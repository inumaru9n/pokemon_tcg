"""教師データから「削除すべき列」を算出する（EXP-122）。

**デッキごとに実データで測ること**。重複も定数もデッキと教師データに依存するので、
あるデッキで作ったリストを別のデッキに流用すると、そこでは重複でない列を消して
情報を失う（例: `me_dot` と `me_dot_ability` は Alakazam の教師データで偶然一致した
だけで、特性ダメージを多用するデッキでは分岐しうる。ACE SPEC もデッキごとに別カード）。

出すもの:
  (a) **完全重複列** — 全決定でビット単位に一致する列群。各群から1本だけ残す。
      重複列は「その方向の実効学習率を倍にする」装置として働く（2列は毎ステップ
      同一の勾配を受けるので w = init + Δ が2本ぶん乗る）。意図しない強調なので整理する。
  (b) **定数列** — 分散ゼロ。訓練中ずっと同じ値なので情報を運ばないが、標準化の sd が
      下限(1e-3)に張り付くため、本番でその特徴に値が入ると1,000シグマがネットワークに
      流れて方策が壊れる（EXP-111 の744シグマと同じ経路。実測で本番1.1%の試合が該当）。
  (c) **語彙の②違反** — `opp_seen_*` のうち**正例数**（その特徴が発火した決定の数）
      が閾値未満のもの。
      META_CARDS の選抜は ①識別力（ARCH_DECK 1〜2系統 ＋ "Basic " 除外）が
      グローバル定義、②学習可能性（**正例数 >= 500**）が**データセットごと**の条件。
      ②は「誰の試合を模倣するか」で変わる（Yushin/Alakazam で観測される札と
      Luca/Marnie で観測される札は違う）ので、ここで測り直して落とす。
      根拠は166の実測: ②を外して死に次元を18個入れると **-2.29pt**。
      **mu（観測平均）で切ってはいけない**。mu は「1回に何枚見えるか」に引きずられ、
      1枚積みの札を不当に落とす。実測(Luca 483,798決定): Mega Lopunny ex は
      mu 0.0190 で正例 **5,010** なのに落ち、Applin は mu 0.0223 で正例 4,500 なのに残る
      という逆転が起きていた。学習に効くのはサンプル数なので**正例数**で測る。
      「稀＝無価値」ではない点にも注意（Lively Stadium は正例1,387だが、1回見えれば
      相手が Ogerpon だとほぼ確定する＝1回あたりの情報量は最大級）。

残す列の選び方: 各群で**カードIDに直結しない一般的な名前**を優先する
（`hand_1266` より `hand_n_stad`、`line_shaymin` より `my_has_protect`）。
デッキを差し替えても意味を保つほうを残しておくと、後でデッキ探索に移るとき再利用できる。

usage:
  uv run python tools/calc_drop_cols.py --npz data/nn/marnie176.npz \\
      --out data/nn/drop176.json
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re

import numpy as np

# 群の代表を選ぶ規則。上から順に当てる。
_PREFER = [
    (re.compile(r"^(ms|os)0_"), "スロット表現に統一（ms0_/os0_ がアクティブ）"),
    (re.compile(r"^pool_"), "left_ より pool_（超幾何確率の計算が pool_ を参照）"),
]
# カードID直結の名前より一般名を残す（明示指定）
_GENERIC_FIRST = re.compile(r"_\d+$")


def pick_keep(group: list[str]) -> str:
    """群から残す1本を選ぶ。"""
    for rx, _ in _PREFER:
        for n in group:
            if rx.match(n):
                return n
    # カードID直結でない名前を優先
    generic = [n for n in group if not _GENERIC_FIRST.search(n)]
    if generic:
        return sorted(generic)[0]
    return sorted(group)[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", required=True, help="pack_dataset.py が出した npz")
    ap.add_argument("--out", required=True, help="削除列JSONの出力先")
    ap.add_argument("--sd-floor", type=float, default=1e-3,
                    help="これ以下の sd を定数扱いにする（学習側の下限と揃える）")
    ap.add_argument("--min-pos", type=int, default=500,
                    help="opp_seen_* の**正例数**がこれ未満なら落とす（語彙の②）")
    a = ap.parse_args()

    z = np.load(a.npz, allow_pickle=True)
    B = z["board"]
    fk = [str(x) for x in z["feat_keys"]]
    print(f"■ {a.npz}  board {B.shape}")

    sd = B.std(0)
    const = sorted(fk[i] for i in np.nonzero(sd <= a.sd_floor)[0])

    h = collections.defaultdict(list)
    for j in range(B.shape[1]):
        key = hashlib.blake2b(np.ascontiguousarray(B[:, j]).tobytes(), digest_size=16).digest()
        h[key].append(j)
    groups = [sorted(fk[j] for j in v) for v in h.values()
              if len(v) > 1 and sd[v[0]] > a.sd_floor]

    drop_dup, keeps = [], []
    for g in groups:
        k = pick_keep(g)
        keeps.append((k, [x for x in g if x != k]))
        drop_dup += [x for x in g if x != k]

    print(f"\n■ 完全重複 {len(groups)}群 / 余剰 {len(drop_dup)}列")
    for k, d in sorted(keeps)[:60]:
        print(f"   残す {k:26} 消す {d}")
    if len(keeps) > 60:
        print(f"   … 他 {len(keeps)-60}群")
    print(f"\n■ 定数列 {len(const)}列")
    for i in range(0, len(const), 4):
        print("   " + "  ".join(f"{x:24}" for x in const[i:i + 4]))

    # (c) 語彙の②をこの学習データで測り直す
    voc = [(fk[i], int((B[:, i] > 0).sum()), float(B[:, i].mean()))
           for i in range(len(fk)) if fk[i].startswith("opp_seen_")]
    thin = sorted(n for n, pos, _ in voc if pos < a.min_pos)
    print(f"\n■ 語彙の②（**正例数 >= {a.min_pos}**）を測り直し: "
          f"opp_seen_* {len(voc)}枚中 **{len(thin)}枚が不合格**")
    for n, pos, mu in sorted(voc, key=lambda x: x[1])[:len(thin) + 5]:
        mark = " **落とす**" if pos < a.min_pos else ""
        print(f"   {n:22} 正例 {pos:>7,}  (mu {mu:.4f}){mark}")

    drop = sorted(set(drop_dup) | set(const) | set(thin))
    json.dump({"drop": drop, "n_dup": len(drop_dup), "n_const": len(const),
               "n_vocab_thin": len(thin), "min_pos": a.min_pos,
               "source": a.npz, "board_dim": int(B.shape[1]),
               "after": int(B.shape[1]) - len(drop)},
              open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"\n   重複 {len(drop_dup)} + 定数 {len(const)} + 語彙② {len(thin)} "
          f"（重複を除いた実数 {len(drop)}）")
    print(f"**{B.shape[1]} - {len(drop)} = {B.shape[1]-len(drop)}次元**  -> {a.out}")


if __name__ == "__main__":
    main()
