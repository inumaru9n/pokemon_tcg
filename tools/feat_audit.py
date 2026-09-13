"""特徴の**被覆監査**。エンジンが出せるデータを全列挙し、特徴器が使っているか照合する。

**なぜ必要か（EXP-097の反省）**: EXP-097 の特徴設計は「強いプレイヤーは何を見るか」からの
トップダウンだった。この方向は終わりがなく、**「全部挙げた」と言える根拠が原理的に無い**。
実際 EXP-099 で、弱点（ダメージ2倍）が打点計算に入っておらず、教師の試合の54.5%で
脅威特徴が2倍ぶん過小評価されていたことが判明した。**特徴が静かに間違っていても
例外は出ない**ので、テストでは捕まらない。

ボトムアップの被覆監査なら「使っていないもの」が有限リストとして出る。ただし
**監査で網羅できるのは3層のうち1層だけ**なので、この道具は L1 と L2 の両方を出す:

  L0 可視性  どのゾーンが観測できるか。掛け合わせのどのセルが存在するかを決める。
             「見えていない」こと自体も情報（相手の未公開カードの推定）
  L1 状態    ゾーン × 主体 × 属性。**機械的に列挙でき、この道具で網羅できる**
  L2 関係    2つ以上のオブジェクトにまたがる量（弱点=自分の型×相手の弱点、
             逃走=逃げコスト×付いているエネ、クロック=打点×残HP…）。
             **掛け合わせからは絶対に出ない**。ルール上の相互作用の一覧から別途列挙する
  L3 履歴    状態のスナップショットに存在しない（何ターン前線にいるか等）。
             logs と LSTM が部分的に担う。この道具の対象外

粒度の原則（EXP-099）:
  自分側 = カード単位でよい（デッキで確定・22種）
  相手側 = **属性単位が本体**（型/弱点/抵抗/HP/stage/逃げコスト/特性有無）。
           カード単位は「メタ上位N種」に限れば可能だが**メタが動くと陳腐化する**
           （直近1週間で相手のポケモンは138種、上位50種で97.7%）

usage:
  uv run python tools/feat_audit.py                       # 現行コアを監査
  uv run python tools/feat_audit.py --agent agents/103_alakazam_feat3
"""

from __future__ import annotations

import argparse
import os
import random
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO, os.path.join(REPO, "arena")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---- L2: ルールが定める相互作用の一覧（掛け合わせからは出ないので手で列挙する） ----
# 「実装済みか」を判定する目印の識別子を添える。増やすときはルールブック由来のものだけ。
L2_RELATIONS = [
    ("弱点2倍・抵抗-30", "自分の攻撃役の型 × 相手の弱点/抵抗", "_wr_mult"),
    ("技が撃てるか（型付き）", "技のコスト型 × 付いているエネの型", "_afford"),
    ("逃げられるか", "逃げコスト × 付いているエネ × 行動不能", "can_retreat"),
    ("クロック", "打点 × 残HP", "_clock"),
    ("サイドレース", "KO時のサイド枚数 × 残サイド", "lethal_for_me"),
    ("進化できるか", "手札のカード × 場のポケモン", "enables_evo"),
    ("状態異常の継続ダメージ", "毒/火傷 × ターン経過", "_dot"),
    ("デッキアウト", "山残り × ターン経過", "deckout_me"),
]

SKIP_ATTRS = {
    "serial", "result", "yourIndex",          # ID・答えなので使ってはいけない
    "search_begin_input", "searchId",         # 探索用
    "context",                                # ctx として別経路で使用
}


def _attrs(o):
    out = {}
    for f in sorted(dir(o)):
        if f.startswith("_"):
            continue
        try:
            v = getattr(o, f)
        except Exception:  # noqa: BLE001
            continue
        if callable(v):
            continue
        out[f] = v
    return out


def _used(name: str, src: str) -> bool:
    return bool(re.search(rf"\.{re.escape(name)}\b", src)
                or re.search(rf'"{re.escape(name)}"', src))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--agent", default="agents/103_alakazam_feat3")
    ap.add_argument("--opp", default="agents/101_marnie_luca")
    a = ap.parse_args()

    import cg.game as cgg
    import run_match as rm
    from cg.api import all_card_data, to_observation_class

    ag = os.path.join(REPO, a.agent)
    feat = next(f for f in sorted(os.listdir(ag))
                if f.startswith("feat_") and f.endswith(".py"))
    src = open(os.path.join(ag, feat)).read()

    pa = rm.load_agent_module(ag, "A")
    pb = rm.load_agent_module(os.path.join(REPO, a.opp), "B")
    da, db = rm.read_deck(ag), rm.read_deck(os.path.join(REPO, a.opp))
    random.seed(3)
    obs, _ = cgg.battle_start_seeded(da, db, 777001)
    got = None
    while obs is not None:
        if obs["current"]["result"] != -1:
            break
        me = obs["current"]["yourIndex"]
        if me == 0:
            o = to_observation_class(obs)
            st = o.current
            if (st.turn >= 6 and st.players[0].hand and st.players[1].bench[0]
                    and st.players[1].active and st.players[1].active[0]):
                got = (o, me)
                break
        obs = cgg.battle_select((pa if me == 0 else pb)(obs))
    if got is None:
        raise SystemExit("監査に使える盤面が取れなかった")
    o, me = got
    st = o.current
    tbl = {c.cardId: c for c in all_card_data()}

    print(f"監査対象: {a.agent}/{feat}\n")
    print("=" * 66)
    print("L0 可視性（掛け合わせのどのセルが存在するか）")
    print("=" * 66)
    print(f"  {'ゾーン':<12}{'自分':<26}{'相手':<26}")
    for z in ("hand", "deck", "prize", "discard", "active", "bench"):
        cells = []
        for pi in (me, 1 - me):
            ps = st.players[pi]
            v = getattr(ps, z, None)
            if v is None:
                cells.append("枚数のみ")
            elif isinstance(v, list) and v and all(x is None for x in v):
                cells.append(f"枚数のみ（中身は伏せ, {len(v)}）")
            elif isinstance(v, list):
                cells.append("中身が見える")
            else:
                cells.append(str(v)[:24])
        print(f"  {z:<12}{cells[0]:<26}{cells[1]:<26}")
    print(f"  {'stadium':<12}{'共有・中身が見える':<26}")

    print()
    print("=" * 66)
    print("L1 状態（ゾーン × 主体 × 属性）— 機械的に網羅できる層")
    print("=" * 66)
    targets = [
        ("Observation", o, ()),
        ("State（盤面全体）", st, ("players",)),
        ("PlayerState", st.players[me],
         ("active", "bench", "hand", "discard", "prize", "deck", "stadium")),
        ("場のポケモン", st.players[me].active[0] if st.players[me].active
         and st.players[me].active[0] else st.players[me].bench[0],
         ("energyCards", "tools", "preEvolution")),
        ("手札のカード", st.players[me].hand[0], ()),
        ("SelectData", o.select, ("option",)),
        ("Option（選択肢）", o.select.option[0], ()),
        ("CardData（カード属性）", tbl[st.players[1 - me].active[0].id],
         ("attacks", "skills")),
    ]
    total_un = 0
    for title, obj, skip in targets:
        if obj is None:
            continue
        used, unused = [], []
        for f, v in _attrs(obj).items():
            if f in skip or f in SKIP_ATTRS:
                continue
            (used if _used(f, src) else unused).append((f, str(v)[:40]))
        mark = "✅" if not unused else "⚠️"
        print(f"\n{mark} {title}: 使用 {len(used)} / 未使用 {len(unused)}")
        for f, v in unused:
            print(f"     ✗ {f:<22}= {v}")
        total_un += len(unused)

    print()
    print("=" * 66)
    print("L2 関係（2つ以上のオブジェクトにまたがる量）— 掛け合わせでは出ない層")
    print("=" * 66)
    miss = 0
    for name, how, marker in L2_RELATIONS:
        hit = marker in src
        if not hit:
            miss += 1
        print(f"  {'✅' if hit else '✗ '} {name:<24}{how}")

    print()
    print("=" * 66)
    print(f"未使用フィールド {total_un} 件 / 未実装のL2関係 {miss} 件")
    print("L3（履歴）はこの監査の対象外。logs と LSTM が担う")
    print("=" * 66)
    cgg.battle_finish()


if __name__ == "__main__":
    main()
