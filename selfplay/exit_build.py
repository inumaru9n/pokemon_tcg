"""ExItの補正を「試合系列に埋め込んだ」教師データを作る（EXP-124）。

**なぜ系列に埋め込むのか**: 補正局面だけを長さ1の系列として学習させると、LSTMが
ゼロ状態で入力される。実測でゼロ状態と実状態は **22.2%の決定で判断が変わる**ので、
「実状態で見たときの正解」を「ゼロ状態の入力」に貼ることになり、教師そのものが壊れる。
exit_replay.py が生成と同一の試合を打ち直して全決定を持っているので、そこへ
補正を差し込めば、LSTMは本物の履歴を累積した上で補正を受け取れる。

作り方:
  1. ExItの**教師行**を選ぶ。既定 = M=24（逐次打ち切りを通過＝勝負がついていない局面）
     かつ候補≥2 かつ **best≠top1**。M=8で止まった行は「全候補同結果」か「決着済み」で
     打ち切られたものなので、best≠top1 は 1,920件中 6件しかない＝教師にならない。
  2. 指紋（board+logs+prev+opt_cid+ctx+turn の blake2b）で再生側の決定を特定する。
     **1件でも取りこぼしたら中止する**（取りこぼしは相手割り当てズレ等の再現性事故の兆候）。
  3. 各試合を**最後の補正決定まで**で切る。その先は勾配も状態も要らない。
  4. 補正決定の chosen を **best候補**に差し替え、重み dw を与える。
     他の決定は dw=0（**LSTMの文脈としてだけ効く**）。
  5. 既存 npz（dw=1）と連結する。dw は決定ごとの損失重みとして学習側が読む。

重みの決め方: `--new-ratio r` は**重みの総和**の比。
  dw = (既存決定数 * r / (1-r)) / 補正件数。r=0.1 なら 435,150/9/1,091 = **44.3**。

usage:
  uv run python selfplay/exit_build.py --base data/nn/alakazam186.npz \\
      --exit data/nn/exit186.pkl --games data/nn/exit186_games.pkl \\
      --new-ratio 0.1 --out data/nn/alakazam186_exit.npz
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import pickle
from math import comb

import numpy as np


def sign_stat(R, bi):
    """best列 vs top1列 の**CRNペア符号検定**（両側p）と、対応差の実効量を返す。

    R は (候補, M) の勝敗行列で、**全候補が同一の search_begin から分岐している**ので
    列ごとに対応がある。「差の平均」だけでは 24本で偶然入れ替わった行と、
    一貫して勝っている行を区別できない（実測: 全2,557件の不偏gap +0.081 に対し、
    **p<=0.05 を通る265件だけなら +0.436 と5.4倍**）。
    """
    a, b = R[bi], R[0]
    w = int((a > b).sum()); l = int((a < b).sum()); m = w + l
    if m == 0:
        return 1.0, 0.0
    pv = min(1.0, sum(comb(m, k) for k in range(w, m + 1)) / 2 ** m * 2)
    return pv, (w - l) / R.shape[1]        # 対応差の実効量（−1..1）


def fp(board, logs, prev, opt_cid, ctx, turn) -> bytes:
    """決定の指紋。生成側と再生側で同じ局面かを突き合わせる唯一の鍵。"""
    h = hashlib.blake2b(digest_size=16)
    for a in (np.asarray(board, np.float32), np.asarray(logs, np.float32),
              np.asarray(prev, np.float32), np.asarray(opt_cid, np.int64)):
        h.update(np.ascontiguousarray(a).tobytes())
    h.update(bytes([int(ctx) & 255, int(turn) & 255]))
    return h.digest()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="既存の教師 npz（模倣学習ぶん）")
    # **複数バッチを合算できる**（--exit A B --games a b で対応順に組にする）。
    # 生成はシードごとに別の試合群なので、指紋の衝突は起きない。
    ap.add_argument("--exit", required=True, nargs="+", help="exit_gen.py の pkl（複数可）")
    ap.add_argument("--games", required=True, nargs="+", help="exit_replay.py の pkl（--exit と同順）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-M", type=int, default=24, dest="min_M",
                    help="この本数まで回した行だけを教師にする")
    ap.add_argument("--new-ratio", type=float, default=0.1, dest="new_ratio",
                    help="補正ぶんが占める**重みの割合**")
    # **ラベルの質で選別する**（EXP-131）。既定は従来どおり全件・一律重み。
    ap.add_argument("--max-p", type=float, default=1.0, dest="max_p",
                    help="CRNペア符号検定の両側pがこれ以下の補正だけを教師にする")
    ap.add_argument("--weight-by", default="uniform", choices=("uniform", "evidence"),
                    dest="weight_by",
                    help="uniform=全補正に同じ重み / evidence=対応差(w-l)/M に比例")
    a = ap.parse_args()

    assert len(a.exit) == len(a.games), "--exit と --games の数が違う"

    # ---- 1) 教師行を選ぶ ----
    E = []
    for _b, _p in enumerate(a.exit):
        _r = pickle.load(open(_p, "rb"))["rows"]
        print(f"   {_p}: {len(_r):,}局面")
        for _x in _r:
            _x["_batch"] = _b      # **照合はバッチ内に限る**（下記参照）
        E += _r

    # **候補0が方策の貪欲手**。exit_gen._cands が `sorted(order[:k])` を先頭に置く
    # 仕様なので、baseline は常に index 0 である。
    # `sc[c[0]]` で候補を並べ替えて最大を取るのは**誤り**: k>1（複数選択）の候補は
    # スコア順でなく**オプションindex順**に sorted されているため、c[0] は
    # 「その集合で最もスコアが高い1枚」ではない。実測で63行中8行が食い違った。
    TOP1 = 0

    T = [r for r in E
         if r["M"] >= a.min_M and len(r["cands"]) >= 2
         and int(np.argmax(r["q"])) != TOP1]
    for r in T:
        r["_p"], r["_eff"] = sign_stat(np.asarray(r["R"], float), int(np.argmax(r["q"])))
    if a.max_p < 1.0:
        _n0 = len(T)
        T = [r for r in T if r["_p"] <= a.max_p]
        print(f"■ **符号検定 p<={a.max_p} で選別: {_n0:,} → {len(T):,}件**")
    print(f"■ ExIt {len(E):,}局面 → 教師行 **{len(T):,}件** "
          f"(M>={a.min_M} / 候補>=2 / best!=top1)")
    # 選択バイアスを抜いた真の利得を出しておく（前半でbestを選び、後半で測る）
    ub = []
    for r in T:
        R = np.asarray(r["R"]); half = R.shape[1] // 2
        ub.append(R[:, half:][int(np.argmax(R[:, :half].mean(1)))].mean()
                  - R[:, half:][TOP1].mean())
    ub = np.array(ub)
    print(f"   不偏 gap {ub.mean():+.4f} ±{1.96*ub.std(ddof=1)/np.sqrt(len(ub)):.4f}"
          f"（この局面群で best に替えたときの勝率上昇）")

    # ---- 2) 再生側で位置を特定する ----
    # **バッチ間で gi が重複する**ので、試合IDにバッチ番号を混ぜて一意にする。
    G = []
    # **指紋はバッチをまたぐと衝突する**。序盤の定型局面は別の試合でもビット一致しうる
    # （実測: 2バッチ合算で多重ヒット927件。バッチ単体では0件）。
    # ExIt行は自分が出た試合の中にしか対応物が無いので、**索引をバッチごとに分ける**。
    idx = [collections.defaultdict(list) for _ in a.games]
    for _b, _p in enumerate(a.games):
        _r = pickle.load(open(_p, "rb"))["rows"]
        print(f"   {_p}: {len(_r):,}決定")
        for _x in _r:
            _x["gi"] = (_b, _x["gi"])
        for _j, _x in enumerate(_r):
            idx[_b][fp(_x["board"], _x["logs"], _x["prev"],
                       _x["opt_cid"], _x["ctx"], _x["turn"])].append(len(G) + _j)
        G += _r
    hit, miss, dup = {}, 0, 0
    for r in T:
        v = idx[r["_batch"]][fp(r["board"], r["logs"], r["prev"],
                                r["opt_cid"], r["ctx"], r["turn"])]
        if not v:
            miss += 1
        elif len(v) > 1:
            dup += 1
        else:
            hit[v[0]] = r
    print(f"■ 再生 {len(G):,}決定 / {len({r['gi'] for r in G})}試合 と照合: "
          f"一致 {len(hit):,} / 不一致 {miss} / 多重 {dup}")
    if miss or dup:
        raise SystemExit("**照合に失敗**。再生の再現性が崩れている（相手割り当て・シード・"
                         "特徴器のいずれかが生成時と違う）。学習に進んではいけない。")

    # ---- 3) 各試合を最後の補正まで切る ----
    by_game = collections.defaultdict(list)
    for i, r in enumerate(G):
        by_game[r["gi"]].append(i)
    for v in by_game.values():
        v.sort(key=lambda i: G[i]["n"])
    keep_games = {}
    for gi, rows in by_game.items():
        last = max((k for k, i in enumerate(rows) if i in hit), default=-1)
        if last >= 0:
            keep_games[gi] = rows[:last + 1]
    n_new = sum(len(v) for v in keep_games.values())
    print(f"■ 補正を含む {len(keep_games)}試合 を最後の補正まで切り出し: "
          f"**{n_new:,}決定**（うち補正 {len(hit):,} = {100*len(hit)/n_new:.1f}%）")

    # ---- 4) 既存 npz を読み、形を合わせる ----
    z = np.load(a.base, allow_pickle=True)
    O = int(z["opt_cid"].shape[1])
    nb = int(z["board"].shape[0])
    ng = int(len(z["gres"]))
    D = int(z["opt_vec"].shape[2])
    mx_new = max(len(G[i]["opt_cid"]) for v in keep_games.values() for i in v)
    print(f"■ 既存 {nb:,}決定 / {ng}試合 / O={O}  ← 再生側の最大オプション {mx_new}")
    over = [i for v in keep_games.values() for i in v if len(G[i]["opt_cid"]) > O]
    if over:
        bad = [i for i in over if i in hit]
        # LSTMは選択肢を見ない（board/logs/ctx/prev のみ）ので、dw=0 の文脈決定は
        # 先頭Oまで切り詰めて構わない。**補正決定が該当したら中止**する。
        if bad:
            raise SystemExit(f"補正決定に O={O} を超えるものがある: {len(bad)}件")
        print(f"   オプション数 >{O} の文脈決定 {len(over)}件を先頭{O}に切り詰め（dw=0なので無害）")

    _budget = nb * a.new_ratio / (1.0 - a.new_ratio)      # 補正に配る重みの総和
    if a.weight_by == "evidence":
        # **証拠の強さに比例**して配分する。24本中の対応差 (勝−負)/M が小さい行＝
        # 偶然入れ替わっただけの行は、捨てずに寄与だけ小さくする。
        _ev = {i: max(0.0, float(r["_eff"])) for i, r in hit.items()}
        _s = sum(_ev.values()) or 1.0
        dw_of = {i: _budget * v / _s for i, v in _ev.items()}
        _v = np.array(list(dw_of.values()))
        print(f"■ 重み(evidence): 総和 {_v.sum():,.0f} / 中央 {np.median(_v):.1f} / "
              f"最大 {_v.max():.1f} / 最小 {_v.min():.2f} → 新規比 "
              f"**{_v.sum()/(nb+_v.sum()):.1%}**")
    else:
        dw_new = _budget / len(hit)
        dw_of = {i: dw_new for i in hit}
        print(f"■ 重み(uniform): 既存 dw=1 × {nb:,} / 補正 **dw={dw_new:.1f}** × "
              f"{len(hit):,} → 新規比 **{dw_new*len(hit)/(nb+dw_new*len(hit)):.1%}**")

    N = nb + n_new
    out = {}
    for k, shp, dt in (("board", (N, z["board"].shape[1]), np.float32),
                       ("logs", (N, z["logs"].shape[1]), np.float32),
                       ("prev", (N, 3), np.float32),
                       ("opt_vec", (N, O, D), np.float32)):
        out[k] = np.zeros(shp, dt); out[k][:nb] = z[k]
    for k, fill, dt in (("ctx", 0, np.int16), ("turn", 0, np.int16), ("nopt", 0, np.int16),
                        ("opt_cid", -2, np.int32), ("opt_cls", -1, np.int16),
                        ("chosen", 0, np.int8), ("first", 0, np.int8), ("gidx", 0, np.int32)):
        shp = (N, O) if k in ("opt_cid", "opt_cls", "chosen") else (N,)
        out[k] = np.full(shp, fill, dt); out[k][:nb] = z[k]
    dw = np.zeros(N, np.float32); dw[:nb] = 1.0
    gres = np.zeros(ng + len(keep_games), np.int8); gres[:ng] = z["gres"]
    g_new = np.zeros(ng + len(keep_games), np.int8)
    is_new = np.zeros(N, np.int8)

    # ---- 5) 補正ぶんを書き込む ----
    i, gj = nb, ng
    for gi in sorted(keep_games):
        rows = keep_games[gi]
        r0 = G[rows[0]]
        gres[gj] = {1.0: 1, 0.0: -1}.get(r0["game_result"], 0)
        g_new[gj] = 1
        for k, ri in enumerate(rows):
            r = G[ri]
            m = min(len(r["opt_cid"]), O)
            out["board"][i] = r["board"]; out["logs"][i] = r["logs"]
            out["prev"][i] = r["prev"]
            out["ctx"][i] = r["ctx"]; out["turn"][i] = r["turn"]; out["nopt"][i] = m
            out["opt_cid"][i, :m] = r["opt_cid"][:m]
            out["opt_vec"][i, :m] = np.asarray(r["opt_vec"][:m], np.float32)
            out["opt_cls"][i, :m] = r["opt_cls"][:m]
            out["first"][i] = 1 if k == 0 else 0
            out["gidx"][i] = gj
            is_new[i] = 1
            if ri in hit:
                e = hit[ri]
                # **best候補**（qが最大の候補）に差し替える。複数選択(k>1)なら全枚を正例に。
                for c in e["cands"][int(np.argmax(e["q"]))]:
                    out["chosen"][i, c] = 1
                dw[i] = dw_of[ri]
            else:
                for c in r["chosen"]:      # 実際に打った手（dw=0なので損失には効かない）
                    if c < m:
                        out["chosen"][i, c] = 1
            i += 1
        gj += 1
    assert i == N and gj == ng + len(keep_games)

    np.savez_compressed(a.out, n_cls=z["n_cls"], gres=gres, g_new=g_new,
                        is_new=is_new, dw=dw, feat_cap=z["feat_cap"],
                        feat_keys=z["feat_keys"], classes=z["classes"], meta=z["meta"],
                        **out)
    import os
    print(f"\n保存 -> {a.out} ({os.path.getsize(a.out)/1e6:.1f} MB)  "
          f"決定 {N:,} / 試合 {ng+len(keep_games)} / **dw>0 は {int((dw>0).sum()):,}件**")


if __name__ == "__main__":
    main()
