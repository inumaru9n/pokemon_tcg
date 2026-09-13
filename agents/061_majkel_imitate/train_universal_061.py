"""EXP-061 universal imitation ranker の学習。

各選択肢を「選ばれるスコア」の二値GBTで採点し、決定内argmax（複数選択はtop-k）。
評価= pick一致率（単一選択）/ 枚単位一致（複数選択）。ctx別に集計して
どのコンテキストがモデルで写せているか（=どこにルールが要るか）を出す。

usage:
  uv run python agents/061_majkel_imitate/train_universal_061.py \
      --dev dev.pkl --val val.pkl --save agents/061_majkel_imitate/model_universal_061
"""

from __future__ import annotations

import argparse
import pickle
from collections import defaultdict

import numpy as np


def load(path):
    with open(path, "rb") as f:
        d = pickle.load(f)
    rows = d["rows"]
    X = np.array([r["x"] for r in rows], dtype=np.float32)
    y = np.array([r["y"] for r in rows], dtype=np.int32)
    dec = [(r["ep"], r["step"]) for r in rows]
    ctx = np.array([r["ctx"] for r in rows], dtype=np.int32)
    minc = np.array([r["minc"] for r in rows], dtype=np.int32)
    return d["feat_keys"], X, y, dec, ctx, minc


def _groups(dec):
    g = defaultdict(list)
    for i, key in enumerate(dec):
        g[key].append(i)
    return g


def coverage(score, y, dec, ctx, minc):
    """ctx別のpick/枚単位一致率。single: argmax==chosen, multi: top-k重なり/k。"""
    g = _groups(dec)
    per = defaultdict(lambda: [0.0, 0])  # ctx -> [agreement_sum, n_decisions]
    for key, idxs in g.items():
        idxs = np.array(idxs)
        c = int(ctx[idxs[0]])
        k = int(minc[idxs[0]])
        chosen = set(int(i) for i in idxs if y[i] == 1)
        kk = max(1, len(chosen))
        order = idxs[np.argsort(-score[idxs])]
        pred = set(int(i) for i in order[:kk])
        agree = len(pred & chosen) / kk
        per[c][0] += agree
        per[c][1] += 1
    return per


CTX_NAME = {0: "MAIN", 3: "SWITCH", 4: "TO_ACTIVE", 5: "TO_BENCH", 7: "TO_HAND",
            8: "DISCARD", 9: "TO_DECK", 30: "DISCARD_ENERGY", 37: "EVOLVE",
            38: "DRAW_COUNT", 41: "IS_FIRST", 43: "ACTIVATE", 1: "SETUP_ACTIVE",
            2: "SETUP_BENCH"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--leaves", type=int, default=63)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--rounds", type=int, default=2000)
    ap.add_argument("--save", default=None)
    args = ap.parse_args()

    keys, Xd, yd, decd, cxd, mnd = load(args.dev)
    kv, Xv, yv, decv, cxv, mnv = load(args.val)
    assert keys == kv
    print(f"dev rows={len(Xd)} decisions={len(set(decd))}  "
          f"val rows={len(Xv)} decisions={len(set(decv))}  feat={len(keys)}")

    import lightgbm as lgb
    dtr = lgb.Dataset(Xd, label=yd, feature_name=list(keys))
    dva = lgb.Dataset(Xv, label=yv, reference=dtr)
    params = dict(objective="binary", metric="binary_logloss",
                  learning_rate=args.lr, num_leaves=args.leaves,
                  min_data_in_leaf=50, feature_fraction=0.9,
                  bagging_fraction=0.9, bagging_freq=1, verbose=-1, seed=42)
    bst = lgb.train(params, dtr, num_boost_round=args.rounds, valid_sets=[dva],
                    callbacks=[lgb.early_stopping(80), lgb.log_evaluation(200)])
    sc = bst.predict(Xv, num_iteration=bst.best_iteration)
    per = coverage(sc, yv, decv, cxv, mnv)

    print(f"\n=== ctx別カバレッジ（val、model pick/枚単位一致率）best_iter={bst.best_iteration} ===")
    print(f"  {'ctx':>4s} {'name':16s} {'n決定':>6s} {'一致率':>7s}")
    tot_a = tot_n = 0
    for c, (a, n) in sorted(per.items(), key=lambda kv_: -kv_[1][1]):
        print(f"  {c:4d} {CTX_NAME.get(c,'?'):16s} {n:6d}  {a/n:.3f}")
        tot_a += a
        tot_n += n
    print(f"  {'ALL':>21s} {tot_n:6d}  {tot_a/tot_n:.3f}")

    imp = sorted(zip(keys, bst.feature_importance("gain")),
                 key=lambda kv_: -kv_[1])[:15]
    print("\ntop gains:", [(k, int(v)) for k, v in imp])
    if args.save:
        bst.save_model(args.save + ".txt", num_iteration=bst.best_iteration)
        print(f"saved -> {args.save}.txt")


if __name__ == "__main__":
    main()
