"""EXP-060 アクティブ選出ランカーの学習 (I-117)。

候補ポケモンごとに「選ぶスコア」を二値GBTで出し、決定内argmaxを選択とする。
評価=pick一致率（argmaxが本人の選択か）。ctx別（SWITCH/TO_ACTIVE）に集計し、
052ベース（現policyの選択）と比較する。052ベースはcollect時に別途測る必要があるため、
ここでは学習器のpick一致率のみ算出し、ベースは calib（policyのagent実行）で別測。

usage:
  uv run python agents/060_alakazam_active/train_active_060.py \
      --dev dev.pkl --val val.pkl --save agents/060_alakazam_active/model_active_060
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
    return d["feat_keys"], X, y, dec, ctx


def _groups(dec):
    g = defaultdict(list)
    for i, key in enumerate(dec):
        g[key].append(i)
    return g


def pick_acc(score, y, dec, ctx, which=None):
    g = _groups(dec)
    ok = tot = 0
    for key, idxs in g.items():
        idxs = np.array(idxs)
        if which is not None and ctx[idxs[0]] != which:
            continue
        pick = idxs[np.argmax(score[idxs])]
        ok += int(y[pick] == 1)
        tot += 1
    return ok / max(1, tot), tot


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--save", default=None)
    args = ap.parse_args()

    keys, Xd, yd, decd, cxd = load(args.dev)
    kv, Xv, yv, decv, cxv = load(args.val)
    assert keys == kv
    print(f"dev rows={len(Xd)} decisions={len(set(decd))}  "
          f"val rows={len(Xv)} decisions={len(set(decv))}  feat={len(keys)}")

    import lightgbm as lgb
    dtr = lgb.Dataset(Xd, label=yd, feature_name=list(keys))
    dva = lgb.Dataset(Xv, label=yv, reference=dtr)
    params = dict(objective="binary", metric="binary_logloss",
                  learning_rate=0.03, num_leaves=31, min_data_in_leaf=30,
                  feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1,
                  verbose=-1, seed=42)
    bst = lgb.train(params, dtr, num_boost_round=2000, valid_sets=[dva],
                    callbacks=[lgb.early_stopping(80), lgb.log_evaluation(200)])
    sc = bst.predict(Xv, num_iteration=bst.best_iteration)
    a_all, n_all = pick_acc(sc, yv, decv, cxv)
    a_sw, n_sw = pick_acc(sc, yv, decv, cxv, which=3)
    a_ta, n_ta = pick_acc(sc, yv, decv, cxv, which=4)
    print(f"model pick acc: all {a_all:.4f} (n={n_all})  "
          f"SWITCH {a_sw:.4f} (n={n_sw})  TO_ACTIVE {a_ta:.4f} (n={n_ta})")
    imp = sorted(zip(keys, bst.feature_importance("gain")),
                 key=lambda kvp: -kvp[1])[:12]
    print("top gains:", [(k, int(v)) for k, v in imp])
    if args.save:
        bst.save_model(args.save + ".txt", num_iteration=bst.best_iteration)
        print(f"saved -> {args.save}.txt")


if __name__ == "__main__":
    main()
