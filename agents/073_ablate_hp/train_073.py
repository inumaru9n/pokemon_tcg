"""EXP-073 070の利得の要因分解（データ鮮度 vs ハイパラ）。

070は058から「データ」と「ハイパラ」を同時に変えたため、vs058 +3.2pt の出どころが
分離できていない。本スクリプトは **058のハイパラを070のデータで**学習し、中間点Xを作る。

  058 = 旧データ(07-09..12) × 旧ハイパラ(lr0.05/1500固定/leaves63/min_data50)
  X   = 新データ(07-09..13) × 旧ハイパラ          ← これを作る
  070 = 新データ            × 新ハイパラ(lr0.01/early stopping/leaves31/min_data100)

X vs 070 = ハイパラの純効果 / 058 vs X = データの純効果。
評価は070と同一の test 07-15（同一日・同一データなので一致率を直接比較できる）。

usage:
  uv run python agents/073_ablate_hp/train_073.py --train tr.pkl --val va.pkl --test te.pkl
"""

from __future__ import annotations

import argparse
import os
import pickle

import numpy as np

# EXP-058 の train_058.py:90-97 と同一（early stopping patience=80 も含む）
PARAMS_058 = dict(objective="multiclass", metric="multi_logloss",
                  learning_rate=0.05, num_leaves=63, min_data_in_leaf=50,
                  feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1,
                  verbose=-1, seed=42)
ROUNDS_058 = 1500
PATIENCE_058 = 80

AGENT_DIR = os.path.dirname(os.path.abspath(__file__))


def load(path):
    with open(path, "rb") as f:
        d = pickle.load(f)
    rows = d["rows"]
    out = dict(
        X=np.array([r["x"] for r in rows], dtype=np.float32),
        y=np.array([r["y"] for r in rows], dtype=np.int32),
        avail=np.array([r["avail"] for r in rows], dtype=bool),
        pb=np.array([r["pb_cls"] for r in rows], dtype=np.int32),
        turn=np.array([r["turn"] for r in rows], dtype=np.int32),
    )
    return d["feat_keys"], d["classes"], out


def masked_argmax(proba, avail):
    return np.where(avail, proba, -1.0).argmax(axis=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--save", default=os.path.join(AGENT_DIR, "model_073"))
    args = ap.parse_args()

    import lightgbm as lgb

    keys, classes, tr = load(args.train)
    _, _, va = load(args.val)
    _, _, te = load(args.test)
    print(f"train={len(tr['X'])} val={len(va['X'])} test={len(te['X'])}")

    # 058の設定を忠実に再現（early stopping patience=80 を含む。valで停止判定）
    dtr = lgb.Dataset(tr["X"], label=tr["y"], feature_name=list(keys))
    dva = lgb.Dataset(va["X"], label=va["y"], reference=dtr)
    p = dict(PARAMS_058, num_class=len(classes))
    b = lgb.train(p, dtr, num_boost_round=ROUNDS_058, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(PATIENCE_058)])
    ni = b.best_iteration
    print(f"trained: best_iter={ni} → {ni * len(classes)} trees")

    # 参考: 実物の058モデル（旧データ×旧ハイパラ）を同じtestで評価
    m058 = os.path.join(os.path.dirname(AGENT_DIR),
                        "058_alakazam_policyml2", "model_058.txt")
    if os.path.exists(m058):
        b58 = lgb.Booster(model_file=m058)
        assert b58.num_feature() == len(keys), (b58.num_feature(), len(keys))
        pr58 = masked_argmax(b58.predict(te["X"]), te["avail"])
        print(f"  [参考] 実物058モデル test 07-15 class acc = "
              f"{(pr58 == te['y']).mean():.4f}  ({b58.num_trees()} trees)")

    for tag, d in (("val 07-14", va), ("test 07-15", te)):
        pred = masked_argmax(b.predict(d["X"], num_iteration=ni), d["avail"])
        print(f"  {tag}: X class acc = {(pred == d['y']).mean():.4f}   "
              f"(pb=070実機 {(d['pb'] == d['y']).mean():.4f})")

    pred = masked_argmax(b.predict(te["X"], num_iteration=ni), te["avail"])
    yt, tt = te["y"], te["turn"]
    for lo, hi, tag in ((0, 2, "t1-2"), (3, 5, "t3-5"), (6, 9, "t6-9"),
                        (10, 99, "t10+")):
        m = (tt >= lo) & (tt <= hi)
        if m.sum():
            print(f"    {tag:5s} n={m.sum():6d} X={(pred[m]==yt[m]).mean():.3f}")

    b.save_model(args.save + ".txt")
    print(f"saved -> {args.save}.txt")


if __name__ == "__main__":
    main()
