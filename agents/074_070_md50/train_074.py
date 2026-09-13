"""EXP-074 「正しく選定した構成」のアリーナ検証。

EXP-070はグリッド選定を test(07-15) で行い、同じ test で報告していた（選定バイアス）。
正しくは選定も val(07-14) で行うべきで、その場合に選ばれるのは:

  num_leaves=31 / min_data_in_leaf=50   (valid_acc 0.7858 = グリッド最良)
  ※070が出荷したのは 31/100 (valid 0.7848、test 0.7767 = testで最良)

両者のオフライン差は test で 0.13pt（0.7767 vs 0.7754）だが、
**オフラインの差からアリーナの差は推論できない**（EXP-058: オフライン-0.5ptでアリーナ+3.1pt、
EXP-060: オフライン+9ptでアリーナ±0）。よってアリーナで直接測る。

- 中立   → ハイパラ微差は平坦領域。31/100の出荷継続を正当化＋今後グリッドに費用をかけない根拠
- 非中立 → **オフライン一致率で選ぶ手法自体が破綻**。EXP-070/071/073の判断根拠が揺らぐ

データ・early stopping条件は070と完全に同一（train 07-09..13 / val 07-14 / patience 150）。

usage:
  uv run python agents/074_070_md50/train_074.py --train tr.pkl --val va.pkl --test te.pkl
"""

from __future__ import annotations

import argparse
import os
import pickle

import numpy as np

# 070のグリッドから「validで最良」の構成。他は070と同一
PARAMS = dict(objective="multiclass", metric="multi_logloss",
              learning_rate=0.01, num_leaves=31, min_data_in_leaf=50,
              feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1,
              verbose=-1, seed=42)
ROUNDS = 8000
PATIENCE = 150

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
    ap.add_argument("--save", default=os.path.join(AGENT_DIR, "model_074"))
    args = ap.parse_args()

    import lightgbm as lgb

    keys, classes, tr = load(args.train)
    _, _, va = load(args.val)
    _, _, te = load(args.test)
    print(f"train={len(tr['X'])} val={len(va['X'])} test={len(te['X'])}")

    dtr = lgb.Dataset(tr["X"], label=tr["y"], feature_name=list(keys))
    dva = lgb.Dataset(va["X"], label=va["y"], reference=dtr)
    p = dict(PARAMS, num_class=len(classes))
    b = lgb.train(p, dtr, num_boost_round=ROUNDS, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(PATIENCE)])
    ni = b.best_iteration
    print(f"trained: best_iter={ni} → {ni * len(classes)} trees "
          f"(070は1291 → 47767木)")

    for tag, d in (("val 07-14", va), ("test 07-15", te)):
        pred = masked_argmax(b.predict(d["X"], num_iteration=ni), d["avail"])
        print(f"  {tag}: 074 class acc = {(pred == d['y']).mean():.4f}")
    print("  参考: 070のグリッド記録 valid=0.7858 test=0.7754 (31/50)")
    print("        070が出荷した構成   valid=0.7848 test=0.7767 (31/100)")

    b.save_model(args.save + ".txt")
    print(f"saved -> {args.save}.txt")


if __name__ == "__main__":
    main()
