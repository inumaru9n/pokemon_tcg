"""EXP-057 選択クラス予測の学習器比較 (I-111)。

collect_057.py の pickle から (ゾーン特徴 → 本人の行動クラス) の多クラス分類を
複数モデルで学習・比較する。推論は「選択可能クラスでマスクした argmax」=
実運用と同じ形で評価する。

比較基準（ベース）= 052policyの選択クラス（pb_cls）。
usage:
  uv run python agents/057_alakazam_policyml/train_057.py \
      --dev dev.pkl --val val.pkl --models gbt logreg mlp --save-best model_057
"""

from __future__ import annotations

import argparse
import pickle

import numpy as np


def load(path):
    with open(path, "rb") as f:
        d = pickle.load(f)
    rows = d["rows"]
    X = np.array([r["x"] for r in rows], dtype=np.float32)
    y = np.array([r["y"] for r in rows], dtype=np.int32)
    avail = np.array([r["avail"] for r in rows], dtype=bool)
    pb = np.array([r["pb_cls"] for r in rows], dtype=np.int32)
    mb = np.array([r["match_base"] for r in rows], dtype=bool)
    turn = np.array([r["turn"] for r in rows], dtype=np.int32)
    return d["feat_keys"], d["classes"], X, y, avail, pb, mb, turn


def masked_argmax(proba, avail):
    p = np.where(avail, proba, -1.0)
    return p.argmax(axis=1)


def report(name, pred, y, pb, mb, turn, classes):
    acc = (pred == y).mean()
    base_cls_acc = (pb == y).mean()
    fire = pred != pb
    f_ok = (pred[fire] == y[fire]).sum()
    f_base = (pb[fire] == y[fire]).sum()
    print(f"\n== {name} ==")
    print(f"class acc = {acc:.4f}   (base class acc = {base_cls_acc:.4f})")
    print(f"fire(クラス変化)={fire.sum()}  model正解={f_ok}  base正解={f_base}  "
          f"net={f_ok - f_base:+d}")
    for lo, hi, tag in ((0, 2, "t1-2"), (3, 5, "t3-5"), (6, 9, "t6-9"),
                        (10, 99, "t10+")):
        m = (turn >= lo) & (turn <= hi)
        if m.sum():
            print(f"  {tag:5s} n={m.sum():6d} model={(pred[m]==y[m]).mean():.3f} "
                  f"base={(pb[m]==y[m]).mean():.3f}")
    # クラス別（本人の選択クラス頻度上位）
    print("  class別 (n / model / base):")
    cnt = np.bincount(y, minlength=len(classes))
    for ci in np.argsort(-cnt)[:10]:
        m = y == ci
        if m.sum() < 30:
            continue
        print(f"    {classes[ci]:12s} {m.sum():5d}  {(pred[m]==ci).mean():.3f}  "
              f"{(pb[m]==ci).mean():.3f}")
    return acc


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--models", nargs="+", default=["gbt", "logreg", "mlp"])
    ap.add_argument("--save-best", default=None, dest="save_best")
    args = ap.parse_args()

    keys, classes, Xd, yd, avd, pbd, mbd, td = load(args.dev)
    kv, cv, Xv, yv, avv, pbv, mbv, tv = load(args.val)
    assert keys == kv and classes == cv
    print(f"dev={len(Xd)} val={len(Xv)} feat={len(keys)} classes={len(classes)}")
    print(f"base option-acc: dev={mbd.mean():.4f} val={mbv.mean():.4f}")

    mu = Xd.mean(axis=0)
    sd = Xd.std(axis=0)
    sd[sd < 1e-6] = 1.0

    results = {}
    if "gbt" in args.models:
        import lightgbm as lgb
        dtr = lgb.Dataset(Xd, label=yd, feature_name=list(keys))
        dva = lgb.Dataset(Xv, label=yv, reference=dtr)
        params = dict(objective="multiclass", num_class=len(classes),
                      metric="multi_logloss", learning_rate=0.05,
                      num_leaves=63, min_data_in_leaf=50,
                      feature_fraction=0.9, bagging_fraction=0.9,
                      bagging_freq=1, verbose=-1, seed=42)
        bst = lgb.train(params, dtr, num_boost_round=1500, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(80),
                                   lgb.log_evaluation(200)])
        pred = masked_argmax(bst.predict(Xv, num_iteration=bst.best_iteration),
                             avv)
        results["gbt"] = (report("GBT (lightgbm multiclass)", pred, yv, pbv,
                                 mbv, tv, classes), bst)
        imp = sorted(zip(keys, bst.feature_importance("gain")),
                     key=lambda kv_: -kv_[1])[:15]
        print("  top gains:", [(k, int(v)) for k, v in imp])

    if "logreg" in args.models:
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(max_iter=2000, C=0.5, n_jobs=-1)
        lr.fit((Xd - mu) / sd, yd)
        proba = np.zeros((len(Xv), len(classes)), dtype=np.float64)
        proba[:, lr.classes_] = lr.predict_proba((Xv - mu) / sd)
        pred = masked_argmax(proba, avv)
        results["logreg"] = (report("Logistic Regression", pred, yv, pbv, mbv,
                                    tv, classes), lr)

    if "mlp" in args.models:
        from sklearn.neural_network import MLPClassifier
        nn = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=60,
                           early_stopping=True, n_iter_no_change=6,
                           random_state=42)
        nn.fit((Xd - mu) / sd, yd)
        proba = np.zeros((len(Xv), len(classes)), dtype=np.float64)
        proba[:, nn.classes_] = nn.predict_proba((Xv - mu) / sd)
        pred = masked_argmax(proba, avv)
        results["mlp"] = (report("MLP (sklearn 128-64)", pred, yv, pbv, mbv,
                                 tv, classes), nn)

    best = max(results.items(), key=lambda kv_: kv_[1][0])
    print(f"\nbest = {best[0]} (val class acc {best[1][0]:.4f})")
    if args.save_best:
        name, (acc, model) = best
        if name == "gbt":
            model.save_model(args.save_best + ".txt")
            print(f"saved -> {args.save_best}.txt")
        else:
            with open(args.save_best + ".pkl", "wb") as f:
                pickle.dump({"model": model, "mu": mu, "sd": sd,
                             "classes": classes, "keys": keys}, f)
            print(f"saved -> {args.save_best}.pkl")


if __name__ == "__main__":
    main()
