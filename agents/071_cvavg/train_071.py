"""EXP-071 時系列CV平均（rolling-origin CV averaging）の効果測定。

070と同一のデータ（train 07-09..13 / val 07-14 / test 07-15）・同一ハイパラ
（lr=0.01 / num_leaves=31 / min_data_in_leaf=100）のまま、
**時系列でtrain-valを切ったK個のfold**をそれぞれ学習し（early stoppingは各foldのval）、
K本のモデルを確率平均した場合の一致率増分を test 07-15 で測る。

fold構成（1日=1連続ブロック。ランダムfoldは使わない）:
  f1: val 07-09  train 07-10..14
  f2: val 07-10  train 07-09, 07-11..14
  ...
  f6: val 07-14  train 07-09..13   ← 070と同一の切り方
各fold n_train ≈ 100k でほぼ同等。f6の単発が070のモデルに相当するため、
「f6単発 vs K本平均」が求める増分。test 07-15 は学習にもearly stoppingにも一切使わない。

ハイパラは固定＝ここでは選定を行わないので、070でグリッド選定に使ったtestを
そのまま増分測定に流用できる（単発 vs 平均の差分のみを読む）。
K倍の推論コストを払う前のゲート: 増分が +1pt 未満なら破棄（アリーナで検出不能）。

usage:
  uv run python agents/071_cvavg/train_071.py \
      --train tr.pkl --val va.pkl --test te.pkl --save-dir agents/071_cvavg
"""

from __future__ import annotations

import argparse
import os
import pickle

import numpy as np

PARAMS = dict(objective="multiclass", metric="multi_logloss",
              learning_rate=0.01, num_leaves=31, min_data_in_leaf=100,
              feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1,
              verbose=-1, seed=42)


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
        date=np.array([r["date"] for r in rows]),
    )
    return d["feat_keys"], d["classes"], out


def masked_argmax(proba, avail):
    return np.where(avail, proba, -1.0).argmax(axis=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", required=True)   # 07-09..13
    ap.add_argument("--val", required=True)     # 07-14
    ap.add_argument("--test", required=True)    # 07-15（報告用）
    ap.add_argument("--save-dir", default=None, dest="save_dir")
    args = ap.parse_args()

    import lightgbm as lgb

    keys, classes, tr = load(args.train)
    k2, c2, va = load(args.val)
    k3, c3, te = load(args.test)
    assert keys == k2 == k3 and classes == c2 == c3

    # 全trainデータを日付順に連結（tr 07-09..13 + va 07-14）
    X = np.concatenate([tr["X"], va["X"]])
    y = np.concatenate([tr["y"], va["y"]])
    dts = np.concatenate([tr["date"], va["date"]])
    dates = sorted(set(dts.tolist()))
    print(f"dates={dates}")
    for d in dates:
        print(f"  {d}: n={(dts == d).sum()}")
    Xt, yt, avt, pbt, tt = te["X"], te["y"], te["avail"], te["pb"], te["turn"]
    print(f"test={len(Xt)}  baseline(070 agent) class acc = {(pbt == yt).mean():.4f}")

    # 連続ブロックfold: val=dates[i]、train=それ以外の全ブロック
    probs, raws, caps = [], [], []
    for fi, va_d in enumerate(dates):
        mva = dts == va_d
        mtr = ~mva
        dtr = lgb.Dataset(X[mtr], label=y[mtr], feature_name=list(keys))
        dva = lgb.Dataset(X[mva], label=y[mva], reference=dtr)
        p = dict(PARAMS, num_class=len(classes))
        b = lgb.train(p, dtr, num_boost_round=8000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(150)])
        pr = b.predict(Xt, num_iteration=b.best_iteration)
        rw = b.predict(Xt, num_iteration=b.best_iteration, raw_score=True)
        probs.append(pr)
        raws.append(rw)
        caps.append(b.best_iteration)
        acc = float((masked_argmax(pr, avt) == yt).mean())
        ens = float((masked_argmax(np.mean(probs, axis=0), avt) == yt).mean())
        print(f"  [f{fi+1}] val={va_d} (n_train={mtr.sum()})  "
              f"best_iter={b.best_iteration:5d}  single_acc={acc:.4f}  "
              f"cumCV平均(K={fi+1})={ens:.4f}")
        if args.save_dir:
            b.save_model(os.path.join(args.save_dir, f"model_071_f{fi+1}.txt"))

    if max(caps) > 7000:
        print(f"!! WARNING: best_iter={max(caps)} は cap(8000) 拘束の疑い。capを上げて再実行")

    pred_e = masked_argmax(np.mean(probs, axis=0), avt)      # 確率平均
    pred_r = masked_argmax(np.mean(raws, axis=0), avt)       # 生スコア平均
    # f6 = val 07-14 / train 07-09..13 = 070と同一の切り方
    i070 = dates.index(max(dates))
    pred_1 = masked_argmax(probs[i070], avt)
    ae = float((pred_e == yt).mean())
    ar = float((pred_r == yt).mean())
    al = float((pred_1 == yt).mean())
    print("\n== test 07-15 ==")
    print(f"f{i070+1}単発(=070と同一の切り方) = {al:.4f}")
    print(f"CV平均 K={len(probs)} 確率平均        = {ae:.4f}   増分 = {ae-al:+.4f}")
    print(f"CV平均 K={len(probs)} 生スコア平均    = {ar:.4f}   増分 = {ar-al:+.4f}"
          f"   (確率平均との一致 {(pred_r == pred_e).mean():.4f})")
    print(f"baseline(070 npz)              = {(pbt == yt).mean():.4f}")
    # K本平均の飽和曲線（f1から順に足した場合）
    for k in range(2, len(probs) + 1):
        sub = float((masked_argmax(np.mean(probs[:k], axis=0), avt) == yt).mean())
        print(f"  K={k}平均 = {sub:.4f}")
    pred_last = pred_1
    fire = pred_e != pred_last
    print(f"fire(単発との差)={fire.sum()} CV正解={(pred_e[fire]==yt[fire]).sum()} "
          f"単発正解={(pred_last[fire]==yt[fire]).sum()}")
    for lo, hi, tag in ((0, 2, "t1-2"), (3, 5, "t3-5"), (6, 9, "t6-9"),
                        (10, 99, "t10+")):
        m = (tt >= lo) & (tt <= hi)
        if m.sum():
            print(f"  {tag:5s} n={m.sum():6d} CV={(pred_e[m]==yt[m]).mean():.3f} "
                  f"単発={(pred_last[m]==yt[m]).mean():.3f} "
                  f"base070={(pbt[m]==yt[m]).mean():.3f}")


if __name__ == "__main__":
    main()
