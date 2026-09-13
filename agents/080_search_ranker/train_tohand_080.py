"""EXP-080 TO_HAND二値ランカーの学習 (I-118)。

各TO_HAND決定について候補カードを「取る」スコアでランクし、上位k枚を選ぶ。
評価は「枚単位一致率」（本人の選択集合と予測集合の重なり / k）を主指標にし、
ベース方策（075のRULE_C、収集時に記録したrp）と比較する。

usage:
  uv run python agents/080_search_ranker/train_tohand_080.py \
      --dev dev.pkl --val val.pkl --save model_tohand_080
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from collections import defaultdict

import numpy as np

REPO_ROOT = "/Users/akira/kaggle/pokemon_tcg"
AGENT_DIR = os.path.join(REPO_ROOT, "agents/080_search_ranker")
for _p in (REPO_ROOT, AGENT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def load(path):
    with open(path, "rb") as f:
        d = pickle.load(f)
    rows = d["rows"]
    X = np.array([r["x"] for r in rows], dtype=np.float32)
    y = np.array([r["y"] for r in rows], dtype=np.int32)
    rp = np.array([r["rp"] for r in rows], dtype=np.int32)
    dec = [(r["ep"], r["step"]) for r in rows]
    k = np.array([r["k"] for r in rows], dtype=np.int32)
    cid = np.array([r["cid"] for r in rows], dtype=np.int64)
    return d["feat_keys"], X, y, rp, dec, k, cid


def _groups(dec):
    g = defaultdict(list)
    for i, key in enumerate(dec):
        g[key].append(i)
    return g


def piecewise_agreement(score, y, dec, k):
    """各決定で上位k枚を予測 → 枚単位一致率（∑重なり / ∑k）。"""
    g = _groups(dec)
    inter = tot = 0
    for key, idxs in g.items():
        idxs = np.array(idxs)
        kk = int(k[idxs[0]])
        order = idxs[np.argsort(-score[idxs])]
        pred = set(order[:kk].tolist())
        actual = set(i for i in idxs.tolist() if y[i] == 1)
        inter += len(pred & actual)
        tot += kk
    return inter / max(1, tot), len(g)


def rp_agreement(rp, y, dec, k):
    """ベース方策(rp=収集時のRULE_C選択)の枚単位一致率。"""
    g = _groups(dec)
    inter = tot = 0
    for key, idxs in g.items():
        kk = int(k[idxs[0]])
        pred = set(i for i in idxs if rp[i] == 1)
        actual = set(i for i in idxs if y[i] == 1)
        inter += len(pred & actual)
        tot += kk
    return inter / max(1, tot), len(g)


def exact_rate(score_or_rp, y, dec, k, is_rp=False):
    g = _groups(dec)
    ok = 0
    for key, idxs in g.items():
        ia = np.array(idxs)
        kk = int(k[ia[0]])
        if is_rp:
            pred = set(i for i in idxs if score_or_rp[i] == 1)
        else:
            pred = set(ia[np.argsort(-score_or_rp[ia])][:kk].tolist())
        actual = set(i for i in idxs if y[i] == 1)
        ok += pred == actual
    return ok / len(g)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--save", default=None)
    args = ap.parse_args()

    keys, Xd, yd, rpd, decd, kd, cidd = load(args.dev)
    kv_, Xv, yv, rpv, decv, kv, cidv = load(args.val)
    assert keys == kv_
    print(f"dev rows={len(Xd)} decisions={len(set(decd))}  "
          f"val rows={len(Xv)} decisions={len(set(decv))}  feat={len(keys)}")

    b_agr, _ = rp_agreement(rpv, yv, decv, kv)
    print(f"base(RULE_C rp) piecewise agreement = {b_agr:.4f}  "
          f"exact = {exact_rate(rpv, yv, decv, kv, is_rp=True):.4f}")

    import lightgbm as lgb
    dtr = lgb.Dataset(Xd, label=yd, feature_name=list(keys))
    dva = lgb.Dataset(Xv, label=yv, reference=dtr)
    params = dict(objective="binary", metric="binary_logloss",
                  learning_rate=0.05, num_leaves=31, min_data_in_leaf=50,
                  feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1,
                  verbose=-1, seed=42)
    bst = lgb.train(params, dtr, num_boost_round=1000, valid_sets=[dva],
                    callbacks=[lgb.early_stopping(60), lgb.log_evaluation(100)])
    sc_val = bst.predict(Xv, num_iteration=bst.best_iteration)
    m_agr, ndec = piecewise_agreement(sc_val, yv, decv, kv)
    print(f"model piecewise agreement = {m_agr:.4f}  (base {b_agr:.4f}, "
          f"n_decisions={ndec}, best_iter={bst.best_iteration})")
    print(f"exact-set: model {exact_rate(sc_val, yv, decv, kv):.4f}")
    imp = sorted(zip(keys, bst.feature_importance("gain")),
                 key=lambda kvp: -kvp[1])[:12]
    print("top gains:", [(k, int(v)) for k, v in imp])

    if args.save:
        bst.save_model(args.save + ".txt", num_iteration=bst.best_iteration)
        print(f"saved -> {args.save}.txt")


if __name__ == "__main__":
    main()
