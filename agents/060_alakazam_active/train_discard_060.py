"""EXP-059 DISCARD二値分類器の学習 (I-116)。

各DISCARD決定について候補カードを捨てるスコアでランクし、上位k枚を捨てとする。
評価は「枚単位一致率」（本人の捨て集合と予測捨て集合の重なり / k）を主指標にし、
052ベース（RULE_DのカードID固定スコア）と比較する。

usage:
  uv run python agents/060_alakazam_active/train_discard_060.py \
      --dev dev.pkl --val val.pkl --save model_discard_060
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from collections import defaultdict

import numpy as np

REPO_ROOT = "/Users/akira/kaggle/pokemon_tcg"
AGENT_DIR = os.path.join(REPO_ROOT, "agents/060_alakazam_active")
for _p in (REPO_ROOT, AGENT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def load(path):
    with open(path, "rb") as f:
        d = pickle.load(f)
    rows = d["rows"]
    X = np.array([r["x"] for r in rows], dtype=np.float32)
    y = np.array([r["y"] for r in rows], dtype=np.int32)
    dec = [(r["ep"], r["step"]) for r in rows]
    k = np.array([r["k"] for r in rows], dtype=np.int32)
    cid = np.array([r["cid"] for r in rows], dtype=np.int64)
    return d["feat_keys"], X, y, dec, k, cid


def _groups(dec):
    """決定ID → 行indexリスト。"""
    g = defaultdict(list)
    for i, key in enumerate(dec):
        g[key].append(i)
    return g


def piecewise_agreement(score, y, dec, k):
    """各決定で上位k枚を捨て予測 → 枚単位一致率（∑重なり / ∑k）。"""
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


def base_rule_d_scores(cid):
    """052のRULE_D固定スコア（main.pyと同値）。"""
    import feat_060 as ft
    table = {
        ft.Nighttime_Mine: 100, ft.Shaymin: 92, ft.Hilda: 88, ft.Boss_Orders: 84,
        ft.Dawn: 80, ft.Rare_Candy: 76, ft.Sacred_Ash: 72, ft.Buddy_Buddy_Poffin: 68,
        ft.Enhanced_Hammer: 64, ft.Alakazam: 60, ft.Basic_Psychic_Energy: 56,
        ft.Xerosic: 50, ft.Fezandipiti_ex: 46, ft.Poke_Pad: 44, ft.Kadabra: 42,
        ft.Telepath_Psychic_Energy: 40, ft.Dudunsparce: 38, ft.Dunsparce: 34,
        ft.Enriching_Energy: 30, ft.Night_Stretcher: 26, ft.Lanas_Aid: 22, ft.Abra: 8,
    }
    return np.array([table.get(int(c), 55) for c in cid], dtype=np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--save", default=None)
    args = ap.parse_args()

    keys, Xd, yd, decd, kd, cidd = load(args.dev)
    kv_, Xv, yv, decv, kv, cidv = load(args.val)
    assert keys == kv_
    print(f"dev rows={len(Xd)} decisions={len(set(decd))}  "
          f"val rows={len(Xv)} decisions={len(set(decv))}  feat={len(keys)}")

    # ベース（RULE_D）
    base_val = base_rule_d_scores(cidv)
    b_agr, _ = piecewise_agreement(base_val, yv, decv, kv)
    print(f"base(RULE_D) piecewise agreement = {b_agr:.4f}")

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
    # 決定単位の完全一致も参考に
    def exact(score):
        g = _groups(decv)
        ok = 0
        for key, idxs in g.items():
            idxs = np.array(idxs)
            kk = int(kv[idxs[0]])
            pred = set(idxs[np.argsort(-score[idxs])][:kk].tolist())
            actual = set(i for i in idxs.tolist() if yv[i] == 1)
            ok += pred == actual
        return ok / len(g)
    print(f"exact-set: base {exact(base_val):.4f}  model {exact(sc_val):.4f}")
    imp = sorted(zip(keys, bst.feature_importance("gain")),
                 key=lambda kvp: -kvp[1])[:12]
    print("top gains:", [(k, int(v)) for k, v in imp])

    if args.save:
        bst.save_model(args.save + ".txt", num_iteration=bst.best_iteration)
        print(f"saved -> {args.save}.txt")


if __name__ == "__main__":
    main()
