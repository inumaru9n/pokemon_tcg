"""model_tohand_080.txt (lightgbm) → model_tohand_080.npz (純Python推論用) 変換 (EXP-080)。

Kaggle評価環境にlightgbmが無いため、全ツリーをフラット配列に落とす（二値なのでtree_cls不要）。
検証: val決定データ上で「決定内の選択集合（上位k）」の一致率を lightgbm予測と比較する。

usage:
  uv run python agents/080_search_ranker/export_tohand_080.py --check val.pkl
"""

from __future__ import annotations

import argparse
import os
import pickle
from collections import defaultdict

import numpy as np

AGENT_DIR = os.path.dirname(os.path.abspath(__file__))


def export():
    import lightgbm as lgb

    bst = lgb.Booster(model_file=os.path.join(AGENT_DIR, "model_tohand_080.txt"))
    dump = bst.dump_model()

    feat, thr, left, right, val = [], [], [], [], []
    roots = []

    def add(node) -> int:
        idx = len(feat)
        feat.append(-1)
        thr.append(0.0)
        left.append(-1)
        right.append(-1)
        val.append(0.0)
        if "split_feature" in node:
            assert node.get("decision_type", "<=") == "<=", node["decision_type"]
            feat[idx] = node["split_feature"]
            thr[idx] = node["threshold"]
            li = add(node["left_child"])
            ri = add(node["right_child"])
            left[idx] = li
            right[idx] = ri
        else:
            val[idx] = node["leaf_value"]
        return idx

    for ti in dump["tree_info"]:
        roots.append(len(feat))
        add(ti["tree_structure"])

    out = os.path.join(AGENT_DIR, "model_tohand_080.npz")
    np.savez_compressed(
        out,
        feat=np.array(feat, dtype=np.int16),
        thr=np.array(thr, dtype=np.float64),
        left=np.array(left, dtype=np.int32),
        right=np.array(right, dtype=np.int32),
        val=np.array(val, dtype=np.float64),
        roots=np.array(roots, dtype=np.int32),
        num_feature=np.int32(bst.num_feature()),
    )
    sz = os.path.getsize(out) / 1e6
    print(f"exported: {len(roots)} trees, {len(feat)} nodes, {sz:.2f}MB -> {out}")
    return bst


def score_pure(z, x) -> float:
    feat, thr, left, right, val = z["feat"], z["thr"], z["left"], z["right"], z["val"]
    tot = 0.0
    for r in z["roots"]:
        i = int(r)
        while feat[i] >= 0:
            i = int(left[i]) if x[feat[i]] <= thr[i] else int(right[i])
        tot += float(val[i])
    return tot


def check(bst, pkl_path):
    with open(pkl_path, "rb") as f:
        rows = pickle.load(f)["rows"]
    z = np.load(os.path.join(AGENT_DIR, "model_tohand_080.npz"))
    X = np.array([r["x"] for r in rows], dtype=np.float64)
    P = bst.predict(X)
    import time
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups[(r["ep"], r["step"])].append(i)
    t0 = time.perf_counter()
    S = [score_pure(z, X[i]) for i in range(len(rows))]
    dt = (time.perf_counter() - t0) / len(rows) * 1000
    same = 0
    for key, idxs in groups.items():
        ia = np.array(idxs)
        kk = rows[ia[0]]["k"]
        p1 = set(ia[np.argsort(-P[ia])][:kk].tolist())
        p2 = set(ia[np.argsort(-np.array([S[i] for i in ia]))][:kk].tolist())
        same += p1 == p2
    print(f"selection parity: {same}/{len(groups)} = {same/len(groups):.4f}   "
          f"pure-python {dt:.2f} ms/candidate")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", default=None)
    args = ap.parse_args()
    b = export()
    if args.check:
        check(b, args.check)
