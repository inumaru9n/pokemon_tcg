"""model_070.txt (lightgbm) → model_070.npz (純Python/numpy推論用) 変換 (EXP-057)。

Kaggle評価環境にlightgbmが無いため、全ツリーをフラット配列に落とす。
検証: val決定データ上で lightgbm予測との argmax一致率を報告する。

usage:
  uv run python agents/070_058_retune/export_model_070.py [--check val.pkl]
"""

from __future__ import annotations

import argparse
import os
import pickle

import numpy as np

AGENT_DIR = os.path.dirname(os.path.abspath(__file__))


def export():
    import lightgbm as lgb

    bst = lgb.Booster(model_file=os.path.join(AGENT_DIR, "model_070.txt"))
    dump = bst.dump_model()
    num_class = bst.num_model_per_iteration()

    feat, thr, left, right, val, dflt = [], [], [], [], [], []
    roots, tree_cls = [], []

    def add(node) -> int:
        idx = len(feat)
        feat.append(-1)
        thr.append(0.0)
        left.append(-1)
        right.append(-1)
        val.append(0.0)
        dflt.append(0)
        if "split_feature" in node:
            assert node.get("decision_type", "<=") == "<=", node["decision_type"]
            feat[idx] = node["split_feature"]
            thr[idx] = node["threshold"]
            dflt[idx] = 1 if node.get("default_left", True) else 0
            li = add(node["left_child"])
            ri = add(node["right_child"])
            left[idx] = li
            right[idx] = ri
        else:
            val[idx] = node["leaf_value"]
        return idx

    for ti in dump["tree_info"]:
        roots.append(len(feat))
        tree_cls.append(ti["tree_index"] % num_class)
        add(ti["tree_structure"])

    out = os.path.join(AGENT_DIR, "model_070.npz")
    np.savez_compressed(
        out,
        feat=np.array(feat, dtype=np.int16),
        thr=np.array(thr, dtype=np.float64),
        left=np.array(left, dtype=np.int32),
        right=np.array(right, dtype=np.int32),
        val=np.array(val, dtype=np.float64),
        roots=np.array(roots, dtype=np.int32),
        tree_cls=np.array(tree_cls, dtype=np.int16),
        num_class=np.int32(num_class),
        num_feature=np.int32(bst.num_feature()),
    )
    sz = os.path.getsize(out) / 1e6
    print(f"exported: {len(roots)} trees, {len(feat)} nodes, {sz:.1f}MB -> {out}")
    return bst


def predict_pure(z, x):
    """純Python推論（main.pyに同じロジックを実装。ここでは検証用）。"""
    feat, thr, left, right, val = z["feat"], z["thr"], z["left"], z["right"], z["val"]
    scores = [0.0] * int(z["num_class"])
    for r, c in zip(z["roots"], z["tree_cls"]):
        i = int(r)
        while feat[i] >= 0:
            i = int(left[i]) if x[feat[i]] <= thr[i] else int(right[i])
        scores[int(c)] += float(val[i])
    return scores


def check(bst, pkl_path, n=3000):
    with open(pkl_path, "rb") as f:
        rows = pickle.load(f)["rows"][:n]
    z = np.load(os.path.join(AGENT_DIR, "model_070.npz"))
    X = np.array([r["x"] for r in rows], dtype=np.float64)
    P = bst.predict(X)
    import time
    t0 = time.perf_counter()
    agree = 0
    for i, r in enumerate(rows):
        s = predict_pure(z, X[i])
        avail = np.array(r["avail"], dtype=bool)
        a1 = int(np.where(avail, P[i], -np.inf).argmax())
        s_m = [(s[j] if avail[j] else -1e18) for j in range(len(s))]
        a2 = int(np.argmax(s_m))
        agree += a1 == a2
    dt = (time.perf_counter() - t0) / len(rows) * 1000
    print(f"argmax parity: {agree}/{len(rows)} = {agree/len(rows):.4f}   "
          f"pure-python {dt:.2f} ms/pred")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", default=None)
    args = ap.parse_args()
    b = export()
    if args.check:
        check(b, args.check)
