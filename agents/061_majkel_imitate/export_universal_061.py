"""model_universal_061.txt (lightgbm binary) → .npz（純Python推論用、Kaggle）。"""
from __future__ import annotations

import os

import numpy as np

AD = os.path.dirname(os.path.abspath(__file__))


def export():
    import lightgbm as lgb
    bst = lgb.Booster(model_file=os.path.join(AD, "model_universal_061.txt"))
    dump = bst.dump_model()
    feat, thr, left, right, val, roots = [], [], [], [], [], []

    def add(node):
        idx = len(feat)
        feat.append(-1); thr.append(0.0); left.append(-1); right.append(-1)
        val.append(0.0)
        if "split_feature" in node:
            assert node.get("decision_type", "<=") == "<="
            feat[idx] = node["split_feature"]; thr[idx] = node["threshold"]
            left[idx] = add(node["left_child"]); right[idx] = add(node["right_child"])
        else:
            val[idx] = node["leaf_value"]
        return idx

    for ti in dump["tree_info"]:
        roots.append(len(feat)); add(ti["tree_structure"])
    out = os.path.join(AD, "model_universal_061.npz")
    np.savez_compressed(out,
                        feat=np.array(feat, dtype=np.int16),
                        thr=np.array(thr, dtype=np.float64),
                        left=np.array(left, dtype=np.int32),
                        right=np.array(right, dtype=np.int32),
                        val=np.array(val, dtype=np.float64),
                        roots=np.array(roots, dtype=np.int32))
    print(f"exported {len(roots)} trees, {len(feat)} nodes, "
          f"{os.path.getsize(out)/1e6:.1f}MB -> {out}")
    return bst


def check(bst, pkl, n=2000):
    import pickle, math
    rows = pickle.load(open(pkl, "rb"))["rows"][:n]
    z = np.load(os.path.join(AD, "model_universal_061.npz"))
    X = np.array([r["x"] for r in rows], dtype=np.float64)
    P = bst.predict(X)

    def pure(x):
        tot = 0.0
        for r in z["roots"]:
            i = int(r)
            while z["feat"][i] >= 0:
                i = int(z["left"][i]) if x[z["feat"][i]] <= z["thr"][i] else int(z["right"][i])
            tot += float(z["val"][i])
        return tot
    mx = max(abs(1/(1+math.exp(-pure(X[i]))) - P[i]) for i in range(len(rows)))
    print(f"npz parity: max|sigmoid(pure)-lgb| = {mx:.2e}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", default=None)
    a = ap.parse_args()
    b = export()
    if a.check:
        check(b, a.check)
