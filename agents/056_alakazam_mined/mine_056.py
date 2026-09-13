"""EXP-056 dtype別ルールマイニング (I-110)。

決定pickle（calib_056_decisions.py collect産、t_*/c_*/o_*入り）から、ターゲット行動
クラスごとに「本人はこのターンこの行動を使うか」のターン単位ラベルを作り、root状態
特徴で浅い決定木を学習して分岐をテキスト出力する。人間が読んでルール候補を選別する。

- 決定単位でなくターン単位のラベル: 順序等価（今でなく後で使う）のノイズを除去。
  「クラスが最初に選択可能になった決定」の root 特徴 → 「そのターン中に使ったか」
- root特徴 = F行列のうち a_*全0 かつ attack=0 の行（=無行動END終端）= 現盤面の記述

usage:
  uv run python agents/056_alakazam_mined/mine_056.py --dev dec56_dev.pkl \
      --val dec56_val.pkl [--target a_dud a_fez a_attach attack] [--depth 3]
"""

from __future__ import annotations

import argparse
import pickle
from collections import defaultdict

import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text

# root状態を記述する特徴（ライン依存の差分特徴は除外）
STATE_PREFIXES = ("s_", "t_", "c_", "o_", "h_")
EXTRA_STATE = ["hand"]  # END終端のhand=現手札+unk


def _load_turns(pkl_path):
    """(ep, turn) ごとに [record,...]（step順）を返す。qualifying recordのみ。"""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    keys = data["feat_keys"]
    ki = {k: i for i, k in enumerate(keys)}
    acols = [i for k, i in ki.items() if k.startswith("a_")]
    atk_c = ki["attack"]
    turns = defaultdict(list)
    for r in data["records"]:
        if r["forced"] or r["fixed"] or "F" not in r:
            continue
        turns[(r["ep"], r["turn"])].append(r)
    return turns, keys, ki, acols, atk_c


def _root_feat(r, acols, atk_c):
    F = r["F"]
    m = (np.abs(F[:, acols]).sum(axis=1) == 0) & (F[:, atk_c] == 0)
    idx = np.where(m)[0]
    return F[idx[0]].astype(np.float32) if len(idx) else None


def _cls_of(r, i):
    ocls = r.get("ocls") or []
    return ocls[i] if 0 <= i < len(ocls) else None


def build_dataset(turns, ki, acols, atk_c, target):
    """target='a_dud'等: 最初に選択可能になった決定のroot特徴 → ターン内使用ラベル。
    target='attack': ターン最初の決定のroot特徴 → このターン攻撃で締めたか。"""
    X, y, base_pred, meta = [], [], [], []
    for (ep, turn), recs in turns.items():
        recs = sorted(recs, key=lambda r: r["step"])
        if target == "attack":
            r0 = recs[0]
            rf = _root_feat(r0, acols, atk_c)
            if rf is None:
                continue
            lbl = any(r["dtype"] == "main_attack" for r in recs)
            bp = None  # baseのターン内挙動は多段のため予測列にしない
            X.append(rf)
            y.append(lbl)
            base_pred.append(bp)
            meta.append((ep, turn))
            continue
        first = None
        for r in recs:
            ocls = r.get("ocls") or []
            if target in ocls:
                first = r
                break
        if first is None:
            continue
        rf = _root_feat(first, acols, atk_c)
        if rf is None:
            continue
        lbl = any(_cls_of(r, r["actual"][0]) == target
                  for r in recs if r["actual"])
        bp = _cls_of(first, first.get("pb", -1)) == target  # baseは「今すぐ使う」か
        X.append(rf)
        y.append(lbl)
        base_pred.append(bp)
        meta.append((ep, turn))
    return np.array(X), np.array(y, dtype=bool), base_pred, meta


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--target", nargs="+",
                    default=["a_dud", "a_fez", "a_attach", "attack"])
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--min-leaf", type=int, default=80, dest="min_leaf")
    args = ap.parse_args()

    dev_turns, keys, ki, acols, atk_c = _load_turns(args.dev)
    val_turns, keys_v, _ki2, _ac2, _at2 = _load_turns(args.val)
    assert keys == keys_v
    fsel = [i for i, k in enumerate(keys)
            if k.startswith(STATE_PREFIXES) or k in EXTRA_STATE]
    fnames = [keys[i] for i in fsel]

    for tgt in args.target:
        Xd, yd, _bd, _md = build_dataset(dev_turns, ki, acols, atk_c, tgt)
        Xv, yv, bv, _mv = build_dataset(val_turns, ki, acols, atk_c, tgt)
        if len(Xd) < 200:
            print(f"== {tgt}: dev n={len(Xd)} 不足、スキップ ==")
            continue
        Xd_, Xv_ = Xd[:, fsel], Xv[:, fsel]
        maj = max(yv.mean(), 1 - yv.mean())
        print(f"\n===== target={tgt} dev n={len(Xd)} (pos {yd.mean():.2f})  "
              f"val n={len(Xv)} (pos {yv.mean():.2f}, majority {maj:.3f}) =====")
        for depth in (2, args.depth, args.depth + 1):
            t = DecisionTreeClassifier(max_depth=depth,
                                       min_samples_leaf=args.min_leaf,
                                       random_state=0)
            t.fit(Xd_, yd)
            acc_d = t.score(Xd_, yd)
            acc_v = t.score(Xv_, yv)
            print(f"-- depth={depth}: dev {acc_d:.3f} / val {acc_v:.3f}")
            if depth == args.depth:
                print(export_text(t, feature_names=fnames,
                                  max_depth=depth).rstrip())
        # base（052の「最初の可能決定で即使用」）とターン内実挙動の一致
        bvv = [b for b in bv if b is not None]
        if bvv and tgt != "attack":
            base_acc = np.mean([(b == yy) for b, yy in zip(bv, yv)
                                if b is not None])
            print(f"base(即時使用=ターン内使用とみなす)一致: {base_acc:.3f}")


if __name__ == "__main__":
    main()
