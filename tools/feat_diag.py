"""抽出済み教師データの**識別不能率**を測る（特徴器の品質を学習前に判定する）。

識別不能 = 教師が選んだ選択肢と**特徴が完全一致する別の選択肢**が同じ決定内に存在する状態。
このとき、どんなモデルでも教師の選択を再現できない（原理的な上限が下がる）。
EXP-094ではこの診断でv5の穴 A-1（PLAYのカード未解決）と A-2（貼り先未記録）を発見した。

学習を回さずに特徴器の良否を比較できるので、汎用ジェネレータ版と手作り版の比較に使う。

usage:
  uv run python tools/feat_diag.py data/nn/train.pkl
  uv run python tools/feat_diag.py a.pkl b.pkl --label handmade gen
"""

from __future__ import annotations

import argparse
import pickle
from collections import Counter, defaultdict


def diag(path: str) -> dict:
    with open(path, "rb") as fh:
        d = pickle.load(fh)
    games = d["games"]
    n_dec = n_bad = 0
    n_main = n_main_bad = 0
    per_ctx = defaultdict(lambda: [0, 0])
    n_opt = 0
    dup_pairs = Counter()
    for g in games:
        for dec in g["decisions"]:
            ch = dec["chosen"]
            if not ch:
                continue
            keys = [(c, tuple(v), cl) for c, v, cl
                    in zip(dec["opt_cid"], dec["opt_vec"], dec["opt_cls"])]
            cnt = Counter(keys)
            bad = any(cnt[keys[i]] > 1 for i in ch if i < len(keys))
            ctx = dec["ctx"]
            n_dec += 1
            n_opt += len(keys)
            per_ctx[ctx][0] += 1
            if bad:
                n_bad += 1
                per_ctx[ctx][1] += 1
                dup_pairs[ctx] += 1
            if ctx == 0:
                n_main += 1
                n_main_bad += int(bad)
    # 教師が選んだMAIN行動のクラス分布（クラス分類の設計が妥当かを見る。
    # 「other」に大量に落ちていたら、その行動タイプがクラス頭で区別できていない）
    cls_names = d.get("classes") or []
    chosen_cls = Counter()
    for g in games:
        for dec in g["decisions"]:
            if dec["ctx"] != 0:
                continue
            for i in dec["chosen"]:
                if i < len(dec["opt_cls"]) and dec["opt_cls"][i] >= 0:
                    c = dec["opt_cls"][i]
                    chosen_cls[cls_names[c] if c < len(cls_names) else f"#{c}"] += 1
    return {"path": path, "games": len(games), "dec": n_dec, "bad": n_bad,
            "main": n_main, "main_bad": n_main_bad, "per_ctx": dict(per_ctx),
            "n_feat": len(d["feat_keys"]), "avg_opt": n_opt / max(1, n_dec),
            "n_cls": int(d.get("n_cls") or 0), "chosen_cls": chosen_cls}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pkl", nargs="+")
    ap.add_argument("--label", nargs="*", default=None)
    ap.add_argument("--top-ctx", type=int, default=8)
    args = ap.parse_args()
    labels = args.label or [p.split("/")[-1] for p in args.pkl]

    rows = [diag(p) for p in args.pkl]
    print(f"{'':<16}{'試合':>7}{'決定':>9}{'盤面次元':>9}{'平均選択肢':>11}"
          f"{'識別不能':>10}{'MAIN識別不能':>14}")
    for lb, r in zip(labels, rows):
        print(f"{lb:<16}{r['games']:>7}{r['dec']:>9}{r['n_feat']:>9}"
              f"{r['avg_opt']:>11.2f}"
              f"{r['bad']/max(1,r['dec']):>9.2%}"
              f"{r['main_bad']/max(1,r['main']):>13.2%}")

    print("\nctx別の識別不能率（決定数の多い順）")
    for lb, r in zip(labels, rows):
        top = sorted(r["per_ctx"].items(), key=lambda kv: -kv[1][0])[:args.top_ctx]
        s = "  ".join(f"ctx{c}:{b}/{n}={b/max(1,n):.1%}" for c, (n, b) in top)
        print(f"  {lb}: {s}")

    print("\n教師が選んだMAIN行動のクラス分布（上位10 / クラス総数）")
    for lb, r in zip(labels, rows):
        cc = r["chosen_cls"]
        tot = sum(cc.values()) or 1
        oth = sum(v for k, v in cc.items() if k in ("other", "ab_other", "atk_other"))
        top = "  ".join(f"{k}:{v/tot:.1%}" for k, v in cc.most_common(10))
        print(f"  {lb} (n_cls={r['n_cls']}, 使用{len(cc)}種, "
              f"other系{oth/tot:.1%}): {top}")


if __name__ == "__main__":
    main()
