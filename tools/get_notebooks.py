"""コンペ公開ノートブックの取得（kaggle CLI の kernels list/pull を利用）。

使い方（リポジトリルートから）:
  uv run tools/get_notebooks.py                  # vote数30以上の新規・更新ノートブックのみ取得
  uv run tools/get_notebooks.py --min-votes 10   # 閾値を下げて取得
  uv run tools/get_notebooks.py --force          # 閾値内の全ノートブック再取得

出力:
  data/notebooks/index.json    # ノートブック一覧（ref, title, author, totalVotes, lastRunTime）
  data/notebooks/<slug>.ipynb  # ノートブック本体（分析の入力）

仕組みのメモ:
  - 新規/更新の判定は lastRunTime の変化（indexに保存した前回値と比較）。
    更新されたノートブックは再取得されるので、要約の更新要否は収穫スキル側で判断する
  - vote数フィルタは取得対象の絞り込みのみ。indexには閾値未満も含めて全件記録する
"""

from __future__ import annotations

import argparse
from pathlib import Path

from kaggle_util import fetch_all, kaggle_json, kaggle_text, load_index, save_index

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "notebooks"
COMPETITION = "pokemon-tcg-ai-battle"


def list_all_notebooks() -> list[dict]:
    notebooks = []
    page = 1
    while True:
        batch = kaggle_json(["kernels", "list", "--competition", COMPETITION,
                             "--sort-by", "voteCount", "--page-size", "200",
                             "-p", str(page), "--format", "json"])
        if not batch:
            break
        notebooks.extend(batch)
        if len(batch) < 200:
            break
        page += 1
    return notebooks


def fetch_notebook(nb: dict) -> None:
    kaggle_text(["kernels", "pull", nb["ref"], "-p", str(OUT_DIR)])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-votes", type=int, default=30, help="取得対象の最低vote数")
    ap.add_argument("--force", action="store_true", help="閾値内の全ノートブック再取得")
    args = ap.parse_args()

    index_path = OUT_DIR / "index.json"
    old = load_index(index_path, key="ref", force=args.force)

    notebooks = list_all_notebooks()
    print(f"{len(notebooks)} notebooks found")
    targets = [n for n in notebooks if n.get("totalVotes", 0) >= args.min_votes]
    changed = []
    for n in targets:
        prev = old.get(n["ref"])
        slug = n["ref"].split("/")[-1]
        if prev is None or not (OUT_DIR / f"{slug}.ipynb").exists():
            changed.append((n, "NEW"))
        elif prev.get("lastRunTime") != n.get("lastRunTime"):
            changed.append((n, "UPDATED"))

    n_ok = fetch_all(changed, fetch_notebook,
                     lambda n: f"{n['title'][:70]} by {n.get('author', '')} | Votes: {n.get('totalVotes', 0)}")
    save_index(index_path, notebooks)
    print(f"done: {n_ok}/{len(changed)} fetched, index -> {index_path}")


if __name__ == "__main__":
    main()
