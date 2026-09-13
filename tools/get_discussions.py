"""コンペ公式Discussionの取得（kaggle CLI の competitions topics list/show を利用）。

使い方（リポジトリルートから）:
  uv run tools/get_discussions.py                 # vote数10以上の新規・更新トピックのみ取得
  uv run tools/get_discussions.py --min-votes 0   # 全トピックを対象にする
  uv run tools/get_discussions.py --force         # 閾値内の全トピック再取得

出力:
  data/discussions/index.json      # トピック一覧（id, title, votes, commentCount, postDate）
  data/discussions/<id>.md         # 本文+全コメントのトランスクリプト（分析の入力）

仕組みのメモ:
  - `topics show --format json` は本文を含まない。本文はtable出力から抽出し、
    コメント全文はJSONから取る（table側のコメントは省略される）
  - 新規/更新の判定は commentCount の変化（indexに保存した前回値と比較）
  - vote数フィルタは取得対象の絞り込みのみ。indexには閾値未満も含めて全件記録する
"""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

from kaggle_util import fetch_all, kaggle_json, kaggle_text, load_index, save_index

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "discussions"
COMPETITION = "pokemon-tcg-ai-battle"


def strip_html(s: str) -> str:
    s = re.sub(r"<br\s*/?>|</p>", "\n", s)
    s = re.sub(r"<[^>]+>", "", s)
    return html.unescape(s).strip()


def list_all_topics() -> list[dict]:
    topics = []
    page = 1
    while True:
        try:
            batch = kaggle_json(["competitions", "topics", "list", COMPETITION,
                                 "--format", "json", "-p", str(page)])
        except Exception:
            break  # 最終ページを越えると "No topics found"（非JSON）が返る
        if isinstance(batch, dict):
            batch = batch.get("items") or batch.get("topics") or []
        if not batch:
            break
        topics.extend(batch)
        if len(batch) < 20:
            break
        page += 1
    return topics


def extract_body_from_table(table_out: str) -> str:
    """table出力のヘッダ部（Votes行）と 'Comments:' の間が本文。"""
    lines = table_out.splitlines()
    start = 0
    for i, line in enumerate(lines):
        if line.strip().startswith("Votes:"):
            start = i + 1
            break
    end = len(lines)
    for i, line in enumerate(lines):
        if line.strip() == "Comments:":
            end = i
            break
    return "\n".join(lines[start:end]).strip()


def fetch_topic(topic: dict) -> None:
    tid = topic["id"]
    ref = f"{COMPETITION}/{tid}"
    detail = kaggle_json(["competitions", "topics", "show", ref,
                          "--format", "json", "--page-size", "200"])
    table = kaggle_text(["competitions", "topics", "show", ref, "--page-size", "200"])
    body = extract_body_from_table(table)

    lines = [
        f"# {topic['title']}",
        "",
        f"- topic_id: {tid} / author: {topic.get('authorName') or '(不明)'} / votes: {topic.get('votes', 0)} / comments: {topic.get('commentCount', 0)} / posted: {topic.get('postDate', '')}",
        "",
        "## 本文",
        "",
        body or "(本文なし)",
        "",
        "## コメント",
        "",
    ]
    comments = detail.get("comments") or []
    if not comments:
        lines.append("(コメントなし)")
    for c in comments:
        lines.append(f"### {c.get('authorName') or '(不明)'} ({c.get('postDate', '')}) [+{c.get('votes', 0)}]")
        lines.append("")
        lines.append(strip_html(c.get("content", "")))
        lines.append("")
    (OUT_DIR / f"{tid}.md").write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-votes", type=int, default=10, help="取得対象の最低vote数")
    ap.add_argument("--force", action="store_true", help="閾値内の全トピック再取得")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    index_path = OUT_DIR / "index.json"
    old = load_index(index_path, key="id", force=args.force)

    topics = list_all_topics()
    print(f"{len(topics)} topics found")
    targets = [t for t in topics if t.get("votes", 0) >= args.min_votes]
    changed = []
    for t in targets:
        prev = old.get(t["id"])
        if prev is None or not (OUT_DIR / f"{t['id']}.md").exists():
            changed.append((t, "NEW"))
        elif prev.get("commentCount") != t.get("commentCount"):
            changed.append((t, "UPDATED"))

    n_ok = fetch_all(changed, fetch_topic,
                     lambda t: f"{t['id']} {t['title'][:70]} | Votes: {t.get('votes', 0)}")
    save_index(index_path, topics)
    print(f"done: {n_ok}/{len(changed)} fetched, index -> {index_path}")


if __name__ == "__main__":
    main()
