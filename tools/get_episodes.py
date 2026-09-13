"""公式日次エピソードデータセット（pokemon-tcg-ai-battle-episodes-*）の取得・要約・メタレポート生成。

使い方（リポジトリルートから）:
  uv run tools/get_episodes.py download --latest            # indexを見て最新日をdata/episodes/へ
  uv run tools/get_episodes.py download --date 2026-07-03
  uv run tools/get_episodes.py summarize --date 2026-07-03  # zipを直接読んで summary.csv を生成
  uv run tools/get_episodes.py report --date 2026-07-03     # knowledge/episodes/<date>.md を生成

データの置き場所:
  data/episodes/<date>/pokemon-tcg-ai-battle-episodes-<date>.zip  (約750MB、リポジトリに残る)
  data/episodes/<date>/summary.csv   (1プレイヤー1行の要約。zipの展開はしない)
  knowledge/episodes/<date>.md           (アーキタイプ分布・対面勝率などのレポート)

エピソードJSONはKaggle標準リプレイ形式:
  steps[i][player] = {action, observation, reward, status}
  - デッキ提出: 各プレイヤー最初の len(action)==60 のstep（カードIDのリスト）
  - observation.remainingOverageTime: 600秒から減る時間プール（actTimeout=0のため全消費がここから引かれる）
  - observation.search_begin_input: 探索API入力のシリアライズ状態。実対戦ログからの方策再現・模倣学習に使える可能性
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
import zipfile
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

from kaggle_util import kaggle_run

ROOT = Path(__file__).resolve().parent.parent
EPISODES_DIR = ROOT / "data" / "episodes"
META_DIR = ROOT / "knowledge" / "episodes"
INDEX_SLUG = "kaggle/pokemon-tcg-ai-battle-episodes-index"
DAILY_SLUG = "kaggle/pokemon-tcg-ai-battle-episodes-{date}"

sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------- card data

def _card_table() -> dict[int, dict]:
    from cg.api import all_card_data

    table = {}
    for c in all_card_data():
        table[c.cardId] = {
            "name": c.name,
            "is_pokemon": c.cardType == 0,
            "stage": 2 if c.stage2 else (1 if c.stage1 else 0),
            "mega": c.megaEx,
            "ex": c.ex,
        }
    return table


def classify_deck(card_ids: list[int], table: dict[int, dict]) -> str:
    """デッキの看板ポケモンを推定してアーキタイプ名とする簡易ヒューリスティック。

    採用基準: megaEx > 進化段階が高い > ex > 採用枚数が多い。megaExを最優先にする理由は
    Mega Starmie ex（Staryuからのstage1進化）が同居するstage2ポケモンに看板を奪われる誤分類を
    避けるため（EXP-017で発覚。megaExはサイド3枚・高HPでほぼ確実にデッキの主軸）。
    stageをexより優先するのは、stage2の主軸（例: Alakazam）がテックのbasic ex（例: Fezandipiti ex）に
    看板を奪われるのを防ぐため。誤分類率は未検証。
    """
    counts = Counter(card_ids)
    best = None
    for cid, n in counts.items():
        c = table.get(cid)
        if not c or not c["is_pokemon"]:
            continue
        key = (c["mega"], c["stage"], c["ex"], n)
        if best is None or key > best[0]:
            best = (key, c["name"])
    return best[1] if best else "unknown"


def deck_signature(card_ids: list[int]) -> str:
    sig = ",".join(f"{cid}x{n}" for cid, n in sorted(Counter(card_ids).items()))
    return hashlib.sha1(sig.encode()).hexdigest()[:10]


# ---------------------------------------------------------------- download

def latest_date() -> str:
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        kaggle_run(["datasets", "download", INDEX_SLUG, "--unzip", "-p", td])
        rows = list(csv.DictReader(open(Path(td) / "manifest.csv")))
    return max(r["date"] for r in rows)


def cmd_download(args: argparse.Namespace) -> None:
    date = latest_date() if args.latest else args.date
    if not date:
        sys.exit("--date か --latest を指定してください")
    dest = EPISODES_DIR / date
    zip_path = dest / f"pokemon-tcg-ai-battle-episodes-{date}.zip"
    if zip_path.exists():
        print(f"既に存在: {zip_path}")
        return
    dest.mkdir(parents=True, exist_ok=True)
    kaggle_run(["datasets", "download", DAILY_SLUG.format(date=date), "-p", str(dest)])
    print(f"done: {zip_path}")


# ---------------------------------------------------------------- summarize

_worker_zip: zipfile.ZipFile | None = None
_worker_zip_path: str = ""


def _init_worker(zip_path: str) -> None:
    global _worker_zip, _worker_zip_path
    _worker_zip_path = zip_path
    _worker_zip = zipfile.ZipFile(zip_path)


def _parse_episode(name: str) -> list[dict] | None:
    try:
        with _worker_zip.open(name) as f:
            d = json.load(io.TextIOWrapper(f, encoding="utf-8"))
    except Exception as e:
        print(f"parse error {name}: {e}", file=sys.stderr)
        return None

    steps = d.get("steps") or []
    if not steps:
        return None
    teams = d.get("info", {}).get("TeamNames") or ["?", "?"]
    rewards = d.get("rewards") or [None, None]
    statuses = d.get("statuses") or ["?", "?"]

    decks: list[list[int] | None] = [None, None]
    for st in steps:
        for pi in range(2):
            if decks[pi] is None and isinstance(st[pi].get("action"), list) and len(st[pi]["action"]) == 60:
                decks[pi] = st[pi]["action"]
        if all(decks):
            break

    last = steps[-1]
    turn = (last[0].get("observation") or {}).get("current", {}).get("turn")
    rows = []
    for pi in range(2):
        overage = (last[pi].get("observation") or {}).get("remainingOverageTime")
        rows.append({
            "episode_id": Path(name).stem,
            "player": pi,
            "team": teams[pi] if pi < len(teams) else "?",
            "reward": rewards[pi] if pi < len(rewards) else None,
            "status": statuses[pi] if pi < len(statuses) else "?",
            "deck_ids": " ".join(map(str, decks[pi])) if decks[pi] else "",
            "turns": turn,
            "num_steps": len(steps),
            "overage_left": overage,
        })
    return rows


def cmd_summarize(args: argparse.Namespace) -> None:
    date = args.date
    dest = EPISODES_DIR / date
    zip_path = dest / f"pokemon-tcg-ai-battle-episodes-{date}.zip"
    if not zip_path.exists():
        sys.exit(f"先に download してください: {zip_path}")

    table = _card_table()
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if n.endswith(".json")]
    print(f"{len(names)} episodes を処理中 (workers={args.workers}) ...")

    out_path = dest / "summary.csv"
    fields = ["episode_id", "player", "team", "reward", "status", "archetype",
              "deck_signature", "deck_ids", "turns", "num_steps", "overage_left"]
    n_done = 0
    with Pool(args.workers, initializer=_init_worker, initargs=(str(zip_path),)) as pool, \
            open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for rows in pool.imap_unordered(_parse_episode, names, chunksize=16):
            if not rows:
                continue
            for r in rows:
                ids = [int(x) for x in r["deck_ids"].split()] if r["deck_ids"] else []
                r["archetype"] = classify_deck(ids, table) if ids else "unknown"
                r["deck_signature"] = deck_signature(ids) if ids else ""
                w.writerow(r)
            n_done += 1
            if n_done % 500 == 0:
                print(f"  {n_done}/{len(names)}")
    print(f"done: {out_path}")


# ---------------------------------------------------------------- report

def _load_summary(date: str) -> list[dict]:
    path = EPISODES_DIR / date / "summary.csv"
    if not path.exists():
        sys.exit(f"先に summarize してください: {path}")
    return list(csv.DictReader(open(path)))


def cmd_report(args: argparse.Namespace) -> None:
    date = args.date
    rows = _load_summary(date)
    by_ep: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_ep[r["episode_id"]].append(r)

    manifest_path = EPISODES_DIR / date / "manifest.csv"
    scores = {}
    if not manifest_path.exists():
        zip_path = EPISODES_DIR / date / f"pokemon-tcg-ai-battle-episodes-{date}.zip"
        with zipfile.ZipFile(zip_path) as z, z.open("manifest.csv") as f:
            manifest_path.write_bytes(f.read())
    for m in csv.DictReader(open(manifest_path)):
        scores[m["episode_id"]] = float(m["avg_score"])

    # アーキタイプ別 出現数・勝率
    arch_games: Counter[str] = Counter()
    arch_wins: Counter[str] = Counter()
    matchup: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])  # (a,b): [aの勝ち, 試合数]
    sig_stats: dict[str, list] = defaultdict(lambda: [0, 0, ""])  # sig: [wins, games, archetype]
    max_time_used = 0.0
    errors = Counter()
    for ep, prs in by_ep.items():
        if len(prs) != 2:
            continue
        for r in prs:
            arch_games[r["archetype"]] += 1
            if r["reward"] == "1":
                arch_wins[r["archetype"]] += 1
            if r["deck_signature"]:
                s = sig_stats[r["deck_signature"]]
                s[0] += int(r["reward"] == "1")
                s[1] += 1
                s[2] = r["archetype"]
            if r["overage_left"]:
                max_time_used = max(max_time_used, 600 - float(r["overage_left"]))
            if r["status"] != "DONE":
                errors[r["status"]] += 1
        a, b = prs
        m = matchup[(a["archetype"], b["archetype"])]
        m[0] += int(a["reward"] == "1")
        m[1] += 1
        m2 = matchup[(b["archetype"], a["archetype"])]
        m2[0] += int(b["reward"] == "1")
        m2[1] += 1

    top_arch = [a for a, _ in arch_games.most_common(10)]
    lines = [
        f"# 公式エピソード メタレポート {date}",
        "",
        f"- エピソード数: {len(by_ep)}（daily datasetは20GiB上限で切り詰めの可能性あり、全対戦の完全収録ではない）",
        f"- アーキタイプ分類は tools/get_episodes.py の簡易ヒューリスティック（看板ポケモン推定）。誤分類率未検証",
        f"- 1プレイヤーの最大消費時間: {max_time_used:.1f}秒 / 600秒（remainingOverageTimeより）",
        f"- 異常終了status: {dict(errors) if errors else 'なし'}",
        "",
        "## アーキタイプ分布と勝率",
        "",
        "| アーキタイプ | 出現数 | シェア | 勝率 |",
        "|---|---|---|---|",
    ]
    total = sum(arch_games.values())
    for a in top_arch:
        g = arch_games[a]
        lines.append(f"| {a} | {g} | {g/total:.1%} | {arch_wins[a]/g:.1%} |")

    lines += ["", "## 対面勝率（行 vs 列、上位アーキタイプ、試合数30未満は - ）", "",
              "| vs | " + " | ".join(top_arch[:8]) + " |", "|---|" + "---|" * min(8, len(top_arch))]
    for a in top_arch[:8]:
        cells = []
        for b in top_arch[:8]:
            w, n = matchup[(a, b)]
            cells.append(f"{w/n:.0%} ({n})" if n >= 30 else "-")
        lines.append(f"| {a} | " + " | ".join(cells) + " |")

    lines += ["", "## 勝率上位デッキ（同一60枚、20戦以上）", "",
              "| deck_signature | アーキタイプ | 戦数 | 勝率 |", "|---|---|---|---|"]
    best = sorted((s for s in sig_stats.items() if s[1][1] >= 20),
                  key=lambda kv: kv[1][0] / kv[1][1], reverse=True)[:15]
    for sig, (w, n, arch) in best:
        lines.append(f"| {sig} | {arch} | {n} | {w/n:.1%} |")
    lines += ["", f"（デッキの60枚内訳は data/episodes/{date}/summary.csv の deck_ids を参照）", ""]

    META_DIR.mkdir(parents=True, exist_ok=True)
    out = META_DIR / f"{date}.md"
    out.write_text("\n".join(lines))
    print(f"done: {out}")


# ---------------------------------------------------------------- main

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download", help="日次データセットのzipを取得")
    d.add_argument("--date")
    d.add_argument("--latest", action="store_true")
    d.set_defaults(func=cmd_download)

    s = sub.add_parser("summarize", help="zipからsummary.csvを生成（展開しない）")
    s.add_argument("--date", required=True)
    s.add_argument("-w", "--workers", type=int, default=4)
    s.set_defaults(func=cmd_summarize)

    r = sub.add_parser("report", help="summary.csvからknowledge/episodes/<date>.mdを生成")
    r.add_argument("--date", required=True)
    r.set_defaults(func=cmd_report)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
