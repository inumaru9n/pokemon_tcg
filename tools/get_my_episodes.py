"""自分の提出（submission）の対戦リプレイを取得・分析する。

  uv run tools/get_my_episodes.py <submission_id> [--limit N] [--team haruto]

- kaggle competitions episodes でエピソードID一覧を取得
- kaggle competitions replay で各リプレイJSONをdata/my_episodes/<sid>/に取得（既存はスキップ）
- 各リプレイを解析: 自チームの勝敗、相手デッキのアーキタイプ、自分の消費時間・ターン数
- 対戦相手アーキタイプ別の勝率と、消費時間分布（lethal探索が本番で動いているかの傍証）を報告

日次エピソードデータセット(get_episodes.py)と違い、レートに関係なく自分の全対戦が取れる。
ダウンロードには日次上限があるので --limit で件数を絞る。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from kaggle_util import kaggle_text

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "my_episodes"
sys.path.insert(0, str(ROOT))


def list_episode_ids(sid: str) -> list[str]:
    out = kaggle_text(["competitions", "episodes", str(sid), "--format", "csv"])
    ids = []
    for line in out.splitlines()[1:]:
        if line.strip():
            ids.append(line.split(",")[0])
    return ids


def replay(eid: str, dest: Path) -> Path | None:
    path = dest / f"episode-{eid}-replay.json"
    if path.exists():
        return path
    kaggle_text(["competitions", "replay", str(eid), "-p", str(dest)])
    return path if path.exists() else None


def classify(card_ids: list[int], table) -> str:
    from get_episodes import classify_deck
    return classify_deck(card_ids, table)


def analyze(path: Path, team: str, table) -> dict | None:
    d = json.loads(path.read_text())
    teams = d.get("info", {}).get("TeamNames") or []
    if team not in teams:
        return None
    me = teams.index(team)
    steps = d.get("steps") or []
    rewards = d.get("rewards") or [None, None]
    statuses = d.get("statuses") or ["?", "?"]

    # 各プレイヤーの提出デッキ（最初のlen==60アクション）
    decks = [None, None]
    for st in steps:
        for pi in range(2):
            a = st[pi].get("action")
            if decks[pi] is None and isinstance(a, list) and len(a) == 60:
                decks[pi] = a
        if all(decks):
            break
    opp_deck = decks[1 - me]
    last = steps[-1]
    turn = (last[0].get("observation") or {}).get("current", {}).get("turn")
    my_overage = (last[me].get("observation") or {}).get("remainingOverageTime")

    r = rewards[me]
    result = "win" if r == 1 else "loss" if r == -1 else "draw"
    return {
        "opp_archetype": classify(opp_deck, table) if opp_deck else "unknown",
        "result": result,
        "status": statuses[me] if me < len(statuses) else "?",
        "turns": turn,
        "time_used": (600 - my_overage) if my_overage is not None else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("submission_id")
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--team", default="haruto")
    args = ap.parse_args()

    from get_episodes import _card_table
    table = _card_table()

    dest = OUT / str(args.submission_id)
    dest.mkdir(parents=True, exist_ok=True)
    ids = list_episode_ids(args.submission_id)[: args.limit]
    print(f"{len(ids)} episodes to fetch (submission {args.submission_id})")

    recs = []
    for i, eid in enumerate(ids):
        try:
            p = replay(eid, dest)
            if p:
                rec = analyze(p, args.team, table)
                if rec:
                    recs.append(rec)
        except Exception as e:  # noqa: BLE001 - レート上限等で失敗したらそこまでで分析
            print(f"  stop at {i} ({eid}): {str(e)[:120]}")
            break
    if not recs:
        sys.exit("解析できた対戦なし")

    wins = sum(r["result"] == "win" for r in recs)
    print(f"\n=== {len(recs)}戦 全体勝率 {wins / len(recs):.1%} ===")
    print(f"異常終了: {Counter(r['status'] for r in recs if r['status'] != 'DONE') or 'なし'}")
    times = [r["time_used"] for r in recs if r["time_used"] is not None]
    if times:
        times.sort()
        print(f"消費時間/ゲーム: 中央{times[len(times)//2]:.1f}秒 最大{times[-1]:.1f}秒 (600秒中)")

    by = defaultdict(lambda: [0, 0])
    for r in recs:
        by[r["opp_archetype"]][0] += int(r["result"] == "win")
        by[r["opp_archetype"]][1] += 1
    print("\n=== 対戦相手アーキタイプ別 ===")
    for a, (w, n) in sorted(by.items(), key=lambda kv: -kv[1][1]):
        print(f"  {a}: {w}/{n} ({w/n:.0%})")


if __name__ == "__main__":
    main()
