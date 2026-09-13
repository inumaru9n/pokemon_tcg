"""自分の提出リプレイを走査し、**勝った試合だけ**を抽出タスクの形で出す（EXP-114）。

継続学習の教師データにするので、負け試合は使わない（自分の悪手を学習してしまう）。
出力は extract_yushin.py の find_episodes と同じ (date, ep_id, me, meta) タプル。
date は "mine:<submission_id>" という擬似日付にする。

usage:
  uv run python tools/list_my_games.py 55235376 --team haruto --only-win
"""
from __future__ import annotations
import argparse, collections, json, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def list_games(sid: str, team: str = "haruto", only_win: bool = True):
    d = ROOT / "data" / "my_episodes" / str(sid)
    out, stat = [], collections.Counter()
    for f in sorted(d.glob("*.json")):
        try:
            j = json.load(open(f))
        except Exception:
            stat["読めない"] += 1
            continue
        names = (j.get("info") or {}).get("TeamNames") or []
        if team not in names:
            stat["自分がいない"] += 1
            continue
        me = names.index(team)
        rw = j.get("rewards") or [None, None]
        res = ("win" if (rw[me] or 0) > (rw[1 - me] or 0)
               else "loss" if (rw[me] or 0) < (rw[1 - me] or 0) else "draw")
        stat[res] += 1
        if only_win and res != "win":
            continue
        opp = j["steps"][0][1 - me].get("observation") or {}
        out.append((f"mine:{sid}", f.stem, me,
                    {"me": me, "result": res, "opp_arch": "?", "opp_sig": "?"}))
    return out, stat


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sids", nargs="+")
    ap.add_argument("--team", default="haruto")
    ap.add_argument("--all", action="store_true", help="負け試合も含める")
    a = ap.parse_args()
    tot = []
    for sid in a.sids:
        g, st = list_games(sid, a.team, not a.all)
        print(f"submission {sid}: {dict(st)} → 採用 {len(g)}件")
        tot += g
    print(f"合計 {len(tot)}件")


if __name__ == "__main__":
    main()
