"""EXP-090 instrumentation: in-process CRN games vs 089 reading F090_STATS.

play_game() is deterministic per game_index (python RNG reseeded per game +
seeded engine RNG), so a workers=1 in-process run reproduces the same games
as the official -w 2 gate run and lets us read agent_A.F090_STATS deltas
between games.

Usage:
    uv run agents/090_marnie_ml_lethal/instrument_090.py [START] [END] [ENGINE_SEED] [OUT_JSON]
"""

import json
import os
import sys
from dataclasses import asdict

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "arena"))

import run_match as rm  # noqa: E402

A_DIR = os.path.join(REPO, "agents/090_marnie_ml_lethal")
B_DIR = os.path.join(REPO, "agents/089_marnie_ml")


def main() -> None:
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    end = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    engine_seed = int(sys.argv[3]) if len(sys.argv) > 3 else 1350001
    out = sys.argv[4] if len(sys.argv) > 4 else os.path.join(
        REPO, "arena/results/EXP-090_instrument.json")

    records = []
    prev = {"search_calls": 0, "fires": 0, "overrides": 0}
    for i in range(start, end):
        rec = rm.play_game((i, A_DIR, B_DIR, 30000, 42, engine_seed))
        stats = dict(sys.modules["agent_A"].F090_STATS)
        delta = {k: stats[k] - prev[k] for k in stats}
        prev = stats
        records.append({**asdict(rec), "stats": delta})
        if (i + 1) % 50 == 0:
            print(f"[{i + 1}/{end}]", flush=True)

    fired = [r for r in records if r["stats"]["fires"] > 0]
    fired_wins = [r for r in fired if r["winner"] == "A"]
    a_wins = sum(r["winner"] == "A" for r in records)
    summary = {
        "games": end - start,
        "game_range": [start, end],
        "engine_seed": engine_seed,
        "a_wins": a_wins,
        "b_wins": sum(r["winner"] == "B" for r in records),
        "draws": sum(r["winner"] == "draw" for r in records),
        "errors": sum(r["reason"] in ("agent_error", "invalid_selection")
                      for r in records),
        "search_call_total": prev["search_calls"],
        "fire_total": prev["fires"],
        "override_total": prev["overrides"],
        "fired_games": len(fired),
        "fired_game_wins": len(fired_wins),
        "fired_game_losses": [r["game_index"] for r in fired
                              if r["winner"] != "A"],
        "override_games": sum(r["stats"]["overrides"] > 0 for r in records),
        "unfired_games": (end - start) - len(fired),
        "unfired_game_wins": a_wins - len(fired_wins),
    }
    with open(out, "w") as f:
        json.dump({"summary": summary, "records": records}, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
