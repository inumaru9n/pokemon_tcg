"""Round-robin league over agent directories, with a local Elo rating.

Usage:
    uv run arena/league.py                      # all dirs under agents/
    uv run arena/league.py agents/001_greedy agents/002_x -n 100

Writes arena/results/league.json and league.md (pairwise table + ranking).
Match processes are spawned per pair, so each pair runs with clean state.
"""

import argparse
import itertools
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_match import run_series  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO_ROOT, "arena", "results")


def discover_agents() -> list[str]:
    base = os.path.join(REPO_ROOT, "agents")
    dirs = []
    for name in sorted(os.listdir(base)):
        d = os.path.join(base, name)
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "main.py")):
            dirs.append(d)
    return dirs


def elo_from_scores(names: list[str], score: dict, games: dict,
                    iters: int = 500, k: float = 400.0) -> dict:
    """Fit Elo-like ratings to pairwise score rates by gradient steps."""
    rating = {n: 1000.0 for n in names}
    for _ in range(iters):
        for a, b in itertools.combinations(names, 2):
            n = games.get((a, b), 0)
            if n == 0:
                continue
            actual = score[(a, b)] / n
            expected = 1.0 / (1.0 + 10 ** ((rating[b] - rating[a]) / 400.0))
            delta = 0.1 * k * (actual - expected)
            rating[a] += delta
            rating[b] -= delta
    return rating


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("agents", nargs="*", help="agent dirs (default: all under agents/)")
    ap.add_argument("-n", "--games", type=int, default=100, help="games per pair")
    ap.add_argument("-w", "--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--step-cap", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    agent_dirs = [os.path.abspath(a) for a in args.agents] or discover_agents()
    if len(agent_dirs) < 2:
        sys.exit("Need at least 2 agents.")
    names = [os.path.basename(os.path.normpath(d)) for d in agent_dirs]
    by_name = dict(zip(names, agent_dirs))

    t0 = time.time()
    score: dict = {}
    games: dict = {}
    pair_summary = {}
    for a, b in itertools.combinations(names, 2):
        print(f"[league] {a} vs {b} ({args.games} games)...", file=sys.stderr)
        s = run_series(by_name[a], by_name[b], args.games, args.workers,
                       args.step_cap, args.seed)
        score[(a, b)] = s["a_wins"] + 0.5 * s["draws"]
        games[(a, b)] = s["games"]
        pair_summary[f"{a} vs {b}"] = {k: v for k, v in s.items() if k != "records"}

    rating = elo_from_scores(names, score, games)
    ranking = sorted(names, key=lambda n: -rating[n])

    # Markdown report
    lines = ["# League Results", "",
             f"- date: {time.strftime('%Y-%m-%d %H:%M')}",
             f"- games per pair: {args.games}, seed: {args.seed}",
             f"- duration: {time.time() - t0:.0f}s", "",
             "## Ranking", "", "| # | agent | Elo |", "|---|---|---|"]
    for i, n in enumerate(ranking, 1):
        lines.append(f"| {i} | {n} | {rating[n]:.0f} |")
    lines += ["", "## Pairwise score rate (row vs column)", ""]
    header = "| | " + " | ".join(ranking) + " |"
    lines += [header, "|---" * (len(ranking) + 1) + "|"]
    for a in ranking:
        row = [a]
        for b in ranking:
            if a == b:
                row.append("-")
            elif (a, b) in score:
                row.append(f"{score[(a, b)] / games[(a, b)]:.0%}")
            elif (b, a) in score:
                row.append(f"{1 - score[(b, a)] / games[(b, a)]:.0%}")
            else:
                row.append("?")
        lines.append("| " + " | ".join(row) + " |")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    md_path = os.path.join(RESULTS_DIR, "league.md")
    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(RESULTS_DIR, "league.json"), "w") as f:
        json.dump({"rating": rating, "pairs": pair_summary}, f, indent=1)

    print("\n".join(lines))
    print(f"\nwritten: {md_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
