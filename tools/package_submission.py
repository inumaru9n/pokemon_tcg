"""Package an agent directory into a Kaggle submission tar.gz.

Usage:
    uv run tools/package_submission.py agents/001_greedy

Steps:
 1. Validate deck.csv (60 cards, <=4 copies per card name except basic
    energy, <=1 ACE SPEC, card IDs exist).
 2. Self-play validation games (like Kaggle's validation episode).
 3. Bundle main.py + deck.csv + any extra agent files + the ORIGINAL cg/
    from the competition data (not the locally built dylib) into
    submissions/<agent>_<timestamp>.tar.gz with main.py at top level.
"""

import argparse
import collections
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORIGINAL_CG = os.path.join(
    REPO_ROOT, "pokemon-tcg-ai-battle", "data", "sample_submission",
    "sample_submission", "cg")
SUBMISSIONS_DIR = os.path.join(REPO_ROOT, "submissions")

sys.path.insert(0, REPO_ROOT)


def validate_deck(deck_path: str) -> list[str]:
    from cg.api import all_card_data, CardType

    problems = []
    with open(deck_path) as f:
        ids = [int(x) for x in f.read().split()]
    if len(ids) != 60:
        problems.append(f"deck has {len(ids)} cards, expected 60")

    cards = {c.cardId: c for c in all_card_data()}
    unknown = [i for i in ids if i not in cards]
    if unknown:
        problems.append(f"unknown card IDs: {sorted(set(unknown))}")
        return problems

    by_name = collections.Counter()
    ace_spec = 0
    basics = 0
    for i in ids:
        c = cards[i]
        if c.cardType != CardType.BASIC_ENERGY:
            by_name[c.name] += 1
        if c.aceSpec:
            ace_spec += 1
        if c.cardType == CardType.POKEMON and c.basic:
            basics += 1
    for name, cnt in by_name.items():
        if cnt > 4:
            problems.append(f"more than 4 copies of '{name}' ({cnt})")
    if ace_spec > 1:
        problems.append(f"{ace_spec} ACE SPEC cards (max 1)")
    if basics == 0:
        problems.append("deck contains no Basic Pokemon")
    return problems


def self_play(agent_dir: str, games: int) -> None:
    cmd = ["uv", "run", os.path.join(REPO_ROOT, "arena", "run_match.py"),
           agent_dir, agent_dir, "-n", str(games), "-w", "1"]
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    if out.returncode != 0:
        sys.exit(f"self-play run failed:\n{out.stderr}")
    import json
    summary = json.loads(out.stdout)
    if summary["errors"] > 0:
        sys.exit(f"self-play produced errors:\n{summary['error_details']}")
    print(f"self-play OK: {games} games, 0 errors, "
          f"avg {summary['avg_steps']} steps")


def package(agent_dir: str) -> str:
    agent_dir = os.path.abspath(agent_dir)
    name = os.path.basename(os.path.normpath(agent_dir))
    ts = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs(SUBMISSIONS_DIR, exist_ok=True)
    out_path = os.path.join(SUBMISSIONS_DIR, f"{name}_{ts}.tar.gz")

    with tempfile.TemporaryDirectory() as tmp:
        for entry in os.listdir(agent_dir):
            if entry in ("cg", "__pycache__") or entry.endswith(".pyc"):
                continue
            src = os.path.join(agent_dir, entry)
            dst = os.path.join(tmp, entry)
            (shutil.copytree if os.path.isdir(src) else shutil.copy2)(src, dst)
        shutil.copytree(ORIGINAL_CG, os.path.join(tmp, "cg"),
                        ignore=shutil.ignore_patterns("__pycache__"))
        with tarfile.open(out_path, "w:gz") as tar:
            for entry in sorted(os.listdir(tmp)):
                tar.add(os.path.join(tmp, entry), arcname=entry)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("agent_dir")
    ap.add_argument("--self-play-games", type=int, default=4)
    ap.add_argument("--skip-self-play", action="store_true")
    args = ap.parse_args()

    main_py = os.path.join(args.agent_dir, "main.py")
    deck_csv = os.path.join(args.agent_dir, "deck.csv")
    for p in (main_py, deck_csv):
        if not os.path.exists(p):
            sys.exit(f"missing: {p}")

    problems = validate_deck(deck_csv)
    if problems:
        sys.exit("deck validation failed:\n- " + "\n- ".join(problems))
    print("deck OK")

    if not args.skip_self_play:
        self_play(args.agent_dir, args.self_play_games)

    out = package(args.agent_dir)
    with tarfile.open(out) as tar:
        top = sorted({m.name.split("/")[0] for m in tar.getmembers()})
    print(f"created: {out}")
    print(f"top-level entries: {top}")
    if "main.py" not in top or "deck.csv" not in top:
        sys.exit("ERROR: main.py/deck.csv not at top level")


if __name__ == "__main__":
    main()
