"""Evaluation CLI: run N games between two agents and report win rates.

Usage:
    uv run arena/run_match.py agents/000_random agents/001_greedy -n 100
    uv run arena/run_match.py A B -n 200 -w 4 --json arena/results/exp001.json

Each agent directory must contain:
    main.py   -- defines agent(obs_dict: dict) -> list[int]
    deck.csv  -- 60 card IDs, one per line

Seats alternate every game (even game index: agent A is player 0).
Invalid selections or agent exceptions count as an immediate loss (recorded
as an error). Games exceeding --step-cap are draws (recorded as step_cap).
"""

import argparse
import importlib.util
import json
import math
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, asdict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class GameRecord:
    game_index: int
    seat0: str  # "A" or "B"
    winner: str  # "A" | "B" | "draw"
    steps: int
    reason: str  # "result" | "agent_error" | "invalid_selection" | "step_cap"
    error_detail: str = ""
    a_move_time: float = 0.0
    b_move_time: float = 0.0
    a_moves: int = 0
    b_moves: int = 0


# Per-process cache (populated lazily inside each worker process).
_worker_state: dict = {}


def load_agent_module(agent_dir: str, alias: str):
    """Load main.py from agent_dir as a uniquely named module."""
    agent_dir = os.path.abspath(agent_dir)
    main_path = os.path.join(agent_dir, "main.py")
    if not os.path.exists(main_path):
        raise FileNotFoundError(f"main.py not found in {agent_dir}")
    for p in (REPO_ROOT, agent_dir):
        if p not in sys.path:
            sys.path.insert(0, p)
    spec = importlib.util.spec_from_file_location(f"agent_{alias}", main_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    if not hasattr(mod, "agent"):
        raise AttributeError(f"{main_path} does not define agent()")
    return mod.agent


def read_deck(agent_dir: str) -> list[int]:
    path = os.path.join(agent_dir, "deck.csv")
    with open(path) as f:
        ids = [int(line) for line in f.read().split()]
    if len(ids) != 60:
        raise ValueError(f"{path}: expected 60 cards, got {len(ids)}")
    return ids


def _ensure_worker(a_dir: str, b_dir: str) -> None:
    if _worker_state.get("loaded"):
        return
    _worker_state["agent_a"] = load_agent_module(a_dir, "A")
    _worker_state["agent_b"] = load_agent_module(b_dir, "B")
    _worker_state["deck_a"] = read_deck(a_dir)
    _worker_state["deck_b"] = read_deck(b_dir)
    _worker_state["loaded"] = True


def _validate_selection(sel, select) -> str:
    """Return "" if valid, else an error description."""
    if not isinstance(sel, list) or not all(isinstance(i, int) for i in sel):
        return f"selection is not list[int]: {sel!r}"
    n = len(select["option"])
    if not (select["minCount"] <= len(sel) <= select["maxCount"]):
        return (f"selection count {len(sel)} outside "
                f"[{select['minCount']}, {select['maxCount']}]")
    if len(set(sel)) != len(sel):
        return f"duplicate elements: {sel}"
    if any(i < 0 or i >= n for i in sel):
        return f"index out of range (n={n}): {sel}"
    return ""


def derive_engine_seed(engine_seed_base: int, game_index: int) -> int:
    """Deterministic per-game engine seed for CRN pairing.

    Two runs with the same (engine_seed_base, game_index, decks) get identical
    engine RNG streams (deck order, coin flips), so candidate-vs-candidate
    comparisons share初期条件 and the opening-luck variance cancels.
    """
    return ((engine_seed_base * 1000003 + game_index * 7919 + 1) & 0xFFFFFFFF) or 1


def play_game(args: tuple) -> GameRecord:
    game_index, a_dir, b_dir, step_cap, base_seed, engine_seed_base = args
    _ensure_worker(a_dir, b_dir)
    from cg.game import battle_start, battle_start_seeded, battle_select, battle_finish

    random.seed(base_seed + game_index)
    a_is_seat0 = game_index % 2 == 0
    seat = {0: "A" if a_is_seat0 else "B", 1: "B" if a_is_seat0 else "A"}
    agents = {"A": _worker_state["agent_a"], "B": _worker_state["agent_b"]}
    deck0 = _worker_state["deck_a"] if a_is_seat0 else _worker_state["deck_b"]
    deck1 = _worker_state["deck_b"] if a_is_seat0 else _worker_state["deck_a"]

    rec = GameRecord(game_index=game_index, seat0=seat[0], winner="draw",
                     steps=0, reason="result")

    if engine_seed_base is not None:
        obs_dict, start = battle_start_seeded(
            deck0, deck1, derive_engine_seed(engine_seed_base, game_index))
    else:
        obs_dict, start = battle_start(deck0, deck1)
    if obs_dict is None:
        rec.winner = "draw"
        rec.reason = "agent_error"
        rec.error_detail = (f"battle_start failed: errorPlayer="
                            f"{start.errorPlayer} errorType={start.errorType}")
        return rec

    try:
        while True:
            result = obs_dict["current"]["result"]
            if result != -1:
                rec.winner = "draw" if result == 2 else seat[result]
                return rec
            if rec.steps >= step_cap:
                rec.reason = "step_cap"
                rec.winner = "draw"
                return rec

            player = obs_dict["current"]["yourIndex"]
            tag = seat[player]
            t0 = time.perf_counter()
            try:
                sel = agents[tag](obs_dict)
            except Exception as e:  # noqa: BLE001 - agent error means loss
                rec.winner = "B" if tag == "A" else "A"
                rec.reason = "agent_error"
                rec.error_detail = f"{tag}: {type(e).__name__}: {e}"
                return rec
            dt = time.perf_counter() - t0
            if tag == "A":
                rec.a_move_time += dt
                rec.a_moves += 1
            else:
                rec.b_move_time += dt
                rec.b_moves += 1

            err = _validate_selection(sel, obs_dict["select"])
            if err:
                rec.winner = "B" if tag == "A" else "A"
                rec.reason = "invalid_selection"
                rec.error_detail = f"{tag}: {err}"
                return rec

            obs_dict = battle_select(sel)
            rec.steps += 1
    finally:
        battle_finish()


def wilson_ci(score: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval for a proportion (draws counted as 0.5 wins)."""
    if n == 0:
        return (0.0, 1.0)
    p = score / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def run_series(a_dir: str, b_dir: str, n_games: int, workers: int,
               step_cap: int, seed: int, engine_seed_base: int = None) -> dict:
    t0 = time.time()
    args = [(i, a_dir, b_dir, step_cap, seed, engine_seed_base)
            for i in range(n_games)]
    if workers <= 1:
        records = [play_game(a) for a in args]
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            records = list(ex.map(play_game, args, chunksize=1))

    a_wins = sum(r.winner == "A" for r in records)
    b_wins = sum(r.winner == "B" for r in records)
    draws = sum(r.winner == "draw" for r in records)
    errors = [r for r in records if r.reason in ("agent_error", "invalid_selection")]
    score = a_wins + 0.5 * draws
    lo, hi = wilson_ci(score, n_games)
    a_moves = sum(r.a_moves for r in records)
    b_moves = sum(r.b_moves for r in records)

    return {
        "agent_a": os.path.abspath(a_dir),
        "agent_b": os.path.abspath(b_dir),
        "games": n_games,
        "a_wins": a_wins,
        "b_wins": b_wins,
        "draws": draws,
        "a_score_rate": round(score / n_games, 4),
        "a_score_rate_ci95": [round(lo, 4), round(hi, 4)],
        "errors": len(errors),
        "error_details": [r.error_detail for r in errors][:10],
        "avg_steps": round(sum(r.steps for r in records) / n_games, 1),
        "step_cap_draws": sum(r.reason == "step_cap" for r in records),
        "a_avg_move_ms": round(1000 * sum(r.a_move_time for r in records) / max(1, a_moves), 3),
        "b_avg_move_ms": round(1000 * sum(r.b_move_time for r in records) / max(1, b_moves), 3),
        "duration_sec": round(time.time() - t0, 1),
        "seed": seed,
        "engine_seed_base": engine_seed_base,
        "records": [asdict(r) for r in records],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("agent_a", help="directory of agent A")
    ap.add_argument("agent_b", help="directory of agent B")
    ap.add_argument("-n", "--games", type=int, default=100)
    ap.add_argument("-w", "--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--step-cap", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--engine-seed", type=int, default=None,
                    help="CRN mode: fix engine RNG per game from this base seed "
                         "(requires local libcg with BattleStartSeeded)")
    ap.add_argument("--json", dest="json_path", default=None,
                    help="write full results (incl. per-game records) to this file")
    args = ap.parse_args()

    summary = run_series(args.agent_a, args.agent_b, args.games, args.workers,
                         args.step_cap, args.seed, args.engine_seed)

    if args.json_path:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_path)), exist_ok=True)
        with open(args.json_path, "w") as f:
            json.dump(summary, f, indent=1)

    brief = {k: v for k, v in summary.items() if k != "records"}
    print(json.dumps(brief, indent=1))
    a = os.path.basename(os.path.normpath(args.agent_a))
    b = os.path.basename(os.path.normpath(args.agent_b))
    lo, hi = summary["a_score_rate_ci95"]
    print(f"\n{a} vs {b}: {summary['a_wins']}W-{summary['b_wins']}L-{summary['draws']}D "
          f"| score {summary['a_score_rate']:.1%} (95% CI {lo:.1%}-{hi:.1%}) "
          f"| {summary['duration_sec']}s", file=sys.stderr)


if __name__ == "__main__":
    main()
