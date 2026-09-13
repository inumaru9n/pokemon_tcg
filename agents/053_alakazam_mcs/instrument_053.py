"""In-process instrumentation for 053 MCS layer.

Runs N games of 053 vs an opponent agent inside one process (no arena),
collecting per-decision MCS stats: calls, fired, time distribution.
Usage:
  uv run python instrument_053.py --opp agents/052_alakazam_mimic -n 20 --seed 42
"""
import argparse
import importlib.util
import os
import random
import statistics
import sys
import time

REPO_ROOT = "/Users/akira/kaggle/pokemon_tcg"
sys.path.insert(0, REPO_ROOT)


def load_agent(agent_dir, alias):
    agent_dir = os.path.abspath(os.path.join(REPO_ROOT, agent_dir))
    main_path = os.path.join(agent_dir, "main.py")
    for p in (REPO_ROOT, agent_dir):
        if p not in sys.path:
            sys.path.insert(0, p)
    spec = importlib.util.spec_from_file_location(f"agent_{alias}", main_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def read_deck(agent_dir):
    with open(os.path.join(REPO_ROOT, agent_dir, "deck.csv")) as f:
        return [int(x) for x in f.read().split("\n")[:60]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opp", default="agents/052_alakazam_mimic")
    ap.add_argument("-n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--step-cap", type=int, default=3000)
    args = ap.parse_args()

    me = load_agent("agents/053_alakazam_mcs", "me")
    opp = load_agent(args.opp, "opp")
    me._F053_LOG_CAP = 100000
    my_deck = read_deck("agents/053_alakazam_mcs")
    opp_deck = read_deck(args.opp)

    from cg.game import battle_start, battle_select, battle_finish

    wins = losses = draws = 0
    game_times = []
    move_times_all = []
    for gi in range(args.n):
        random.seed(args.seed + gi)
        my_pi = gi % 2
        agents = {my_pi: me.agent, 1 - my_pi: opp.agent}
        deck0 = my_deck if my_pi == 0 else opp_deck
        deck1 = opp_deck if my_pi == 0 else my_deck
        obs_dict, start = battle_start(deck0, deck1)
        if obs_dict is None:
            print(f"game {gi}: battle_start failed")
            continue
        me_time, me_moves, steps, result = 0.0, 0, 0, None
        try:
            while steps < args.step_cap:
                result = obs_dict["current"]["result"]
                if result != -1:
                    break
                result = None
                pi = obs_dict["current"]["yourIndex"]
                t0 = time.perf_counter()
                sel = agents[pi](obs_dict)
                dt = time.perf_counter() - t0
                if pi == my_pi:
                    me_time += dt
                    me_moves += 1
                    move_times_all.append(dt)
                obs_dict = battle_select(sel)
                steps += 1
        finally:
            battle_finish()
        game_times.append(me_time)
        if result == my_pi:
            wins += 1
            tag = "W"
        elif result is None or result == 2:
            draws += 1
            tag = "D"
        else:
            losses += 1
            tag = "L"
        print(f"game {gi}: {tag} steps={steps} 053_time={me_time:.1f}s moves={me_moves}")

    st = me.F053_STATS
    print("\n=== summary ===")
    print(f"record: {wins}-{losses}-{draws}")
    print(f"STATS: {dict(st)}")
    if st["mcs_calls"]:
        print(f"avg time/call: {st['mcs_time_ms']/st['mcs_calls']:.0f} ms, "
              f"fired rate: {st['mcs_fired']/st['mcs_calls']:.2%}")
    if game_times:
        print(f"game search time: med={statistics.median(game_times):.1f}s "
              f"max={max(game_times):.1f}s")
    if move_times_all:
        s = sorted(move_times_all)
        print(f"move time: med={s[len(s)//2]*1000:.0f}ms p90={s[int(len(s)*0.9)]*1000:.0f}ms "
              f"max={s[-1]*1000:.0f}ms")
    fired = [r for r in me.F053_LOG if r[6] is not None]
    evaluated = len(me.F053_LOG)
    print(f"\nMCS decisions logged: {evaluated}, fired: {len(fired)}")
    for r in fired[:12]:
        turn, n, cls, pts, done, g_idx, f_idx = r
        print(f"  turn{turn} n={n} cands={cls} pts={pts} greedy={g_idx} fired={f_idx}")


if __name__ == "__main__":
    main()
