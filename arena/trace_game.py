"""CRNシードで特定ゲームを決定的に再現し、構造化トレースを出力する（I-126/I-133）。

run_matchと同一のシード・席順ロジックで game_index のゲームを再生し、
ターンごとの盤面サマリと全選択を記録する。負けトレース精読・分岐点特定用。

usage:
  uv run arena/trace_game.py <agentA_dir> <agentB_dir> --engine-seed BASE --game IDX \
      [--json out.jsonl] [--quiet]
"""

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_match import (derive_engine_seed, load_agent_module, read_deck)  # noqa: E402


def snap(cur: dict) -> dict:
    """盤面の要約（両プレイヤー）。"""
    out = {"turn": cur["turn"]}
    for p in (0, 1):
        ps = cur["players"][p]
        act = ps.get("active") or []
        a = act[0] if act and act[0] else None
        out[f"p{p}"] = {
            "prize": len(ps.get("prize") or []),
            "deck": ps.get("deckCount"),
            "hand": ps.get("handCount"),
            "active": (a or {}).get("id"),
            "active_dmg": (a or {}).get("damage"),
            "bench": [(b or {}).get("id") for b in (ps.get("bench") or []) if b],
        }
    return out


def opt_desc(obs: dict, sel_idx: list[int]) -> str:
    sel = obs.get("select") or {}
    opts = sel.get("option") or []
    parts = []
    for i in sel_idx:
        if i < len(opts):
            o = opts[i]
            parts.append(f"t{o.get('type')}a{o.get('area')}i{o.get('index')}")
    return f"ctx{sel.get('context')}:[" + ",".join(parts) + "]"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("agent_a")
    ap.add_argument("agent_b")
    ap.add_argument("--engine-seed", type=int, required=True)
    ap.add_argument("--game", type=int, required=True)
    ap.add_argument("--seed", type=int, default=42, help="run_matchのbase_seed")
    ap.add_argument("--json", dest="json_path", default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    from cg.game import battle_start_seeded, battle_select, battle_finish

    agent_a = load_agent_module(args.agent_a, "A")
    agent_b = load_agent_module(args.agent_b, "B")
    deck_a, deck_b = read_deck(args.agent_a), read_deck(args.agent_b)

    gi = args.game
    random.seed(args.seed + gi)
    a_is_seat0 = gi % 2 == 0
    seat = {0: "A" if a_is_seat0 else "B", 1: "B" if a_is_seat0 else "A"}
    agents = {"A": agent_a, "B": agent_b}
    deck0 = deck_a if a_is_seat0 else deck_b
    deck1 = deck_b if a_is_seat0 else deck_a

    es = derive_engine_seed(args.engine_seed, gi)
    obs, start = battle_start_seeded(deck0, deck1, es)
    assert obs is not None, f"battle_start failed: {start.errorType}"

    events = [{"ev": "meta", "game": gi, "engine_seed": es,
               "seat0": seat[0], "a_dir": args.agent_a, "b_dir": args.agent_b}]
    last_turn = -1
    last_prize = {0: 0, 1: 0}  # セットアップで0→6に充填され、以後は自分がサイドを取ると減る
    steps = 0
    try:
        while True:
            cur = obs["current"]
            if cur["result"] != -1:
                events.append({"ev": "result", "winner_seat": cur["result"],
                               "winner": ("draw" if cur["result"] == 2 else seat[cur["result"]]),
                               "steps": steps, "final": snap(cur)})
                break
            if cur["turn"] != last_turn:
                last_turn = cur["turn"]
                events.append({"ev": "turn", **snap(cur)})
            for p in (0, 1):
                pr = len((cur["players"][p].get("prize") or []))
                if pr > last_prize[p]:      # セットアップ充填
                    last_prize[p] = pr
                elif pr < last_prize[p]:    # p が自分のサイドを取った（=相手をKO）
                    events.append({"ev": "prize_taken", "by_seat": p, "by": seat[p],
                                   "from": last_prize[p], "to": pr, "turn": cur["turn"]})
                    last_prize[p] = pr
            player = cur["yourIndex"]
            tag = seat[player]
            sel_idx = agents[tag](obs)
            events.append({"ev": "sel", "turn": cur["turn"], "who": tag,
                           "d": opt_desc(obs, sel_idx)})
            obs = battle_select(sel_idx)
            steps += 1
            if steps > 30000:
                events.append({"ev": "step_cap"})
                break
    finally:
        battle_finish()

    if args.json_path:
        with open(args.json_path, "w") as f:
            for e in events:
                f.write(json.dumps(e) + "\n")
    if not args.quiet:
        for e in events:
            if e["ev"] in ("meta", "turn", "prize_taken", "result"):
                print(json.dumps(e, ensure_ascii=False))


if __name__ == "__main__":
    main()
