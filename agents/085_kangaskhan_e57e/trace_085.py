"""EXP-085 デバッグ用: 085 vs 相手エージェントの1ゲームを観戦ログ付きで実行。

usage: uv run python agents/085_kangaskhan_e57e/trace_085.py <opp_dir> <win|loss> [seed_start]
  win  = 085が勝ったゲームを最初に見つけて表示
  loss = 085が負けたゲームを表示
"""
import random
import sys

sys.path.insert(0, "/Users/akira/kaggle/pokemon_tcg")
from arena.run_match import load_agent_module, read_deck
from cg.game import battle_start, battle_select, battle_finish
from tools import render_replay as rr

A_DIR = "/Users/akira/kaggle/pokemon_tcg/agents/085_kangaskhan_e57e"


def play(agent_a, agent_b, deck_a, deck_b, a_is_seat0, record):
    """record=list to append (obs, player, sel). returns winner tag 'A'/'B'/'draw'."""
    deck0 = deck_a if a_is_seat0 else deck_b
    deck1 = deck_b if a_is_seat0 else deck_a
    seat = {0: "A" if a_is_seat0 else "B", 1: "B" if a_is_seat0 else "A"}
    obs, start = battle_start(deck0, deck1)
    assert obs is not None
    try:
        steps = 0
        while True:
            result = obs["current"]["result"]
            if result != -1:
                return "draw" if result == 2 else seat[result]
            if steps >= 30000:
                return "draw"
            player = obs["current"]["yourIndex"]
            sel = (agent_a if seat[player] == "A" else agent_b)(obs)
            record.append((obs, player, list(sel)))
            obs = battle_select(sel)
            steps += 1
    finally:
        battle_finish()


def render(record, me_seat_player, seat):
    last_turn = -1
    prev_logs = None
    for obs, player, sel in record:
        cur = obs["current"]
        logs = obs.get("logs") or []
        if logs != prev_logs:
            new = logs
            if prev_logs and len(logs) > len(prev_logs) and logs[: len(prev_logs)] == prev_logs:
                new = logs[len(prev_logs):]
            for lg in new:
                s = rr.render_log(lg, me_seat_player)
                if s:
                    print(s)
            prev_logs = logs
        if cur["turn"] != last_turn:
            print(f"\n### turn {cur['turn']} (first={cur['firstPlayer']}) board:")
            print(rr.board_str(cur, me_seat_player))
            last_turn = cur["turn"]
        sel_obj = obs.get("select") or {}
        opts = sel_obj.get("option") or []
        ctx = rr.CTX.get(sel_obj.get("context"), sel_obj.get("context"))
        chosen = [rr.opt_str(opts[j], cur, player) for j in sel if j < len(opts)]
        tag = seat[player]
        if not chosen and sel_obj.get("minCount", 0) == 0:
            chosen = ["(pass/none)"]
        print(f"  >> [{tag} t{cur['turn']} {ctx} {len(sel)}/{len(opts)}] " + " | ".join(chosen))


def main():
    opp_dir = sys.argv[1]
    want = sys.argv[2]  # win / loss
    seed0 = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    agent_a = load_agent_module(A_DIR, "T085")
    agent_b = load_agent_module(opp_dir, "TOPP")
    deck_a = read_deck(A_DIR)
    deck_b = read_deck(opp_dir)
    for g in range(200):
        random.seed(seed0 + g)
        a_is_seat0 = g % 2 == 0
        record = []
        winner = play(agent_a, agent_b, deck_a, deck_b, a_is_seat0, record)
        if (want == "win" and winner == "A") or (want == "loss" and winner == "B"):
            seat = {0: "A" if a_is_seat0 else "B", 1: "B" if a_is_seat0 else "A"}
            me = 0 if a_is_seat0 else 1  # 085's player index
            print(f"# game {g}: winner={winner}, 085 is player {me} (A={A_DIR.split('/')[-1]}, B={opp_dir.split('/')[-1]})")
            render(record, me, seat)
            return
    print("not found in 200 games")


if __name__ == "__main__":
    main()
