"""EXP-085 デバッグ用: 085 vs 相手の複数ゲームを回し、勝敗理由と最終盤面を集計。

usage: uv run python agents/085_kangaskhan_e57e/stats_085.py <opp_dir> <n_games>
"""
import random
import sys
from collections import Counter

sys.path.insert(0, "/Users/akira/kaggle/pokemon_tcg")
from arena.run_match import load_agent_module, read_deck
from cg.game import battle_start, battle_select, battle_finish
from cg import api

A_DIR = "/Users/akira/kaggle/pokemon_tcg/agents/085_kangaskhan_e57e"
REASON = {1: "prizes", 2: "deck_out", 3: "no_active", 4: "card_effect"}
CARDS = {c.cardId: c for c in api.all_card_data()}
ATKS = {a.attackId: a for a in api.all_attack()}


def main():
    opp_dir = sys.argv[1]
    n = int(sys.argv[2])
    agent_a = load_agent_module(A_DIR, "S085")
    agent_b = load_agent_module(opp_dir, "SOPP")
    deck_a = read_deck(A_DIR)
    deck_b = read_deck(opp_dir)

    tally = Counter()
    turns_by = Counter()
    turns_n = Counter()
    a_plays = Counter()
    b_plays = Counter()
    prz_left = Counter()  # (winner, loser_prizes_remaining)
    for g in range(n):
        random.seed(42 + g)
        a_is_seat0 = g % 2 == 0
        seat = {0: "A" if a_is_seat0 else "B", 1: "B" if a_is_seat0 else "A"}
        deck0 = deck_a if a_is_seat0 else deck_b
        deck1 = deck_b if a_is_seat0 else deck_a
        obs, start = battle_start(deck0, deck1)
        try:
            steps = 0
            while True:
                result = obs["current"]["result"]
                if result != -1:
                    reason = None
                    for lg in reversed(obs.get("logs") or []):
                        if lg.get("type") == 26 or "result" in lg:
                            reason = lg.get("reason")
                            break
                    win = "draw" if result == 2 else seat[result]
                    tally[(win, REASON.get(reason, reason))] += 1
                    turns_by[win] += obs["current"]["turn"]
                    turns_n[win] += 1
                    if win != "draw":
                        # A(085)側の残プライズ
                        a_pi = 0 if a_is_seat0 else 1
                        pa = len(obs["current"]["players"][a_pi]["prize"])
                        pb = len(obs["current"]["players"][1 - a_pi]["prize"])
                        prz_left[(win, f"A_prz={pa},B_prz={pb}")] += 1
                    break
                if steps >= 30000:
                    tally[("draw", "step_cap")] += 1
                    break
                player = obs["current"]["yourIndex"]
                sel = (agent_a if seat[player] == "A" else agent_b)(obs)
                sel_obj = obs.get("select") or {}
                opts = sel_obj.get("option") or []
                mypl = obs["current"]["players"][player]
                bucket = a_plays if seat[player] == "A" else b_plays
                for j in sel:
                    if not (isinstance(j, int) and j < len(opts)):
                        continue
                    o = opts[j]
                    if o["type"] == api.OptionType.PLAY:
                        h = mypl.get("hand")
                        if h and o["index"] < len(h):
                            c = CARDS.get(h[o["index"]]["id"])
                            bucket[c.name if c else "?"] += 1
                    elif o["type"] == api.OptionType.ATTACK:
                        act_p = mypl["active"][0] if mypl.get("active") else None
                        c = CARDS.get(act_p["id"]) if act_p else None
                        a = ATKS.get(o.get("attackId"))
                        bucket[f"ATK:{c.name if c else '?'}:{a.name if a else '?'}"] += 1
                obs = battle_select(sel)
                steps += 1
        finally:
            battle_finish()

    print("winner/reason:")
    for k, v in tally.most_common():
        print(f"  {k}: {v}")
    for w in ("A", "B"):
        if turns_n[w]:
            print(f"avg turns when {w} wins: {turns_by[w]/turns_n[w]:.1f} (n={turns_n[w]})")
    print("\nprize-remaining distribution (top15):")
    for k, v in prz_left.most_common(15):
        print(f"  {k}: {v}")
    print(f"\nA(085) plays per game (n={n}):")
    for k, v in a_plays.most_common(25):
        print(f"  {k}: {v/n:.2f}")
    print(f"\nB(opp) plays per game (n={n}):")
    for k, v in b_plays.most_common(25):
        print(f"  {k}: {v/n:.2f}")


if __name__ == "__main__":
    main()
