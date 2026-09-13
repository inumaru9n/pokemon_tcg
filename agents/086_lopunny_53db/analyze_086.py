"""tw_shin Lopunny (sig 53db4c87b8) の行動頻度集計スクリプト（EXP-086）。"""
import csv
import json
import sys
import zipfile
from collections import Counter, defaultdict

sys.path.insert(0, "/Users/akira/kaggle/pokemon_tcg")
from cg.api import all_card_data, all_attack

ROOT = "/Users/akira/kaggle/pokemon_tcg"
ZIP = f"{ROOT}/data/episodes/2026-07-22/pokemon-tcg-ai-battle-episodes-2026-07-22.zip"
SUM = f"{ROOT}/data/episodes/2026-07-22/summary.csv"
SIG = "53db4c87b8"

table = {c.cardId: c for c in all_card_data()}
atable = {a.attackId: a for a in all_attack()}


def name(cid):
    c = table.get(cid)
    return c.name if c else f"?{cid}"


rows = [r for r in csv.DictReader(open(SUM)) if r["deck_signature"] == SIG]
targets = {r["episode_id"]: int(r["player"]) for r in rows}
print(f"episodes: {len(targets)}")

z = zipfile.ZipFile(ZIP)

is_first = Counter()
setup_active = Counter()
setup_bench = Counter()
attach = Counter()          # (card, target_pokemon)
attacks = Counter()         # (pokemon?, attack name) — attackIdのみ
plays = Counter()           # played card names
abilities = Counter()
retreats = Counter()        # retreat時のactive pokemon
switch_choice = Counter()   # SWITCH/TO_ACTIVE で選んだ pokemon
to_active_ctx = Counter()
boss_target = Counter()     # 相手盤面から選んだカード（playerIndex != me の CARD）
to_hand = Counter()
discard = Counter()
evolve = Counter()
ctx_counter = Counter()
end_turn = 0
attack_turnstats = defaultdict(Counter)  # attack name -> moved_this_turn? 集計不能なので省略

n_ep = 0
for ep, pi in targets.items():
    try:
        d = json.loads(z.read(f"{ep}.json"))
    except KeyError:
        continue
    n_ep += 1
    steps_d = d["steps"]
    for t in range(1, len(steps_d)):
        # action[t] は observation[t-1] への応答（トレースで確認済み）
        act = steps_d[t][pi].get("action")
        obs = steps_d[t - 1][pi].get("observation") or {}
        sel = obs.get("select")
        if not act or not sel or not sel.get("option"):
            continue
        ctx = sel["context"]
        cur = obs.get("current") or {}
        players = cur.get("players") or [{}, {}]
        me = players[pi] if pi < len(players) else {}
        opp = players[1 - pi] if len(players) > 1 else {}
        hand = me.get("hand") or []
        my_board = (me.get("active") or []) + (me.get("bench") or [])
        opp_board = (opp.get("active") or []) + (opp.get("bench") or [])

        for i in act:
            if i >= len(sel["option"]):
                continue
            o = sel["option"][i]
            t = o["type"]
            ctx_counter[ctx] += 1
            if ctx == 41:  # IS_FIRST
                is_first["yes" if t == 1 else "no"] += 1
            elif ctx == 1 and t == 3:  # SETUP_ACTIVE
                cards = hand
                cid = cards[o["index"]]["id"] if o["index"] < len(cards) else -1
                setup_active[name(cid)] += 1
            elif ctx == 2 and t == 3:  # SETUP_BENCH
                cid = hand[o["index"]]["id"] if o["index"] < len(hand) else -1
                setup_bench[name(cid)] += 1
            elif t == 7:  # PLAY
                cid = hand[o["index"]]["id"] if o["index"] < len(hand) else -1
                plays[name(cid)] += 1
            elif t == 8:  # ATTACH
                cid = hand[o["index"]]["id"] if o["area"] == 2 and o["index"] < len(hand) else -1
                board = my_board
                bi = o["inPlayIndex"] if o["inPlayArea"] == 5 else -1
                if o["inPlayArea"] == 4:
                    pk = (me.get("active") or [None])[0]
                elif o["inPlayArea"] == 5:
                    bench = me.get("bench") or []
                    pk = bench[o["inPlayIndex"]] if o["inPlayIndex"] < len(bench) else None
                else:
                    pk = None
                pname = name(pk["id"]) if pk else "?"
                loc = "act" if o["inPlayArea"] == 4 else "bench"
                attach[(name(cid), pname, loc)] += 1
            elif t == 9:  # EVOLVE
                cid = hand[o["index"]]["id"] if o["area"] == 2 and o["index"] < len(hand) else -1
                evolve[name(cid)] += 1
            elif t == 10:  # ABILITY
                if o["area"] == 4:
                    pk = (me.get("active") or [None])[0]
                elif o["area"] == 5:
                    bench = me.get("bench") or []
                    pk = bench[o["index"]] if o["index"] < len(bench) else None
                else:
                    pk = None
                abilities[name(pk["id"]) if pk else f'area{o["area"]}'] += 1
            elif t == 12:  # RETREAT
                pk = (me.get("active") or [None])[0]
                retreats[name(pk["id"]) if pk else "?"] += 1
            elif t == 13:  # ATTACK
                a = atable.get(o["attackId"])
                pk = (me.get("active") or [None])[0]
                attacks[(name(pk["id"]) if pk else "?", a.name if a else o["attackId"])] += 1
            elif t == 14:
                end_turn += 1
            elif t == 3:  # CARD in other contexts
                area, idx, owner = o["area"], o["index"], o["playerIndex"]
                if area == 2:
                    cid = hand[idx]["id"] if idx < len(hand) else -1
                elif area == 4:
                    b = (players[owner].get("active") or []) if owner < len(players) else []
                    cid = b[0]["id"] if b else -1
                elif area == 5:
                    b = (players[owner].get("bench") or []) if owner < len(players) else []
                    cid = b[idx]["id"] if idx < len(b) else -1
                elif area == 12:
                    lk = cur.get("looking") or []
                    cid = lk[idx]["id"] if idx < len(lk) else -1
                elif area == 1:
                    dk = sel.get("deck") or []
                    cid = dk[idx]["id"] if idx < len(dk) else -1
                elif area == 3:
                    dc = players[owner].get("discard") or [] if owner < len(players) else []
                    cid = dc[idx]["id"] if idx < len(dc) else -1
                else:
                    cid = -1
                nm = name(cid)
                if owner != pi:
                    boss_target[(ctx, nm)] += 1
                elif ctx in (3, 4):
                    switch_choice[(ctx, nm)] += 1
                elif ctx == 7:
                    to_hand[nm] += 1
                elif ctx in (8, 9, 10):
                    discard[(ctx, nm)] += 1
                elif ctx == 21:  # ATTACH_FROM
                    attach[("(effect)", nm, "?")] += 1
                elif ctx == 17:  # HEAL (Wally target)
                    switch_choice[("heal", nm)] += 1

print(f"parsed {n_ep} episodes")
print("\n== IS_FIRST ==", dict(is_first))
print("\n== SETUP_ACTIVE ==", setup_active.most_common())
print("\n== SETUP_BENCH ==", setup_bench.most_common())
print("\n== EVOLVE ==", evolve.most_common())
print("\n== PLAYS ==")
for k, v in plays.most_common(30):
    print(f"  {v:5d}  {k}")
print("\n== ATTACH (card -> pokemon, loc) ==")
for k, v in attach.most_common(30):
    print(f"  {v:5d}  {k}")
print("\n== ATTACKS (active, attack) ==")
for k, v in attacks.most_common(20):
    print(f"  {v:5d}  {k}")
print("\n== ABILITIES ==", abilities.most_common())
print("\n== RETREATS (from) ==", retreats.most_common())
print("\n== SWITCH/TO_ACTIVE choice ==")
for k, v in switch_choice.most_common(20):
    print(f"  {v:5d}  {k}")
print("\n== OPP-BOARD picks (ctx, card) ==")
for k, v in boss_target.most_common(20):
    print(f"  {v:5d}  {k}")
print("\n== TO_HAND ==")
for k, v in to_hand.most_common(25):
    print(f"  {v:5d}  {k}")
print("\n== DISCARD/TO_DECK ==")
for k, v in discard.most_common(25):
    print(f"  {v:5d}  {k}")
print("\n== END_TURN ==", end_turn)
print("\n== ctx counts ==", ctx_counter.most_common())
