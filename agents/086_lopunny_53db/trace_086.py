"""086 vs 相手役の1ゲームトレース（負け筋精読用、EXP-086）。"""
import importlib.util
import json
import random
import sys

sys.path.insert(0, "/Users/akira/kaggle/pokemon_tcg")
from cg.api import all_card_data, all_attack
from cg.game import battle_start, battle_select, battle_finish

table = {c.cardId: c for c in all_card_data()}
atable = {a.attackId: a for a in all_attack()}


def nm(cid):
    return table[cid].name if cid in table else str(cid)


def load_agent(agent_dir, alias):
    spec = importlib.util.spec_from_file_location(alias, f"{agent_dir}/main.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod.agent


def read_deck(agent_dir):
    return [int(x) for x in open(f"{agent_dir}/deck.csv").read().split()]


ROOT = "/Users/akira/kaggle/pokemon_tcg"
A_DIR = f"{ROOT}/agents/086_lopunny_53db"
B_DIR = sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/agents/082_marnie_1ec0"
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0
ME_SEAT = int(sys.argv[3]) if len(sys.argv) > 3 else 0

agent_a = load_agent(A_DIR, "agent086")
agent_b = load_agent(B_DIR, "agentopp")
random.seed(SEED)

if ME_SEAT == 0:
    obs, start = battle_start(read_deck(A_DIR), read_deck(B_DIR))
else:
    obs, start = battle_start(read_deck(B_DIR), read_deck(A_DIR))
me = ME_SEAT

def board_str(pl):
    parts = []
    for i, p in enumerate((pl.get("active") or []) + (pl.get("bench") or [])):
        if p is None:
            parts.append("?")
            continue
        e = len(p.get("energies") or [])
        t = "+B" if any((tl or {}).get("id") == 1175 for tl in (p.get("tools") or [])) else ""
        parts.append(f"{'*' if i==0 else ''}{nm(p['id'])}({p['hp']}/{p['maxHp']}e{e}{t})")
    return " | ".join(parts)


steps = 0
try:
    while steps < 2000:
        result = obs["current"]["result"]
        if result != -1:
            print(f"RESULT: winner seat {result} ({'ME' if result == me else 'OPP' if result in (0,1) else 'DRAW'})")
            break
        pi = obs["current"]["yourIndex"]
        act = agent_a(obs) if pi == me else agent_b(obs)
        cur = obs["current"]
        sel = obs["select"]
        if pi == me and sel and sel.get("option"):
            mepl = cur["players"][me]
            opp = cur["players"][1 - me]
            ctx = sel["context"]
            hand = mepl.get("hand") or []
            descs = []
            for i in act:
                o = sel["option"][i] if i < len(sel["option"]) else None
                if o is None:
                    continue
                ty = o["type"]
                if ty == 7 and o["index"] < len(hand):
                    descs.append("PLAY " + nm(hand[o["index"]]["id"]))
                elif ty == 8:
                    cid = hand[o["index"]]["id"] if o["area"] == 2 and o["index"] < len(hand) else -1
                    descs.append(f"ATT {nm(cid)}->{'act' if o['inPlayArea']==4 else 'b'+str(o['inPlayIndex'])}")
                elif ty == 9:
                    descs.append("EVO " + (nm(hand[o["index"]]["id"]) if o["area"] == 2 and o["index"] < len(hand) else "?"))
                elif ty == 10:
                    descs.append(f"ABILITY a{o['area']}i{o['index']}")
                elif ty == 12:
                    descs.append("RETREAT")
                elif ty == 13:
                    a = atable.get(o["attackId"])
                    op_act = (opp.get("active") or [None])[0]
                    descs.append(f"ATK {a.name if a else o['attackId']} vs {nm(op_act['id']) if op_act else '?'}(hp{op_act['hp'] if op_act else '?'}) opph{opp.get('handCount')}")
                elif ty == 14:
                    descs.append("END")
                elif ty in (1, 2):
                    descs.append("YES" if ty == 1 else "NO")
                elif ty == 3:
                    descs.append(f"CARD(ctx{ctx},a{o['area']}i{o['index']}p{o['playerIndex']})")
                else:
                    descs.append(f"ty{ty}")
            if descs:
                print(f"T{cur.get('turn')} prz{len(mepl.get('prize') or [])}-{len(opp.get('prize') or [])} "
                      f"[{board_str(mepl)}] VS [{board_str(opp)}] :: {'; '.join(descs)}")
        obs = battle_select(act)
        steps += 1
finally:
    battle_finish()
