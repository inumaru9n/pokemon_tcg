"""Play one local game 087 vs opponent and print a narrative log from 087's view.

usage: uv run python agents/087_dragapult_75cb/debug_087.py [opp_dir] [game_index]
"""
import importlib.util
import json
import random
import sys

sys.path.insert(0, "/Users/akira/kaggle/pokemon_tcg")
from cg import api
from cg.game import battle_start, battle_select, battle_finish

CARDS = {c.cardId: c for c in api.all_card_data()}
ATKS = {a.attackId: a for a in api.all_attack()}
CTX = {c.value: c.name for c in api.SelectContext}
OT = api.OptionType
AREA = {a.value: a.name for a in api.AreaType}


def load_agent(agent_dir, alias):
    spec = importlib.util.spec_from_file_location(f"agent_{alias}", f"{agent_dir}/main.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.agent


def read_deck(agent_dir):
    return [int(x) for x in open(f"{agent_dir}/deck.csv").read().split()]


def cname(cid):
    c = CARDS.get(cid)
    return c.name if c else f"card{cid}"


def poke_str(p):
    if p is None:
        return "(facedown)"
    en = "".join(str(e) for e in (p.get("energies") or []))
    s = f"{cname(p['id'])}[{p['hp']}/{p['maxHp']}"
    if en:
        s += f" E:{en}"
    return s + "]"


def board(cur, me):
    out = []
    for pi in (me, 1 - me):
        pl = cur["players"][pi]
        tag = "ME " if pi == me else "OPP"
        act = pl["active"][0] if pl["active"] else None
        bench = " ".join(poke_str(b) for b in pl["bench"]) or "-"
        hand = pl.get("hand")
        hands = " ".join(sorted(cname(c["id"]) for c in hand)) if hand else f"({pl['handCount']})"
        out.append(f"  {tag} prz={len(pl['prize'])} deck={pl['deckCount']} | ACT {poke_str(act)} | BENCH {bench} | hand: {hands}")
    return "\n".join(out)


def opt_str(o, cur, me):
    t = o["type"]
    if t == OT.NUMBER: return f"count={o.get('number')}"
    if t == OT.YES: return "YES"
    if t == OT.NO: return "NO"
    if t == OT.END: return "END"
    if t == OT.RETREAT: return "RETREAT"
    if t == OT.ATTACK:
        a = ATKS.get(o.get("attackId"))
        return f"ATTACK {a.name if a else o.get('attackId')}"
    if t == OT.ABILITY:
        return f"ABILITY {AREA.get(o.get('area'),'?')}[{o.get('index')}]"
    if t == OT.PLAY:
        pl = cur["players"][me]
        h = pl.get("hand")
        nm = cname(h[o["index"]]["id"]) if h and o.get("index") is not None and o["index"] < len(h) else "?"
        return f"PLAY {nm}"
    if t == OT.EVOLVE:
        return f"EVOLVE hand[{o.get('index')}] onto {AREA.get(o.get('inPlayArea'),'?')}[{o.get('inPlayIndex')}]"
    if t == OT.ATTACH:
        return f"ATTACH hand[{o.get('index')}] -> {AREA.get(o.get('inPlayArea'),'?')}[{o.get('inPlayIndex')}]"
    if t == OT.CARD:
        pi = o.get("playerIndex")
        area = o.get("area")
        idx = o.get("index")
        try:
            pl = cur["players"][pi]
            if area == api.AreaType.ACTIVE:
                return f"CARD {'my' if pi==me else 'opp'} ACT {poke_str(pl['active'][idx])}"
            if area == api.AreaType.BENCH:
                return f"CARD {'my' if pi==me else 'opp'} BENCH {poke_str(pl['bench'][idx])}"
            if area == api.AreaType.HAND:
                return f"CARD my HAND {cname(pl['hand'][idx]['id'])}"
            if area == api.AreaType.DISCARD:
                return f"CARD {'my' if pi==me else 'opp'} DISC {cname(pl['discard'][idx]['id'])}"
        except Exception:
            pass
        return f"CARD {AREA.get(area,'?')}[{idx}]p{pi}"
    if t in (OT.ENERGY, OT.ENERGY_CARD):
        return f"ENERGY {AREA.get(o.get('area'),'?')}[{o.get('index')}]p{o.get('playerIndex')}"
    return f"opt{t}"


def main():
    opp_dir = sys.argv[1] if len(sys.argv) > 1 else "agents/045_alakazam_full"
    gi = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    me_dir = "agents/087_dragapult_75cb"
    a = load_agent(me_dir, "A")
    b = load_agent(opp_dir, "B")
    random.seed(int(__import__("os").environ.get("SEEDBASE","42")) + gi)
    me = gi % 2  # seat of 087
    decks = {me: read_deck(me_dir), 1 - me: read_deck(opp_dir)}
    agents = {me: a, 1 - me: b}
    obs, start = battle_start(decks[0], decks[1])
    steps = 0
    last_turn = -1
    try:
        while True:
            res = obs["current"]["result"]
            if res != -1:
                print(f"##### RESULT: {'ME WINS' if res == me else ('OPP WINS' if res == 1 - me else 'DRAW')}")
                break
            if steps > 4000:
                print("step cap")
                break
            pi = obs["current"]["yourIndex"]
            sel_meta = obs["select"]
            cur = obs["current"]
            if pi == me:
                if cur["turn"] != last_turn:
                    last_turn = cur["turn"]
                    print(f"\n### turn {last_turn}")
                    print(board(cur, me))
                ctx = CTX.get(sel_meta["context"], "?") if sel_meta else "INIT"
            sel = agents[pi](obs)
            if pi == me and sel_meta is not None:
                opts = sel_meta["option"]
                chosen = " | ".join(opt_str(opts[j], cur, me) for j in sel if j < len(opts))
                print(f"  [{ctx} {len(sel)}/{len(opts)}] {chosen}")
            obs = battle_select(sel)
            steps += 1
    finally:
        battle_finish()


if __name__ == "__main__":
    main()
