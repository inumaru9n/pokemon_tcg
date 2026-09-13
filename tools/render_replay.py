"""Render a Kaggle Pokemon TCG episode JSON into a readable narrative log
from the perspective of one player (default: the 491b8bfb26 Starmie player)."""
import sys, json, zipfile, argparse

sys.path.insert(0, "/Users/akira/kaggle/pokemon_tcg")
from cg import api

CARDS = {c.cardId: c for c in api.all_card_data()}
ATKS = {a.attackId: a for a in api.all_attack()}
ETYPE = {e.value: e.name[:2] for e in api.EnergyType}
AREA = {a.value: a.name for a in api.AreaType}

def cname(cid):
    c = CARDS.get(cid)
    return c.name if c else f"card{cid}"

def aname(aid):
    a = ATKS.get(aid)
    return f"{a.name}({a.damage})" if a else f"atk{aid}"


def poke_str(p):
    if p is None:
        return "(facedown)"
    en = "".join(ETYPE.get(e, "?") for e in (p.get("energies") or []))
    tools = ",".join(cname(t["id"]) for t in (p.get("tools") or []))
    s = f"{cname(p['id'])}[{p['hp']}/{p['maxHp']}"
    if en:
        s += f" E:{en}"
    if tools:
        s += f" T:{tools}"
    return s + "]"


def board_str(cur, me):
    out = []
    for pi in (me, 1 - me):
        pl = cur["players"][pi]
        tag = "ME " if pi == me else "OPP"
        act = pl["active"][0] if pl["active"] else None
        bench = " ".join(poke_str(b) for b in pl["bench"]) or "-"
        prz = sum(1 for x in pl["prize"] if x is not None or True)
        prz = len([x for x in pl["prize"]])
        cond = "".join(
            k[0].upper() for k in ("poisoned", "burned", "asleep", "paralyzed", "confused") if pl.get(k)
        )
        hand = pl.get("hand")
        hands = " ".join(sorted(cname(c["id"]) for c in hand)) if hand else f"({pl['handCount']} cards)"
        out.append(
            f"  {tag} prz={prz} deck={pl['deckCount']} | ACT {poke_str(act)}{('<'+cond+'>') if cond else ''} | BENCH {bench}\n"
            f"      hand: {hands}"
        )
    return "\n".join(out)


def render_log(lg, me):
    t = lg.get("type")
    pi = lg.get("playerIndex")
    who = "ME" if pi == me else "OPP"
    T = api.LogType
    if t == T.SHUFFLE: return None
    if t == T.HAS_BASIC_POKEMON: return None
    if t == T.TURN_START: return f"===== TURN START ({who}) ====="
    if t == T.TURN_END: return f"----- turn end ({who})"
    if t == T.DRAW: return f"{who} draws {cname(lg['cardId'])}"
    if t == T.DRAW_REVERSE: return f"{who} draws a card"
    if t == T.MOVE_CARD:
        return f"{who} moves {cname(lg['cardId'])}: {AREA.get(lg.get('fromArea'),'?')} -> {AREA.get(lg.get('toArea'),'?')}"
    if t == T.MOVE_CARD_REVERSE:
        return f"{who} moves facedown card: {AREA.get(lg.get('fromArea'),'?')} -> {AREA.get(lg.get('toArea'),'?')}"
    if t == T.SWITCH:
        return f"{who} switches: {cname(lg.get('cardIdBench'))} out, {cname(lg.get('cardIdActive'))} in"
    if t == T.CHANGE:
        return f"{who} change: {cname(lg.get('cardIdBefore'))} -> {cname(lg.get('cardIdAfter'))}"
    if t == T.PLAY: return f"{who} PLAYS {cname(lg['cardId'])}"
    if t == T.ATTACH:
        return f"{who} attaches {cname(lg['cardId'])} -> {cname(lg.get('cardIdTarget'))}"
    if t == T.EVOLVE:
        return f"{who} EVOLVES {cname(lg.get('cardIdTarget'))} -> {cname(lg['cardId'])}"
    if t == T.DEVOLVE:
        return f"{who} devolves {cname(lg.get('cardIdTarget'))} removing {cname(lg['cardId'])}"
    if t == T.MOVE_ATTACHED:
        return f"{who} moves attached {cname(lg['cardId'])}: {cname(lg.get('cardIdBefore'))} -> {cname(lg.get('cardIdAfter'))}"
    if t == T.ATTACK:
        return f"{who} *** ATTACK: {cname(lg['cardId'])} uses {aname(lg.get('attackId'))} ***"
    if t == T.HP_CHANGE:
        v = lg.get("value", 0)
        kind = "damage counters" if lg.get("putDamageCounter") else "HP"
        sign = "+" if v > 0 else ""
        return f"    {who} {cname(lg['cardId'])} {kind} {sign}{v}"
    if t in (T.POISONED, T.BURNED, T.ASLEEP, T.PARALYZED, T.CONFUSED):
        cond = T(t).name
        return f"{who} {cname(lg.get('cardId'))} {'recovers from' if lg.get('isRecover') else 'is'} {cond}"
    if t == T.COIN:
        return f"coin: {'HEADS' if lg.get('head') else 'TAILS'}"
    if t == T.RESULT:
        r = lg.get("result"); reason = lg.get("reason")
        rs = {1: "prizes taken", 2: "deck out", 3: "no active", 4: "card effect"}.get(reason, reason)
        w = "ME" if r == me else ("OPP" if r == 1 - me else "DRAW")
        return f"##### RESULT: winner={w} reason={rs} #####"
    return f"{who} log{t}: {json.dumps(lg)}"


def find_card_in_state(cur, opt):
    """Resolve a CARD option to a name using current state."""
    area, idx, pi = opt.get("area"), opt.get("index"), opt.get("playerIndex")
    pl = cur["players"][pi] if pi is not None else None
    try:
        if area == api.AreaType.ACTIVE:
            p = pl["active"][idx]
            return poke_str(p) if p else "(facedown active)"
        if area == api.AreaType.BENCH:
            return poke_str(pl["bench"][idx])
        if area == api.AreaType.HAND:
            return cname(pl["hand"][idx]["id"]) if pl.get("hand") else f"hand[{idx}]"
        if area == api.AreaType.DISCARD:
            return cname(pl["discard"][idx]["id"])
        if area == api.AreaType.LOOKING:
            lk = cur.get("looking")
            c = lk[idx] if lk else None
            return cname(c["id"]) if c else f"looking[{idx}]"
        if area == api.AreaType.STADIUM:
            return cname(cur["stadium"][idx]["id"])
        if area == api.AreaType.PRIZE:
            c = pl["prize"][idx]
            return cname(c["id"]) if c else f"prize[{idx}](facedown)"
    except (IndexError, TypeError, KeyError):
        pass
    return f"{AREA.get(area,'?')}[{idx}]p{pi}"


def opt_str(opt, cur, me):
    T = api.OptionType
    t = opt["type"]
    if t == T.NUMBER: return f"count={opt.get('number')}"
    if t == T.YES: return "YES"
    if t == T.NO: return "NO"
    if t == T.CARD:
        who = "my" if opt.get("playerIndex") == me else "opp"
        return f"CARD {who} {AREA.get(opt.get('area'),'?')}[{opt.get('index')}] {find_card_in_state(cur, opt)}"
    if t == T.TOOL_CARD: return f"TOOL on {find_card_in_state(cur, opt)}"
    if t in (T.ENERGY_CARD, T.ENERGY):
        return f"ENERGY(idx{opt.get('energyIndex')}) on {find_card_in_state(cur, opt)}"
    if t == T.PLAY:
        pl = cur["players"][me]
        h = pl.get("hand")
        nm = cname(h[opt["index"]]["id"]) if h and opt.get("index") is not None and opt["index"] < len(h) else "?"
        return f"PLAY {nm}"
    if t == T.ATTACH:
        src = find_card_in_state(cur, {"area": opt.get("area"), "index": opt.get("index"), "playerIndex": me})
        dst = find_card_in_state(cur, {"area": opt.get("inPlayArea"), "index": opt.get("inPlayIndex"), "playerIndex": me})
        return f"ATTACH {src} -> {dst}"
    if t == T.EVOLVE:
        src = find_card_in_state(cur, {"area": opt.get("area"), "index": opt.get("index"), "playerIndex": me})
        dst = find_card_in_state(cur, {"area": opt.get("inPlayArea"), "index": opt.get("inPlayIndex"), "playerIndex": me})
        return f"EVOLVE {src} onto {dst}"
    if t == T.ABILITY:
        return f"ABILITY of {find_card_in_state(cur, {'area': opt.get('area'), 'index': opt.get('index'), 'playerIndex': me})}"
    if t == T.DISCARD:
        return f"DISCARD-inplay {find_card_in_state(cur, {'area': opt.get('area'), 'index': opt.get('index'), 'playerIndex': me})}"
    if t == T.RETREAT: return "RETREAT"
    if t == T.ATTACK: return f"ATTACK {aname(opt.get('attackId'))}"
    if t == T.END: return "END TURN"
    if t == T.SKILL: return f"SKILL {cname(opt.get('cardId'))}"
    if t == T.SPECIAL_CONDITION: return f"COND {opt.get('specialConditionType')}"
    return json.dumps(opt)


CTX = {c.value: c.name for c in api.SelectContext}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episode_id")
    ap.add_argument("--player", type=int, default=None, help="perspective player index")
    ap.add_argument("--zip", default="/Users/akira/kaggle/pokemon_tcg/data/episodes/2026-07-08/pokemon-tcg-ai-battle-episodes-2026-07-08.zip")
    ap.add_argument("--full-options", action="store_true", help="print all options, not just chosen")
    args = ap.parse_args()

    z = zipfile.ZipFile(args.zip)
    with z.open(f"{args.episode_id}.json") as f:
        d = json.load(f)
    steps = d["steps"]

    # find our player by deck signature if not given
    me = args.player
    if me is None:
        for st in steps:
            for pi in (0, 1):
                a = st[pi].get("action")
                if isinstance(a, list) and len(a) == 60 and a.count(3) == 9 and a.count(666) == 4:
                    me = pi
        if me is None:
            print("could not find 491b player; use --player"); return
    print(f"# episode {args.episode_id}, ME = player {me} (Starmie/Cinderace 491b)")

    last_turn_printed = -1
    prev_logs = None
    for i, st in enumerate(steps):
        p = st[me]
        obs = p.get("observation") or {}
        cur = obs.get("current")
        sel = obs.get("select")
        logs = obs.get("logs") or []

        # dedupe logs: repeated identical blocks while waiting
        if logs != prev_logs:
            new = logs
            if prev_logs and len(logs) > len(prev_logs) and logs[: len(prev_logs)] == prev_logs:
                new = logs[len(prev_logs):]
            for lg in new:
                s = render_log(lg, me)
                if s:
                    print(s)
            prev_logs = logs

        if p.get("status") != "ACTIVE" or cur is None or sel is None:
            continue
        # action responding to this observation is recorded at the next step
        act = steps[i + 1][me].get("action") if i + 1 < len(steps) else None

        # print board at each new turn
        if cur["turn"] != last_turn_printed:
            print(f"\n### turn {cur['turn']} (first={cur['firstPlayer']}) board:")
            print(board_str(cur, me))
            last_turn_printed = cur["turn"]

        if not isinstance(act, list):
            continue
        ctx = CTX.get(sel.get("context"), sel.get("context"))
        opts = sel.get("option") or []
        chosen = [opt_str(opts[j], cur, me) for j in act if isinstance(j, int) and j < len(opts)]
        n = len(opts)
        if args.full_options and n > 1:
            print(f"  [t{cur['turn']} {ctx}] options({n}):")
            for j, o in enumerate(opts):
                mark = "*" if j in act else " "
                print(f"   {mark} {j}: {opt_str(o, cur, me)}")
        else:
            if not chosen and sel.get("minCount", 0) == 0:
                chosen = ["(pass/none)"]
            print(f"  >> [t{cur['turn']} {ctx} {len(act)}/{n}] " + " | ".join(chosen))
    # final result
    rew = d.get("rewards")
    print(f"\n# rewards={rew} (ME=player{me})")


if __name__ == "__main__":
    main()
