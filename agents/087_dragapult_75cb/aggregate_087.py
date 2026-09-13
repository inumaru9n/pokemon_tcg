"""EXP-087: 75cb2ab957 (Dragapult ex / Dusknoir, 07-22 n=202, LumenLiquidity) の全リプレイ行動頻度集計。

usage: uv run python agents/087_dragapult_75cb/aggregate_087.py
"""
import csv
import json
import sys
import zipfile
from collections import Counter, defaultdict

sys.path.insert(0, "/Users/akira/kaggle/pokemon_tcg")
from cg import api

ROOT = "/Users/akira/kaggle/pokemon_tcg"
DAY = "2026-07-22"
SIG = "75cb2ab957"

CARDS = {c.cardId: c for c in api.all_card_data()}
ATKS = {a.attackId: a for a in api.all_attack()}
CTX = {c.value: c.name for c in api.SelectContext}
OT = api.OptionType
AREA = api.AreaType

RULEBOX = {cid for cid, c in CARDS.items() if c.ex or c.megaEx}


def cname(cid):
    c = CARDS.get(cid)
    return c.name if c else f"card{cid}"


def resolve(cur, opt, me, sel=None):
    area, idx = opt.get("area"), opt.get("index")
    pi = opt.get("playerIndex", me)
    pl = cur["players"][pi] if pi is not None else None
    try:
        if area == AREA.DECK:
            dk = (sel or {}).get("deck")
            return dk[idx] if dk else None
        if area == AREA.ACTIVE:
            p = pl["active"][idx]
            return p
        if area == AREA.BENCH:
            p = pl["bench"][idx]
            return p
        if area == AREA.HAND:
            return pl["hand"][idx] if pl.get("hand") else None
        if area == AREA.DISCARD:
            return pl["discard"][idx]
        if area == AREA.LOOKING:
            lk = cur.get("looking")
            return lk[idx] if lk else None
        if area == AREA.STADIUM:
            return cur["stadium"][idx]
        if area == AREA.PRIZE:
            return pl["prize"][idx]
    except (IndexError, TypeError, KeyError):
        return None
    return None


def rid(cur, opt, me, sel=None):
    c = resolve(cur, opt, me, sel)
    return c["id"] if isinstance(c, dict) else None


def main():
    rows = list(csv.DictReader(open(f"{ROOT}/data/episodes/{DAY}/summary.csv")))
    byep = defaultdict(list)
    for r in rows:
        byep[r["episode_id"]].append(r)
    mine = [r for r in rows if r["deck_signature"] == SIG]

    z = zipfile.ZipFile(f"{ROOT}/data/episodes/{DAY}/pokemon-tcg-ai-battle-episodes-{DAY}.zip")
    names = set(z.namelist())

    S = defaultdict(Counter)
    n_games = 0

    for r in mine:
        ep = r["episode_id"]
        fn = f"{ep}.json"
        if fn not in names:
            continue
        me = int(r["player"])
        opp_row = [x for x in byep[ep] if x["player"] != r["player"]][0]
        oa = opp_row["archetype"]
        vsalak = "vsAlak" if oa == "Alakazam" else "vsOther"
        with z.open(fn) as f:
            d = json.load(f)
        steps = d["steps"]
        n_games += 1
        for i, st in enumerate(steps):
            p = st[me]
            obs = p.get("observation") or {}
            cur, sel = obs.get("current"), obs.get("select")
            if p.get("status") != "ACTIVE" or cur is None or sel is None:
                continue
            act = steps[i + 1][me].get("action") if i + 1 < len(steps) else None
            if not isinstance(act, list):
                continue
            ctx = CTX.get(sel.get("context"), str(sel.get("context")))
            opts = sel.get("option") or []
            mypl = cur["players"][me]
            oppl = cur["players"][1 - me]
            turn = cur["turn"]
            chosen = [opts[j] for j in act if isinstance(j, int) and j < len(opts)]
            eff = sel.get("effect") or {}
            ename = cname(eff.get("id")) if eff.get("id") else "?"

            if ctx == "IS_FIRST":
                for o in chosen:
                    S["is_first"]["YES" if o["type"] == OT.YES else "NO"] += 1
            elif ctx == "SETUP_ACTIVE_POKEMON":
                for o in chosen:
                    S["setup_active"][cname(rid(cur, o, me))] += 1
            elif ctx == "SETUP_BENCH_POKEMON":
                S["setup_bench_count"][len(chosen)] += 1
                for o in chosen:
                    S["setup_bench"][cname(rid(cur, o, me))] += 1
            elif ctx == "MAIN":
                for o in chosen:
                    t = o["type"]
                    if t == OT.PLAY:
                        cid = rid(cur, {"area": AREA.HAND, "index": o["index"]}, me)
                        S["play"][cname(cid)] += 1
                    elif t == OT.ATTACK:
                        a = ATKS.get(o.get("attackId"))
                        atk_name = a.name if a else str(o.get("attackId"))
                        act_p = mypl["active"][0] if mypl.get("active") else None
                        att = cname(act_p["id"]) if act_p else "?"
                        S["attack"][f"{att}:{atk_name}"] += 1
                        S[f"attack_{vsalak}"][f"{att}:{atk_name}"] += 1
                    elif t == OT.RETREAT:
                        S["retreat"]["count"] += 1
                    elif t == OT.ABILITY:
                        c = resolve(cur, o, me)
                        cid = c["id"] if c else None
                        S["ability"][cname(cid)] += 1
                        if cid in (132, 133):  # Cursed Blast
                            S["cursed_blast_turn"][f"{cname(cid)}:T{min(turn,12)}"] += 1
                    elif t == OT.ATTACH:
                        src = rid(cur, {"area": o.get("area"), "index": o.get("index")}, me)
                        dst = rid(cur, {"area": o.get("inPlayArea"), "index": o.get("inPlayIndex")}, me)
                        S["attach"][f"{cname(src)}->{cname(dst)}"] += 1
                    elif t == OT.EVOLVE:
                        src = rid(cur, {"area": o.get("area"), "index": o.get("index")}, me)
                        dst = rid(cur, {"area": o.get("inPlayArea"), "index": o.get("inPlayIndex")}, me)
                        S["evolve"][f"{cname(dst)}->{cname(src)}"] += 1
                    elif t == OT.END:
                        S["end_turn"]["count"] += 1
            elif ctx in ("SWITCH", "TO_ACTIVE"):
                for o in chosen:
                    if o.get("playerIndex") == me:
                        S["to_active_mine"][cname(rid(cur, o, me))] += 1
                    else:
                        c = resolve(cur, o, me)
                        tag = f"{cname(c['id'])}(hp{c['hp']},e{len(c.get('energies') or [])})" if c else "?"
                        S["gust_target"][tag] += 1
            elif ctx == "TO_HAND":
                for o in chosen:
                    S[f"tohand[{ename}]"][cname(rid(cur, o, me, sel))] += 1
            elif ctx in ("DISCARD", "TO_DECK", "TO_DECK_BOTTOM"):
                for o in chosen:
                    pi = o.get("playerIndex", me)
                    tag = "self" if pi == me else "opp"
                    S[f"discard_{tag}[{ename}]"][cname(rid(cur, o, me))] += 1
            elif ctx in ("TO_BENCH", "TO_FIELD"):
                for o in chosen:
                    S[f"to_bench[{ename}]"][cname(rid(cur, o, me, sel))] += 1
            elif ctx == "DAMAGE_COUNTER_ANY":
                # Phantom Dive scatter: 6 counters on bench
                tag = []
                for o in chosen:
                    c = resolve(cur, o, me)
                    if c:
                        S[f"pdive_target_{vsalak}"][f"{cname(c['id'])}(hp{c['hp']})"] += 1
                S["pdive_ntargets"][len(chosen)] += 1
            elif ctx == "DAMAGE_COUNTER":
                # Cursed Blast (5 or 13) / Adrena-Brain destination
                for o in chosen:
                    pi = o.get("playerIndex", me)
                    c = resolve(cur, o, me)
                    side = "opp" if pi != me else "self"
                    if c:
                        S[f"dmgctr[{ename}]_{side}_{vsalak}"][f"{cname(c['id'])}(hp{c['hp']})"] += 1
            elif ctx in ("REMOVE_DAMAGE_COUNTER", "HEAL"):
                for o in chosen:
                    c = resolve(cur, o, me)
                    if c:
                        S[f"heal[{ename}]"][cname(c["id"])] += 1
            elif ctx in ("DAMAGE_COUNTER_COUNT", "DRAW_COUNT", "REMOVE_DAMAGE_COUNTER_COUNT"):
                for o in chosen:
                    S[f"count[{ctx}][{ename}]"][o.get("number")] += 1
            elif ctx == "EFFECT_TARGET":
                for o in chosen:
                    c = resolve(cur, o, me)
                    pi = o.get("playerIndex", me)
                    side = "opp" if pi != me else "self"
                    if c:
                        S[f"efftarget[{ename}]_{side}"][f"{cname(c['id'])}(hp{c.get('hp')})"] += 1
            else:
                for o in chosen:
                    S[f"other[{ctx}][{ename}]"][str(o.get("type"))] += 1

    print(f"games analyzed: {n_games}")
    for key in sorted(S):
        print(f"\n== {key} ==")
        for k, v in S[key].most_common(25):
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
