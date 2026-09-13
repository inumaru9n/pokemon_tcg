"""EXP-092: Luca本家（sig 1ec0f47981）の条件付き行動頻度集計（フル精読の定量パート）。

082の頻度集計（EXP-082、100戦）と違い、全2,344戦を対象に
「対壁/非壁」「盤面条件」で条件付けた頻度を取る。

usage: uv run python agents/092_marnie_full/aggregate_092.py [--dates 2026-07-16 ...]
"""
import argparse
import csv
import json
import sys
import zipfile
from collections import Counter, defaultdict

sys.path.insert(0, "/Users/akira/kaggle/pokemon_tcg")
from cg import api

ROOT = "/Users/akira/kaggle/pokemon_tcg"
SIG = "1ec0f47981"
TEAM = "Luca"

CARDS = {c.cardId: c for c in api.all_card_data()}
ATKS = {a.attackId: a for a in api.all_attack()}
CTX = {c.value: c.name for c in api.SelectContext}
OT = api.OptionType
AREA = api.AreaType

CRUSTLE = 345
KANG = 756
GRIMM = 648
MUNKI = 112
MORGREM = 647
IMPIDIMP = 646
SNORUNT = 860
FROSLASS = 104
ENERGY = 7


def cname(cid):
    c = CARDS.get(cid)
    return c.name if c else f"card{cid}"


def resolve_card_id(cur, opt, me, sel=None):
    area, idx = opt.get("area"), opt.get("index")
    pi = opt.get("playerIndex", me)
    pi = me if pi is None else pi
    pl = cur["players"][pi]
    try:
        if area == AREA.ACTIVE:
            p = pl["active"][idx]
            return p["id"] if p else None
        if area == AREA.BENCH:
            p = pl["bench"][idx]
            return p["id"] if p else None
        if area == AREA.HAND:
            return pl["hand"][idx]["id"] if pl.get("hand") else None
        if area == AREA.DISCARD:
            return pl["discard"][idx]["id"]
        if area == AREA.LOOKING:
            lk = cur.get("looking")
            return lk[idx]["id"] if lk and lk[idx] else None
        if area == AREA.STADIUM:
            return cur["stadium"][idx]["id"]
        if area == AREA.PRIZE:
            c = pl["prize"][idx]
            return c["id"] if c else None
        if area == AREA.DECK:
            if sel and sel.get("deck"):
                c = sel["deck"][idx]
                return c["id"] if c else None
            return None
    except (IndexError, TypeError, KeyError):
        return None
    return None


def field(pl):
    return [p for p in (pl.get("active") or []) + (pl.get("bench") or []) if p is not None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", nargs="+",
                    default=["2026-07-16", "2026-07-17", "2026-07-18", "2026-07-22"])
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    S = defaultdict(Counter)
    n_games = 0

    for day in args.dates:
        rows_by_ep = defaultdict(dict)
        for r in csv.DictReader(open(f"{ROOT}/data/episodes/{day}/summary.csv")):
            rows_by_ep[r["episode_id"]][int(r["player"])] = r
        z = zipfile.ZipFile(f"{ROOT}/data/episodes/{day}/pokemon-tcg-ai-battle-episodes-{day}.zip")
        names = set(z.namelist())
        done = 0
        for ep, players in sorted(rows_by_ep.items()):
            tgt = [pi for pi, r in players.items()
                   if r["deck_signature"] == SIG and r["team"] == TEAM]
            if not tgt or f"{ep}.json" not in names:
                continue
            if args.limit and done >= args.limit:
                break
            done += 1
            with z.open(f"{ep}.json") as f:
                steps = json.load(f)["steps"]
            for me in tgt:
                n_games += 1
                opp_arch = players.get(1 - me, {}).get("archetype", "?")
                # game-level trackers
                max_grimm_en = 0
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
                    myhand = len(mypl.get("hand") or []) or mypl.get("handCount", 0)
                    ophand = oppl.get("handCount", 0)
                    deck = mypl.get("deckCount", 0)
                    turn = cur["turn"]
                    my_prz = len(mypl.get("prize") or [])
                    opp_field = field(oppl)
                    wall = any(q["id"] == CRUSTLE for q in opp_field)
                    wk = "wall" if wall else "open"
                    chosen = [opts[j] for j in act if isinstance(j, int) and j < len(opts)]

                    for q in field(mypl):
                        if q["id"] == GRIMM:
                            max_grimm_en = max(max_grimm_en, len(q.get("energies") or []))

                    if ctx == "IS_FIRST":
                        for o in chosen:
                            S["is_first"]["YES" if o["type"] == OT.YES else "NO"] += 1
                    elif ctx == "SETUP_ACTIVE_POKEMON":
                        for o in chosen:
                            S["setup_active"][cname(resolve_card_id(cur, o, me))] += 1
                    elif ctx == "SETUP_BENCH_POKEMON":
                        S["setup_bench_count"][len(chosen)] += 1
                        for o in chosen:
                            S["setup_bench"][cname(resolve_card_id(cur, o, me))] += 1
                    elif ctx == "MAIN":
                        for o in chosen:
                            t = o["type"]
                            if t == OT.PLAY:
                                cid = resolve_card_id(cur, {"area": AREA.HAND, "index": o["index"]}, me)
                                S[f"play_{wk}"][cname(cid)] += 1
                                if cid == 1227:  # Lillie
                                    S["lillie_myhand"][min(myhand, 9)] += 1
                                    S["lillie_deck"][min(deck, 20)] += 1
                                elif cid == 1231:  # Dawn
                                    S["dawn_deck"][min(deck, 20)] += 1
                                elif cid == 1219:  # Petrel
                                    S["petrel_myhand"][min(myhand, 9)] += 1
                                elif cid == 1080:  # Unfair Stamp
                                    S["stamp_ophand"][min(ophand, 12)] += 1
                                    S["stamp_myhand"][min(myhand, 9)] += 1
                                    S["stamp_myprz"][my_prz] += 1
                                elif cid == 1182:  # Boss
                                    S[f"boss_play_{wk}_turn"][min(turn, 14)] += 1
                            elif t == OT.ATTACK:
                                if o.get("attackId") == 934:
                                    S["filch_deck"][min(deck, 25)] += 1
                                a = ATKS.get(o.get("attackId"))
                                atk_name = a.name if a else str(o.get("attackId"))
                                act_p = mypl["active"][0] if mypl.get("active") else None
                                att = cname(act_p["id"]) if act_p else "?"
                                S[f"attack_{wk}"][f"{att}:{atk_name}"] += 1
                                if act_p and act_p["id"] == GRIMM:
                                    S[f"grimm_atk_encount_{wk}"][len(act_p.get("energies") or [])] += 1
                                # what was the opp active when we attacked into wall
                                if wall and oppl.get("active") and oppl["active"][0]:
                                    S["attack_wall_oppactive"][f"{att}->{cname(oppl['active'][0]['id'])}"] += 1
                            elif t == OT.RETREAT:
                                act_p = mypl["active"][0] if mypl.get("active") else None
                                if act_p:
                                    S[f"retreat_{wk}"][f"{cname(act_p['id'])}@hp{min(act_p['hp'], 320)}"] += 1
                                    S[f"retreat_who_{wk}"][cname(act_p["id"])] += 1
                            elif t == OT.ABILITY:
                                cid = resolve_card_id(cur, o, me)
                                S[f"ability_{wk}"][cname(cid)] += 1
                            elif t == OT.ATTACH:
                                src = resolve_card_id(cur, {"area": o.get("area"), "index": o.get("index")}, me)
                                dst_area = o.get("inPlayArea")
                                dst_opt = {"area": dst_area, "index": o.get("inPlayIndex")}
                                dst = resolve_card_id(cur, dst_opt, me)
                                dst_p = None
                                try:
                                    if dst_area == AREA.ACTIVE:
                                        dst_p = mypl["active"][o.get("inPlayIndex")]
                                    elif dst_area == AREA.BENCH:
                                        dst_p = mypl["bench"][o.get("inPlayIndex")]
                                except (IndexError, TypeError):
                                    pass
                                en = len(dst_p.get("energies") or []) if dst_p else -1
                                S[f"attach_{wk}"][f"{cname(dst)}(en{en})"] += 1
                            elif t == OT.END:
                                S["end_turn"][wk] += 1
                                if any(oo.get("type") == OT.ATTACK for oo in opts):
                                    S["end_with_attack_avail"][wk] += 1
                    elif ctx in ("SWITCH", "TO_ACTIVE"):
                        for o in chosen:
                            pi = o.get("playerIndex", me)
                            pi = me if pi is None else pi
                            cid = resolve_card_id(cur, o, me)
                            if pi == me:
                                S[f"promote_mine_{wk}"][cname(cid)] += 1
                            else:
                                # gust target: id + hp band + energies
                                p2 = None
                                try:
                                    if o.get("area") == AREA.BENCH:
                                        p2 = oppl["bench"][o.get("index")]
                                except (IndexError, TypeError):
                                    pass
                                en = len(p2.get("energies") or []) if p2 else -1
                                hp = p2["hp"] if p2 else -1
                                S[f"gust_{wk}"][f"{cname(cid)}(hp{hp},en{en})"] += 1
                                S[f"gust_who_{wk}"][cname(cid)] += 1
                    elif ctx == "TO_HAND":
                        eff = sel.get("effect") or {}
                        ename = cname(eff.get("id")) if eff.get("id") else "?"
                        for o in chosen:
                            S[f"tohand[{ename}]"][cname(resolve_card_id(cur, o, me, sel))] += 1
                    elif ctx in ("DISCARD", "TO_DECK", "TO_DECK_BOTTOM"):
                        eff = sel.get("effect") or {}
                        ename = cname(eff.get("id")) if eff.get("id") else "?"
                        for o in chosen:
                            pi = o.get("playerIndex", me)
                            pi = me if pi is None else pi
                            tag = "self" if pi == me else "opp"
                            S[f"{ctx.lower()}_{tag}[{ename}]"][cname(resolve_card_id(cur, o, me))] += 1
                    elif ctx in ("TO_BENCH", "TO_FIELD"):
                        for o in chosen:
                            S["to_bench"][cname(resolve_card_id(cur, o, me, sel))] += 1
                    elif ctx == "ATTACH_FROM":
                        S[f"punkup_count_{wk}"][len(chosen)] += 1
                        for o in chosen:
                            pi2 = o.get("playerIndex", me)
                            pi2 = me if pi2 is None else pi2
                            p2 = None
                            try:
                                if o.get("area") == AREA.ACTIVE:
                                    p2 = cur["players"][pi2]["active"][o.get("index")]
                                elif o.get("area") == AREA.BENCH:
                                    p2 = cur["players"][pi2]["bench"][o.get("index")]
                            except (IndexError, TypeError):
                                pass
                            en = len(p2.get("energies") or []) if p2 else -1
                            cid = resolve_card_id(cur, o, me, sel)
                            S[f"punkup_dst_{wk}"][f"{cname(cid)}(en{en})"] += 1
                    elif ctx in ("DAMAGE_COUNTER", "DAMAGE"):
                        for o in chosen:
                            pi = o.get("playerIndex", me)
                            pi = me if pi is None else pi
                            cid = resolve_card_id(cur, o, me)
                            p2 = None
                            try:
                                if o.get("area") == AREA.BENCH:
                                    p2 = cur["players"][pi]["bench"][o.get("index")]
                                elif o.get("area") == AREA.ACTIVE:
                                    p2 = cur["players"][pi]["active"][o.get("index")]
                            except (IndexError, TypeError):
                                pass
                            hp = p2["hp"] if p2 else -1
                            side = "my" if pi == me else "op"
                            S[f"dmgctr_{wk}_{side}"][f"{cname(cid)}(hp{hp})"] += 1
                    elif ctx == "REMOVE_DAMAGE_COUNTER":
                        for o in chosen:
                            pi = o.get("playerIndex", me)
                            pi = me if pi is None else pi
                            side = "my" if pi == me else "op"
                            S[f"rmctr_{wk}_{side}"][cname(resolve_card_id(cur, o, me))] += 1
                S["max_grimm_energy_per_game_" + ("wallgame" if any(
                    True for _ in [1] if False) else "all")][max_grimm_en] += 1
                # game-level: was it a wall game? re-scan quickly
                # (approximation: opp archetype)
                S[f"maxgrimm_en[{opp_arch}]"][max_grimm_en] += 1

    print(f"games analyzed: {n_games}")
    for key in sorted(S):
        total = sum(S[key].values())
        print(f"\n== {key} (n={total}) ==")
        for k, v in S[key].most_common(24):
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
