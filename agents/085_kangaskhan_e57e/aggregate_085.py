"""EXP-085: e57eb95734 (Mega Kangaskhan, 07-22 n=440) の全リプレイ行動頻度集計。

usage: uv run python agents/085_kangaskhan_e57e/aggregate_085.py
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
SIG = "e57eb95734"

CARDS = {c.cardId: c for c in api.all_card_data()}
ATKS = {a.attackId: a for a in api.all_attack()}
CTX = {c.value: c.name for c in api.SelectContext}
OT = api.OptionType
AREA = api.AreaType

RULEBOX = {cid for cid, c in CARDS.items() if c.ex or c.megaEx}

KANG, DWEBBLE, CRUSTLE = 756, 344, 345


def cname(cid):
    c = CARDS.get(cid)
    return c.name if c else f"card{cid}"


def opp_has_rulebox(cur, me):
    op = cur["players"][1 - me]
    for p in (op.get("active") or []) + (op.get("bench") or []):
        if p is not None and p["id"] in RULEBOX:
            return True
    for c in op.get("discard") or []:
        if c["id"] in RULEBOX:
            return True
    return False


def resolve_card_id(cur, opt, me):
    area, idx = opt.get("area"), opt.get("index")
    pi = opt.get("playerIndex", me)
    pl = cur["players"][pi] if pi is not None else None
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
    except (IndexError, TypeError, KeyError):
        return None
    return None


def main():
    rows = [r for r in csv.DictReader(open(f"{ROOT}/data/episodes/{DAY}/summary.csv"))
            if r["deck_signature"] == SIG]
    eps = {}  # episode_id -> set of players with sig
    for r in rows:
        eps.setdefault(r["episode_id"], set()).add(int(r["player"]))

    z = zipfile.ZipFile(f"{ROOT}/data/episodes/{DAY}/pokemon-tcg-ai-battle-episodes-{DAY}.zip")
    names = set(z.namelist())

    S = defaultdict(Counter)  # stats
    n_games = 0

    for ep, players in sorted(eps.items()):
        fn = f"{ep}.json"
        if fn not in names:
            continue
        with z.open(fn) as f:
            d = json.load(f)
        steps = d["steps"]
        for me in players:
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
                rb = opp_has_rulebox(cur, me)
                rbk = "rb" if rb else "nonrb"
                mypl = cur["players"][me]
                oppl = cur["players"][1 - me]
                myhand = len(mypl.get("hand") or []) or mypl.get("handCount", 0)
                ophand = oppl.get("handCount", 0)
                turn = cur["turn"]
                chosen = [opts[j] for j in act if isinstance(j, int) and j < len(opts)]

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
                            S["play"][cname(cid)] += 1
                            S[f"play_{rbk}"][cname(cid)] += 1
                            if cid == 1087:  # Hand Trimmer
                                S["trimmer_opphand"][ophand] += 1
                                S["trimmer_myhand"][myhand] += 1
                            if cid == 1197:  # Xerosic
                                S["xerosic_opphand"][ophand] += 1
                            if cid == 1227:  # Lillie
                                S["lillie_myhand"][myhand] += 1
                            if cid == 756:
                                # count kangaskhan already on my board
                                nk = sum(1 for q in (mypl.get("active") or []) + (mypl.get("bench") or [])
                                         if q is not None and q["id"] == KANG)
                                S["kang_play_boardkang"][nk] += 1
                                S[f"kang_play_{rbk}_turn"][min(turn, 10)] += 1
                            if cid == 1264:  # Battle Cage
                                st_id = cur["stadium"][0]["id"] if cur.get("stadium") else 0
                                mine = bool(cur.get("stadium")) and cur["stadium"][0].get("playerIndex") == me
                                S["cage_when"][f"stadium={cname(st_id) if st_id else 'none'}|mine={mine}"] += 1
                        elif t == OT.ATTACK:
                            a = ATKS.get(o.get("attackId"))
                            atk_name = a.name if a else str(o.get("attackId"))
                            act_p = mypl["active"][0] if mypl.get("active") else None
                            att = cname(act_p["id"]) if act_p else "?"
                            S[f"attack_{rbk}"][f"{att}:{atk_name}"] += 1
                        elif t == OT.RETREAT:
                            S["retreat"]["count"] += 1
                        elif t == OT.ABILITY:
                            cid = resolve_card_id(cur, o, me)
                            S["ability"][cname(cid)] += 1
                        elif t == OT.ATTACH:
                            src = resolve_card_id(cur, {"area": o.get("area"), "index": o.get("index")}, me)
                            dst = resolve_card_id(cur, {"area": o.get("inPlayArea"), "index": o.get("inPlayIndex")}, me)
                            if src is not None and CARDS.get(src) and CARDS[src].cardType in (
                                api.CardType.BASIC_ENERGY, api.CardType.SPECIAL_ENERGY
                            ):
                                S[f"attach_energy_{rbk}"][f"{cname(src)}->{cname(dst)}"] += 1
                            else:
                                S["attach_tool"][f"{cname(src)}->{cname(dst)}"] += 1
                        elif t == OT.END:
                            S["end_turn"]["count"] += 1
                elif ctx in ("SWITCH", "TO_ACTIVE"):
                    for o in chosen:
                        if o.get("playerIndex") == me:
                            S["to_active_mine"][cname(resolve_card_id(cur, o, me))] += 1
                        else:
                            S["to_active_opp_gust"][cname(resolve_card_id(cur, o, me))] += 1
                elif ctx == "TO_HAND":
                    eff = sel.get("effect") or {}
                    ename = cname(eff.get("id")) if eff.get("id") else "?"
                    for o in chosen:
                        S[f"tohand[{ename}]"][cname(resolve_card_id(cur, o, me))] += 1
                elif ctx in ("DISCARD", "TO_DECK", "TO_DECK_BOTTOM"):
                    eff = sel.get("effect") or {}
                    ename = cname(eff.get("id")) if eff.get("id") else "?"
                    for o in chosen:
                        pi = o.get("playerIndex", me)
                        tag = "self" if pi == me else "opp"
                        S[f"discard_{tag}[{ename}]"][cname(resolve_card_id(cur, o, me))] += 1
                elif ctx in ("TO_BENCH", "TO_FIELD"):
                    for o in chosen:
                        S["to_bench"][cname(resolve_card_id(cur, o, me))] += 1

    print(f"games analyzed: {n_games}")
    for key in sorted(S):
        print(f"\n== {key} ==")
        for k, v in S[key].most_common(20):
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
