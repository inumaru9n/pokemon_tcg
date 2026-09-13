"""行動一致率測定CLI（Marnie/082系用の改作、EXP-092）。

tools/agreement.py（045系用）からの変更点:
- 教師強制: 082系は `pre_turn != turn` で plan をリセットする。各ターンの最初の決定の前に
  mod.pre_turn = turn-1 を入れて自然リセットを発火させる（agreement.pyはpre_turn=turnを
  常時セットするためplanリセットが再現されない）
- _lethal_search のシグネチャが (obs, heuristic_scores) の2引数（090/092系）に対応
- Dudunsparce/Fezandipitiのabilityトラッカーは無し（hasattrガードのため実質無害だが削除）

usage:
  F092_ML=off uv run tools/agreement_marnie.py agents/092_marnie_full \
      --team Luca --sig 1ec0f47981 --dates 2026-07-22 [-w 2] [--limit N] [--records out.jsonl]
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import random
import sys
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

# OptionType values (cg/api.py)
OT_NUMBER, OT_YES, OT_NO, OT_CARD, OT_TOOL, OT_ENERGY_CARD, OT_ENERGY = 0, 1, 2, 3, 4, 5, 6
OT_PLAY, OT_ATTACH, OT_EVOLVE, OT_ABILITY, OT_DISCARD, OT_RETREAT, OT_ATTACK, OT_END = (
    7, 8, 9, 10, 11, 12, 13, 14)
# AreaType values
AREA_DECK, AREA_HAND, AREA_DISCARD, AREA_ACTIVE, AREA_BENCH = 1, 2, 3, 4, 5
AREA_PRIZE, AREA_STADIUM, AREA_LOOKING = 6, 7, 12

CTX_MAIN = 0

# ---------------------------------------------------------------------------
_W: dict = {}


def _load_agent(agent_dir: str):
    agent_dir = os.path.abspath(agent_dir)
    main_path = os.path.join(agent_dir, "main.py")
    for p in (REPO_ROOT, agent_dir):
        if p not in sys.path:
            sys.path.insert(0, p)
    spec = importlib.util.spec_from_file_location("agreement_agent", main_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _ensure_worker(agent_dir: str, no_search: bool):
    if _W.get("loaded"):
        return
    from cg.api import all_card_data, all_attack, SelectContext
    mod = _load_agent(agent_dir)
    _W["mod"] = mod
    _W["table"] = {c.cardId: c for c in all_card_data()}
    _W["attacks"] = {a.attackId: a.name for a in all_attack()}
    _W["ctx_names"] = {int(v): v.name for v in SelectContext}
    _W["zips"] = {}
    _W["search_flag"] = {"used": False}

    if hasattr(mod, "_lethal_search"):
        orig = mod._lethal_search
        if no_search:
            mod._lethal_search = lambda *a, **k: None
        else:
            flag = _W["search_flag"]

            def wrapped(*a, _orig=orig, _flag=flag, **k):
                r = _orig(*a, **k)
                if r is not None:
                    _flag["used"] = True
                return r

            mod._lethal_search = wrapped
    _W["loaded"] = True


def _zip_for(date: str) -> zipfile.ZipFile:
    if date not in _W["zips"]:
        d = os.path.join(REPO_ROOT, "data", "episodes", date)
        zpath = next(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".zip"))
        _W["zips"][date] = zipfile.ZipFile(zpath)
    return _W["zips"][date]


# ---------------------------------------------------------------------------
# Option description helpers (raw dict based)

def _pokemon_at(cur: dict, area: int, index: int, player: int) -> dict | None:
    ps = cur["players"][player]
    try:
        if area == AREA_ACTIVE:
            return ps["active"][index]
        if area == AREA_BENCH:
            return ps["bench"][index]
    except (IndexError, TypeError):
        return None
    return None


def _card_at(obs: dict, area: int, index: int, player: int) -> dict | None:
    cur = obs["current"]
    ps = cur["players"][player]
    try:
        if area == AREA_DECK:
            return (obs["select"].get("deck") or [None] * (index + 1))[index]
        if area == AREA_HAND:
            return (ps.get("hand") or [None] * (index + 1))[index]
        if area == AREA_DISCARD:
            return ps["discard"][index]
        if area in (AREA_ACTIVE, AREA_BENCH):
            return _pokemon_at(cur, area, index, player)
        if area == AREA_PRIZE:
            return ps["prize"][index]
        if area == AREA_STADIUM:
            return cur["stadium"][index]
        if area == AREA_LOOKING:
            return cur["looking"][index]
    except (IndexError, TypeError):
        return None
    return None


def _name(cid: int | None) -> str:
    if cid is None:
        return "?"
    c = _W["table"].get(cid)
    return c.name if c else f"id{cid}"


def _opt_desc(obs: dict, opt: dict) -> str:
    cur = obs["current"]
    my = cur["yourIndex"]
    t = opt["type"]
    if t == OT_NUMBER:
        return f"NUM:{opt.get('number')}"
    if t == OT_YES:
        return "YES"
    if t == OT_NO:
        return "NO"
    if t == OT_END:
        return "END"
    if t == OT_RETREAT:
        return "RETREAT"
    if t == OT_ATTACK:
        return f"ATK:{_W['attacks'].get(opt.get('attackId'), opt.get('attackId'))}"
    if t == OT_PLAY:
        card = _card_at(obs, AREA_HAND, opt.get("index", -1), my)
        return f"PLAY:{_name(card and card.get('id'))}"
    if t == OT_ABILITY:
        card = _card_at(obs, opt.get("area", -1), opt.get("index", -1), my)
        return f"ABILITY:{_name(card and card.get('id'))}"
    if t == OT_EVOLVE:
        card = _card_at(obs, opt.get("area", -1), opt.get("index", -1), my)
        pk = _pokemon_at(cur, opt.get("inPlayArea", -1), opt.get("inPlayIndex", -1), my)
        return f"EVOLVE:{_name(card and card.get('id'))}<-{_name(pk and pk.get('id'))}"
    if t == OT_ATTACH:
        card = _card_at(obs, opt.get("area", AREA_HAND), opt.get("index", -1), my)
        pk = _pokemon_at(cur, opt.get("inPlayArea", -1), opt.get("inPlayIndex", -1), my)
        loc = "act" if opt.get("inPlayArea") == AREA_ACTIVE else "bench"
        return f"ATTACH:{_name(card and card.get('id'))}->{_name(pk and pk.get('id'))}({loc})"
    if t == OT_CARD:
        pl = opt.get("playerIndex", my)
        pl = my if pl is None else pl
        card = _card_at(obs, opt.get("area", -1), opt.get("index", -1), pl)
        side = "my" if pl == my else "op"
        area_n = {AREA_DECK: "deck", AREA_HAND: "hand", AREA_DISCARD: "dis",
                  AREA_ACTIVE: "act", AREA_BENCH: "bench", AREA_PRIZE: "prz",
                  AREA_STADIUM: "stad", AREA_LOOKING: "look"}.get(opt.get("area"), "?")
        # ATTACH_FROM等でPokemonが返る場合はhpでなくidで名前化
        cid = card and card.get("id")
        return f"CARD:{_name(cid)}@{side}.{area_n}"
    if t in (OT_ENERGY, OT_ENERGY_CARD):
        pl = opt.get("playerIndex", my)
        pl = my if pl is None else pl
        pk = _pokemon_at(cur, opt.get("area", -1), opt.get("index", -1), pl)
        side = "my" if pl == my else "op"
        return f"ENERGY@{side}.{_name(pk and pk.get('id'))}"
    return f"T{t}"


def _sel_desc(obs: dict, sel_idx: list[int]) -> str:
    opts = obs["select"]["option"]
    descs = [_opt_desc(obs, opts[i]) for i in sel_idx if 0 <= i < len(opts)]
    return " + ".join(sorted(descs)) if descs else "(none)"


def _dtype(obs: dict, actual: list[int]) -> str:
    sel = obs["select"]
    ctx = sel["context"]
    if ctx != CTX_MAIN:
        base = _W["ctx_names"].get(ctx, f"CTX{ctx}").lower()
        if base in ("to_hand", "discard", "to_bench", "to_field", "to_deck"):
            eff = (sel.get("effect") or {}).get("id")
            if eff:
                return f"{base}[{_name(eff)}]"
        return base
    if not actual:
        return "main_?"
    t = sel["option"][actual[0]]["type"]
    if t == OT_ATTACK:
        return "main_attack"
    if t == OT_PLAY:
        my = obs["current"]["yourIndex"]
        card = _card_at(obs, AREA_HAND, sel["option"][actual[0]].get("index", -1), my)
        ct = _W["table"].get(card["id"]).cardType if card else None
        return {0: "main_play_pokemon", 1: "main_play_item", 2: "main_play_tool",
                3: "main_play_supporter", 4: "main_play_stadium"}.get(
                    int(ct) if ct is not None else -1, "main_play_other")
    return {OT_ATTACH: "main_attach_energy", OT_EVOLVE: "main_evolve",
            OT_ABILITY: "main_ability", OT_RETREAT: "main_retreat",
            OT_END: "main_end_turn"}.get(t, f"main_t{t}")


# ---------------------------------------------------------------------------

def process_episode(task: tuple) -> list[dict]:
    (agent_dir, no_search, date, ep_id, me, meta) = task
    _ensure_worker(agent_dir, no_search)
    mod = _W["mod"]
    steps = json.loads(_zip_for(date).read(f"{ep_id}.json"))["steps"]

    # per-episode agent state reset
    if hasattr(mod, "_search_time_used"):
        mod._search_time_used = 0.0
    trk_turn = -1

    records = []
    for t, st in enumerate(steps):
        s = st[me]
        if s.get("status") != "ACTIVE":
            continue
        obs = s.get("observation") or {}
        sel = obs.get("select")
        if not sel or not obs.get("current"):
            continue
        if t + 1 >= len(steps):
            continue
        actual = steps[t + 1][me].get("action")
        if not isinstance(actual, list):
            continue
        if not actual and sel.get("minCount", 1) > 0:
            continue
        cur = obs["current"]
        turn = cur.get("turn", 0)
        if turn != trk_turn:
            trk_turn = turn
            # 082系の教師強制: 新ターンの最初の決定でplanの自然リセットを発火させる
            if hasattr(mod, "pre_turn"):
                mod.pre_turn = turn - 1
        _W["search_flag"]["used"] = False

        random.seed((int(ep_id) * 1000003 + t) & 0x7FFFFFFF)
        try:
            pred = mod.agent(obs)
        except Exception as e:  # noqa: BLE001
            records.append({"ep": ep_id, "date": date, "step": t, "error": f"{type(e).__name__}: {e}"})
            continue

        maxc = sel.get("maxCount", 1)
        match = sorted(pred[:maxc]) == sorted(actual)
        if not match:
            # 等価一致: 同名カード（山の添字違い）・裏向き等、記述の多重集合が同じなら一致
            match = _sel_desc(obs, pred[:maxc]) == _sel_desc(obs, actual)

        opts = sel["option"]
        nopt = len(opts)
        minc = sel.get("minCount", 1)
        forced = (nopt == 1 and minc >= 1) or (minc == nopt)
        dtype = _dtype(obs, actual)
        my = cur["yourIndex"]

        rec = {
            "ep": ep_id, "date": date, "step": t, "turn": turn,
            "ctx": sel["context"], "dtype": dtype,
            "nopt": nopt, "minc": minc, "maxc": maxc, "forced": forced,
            "pred": pred[:maxc], "actual": actual, "match": match,
            "search_used": _W["search_flag"]["used"],
            "prz_me": len(cur["players"][my].get("prize") or []),
            "prz_op": len(cur["players"][1 - my].get("prize") or []),
            "hand": len(cur["players"][my].get("hand") or []),
            "deck": cur["players"][my].get("deckCount"),
            "op_hand": cur["players"][1 - my].get("handCount"),
            "pred_desc": _sel_desc(obs, pred[:maxc]),
            "actual_desc": _sel_desc(obs, actual),
            **meta,
        }
        if not match:
            rec["flag_search"] = rec["search_used"]
        records.append(rec)
    return records


# ---------------------------------------------------------------------------

def find_episodes(dates: list[str], team: str, sig: str) -> list[tuple]:
    out = []
    for date in dates:
        path = os.path.join(REPO_ROOT, "data", "episodes", date, "summary.csv")
        rows_by_ep = defaultdict(dict)
        with open(path) as f:
            for row in csv.DictReader(f):
                rows_by_ep[row["episode_id"]][int(row["player"])] = row
        for ep, players in rows_by_ep.items():
            for pi, row in players.items():
                if row["team"] == team and row["deck_signature"] == sig:
                    opp = players.get(1 - pi, {})
                    meta = {
                        "me": pi,
                        "result": {"1": "win", "-1": "loss"}.get(row["reward"], "draw"),
                        "opp_arch": opp.get("archetype", "?"),
                        "opp_sig": opp.get("deck_signature", "?"),
                    }
                    out.append((date, ep, pi, meta))
                    break
    return sorted(out, key=lambda x: int(x[1]))


def wilson(k: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    c = (p + z * z / (2 * n)) / denom
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, c - m), min(1.0, c + m))


def rate(recs: list[dict]) -> str:
    n = len(recs)
    k = sum(r["match"] for r in recs)
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {k/n:.1%} [{lo:.1%}-{hi:.1%}]" if n else "n=0"


def summarize(records: list[dict], label: str, top_clusters: int = 14) -> None:
    errs = [r for r in records if "error" in r]
    recs = [r for r in records if "error" not in r]
    print(f"\n{'='*70}\n== {label}: {len(recs)} decisions, {len(errs)} agent errors")
    if errs:
        for e in errs[:5]:
            print(f"   ERROR ep={e['ep']} step={e['step']}: {e['error']}")
    if not recs:
        return
    nonforced = [r for r in recs if not r["forced"]]
    print(f"全体一致率        : {rate(recs)}")
    print(f"非強制のみ        : {rate(nonforced)}  (forced={len(recs)-len(nonforced)})")

    print("\n-- 種別ごと（非強制） --")
    by = defaultdict(list)
    for r in nonforced:
        by[r["dtype"]].append(r)
    for dt, rs in sorted(by.items(), key=lambda kv: -len(kv[1])):
        print(f"  {dt:34s} {rate(rs)}")

    print("\n-- 対面アーキタイプごと（非強制） --")
    by = defaultdict(list)
    for r in nonforced:
        by[r["opp_arch"]].append(r)
    for a, rs in sorted(by.items(), key=lambda kv: -len(kv[1])):
        print(f"  {a:36s} {rate(rs)}")

    print(f"\n-- 不一致クラスタ上位{top_clusters}（非強制） --")
    mism = [r for r in nonforced if not r["match"]]
    clusters = defaultdict(list)
    for r in mism:
        clusters[(r["dtype"], r["actual_desc"], r["pred_desc"])].append(r)
    for (dt, ad, pd), rs in sorted(clusters.items(), key=lambda kv: -len(kv[1]))[:top_clusters]:
        ex = rs[0]
        print(f"  x{len(rs):4d} [{dt}] 本人={ad} | 予測={pd}"
              f"  (例: ep{ex['ep']} step{ex['step']} turn{ex['turn']} vs {ex['opp_arch']})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("agent_dir")
    ap.add_argument("--team", required=True)
    ap.add_argument("--sig", required=True)
    ap.add_argument("--dates", nargs="+", required=True)
    ap.add_argument("--limit", type=int, default=None, help="episodes per date")
    ap.add_argument("--no-search", action="store_true")
    ap.add_argument("-w", "--workers", type=int, default=2)
    ap.add_argument("--records", default=None, help="write per-decision JSONL here")
    args = ap.parse_args()

    eps = find_episodes(args.dates, args.team, args.sig)
    if args.limit:
        per = defaultdict(int)
        eps = [e for e in eps if (per.__setitem__(e[0], per[e[0]] + 1) or per[e[0]] <= args.limit)]
    print(f"{len(eps)} episodes ({Counter(e[0] for e in eps)})")

    tasks = [(os.path.abspath(args.agent_dir), args.no_search, date, ep, me, meta)
             for (date, ep, me, meta) in eps]
    if args.workers <= 1:
        all_recs = [process_episode(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            all_recs = list(ex.map(process_episode, tasks, chunksize=4))
    records = [r for rs in all_recs for r in rs]

    if args.records:
        os.makedirs(os.path.dirname(os.path.abspath(args.records)), exist_ok=True)
        with open(args.records, "w") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"records -> {args.records}")

    for date in args.dates:
        summarize([r for r in records if r.get("date") == date], f"date {date}")
    if len(args.dates) > 1:
        summarize(records, "ALL")


if __name__ == "__main__":
    main()
