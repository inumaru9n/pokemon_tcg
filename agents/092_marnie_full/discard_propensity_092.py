"""EXP-092: Xerosic/Hand Trimmer被弾時のLucaの捨て札傾向スコア。
各DISCARD選択で「手札にあった×捨てられた」を数え、捨て率=discarded/offeredを出す。
usage: uv run python agents/092_marnie_full/discard_propensity_092.py
"""
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
CTX = {c.value: c.name for c in api.SelectContext}
AREA = api.AreaType


def cname(cid):
    c = CARDS.get(cid)
    return c.name if c else f"card{cid}"


offered = Counter()
discarded = Counter()
pair_kept = Counter()  # (kept_card) when hand also had (discarded card)

for day in ["2026-07-16", "2026-07-17", "2026-07-18", "2026-07-22"]:
    rows_by_ep = defaultdict(dict)
    for r in csv.DictReader(open(f"{ROOT}/data/episodes/{day}/summary.csv")):
        rows_by_ep[r["episode_id"]][int(r["player"])] = r
    z = zipfile.ZipFile(f"{ROOT}/data/episodes/{day}/pokemon-tcg-ai-battle-episodes-{day}.zip")
    names = set(z.namelist())
    for ep, players in sorted(rows_by_ep.items()):
        tgt = [pi for pi, r in players.items()
               if r["deck_signature"] == SIG and r["team"] == TEAM]
        if not tgt or f"{ep}.json" not in names:
            continue
        steps = json.load(z.open(f"{ep}.json"))["steps"]
        for me in tgt:
            for i, st in enumerate(steps):
                p = st[me]
                obs = p.get("observation") or {}
                cur, sel = obs.get("current"), obs.get("select")
                if p.get("status") != "ACTIVE" or cur is None or sel is None:
                    continue
                if CTX.get(sel.get("context")) != "DISCARD":
                    continue
                eff = (sel.get("effect") or {}).get("id")
                if eff not in (1197, 1087):  # Xerosic / Hand Trimmer
                    continue
                act = steps[i + 1][me].get("action") if i + 1 < len(steps) else None
                if not isinstance(act, list):
                    continue
                opts = sel.get("option") or []
                mypl = cur["players"][me]
                ids = []
                for j, o in enumerate(opts):
                    if o.get("area") == AREA.HAND and o.get("playerIndex", me) == me:
                        try:
                            cid = mypl["hand"][o["index"]]["id"]
                        except (IndexError, TypeError, KeyError):
                            cid = None
                        ids.append((j, cid))
                chosen = set(a for a in act if isinstance(a, int))
                for j, cid in ids:
                    if cid is None:
                        continue
                    offered[cid] += 1
                    if j in chosen:
                        discarded[cid] += 1

print(f"{'card':34s} {'offered':>8s} {'discarded':>9s} {'rate':>6s}")
for cid, n in sorted(offered.items(), key=lambda kv: -(discarded[kv[0]] / kv[1])):
    print(f"{cname(cid):34s} {n:8d} {discarded[cid]:9d} {discarded[cid]/n:6.1%}")
