"""EXP-061 universal imitation の教師データ収集 (I-NEW / Majkel切替)。

team=Majkel1337 sig=156952a871 の**全コンテキスト**の決定を対象に、
1決定 × 各選択肢 = 1行として (universal特徴, ラベル=本人が選んだか) を収集する。
固定クラス無し・全選択肢を採点する universal ranker 用。

usage:
  uv run python agents/061_majkel_imitate/collect_universal_061.py \
      --dates 2026-07-10 2026-07-11 2026-07-12 2026-07-13 --out dev.pkl -w 4
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pickle
import sys
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

REPO_ROOT = "/Users/akira/kaggle/pokemon_tcg"
AGENT_DIR = os.path.join(REPO_ROOT, "agents/061_majkel_imitate")
for _p in (REPO_ROOT, AGENT_DIR, os.path.join(REPO_ROOT, "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_W: dict = {}


def _ensure_worker():
    if _W.get("loaded"):
        return
    spec = importlib.util.spec_from_file_location(
        "agent061u", os.path.join(AGENT_DIR, "main.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    import feat_061
    _W["mod"] = mod
    _W["ft"] = feat_061
    _W["zips"] = {}
    _W["loaded"] = True


def _zip_for(date: str) -> zipfile.ZipFile:
    if date not in _W["zips"]:
        d = os.path.join(REPO_ROOT, "data", "episodes", date)
        zpath = next(os.path.join(d, x) for x in os.listdir(d) if x.endswith(".zip"))
        _W["zips"][date] = zipfile.ZipFile(zpath)
    return _W["zips"][date]


def process_episode(task: tuple) -> dict:
    (date, ep_id, me, meta) = task
    _ensure_worker()
    mod, ft = _W["mod"], _W["ft"]
    try:
        steps = json.loads(_zip_for(date).read(f"{ep_id}.json"))["steps"]
    except KeyError:
        return {"rows": [], "errors": []}

    rows, errs = [], []
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
        if not isinstance(actual, list) or not actual:
            continue
        nopt = len(sel["option"])
        minc = sel.get("minCount", 1)
        # 選ぶ余地のある決定のみ（全選択強制などは学習に無意味）
        if nopt < 2 or minc >= nopt:
            continue
        a0max = max(actual)
        if a0max >= nopt:
            continue
        ctx = sel["context"]
        try:
            oc = mod.to_observation_class(obs)
            state = oc.current
            my = state.yourIndex
            chosen = set(actual)
            for i in range(nopt):
                x, _keys = ft.universal_row(state, oc.select, i, my)
                rows.append({
                    "ep": ep_id, "step": t, "ctx": ctx, "turn": state.turn,
                    "nopt": nopt, "minc": minc, "x": x,
                    "y": 1 if i in chosen else 0,
                })
        except Exception as e:  # noqa: BLE001
            errs.append(f"ctx{ctx} step{t}: {type(e).__name__}: {e}")
            continue
    return {"rows": rows, "errors": errs, "keys": ft.UNIVERSAL_FEAT_KEYS}


def cmd_collect(args):
    from agreement import find_episodes

    eps = find_episodes(args.dates, args.team, args.sig)
    if args.limit:
        per: dict = defaultdict(int)
        kept = []
        for e in eps:
            per[e[0]] += 1
            if per[e[0]] <= args.limit:
                kept.append(e)
        eps = kept
    print(f"{len(eps)} episodes ({Counter(e[0] for e in eps)})")
    if args.workers <= 1:
        results = [process_episode(e) for e in eps]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            results = list(ex.map(process_episode, eps, chunksize=4))
    rows = [r for res in results for r in res["rows"]]
    errs = [e for res in results for e in res["errors"]]
    keys = next((res.get("keys") for res in results if res.get("keys")), None)
    n_dec = len({(r["ep"], r["step"]) for r in rows})
    by_ctx = Counter(r["ctx"] for r in rows)
    with open(args.out, "wb") as f:
        pickle.dump({"feat_keys": keys, "rows": rows}, f, protocol=4)
    print(f"decisions={n_dec} rows={len(rows)} feat={len(keys) if keys else 0} "
          f"errors={len(errs)} -> {args.out}")
    print(f"  by_ctx(rows): {dict(sorted(by_ctx.items(), key=lambda kv: -kv[1]))}")
    for e in errs[:5]:
        print("  ERROR", e)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dates", nargs="+", required=True)
    ap.add_argument("--team", default="Majkel1337")
    ap.add_argument("--sig", default="156952a871")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("-w", "--workers", type=int, default=2)
    cmd_collect(ap.parse_args())


if __name__ == "__main__":
    main()
