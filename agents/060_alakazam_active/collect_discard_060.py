"""EXP-059 DISCARD二値分類器の教師データ収集 (I-116)。

Yushin(81f1)のDISCARD決定（ctx8、複数枚から選ぶ）を対象に、各候補カードについて
  - discard_feat_vector（盤面ゾーン特徴 + そのカードの属性）
  - ラベル = 本人がそれを捨てたか（1/0）
  - 決定ID（ep,step）と、その決定で捨てる枚数k
を記録して pickle 保存する。1決定 × 候補カード数 = 複数行。

usage:
  uv run python agents/060_alakazam_active/collect_discard_060.py \
      --dates 2026-07-09 2026-07-11 --out dev.pkl -w 4
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
AGENT_DIR = os.path.join(REPO_ROOT, "agents/060_alakazam_active")
for _p in (REPO_ROOT, AGENT_DIR, os.path.join(REPO_ROOT, "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_W: dict = {}


def _ensure_worker():
    if _W.get("loaded"):
        return
    spec = importlib.util.spec_from_file_location(
        "agent060d", os.path.join(AGENT_DIR, "main.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    import feat_060
    _W["mod"] = mod
    _W["ft"] = feat_060
    _W["zips"] = {}
    _W["loaded"] = True


def _zip_for(date: str) -> zipfile.ZipFile:
    if date not in _W["zips"]:
        d = os.path.join(REPO_ROOT, "data", "episodes", date)
        zpath = next(os.path.join(d, x) for x in os.listdir(d) if x.endswith(".zip"))
        _W["zips"][date] = zipfile.ZipFile(zpath)
    return _W["zips"][date]


def _opt_card_id(state, my_index, opt):
    """DISCARD候補option → カードID（手札área想定）。"""
    from cg.api import AreaType
    me = state.players[my_index]
    a, i = opt.area, opt.index
    try:
        if a == AreaType.HAND:
            return me.hand[i].id
        if a == AreaType.DISCARD:
            return me.discard[i].id
    except (IndexError, TypeError, AttributeError):
        return None
    return None


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
        if not sel or not obs.get("current") or sel.get("context") != 8:
            continue
        if t + 1 >= len(steps):
            continue
        actual = steps[t + 1][me].get("action")
        if not isinstance(actual, list) or not actual:
            continue
        nopt = len(sel["option"])
        k = len(actual)
        if nopt < 2 or k >= nopt:  # 全部捨て等は選択の余地なし
            continue
        try:
            obs_cls = mod.to_observation_class(obs)
            state = obs_cls.current
            my = state.yourIndex
            chosen = set(actual)
            for i in range(nopt):
                cid = _opt_card_id(state, my, obs_cls.select.option[i])
                if cid is None:
                    continue
                x, _keys = ft.discard_feat_vector(state, my, cid)
                rows.append({
                    "ep": ep_id, "step": t, "k": k, "nopt": nopt,
                    "x": x, "y": 1 if i in chosen else 0, "cid": cid,
                })
        except Exception as e:  # noqa: BLE001
            errs.append(f"step{t}: {type(e).__name__}: {e}")
            continue
    return {"rows": rows, "errors": errs, "keys": ft.DISCARD_FEAT_KEYS}


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
    with open(args.out, "wb") as f:
        pickle.dump({"feat_keys": keys, "rows": rows}, f, protocol=4)
    print(f"decisions={n_dec} rows={len(rows)} discard-rate={sum(r['y'] for r in rows)/max(1,len(rows)):.3f} "
          f"errors={len(errs)} -> {args.out}")
    for e in errs[:3]:
        print("  ERROR", e)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dates", nargs="+", required=True)
    ap.add_argument("--team", default="Yushin Ito")
    ap.add_argument("--sig", default="81f1758c92")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("-w", "--workers", type=int, default=2)
    cmd_collect(ap.parse_args())


if __name__ == "__main__":
    main()
