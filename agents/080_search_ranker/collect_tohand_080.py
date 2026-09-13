"""EXP-080 TO_HAND（サーチ先）二値ランカーの教師データ収集 (I-118)。

Yushin(81f1)のTO_HAND決定（ctx7、k固定・2択以上）を対象に、各候補カードについて
  - to_hand_feat_vector（盤面ゾーン特徴 + カード属性 + サーチ元）
  - ラベル y = 本人がそれを取ったか（1/0）
  - rp = ベース方策(075のRULE_C)がそれを取ったか（1/0。ベースライン一致率算出用）
を記録して pickle 保存する。1決定 × 候補カード数 = 複数行。

usage:
  uv run python agents/080_search_ranker/collect_tohand_080.py \
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
AGENT_DIR = os.path.join(REPO_ROOT, "agents/080_search_ranker")
for _p in (REPO_ROOT, AGENT_DIR, os.path.join(REPO_ROOT, "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_W: dict = {}


def _ensure_worker():
    if _W.get("loaded"):
        return
    os.environ["F080_TOHAND"] = "off"  # 収集時は学習層を無効化（rpはRULE_Cの素の選択）
    spec = importlib.util.spec_from_file_location(
        "agent080c", os.path.join(AGENT_DIR, "main.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    import feat_080
    _W["mod"] = mod
    _W["ft"] = feat_080
    _W["zips"] = {}
    _W["loaded"] = True


def _zip_for(date: str) -> zipfile.ZipFile:
    if date not in _W["zips"]:
        d = os.path.join(REPO_ROOT, "data", "episodes", date)
        zpath = next(os.path.join(d, x) for x in os.listdir(d) if x.endswith(".zip"))
        _W["zips"][date] = zipfile.ZipFile(zpath)
    return _W["zips"][date]


def _opt_card(mod, obs_cls, opt):
    """TO_HAND候補option → (カードID, from_discard)。解決不能ならNone。"""
    from cg.api import AreaType
    my = obs_cls.current.yourIndex
    me = obs_cls.current.players[my]
    try:
        if opt.area == AreaType.DECK:
            return obs_cls.select.deck[opt.index].id, 0.0
        if opt.area == AreaType.DISCARD:
            return me.discard[opt.index].id, 1.0
        if opt.area == AreaType.LOOKING:
            return obs_cls.current.looking[opt.index].id, 0.0
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
        if not sel or not obs.get("current") or sel.get("context") != 7:
            continue
        if t + 1 >= len(steps):
            continue
        actual = steps[t + 1][me].get("action")
        if not isinstance(actual, list) or not actual:
            continue
        nopt = len(sel["option"])
        k = len(actual)
        if nopt < 2 or k >= nopt:
            continue
        # 管轄はmainの_tohand_kと同一: k固定 or 「1枚まで」型（0/1）
        mn, mx = sel.get("minCount"), sel.get("maxCount")
        if not ((mn == mx == k and k >= 1) or (mn == 0 and mx == 1 and k == 1)):
            continue
        try:
            obs_cls = mod.to_observation_class(obs)
            state = obs_cls.current
            my = state.yourIndex
            # ベース方策(RULE_C)の選択: policy()のランキング上位k
            try:
                rp_set = set(mod.policy(obs_cls)[:k])
            except Exception:
                rp_set = set()
            chosen = set(actual)
            # 全候補が解決できる決定のみ採用（mainの_tohand_layerと同一条件。
            # PRIZE=裏向きサイド取り等は候補IDが無く学習不能なので管轄外）
            resolved = []
            for i in range(nopt):
                r = _opt_card(mod, obs_cls, obs_cls.select.option[i])
                if r is None:
                    resolved = None
                    break
                resolved.append((i, r))
            if resolved is None:
                continue
            for i, (cid, fd) in resolved:
                x, _keys = ft.to_hand_feat_vector(state, my, cid, fd)
                rows.append({
                    "ep": ep_id, "step": t, "k": k, "nopt": nopt,
                    "x": x, "y": 1 if i in chosen else 0,
                    "rp": 1 if i in rp_set else 0, "cid": cid, "fd": fd,
                })
        except Exception as e:  # noqa: BLE001
            errs.append(f"step{t}: {type(e).__name__}: {e}")
            continue
    return {"rows": rows, "errors": errs, "keys": ft.TOHAND_FEAT_KEYS}


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
    print(f"decisions={n_dec} rows={len(rows)} take-rate={sum(r['y'] for r in rows)/max(1,len(rows)):.3f} "
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
