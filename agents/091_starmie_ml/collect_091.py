"""EXP-091 Starmie(491b8bfb26)ダイレクト方策学習の教師データ収集 (I-136)。

教師=Yushin Ito（491bレシピの本家。他チームは晩期の少数コピー勢で除外）の
全MAIN決定（ctx0・min=max=1・選択肢2以上）について
  - feat_091.featurize のゾーン特徴（~115次元）
  - 選択可能な行動クラスのマスク / 本人が選んだクラス
  - 036ルール方策（lethal探索なしのStarmiePolicy素身）の予測クラスと
    option一致（比較基準）
を記録して pickle 保存する。列挙なし=軽量。

usage:
  uv run python agents/091_starmie_ml/collect_091.py \
      --dates 2026-07-03 2026-07-05 2026-07-06 2026-07-08 \
      --out agents/091_starmie_ml/dev_091.pkl -w 4
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
AGENT_DIR = os.path.join(REPO_ROOT, "agents/091_starmie_ml")
for _p in (REPO_ROOT, AGENT_DIR, os.path.join(REPO_ROOT, "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_W: dict = {}


def _ensure_worker():
    if _W.get("loaded"):
        return
    os.environ["F091_ML"] = "off"  # 収集時はML層を無効化
    for p in (REPO_ROOT, AGENT_DIR, os.path.join(REPO_ROOT, "tools")):
        if p not in sys.path:
            sys.path.insert(0, p)
    spec = importlib.util.spec_from_file_location(
        "agent091c", os.path.join(AGENT_DIR, "main.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    import feat_091
    _W["mod"] = mod
    _W["ft"] = feat_091
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
        if not sel or not obs.get("current") or sel.get("context") != 0:
            continue
        if t + 1 >= len(steps):
            continue
        # steps[t+1][me]['action'] は steps[t][me]['observation'] への応答（EXP-086）
        actual = steps[t + 1][me].get("action")
        if not isinstance(actual, list) or not actual:
            continue
        nopt = len(sel["option"])
        if sel.get("minCount", 1) != 1 or sel.get("maxCount", 1) != 1 or nopt < 2:
            continue
        a0 = actual[0]
        if a0 >= nopt:
            continue
        cur = obs["current"]
        turn = cur.get("turn", 0)

        try:
            obs_cls = mod.to_observation_class(obs)
            state = obs_cls.current
            my = state.yourIndex
            feats, _keys = ft.feat_vector(state, my)
            cls_of = [ft.option_class(state, obs_cls.select, i, my)
                      for i in range(nopt)]
            # ベース=036ルール（lethal探索なしの素身policy）
            pred_base = mod.StarmiePolicy(obs_cls).choose()
        except Exception as e:  # noqa: BLE001
            errs.append(f"step{t}: {type(e).__name__}: {e}")
            continue
        pb = pred_base[0] if pred_base else -1
        avail = [0] * len(ft.CLASSES)
        for c in cls_of:
            avail[ft.CLASS_ID[c]] = 1
        rows.append({
            "ep": ep_id, "date": date, "step": t, "turn": turn,
            "x": feats,
            "avail": avail,
            "y": ft.CLASS_ID[cls_of[a0]],
            "pb_cls": ft.CLASS_ID[cls_of[pb]] if 0 <= pb < nopt else -1,
            "match_base": bool(pb == a0),
            "deck": state.players[my].deckCount,
            "result": meta.get("result", "?"),
            "opp_arch": meta.get("opp_arch", "?"),
        })
    return {"rows": rows, "errors": errs, "keys": ft.FEAT_KEYS}


def cmd_collect(args):
    from agreement import find_episodes  # tools/agreement.py

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
    _ensure_worker()
    rows = [r for res in results for r in res["rows"]]
    errs = [e for res in results for e in res["errors"]]
    keys = next((res.get("keys") for res in results if res.get("keys")), None)
    data = {"feat_keys": keys, "classes": _W["ft"].CLASSES, "rows": rows}
    with open(args.out, "wb") as f:
        pickle.dump(data, f, protocol=4)
    base_acc = sum(r["match_base"] for r in rows) / max(1, len(rows))
    print(f"rows={len(rows)} base(option)acc={base_acc:.4f} "
          f"errors={len(errs)} -> {args.out}")
    cls = _W["ft"].CLASSES
    cnt = Counter(cls[r["y"]] for r in rows)
    for c, n in cnt.most_common():
        print(f"  {c:18s} {n:6d} {n/len(rows):.4f}")
    for e in errs[:3]:
        print("  ERROR", e)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dates", nargs="+", required=True)
    ap.add_argument("--team", default="Yushin Ito")
    ap.add_argument("--sig", default="491b8bfb26")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("-w", "--workers", type=int, default=2)
    cmd_collect(ap.parse_args())


if __name__ == "__main__":
    main()
