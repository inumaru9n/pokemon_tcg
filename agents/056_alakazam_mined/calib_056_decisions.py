"""EXP-055 一致率評価用の決定データ収集（054較正基盤の055版）。

collect: 本人(Yushin 81f1)の記録局面を再生し、決定ごとに
  - pred_base: 054のLINES無効時の予測（=052+lethal と同一挙動）
  - LINES較正データ: 全終端の (feature int8ベクトル, マッピング済みpred option idx)
  を収集して pickle 保存する。ドロー等の教師強制は tools/agreement.py と同一。

fit: collectデータ上で重みwの座標降下。objective = 非強制の全体一致率
  （weight依存の決定は argmax(F@w) → pred、weight非依存は match_base 固定）。
  fire精度（pred(w) != pred_base の決定での本人一致）も併記。

usage:
  uv run python agents/056_alakazam_mined/calib_056_decisions.py collect --dates 2026-07-09 \
      --out /tmp/dev.pkl [-w 2] [--limit 30]
  uv run python agents/056_alakazam_mined/calib_056_decisions.py fit --dev /tmp/dev.pkl \
      [--val /tmp/val.pkl] [--passes 3]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pickle
import random
import sys
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np

REPO_ROOT = "/Users/akira/kaggle/pokemon_tcg"
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "tools"))

AGENT_DIR = os.path.join(REPO_ROOT, "agents/056_alakazam_mined")

# ---------------------------------------------------------------------------
_W: dict = {}


def _ensure_worker():
    if _W.get("loaded"):
        return
    main_path = os.path.join(AGENT_DIR, "main.py")
    for p in (REPO_ROOT, AGENT_DIR):
        if p not in sys.path:
            sys.path.insert(0, p)
    spec = importlib.util.spec_from_file_location("calib056_agent", main_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    _W["mod"] = mod
    _W["zips"] = {}
    _W["feat_keys"] = sorted(k for k in mod._W054 if "@" not in k)
    # lethal探索の発火フラグ
    flag = {"used": False}
    _W["search_flag"] = flag
    orig = mod._lethal_search

    def wrapped(obs, _orig=orig, _flag=flag):
        r = _orig(obs)
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


def process_episode(task: tuple) -> list[dict]:
    (date, ep_id, me, meta) = task
    _ensure_worker()
    mod = _W["mod"]
    feat_keys = _W["feat_keys"]
    nk = len(feat_keys)
    steps = json.loads(_zip_for(date).read(f"{ep_id}.json"))["steps"]

    mod._search_time_used = 0.0
    trk_turn, trk_dud, trk_fez = 0, False, False

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
            trk_turn, trk_dud, trk_fez = turn, False, False

        def sync():
            mod.pre_turn = trk_turn
            mod.ability_used_dudunsparce = trk_dud
            mod.ability_used_fezandipiti = trk_fez
            mod._lines_boss_bench = None

        nopt = len(sel["option"])
        minc = sel.get("minCount", 1)
        maxc = sel.get("maxCount", 1)
        forced = (nopt == 1 and minc >= 1) or (minc == nopt)

        # pred_base（LINES無効=052+lethal同一）
        sync()
        _W["search_flag"]["used"] = False
        mod.F054_LINES = False
        random.seed((int(ep_id) * 1000003 + t) & 0x7FFFFFFF)
        err = None
        try:
            pred_base = mod.agent(obs)
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
            pred_base = []
        finally:
            mod.F054_LINES = True
        lethal_fired = _W["search_flag"]["used"]
        match_base = sorted(pred_base[:maxc]) == sorted(actual)

        my = cur["yourIndex"]
        ps_me = cur["players"][my]
        ps_op = cur["players"][1 - my]
        dtype = f"ctx{sel['context']}"
        if sel["context"] == 0 and actual:
            a_opt = sel["option"][actual[0]]
            at = a_opt["type"]
            dtype = {13: "main_attack", 8: "main_attach", 9: "main_evolve",
                     10: "main_ability", 12: "main_retreat", 14: "main_end",
                     11: "main_discard"}.get(at, f"main_t{at}")
            if at == 7:
                try:
                    cid = ps_me["hand"][a_opt.get("index", -1)]["id"]
                    ct = int(mod.card_table[cid].cardType)
                    dtype = {0: "main_play_pokemon", 1: "main_play_item",
                             3: "main_play_supporter", 4: "main_play_stadium"}.get(
                                 ct, "main_play_other")
                except (IndexError, TypeError, KeyError):
                    dtype = "main_play_?"
        rec = {
            "ep": ep_id, "date": date, "step": t, "turn": turn,
            "ctx": sel["context"], "nopt": nopt, "minc": minc, "maxc": maxc,
            "forced": forced, "actual": actual, "dtype": dtype,
            "deck": ps_me.get("deckCount"),
            "prz_me": len(ps_me.get("prize") or []),
            "prz_op": len(ps_op.get("prize") or []),
            "pb": pred_base[0] if pred_base else -1,
            "match_base": bool(match_base), "fixed": True,
            "opp_arch": meta.get("opp_arch", "?"),
        }
        if err:
            rec["error"] = err

        qualifying = (sel["context"] == 0 and minc == 1 and maxc == 1
                      and turn >= 2 and nopt >= 2 and not lethal_fired and not err)
        if qualifying:
            sync()
            random.seed((int(ep_id) * 1000003 + t) & 0x7FFFFFFF)
            terms = []
            try:
                obs_cls = mod.to_observation_class(obs)
                mod._lines_layer(obs_cls, _collect=terms)
            except Exception as e:  # noqa: BLE001
                rec["error"] = f"lines: {type(e).__name__}: {e}"
                terms = []
            if terms:
                a0 = actual[0] if actual else -1
                dedup = {}
                for feat, pred, mask in terms:
                    key = (tuple(sorted(feat.items())), pred, mask)
                    if key not in dedup:
                        dedup[key] = (feat, pred, mask)
                F = np.zeros((len(dedup), nk), dtype=np.int16)
                P = np.zeros(len(dedup), dtype=np.int16)
                A = np.zeros(len(dedup), dtype=bool)
                M = np.zeros(len(dedup), dtype=np.uint64)
                ki = {k: i for i, k in enumerate(feat_keys)}
                for r_i, (feat, pred, mask) in enumerate(dedup.values()):
                    for k, v in feat.items():
                        F[r_i, ki[k]] = v
                    P[r_i] = pred
                    A[r_i] = bool(a0 >= 0 and (mask >> a0) & 1)
                    M[r_i] = mask if (nopt <= 63 and mask < (1 << 63)) else 0
                rec["fixed"] = False
                rec["F"] = F
                rec["P"] = P
                rec["A"] = A
                rec["M"] = M
                dbg = getattr(mod, "_lines_debug", None) or {}
                rec["ops"] = np.array(dbg.get("pscores", []), dtype=np.float32)
                rec["ocls"] = dbg.get("opt_cls", [])
        records.append(rec)

        # 教師強制トラッカー更新（実選択ベース）: MAINのABILITYのみ
        if sel["context"] == 0 and actual:
            a_opt = sel["option"][actual[0]]
            if a_opt["type"] == 10:  # OT_ABILITY
                area, idx = a_opt.get("area", -1), a_opt.get("index", -1)
                ps = cur["players"][cur["yourIndex"]]
                card = None
                try:
                    if area == 4:
                        card = ps["active"][idx]
                    elif area == 5:
                        card = ps["bench"][idx]
                except (IndexError, TypeError):
                    card = None
                if card is not None:
                    if card["id"] == 66:
                        trk_dud = True
                    elif card["id"] == 140:
                        trk_fez = True
    return records


def cmd_collect(args):
    from agreement import find_episodes  # tools/agreement.py

    eps = find_episodes(args.dates, args.team, args.sig)
    if args.limit:
        per = defaultdict(int)
        kept = []
        for e in eps:
            per[e[0]] += 1
            if per[e[0]] <= args.limit:
                kept.append(e)
        eps = kept
    print(f"{len(eps)} episodes ({Counter(e[0] for e in eps)})")
    tasks = [(date, ep, me, meta) for (date, ep, me, meta) in eps]
    if args.workers <= 1:
        all_recs = [process_episode(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            all_recs = list(ex.map(process_episode, tasks, chunksize=2))
    records = [r for rs in all_recs for r in rs]
    _ensure_worker()
    data = {"feat_keys": _W["feat_keys"], "records": records,
            "defaults": dict(_W["mod"]._W054)}
    with open(args.out, "wb") as f:
        pickle.dump(data, f, protocol=4)
    nerr = sum(1 for r in records if "error" in r)
    ndep = sum(1 for r in records if not r["fixed"])
    print(f"records={len(records)} weight-dependent={ndep} errors={nerr} -> {args.out}")
    for r in records:
        if "error" in r:
            print("  ERROR", r["ep"], r["step"], r["error"])
            break


# ---------------------------------------------------------------------------

PHASE_KEYS = ["hand", "burn", "chip", "attack", "prize", "xero", "next_ready",
              "v_dudunsparce", "v_abra", "v_dunsparce", "v_kadabra", "b_alakazam"]
RACE_KEYS = ["prize", "chip", "attack", "hand", "burn"]


def _ext_keys(keys):
    ext = []
    for k in PHASE_KEYS:
        if k in keys:
            ext += [f"{k}@early", f"{k}@late"]
    for k in RACE_KEYS:
        if k in keys:
            ext.append(f"{k}@race")
    return ext


def _prep(data):
    """fit用に前処理: 非強制のみ対象。拡張特徴（フェーズ/レース×基本特徴）を実体化。"""
    keys = data["feat_keys"]
    ext = _ext_keys(keys)
    all_keys = list(keys) + ext
    ki = {k: i for i, k in enumerate(keys)}
    fixed_match = 0
    fixed_n = 0
    dep = []
    for r in data["records"]:
        if r["forced"]:
            continue
        if r["fixed"]:
            fixed_n += 1
            fixed_match += 1 if r["match_base"] else 0
        else:
            F = r["F"].astype(np.float32)
            deck = r.get("deck") or 0
            early = 1.0 if deck >= 30 else 0.0
            late = 1.0 if deck <= 14 else 0.0
            racef = 1.0 if (r.get("prz_me") or 6) <= 2 else 0.0
            cols = [F]
            for k in PHASE_KEYS:
                if k in ki:
                    cols.append((F[:, ki[k]] * early)[:, None])
                    cols.append((F[:, ki[k]] * late)[:, None])
            for k in RACE_KEYS:
                if k in ki:
                    cols.append((F[:, ki[k]] * racef)[:, None])
            Fe = np.hstack(cols)
            # 終端ごとの候補option indexリスト（順序最適化用）。None=マスク無し→P使用
            bits = None
            if "M" in r:
                bits = []
                nopt = r["nopt"]
                for m in r["M"]:
                    m = int(m)
                    bits.append(None if m == 0 else
                                [i for i in range(nopt) if (m >> i) & 1])
            dep.append((Fe, r["P"], r.get("A"), bits,
                        r.get("ops"), r.get("ocls"),
                        r["actual"][0] if r["actual"] else -1,
                        r["pb"], r["match_base"], r.get("dtype", "?")))
    return all_keys, fixed_match, fixed_n, dep


def _pred_one(P, bits, ops, ocls, prio, ti, pb):
    b = bits[ti] if bits is not None else None
    if b is None or ops is None or len(ops) == 0:
        pred = int(P[ti])
    else:
        bestkey = None
        pred = -1
        for i in b:
            c = ocls[i] if ocls else None
            key = ((1, prio.get(c, 0.0), float(ops[i])) if c is not None
                   else (0, 0.0, float(ops[i])))
            if bestkey is None or key > bestkey:
                bestkey, pred = key, i
    return pb if pred < 0 else pred


def _eval(wvec, prio, fixed_match, fixed_n, dep, detail=False):
    match = fixed_match
    n = fixed_n
    fire = fire_base_ok = fire_new_ok = 0
    order_err = select_err = 0
    by_type = defaultdict(lambda: [0, 0, 0])  # dtype -> [n, ok, base_ok]
    for Fe, P, A, bits, ops, ocls, a0, pb, mb, dt in dep:
        s = Fe @ wvec
        ti = int(np.argmax(s))
        pred = _pred_one(P, bits, ops, ocls, prio, ti, pb)
        ok = pred == a0
        match += 1 if ok else 0
        n += 1
        if detail:
            bt = by_type[dt]
            bt[0] += 1
            bt[1] += 1 if ok else 0
            bt[2] += 1 if mb else 0
            if not ok and A is not None:
                if bool(A[ti]):
                    order_err += 1   # 正解行動はライン内にあるが順序で外した
                else:
                    select_err += 1  # ライン選択自体が外れ
        if pred != pb:
            fire += 1
            fire_base_ok += 1 if mb else 0
            fire_new_ok += 1 if ok else 0
    if detail:
        return (match / max(1, n), n, fire, fire_base_ok, fire_new_ok,
                order_err, select_err, dict(by_type))
    return match / max(1, n)


GRIDS = {
    "hand": [8, 12, 16, 20, 25, 32, 42, 55],
    "burn": [0, -2, -4, -6, -9, -14, -20, -30],
    "chip": [0, 1, 2, 3, 5, 8, 12],
    "attack": [0, 15, 40, 80, 150, 300, 600],
    "next_ready": [0, 10, 25, 50, 90],
    "xero": [0, 8, 15, 25, 40, 70, 120],
    "hammer": [0, 8, 15, 25, 45, 80],
    "mine": [0, 4, 8, 15, 30, 60],
    "prize": [200, 400, 650, 900, 1300, 2000],
    "v_abra": [0, 5, 12, 22, 30, 45, 65],
    "v_kadabra": [0, 10, 25, 42, 60, 85],
    "v_alakazam": [0, 10, 25, 50, 75, 105],
    "v_dunsparce": [0, 5, 14, 25, 40, 60],
    "v_dudunsparce": [0, 8, 16, 26, 40, 60, 90],
    "v_fez": [0, 10, 20, 35, 55],
    "v_shaymin": [0, 5, 10, 20, 35],
    "b_alakazam": [0, 10, 25, 45, 70, 110],
    "b_kadabra": [0, 6, 12, 25, 45],
    "b_abra": [0, 4, 10, 20],
    "b_dudun3": [0, 10, 25, 50],
    "fez_bench": [0, -15, -35, -55, -90, -150],
    "shaymin_bench": [0, -8, -20, -40, -80],
    "lowdeck": [0, -30, -80, -150, -300, -600],
    "middeck": [0, -10, -25, -50, -100],
}
_DELTA_GRIDS = {
    "hand": [-20, -10, -5, 0, 5, 10, 20],
    "burn": [-12, -6, -3, 0, 3, 6, 12],
    "chip": [-10, -5, 0, 5, 10],
    "attack": [-150, -60, -25, 0, 25, 60, 150],
    "prize": [-500, -250, 0, 250, 600],
    "xero": [-25, -10, 0, 10, 25],
    "next_ready": [-40, -20, 0, 20, 40],
    "v_dudunsparce": [-40, -20, 0, 20, 40],
    "v_abra": [-25, -10, 0, 10, 25],
    "v_dunsparce": [-15, 0, 15, 30],
    "v_kadabra": [-40, -20, 0, 20, 40],
    "b_alakazam": [-40, -20, 0, 20, 40],
}
for _k in PHASE_KEYS:
    GRIDS[f"{_k}@early"] = _DELTA_GRIDS[_k]
    GRIDS[f"{_k}@late"] = _DELTA_GRIDS[_k]
for _k in RACE_KEYS:
    GRIDS[f"{_k}@race"] = _DELTA_GRIDS[_k]
# 行動バイアス（ライン選択の性向ダイヤル）
_ABIAS_GRID = [-120, -80, -50, -30, -15, 0, 15, 30, 50, 80]
for _k in ("a_dud", "a_fez", "a_evo_kad", "a_evo_zam", "a_candy", "a_evo_dud",
           "a_attach", "a_enrich", "a_poffin", "a_pad", "a_stretcher", "a_ash",
           "a_hammer", "a_mine", "a_abra", "a_dunsparce", "a_fezmon", "a_shaymon",
           "a_dawn", "a_hilda", "a_lana", "a_xero", "a_retreat", "a_boss"):
    GRIDS[_k] = _ABIAS_GRID


ORDER_CLASSES = ["a_dud", "a_fez", "a_evo_kad", "a_evo_zam", "a_candy", "a_evo_dud",
                 "a_attach", "a_enrich", "a_poffin", "a_pad", "a_stretcher", "a_ash",
                 "a_hammer", "a_mine", "a_abra", "a_dunsparce", "a_fezmon", "a_shaymon",
                 "a_dawn", "a_hilda", "a_lana", "a_xero", "a_retreat", "a_boss"]
PRIO_GRID = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]


def _report(tag, wvec, prio, prep):
    all_keys, fm, fn_, dep = prep
    acc, n, fire, fb, fnew, oerr, serr, by = _eval(wvec, prio, fm, fn_, dep, detail=True)
    print(f"{tag}: acc={acc:.4f} n={n} fire={fire} fire_base_ok={fb} fire_new_ok={fnew} "
          f"| mismatch: order_err={oerr} select_err={serr}")
    print(f"{tag} per-dtype (n / lines_acc / base_acc):")
    for dt, (tn, tok, tbase) in sorted(by.items(), key=lambda kv: -kv[1][0])[:12]:
        print(f"    {dt:22s} {tn:5d}  {tok/tn:.3f}  {tbase/tn:.3f}")
    return acc


def cmd_fit(args):
    with open(args.dev, "rb") as f:
        dev = pickle.load(f)
    devp = _prep(dev)
    all_keys = devp[0]
    valp = None
    if args.val:
        with open(args.val, "rb") as f:
            val = pickle.load(f)
        valp = _prep(val)

    w = {k: 0.0 for k in all_keys}
    w.update({k: v for k, v in dev["defaults"].items() if k in w})
    prio = {k: 0.0 for k in ORDER_CLASSES}
    if args.init:
        init = json.loads(args.init)
        w.update({k: v for k, v in init.get("w", init).items() if k in w})
        prio.update({k: v for k, v in init.get("o", {}).items() if k in prio})

    def wv(wd):
        return np.array([wd[k] for k in all_keys], dtype=np.float32)

    cur = _eval(wv(w), prio, devp[1], devp[2], devp[3])
    print(f"init dev acc = {cur:.4f}")
    worder = [k for k in GRIDS if k in all_keys]
    for p in range(args.passes):
        improved = False
        for k in worder:
            best_v, best_acc = w[k], cur
            for v in GRIDS[k]:
                if v == w[k]:
                    continue
                w2 = dict(w)
                w2[k] = float(v)
                acc = _eval(wv(w2), prio, devp[1], devp[2], devp[3])
                if acc > best_acc + 1e-9:
                    best_acc, best_v = acc, float(v)
            if best_v != w[k]:
                print(f"  pass{p} w {k}: {w[k]} -> {best_v}  dev {cur:.4f} -> {best_acc:.4f}")
                w[k] = best_v
                cur = best_acc
                improved = True
        if not args.no_order:
            wvec = wv(w)
            for k in ORDER_CLASSES:
                best_v, best_acc = prio[k], cur
                for v in PRIO_GRID:
                    if v == prio[k]:
                        continue
                    p2 = dict(prio)
                    p2[k] = v
                    acc = _eval(wvec, p2, devp[1], devp[2], devp[3])
                    if acc > best_acc + 1e-9:
                        best_acc, best_v = acc, v
                if best_v != prio[k]:
                    print(f"  pass{p} o {k}: {prio[k]} -> {best_v}  dev {cur:.4f} -> {best_acc:.4f}")
                    prio[k] = best_v
                    cur = best_acc
                    improved = True
        if not improved:
            break
    print(f"\nfinal dev acc = {cur:.4f}\n")
    _report("dev", wv(w), prio, devp)
    if valp:
        _report("val", wv(w), prio, valp)
    out = {"w": {k: v for k, v in sorted(w.items())},
           "o": {k: v for k, v in sorted(prio.items())}}
    if args.save:
        with open(args.save, "w") as f:
            json.dump(out, f, indent=1)
        print(f"saved -> {args.save}")
    print("\nweights (non-zero):")
    print(json.dumps({k: v for k, v in out["w"].items() if v != 0.0}, indent=1))
    print("order prios (non-zero):")
    print(json.dumps({k: v for k, v in out["o"].items() if v != 0.0}, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--dates", nargs="+", required=True)
    c.add_argument("--team", default="Yushin Ito")
    c.add_argument("--sig", default="81f1758c92")
    c.add_argument("--limit", type=int, default=None)
    c.add_argument("--out", required=True)
    c.add_argument("-w", "--workers", type=int, default=2)
    c.set_defaults(func=cmd_collect)
    f = sub.add_parser("fit")
    f.add_argument("--dev", required=True)
    f.add_argument("--val", default=None)
    f.add_argument("--passes", type=int, default=3)
    f.add_argument("--init", default=None,
                   help='JSON: {"w": {...}, "o": {...}} または重みdict')
    f.add_argument("--save", default=None, help="最終 w/o をJSON保存")
    f.add_argument("--no-order", action="store_true", help="順序優先度の最適化を省略")
    f.set_defaults(func=cmd_fit)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
