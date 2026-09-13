"""EXP-055 価値ネット V(終端盤面)=勝率 の学習パイプライン (I-107)。

collect: sig 81f1758c92 が登場する全対局（チーム不問、両者DONE）をリプレイし、
  各ターンの最終決定（実選択がATTACK/END）で055の列挙器を_collectモードで走らせ、
  実現された終端（=そのターン終了盤面）の特徴ベクトル+そのゲームの勝敗ラベルをnpz保存。
  特徴は推論時と同一コードパス（terminals()）で生成される=パリティ問題なし。

fit: numpy MLP（BCE、Adam）で盤面→勝率を回帰。day分割でval AUC・ターン帯別AUCを報告し、
  value_055.npz（main.pyが起動時ロード）を保存。

eval: calib_055_decisions.py collect の決定pickle上で、V選択の一致率/fire精度を
  052ベースと比較（dtype・ターン帯別）。win終端の厳密優先も運用と同一に適用。

usage:
  uv run python agents/055_alakazam_value/value_055.py collect \
      --dates 2026-07-08 2026-07-09 2026-07-11 --out /tmp/tv_dev.npz -w 4
  uv run python agents/055_alakazam_value/value_055.py fit \
      --train /tmp/tv_dev.npz --val /tmp/tv_val.npz \
      --save agents/055_alakazam_value/value_055.npz
  uv run python agents/055_alakazam_value/value_055.py eval \
      --pkl /tmp/dec_val.pkl --value agents/055_alakazam_value/value_055.npz
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import pickle
import sys
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np

REPO_ROOT = "/Users/akira/kaggle/pokemon_tcg"
AGENT_DIR = os.path.join(REPO_ROOT, "agents/055_alakazam_value")
SIG = "81f1758c92"

_W: dict = {}


def _ensure_worker():
    if _W.get("loaded"):
        return
    os.environ["F055_VALUE"] = "off"  # 収集はV非依存（列挙+特徴のみ使う）
    # 使うのはroot直下の終端（実現された最終行動）だけなのでBFS展開を止める＝大幅高速化
    os.environ["F054_CAP"] = "1"
    main_path = os.path.join(AGENT_DIR, "main.py")
    for p in (REPO_ROOT, AGENT_DIR):
        if p not in sys.path:
            sys.path.insert(0, p)
    spec = importlib.util.spec_from_file_location("value055_agent", main_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    _W["mod"] = mod
    _W["zips"] = {}
    _W["feat_keys"] = sorted(k for k in mod._W054 if "@" not in k)
    _W["abias"] = [k for k in _W["feat_keys"] if k.startswith("a_")]
    _W["loaded"] = True


def _zip_for(date: str) -> zipfile.ZipFile:
    if date not in _W["zips"]:
        d = os.path.join(REPO_ROOT, "data", "episodes", date)
        zpath = next(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".zip"))
        _W["zips"][date] = zipfile.ZipFile(zpath)
    return _W["zips"][date]


def find_sig_games(dates: list[str]) -> list[tuple]:
    """(date, ep_id, me_side, y) のリスト。ミラーは両側から1サンプルずつ。"""
    out = []
    for date in dates:
        path = os.path.join(REPO_ROOT, "data", "episodes", date, "summary.csv")
        byep: dict = defaultdict(dict)
        for row in csv.DictReader(open(path)):
            byep[row["episode_id"]][row.get("player")] = row
        for ep, d in byep.items():
            if len(d) != 2:
                continue
            if any(r.get("status") != "DONE" for r in d.values()):
                continue
            for pl, r in d.items():
                if r.get("deck_signature") == SIG and r.get("reward") in ("1", "-1"):
                    out.append((date, ep, int(pl), 1 if r["reward"] == "1" else 0))
    return out


def process_episode(task: tuple) -> dict:
    (date, ep_id, me, y) = task
    _ensure_worker()
    mod = _W["mod"]
    feat_keys = _W["feat_keys"]
    abias = _W["abias"]
    try:
        steps = json.loads(_zip_for(date).read(f"{ep_id}.json"))["steps"]
    except KeyError:
        return {"ep": ep_id, "skip": "no-json", "rows": [], "meta": []}

    mod._search_time_used = 0.0
    rows, meta, errs = [], [], []
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
        actual = steps[t + 1][me].get("action")
        if not isinstance(actual, list) or not actual:
            continue
        a0 = actual[0]
        if a0 >= len(sel["option"]):
            continue
        a_opt = sel["option"][a0]
        if a_opt.get("type") not in (13, 14):  # ATTACK / END のみ=ターン最終決定
            continue
        cur = obs["current"]
        turn = cur.get("turn", 0)
        mod._lines_boss_bench = None
        terms: list = []
        try:
            obs_cls = mod.to_observation_class(obs)
            mod._lines_layer(obs_cls, _collect=terms)
        except Exception as e:  # noqa: BLE001
            errs.append(f"{type(e).__name__}: {e}")
            continue
        if not terms:
            continue
        is_attack = a_opt["type"] == 13
        cands = []
        for feat, pred, mask in terms:
            if any(feat.get(k) for k in abias):
                continue  # ライン内に非終端行動あり=root直下の終端でない
            if bool(feat.get("attack")) != is_attack:
                continue
            if mask and a0 <= 63 and not (mask >> a0) & 1:
                continue  # この終端のマッピング候補に実選択が含まれない
            cands.append(feat)
        if not cands:
            continue
        if len(cands) > 1:
            # 攻撃対象の多義性（Cruel Arrow等）: 価値の高い対象を選んだと仮定
            cands.sort(key=lambda f: (f.get("win", 0), f.get("prize", 0),
                                      f.get("chip", 0)), reverse=True)
        feat = cands[0]
        rows.append([float(feat.get(k, 0)) for k in feat_keys])
        deck = (cur["players"][cur["yourIndex"]].get("deckCount")) or 0
        meta.append((turn, deck))
    return {"ep": ep_id, "rows": rows, "meta": meta, "y": y, "date": date,
            "errors": errs}


def cmd_collect(args):
    games = find_sig_games(args.dates)
    if args.limit:
        per: dict = defaultdict(int)
        kept = []
        for g in games:
            per[g[0]] += 1
            if per[g[0]] <= args.limit:
                kept.append(g)
        games = kept
    print(f"{len(games)} game-sides ({Counter(g[0] for g in games)})")
    if args.workers <= 1:
        results = [process_episode(g) for g in games]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            results = list(ex.map(process_episode, games, chunksize=4))
    _ensure_worker()
    X, y, turn, deck, dates = [], [], [], [], []
    nerr = 0
    for r in results:
        nerr += len(r.get("errors") or [])
        for row, (tn, dk) in zip(r["rows"], r["meta"]):
            X.append(row)
            y.append(r["y"])
            turn.append(tn)
            deck.append(dk)
            dates.append(r["date"])
    X = np.array(X, dtype=np.float32)
    np.savez_compressed(
        args.out, X=X, y=np.array(y, dtype=np.int8),
        turn=np.array(turn, dtype=np.int16), deck=np.array(deck, dtype=np.int16),
        date=np.array(dates, dtype="U10"),
        feat_keys=np.array(_W["feat_keys"], dtype="U24"))
    print(f"samples={len(X)} win-rate={np.mean(y):.3f} feat={X.shape[1]} "
          f"errors={nerr} -> {args.out}")
    for r in results:
        for e in (r.get("errors") or [])[:1]:
            print("  ERROR", r["ep"], e)
            return


# ---------------------------------------------------------------------------


def _auc(y, p):
    order = np.argsort(p)
    rank = np.empty(len(p))
    rank[order] = np.arange(1, len(p) + 1)
    pos = y == 1
    npos, nneg = pos.sum(), (~pos).sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    return (rank[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg)


class MLP:
    def __init__(self, sizes, seed=0):
        rng = np.random.default_rng(seed)
        self.Ws = [rng.normal(0, np.sqrt(2.0 / sizes[i]),
                              (sizes[i], sizes[i + 1])).astype(np.float32)
                   for i in range(len(sizes) - 1)]
        self.bs = [np.zeros(sizes[i + 1], dtype=np.float32)
                   for i in range(len(sizes) - 1)]
        self.m = [np.zeros_like(w) for w in self.Ws + self.bs]
        self.v = [np.zeros_like(w) for w in self.Ws + self.bs]
        self.t = 0

    def forward(self, X, keep=None):
        acts = [X]
        for i, (W, b) in enumerate(zip(self.Ws, self.bs)):
            X = X @ W + b
            if i < len(self.Ws) - 1:
                X = np.maximum(X, 0.0)
            acts.append(X)
        return (acts if keep else X[:, 0])

    def step(self, X, y, lr=1e-3, wd=1e-5):
        acts = self.forward(X, keep=True)
        logit = acts[-1][:, 0]
        p = 1.0 / (1.0 + np.exp(-logit))
        n = len(y)
        gWs = [None] * len(self.Ws)
        gbs = [None] * len(self.bs)
        delta = ((p - y) / n)[:, None].astype(np.float32)
        for i in range(len(self.Ws) - 1, -1, -1):
            a = acts[i]
            gWs[i] = a.T @ delta + wd * self.Ws[i]
            gbs[i] = delta.sum(axis=0)
            if i > 0:
                delta = (delta @ self.Ws[i].T) * (acts[i] > 0)
        self.t += 1
        grads = gWs + gbs
        params = self.Ws + self.bs
        b1, b2, eps = 0.9, 0.999, 1e-8
        for j, (prm, g) in enumerate(zip(params, grads)):
            self.m[j] = b1 * self.m[j] + (1 - b1) * g
            self.v[j] = b2 * self.v[j] + (1 - b2) * g * g
            mh = self.m[j] / (1 - b1 ** self.t)
            vh = self.v[j] / (1 - b2 ** self.t)
            prm -= lr * mh / (np.sqrt(vh) + eps)
        return float(-(y * np.log(p + 1e-9)
                       + (1 - y) * np.log(1 - p + 1e-9)).mean())


def _load_npzs(paths):
    Xs, ys, turns = [], [], []
    keys = None
    for p in paths:
        z = np.load(p, allow_pickle=False)
        k = [str(x) for x in z["feat_keys"]]
        if keys is None:
            keys = k
        assert k == keys, f"feat_keys mismatch: {p}"
        Xs.append(z["X"])
        ys.append(z["y"])
        turns.append(z["turn"])
    return np.vstack(Xs), np.concatenate(ys).astype(np.float32), \
        np.concatenate(turns), keys


def _turn_buckets(turn):
    return [("t1-2", turn <= 2), ("t3-5", (turn >= 3) & (turn <= 5)),
            ("t6-9", (turn >= 6) & (turn <= 9)), ("t10+", turn >= 10)]


def cmd_fit(args):
    Xtr, ytr, ttr, keys = _load_npzs(args.train)
    Xva, yva, tva, kv = _load_npzs(args.val)
    assert kv == keys
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0)
    sd[sd < 1e-6] = 1.0
    Xtr = (Xtr - mu) / sd
    Xva = (Xva - mu) / sd
    print(f"train={len(Xtr)} (win {ytr.mean():.3f})  val={len(Xva)} "
          f"(win {yva.mean():.3f})  feat={len(keys)}")

    rng = np.random.default_rng(42)
    net = MLP([len(keys), args.hidden, args.hidden, 1], seed=42)
    best = (-1.0, None)
    patience = 0
    for ep in range(args.epochs):
        idx = rng.permutation(len(Xtr))
        losses = []
        for i in range(0, len(idx), args.batch):
            b = idx[i:i + args.batch]
            losses.append(net.step(Xtr[b], ytr[b], lr=args.lr))
        pva = net.forward(Xva)
        auc = _auc(yva, pva)
        print(f"  epoch {ep:2d} loss={np.mean(losses):.4f} val_auc={auc:.4f}")
        if auc > best[0] + 1e-4:
            best = (auc, [w.copy() for w in net.Ws + net.bs])
            patience = 0
        else:
            patience += 1
            if patience >= args.patience:
                break
    nl = len(net.Ws)
    net.Ws = best[1][:nl]
    net.bs = best[1][nl:]
    print(f"best val AUC = {best[0]:.4f}")

    pva = 1.0 / (1.0 + np.exp(-net.forward(Xva)))
    for name, m in _turn_buckets(tva):
        if m.sum() > 50:
            print(f"  val {name:5s} n={m.sum():6d} auc={_auc(yva[m], pva[m]):.4f} "
                  f"base_win={yva[m].mean():.3f}")
    # 較正表（予測decile→実勝率）
    q = np.quantile(pva, np.linspace(0, 1, 11))
    print("  calibration (pred -> actual):")
    for i in range(10):
        m = (pva >= q[i]) & (pva <= q[i + 1])
        if m.sum() > 0:
            print(f"    [{q[i]:.2f},{q[i+1]:.2f}] n={m.sum():5d} "
                  f"pred={pva[m].mean():.3f} actual={yva[m].mean():.3f}")

    if args.save:
        out = {"n_layers": np.int32(nl), "mu": mu.astype(np.float32),
               "sd": sd.astype(np.float32),
               "feat_keys": np.array(keys, dtype="U24")}
        for i, (w, b) in enumerate(zip(net.Ws, net.bs)):
            out[f"W{i}"] = w
            out[f"b{i}"] = b
        np.savez(args.save, **out)
        print(f"saved -> {args.save}")


# ---------------------------------------------------------------------------


def _rank_prep(pkl_path, keys_v, max_neg=63, seed=0):
    """決定pickle→ランカー学習用の(連結X, セグメント境界, 正解行フラグ)。
    正解行 = P[i]==actual（マッピングが本人選択に一致する終端）。負例はサブサンプル。"""
    rng = np.random.default_rng(seed)
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    keys = data["feat_keys"]
    col = np.array([keys.index(k) for k in keys_v], dtype=np.int64)
    Xs, seg_sizes, pos_flags, metas = [], [], [], []
    for r in data["records"]:
        if r["forced"] or r["fixed"]:
            continue
        a0 = r["actual"][0] if r["actual"] else -1
        P = r["P"]
        pos = np.where(P == a0)[0]
        if len(pos) == 0 or a0 < 0:
            continue  # oracle miss
        neg = np.where(P != a0)[0]
        if len(neg) > max_neg:
            neg = rng.choice(neg, size=max_neg, replace=False)
        idx = np.concatenate([pos, neg])
        F = r["F"][idx][:, col].astype(np.float32)
        Xs.append(F)
        seg_sizes.append(len(idx))
        flag = np.zeros(len(idx), dtype=np.float32)
        flag[:len(pos)] = 1.0 / len(pos)
        pos_flags.append(flag)
        metas.append((r.get("turn", 0), r.get("dtype", "?")))
    X = np.vstack(Xs)
    sizes = np.array(seg_sizes, dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    y = np.concatenate(pos_flags)
    print(f"  {pkl_path}: decisions={len(sizes)} rows={len(X)}")
    return X, starts, sizes, y, metas


def _seg_softmax(z, starts, sizes):
    """セグメントごとのsoftmax（行が決定ごとに連続配置されている前提）。"""
    zmax = np.maximum.reduceat(z, starts)
    z = np.exp(z - np.repeat(zmax, sizes))
    denom = np.add.reduceat(z, starts)
    return z / np.repeat(denom, sizes)


def _rank_prep_multi(paths, keys_v, max_neg, seed=0):
    Xs, sizes_l, ys, metas = [], [], [], []
    for i, p in enumerate(paths):
        X, _st, sz, y, m = _rank_prep(p, keys_v, max_neg=max_neg, seed=seed + i)
        Xs.append(X)
        sizes_l.append(sz)
        ys.append(y)
        metas += m
    X = np.vstack(Xs)
    sizes = np.concatenate(sizes_l)
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    return X, starts, sizes, np.concatenate(ys), metas


def cmd_fit_gbt(args):
    import lightgbm as lgb

    with open(args.dev[0], "rb") as f:
        probe = pickle.load(f)["feat_keys"]
    keys = probe if args.with_abias else [k for k in probe
                                          if not k.startswith("a_")]
    Xd, _std, szd, yd, _ = _rank_prep_multi(args.dev, keys, args.max_neg, seed=1)
    Xv, stv, szv, yv, _ = _rank_prep_multi([args.val], keys, args.max_neg, seed=99)
    print(f"train rows={len(Xd)} decisions={len(szd)}  "
          f"val rows={len(Xv)} decisions={len(szv)}  feat={len(keys)}")
    dtr = lgb.Dataset(Xd, label=(yd > 0).astype(np.int8), group=szd,
                      feature_name=list(keys))
    dva = lgb.Dataset(Xv, label=(yv > 0).astype(np.int8), group=szv,
                      reference=dtr, feature_name=list(keys))
    params = dict(objective="lambdarank", metric="ndcg", ndcg_eval_at=[1],
                  learning_rate=args.lr, num_leaves=args.leaves,
                  min_data_in_leaf=50, feature_fraction=0.9,
                  bagging_fraction=0.9, bagging_freq=1,
                  lambdarank_truncation_level=20, verbose=-1, seed=42)
    bst = lgb.train(params, dtr, num_boost_round=args.rounds,
                    valid_sets=[dva],
                    callbacks=[lgb.early_stopping(args.early),
                               lgb.log_evaluation(100)])
    # val pick accuracy（argmax行が正例か）
    z = bst.predict(Xv, num_iteration=bst.best_iteration)
    ok = sum(1 for s, n in zip(stv, szv)
             if yv[s + int(np.argmax(z[s:s + n]))] > 0)
    print(f"val pick acc = {ok}/{len(szv)} = {ok/len(szv):.4f} "
          f"(best_iter={bst.best_iteration})")
    if args.save:
        bst.save_model(args.save, num_iteration=bst.best_iteration)
        print(f"saved -> {args.save}")
    imp = sorted(zip(keys, bst.feature_importance("gain")),
                 key=lambda kv: -kv[1])[:15]
    print("top gains:", [(k, int(v)) for k, v in imp])


def cmd_fit_rank(args):
    _keys_probe = None
    with open(args.dev, "rb") as f:
        _keys_probe = pickle.load(f)["feat_keys"]
    keys = [k for k in _keys_probe if not k.startswith("a_")]  # 行動バイアスは状態でない
    if args.state_only:
        drop = {"attack", "win", "prize", "chip", "burn", "xero", "hammer",
                "mine", "next_ready", "deckout", "lowdeck", "middeck"}
        keys = [k for k in keys if k not in drop]
    Xd, std, szd, yd, _md = _rank_prep(args.dev, keys, seed=1)
    Xv, stv, szv, yv, mv = _rank_prep(args.val, keys, max_neg=10 ** 9, seed=2)
    mu = Xd.mean(axis=0)
    sd = Xd.std(axis=0)
    sd[sd < 1e-6] = 1.0
    Xd = (Xd - mu) / sd
    Xv = (Xv - mu) / sd

    net = MLP([len(keys), args.hidden, args.hidden, 1], seed=42)
    rng = np.random.default_rng(7)
    nseg = len(szd)

    def val_acc():
        z = net.forward(Xv)
        pick = []
        for s, n in zip(stv, szv):
            pick.append(s + int(np.argmax(z[s:s + n])))
        pick = np.array(pick)
        return float(np.mean(yv[pick] > 0))

    best = (-1.0, None)
    patience = 0
    for ep in range(args.epochs):
        order = rng.permutation(nseg)
        losses = []
        for i in range(0, nseg, args.batch):
            segs = order[i:i + args.batch]
            rows = np.concatenate([np.arange(std[s], std[s] + szd[s]) for s in segs])
            bsz = szd[segs]
            bst = np.concatenate([[0], np.cumsum(bsz)[:-1]])
            Xb, yb = Xd[rows], yd[rows]
            acts = net.forward(Xb, keep=True)
            z = acts[-1][:, 0]
            p = _seg_softmax(z, bst, bsz)
            loss = -np.log((p * (yb > 0)).reshape(-1)[yb > 0].clip(1e-9)).mean()
            losses.append(float(loss))
            delta = ((p - yb) / len(bsz))[:, None].astype(np.float32)
            gWs = [None] * len(net.Ws)
            gbs = [None] * len(net.bs)
            for li in range(len(net.Ws) - 1, -1, -1):
                a = acts[li]
                gWs[li] = a.T @ delta + 1e-5 * net.Ws[li]
                gbs[li] = delta.sum(axis=0)
                if li > 0:
                    delta = (delta @ net.Ws[li].T) * (acts[li] > 0)
            net.t += 1
            grads = gWs + gbs
            params = net.Ws + net.bs
            b1, b2, eps = 0.9, 0.999, 1e-8
            for j, (prm, g) in enumerate(zip(params, grads)):
                net.m[j] = b1 * net.m[j] + (1 - b1) * g
                net.v[j] = b2 * net.v[j] + (1 - b2) * g * g
                mh = net.m[j] / (1 - b1 ** net.t)
                vh = net.v[j] / (1 - b2 ** net.t)
                prm -= args.lr * mh / (np.sqrt(vh) + eps)
        va = val_acc()
        print(f"  epoch {ep:2d} loss={np.mean(losses):.4f} val_pick_acc={va:.4f}")
        if va > best[0] + 1e-4:
            best = (va, [w.copy() for w in net.Ws + net.bs])
            patience = 0
        else:
            patience += 1
            if patience >= args.patience:
                break
    nl = len(net.Ws)
    net.Ws = best[1][:nl]
    net.bs = best[1][nl:]
    print(f"best val pick acc = {best[0]:.4f}")
    if args.save:
        out = {"n_layers": np.int32(nl), "mu": mu.astype(np.float32),
               "sd": sd.astype(np.float32),
               "feat_keys": np.array(keys, dtype="U24")}
        for i, (w, b) in enumerate(zip(net.Ws, net.bs)):
            out[f"W{i}"] = w
            out[f"b{i}"] = b
        np.savez(args.save, **out)
        print(f"saved -> {args.save}")


def _mlp_load(path):
    z = np.load(path, allow_pickle=False)
    nl = int(z["n_layers"])
    return {"keys": [str(k) for k in z["feat_keys"]], "mu": z["mu"], "sd": z["sd"],
            "Ws": [z[f"W{i}"] for i in range(nl)],
            "bs": [z[f"b{i}"] for i in range(nl)]}


def _mlp_scores(V, X):
    X = (X - V["mu"]) / V["sd"]
    for i, (W, b) in enumerate(zip(V["Ws"], V["bs"])):
        X = X @ W + b
        if i < len(V["Ws"]) - 1:
            X = np.maximum(X, 0.0)
    return X[:, 0]


def cmd_eval(args):
    """決定pickle（calib_055_decisions.py collect産）上でV選択の一致率/fire精度。"""
    with open(args.pkl, "rb") as f:
        data = pickle.load(f)
    keys = data["feat_keys"]
    if args.value.endswith(".txt"):  # LightGBM model
        import lightgbm as lgb
        bst = lgb.Booster(model_file=args.value)
        keys_v = bst.feature_name()

        def _score(F):
            return bst.predict(F)
    else:
        V = _mlp_load(args.value)
        keys_v = V["keys"]

        def _score(F):
            return _mlp_scores(V, F)
    col = [keys.index(k) for k in keys_v]
    wcol = keys.index("win")
    tot = ok = 0
    fire = fb_ok = fv_ok = 0
    by_dt: dict = defaultdict(lambda: [0, 0, 0])
    by_turn: dict = defaultdict(lambda: [0, 0, 0, 0, 0])  # n, ok, base_ok, fire, fire_net
    for r in data["records"]:
        if r["forced"]:
            continue
        a0 = r["actual"][0] if r["actual"] else -1
        turn = r.get("turn", 0)
        tb = ("t1-2" if turn <= 2 else "t3-5" if turn <= 5
              else "t6-9" if turn <= 9 else "t10+")
        if r["fixed"]:
            tot += 1
            ok += 1 if r["match_base"] else 0
            continue
        F = r["F"].astype(np.float32)
        sc = _score(F[:, col])
        winm = F[:, wcol] > 0
        pool = np.where(winm)[0] if winm.any() else np.arange(len(sc))
        pick = int(pool[np.argmax(sc[pool])])
        pred = int(r["P"][pick])
        if pred < 0:
            correct = r["match_base"]  # マッピング失敗→policy委譲
        else:
            correct = (pred == a0)
        tot += 1
        ok += 1 if correct else 0
        dt = r.get("dtype", "?")
        by_dt[dt][0] += 1
        by_dt[dt][1] += 1 if correct else 0
        by_dt[dt][2] += 1 if r["match_base"] else 0
        b = by_turn[tb]
        b[0] += 1
        b[1] += 1 if correct else 0
        b[2] += 1 if r["match_base"] else 0
        if pred >= 0 and pred != r["pb"]:
            fire += 1
            fb_ok += 1 if r["match_base"] else 0
            fv_ok += 1 if correct else 0
            b[3] += 1
            b[4] += (1 if correct else 0) - (1 if r["match_base"] else 0)
    nb = sum(1 for r in data["records"]
             if not r["forced"] and r["match_base"]) or 1
    print(f"V acc = {ok}/{tot} = {ok/tot:.4f}   (base = {nb}/{tot} = {nb/tot:.4f})")
    print(f"fire = {fire}  base_ok={fb_ok}  V_ok={fv_ok}  net={fv_ok - fb_ok:+d}")
    print("\nby turn-bucket (n / V_acc / base_acc / fire / fire_net):")
    for tb in ("t1-2", "t3-5", "t6-9", "t10+"):
        n, o, bo, fr, fn = by_turn[tb]
        if n:
            print(f"  {tb:5s} {n:6d}  {o/n:.3f}  {bo/n:.3f}  {fr:5d}  {fn:+d}")
    print("\nby dtype (n / V_acc / base_acc):")
    for dt, (n, o, bo) in sorted(by_dt.items(), key=lambda kv: -kv[1][0])[:12]:
        print(f"  {dt:22s} {n:5d}  {o/n:.3f}  {bo/n:.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--dates", nargs="+", required=True)
    c.add_argument("--limit", type=int, default=None)
    c.add_argument("--out", required=True)
    c.add_argument("-w", "--workers", type=int, default=2)
    f = sub.add_parser("fit")
    f.add_argument("--train", nargs="+", required=True)
    f.add_argument("--val", nargs="+", required=True)
    f.add_argument("--hidden", type=int, default=64)
    f.add_argument("--epochs", type=int, default=60)
    f.add_argument("--batch", type=int, default=512)
    f.add_argument("--lr", type=float, default=1e-3)
    f.add_argument("--patience", type=int, default=6)
    f.add_argument("--save", default=None)
    e = sub.add_parser("eval")
    e.add_argument("--pkl", required=True)
    e.add_argument("--value", required=True)
    r = sub.add_parser("fit-rank")
    r.add_argument("--dev", required=True)
    r.add_argument("--val", required=True)
    r.add_argument("--hidden", type=int, default=64)
    r.add_argument("--epochs", type=int, default=40)
    r.add_argument("--batch", type=int, default=256)
    r.add_argument("--lr", type=float, default=1e-3)
    r.add_argument("--patience", type=int, default=5)
    r.add_argument("--state-only", action="store_true", dest="state_only")
    r.add_argument("--save", default=None)
    g = sub.add_parser("fit-gbt")
    g.add_argument("--dev", nargs="+", required=True)
    g.add_argument("--val", required=True)
    g.add_argument("--max-neg", type=int, default=200, dest="max_neg")
    g.add_argument("--lr", type=float, default=0.05)
    g.add_argument("--leaves", type=int, default=63)
    g.add_argument("--rounds", type=int, default=2000)
    g.add_argument("--early", type=int, default=100)
    g.add_argument("--with-abias", action="store_true", dest="with_abias")
    g.add_argument("--save", default=None)
    args = ap.parse_args()
    {"collect": cmd_collect, "fit": cmd_fit, "eval": cmd_eval,
     "fit-rank": cmd_fit_rank, "fit-gbt": cmd_fit_gbt}[args.cmd](args)


if __name__ == "__main__":
    main()
