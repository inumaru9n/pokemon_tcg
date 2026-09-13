"""094方策の自己対戦を生成し、価値関数の学習データを作る（I-109 Phase RL Stage 0/1）。

出力する1行 = 「ある決定時点の**完全情報**盤面 → そのゲームの勝敗」。
AlphaStar の centralised (privileged) value baseline と同じで、**相手の手札まで見える特徴**を
入力にする。提出物には載らないので合法。

**なぜ自己対戦か**: EXP-055 は「観察データ（本番episodes）から学んだ勝率回帰は交絡で使えない
＝勝者の盤面と相関するものを選んでしまう。根治はオンポリシー自己対戦のみ」と機序まで特定した。
その未検証の処方をここで初めて実行する。

usage:
  uv run python selfplay/gen_selfplay.py --games 2000 -w 6 --out data/rl/sp_000.npz
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "arena"),
           os.path.join(_ROOT, "agents/094_yushin_nn")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

AGENT_DIR = os.path.join(_ROOT, "agents/094_yushin_nn")
MODEL = os.path.join(AGENT_DIR, "model_094.npz")

_W: dict = {}


def _init_worker(batch: int, backend: str = "numpy"):
    if _W.get("ready"):
        return
    import feat_094 as ft
    from cg.api import to_observation_class

    import run_match as rm

    _W["ft"] = ft
    _W["to_obs"] = to_observation_class
    if backend == "torch":
        from selfplay.batch_policy import TorchBatchPolicy
        _W["pol"] = TorchBatchPolicy(MODEL, n_slots=2 * batch)
    else:
        from selfplay.batch_policy import BatchLSTMPolicy
        _W["pol"] = BatchLSTMPolicy(MODEL, n_slots=2 * batch)
    _W["deck"] = rm.read_deck(AGENT_DIR)
    _W["rm"] = rm
    _W["ready"] = True


def _decide(obs_list):
    """[(battle_idx, player, obs_dict)] → ({battle_idx: 選択}, [記録用の行])。

    強制手（選択肢1個 かつ minCount>0）は**LSTMを進めない**。これは094のmain.pyと
    同じ規約で、学習系列と推論系列のステップ数を一致させるために必須（EXP-094 v7）。
    """
    ft, to_obs, pol = _W["ft"], _W["to_obs"], _W["pol"]
    acts, rows = {}, []
    slots, boards, logs, prevs, ctxs, cids_l, vecs_l, cls_l, meta = \
        [], [], [], [], [], [], [], [], []
    for (bi, o) in obs_list:
        ob = to_obs(o)
        st = ob.current
        me = st.yourIndex
        sel = ob.select
        n = len(sel.option)
        k = min(sel.maxCount or sel.minCount, n)
        k = max(k, sel.minCount)
        if n == 0:
            acts[bi] = []
            continue
        if n < 2 and sel.minCount != 0:
            acts[bi] = list(range(min(max(1, k), n)))
            continue
        board, _ = ft.feat_vector(st, me)
        opp_board, _ = ft.feat_vector(st, 1 - me)     # ← 完全情報（criticの入力）
        lg = ft.logs_vector(ob, me)
        hc, pl = ft.pool_counts(st, me)
        cids, vecs = [], []
        for j in range(n):
            c, v = ft.option_vector(ob, sel.option[j], me, hc, pl)
            cids.append(c)
            vecs.append(v)
        ctx = int(sel.context)
        cls = None
        if ctx == 0:
            cls = [ft.CLASS_ID.get(ft.option_class(st, sel, j, me), -1) for j in range(n)]
        n_real = n
        if sel.minCount == 0:
            cids = cids + [ft.NULL_CID]
            vecs = vecs + [ft.null_option_vector()]
            if cls is not None:
                cls = cls + [-1]
        slots.append(2 * bi + me)
        boards.append(board)
        logs.append(lg)
        prevs.append(_W["prev"][2 * bi + me])
        ctxs.append(ctx)
        cids_l.append(cids)
        vecs_l.append(vecs)
        cls_l.append(cls)
        # prev はこの決定の**入力**なので、更新前の値を meta に退避しておく
        meta.append((bi, me, n_real, k, cids, vecs, board, opp_board, st.turn,
                     lg, list(_W["prev"][2 * bi + me]), ctx))
    if slots:
        scs = pol.scores(slots, boards, logs, prevs, ctxs, cids_l, vecs_l, cls_l)
        for m, sc in zip(meta, scs):
            (bi, me, n_real, k, cids, vecs, board, opp_board, turn, lg, prev_in, ctx) = m
            order = sorted(range(len(sc)), key=lambda i: -sc[i])
            if len(cids) > n_real and order[0] == n_real:      # 棄権(NULL)
                acts[bi] = []
                _W["prev"][2 * bi + me] = [float(_W["ft"].NULL_CID), -1.0, -1.0]
            else:
                order = [i for i in order if i < n_real]
                pick = sorted(order[:max(1, min(k, n_real))])
                acts[bi] = pick
                _W["prev"][2 * bi + me] = [float(cids[pick[0]]),
                                           vecs[pick[0]][0], vecs[pick[0]][1]]
            # 棄権も含めて**全ての非強制決定**を記録する（LSTMのステップと1対1にする）
            rows.append((bi, me, turn, board, opp_board, lg, prev_in, ctx))
    return acts, rows


def _run_chunk(task):
    """1ワーカーが batch 試合を同時進行させ、**(試合, プレイヤー) 単位の決定列**を返す。

    Vは方策と同じLSTMアーキテクチャで学習するので、**間引かずに系列のまま**出す。
    間引くとVのLSTMステップが方策と1対1でなくなり、advantageの計算がズレる。
    """
    (seed_base, g0, batch, backend) = task
    _init_worker(batch, backend)
    from selfplay.vecenv import VecBattles
    rm, deck = _W["rm"], _W["deck"]
    seeds = [rm.derive_engine_seed(seed_base, g0 + j) for j in range(batch)]
    vb = VecBattles([deck] * batch, [deck] * batch, seeds)
    _W["prev"] = [[-1.0, -1.0, -1.0] for _ in range(2 * batch)]
    _W["pol"].reset_all()
    per_slot: list[list] = [[] for _ in range(2 * batch)]
    while vb.live:
        cur = vb.observations()
        acts, rows = _decide(cur)
        for (bi, me, turn, board, opp_board, lg, prev_in, ctx) in rows:
            per_slot[2 * bi + me].append((turn, ctx, board, opp_board, lg, prev_in))
        vb.step(acts)
    # **ワーカー内でnumpyに詰めてから返す**。Pythonのリストのまま返すと、親が
    # 全チャンクを溜め込んだ時点でメモリが数十GBに膨れてスラッシングする
    # （実際に4万試合の実行で親RSS 2GB・スワップ4.3GB・CPU 7秒/27分で停止した）。
    keep = []
    for bi in range(batch):
        r = vb.results[bi]
        if r not in (0, 1):
            continue                      # 引き分け/未完は捨てる
        for me in (0, 1):
            rows = per_slot[2 * bi + me]
            if rows:
                keep.append((g0 + bi, me, 1.0 if r == me else 0.0, rows))
    n_rows = sum(len(k[3]) for k in keep)
    F = len(keep[0][3][0][2]) if keep else 0
    L = len(keep[0][3][0][4]) if keep else 0
    out = dict(
        board=np.empty((n_rows, F), np.float16),
        opp_board=np.empty((n_rows, F), np.float16),
        logs=np.empty((n_rows, L), np.float16),
        prev=np.empty((n_rows, 3), np.float32),
        ctx=np.empty(n_rows, np.int16),
        turn=np.empty(n_rows, np.int16),
        seq_id=np.empty(n_rows, np.int32),
        label=np.empty(len(keep), np.float32),
        game_id=np.empty(len(keep), np.int32))
    i = 0
    for si, (gid, me, y, rows) in enumerate(keep):
        out["label"][si] = y
        out["game_id"][si] = gid
        for (tn, cx, b, ob, lg, pv) in rows:
            out["board"][i] = b
            out["opp_board"][i] = ob
            out["logs"][i] = lg
            out["prev"][i] = pv
            out["ctx"][i] = cx
            out["turn"][i] = tn
            out["seq_id"][i] = si
            i += 1
    res = list(vb.results)
    steps = list(vb.steps)
    vb.close()
    return out, res, steps


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=32, help="1ワーカーの同時バトル数")
    ap.add_argument("-w", "--workers", type=int, default=6)
    ap.add_argument("--seed-base", type=int, default=20260727)
    # **既定で無効**。max_tasks_per_child によるワーカー再生成は、macOS+spawn で
    # 再生成に失敗してデッドロックする（実測: シャード1本目の直後にワーカーが全滅し、
    # 親とresource_trackerだけが残って空転。無効化すると毎分1シャードで安定）。
    # libcgのリーク対策は「1プロセスあたりの試合数を数万に抑える」ことで代替する。
    ap.add_argument("--tasks-per-child", type=int, default=0,
                    help="0=無効。>0 でワーカーを作り直す（macOSではデッドロックする）")
    ap.add_argument("--policy-backend", choices=["numpy", "torch"], default="numpy",
                    help="torch=GPUで行列積を回す（PPOの生成コスト削減用）")
    ap.add_argument("--rows-per-shard", type=int, default=800_000,
                    help="この決定数を超えたらシャードを書き出して親のメモリを解放する")
    ap.add_argument("--out", default="data/rl/selfplay.npz")
    args = ap.parse_args()

    tasks = []
    g = 0
    while g < args.games:
        b = min(args.batch, args.games - g)
        tasks.append((args.seed_base, g, b, args.policy_backend))
        g += b

    t0 = time.perf_counter()
    # **結果を溜め込まず、一定行数ごとにシャードへ書き出す**。ex.map の戻りを
    # list() で全部持つと親が数十GBに膨れる（実測: 4万試合で親RSS 2GB＋スワップ4.3GB）。
    buf: list[dict] = []
    buf_rows = 0
    shard = 0
    res: list[int] = []
    steps: list[int] = []
    written = {"rows": 0, "seqs": 0, "files": []}
    base, ext = os.path.splitext(os.path.abspath(args.out))
    os.makedirs(os.path.dirname(base), exist_ok=True)

    def flush():
        nonlocal buf, buf_rows, shard
        if not buf:
            return
        s_off = 0
        merged = {}
        for k in ("board", "opp_board", "logs", "prev", "ctx", "turn"):
            merged[k] = np.concatenate([b[k] for b in buf])
        sids, labs, gids = [], [], []
        for b in buf:
            sids.append(b["seq_id"] + s_off)
            labs.append(b["label"])
            gids.append(b["game_id"])
            s_off += len(b["label"])
        merged["seq_id"] = np.concatenate(sids)
        merged["label"] = np.concatenate(labs)
        merged["game_id"] = np.concatenate(gids)
        path = f"{base}_p{shard:03d}{ext}"
        np.savez_compressed(path, **merged)
        written["rows"] += len(merged["seq_id"])
        written["seqs"] += len(merged["label"])
        written["files"].append(path)
        print(f"  shard {shard:03d}: 系列{len(merged['label']):,} 決定{len(merged['seq_id']):,} "
              f"-> {os.path.basename(path)} ({os.path.getsize(path)/1e6:.0f}MB)", flush=True)
        shard += 1
        buf = []
        buf_rows = 0

    def take(out, r, s):
        nonlocal buf_rows
        buf.append(out)
        buf_rows += len(out["seq_id"])
        res.extend(r)
        steps.extend(s)
        if buf_rows >= args.rows_per_shard:
            flush()

    if args.workers <= 1:
        for t in tasks:
            take(*_run_chunk(t))
    else:
        kw = ({"max_tasks_per_child": args.tasks_per_child}
              if args.tasks_per_child > 0 else {})
        with ProcessPoolExecutor(max_workers=args.workers, **kw) as ex:
            for out, r, s in ex.map(_run_chunk, tasks):
                take(out, r, s)
    flush()
    dt = time.perf_counter() - t0

    n_dec = sum(res.count(x) for x in (0, 1))
    print(f"{args.games}試合 / {dt:.1f}秒 = {args.games/dt:.1f}試合/秒 "
          f"({args.games/dt*3600:,.0f}試合/時, {args.workers}ワーカー)")
    print(f"  決着 {n_dec} / 引き分け {res.count(2)} / 平均{np.mean(steps):.0f}ステップ")
    print(f"  先手勝率 {res.count(0)/max(1,n_dec):.1%}（自己対戦なので偏りは先後の非対称性）")
    print(f"  系列 {written['seqs']:,}本 / 決定 {written['rows']:,} / "
          f"シャード {len(written['files'])}本")
    tot = sum(os.path.getsize(f) for f in written["files"]) / 1e6
    print(f"saved -> {base}_p*{ext} 計 {tot:.0f}MB")


if __name__ == "__main__":
    main()
