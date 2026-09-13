"""自己対戦生成の1決定あたりコストを**工程別に分解**し、GPU化の効果を測る。

PPOではオンポリシー生成が毎イテレーション走るので、生成コストがそのまま全体のコストになる。
どこを最適化すべきかを推測でなく実測で決めるためのプローブ。

測る工程:
  1. engine   : lib.Select + GetBattleData + json.loads
  2. obs      : to_observation_class（dict → データクラス）
  3. feat     : featurize（自分）+ featurize（相手）+ logs + pool_counts
  4. optvec   : 選択肢ごとの option_vector / option_class 構築（Pythonループ）
  5. nn       : LSTM前向き（numpy or torch-GPU）

さらに **torchバックエンドをバッチ32/128/256/512で計測**し、numpy比の速度を出す。
GPUの利点は同時進行バトル数を上げないと出ない（1.2Mパラメータの小さいLSTMでは
カーネル起動のオーバーヘッドが計算時間と同程度になる）ので、そこも含めて確かめる。

usage (Kaggle GPU notebook):
  python selfplay/profile_gen.py --batches 32 128 256 512
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "arena"),
           os.path.join(_ROOT, "agents/094_yushin_nn")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


class Timer:
    def __init__(self):
        self.t = {}

    def add(self, k, dt):
        self.t[k] = self.t.get(k, 0.0) + dt

    def report(self, n_dec, label=""):
        tot = sum(self.t.values())
        print(f"\n--- 工程別 {label} ({n_dec:,}決定) ---")
        for k, v in sorted(self.t.items(), key=lambda x: -x[1]):
            print(f"  {k:<10} {v/n_dec*1000:7.3f} ms/決定  ({v/tot:5.1%})")
        print(f"  {'合計':<10} {tot/n_dec*1000:7.3f} ms/決定")
        return tot / n_dec * 1000


def run(batch: int, backend: str, games: int, tm: Timer):
    import feat_094 as ft
    from cg.api import to_observation_class
    from selfplay.vecenv import VecBattles
    import run_match as rm

    model = os.path.join(_ROOT, "agents/094_yushin_nn/model_094.npz")
    if backend == "torch":
        from selfplay.batch_policy import TorchBatchPolicy
        pol = TorchBatchPolicy(model, n_slots=2 * batch)
    else:
        from selfplay.batch_policy import BatchLSTMPolicy
        pol = BatchLSTMPolicy(model, n_slots=2 * batch)

    deck = rm.read_deck(os.path.join(_ROOT, "agents/094_yushin_nn"))
    seeds = [rm.derive_engine_seed(4242, i) for i in range(batch)]
    t0 = time.perf_counter()
    vb = VecBattles([deck] * batch, [deck] * batch, seeds)
    prev = [[-1.0, -1.0, -1.0] for _ in range(2 * batch)]
    tm.add("engine", time.perf_counter() - t0)
    n_dec = 0
    while vb.live:
        t = time.perf_counter()
        cur = vb.observations()
        tm.add("engine", time.perf_counter() - t)

        slots, boards, logs, prevs, ctxs, cids_l, vecs_l, cls_l, meta = \
            [], [], [], [], [], [], [], [], []
        acts = {}
        for (bi, o) in cur:
            t = time.perf_counter()
            ob = to_observation_class(o)
            tm.add("obs", time.perf_counter() - t)
            st, sel = ob.current, ob.select
            me = st.yourIndex
            n = len(sel.option)
            k = max(min(sel.maxCount or sel.minCount, n), sel.minCount)
            if n == 0:
                acts[bi] = []
                continue
            if n < 2 and sel.minCount != 0:
                acts[bi] = list(range(min(max(1, k), n)))
                continue
            t = time.perf_counter()
            board, _ = ft.feat_vector(st, me)
            oppb, _ = ft.feat_vector(st, 1 - me)
            lg = ft.logs_vector(ob, me)
            hc, pl = ft.pool_counts(st, me)
            tm.add("feat", time.perf_counter() - t)

            t = time.perf_counter()
            cids, vecs = [], []
            for j in range(n):
                c, v = ft.option_vector(ob, sel.option[j], me, hc, pl)
                cids.append(c)
                vecs.append(v)
            ctx = int(sel.context)
            cls = None
            if ctx == 0:
                cls = [ft.CLASS_ID.get(ft.option_class(st, sel, j, me), -1)
                       for j in range(n)]
            n_real = n
            if sel.minCount == 0:
                cids = cids + [ft.NULL_CID]
                vecs = vecs + [ft.null_option_vector()]
                if cls is not None:
                    cls = cls + [-1]
            tm.add("optvec", time.perf_counter() - t)

            slots.append(2 * bi + me)
            boards.append(board)
            logs.append(lg)
            prevs.append(list(prev[2 * bi + me]))
            ctxs.append(ctx)
            cids_l.append(cids)
            vecs_l.append(vecs)
            cls_l.append(cls)
            meta.append((bi, me, n_real, k, cids, vecs))
        if slots:
            t = time.perf_counter()
            scs = pol.scores(slots, boards, logs, prevs, ctxs, cids_l, vecs_l, cls_l)
            tm.add("nn", time.perf_counter() - t)
            for (bi, me, n_real, k, cids, vecs), sc in zip(meta, scs):
                order = sorted(range(len(sc)), key=lambda i: -sc[i])
                if len(cids) > n_real and order[0] == n_real:
                    acts[bi] = []
                    prev[2 * bi + me] = [float(ft.NULL_CID), -1.0, -1.0]
                else:
                    order = [i for i in order if i < n_real]
                    pick = sorted(order[:max(1, min(k, n_real))])
                    acts[bi] = pick
                    prev[2 * bi + me] = [float(cids[pick[0]]),
                                         vecs[pick[0]][0], vecs[pick[0]][1]]
            n_dec += len(slots)
        t = time.perf_counter()
        vb.step(acts)
        tm.add("engine", time.perf_counter() - t)
    vb.close()
    return n_dec


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batches", type=int, nargs="+", default=[32, 128, 256])
    ap.add_argument("--backends", nargs="+", default=["numpy"])
    args = ap.parse_args()
    print(f"CPU {os.cpu_count()}")
    try:
        import torch
        print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
              torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
    except ImportError:
        print("torch なし")
    rows = []
    for backend in args.backends:
        for b in args.batches:
            tm = Timer()
            t0 = time.perf_counter()
            n = run(b, backend, b, tm)
            wall = time.perf_counter() - t0
            per = tm.report(n, f"{backend} batch={b}")
            rows.append((backend, b, per, n / wall * 3600 / 1))
            print(f"  実測 {n/wall:.0f}決定/秒")
    print("\n=== まとめ ===")
    print(f"{'backend':<8}{'batch':>7}{'ms/決定':>10}{'決定/秒':>12}")
    for backend, b, per, dps in rows:
        print(f"{backend:<8}{b:>7}{per:>10.3f}{dps/3600:>12.0f}")


if __name__ == "__main__":
    main()
