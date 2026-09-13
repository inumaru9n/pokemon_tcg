"""PPO用のオンポリシー軌跡を収集する（I-109 Phase RL Stage 2a）。

Stage 0/1 の `gen_selfplay.py` との違い:
  - **argmax でなく softmax からサンプリング**する（PPOには探索が要る）
  - 各決定の **選んだindexと log π_old(a|s)** を記録する
  - **選択肢の特徴も保持**する（更新時に log π_new(a|s) を計算し直すため）
  - ディスクに書かず**メモリ上で返す**（PPOは生成と更新が同一セッション内なので往復不要）

**勾配対象は maxCount==1 の決定**（ちょうど1つを選ぶカテゴリカル分布）。
実測分布では (1,1)=84.5% と (0,1)=9.5%（末尾のNULLを含めた n+1 択）で **合計94.0%**。
**maxCount>=2 は勾配対象外**（`grad=False`）——上位k個を決定的に取る仕様で joint action の
確率分布になっておらず、無理に確率化するより除外するほうが正直。
除外した決定も実行はするし軌跡にも残す（LSTMのステップ列を壊さないため）。

**強制手（選択肢1個かつminCount>0）ではLSTMを進めない**——094のmain.pyと同一規約。
"""

from __future__ import annotations

import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "arena"),
           os.path.join(_ROOT, "agents/094_yushin_nn")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _softmax(x):
    m = np.max(x)
    e = np.exp(x - m)
    return e / e.sum()


class Rollout:
    """1試合1プレイヤーぶんの決定列。"""

    __slots__ = ("board", "oppb", "logs", "prev", "ctx", "opt_cid", "opt_vec",
                 "opt_cls", "chosen", "logp", "grad", "turn", "reward")

    def __init__(self):
        for s in self.__slots__:
            setattr(self, s, [])
        self.reward = 0.0


def collect(policy, decks, n_games: int, seed_base: int, batch: int = 128,
            temperature: float = 1.0, ft=None, to_obs=None, verbose: bool = False):
    """自己対戦を回して PPO 用の軌跡を返す。

    policy: BatchLSTMPolicy / TorchBatchPolicy（n_slots >= 2*batch）
    decks:  (deck0, deck1)
    戻り: (rollouts, stats)  rollouts は Rollout のリスト（1試合につき2本）
    """
    import time
    from selfplay.vecenv import VecBattles
    import run_match as rm
    if ft is None:
        import feat_094 as ft
    if to_obs is None:
        from cg.api import to_observation_class as to_obs

    rng = np.random.default_rng(seed_base)
    d0, d1 = decks
    out: list[Rollout] = []
    # ent_sum / argmax_hits は温度の効き具合の診断用。**その場で記録する**
    # （後から state を再構成して計算し直すのは、状態の取り違えを招く。実際に1度踏んだ）
    stats = {"games": 0, "decisions": 0, "grad_decisions": 0, "forced": 0,
             "declines": 0, "p0_wins": 0, "steps": 0, "sec": 0.0,
             "ent_sum": 0.0, "argmax_hits": 0}
    t_start = time.perf_counter()
    done = 0
    while done < n_games:
        nb = min(batch, n_games - done)
        seeds = [rm.derive_engine_seed(seed_base, done + j) for j in range(nb)]
        vb = VecBattles([d0] * nb, [d1] * nb, seeds)
        policy.reset_all()
        prev = [[-1.0, -1.0, -1.0] for _ in range(2 * nb)]
        traj = [Rollout() for _ in range(2 * nb)]
        while vb.live:
            cur = vb.observations()
            acts = {}
            slots, boards, logs_l, prevs, ctxs, cids_l, vecs_l, cls_l, meta = \
                [], [], [], [], [], [], [], [], []
            for (bi, o) in cur:
                ob = to_obs(o)
                st, sel = ob.current, ob.select
                me = st.yourIndex
                n = len(sel.option)
                k = max(min(sel.maxCount or sel.minCount, n), sel.minCount)
                if n == 0:
                    acts[bi] = []
                    continue
                if n < 2 and sel.minCount != 0:
                    acts[bi] = list(range(min(max(1, k), n)))
                    stats["forced"] += 1
                    continue
                board, _ = ft.feat_vector(st, me)
                oppb, _ = ft.feat_vector(st, 1 - me)
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
                    cls = [ft.CLASS_ID.get(ft.option_class(st, sel, j, me), -1)
                           for j in range(n)]
                n_real = n
                if sel.minCount == 0:
                    cids = cids + [ft.NULL_CID]
                    vecs = vecs + [ft.null_option_vector()]
                    if cls is not None:
                        cls = cls + [-1]
                slots.append(2 * bi + me)
                boards.append(board)
                logs_l.append(lg)
                prevs.append(list(prev[2 * bi + me]))
                ctxs.append(ctx)
                cids_l.append(cids)
                vecs_l.append(vecs)
                cls_l.append(cls)
                meta.append((bi, me, n_real, k, cids, vecs, board, oppb, lg,
                             list(prev[2 * bi + me]), ctx, cls, st.turn,
                             int(sel.minCount), int(sel.maxCount or sel.minCount)))
            if slots:
                scs = policy.scores(slots, boards, logs_l, prevs, ctxs,
                                    cids_l, vecs_l, cls_l)
                for m, sc in zip(meta, scs):
                    (bi, me, n_real, k, cids, vecs, board, oppb, lg, prev_in,
                     ctx, cls, turn, mn, mx) = m
                    sc = np.asarray(sc, np.float64)
                    # **maxCount==1 ならカテゴリカル分布**。minCount==1 なら「n択」、
                    # minCount==0 なら末尾にNULLを足してあるので「n択+棄権 の n+1択」。
                    # どちらもちょうど1つを選ぶので勾配対象にできる。
                    # (0,1)は全決定の9.5%あり、EXP-094 v7で実装した棄権判断
                    # （ベンチを増やしてボスの指令の的を作らない）が含まれる。
                    # ここを外すと学習も探索もされない。
                    single = (mx == 1)
                    if single:
                        # **単一選択のみ確率的に選ぶ**（PPOの勾配対象）
                        p = _softmax(sc / max(1e-6, temperature))
                        a = int(rng.choice(len(p), p=p))
                        logp = float(np.log(max(p[a], 1e-12)))
                        grad = True
                        stats["ent_sum"] += float(-(p * np.log(p + 1e-12)).sum())
                        stats["argmax_hits"] += int(a == int(np.argmax(sc)))
                    else:
                        a = int(np.argmax(sc))
                        logp = 0.0
                        grad = False
                    r = traj[2 * bi + me]
                    r.board.append(board)
                    r.oppb.append(oppb)
                    r.logs.append(lg)
                    r.prev.append(prev_in)
                    r.ctx.append(ctx)
                    r.opt_cid.append(cids)
                    r.opt_vec.append(vecs)
                    r.opt_cls.append(cls)
                    r.chosen.append(a)
                    r.logp.append(logp)
                    r.grad.append(grad)
                    r.turn.append(turn)
                    stats["decisions"] += 1
                    stats["grad_decisions"] += int(grad)
                    # 実際の行動へ変換
                    if len(cids) > n_real and a == n_real:      # 棄権(NULL)
                        acts[bi] = []
                        prev[2 * bi + me] = [float(ft.NULL_CID), -1.0, -1.0]
                        stats["declines"] += 1
                    elif single:
                        acts[bi] = [a]
                        prev[2 * bi + me] = [float(cids[a]), vecs[a][0], vecs[a][1]]
                    else:
                        order = [i for i in np.argsort(-sc) if i < n_real]
                        pick = sorted(order[:max(1, min(k, n_real))])
                        acts[bi] = pick
                        prev[2 * bi + me] = [float(cids[pick[0]]),
                                             vecs[pick[0]][0], vecs[pick[0]][1]]
            vb.step(acts)
        for bi in range(nb):
            res = vb.results[bi]
            stats["steps"] += vb.steps[bi]
            if res not in (0, 1):
                continue                      # 引き分け/未完は捨てる
            stats["games"] += 1
            stats["p0_wins"] += int(res == 0)
            for me in (0, 1):
                r = traj[2 * bi + me]
                if r.chosen:
                    r.reward = 1.0 if res == me else -1.0
                    out.append(r)
        vb.close()
        done += nb
        if verbose:
            print(f"  {done}/{n_games}試合 経過{time.perf_counter()-t_start:.0f}秒",
                  flush=True)
    stats["sec"] = time.perf_counter() - t_start
    return out, stats
