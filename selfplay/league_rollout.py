"""リーグ用の異種相手ロールアウト（I-109 Stage 3）。**torch非依存**。

`rollout.py` は自己対戦専用（両席を同じ方策・同じ特徴器・同じデッキが駆動）だった。
リーグでは main が **別デッキ・別特徴器・別モデルの相手**と戦うので、
決定ごとに「その席の持ち主」を見て特徴器と方策を切り替える必要がある。

収集するのは **main の決定だけ**（相手の決定は学習データではない）。
ただしLSTMは相手側も進める必要がある（相手も系列方策）。

Criticの入力（`state_encoder.encode`）は**両プレイヤー対称・特権情報つき**なので、
main の決定時に1回だけ計算して軌跡に載せる。

方策オブジェクトに要求するインタフェース（numpy版 BatchLSTMPolicy / torch版ラッパ共通）:
    reset_all()
    scores(slots, boards, logs, prevs, ctxs, opt_cids, opt_vecs, opt_clss) -> list[np.ndarray]
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "arena")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _softmax(x, t):
    z = np.asarray(x, np.float64) / max(1e-6, t)
    z -= z.max()
    e = np.exp(z)
    return e / e.sum()


class Opponent:
    """リーグの相手1体。凍結モデル（学習しない）。

    kind: 'self'（main自身）/ 'past'（過去snapshot）/ 'sl'（SL初期）/ 'ext'（外部レプリカ）
    self の場合 policy は main と同一オブジェクトを渡す（スロットで席を分ける）。
    """

    def __init__(self, name, kind, policy, ft, deck, weight, temp=0.0):
        self.name = name
        self.kind = kind
        self.policy = policy
        self.ft = ft
        self.deck = deck
        self.weight = float(weight)
        self.temp = float(temp)   # >0 なら訓練時にサンプリング（丸暗記の防止）
        self.model_path = None    # 評価時に別インスタンスを作るためのnpzパス（任意）


class Traj:
    """main の1試合ぶんの決定列。Criticの入力も同時に載せる。"""

    __slots__ = ("board", "logs", "prev", "ctx", "opt_cid", "opt_vec", "opt_cls",
                 "chosen", "logp", "grad", "turn", "zone", "attr", "prog",
                 "reward", "opp", "gkey")

    def __init__(self, opp_name, gkey=-1):
        for s in self.__slots__:
            setattr(self, s, [])
        self.reward = 0.0
        self.opp = opp_name
        # **CRN群のID**。同じ gkey の Traj は配札・相手・先後が完全に同一。
        # criticなしの群相対advantage（R_i − 群平均）に使う。
        self.gkey = gkey


def _feat_for(ft, ob, sel, me):
    """その席の特徴器で1決定ぶんの入力を作る。"""
    st = ob.current
    n = len(sel.option)
    board, _ = ft.feat_vector(st, me)
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
    return board, lg, cids, vecs, cls, ctx, n_real


def collect(main_policy, main_ft, main_deck, opponents, n_games, seed_base,
            batch=64, temperature=1.0, rng=None, encode_state=None, verbose=False,
            group=1):
    """リーグのロールアウト。

    opponents: Opponent のリスト（weightで抽選）。'self' は main_policy を指す。
    encode_state: None でなければ (ptr, obs_dict) -> (zone, attr, prog) を呼んで
                  Criticの入力を記録する（state_encoder.encode を想定）。
    group: **CRN群のサイズ**（>1 で criticなしの群相対advantageが使える）。
           同じ engine seed・同じ相手・同じ先後で group 回だけ複製し、
           Traj.gkey に群のIDを入れる。**配札が全く同じ試合を並べる**ので、
           R_i − mean(R_群) を advantage にすればカードゲーム最大のノイズ源
           （配札）が全ターンで厳密に消える（GRPO/RLOO と同型）。
           criticの序盤AUCは0.599しかないので、そこを近似でなく厳密に潰せるのが利点。
    戻り: (trajs, stats)
    """
    from selfplay.vecenv import VecBattles
    import run_match as rm
    from cg.api import to_observation_class as to_obs

    rng = rng or np.random.default_rng(seed_base)
    w = np.array([o.weight for o in opponents], np.float64)
    w = w / w.sum()
    out: list[Traj] = []
    stats = {"games": 0, "decisions": 0, "grad_decisions": 0, "forced": 0,
             "declines": 0, "wins": 0, "sec": 0.0, "by_opp": {}}
    t0 = time.perf_counter()
    done = 0
    while done < n_games:
        nb = min(batch, n_games - done)
        # **群は同一シード・同一相手・同一先後で複製する**。1つでも違うと
        # 群平均が別条件の混合になり、baseline が意味を失う。
        n_uni = (nb + group - 1) // group
        uni_pick = rng.choice(len(opponents), size=n_uni, p=w)
        pick = np.repeat(uni_pick, group)[:nb]
        base_id = [(done // max(1, group)) + (j // group) for j in range(nb)]
        main_seat = [b % 2 for b in base_id]
        gkeys = list(base_id)
        decks0, decks1, seeds = [], [], []
        for j in range(nb):
            opp = opponents[pick[j]]
            if main_seat[j] == 0:
                decks0.append(main_deck)
                decks1.append(opp.deck)
            else:
                decks0.append(opp.deck)
                decks1.append(main_deck)
            seeds.append(rm.derive_engine_seed(seed_base, base_id[j]))
        vb = VecBattles(decks0, decks1, seeds)

        main_policy.reset_all()
        for o in opponents:
            if o.kind != "self":
                o.policy.reset_all()
        prev = [[-1.0, -1.0, -1.0] for _ in range(2 * nb)]
        traj = [Traj(opponents[pick[j]].name, gkeys[j]) for j in range(nb)]

        while vb.live:
            cur = vb.observations()
            acts = {}
            # 持ち主ごとに決定をまとめる（特徴器と方策が違うのでバッチも分ける）
            groups: dict[int, list] = {}
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
                is_main = (me == main_seat[bi])
                gid = -1 if is_main else int(pick[bi])
                groups.setdefault(gid, []).append((bi, ob, sel, me, k, o))

            for gid, items in groups.items():
                is_main = (gid == -1)
                opp = None if is_main else opponents[gid]
                ft = main_ft if is_main else opp.ft
                pol = main_policy if is_main else opp.policy
                slots, boards, logs_l, prevs, ctxs, cids_l, vecs_l, cls_l, meta = \
                    [], [], [], [], [], [], [], [], []
                for (bi, ob, sel, me, k, raw) in items:
                    board, lg, cids, vecs, cls, ctx, n_real = \
                        _feat_for(ft, ob, sel, me)
                    # **スロット規約**: main は席ごとに要る（自己対戦で両席がmain）ので
                    # 2*bi+me。相手は1試合に1席しか無いので bi。
                    slots.append(2 * bi + me if (is_main or opp.kind == "self") else bi)
                    boards.append(board)
                    logs_l.append(lg)
                    prevs.append(list(prev[2 * bi + me]))
                    ctxs.append(ctx)
                    cids_l.append(cids)
                    vecs_l.append(vecs)
                    cls_l.append(cls)
                    meta.append((bi, me, n_real, k, cids, vecs, board, lg, ctx,
                                 cls, ob, sel, raw))
                scs = pol.scores(slots, boards, logs_l, prevs, ctxs,
                                 cids_l, vecs_l, cls_l)
                for m, sc in zip(meta, scs):
                    (bi, me, n_real, k, cids, vecs, board, lg, ctx, cls,
                     ob, sel, raw) = m
                    sc = np.asarray(sc, np.float64)
                    mx = int(sel.maxCount or sel.minCount)
                    single = (mx == 1)
                    temp = temperature if is_main else (opp.temp if opp else 0.0)
                    if single and temp > 0:
                        p = _softmax(sc, temp)
                        a = int(rng.choice(len(p), p=p))
                        logp = float(np.log(max(p[a], 1e-12)))
                        grad = True
                    elif single:
                        a = int(np.argmax(sc))
                        logp = 0.0
                        grad = bool(is_main and temperature > 0)
                    else:
                        a = int(np.argmax(sc))
                        logp = 0.0
                        grad = False
                    if is_main:
                        r = traj[bi]
                        r.board.append(board)
                        r.logs.append(lg)
                        r.prev.append(list(prev[2 * bi + me]))
                        r.ctx.append(ctx)
                        r.opt_cid.append(cids)
                        r.opt_vec.append(vecs)
                        r.opt_cls.append(cls)
                        r.chosen.append(a)
                        r.logp.append(logp)
                        r.grad.append(grad and single)
                        r.turn.append(int(ob.current.turn))
                        if encode_state is not None:
                            z, at, pg = encode_state(vb.ptrs[bi], raw)
                            r.zone.append(z)
                            r.attr.append(at)
                            r.prog.append(pg)
                        stats["decisions"] += 1
                        stats["grad_decisions"] += int(grad and single)
                    if len(cids) > n_real and a == n_real:      # 棄権(NULL)
                        acts[bi] = []
                        # **その席の特徴器のNULL_CIDを使う**。現状は全特徴器が-3で
                        # 揃っているが、どれか1つずれたときに気付けない形にしない。
                        prev[2 * bi + me] = [float(ft.NULL_CID), -1.0, -1.0]
                        if is_main:
                            stats["declines"] += 1
                    elif single:
                        acts[bi] = [a]
                        prev[2 * bi + me] = [float(cids[a]), vecs[a][0], vecs[a][1]]
                    else:
                        order = [i for i in np.argsort(-sc) if i < n_real]
                        pickk = sorted(order[:max(1, min(k, n_real))])
                        acts[bi] = pickk
                        prev[2 * bi + me] = [float(cids[pickk[0]]),
                                             vecs[pickk[0]][0], vecs[pickk[0]][1]]
            vb.step(acts)

        for j in range(nb):
            res = vb.results[j]
            if res not in (0, 1):
                continue
            stats["games"] += 1
            r = traj[j]
            if not r.chosen:
                continue
            won = (res == main_seat[j])
            r.reward = 1.0 if won else -1.0
            stats["wins"] += int(won)
            nm = r.opp
            s = stats["by_opp"].setdefault(nm, [0, 0])
            s[0] += 1
            s[1] += int(won)
            out.append(r)
        vb.close()
        done += nb
        if verbose:
            print(f"  {done}/{n_games}試合 {time.perf_counter()-t0:.0f}秒", flush=True)
    stats["sec"] = time.perf_counter() - t0
    return out, stats
