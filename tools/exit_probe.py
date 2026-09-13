"""ExIt Phase 0: 実施可否を数字で確定する（EXP-115）。

測ること:
  ① **候補間で勝率が割れる局面の割合**（撤退基準: 1%未満）
  ② 選別基準ごとの割れる率（top1-top2 / entropy / 終盤 / 負け試合）
  ③ M=32 で決着する割合（適応打ち切りの実効コスト）
  ④ 較正（実戦の結果 R vs 推定 Q）
  ⑤ スループット

usage:
  uv run python tools/exit_probe.py --games 20 --positions 40 -M 32
"""
from __future__ import annotations
import argparse, collections, importlib.util as ilu, os, random, sys, time
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO, os.path.join(REPO, "arena"), os.path.join(REPO, "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def load_agent(d):
    """agents/<dir> から (featurizer, policy, deck) を読む。"""
    import run_match as rm
    ad = os.path.join(REPO, d)
    fp = [f for f in os.listdir(ad) if f.startswith("feat_") and f.endswith(".py")][0]
    np_ = [f for f in os.listdir(ad) if f.startswith("nn_") and f.endswith(".py")][0]
    mp = [f for f in os.listdir(ad)
          if f.startswith("model_") and f.endswith(".npz") and "_v" not in f][0]
    mods = {}
    for f in (fp, np_):
        sp = ilu.spec_from_file_location(f[:-3], os.path.join(ad, f))
        m = ilu.module_from_spec(sp); sys.modules[f[:-3]] = m; sp.loader.exec_module(m)
        mods[f[:-3]] = m
    ft = mods[fp[:-3]]; net = mods[np_[:-3]].LSTMPolicy(os.path.join(ad, mp))
    return ft, net, rm.read_deck(ad)



def make_cands(sc, sel, ncand):
    """min/max を満たす**選択集合**の候補を作る（EXP-115）。

    --all-ctx にすると minCount>1 / maxCount>1 の決定（DISCARDで複数枚など）が入る。
    候補を1個だけ渡すと `minCount <= len(select) <= maxCount` に反して落ちる。
    k = clamp(maxCount or minCount, minCount, len(option)) 枚を選ぶ組を作る:
      候補0 = 上位k個（=方策の選択そのもの）
      候補i = 上位k個のうち**最下位を i 番目の次点と入れ替えたもの**
    """
    n = len(sel.option)
    mn = int(sel.minCount if sel.minCount is not None else 1)
    mx = int(sel.maxCount if sel.maxCount is not None else mn or 1)
    k = max(mn, min(mx if mx else mn, n))
    k = max(1, min(k, n))
    order = [int(x) for x in np.argsort(-sc)]
    base = sorted(order[:k])
    out = [base]
    for j in range(k, min(n, k + ncand - 1)):
        alt = sorted(order[:k-1] + [order[j]])
        if alt not in out:
            out.append(alt)
    return out[:ncand]

def scores_of(ob, ft, net, prev):
    """1決定ぶんのスコア列と、次の prev を返す。"""
    st = ob.current; my = st.yourIndex; s = ob.select
    b, _ = ft.feat_vector(st, my); lg = ft.logs_vector(ob, my)
    hc, pl = ft.pool_counts(st, my)
    cids, vecs = [], []
    for opt in s.option:
        c, v = ft.option_vector(ob, opt, my, hc, pl)
        cids.append(c); vecs.append(v)
    ctx = int(s.context)
    cls = ([ft.CLASSES.index(ft.option_class(st, s, i, my)) for i in range(len(s.option))]
           if ctx == 0 else None)
    sc = np.asarray(net.scores(b, lg, prev, ctx, cids, vecs, cls), np.float64)
    return sc, cids, vecs


def pick(sc, sel):
    k = max(1, min(sel.maxCount or sel.minCount or 1, len(sel.option)))
    return [int(x) for x in np.argsort(-sc)[:k]]


def softmax(sc):
    e = np.exp(sc - sc.max()); return e / e.sum()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--main", default="agents/104_alakazam_feat4")
    ap.add_argument("--opp", default="agents/101_marnie_luca")
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--positions", type=int, default=40, help="評価する局面数")
    ap.add_argument("-M", type=int, default=32)
    ap.add_argument("--cands", type=int, default=3)
    ap.add_argument("--all-ctx", action="store_true", dest="all_ctx",
                    help="MAIN以外のコンテキストも対象にする（TO_HAND等は決定の40%）")
    ap.add_argument("--select", default="lowgap",
                    choices=("lowgap", "highgap", "random"),
                    help="局面の選び方。gapの予測力を測るには random で全域を見る")
    ap.add_argument("--seed", type=int, default=7000)
    a = ap.parse_args()

    import cg.game as cgg
    from cg.api import to_observation_class, search_begin, search_step, search_end
    from selfplay.state_encoder import _hidden

    FT, NET, DA = load_agent(a.main)
    FTo, NETo, DB = load_agent(a.opp)

    def rollout(o, h0, h1, me, first, st_me=None, st_op=None,
                prev_me=None, prev_op=None):
        """クローンして first を打ち、両者argmaxで終局まで。勝敗を返す。"""
        # **相手アクティブが伏せ状態なら、そのIDも渡す必要がある**（試合開始直後）。
        # --all-ctx で SETUP 系の局面が入ると発生する。真の値を GetHiddenData から取る。
        _oa = []
        _op = o.current.players[1 - me]
        if not (_op.active and _op.active[0]):
            _oa = [int((h1.get("active") or [0])[0])] if h1.get("active") else \
                  [int(sorted(FT.DECK_COUNTS)[0])]
        r = search_begin(o, list(h0.get("deck") or []), list(h0.get("prize") or []),
                         list(h1.get("deck") or []), list(h1.get("prize") or []),
                         list(h1.get("hand") or []), _oa)
        ch = search_step(r.searchId, first)
        # **局面sまでのLSTM状態を復元する**（reset だと記憶を失った方策で評価してしまう）
        if st_me is not None: NET.set_state(st_me)
        else: NET.reset()
        if st_op is not None: NETo.set_state(st_op)
        else: NETo.reset()
        pv = list(prev_me) if prev_me else [-1.,-1.,-1.]
        pvo = list(prev_op) if prev_op else [-1.,-1.,-1.]
        for _ in range(1500):
            ob = ch.observation; st = ob.current
            if st.result != -1:
                search_end(); return 1.0 if st.result == me else (0.5 if st.result == 2 else 0.0)
            s = ob.select
            if s is None or not s.option:
                search_end(); return 0.5
            if st.yourIndex == me:
                sc, cids, vecs = scores_of(ob, FT, NET, pv); idx = pick(sc, s)
                pv = [float(cids[idx[0]]), vecs[idx[0]][0], vecs[idx[0]][1]]
            else:
                sc, cids, vecs = scores_of(ob, FTo, NETo, pvo); idx = pick(sc, s)
                pvo = [float(cids[idx[0]]), vecs[idx[0]][0], vecs[idx[0]][1]]
            ch = search_step(ch.searchId, idx)
        search_end(); return 0.5

    # ---- 局面を集める（実戦は両者argmax。ロールアウトと条件を揃える）----
    pool = []
    t0 = time.perf_counter()
    for g in range(a.games):
        random.seed(g)
        obs, _ = cgg.battle_start_seeded(DA, DB, a.seed + g)
        ptr = cgg.Battle.battle_ptr
        snaps = []; pv = [-1.,-1.,-1.]; pvo = [-1.,-1.,-1.]; n = 0
        while obs is not None and obs["current"]["result"] == -1 and n < 3000:
            ob = to_observation_class(obs); st = ob.current; s = ob.select
            if s is None or not s.option:
                obs = cgg.battle_select([0]); n += 1; continue
            if st.yourIndex == 0:
                sc, cids, vecs = scores_of(ob, FT, NET, pv); idx = pick(sc, s)
                _mn = int(s.minCount if s.minCount is not None else 1)
                # SETUP系（試合開始の配置）は相手アクティブが伏せで、
                # 分岐しても本番の判断とは性質が違うため対象外にする
                _setup = int(s.context) in (1, 2, 41)
                if ((a.all_ctx or int(s.context) == 0) and not _setup
                        and len(s.option) >= 2 and _mn <= len(s.option)):
                    p = softmax(sc)
                    o2 = np.sort(p)[::-1]
                    # **LSTM状態を保存する**。ロールアウトでリセットすると
                    # 局面sまでの記憶を失った弱い方策で評価してしまう
                    # （EXP-094: サブ選択でLSTM状態の効果は TO_HAND +11.0 / TO_ACTIVE +12.8pt）
                    snaps.append(dict(ctx=int(s.context),
                                      st_me=NET.get_state(), st_op=NETo.get_state(),
                                      prev_me=list(pv), prev_op=list(pvo),
                                      ob=ob, h0=_hidden(ptr, 0), h1=_hidden(ptr, 1),
                                      gap=float(o2[0]-o2[1]),
                                      ent=float(-(p*np.log(p+1e-12)).sum()),
                                      turn=int(st.turn), top=make_cands(sc, s, a.cands),
                                      k=len(idx)))
                pv = [float(cids[idx[0]]), vecs[idx[0]][0], vecs[idx[0]][1]]
            else:
                sc, cids, vecs = scores_of(ob, FTo, NETo, pvo); idx = pick(sc, s)
                pvo = [float(cids[idx[0]]), vecs[idx[0]][0], vecs[idx[0]][1]]
            obs = cgg.battle_select(idx); n += 1
        R = 1.0 if (obs or {}).get("current", {}).get("result", -1) == 0 else 0.0
        for i, sn in enumerate(snaps):
            sn["R"] = R; sn["frac"] = i / max(1, len(snaps)-1)
        pool += snaps
        cgg.battle_finish()
    t_gen = time.perf_counter() - t0
    print(f"■ 局面収集: {a.games}試合 → {len(pool)}局面（{t_gen:.0f}秒）")
    print(f"   top1-top2 の分布: 中央{np.median([s['gap'] for s in pool]):.3f} "
          f"/ <0.2 が {np.mean([s['gap']<0.2 for s in pool]):.1%}")

    # ---- 評価する局面を選ぶ（gapが小さい順＝方策が迷っている）----
    # **選別方式**: gap の予測力を検証するには、gapが小さい局面だけを見てはいけない
    # （範囲制限で「差がない」と誤診する。Phase 0 初回で実際に踏んだ）。
    rng = np.random.default_rng(0)
    if a.select == "lowgap":
        pool.sort(key=lambda s: s["gap"]); sel = pool[:a.positions]
    elif a.select == "highgap":
        pool.sort(key=lambda s: -s["gap"]); sel = pool[:a.positions]
    else:                                   # random
        sel = [pool[i] for i in rng.permutation(len(pool))[:a.positions]]
    print(f"\n■ 評価[{a.select}]: {len(sel)}局面 × 候補{a.cands} × M={a.M if hasattr(a,'M') else a.M}")
    t0 = time.perf_counter()
    rows = []
    for si, sn in enumerate(sel):
        o = sn["ob"]; me = o.current.yourIndex
        Q = np.zeros((len(sn["top"]), a.M))
        for m in range(a.M):
            for ci, c in enumerate(sn["top"]):
                Q[ci, m] = rollout(o, sn["h0"], sn["h1"], me, list(c),
                                   sn.get("st_me"), sn.get("st_op"),
                                   sn.get("prev_me"), sn.get("prev_op"))
        q = Q.mean(1)
        # M=32 の時点で有意差があるか（対応のある比較）
        d = Q[0] - Q[1:].max(0) if Q.shape[0] > 1 else np.zeros(a.M)
        se = d.std(ddof=1)/np.sqrt(a.M) if a.M > 1 else 1.0
        rows.append(dict(ctx=sn.get("ctx", 0),
                         gap=sn["gap"], ent=sn["ent"], turn=sn["turn"], R=sn["R"],
                         frac=sn["frac"], q=q, split=float(q.max()-q.min()),
                         se=float(se), best=int(np.argmax(q))))
        if (si+1) % 10 == 0:
            print(f"   {si+1}/{len(sel)} 完了（{time.perf_counter()-t0:.0f}秒）", flush=True)
    dt = time.perf_counter() - t0
    print(f"\n   所要 {dt:.0f}秒 → **{dt/len(sel):.1f}秒/局面**（1コア）")

    sp = np.array([r["split"] for r in rows])
    print(f"\n■ ① 候補間で勝率が割れた局面")
    for th in (0.0, 0.05, 0.10, 0.20):
        print(f"   差 > {th:.2f}: {np.mean(sp > th):.1%}")
    print(f"   差の中央値 {np.median(sp):.3f} / 平均 {sp.mean():.3f}")
    print(f"\n■ ⑥ best が top1 でない局面: {np.mean([r['best']!=0 for r in rows]):.1%}")
    R = np.array([r["R"] for r in rows]); Q0 = np.array([r["q"][0] for r in rows])
    print(f"\n■ ④ 較正: mean(R)={R.mean():.3f} vs mean(Q_top1)={Q0.mean():.3f} "
          f"差{Q0.mean()-R.mean():+.3f} / 相関 r={np.corrcoef(R,Q0)[0,1]:.3f}")
    # **gap帯ごとの「勝率に差がある率」**。これが選別の可否を直接決める。
    # 「方策が確信している局面（gap大）では別候補が勝つ確率は低いはず」という仮説を検証する。
    gg = np.array([r["gap"] for r in rows])
    print(f"\n■ ★ gap帯ごとの内訳（**計算資源をどこに集中すべきか**）")
    print(f"   {'gap帯':<14}{'局面数':>6}{'差>0.10':>9}{'差の平均':>9}{'bestがtop1でない':>16}")
    bins=[(0.0,0.2),(0.2,0.5),(0.5,0.8),(0.8,0.95),(0.95,1.01)]
    for lo,hi in bins:
        m=(gg>=lo)&(gg<hi)
        if m.sum()==0: continue
        nb=np.mean([rows[i]["best"]!=0 for i in range(len(rows)) if m[i]])
        print(f"   [{lo:.2f},{hi:.2f})    {m.sum():>6}{np.mean(sp[m]>0.10):>8.1%}"
              f"{sp[m].mean():>9.3f}{nb:>15.1%}")
    print(f"\n■ ★ コンテキスト別（**MAIN以外にも改善余地があるか**）")
    cx = np.array([r["ctx"] for r in rows])
    try:
        from cg.api import SelectContext as _SC
        nm_ = lambda c: _SC(c).name
    except Exception:
        nm_ = lambda c: str(c)
    print(f"   {'ctx':<22}{'局面数':>6}{'差>0.10':>9}{'差の平均':>9}{'best≠top1':>11}")
    for c in sorted(set(cx.tolist())):
        m = cx == c
        nb = np.mean([rows[i]["best"] != 0 for i in range(len(rows)) if m[i]])
        try: label = f"{c} {nm_(c)}"
        except Exception: label = str(c)
        print(f"   {label[:21]:<22}{m.sum():>6}{np.mean(sp[m]>0.10):>8.1%}"
              f"{sp[m].mean():>9.3f}{nb:>10.1%}")
    print(f"\n■ ② 選別基準との関係（差>0.10 を『割れた』とする）")
    big = sp > 0.10
    for nm, v in (("gap(top1-top2)", np.array([r["gap"] for r in rows])),
                  ("entropy", np.array([r["ent"] for r in rows])),
                  ("turn", np.array([r["turn"] for r in rows])),
                  ("試合内の位置", np.array([r["frac"] for r in rows])),
                  ("負け試合(R=0)", 1.0-R)):
        if big.sum() and (~big).sum():
            print(f"   {nm:<16} 割れた側 {v[big].mean():.3f} / 割れない側 {v[~big].mean():.3f}")


if __name__ == "__main__":
    main()
