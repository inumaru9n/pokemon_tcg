"""模倣レプリカの**実プレイの行動分布**を教師のそれと突合する（EXP-096の診断の再実装）。

模倣学習の covariate shift は「自分の下手な手で教師が到達しない盤面に入り、そこでは
常に選べる安全な選択（多くは `end`）が最高スコアになる」形で出る。**行動分布は教師と
ほぼ一致するのに `end` だけが突出する**のが典型で、膨張量とレプリカの弱さが対応する
（EXP-096実測: 098 +0.9pt→random 99.5% / 100 +8.4pt→random 70.5%）。

val一致率では検出できない（valは教師の盤面上でしか測らないため）。

usage:
  uv run tools/act_freq.py --agent agents/156_kangaskhan_jcox --npz data/nn/kanga156.npz \
      --opp agents/101_marnie_luca -n 60
"""
from __future__ import annotations
import argparse, collections, os, sys
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO, os.path.join(REPO, "arena")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _load(d):
    import importlib.util as ilu
    import run_match as rm
    ad = os.path.join(REPO, d)
    fp = [f for f in os.listdir(ad) if f.startswith("feat_") and f.endswith(".py")][0]
    npf = [f for f in os.listdir(ad) if f.startswith("nn_") and f.endswith(".py")][0]
    mp = [f for f in os.listdir(ad)
          if f.startswith("model_") and f.endswith(".npz") and "_v" not in f][0]
    mods = {}
    for f in (fp, npf):
        sp = ilu.spec_from_file_location(f[:-3], os.path.join(ad, f))
        m = ilu.module_from_spec(sp); sys.modules[f[:-3]] = m; sp.loader.exec_module(m)
        mods[f[:-3]] = m
    return (mods[fp[:-3]], mods[npf[:-3]].LSTMPolicy(os.path.join(ad, mp)),
            rm.read_deck(ad))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--agent", required=True)
    ap.add_argument("--npz", required=True, help="教師データ（分布の基準）")
    ap.add_argument("--opp", required=True)
    ap.add_argument("-n", "--games", type=int, default=60)
    ap.add_argument("--seed", type=int, default=7000)
    ap.add_argument("--top", type=int, default=14)
    a = ap.parse_args()

    import cg.game as cgg
    from cg.api import to_observation_class

    FT, NET, DA = _load(a.agent)
    FTo, NETo, DB = _load(a.opp)
    CLS = list(FT.CLASSES)

    # ---- 教師の分布（MAIN決定で選ばれた行動クラス）----
    z = np.load(a.npz, allow_pickle=True)
    m = z["ctx"] == 0
    tcnt = collections.Counter()
    for cl, ch in zip(z["opt_cls"][m], z["chosen"][m]):
        for i in np.nonzero(ch)[0]:
            c = int(cl[i])
            if 0 <= c < len(CLS):
                tcnt[CLS[c]] += 1
    tot_t = sum(tcnt.values())

    # ---- 実プレイの分布 ----
    pcnt = collections.Counter()
    for g in range(a.games):
        obs, _ = cgg.battle_start_seeded(DA, DB, a.seed + g)
        NET.reset(); NETo.reset()
        pv = [-1., -1., -1.]; pvo = [-1., -1., -1.]
        n = 0
        while obs is not None and obs["current"]["result"] == -1 and n < 3000:
            ob = to_observation_class(obs); st = ob.current; sel = ob.select
            if sel is None or not sel.option:
                obs = cgg.battle_select([0]); n += 1; continue
            me = st.yourIndex
            ft, net, prev = (FT, NET, pv) if me == 0 else (FTo, NETo, pvo)
            b, _ = ft.feat_vector(st, me); lg = ft.logs_vector(ob, me)
            hc, pl = ft.pool_counts(st, me)
            cids, vecs = [], []
            for opt in sel.option:
                c, v = ft.option_vector(ob, opt, me, hc, pl)
                cids.append(c); vecs.append(v)
            ctx = int(sel.context)
            cls = ([ft.CLASSES.index(ft.option_class(st, sel, i, me))
                    for i in range(len(sel.option))] if ctx == 0 else None)
            sc = np.asarray(net.scores(b, lg, prev, ctx, cids, vecs, cls), np.float64)
            mn = int(sel.minCount if sel.minCount is not None else 1)
            mx = int(sel.maxCount if sel.maxCount is not None else mn or 1)
            k = max(1, min(max(mn, min(mx if mx else mn, len(sc))), len(sc)))
            idx = [int(x) for x in np.argsort(-sc)[:k]]
            if me == 0 and ctx == 0 and cls is not None:
                pcnt[ft.CLASSES[cls[idx[0]]]] += 1
            nv = [float(cids[idx[0]]), vecs[idx[0]][0], vecs[idx[0]][1]]
            if me == 0: pv = nv
            else: pvo = nv
            obs = cgg.battle_select(idx); n += 1
        cgg.battle_finish()
    tot_p = sum(pcnt.values())

    print(f"■ MAIN行動クラスの分布   実プレイ {tot_p}決定（{a.games}試合 vs "
          f"{os.path.basename(a.opp)}） / 教師 {tot_t}決定")
    print(f"{'class':30}{'実プレイ':>10}{'教師':>10}{'差':>9}")
    keys = sorted(set(pcnt) | set(tcnt), key=lambda k: -(pcnt[k] / max(tot_p, 1)))
    worst = None
    for k in keys[:a.top]:
        pp, tt = pcnt[k] / max(tot_p, 1), tcnt[k] / max(tot_t, 1)
        d = (pp - tt) * 100
        if worst is None or abs(d) > abs(worst[1]): worst = (k, d)
        print(f"   {k[:27]:30}{pp:>9.1%}{tt:>10.1%}{d:>+9.1f}")
    e_p, e_t = pcnt.get("end", 0) / max(tot_p, 1), tcnt.get("end", 0) / max(tot_t, 1)
    print(f"\n   **end: 実プレイ {e_p:.1%} vs 教師 {e_t:.1%} → 膨張 {(e_p-e_t)*100:+.1f}pt**")
    print(f"   （EXP-096: 098=+0.9pt→random 99.5% / 100=+8.4pt→random 70.5%）")
    print(f"   最大の乖離: {worst[0]} {worst[1]:+.1f}pt")


if __name__ == "__main__":
    main()
