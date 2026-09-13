"""ExItの生成と**同一の試合**を打ち直し、全決定を記録する（EXP-124）。

**なぜ要るか**: exit_gen が保存するのは抽出した局面（1試合6局面）だけで、
その局面に至る決定列を持っていない。ExIt補正を「長さ1の系列」として学習すると
LSTMがゼロ状態で入力され、**実状態と22.2%で判断が変わる**（実測）。
試合ごと記録して系列に埋め込めば、LSTMが本物の履歴で累積する。

**再現性**: exit_gen の run_batch と RNG の使い方を厳密に一致させる。
  random.Random(seed0+gi) / random.seed(seed0+gi) / battle_start_seeded(DA,DB,seed0+gi)
  主方策は T>0 のとき np.random.default_rng(seed0+gi+n) でサンプリング、
  相手は argmax。**ロールアウトを回さないだけ**で他は同一。
一致は「ExIt行の board が再生側の決定に現れるか」で検証する（照合率を出す）。

usage:
  uv run python selfplay/exit_replay.py --main agents/186_alakazam_v8 \\
      --opps agents/189_marnie_v8 ... --opp-weights 45 24 20 9 2 \\
      --games 700 --chunk 10 --seed 20260811 -w 6 --out data/nn/exit186_games.pkl
"""
from __future__ import annotations
import argparse, os, pickle, random, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO, os.path.join(REPO, "arena"), os.path.join(REPO, "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import selfplay.exit_gen as EG  # noqa: E402  _load/_feat/_k_of/SETUP_CTX を再利用

_W: dict = {}


def _init(main_dir, opp_dirs):
    _W["main"] = EG._load(main_dir)
    _W["opps"] = {d: EG._load(d) for d in opp_dirs}


def run_batch(task):
    """games 試合を打ち、**全決定**を記録して返す（ロールアウトなし）。"""
    (gi0, ngames, opp_dir, seed0, temp) = task
    import cg.game as cgg
    from cg.api import to_observation_class
    FT, NET, DA = _W["main"]; FTo, NETo, DB = _W["opps"][opp_dir]
    out = []
    for g in range(ngames):
        gi = gi0 + g
        rng = random.Random(seed0 + gi)          # noqa: F841  exit_gen と同じ消費をする
        random.seed(seed0 + gi)
        obs, _ = cgg.battle_start_seeded(DA, DB, seed0 + gi)
        NET.reset(); NETo.reset(); pv = [-1., -1., -1.]; pvo = [-1., -1., -1.]
        rec = []; n = 0
        while obs is not None and obs["current"]["result"] == -1 and n < 3000:
            ob = to_observation_class(obs); st = ob.current; sel = ob.select
            if sel is None or not sel.option:
                obs = cgg.battle_select([0]); n += 1; continue
            if st.yourIndex == 0:
                sc, b, lg, cids, vecs, cls, ctx = EG._feat(ob, FT, NET, pv)
                k = EG._k_of(sel, len(sel.option))
                if temp > 0:
                    p = np.exp((sc - sc.max()) / temp); p /= p.sum()
                    idx = sorted(rng.sample(range(len(sc)), 0) or
                                 list(np.random.default_rng(seed0 + gi + n).choice(
                                     len(p), size=k, replace=False, p=p)))
                else:
                    idx = [int(x) for x in np.argsort(-sc)[:k]]
                idx = [int(x) for x in idx]
                # **全決定を記録する**。採否は教師データ化の工程で決める。
                rec.append(dict(gi=gi, n=n, ctx=int(ctx), turn=int(st.turn),
                                board=list(b), logs=list(lg), prev=list(pv),
                                opt_cid=cids, opt_vec=vecs, opt_cls=cls,
                                chosen=list(idx),
                                mn=int(sel.minCount or 1), mx=int(sel.maxCount or 1)))
                pv = [float(cids[idx[0]]), vecs[idx[0]][0], vecs[idx[0]][1]]
            else:
                sc, _, _, c_, v_, _, _ = EG._feat(ob, FTo, NETo, pvo)
                idx = [int(x) for x in np.argsort(-sc)[:EG._k_of(sel, len(sel.option))]]
                pvo = [float(c_[idx[0]]), v_[idx[0]][0], v_[idx[0]][1]]
            obs = cgg.battle_select(idx); n += 1
        _r = (obs["current"]["result"] if obs is not None else -1)
        res = 1.0 if _r == 0 else 0.5 if _r == 2 else 0.0 if _r == 1 else None
        cgg.battle_finish()
        for x in rec:
            x["game_result"] = res; x["opp"] = opp_dir
        out += rec
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--main", required=True)
    ap.add_argument("--opps", nargs="+", required=True)
    ap.add_argument("--opp-weights", type=float, nargs="+", default=None, dest="opp_weights")
    ap.add_argument("--games", type=int, default=700, help="実際に再生する試合数")
    ap.add_argument("--total-games", type=int, default=0, dest="total_games",
                    help="**生成時の総試合数**。相手の割り当ては総チャンク数から決まるので、"
                         "一部だけ再生するときも生成時の値を渡す（0なら --games と同じ）")
    ap.add_argument("--chunk", type=int, default=10)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=20260811)
    ap.add_argument("-w", "--workers", type=int, default=6)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    # **チャンクへの相手割り当ては exit_gen と同一でなければならない**
    # （最大剰余法・決定的）。ずれると同じ (seed, gi) でも相手が違う試合になる。
    total = a.total_games or a.games
    nchunk = (total + a.chunk - 1) // a.chunk   # **生成時の総チャンク数**で割り当てる
    W = a.opp_weights or [1.0] * len(a.opps)
    tw = float(sum(W))
    exact = [nchunk * w / tw for w in W]
    alloc = [int(x) for x in exact]
    rem = sorted(range(len(W)), key=lambda i: -(exact[i] - alloc[i]))
    for i in range(nchunk - sum(alloc)):
        alloc[rem[i % len(rem)]] += 1
    order = [d for d, k in zip(a.opps, alloc) for _ in range(k)]
    print("■ 相手の配分（総{}試合={}chunk 基準）: ".format(total, nchunk) + " / ".join(
        f"{d.split('/')[-1]} {k}chunk({k/nchunk:.1%})" for d, k in zip(a.opps, alloc)), flush=True)

    tasks = []; gi = 0; ci = 0
    while gi < a.games:
        nb = min(a.chunk, a.games - gi)
        tasks.append((gi, nb, order[ci % len(order)], a.seed, a.temp)); ci += 1
        gi += nb
    print(f"■ 再生: {a.games}試合 / {a.workers}並列（ロールアウトなし）", flush=True)

    t0 = time.perf_counter(); res = []
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init,
                             initargs=(a.main, a.opps)) as ex:
        futs = [ex.submit(run_batch, t) for t in tasks]
        for i, f in enumerate(as_completed(futs)):
            try:
                res += f.result()
            except Exception as e:  # noqa: BLE001
                print(f"   chunk 失敗: {type(e).__name__}: {str(e)[:120]}", flush=True)
                continue
            el = time.perf_counter() - t0
            print(f"   {i+1}/{len(tasks)} chunk  決定{len(res):,}  {el:.0f}秒", flush=True)
    with open(a.out, "wb") as f:
        pickle.dump(dict(rows=res, main=a.main, opps=a.opps, seed=a.seed, temp=a.temp), f)
    print(f"\n保存 -> {a.out}  決定{len(res):,}  "
          f"試合{len({r['gi'] for r in res})}  {(time.perf_counter()-t0)/60:.1f}分", flush=True)


if __name__ == "__main__":
    main()
