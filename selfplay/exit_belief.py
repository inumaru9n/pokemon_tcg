"""隠れ情報を**推測**したときにExItの利得が何割残るかを測る（EXP-125 / キルゲート）。

**なぜ要るか**: exit_gen のロールアウトは `_hidden(ptr, i)` で相手の手札・山・サイドを
**真値**で渡している。実測した「best に替えると +9.45pt」はその条件下の数字であり、
本番の推論時には推測するしかない（本番観測の `search_begin_input` の引数がすべて
"Predicted Card ID" なのはそのため）。**推測に置き換えて何割残るか**が、
ExItを推論時探索に変えられるかの唯一の分岐点。

測り方（選択バイアスが原理的に入らない形にする）:
  1. **推測した隠れ情報**でロールアウトし直して best' を選ぶ
  2. その best' を、生成時に保存済みの**真値ロールアウト行列 `R`（候補×24）で評価**する
  → 選択（推測標本）と評価（真値標本）が独立なので、二段構えの分割すら要らない

比較対象（真値で選んだ場合の実力）は同じ行集合に対する**前半12本で選び後半12本で測る**値。

**エージェントが正当に知っていること**だけから推測を組む:
  自分側 = deck+prize の**多重集合は既知**（自分の60枚と見えている札から引ける）が、
          **どれがサイドかは不明** → 分割をサンプル
  相手側 = アーキタイプが同定できれば残りの多重集合が引ける（hand+deck+prize）が、
          **手札／山／サイドの分け方は不明** → 3分割をサンプル
これは**相手のリスト読みが完璧な場合の上限**である。ここで落ちるなら推論時探索は死ぬ。

usage:
  uv run python selfplay/exit_belief.py --exit data/nn/exit186.pkl \\
      --main agents/186_alakazam_v8 --n 600 -w 8 --M 24 --out data/nn/belief186.pkl
"""
from __future__ import annotations

import argparse
import os
import pickle
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO, os.path.join(REPO, "arena"), os.path.join(REPO, "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import selfplay.exit_gen as EG  # noqa: E402

_W: dict = {}


def _init(main_dir, opp_dirs):
    _W["main"] = EG._load(main_dir)
    _W["opps"] = {d: EG._load(d) for d in opp_dirs}


def run_chunk(task):
    """局面群について、推測した隠れ情報で best' を選ぶ。"""
    rows, seed0, M, mode = task
    from cg.api import to_observation_class, search_begin, search_step, search_end
    FT, NET, _ = _W["main"]
    out = []

    def roll_from(ch, me, st_me, st_op, pv0, pvo0, FTo, NETo):
        """exit_gen.run_batch の同名関数と同一（search_end は呼び出し側で1回だけ）。"""
        NET.set_state(st_me); NETo.set_state(st_op)
        pv = list(pv0); pvo = list(pvo0)
        for _ in range(1500):
            o2 = ch.observation; s2 = o2.current
            if s2.result != -1:
                return 1.0 if s2.result == me else (0.5 if s2.result == 2 else 0.0)
            sel = o2.select
            if sel is None or not sel.option:
                return 0.5
            if s2.yourIndex == me:
                sc, _, _, c_, v_, _, _ = EG._feat(o2, FT, NET, pv)
                idx = [int(x) for x in np.argsort(-sc)[:EG._k_of(sel, len(sel.option))]]
                pv = [float(c_[idx[0]]), v_[idx[0]][0], v_[idx[0]][1]]
            else:
                sc, _, _, c_, v_, _, _ = EG._feat(o2, FTo, NETo, pvo)
                idx = [int(x) for x in np.argsort(-sc)[:EG._k_of(sel, len(sel.option))]]
                pvo = [float(c_[idx[0]]), v_[idx[0]][0], v_[idx[0]][1]]
            ch = search_step(ch.searchId, idx)
        return 0.5

    for ri, r in rows:
        try:
            ob = to_observation_class(r["obs"])
            me = ob.current.yourIndex
            FTo, NETo, _ = _W["opps"][r["opp"]]
            h0, h1 = r["h0"], r["h1"]
            # **エージェントが知っている多重集合**と、**知らない分割の大きさ**
            pool0 = list(h0["deck"]) + list(h0["prize"])
            n0d, n0p = len(h0["deck"]), len(h0["prize"])
            pool1 = list(h1["deck"]) + list(h1["prize"]) + list(h1["hand"])
            n1d, n1p, n1h = len(h1["deck"]), len(h1["prize"]), len(h1["hand"])
            cands = r["cands"]
            Q = np.zeros((len(cands), M))
            rng = random.Random(seed0 + ri)
            for m in range(M):
                if mode == "true":            # 対照: 真値をそのまま渡す（配管の検証用）
                    d0, p0 = list(h0["deck"]), list(h0["prize"])
                    d1, p1, hh1 = list(h1["deck"]), list(h1["prize"]), list(h1["hand"])
                else:
                    a = list(pool0); rng.shuffle(a)
                    d0, p0 = a[:n0d], a[n0d:n0d + n0p]
                    b = list(pool1); rng.shuffle(b)
                    hh1, d1, p1 = b[:n1h], b[n1h:n1h + n1d], b[n1h + n1d:n1h + n1d + n1p]
                # **1回の search_begin から全候補を分岐させる**（CRN。exit_gen と同じ）
                rt = search_begin(ob, d0, p0, d1, p1, hh1, [])
                for ci, c in enumerate(cands):
                    ch0 = search_step(rt.searchId, list(c))
                    Q[ci, m] = roll_from(ch0, me, r["st_me"], r["st_op"],
                                         r["prev"], r["pvo"], FTo, NETo)
                search_end()
            out.append((ri, Q.astype(np.float32)))
        except Exception as _e:               # noqa: BLE001
            try:
                search_end()
            except Exception:                 # noqa: BLE001
                pass
            out.append((ri, None))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exit", required=True)
    ap.add_argument("--main", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=600, help="測る局面数（M=24の行から抽出）")
    ap.add_argument("--M", type=int, default=24, help="推測側のロールアウト本数")
    ap.add_argument("--mode", default="belief", choices=("belief", "true"))
    ap.add_argument("--chunk", type=int, default=10)
    ap.add_argument("-w", "--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260812)
    a = ap.parse_args()

    E = pickle.load(open(a.exit, "rb"))["rows"]
    # **M=24 まで昇格した行だけ**（M=8 は「全候補同結果」か「決着済み」で打ち切られている）
    idx = [i for i, r in enumerate(E) if r["M"] >= 24 and len(r["cands"]) >= 2]
    rng = np.random.default_rng(a.seed)
    sel = sorted(rng.choice(idx, size=min(a.n, len(idx)), replace=False).tolist())
    opps = sorted({E[i]["opp"] for i in sel})
    print(f"■ 対象 {len(sel)}局面 / M={a.M} / mode={a.mode} / 相手{len(opps)}種", flush=True)
    print(f"   推定コスト {len(sel)*a.M*2.8*0.85/a.workers/60:.0f}分"
          f"（1ロールアウト0.85秒・{a.workers}並列）", flush=True)

    tasks = [([(i, E[i]) for i in sel[s:s + a.chunk]], a.seed, a.M, a.mode)
             for s in range(0, len(sel), a.chunk)]
    t0 = time.perf_counter(); res = {}
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init,
                             initargs=(a.main, opps)) as ex:
        futs = [ex.submit(run_chunk, t) for t in tasks]
        for k, f in enumerate(as_completed(futs)):
            for ri, Q in f.result():
                res[ri] = Q
            el = time.perf_counter() - t0
            print(f"   {k+1}/{len(tasks)}  {len(res)}局面  {el/60:.1f}分 "
                  f"(残り {el/max(1,k+1)*(len(tasks)-k-1)/60:.0f}分)", flush=True)
    with open(a.out, "wb") as f:
        pickle.dump(dict(Q=res, exit=a.exit, mode=a.mode, M=a.M, sel=sel), f)
    print(f"\n保存 -> {a.out}  {(time.perf_counter()-t0)/60:.1f}分", flush=True)

    # ---- 判定 ----
    ok = {i: q for i, q in res.items() if q is not None}
    print(f"\n■ 成功 {len(ok)}/{len(sel)}局面")
    gb, gt, agree = [], [], 0
    for i, Q in ok.items():
        R = np.asarray(E[i]["R"], float)      # 真値ロールアウト（候補 × 24）
        bb = int(np.argmax(Q.mean(1)))        # 推測で選んだ手
        # 推測標本と真値標本は独立 → R 全体で評価してよい（選択バイアスなし）
        gb.append(R[bb].mean() - R[0].mean())
        h = R.shape[1] // 2                   # 真値で選ぶ場合の実力（前半で選び後半で測る）
        gt.append(R[:, h:][int(np.argmax(R[:, :h].mean(1)))].mean() - R[:, h:][0].mean())
        agree += int(bb == int(np.argmax(R.mean(1))))
    gb, gt = np.array(gb), np.array(gt)
    ci = lambda x: 1.96 * x.std(ddof=1) / np.sqrt(len(x))  # noqa: E731
    print(f"\n{'':34}{'1局面あたりの gap':>18}")
    print(f"   真値で選ぶ（前半で選び後半で測る）  **{gt.mean():+.4f}** ±{ci(gt):.4f}")
    print(f"   **推測で選ぶ（本番で取れる値）**     **{gb.mean():+.4f}** ±{ci(gb):.4f}")
    print(f"\n   **残存率 {100*gb.mean()/max(1e-9, gt.mean()):.0f}%**"
          f" / 推測と真値の best 一致率 {100*agree/len(ok):.1f}%")


if __name__ == "__main__":
    main()
