"""ExIt Phase 1b: 保存済み局面を**追加ロールアウトで測り直す**（EXP-115）。

生成時に observation / 隠れ情報 / LSTM状態を保存してあるので、**ゲームを打ち直さずに**
同じ局面の M を増やせる。新規生成に比べて桁違いに安い（実測: 同じ1.5時間で
再評価は350〜500ラベル、新規生成は約28ラベル）。

**検定は追加分のみで行う**。再評価の対象は「最初の24本で best≠top1 だった局面」
＝最初の24本はその候補に有利な方向へ選択されているので、96本をまとめて検定すると
オプショナル・ストッピングと同じ偏りが入る。追加分は選択と独立なので偏らない。

usage:
  uv run python selfplay/exit_reeval.py --pkl data/nn/exit104_v2.pkl \
      --out data/nn/exit104_v2r.pkl --add 72 -w 6
"""
from __future__ import annotations
import argparse, os, pickle, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO, os.path.join(REPO, "arena"), os.path.join(REPO, "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import selfplay.exit_gen as EG  # noqa: E402  _load/_feat/_k_of を再利用

_W: dict = {}


def _init(main_dir, opp_dirs):
    _W["main"] = EG._load(main_dir)
    _W["opps"] = {d: EG._load(d) for d in opp_dirs}


def work(task):
    """1ワーカーが担当する行を再評価し、(index, 追加結果の行列) を返す。"""
    idxs, rows, add = task
    from cg.api import to_observation_class, search_begin, search_step, search_end
    out = []
    for i, r in zip(idxs, rows):
        try:
            FT, NET, _ = _W["main"]
            FTo, NETo, _ = _W["opps"][r["opp"]]
            ob = to_observation_class(r["obs"])
            me = int(ob.current.yourIndex)
            cands = r["cands"]
            new = np.zeros((len(cands), add), np.float32)

            def roll(first):
                """exit_gen.run_batch の roll と同一手順（1本ぶん）。"""
                rr = search_begin(ob, list(r["h0"].get("deck") or []),
                                  list(r["h0"].get("prize") or []),
                                  list(r["h1"].get("deck") or []),
                                  list(r["h1"].get("prize") or []),
                                  list(r["h1"].get("hand") or []), [])
                ch = search_step(rr.searchId, list(first))
                NET.set_state(r["st_me"]); NETo.set_state(r["st_op"])
                pv = list(r["prev"]); pvo = list(r["pvo"])
                for _ in range(1500):
                    o2 = ch.observation; s2 = o2.current
                    if s2.result != -1:
                        search_end()
                        return 1.0 if s2.result == me else (0.5 if s2.result == 2 else 0.0)
                    sel = o2.select
                    if sel is None or not sel.option:
                        search_end(); return 0.5
                    if s2.yourIndex == me:
                        sc, _, _, c_, v_, _, _ = EG._feat(o2, FT, NET, pv)
                        ix = [int(x) for x in np.argsort(-sc)[:EG._k_of(sel, len(sel.option))]]
                        pv = [float(c_[ix[0]]), v_[ix[0]][0], v_[ix[0]][1]]
                    else:
                        sc, _, _, c_, v_, _, _ = EG._feat(o2, FTo, NETo, pvo)
                        ix = [int(x) for x in np.argsort(-sc)[:EG._k_of(sel, len(sel.option))]]
                        pvo = [float(c_[ix[0]]), v_[ix[0]][0], v_[ix[0]][1]]
                    ch = search_step(ch.searchId, ix)
                search_end(); return 0.5

            for m in range(add):
                for ci, c in enumerate(cands):
                    new[ci, m] = roll(c)
            out.append((i, new))
        except Exception as e:  # noqa: BLE001  1局面の失敗で担当分を落とさない
            try:
                search_end()
            except Exception:  # noqa: BLE001
                pass
            print(f"   局面{i}をスキップ: {type(e).__name__}: {str(e)[:80]}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pkl", default="data/nn/exit104_v2.pkl")
    ap.add_argument("--out", default="data/nn/exit104_v2r.pkl")
    ap.add_argument("--add", type=int, default=72, help="追加するロールアウト数")
    ap.add_argument("-w", "--workers", type=int, default=6)
    ap.add_argument("--chunk", type=int, default=20)
    a = ap.parse_args()

    d = pickle.load(open(a.pkl, "rb"))
    rows = d["rows"]
    # **対象は best≠top1 の全局面**（中盤フィルタは掛けない。選別は教師データ化の
    # 段階で重みとして行う）。決着済み（全候補同結果）は増やしても何も出ないので除く。
    tgt = []
    for i, r in enumerate(rows):
        q = np.asarray(r["q"], float)
        if q.max() - q.min() < 1e-9:
            continue
        if int(np.argmax(q)) == 0:
            continue
        tgt.append(i)
    print(f"■ ExIt再評価: 対象{len(tgt)}局面 / 追加{a.add}本 / {a.workers}並列", flush=True)
    print(f"   （全{len(rows)}局面中。決着済みと best=top1 は対象外）", flush=True)

    tasks = [(tgt[s:s + a.chunk], [rows[i] for i in tgt[s:s + a.chunk]], a.add)
             for s in range(0, len(tgt), a.chunk)]
    add_map = {}
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init,
                             initargs=(d["main"], d["opps"])) as ex:
        futs = [ex.submit(work, t) for t in tasks]
        for k, f in enumerate(as_completed(futs)):
            try:
                for i, new in f.result():
                    add_map[i] = new
            except Exception as e:  # noqa: BLE001
                print(f"   chunk失敗: {type(e).__name__}: {str(e)[:100]}", flush=True)
                continue
            el = time.perf_counter() - t0
            eta = el / (k + 1) * (len(tasks) - k - 1)
            print(f"   {k+1}/{len(tasks)} chunk  再評価{len(add_map)}局面  "
                  f"{el:.0f}秒  残り約{eta/60:.0f}分", flush=True)

    for i, new in add_map.items():
        rows[i]["R_add"] = new           # 追加分だけを別キーで持つ（検定はこれだけで行う）
        rows[i]["M_add"] = int(new.shape[1])
    d["reeval"] = dict(n=len(add_map), add=a.add)
    with open(a.out, "wb") as f:
        pickle.dump(d, f)
    dt = time.perf_counter() - t0
    print(f"\n保存 -> {a.out}  再評価{len(add_map)}局面  {dt/60:.1f}分", flush=True)

    # 追加分だけでの改善量（選択バイアスなし）
    g0, g1 = [], []
    for i in add_map:
        q = np.asarray(rows[i]["q"], float); b = int(np.argmax(q))
        A = np.asarray(rows[i]["R_add"], float)
        g0.append(q[b] - q[0])                    # 最初の24本で見えた差（上振れ込み）
        g1.append(float(A[b].mean() - A[0].mean()))  # 追加分だけでの差（不偏）
    g0, g1 = np.array(g0), np.array(g1)
    print(f"   最初の24本での平均差 {g0.mean():.3f} → **追加{a.add}本での平均差 "
          f"{g1.mean():.3f}**（縮小率 {g1.mean()/max(g0.mean(),1e-9):.1%}）")
    for t in (0.0, 0.10, 0.20, 0.35, 0.50):
        print(f"   追加分の差 > {t:.2f}: {int((g1 > t).sum()):>5}局面")


if __name__ == "__main__":
    main()
