"""価値誘導探索が方策の選択をどれだけ変えるかを、実装前に測る。

**なぜ先に測るか**: EXP-053 で探索は5連敗している。実装してから「効かない」と分かるのは
高くつく。変更率を先に見れば、その時点で見込みが判定できる。

  変更率 0〜2%    方策は既にVと整合的。探索の余地なし
  変更率 5〜15%   適度。改善の見込みあり
  変更率 30%超    Vが方策と食い違いすぎ。Vの質を疑うべき

探索の形（EXP-053の死因への対処込み）:
  - 方策の上位k候補だけを展開
  - **相手をロールアウトしない**（死因1: 決定化した受動的な相手が手番を渡す価値を過大評価）
  - 葉は「自分のターンが終わった時点」の盤面、評価は value head（死因2: Vは連続値なので
    1サンプルで差が出る。MCのN=8では±18%の誤差で埋もれた）
  - **同一の search_begin から分岐するので全候補が同じ世界を共有**（CRN）

usage:
  uv run python tools/search_probe.py --agent agents/104_alakazam_feat4 \
      --model model_104_v.npz --games 6 -k 5
"""

from __future__ import annotations

import argparse
import collections
import importlib.util as ilu
import os
import random
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO, os.path.join(REPO, "arena")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--agent", default="agents/104_alakazam_feat4")
    ap.add_argument("--model", default="model_104_v.npz")
    ap.add_argument("--games", type=int, default=6)
    ap.add_argument("-k", type=int, default=5, help="展開する候補数")
    a = ap.parse_args()

    import cg.game as cgg
    import run_match as rm
    from cg.api import (OptionType, search_begin, search_end, search_step,
                        to_observation_class)

    ag = os.path.join(REPO, a.agent)
    if ag not in sys.path:
        sys.path.insert(0, ag)
    fn = next(f for f in sorted(os.listdir(ag))
              if f.startswith("feat_") and f.endswith(".py"))
    sp = ilu.spec_from_file_location(fn[:-3], os.path.join(ag, fn))
    ft = ilu.module_from_spec(sp)
    sys.modules[fn[:-3]] = ft
    sp.loader.exec_module(ft)
    nn_name = next(f for f in sorted(os.listdir(ag))
                   if f.startswith("nn_") and f.endswith(".py"))
    sp2 = ilu.spec_from_file_location(nn_name[:-3], os.path.join(ag, nn_name))
    nnm = ilu.module_from_spec(sp2)
    sys.modules[nn_name[:-3]] = nnm
    sp2.loader.exec_module(nnm)
    net = nnm.LSTMPolicy(os.path.join(ag, a.model))

    def feats(o, me):
        """1決定ぶんの (board, logs, prev用の材料, ctx, 選択肢) を作る。"""
        st = o.current
        board, _ = ft.feat_vector(st, me)
        logs = ft.logs_vector(o, me)
        hc, pl = ft.pool_counts(st, me)
        cids, vecs, clss = [], [], []
        for opt in o.select.option:
            c, v = ft.option_vector(o, opt, me, hc, pl)
            cids.append(c)
            vecs.append(v)
        if int(o.select.context) == 0:
            clss = [ft.CLASSES.index(ft.option_class(st, o.select, i, me))
                    for i in range(len(o.select.option))]
        else:
            clss = None
        return board, logs, int(o.select.context), cids, vecs, clss

    OPPS = [("101_marnie_luca", 4242), ("098_spidops_gen", 777),
            ("085_kangaskhan_e57e", 99)]
    n_dec = n_change = 0
    dv = []
    secs = []
    per = collections.Counter()

    for opp, seed in OPPS:
        pb = rm.load_agent_module(os.path.join(REPO, "agents", opp), "B")
        da, db = rm.read_deck(ag), rm.read_deck(os.path.join(REPO, "agents", opp))
        for g in range(a.games):
            random.seed(g)
            obs, _ = cgg.battle_start_seeded(da, db, seed + g)
            net.reset()
            prev = [-1.0, -1.0, -1.0]
            while obs is not None:
                if obs["current"]["result"] != -1:
                    break
                me = obs["current"]["yourIndex"]
                if me != 0:
                    obs = cgg.battle_select(pb(obs))
                    continue
                o = to_observation_class(obs)
                sel = o.select
                if sel is None or not sel.option:
                    obs = cgg.battle_select([0])
                    continue
                board, logs, ctx, cids, vecs, clss = feats(o, me)
                base = net.get_state()
                sc = net.scores(board, logs, prev, ctx, cids, vecs, clss)
                after = net.get_state()          # 本手を1回進めた状態
                pick = int(np.argmax(sc))
                if ctx == 0 and len(sel.option) >= 2:
                    order = list(np.argsort(-np.asarray(sc)))[:a.k]
                    t0 = time.perf_counter()
                    vals = {}
                    try:
                        mp, op_ = o.current.players[0], o.current.players[1]
                        pad = sorted(ft.DECK_COUNTS)[0]
                        root = search_begin(
                            o, [pad] * mp.deckCount, [pad] * len(mp.prize),
                            [pad] * op_.deckCount, [pad] * len(op_.prize),
                            [pad] * op_.handCount, [])
                        for cand in order:
                            net.set_state(base)          # 各候補は同じ状態から
                            ch = search_step(root.searchId, [int(cand)])
                            pv = list(prev)
                            for _ in range(30):
                                ob2 = ch.observation
                                s2 = ob2.select
                                if (s2 is None or not s2.option
                                        or ob2.current.yourIndex != 0):
                                    break
                                b2, l2, c2, ci2, vv2, cl2 = feats(ob2, 0)
                                s_ = net.scores(b2, l2, pv, c2, ci2, vv2, cl2)
                                kk = max(1, min(s2.maxCount or s2.minCount or 1,
                                                len(s2.option)))
                                idx = list(np.argsort(-np.asarray(s_))[:kk])
                                pv = [float(ci2[idx[0]]),
                                      float(s2.option[idx[0]].type or -1),
                                      float(s2.option[idx[0]].area or -1)]
                                ch = search_step(ch.searchId, [int(x) for x in idx])
                            ob2 = ch.observation
                            b2, l2, c2, _, _, _ = feats(ob2, 0) if (
                                ob2.select and ob2.select.option) else (
                                (ft.feat_vector(ob2.current, 0)[0],
                                 ft.logs_vector(ob2, 0), 0, None, None, None))
                            vals[int(cand)] = net.value(b2, l2, pv, c2)
                    except Exception:  # noqa: BLE001
                        vals = {}
                    finally:
                        try:
                            search_end()
                        except Exception:  # noqa: BLE001
                            pass
                    secs.append(time.perf_counter() - t0)
                    if len(vals) >= 2:
                        n_dec += 1
                        best = max(vals, key=vals.get)
                        if best != pick:
                            n_change += 1
                            per[opp.split("_")[1]] += 1
                        vs = sorted(vals.values(), reverse=True)
                        dv.append(vs[0] - vs[-1])
                net.set_state(after)      # 本番の状態に戻す
                k = max(1, min(sel.maxCount or sel.minCount or 1, len(sel.option)))
                idx = list(np.argsort(-np.asarray(sc))[:k])
                prev = [float(cids[idx[0]]),
                        float(sel.option[idx[0]].type or -1),
                        float(sel.option[idx[0]].area or -1)]
                obs = cgg.battle_select([int(x) for x in idx])
            cgg.battle_finish()

    print(f"探索した MAIN決定: {n_dec} 件（候補 k={a.k}）")
    if not n_dec:
        raise SystemExit("探索が1件も成立しなかった")
    print(f"  **方策の選択を変えた: {n_change} 件 = {n_change/n_dec:.1%}**")
    print(f"  候補間のV差（最大−最小）: 中央値 {np.median(dv):.3f} / "
          f"平均 {np.mean(dv):.3f}")
    print(f"  探索の所要時間: 中央値 {np.median(secs)*1000:.0f} ms/決定 "
          f"（上位陣は約200ms）")
    print(f"\n  対面別の変更件数: {dict(per)}")


if __name__ == "__main__":
    main()
