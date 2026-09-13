"""ExIt Phase 1a: 生データ（局面 + 全候補のQ）を生成する（EXP-115）。

**生成とラベル付けを分離する**。ロールアウト（10.6秒/局面）が高コストで、
ラベル付け（best を選ぶ）は数秒なので、**全候補のQを保存して後から方式を選べる**ようにする。

Phase 0 の結果を反映した設計:
  - 局面は**全コンテキストから無作為抽出**（gap/entropy に予測力なし、と実測）
    差>0.10 は MAIN 37.5% / ACTIVATE 42.9% / TO_ACTIVE 50.0%（差の平均0.198で最大）
    → MAINに絞ると取りこぼす
  - SETUP系(ctx 1/2/41)は除外（相手アクティブが伏せで search_begin が使えない）
  - **逐次打ち切り** M=8 → 24 → 64。決着済み局面を安く弾く
  - **LSTM状態を復元**してロールアウトする（reset すると記憶を失った弱い方策で評価。
    EXP-094: サブ選択でLSTM状態の効果は TO_HAND +11.0 / TO_ACTIVE +12.8pt）

usage:
  uv run python selfplay/exit_gen.py --games 400 --per-game 6 -w 8 --out data/nn/exit104.pkl
"""
from __future__ import annotations
import argparse, os, pickle, random, sys, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO, os.path.join(REPO, "arena"), os.path.join(REPO, "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

SETUP_CTX = (1, 2, 41)          # 相手アクティブが伏せで search_begin が使えない
_W: dict = {}


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


def _init(main_dir, opp_dirs):
    _W["main"] = _load(main_dir)
    _W["opps"] = {d: _load(d) for d in opp_dirs}


def _feat(ob, ft, net, prev):
    st = ob.current; my = st.yourIndex; s = ob.select
    b, _ = ft.feat_vector(st, my); lg = ft.logs_vector(ob, my)
    hc, pl = ft.pool_counts(st, my)
    cids, vecs = [], []
    for opt in s.option:
        c, v = ft.option_vector(ob, opt, my, hc, pl)
        cids.append(c); vecs.append(v)
    ctx = int(s.context)
    cls = ([ft.CLASSES.index(ft.option_class(st, s, i, my)) for i in range(len(s.option))]
           if ctx == 0 else [-1] * len(s.option))
    sc = np.asarray(net.scores(b, lg, prev, ctx, cids, vecs, cls if ctx == 0 else None),
                    np.float64)
    return sc, b, lg, cids, vecs, cls, ctx


def _k_of(sel, n):
    mn = int(sel.minCount if sel.minCount is not None else 1)
    mx = int(sel.maxCount if sel.maxCount is not None else mn or 1)
    return max(1, min(max(mn, min(mx if mx else mn, n)), n))


def _cands(sc, sel, ncand):
    """min/max を満たす**選択集合**の候補。候補0 は方策の選択そのもの。"""
    n = len(sel.option); k = _k_of(sel, n)
    order = [int(x) for x in np.argsort(-sc)]
    out = [sorted(order[:k])]
    for j in range(k, min(n, k + ncand - 1)):
        alt = sorted(order[:k-1] + [order[j]])
        if alt not in out:
            out.append(alt)
    return out[:ncand]


def run_batch(task):
    """1ワーカーが games 試合を打ち、途中の局面を評価して生データを返す。"""
    (gi0, ngames, opp_dir, seed0, per_game, ncand, Ms, temp, mid_lo, mid_hi) = task
    import cg.game as cgg
    from cg.api import to_observation_class, search_begin, search_step, search_end
    from selfplay.state_encoder import _hidden
    FT, NET, DA = _W["main"]; FTo, NETo, DB = _W["opps"][opp_dir]
    out = []

    def begin(ob, h0, h1):
        """局面を複製する。**エンジンが山順を引き直す**ので、呼ぶたびに別の未来になる。"""
        return search_begin(ob, list(h0.get("deck") or []), list(h0.get("prize") or []),
                            list(h1.get("deck") or []), list(h1.get("prize") or []),
                            list(h1.get("hand") or []), [])

    def roll_from(ch, me, st_me, st_op, pv0, pvo0):
        """分岐済みの状態から終局までプレイして結果を返す。

        **search_end() をここで呼んではいけない**。search_end は探索全体を終了させるので、
        同じ root から分岐した兄弟候補の状態まで消える（CRNが成立しなくなる）。
        1つの m の全候補を回し終えてから、呼び出し側で1回だけ呼ぶ。
        """
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
                sc, _, _, c_, v_, _, _ = _feat(o2, FT, NET, pv)
                idx = [int(x) for x in np.argsort(-sc)[:_k_of(sel, len(sel.option))]]
                pv = [float(c_[idx[0]]), v_[idx[0]][0], v_[idx[0]][1]]
            else:
                sc, _, _, c_, v_, _, _ = _feat(o2, FTo, NETo, pvo)
                idx = [int(x) for x in np.argsort(-sc)[:_k_of(sel, len(sel.option))]]
                pvo = [float(c_[idx[0]]), v_[idx[0]][0], v_[idx[0]][1]]
            ch = search_step(ch.searchId, idx)
        return 0.5

    for g in range(ngames):
        gi = gi0 + g
        rng = random.Random(seed0 + gi)
        random.seed(seed0 + gi)
        obs, _ = cgg.battle_start_seeded(DA, DB, seed0 + gi)
        ptr = cgg.Battle.battle_ptr
        NET.reset(); NETo.reset(); pv = [-1., -1., -1.]; pvo = [-1., -1., -1.]
        snaps = []; n = 0
        while obs is not None and obs["current"]["result"] == -1 and n < 3000:
            ob = to_observation_class(obs); st = ob.current; sel = ob.select
            if sel is None or not sel.option:
                obs = cgg.battle_select([0]); n += 1; continue
            if st.yourIndex == 0:
                sc, b, lg, cids, vecs, cls, ctx = _feat(ob, FT, NET, pv)
                k = _k_of(sel, len(sel.option))
                # **相手のアクティブが伏せのままの局面は search_begin できない**
                # （エンジンが opponent_active にポケモンのIDを要求する）。
                # SETUP_CTX の除外だけでは取り切れない局面が実在し、そこで
                # `ValueError: Active card must be the ID of a Pokémon card.` が出て
                # チャンクごと落ちていた。**偽のアクティブを捏造せず局面ごと捨てる**
                # （でっち上げるとロールアウトが実局面と別物になる）。
                _opx = st.players[1 - int(st.yourIndex)]
                _oa_ok = bool(_opx.active and _opx.active[0])
                # **局面は絞り込まない**。Phase 1a の n=2,879 で候補を全て検討した結果:
                #   top1−top2 の gap : 相関 r=+0.016。**無力**（方策が確信している局面ほど
                #                      むしろ割れる＝誤りが分散でなくバイアスだから。EXP-117）
                #   turn≤11          : turn≥12 の83.9%は決着済みで**M=8で安く弾ける**ので、
                #                      事前に除いても計算は6.3%しか減らず改善は6.7%失う＝相殺
                #   選択肢≥3         : 「3倍の差」に見えたのは**測定のアーティファクト**。
                #                      2枚だと代替候補が1本しか作れないため。代替1本あたりでは
                #                      4.8% vs 6.6% の1.4倍で、9枚以上は5.1%と2枚並み
                # 効くのは**相手の混合比**だけだった（下の --opp-weights）。
                if (ctx not in SETUP_CTX and _oa_ok and len(sel.option) >= 2
                        and int(sel.minCount or 1) <= len(sel.option)):
                    snaps.append(dict(ob=ob, obs=obs, h0=_hidden(ptr, 0), h1=_hidden(ptr, 1),
                                      st_me=NET.get_state(), st_op=NETo.get_state(),
                                      pv=list(pv), pvo=list(pvo), ctx=ctx,
                                      turn=int(st.turn), board=list(b), logs=list(lg),
                                      opt_cid=cids, opt_vec=vecs, opt_cls=cls,
                                      mn=int(sel.minCount or 1), mx=int(sel.maxCount or 1),
                                      cands=_cands(sc, sel, ncand), sc=sc.tolist()))
                # **生成は T>0 でサンプリング**（多様な局面を集めるため）
                if temp > 0:
                    p = np.exp((sc - sc.max()) / temp); p /= p.sum()
                    idx = sorted(rng.sample(range(len(sc)), 0) or
                                 list(np.random.default_rng(seed0+gi+n).choice(
                                     len(p), size=k, replace=False, p=p)))
                else:
                    idx = [int(x) for x in np.argsort(-sc)[:k]]
                idx = [int(x) for x in idx]
                pv = [float(cids[idx[0]]), vecs[idx[0]][0], vecs[idx[0]][1]]
            else:
                sc, _, _, c_, v_, _, _ = _feat(ob, FTo, NETo, pvo)
                idx = [int(x) for x in np.argsort(-sc)[:_k_of(sel, len(sel.option))]]
                pvo = [float(c_[idx[0]]), v_[idx[0]][0], v_[idx[0]][1]]
            obs = cgg.battle_select(idx); n += 1
        # ---- 局面をサンプリングして評価 ----
        _r = (obs["current"]["result"] if obs is not None else -1)
        game_res = (1.0 if _r == 0 else 0.5 if _r == 2 else 0.0 if _r == 1 else None)
        if snaps:
            pick = rng.sample(range(len(snaps)), min(per_game, len(snaps)))
            for pi in pick:
              # **1局面の失敗でチャンク60局面を巻き添えにしない**。
              # エンジンが局面固有の理由で search_begin を拒む場合があり、
              # 3.5時間ぶんの生成がこれで消えた（EXP-117の事故）。
              try:
                sn = snaps[pi]; ob = sn["ob"]; me = ob.current.yourIndex
                nc = len(sn["cands"])
                Q = np.zeros((nc, 0))
                done = False
                for M in Ms:                      # 逐次打ち切り 8 → 24 → 64
                    add = M - Q.shape[1]
                    if add <= 0: continue
                    new = np.zeros((nc, add))
                    for m in range(add):
                        # **1回のシャッフルを全候補で共有する（CRN）**。
                        # 候補ごとに search_begin を呼び直すと山順が候補間でバラバラになり、
                        # 「山札の並びが悪い」という共通の運が差に残る。実測（10局面×M=12）:
                        #   独立標本 差のsd 0.443 / **CRNペア 0.297** → **実効Mが2.23倍**
                        # しかも search_begin の呼び出しが候補数ぶんの1に減るので実時間も短い。
                        rt = begin(ob, sn["h0"], sn["h1"])
                        for ci, c in enumerate(sn["cands"]):
                            ch0 = search_step(rt.searchId, list(c))
                            new[ci, m] = roll_from(ch0, me, sn["st_me"], sn["st_op"],
                                                   sn["pv"], sn["pvo"])
                        search_end()      # **全候補を回し終えてから1回だけ**
                    Q = np.concatenate([Q, new], axis=1)
                    q = Q.mean(1)
                    # **昇格をやめる条件**。どちらも「行を捨てる」ことはしない
                    # （採否は教師データ化の工程で決める。EXP-115 の方針）。
                    if q.max() - q.min() < 1e-9:  # 全候補同結果 → 増やしても何も出ない
                        done = True; break
                    # **勝負がついている局面はMを増やさない**（EXP-115 で実測）。
                    # 改善率は q[0] に対して山型で、両端では改善余地が消える:
                    #   q[0]=0（完敗確定） 1,385局面 → 決着済み90.5% / 改善率2.8%
                    #   q[0]=0.4-0.6      565局面 → 決着済み **5.0%** / 改善率 **9.9%**
                    #   q[0]=1（完勝確定） 1,585局面 → 決着済み81.5% / 改善率 **0.0%**
                    # 中盤(0.2-0.8)に絞ると改善率は 2.03倍、決着済みは 47.9%→9.8% に減る。
                    # **行は残す**（捨てるのは教師データ化の工程の仕事）。
                    if not (mid_lo <= q[0] <= mid_hi):
                        done = True; break
                    # **「差が大きいから止める」は廃止した**。
                    # 教師データ化ではCRNペアの符号検定を使うが、M=8 で両側 p<0.05 を
                    # 満たすのは **8/8 の全会一致だけ**（7/8 でも p=0.070 で落ちる）。
                    # つまり M=8 で打ち切った局面は、差がいくら大きく見えても
                    # 検定に通らず教師にできない＝8本を捨てるだけになる。
                    # 決着していない局面は必ず M=24 まで回す（18/24 で p=0.023）。
                q = Q.mean(1)
                out.append(dict(ctx=sn["ctx"], turn=sn["turn"], opp=opp_dir,
                                # **その試合が実際に勝ったか**。「負け試合の局面のほうが
                                # 学びが多いのでは」を後から実測するため（追加コスト0）。
                                # 注: q[0] は前向きの期待勝率で、これは1回ぶんの実現値。
                                game_result=game_res,
                                board=sn["board"], logs=sn["logs"], prev=sn["pv"],
                                opt_cid=sn["opt_cid"], opt_vec=sn["opt_vec"],
                                opt_cls=sn["opt_cls"], mn=sn["mn"], mx=sn["mx"],
                                cands=sn["cands"], q=q.tolist(), M=int(Q.shape[1]),
                                sc=sn["sc"],
                                # **1本ずつの勝敗**（候補 × M）。全候補が同じ root から
                                # 分岐しているのでCRNペアであり、**符号検定ができる**。
                                # 平均だけだとρが分からず有意性を判定できない（前回の穴）。
                                R=Q.astype(np.float32),
                                # **局面を後から再評価するための一式**。これがあれば
                                # 教師データ化の段階で「Mを増やして測り直す」が選べる
                                # ＝生成時に精度を確定させる必要がなくなる。
                                obs=sn["obs"], h0=sn["h0"], h1=sn["h1"],
                                st_me=sn["st_me"], st_op=sn["st_op"], pvo=sn["pvo"]))
              except Exception as _e:              # noqa: BLE001
                  try:
                      search_end()                 # 探索を開きっぱなしにしない
                  except Exception:                # noqa: BLE001
                      pass
                  print(f"   局面をスキップ: {type(_e).__name__}: {str(_e)[:90]}",
                        flush=True)
        cgg.battle_finish()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--main", default="agents/104_alakazam_feat4")
    ap.add_argument("--opps", nargs="+",
                    default=["agents/101_marnie_luca", "agents/131_ogerpon_majkel",
                             "agents/098_spidops_gen", "agents/104_alakazam_feat4"])
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--per-game", type=int, default=6, help="1試合から評価する局面数")
    ap.add_argument("--cands", type=int, default=3)
    # **M=64 まで昇格させない**。前回そこへ上がった840局面は best≠top1 が 56.5%
    # ＝偶然の 66.7% とほぼ同じ＝ノイズしか得られず、予算の47%を食っていた。
    # 精度が要る局面は、保存した隠れ情報から**後で測り直せる**（rows[i]["obs"]等）。
    ap.add_argument("--Ms", type=int, nargs="+", default=[8, 96])
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("-w", "--workers", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=10, help="ワーカーを使い捨てる試合数")
    ap.add_argument("--save-every", type=int, default=5, dest="save_every",
                    help="このチャンク数ごとに途中保存する（0で無効）")
    ap.add_argument("--opp-weights", type=float, nargs="+", default=None,
                    dest="opp_weights",
                    help="--opps と同じ並びの重み（本番メタシェア）。既定は均等")
    # **中盤フィルタの範囲**。M=8時点の q[0] がこの外なら昇格しない。
    ap.add_argument("--mid-lo", type=float, default=0.125, dest="mid_lo")
    ap.add_argument("--mid-hi", type=float, default=0.875, dest="mid_hi")
    ap.add_argument("--seed", type=int, default=9000)
    ap.add_argument("--out", default="data/nn/exit104.pkl")
    a = ap.parse_args()

    tasks = []
    gi = 0
    nchunk = (a.games + a.chunk - 1) // a.chunk
    # **相手をメタシェアで配分する**。均等割りだと局面分布が配備先とズレる
    # （前回: 生成は各25%だが本番メタは Marnie 48.3% / Spidops 2.7%）。
    # ExIt は「見た局面」でしか方策を改善しないので、分布のズレは改善箇所のズレになる。
    # 割り当ては**最大剰余法で決定的**に行う（乱数を使わないので再現できる）。
    W = a.opp_weights or [1.0] * len(a.opps)
    assert len(W) == len(a.opps), "--opp-weights の個数が --opps と違う"
    tw = float(sum(W))
    exact = [nchunk * w / tw for w in W]
    alloc = [int(x) for x in exact]
    rem = sorted(range(len(W)), key=lambda i: -(exact[i] - alloc[i]))
    for i in range(nchunk - sum(alloc)):
        alloc[rem[i % len(rem)]] += 1
    order = [d for d, k in zip(a.opps, alloc) for _ in range(k)]
    print("■ 相手の配分: " + " / ".join(
        f"{d.split('/')[-1]} {k}chunk({k/nchunk:.1%})"
        for d, k in zip(a.opps, alloc)), flush=True)
    ci = 0
    while gi < a.games:
        nb = min(a.chunk, a.games - gi)
        opp = order[ci % len(order)]; ci += 1
        tasks.append((gi, nb, opp, a.seed, a.per_game, a.cands, tuple(a.Ms), a.temp,
                      a.mid_lo, a.mid_hi))
        gi += nb
    print(f"■ ExIt生成: {a.games}試合 / 1試合{a.per_game}局面 / 候補{a.cands} / "
          f"M={a.Ms} / {a.workers}並列 / 相手{len(a.opps)}種", flush=True)
    # **as_completed で終わった順に受け取る**。ex.map は順序を保つので
    # 先頭チャンクが終わるまで一切表示されず、進捗が見えない（実際に1時間気づけなかった）。
    from concurrent.futures import as_completed
    t0 = time.perf_counter(); res = []
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init,
                             initargs=(a.main, a.opps)) as ex:
        futs = [ex.submit(run_batch, t) for t in tasks]
        for i, f in enumerate(as_completed(futs)):
            try:
                res += f.result()
            except Exception as e:                       # 1チャンクの失敗で全体を落とさない
                print(f"   chunk 失敗: {type(e).__name__}: {str(e)[:120]}", flush=True)
                continue
            el = time.perf_counter() - t0
            eta = el / (i + 1) * (len(tasks) - i - 1)
            print(f"   {i+1}/{len(tasks)} chunk 完了  局面{len(res)}  "
                  f"{el:.0f}秒  残り約{eta/60:.0f}分", flush=True)
            # **定期保存**。24〜48時間の生成を途中で止めても成果が残るようにする。
            # 全チャンクを溜めて最後に1回だけ書く作りだと、中断＝全損だった。
            # 一時ファイルへ書いてから rename する（書きかけを読ませない）。
            if a.save_every and (i + 1) % a.save_every == 0:
                _tmp = a.out + ".part"
                with open(_tmp, "wb") as _f:
                    pickle.dump(dict(rows=res, main=a.main, opps=a.opps,
                                     partial=True, done_chunks=i + 1,
                                     total_chunks=len(tasks)), _f)
                os.replace(_tmp, a.out)
                print(f"      → 途中保存 {a.out}（{len(res)}局面）", flush=True)
    dt = time.perf_counter() - t0
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "wb") as f:
        pickle.dump(dict(rows=res, main=a.main, opps=a.opps, Ms=a.Ms,
                         cands=a.cands, games=a.games), f)
    q = np.array([max(r["q"]) - min(r["q"]) for r in res])
    nb = np.mean([int(np.argmax(r["q"])) != 0 for r in res])
    print(f"\n保存 -> {a.out}  局面{len(res)}  {dt/60:.1f}分  "
          f"（{dt/max(1,len(res)):.1f}秒/局面）")
    print(f"   差>0.10 {np.mean(q>0.10):.1%} / 差>0.00 {np.mean(q>0):.1%} / "
          f"**best≠top1 {nb:.1%}**")
    print(f"   平均M {np.mean([r['M'] for r in res]):.1f}（逐次打ち切りの効果）")


if __name__ == "__main__":
    main()
