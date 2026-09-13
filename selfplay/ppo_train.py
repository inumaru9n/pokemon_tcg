"""Stage 2c: 診断ゲート —— PPOで方策が実際に強くなるかを測る。

2bで「動くが壊れない」帯（lr=3e-4, KL係数0.05）を特定した。ここで初めて**強さ**を測る。

**測り方**: 一定イテレーションごとに、更新中の方策と**凍結した094**を直接対戦させる
（両者ともargmax=本番と同じ決定的方策）。先後を交互に入れ替えて先手有利を打ち消す。

**事前登録した判定**:
  勝率が **ノイズ床の2倍（1.81pt×2 = 3.62pt）** を超えて上昇 → Stage 2d（本番ラン）へ
  20イテレーション回して動かない → **Stage 2は撤退**し、Vを活かす後段（価値誘導探索）へ

**注意: ここでの判定は「評価した全イテレーションの最良値」と初期値の差で行っている**。
複数回測定して最良を選ぶ操作なので winner's curse を含み、**これは合否ではなく
「Stage 2dに進む価値があるか」の予備判定**である。最終合否は計画書どおり
「3学習シード平均・vs094 CRN n=800・+4pt以上」で別途測る（Stage 2d）。

argmaxで測るのは、生成はT=0.7でサンプリングするが**提出物はargmax**だから。
サンプリング方策が強くなってもargmaxが強くならなければ意味がない。
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "arena"),
           os.path.join(_ROOT, "agents/094_yushin_nn")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# **どのエージェントを微調整するかは指定させる**（既定の094は古い）。
# agent-dir を渡すと feat_*/nn_*/model_* をそこから取る。
MODEL = os.path.join(_ROOT, "agents/094_yushin_nn/model_094.npz")


@torch.no_grad()
def duel(net_a, net_b, vocab, deck, n_games, batch, device, seed, temp=0.0):
    """net_a vs net_b を対戦させる。先後を交互に入替。戻り: net_a の勝率。

    temp<=0 なら**両者argmax**（提出物と同じ決定的方策）。temp>0 なら両者その温度で
    サンプリング。**両方を測るのが重要**: PPOが最適化しているのは T=0.7 の分布であって
    argmaxではない。「argmaxが決して選ばない裾の悪い手の確率を下げる」だけの改善だと、
    サンプリング方策は強くなるのにargmaxは動かず、『PPOが効かない』と誤診する。
      argmaxもTも上がらない → 本当に学習していない
      Tだけ上がる → 学習はしているがargmaxに転移していない
    """
    _rng = np.random.default_rng(seed ^ 0x5EED)

    def _pick(sc):
        if temp <= 0:
            return int(np.argmax(sc))
        e = np.exp((sc - sc.max()) / temp)
        p = e / e.sum()
        return int(_rng.choice(len(p), p=p))

    ft = globals().get("FEAT")     # **微調整対象の特徴器**（未設定なら094）
    if ft is None:
        import feat_094 as ft
    from cg.api import to_observation_class
    from selfplay.vecenv import VecBattles
    from selfplay.ppo_probe import _TorchAsBatchPolicy
    import run_match as rm

    pa = _TorchAsBatchPolicy(net_a, vocab, 2 * batch, device)
    pb = _TorchAsBatchPolicy(net_b, vocab, 2 * batch, device)
    wins = tot = 0
    done = 0
    while done < n_games:
        nb = min(batch, n_games - done)
        seeds = [rm.derive_engine_seed(seed, done + j) for j in range(nb)]
        vb = VecBattles([deck] * nb, [deck] * nb, seeds)
        pa.reset_all()
        pb.reset_all()
        a_seat = {j: ((done + j) % 2) for j in range(nb)}   # net_a の席を交互に
        prev = [[-1.0, -1.0, -1.0] for _ in range(2 * nb)]
        while vb.live:
            cur = vb.observations()
            acts = {}
            for who_net, pol in ((0, pa), (1, pb)):
                slots, B, L, P, C, CI, VE, CL, meta = [], [], [], [], [], [], [], [], []
                for (bi, o) in cur:
                    me = o["current"]["yourIndex"]
                    is_a = (me == a_seat[bi])
                    if (0 if is_a else 1) != who_net:
                        continue
                    ob = to_observation_class(o)
                    st, sel = ob.current, ob.select
                    n = len(sel.option)
                    k = max(min(sel.maxCount or sel.minCount, n), sel.minCount)
                    if n == 0:
                        acts[bi] = []
                        continue
                    if n < 2 and sel.minCount != 0:
                        acts[bi] = list(range(min(max(1, k), n)))
                        continue
                    b_, _ = ft.feat_vector(st, me)
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
                    nr = n
                    if sel.minCount == 0:
                        cids = cids + [ft.NULL_CID]
                        vecs = vecs + [ft.null_option_vector()]
                        if cls is not None:
                            cls = cls + [-1]
                    slots.append(2 * bi + me)
                    B.append(b_); L.append(lg); P.append(list(prev[2 * bi + me]))
                    C.append(ctx); CI.append(cids); VE.append(vecs); CL.append(cls)
                    meta.append((bi, me, nr, k, cids, vecs,
                                 int(sel.minCount),
                                 int(sel.maxCount or sel.minCount)))
                if slots:
                    scs = pol.scores(slots, B, L, P, C, CI, VE, CL)
                    for (bi, me, nr, k, cids, vecs, mn, mx), sc in zip(meta, scs):
                        sc = np.asarray(sc, np.float64)
                        # **maxCount==1 は rollout.collect と同じ規約でサンプリング**。
                        # ここで a を捨てて argsort の上位k個を取ると、温度が効くのは
                        # 「棄権するか」だけになり、T評価がargmax評価とほぼ同じ値を返す
                        # ＝「転移していない」と誤読する（実際に1度埋め込んだ）。
                        single = (mx == 1)
                        a = _pick(sc) if single else int(np.argmax(sc))
                        if len(cids) > nr and a == nr:
                            acts[bi] = []
                            prev[2 * bi + me] = [float(ft.NULL_CID), -1.0, -1.0]
                        elif single:
                            acts[bi] = [a]
                            prev[2 * bi + me] = [float(cids[a]),
                                                 vecs[a][0], vecs[a][1]]
                        else:
                            order = [i for i in np.argsort(-sc) if i < nr]
                            pick = sorted(order[:max(1, min(k, nr))])
                            acts[bi] = pick
                            prev[2 * bi + me] = [float(cids[pick[0]]),
                                                 vecs[pick[0]][0], vecs[pick[0]][1]]
            vb.step(acts)
        for bi in range(nb):
            r = vb.results[bi]
            if r in (0, 1):
                tot += 1
                wins += int(r == a_seat[bi])
        vb.close()
        done += nb
    return wins / max(1, tot), tot


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--kl", type=float, default=0.05)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--batch", type=int, default=80)
    ap.add_argument("--mb", type=int, default=24)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--vlr", type=float, default=3e-4)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--vbuf", type=int, default=6)
    ap.add_argument("--vinit", default="")
    ap.add_argument("--eval-every", type=int, default=4)
    ap.add_argument("--eval-games", type=int, default=400)
    ap.add_argument("--kl-break", type=float, default=0.5,
                    help="1更新のKLがこれを超えたら打ち切る安全装置（緩めに設定）")
    ap.add_argument("--agent-dir", default=None,
                    help="微調整するエージェント（例 agents/117_alakazam_dspline）")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="/kaggle/working/ppo_policy.npz")
    args = ap.parse_args()

    from selfplay.ppo import (PolicyTorch, ValueTorch, ppo_update, value_update,
                              gae, to_value_rollouts)
    from selfplay.ppo_probe import _TorchAsBatchPolicy
    from selfplay.rollout import collect
    import run_match as rm

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    _agent = getattr(args, "agent_dir", None) or "agents/094_yushin_nn"
    deck = rm.read_deck(os.path.join(_ROOT, _agent))
    _md = MODEL
    if getattr(args, "agent_dir", None):
        import glob as _g
        _c = _g.glob(os.path.join(_ROOT, args.agent_dir, "model_*.npz"))
        _c = [x for x in _c if not x.endswith("_v.npz")]
        assert len(_c) == 1, f"model_*.npz が一意でない: {_c}"
        _md = _c[0]
        _ad = os.path.join(_ROOT, args.agent_dir)
        if _ad not in sys.path:
            sys.path.insert(0, _ad)
    import glob as _g2, importlib.util as _ilu
    _fp = _g2.glob(os.path.join(_ROOT, _agent, "feat_*.py"))
    assert len(_fp) == 1, f"feat_*.py が一意でない: {_fp}"
    _fn = os.path.basename(_fp[0])[:-3]
    _sp = _ilu.spec_from_file_location(_fn, _fp[0])
    _FT = _ilu.module_from_spec(_sp); sys.modules[_fn] = _FT; _sp.loader.exec_module(_FT)
    globals()["FEAT"] = _FT
    print(f"微調整の対象: {_md}  特徴器: {_fn}")
    net, vocab = PolicyTorch.from_npz(_md, device=dev)
    ref, _ = PolicyTorch.from_npz(_md, device=dev)
    for p in ref.parameters():
        p.requires_grad_(False)
    ref.eval()
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    z = np.load(_md, allow_pickle=False)   # **微調整対象と同じ次元でcriticを作る**
    vnet = ValueTorch(int(z["n_board"]), int(z["n_logs"]),
                      z["card_emb.weight"].shape[0]).to(dev)
    if args.vinit and os.path.exists(args.vinit):
        zz = np.load(args.vinit, allow_pickle=False)
        sd = vnet.state_dict()
        for k in sd:
            if k in zz.files and tuple(zz[k].shape) == tuple(sd[k].shape):
                sd[k].copy_(torch.as_tensor(np.asarray(zz[k], np.float32)))
        vnet.load_state_dict(sd)
        print(f"Critic: {os.path.basename(args.vinit)} から初期化", flush=True)
    vopt = torch.optim.Adam(vnet.parameters(), lr=args.vlr)

    print(f"device={dev} lr={args.lr} KL={args.kl} T={args.temp} "
          f"{args.iters}イテレーション×{args.games}試合", flush=True)
    wr0, n0 = duel(net, ref, vocab, deck, args.eval_games, args.batch, dev, 909001)
    wt0, _ = duel(net, ref, vocab, deck, args.eval_games, args.batch, dev, 909001,
                  temp=args.temp)
    print(f"[eval] it=-1 vs凍結094: argmax {wr0:.1%} / T={args.temp} {wt0:.1%} "
          f"(n={n0}) ← 開始時は同一なので約50%のはず", flush=True)
    hist = [(-1, wr0, wt0)]
    snaps = []
    vbuf = []
    for it in range(args.iters):
        pol = _TorchAsBatchPolicy(net, vocab, 2 * args.batch, dev)
        rolls, st = collect(pol, (deck, deck), args.games,
                            args.seed * 100000 + it * 1000,
                            batch=args.batch, temperature=args.temp,
                            ft=globals().get("FEAT"))   # **微調整対象の特徴器を使う**
        # **advantageは更新前のVで計算する**。先にVを現バッチを含むデータで更新すると
        # Vがそのリターンを覚えた分だけ A = R - V が縮み、方策勾配が体系的に小さくなる
        # （2bで「advantageが小さくて動かない」問題を既に1度踏んでいる）。
        advs = gae(vnet, rolls, vocab, dev, lam=args.lam, mb=args.mb)
        vbuf.append(to_value_rollouts(rolls))  # 軽量化して溜める
        vbuf = vbuf[-args.vbuf:]
        flat = [r for rs in vbuf for r in rs]
        vloss = value_update(vnet, vopt, flat, vocab, dev, mb_seqs=args.mb)
        s = ppo_update(net, ref, opt, rolls, vocab, dev, kl_coef=args.kl,
                       temperature=args.temp, epochs=2, mb_seqs=args.mb, advs=advs,
                       kl_break=args.kl_break)
        print(f"it{it:02d}: pg={s['pg']:+.4f} KL(SL)={s['kl']:.4f} "
              f"KL(1更新)={s.get('kl_step', 0):+.4f} ent={s['ent']:.4f} "
              f"|g|={s['gnorm']:.2f} Vloss={vloss:.4f} 生成{st['sec']:.0f}秒"
              + ("  ★安全装置作動" if s.get("broke") else ""), flush=True)
        if (it + 1) % args.eval_every == 0:
            wr, n = duel(net, ref, vocab, deck, args.eval_games, args.batch,
                         dev, 909001)
            wt, _ = duel(net, ref, vocab, deck, args.eval_games, args.batch,
                         dev, 909001, temp=args.temp)
            hist.append((it, wr, wt))
            print(f"[eval] it={it} **vs凍結094: argmax {wr:.1%} / T={args.temp} "
                  f"{wt:.1%}** (n={n}, CI±{1.96*np.sqrt(0.25/max(1,n))*100:.1f}pt)",
                  flush=True)
            sd = {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}
            meta = dict(vocab_keys=np.array(sorted(vocab), np.int64),
                        vocab_vals=np.array([vocab[k] for k in sorted(vocab)], np.int64),
                        n_board=z["n_board"], n_logs=z["n_logs"])
            np.savez_compressed(args.out, **meta, **sd)
            # **イテレーションごとに別名でも保存**。ピーク時点と最終時点を直接対戦させ、
            # 「凍結094への勝率低下」が循環（非推移）なのか本当の劣化なのかを切り分ける
            snap = args.out.replace(".npz", f"_it{it:02d}.npz")
            np.savez_compressed(snap, **meta, **sd)
            snaps.append((it, snap))
    print("\n=== 診断ゲート ===")
    print(f"{'iter':>6}{'argmax':>10}{'T=%.1f' % args.temp:>10}")
    for i, w, t in hist:
        print(f"{i:>6}{w:>10.1%}{t:>10.1%}")
    best = max(w for _, w, _ in hist)
    bestt = max(t for _, _, t in hist)
    print(f"\nargmax 最高 {best:.1%} / 開始 {wr0:.1%} / 差 {(best-wr0)*100:+.1f}pt")
    print(f"T={args.temp} 最高 {bestt:.1%} / 開始 {wt0:.1%} / 差 {(bestt-wt0)*100:+.1f}pt")
    print("（閾値: ノイズ床1.81pt×2 = 3.62pt）")
    # **スナップショット同士の直接対戦**（循環 vs 劣化の切り分け）
    if len(snaps) >= 2:
        print("\n=== スナップショット同士の直接対戦（行が列に勝った率）===")
        nets = []
        for it_, path in snaps:
            n2, _ = PolicyTorch.from_npz(MODEL, device=dev)
            zz = np.load(path, allow_pickle=False)
            sd2 = n2.state_dict()
            for k in sd2:
                if k in zz.files:
                    sd2[k].copy_(torch.as_tensor(np.asarray(zz[k], np.float32)))
            n2.load_state_dict(sd2)
            nets.append((it_, n2))
        M = {}
        print(f"{'':>8}" + "".join(f"{'it'+str(j):>10}" for j, _ in nets))
        for i, (ii, na) in enumerate(nets):
            line = f"{'it'+str(ii):>8}"
            for j, (jj, nb) in enumerate(nets):
                if i == j:
                    line += f"{'—':>10}"
                    continue
                w, _ = duel(na, nb, vocab, deck, 600, args.batch, dev, 717001)
                M[(i, j)] = w
                line += f"{w:>9.1%}"
            print(line, flush=True)
        # **対称性の検査**: 同じ2者を役割を入れ替えて測っているので (i,j)+(j,i) は
        # 100%になるはず。大きく外れるなら先後の扱いか状態管理が壊っている
        worst = max((abs(M[(i, j)] + M[(j, i)] - 1.0), i, j)
                    for i in range(len(nets)) for j in range(len(nets)) if i != j)
        print(f"  対称性検査: (i,j)+(j,i) の100%からの最大ずれ {worst[0]*100:.1f}pt "
              f"(it{nets[worst[1]][0]} vs it{nets[worst[2]][0]}) ← n=600のCI±4%程度が目安")
        print("読み方: 後のitが前のitに勝つ→前進しており094への低下は**循環**。"
              "負ける→**本当の劣化**")
    up_a = best - wr0 > 0.0181 * 2
    up_t = bestt - wt0 > 0.0181 * 2
    if up_a:
        print("判定: **argmaxが上昇** → Stage 3（リーグ）へ")
    elif up_t:
        print("判定: **T側だけ上昇** → 学習はしているがargmaxに転移していない。"
              "評価温度での学習/温度スケジュール等の対処を検討")
    else:
        print("判定: **どちらも上昇せず** → Stage 2 撤退を検討"
              "（Vを活かす後段の価値誘導探索へ）")


if __name__ == "__main__":
    main()
