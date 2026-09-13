"""AlphaStar型リーグ（I-109 Stage 3）。exploiter無しの構成。

    更新対象 = main 1体のみ（099初期）
    訓練相手 = 自己/過去snapshot/SL初期 + **忠実度ゲート合格の外部レプリカのみ**
               （101 Marnie 35 / 131 Ogerpon 5 / 098 Spidops 2）

**Stage 2で残った問題への直接の答え**: Stage 2は純ミラー自己対戦だったため
「自分の過去には単調に勝つが固定参照への改善は確認できない」で終わった。ここでは
SL初期と外部レプリカ（本番メタの61.1%と8.4%）が目的関数に入る。

**外部レプリカを訓練に入れてよい根拠**: EXP-096で忠実度ゲートを通した2体だけを使う
（101=乖離+1.1pt / 098=+0.5pt）。EXP-095の「不忠実な相手にRLがbest-responseすると
偽の弱点を突く」という設計判断は、ゲート通過枠には当たらない。

**評価**: プールは使わない（101/098を訓練に使う以上、それを含むプール評価は in-sample）。
  - 学習内: vs 099（SL初期）を argmax・CRN。選定用エンジンシード
  - 最終判定: **別エンジンシード**で再測定（チェックポイントを選ぶので winner's curse を切る）
  - ホールドアウト: 101-s1/s2、098-s1/s2（**訓練に一度も使わない別学習シードの同一個体**。
    重み固有の癖への過学習を検出する。行動一致率82.7%＝6決定に1回は違う手を打つので
    検出力がある）

usage（Kaggle Notebook想定。torchとcgの両方が要る）:
  uv run python selfplay/league.py --iters 20 --games 400 --eval-games 400
"""

from __future__ import annotations

import argparse
import copy
import importlib.util as ilu
import os
import sys

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "arena")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def load_ft(agent_dir: str, mod: str):
    """エージェントの特徴器をモジュール名衝突なしで読む。"""
    if agent_dir not in sys.path:
        sys.path.insert(0, agent_dir)
    spec = ilu.spec_from_file_location(mod, os.path.join(agent_dir, mod + ".py"))
    m = ilu.module_from_spec(spec)
    sys.modules[mod] = m
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--main-dir", default="agents/099_alakazam_gen")
    ap.add_argument("--main-feat", default="feat_099")
    ap.add_argument("--main-model", default="model_099.npz")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--games", type=int, default=400, help="1イテレーションの生成試合数")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--temp", type=float, default=0.7, help="mainの探索温度")
    ap.add_argument("--opp-temp", type=float, default=0.5,
                    help="凍結相手の温度。**決定的な相手は手順を丸暗記される**"
                         "（外部レプリカ・過去snapshot・SL初期に同じ値を使う）")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--vlr", type=float, default=3e-4)
    ap.add_argument("--kl", type=float, default=0.05, help="SL初期へのKLアンカー係数")
    ap.add_argument("--kl-break", type=float, default=0.05, help="1更新のKL安全装置")
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--mb", type=int, default=24)
    ap.add_argument("--vbuf", type=int, default=6, help="Criticが使う直近イテレーション数")
    ap.add_argument("--critic-warmup", type=int, default=0,
                    help="最初のNイテレーションは**Criticだけ更新し方策は動かさない**。"
                         "Vが無情報だとadvantageが実質±1のコインフリップになり方策勾配の"
                         "信号が消えるため（Stage 2bで実証）。EXP-095の実測では"
                         "Vは2万試合規模で飽和するので、games×N が2万になる値にする")
    ap.add_argument("--snap-every", type=int, default=2)
    ap.add_argument("--max-past", type=int, default=8)
    ap.add_argument("--eval-every", type=int, default=2)
    ap.add_argument("--eval-games", type=int, default=400)
    ap.add_argument("--eval-seed", type=int, default=90001, help="選定用エンジンシード")
    ap.add_argument("--eval-vs", default="sl",
                    help='評価相手。"sl"=SL初期099 / "101" / "098"。'
                         '**単一相手で訓練するときは同じ相手を評価に使う**'
                         '（訓練と評価の指標を揃えないと何を改善したのか読めない）')
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="/kaggle/working/league")
    ap.add_argument("--opp-mix", default="",
                    help='相手構成を上書きする。例 "101:100" / "self:50,101:50"。'
                         '**単一相手にすると判定指標が1本になり解像度が上がる**'
                         '（混合だと改善が相手ごとに分かれて埋もれる）。'
                         '既定は self30/past20/sl10/101:35/098:5')
    ap.add_argument("--no-critic", action="store_true",
                    help="**Criticを使わない**。--group と併用し、同一CRN群の平均を"
                         "baselineにする（GRPO/RLOO型）。ウォームアップ2万試合が不要で、"
                         "特権情報も要らない。失うのは per-step の credit assignment")
    ap.add_argument("--group", type=int, default=1,
                    help="CRN群のサイズ。同一シード・同一相手・同一先後で複製する数")
    ap.add_argument("--no-privileged", action="store_true",
                    help="Criticから**相手の隠れ情報を伏せる**（手札・山∪サイドをゼロに）。"
                         "特権ベースラインは不偏だが**分散削減の意味で最適とは限らない**: "
                         "同じ観測に対し隠れ情報次第でVが動くと、Actorが条件付けられない"
                         "ノイズがadvantageに乗る。その影響を切り分けるための対照条件")
    ap.add_argument("--init-critic", default="",
                    help="**Criticだけ**をチェックポイントから読む（方策はSL初期のまま）。"
                         "KL係数の切り分けのように条件を変えて複数回走らせるとき、"
                         "毎回ウォームアップ2万試合をやり直さずに済む。"
                         "--resume と違い方策・optimizer・過去snapshotは復元しない")
    ap.add_argument("--resume", default="",
                    help="チェックポイント(.pt)から再開する。**方策だけでなく Critic と"
                         "過去snapshotも復元する** — Criticを捨てるとウォームアップの"
                         "2万試合が無駄になり、advantageが無情報の状態から再出発する")
    args = ap.parse_args()
    if args.no_critic:
        # criticなしは群相対baselineが前提。group=1だと A が常に0になり学習しない
        assert args.group > 1, "--no-critic は --group 2 以上と併用すること"
        args.critic_warmup = 0        # ウォームアップは不要
        print(f"**Criticなし**: CRN群サイズ {args.group} の群相対advantageを使う")

    from selfplay.batch_policy import BatchLSTMPolicy
    from selfplay.league_rollout import Opponent, collect
    from selfplay.ppo import PolicyTorch, ppo_update
    from selfplay.ppo_probe import _TorchAsBatchPolicy
    from selfplay.state_encoder import encode
    from selfplay.value_net import (ValueNetV2, gae_v2, group_advantage,
                                    to_value_trajs,
                                    value_update_v2)
    import run_match as rm

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    main_dir = os.path.join(_ROOT, args.main_dir)
    main_ft = load_ft(main_dir, args.main_feat)
    main_deck = rm.read_deck(main_dir)
    model_path = os.path.join(main_dir, args.main_model)

    # --- main（学習対象）と SL初期（凍結・KLアンカー先） ---
    net, vocab = PolicyTorch.from_npz(model_path, device=dev)
    ref, _ = PolicyTorch.from_npz(model_path, device=dev)
    for p in ref.parameters():
        p.requires_grad_(False)
    ref.eval()
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    main_pol = _TorchAsBatchPolicy(net, vocab, 2 * args.batch, dev)
    sl_pol = _TorchAsBatchPolicy(ref, vocab, args.batch, dev)

    # --- Critic（対称・特権つき。スクラッチ学習） ---
    # **logs の次元は main の特徴器から取る**。既定48は099時代の値で、
    # 117(53次元)では critic の入力が5次元ずれて実行時に落ちる（実際に踏んだ）。
    _nlogs = int(np.load(os.path.join(_ROOT, args.main_dir, args.main_model),
                         allow_pickle=False)["n_logs"])
    print(f"critic の n_logs = {_nlogs}")
    vnet = ValueNetV2(n_logs=_nlogs).to(dev)
    vopt = torch.optim.Adam(vnet.parameters(), lr=args.vlr)
    vbuf: list[list] = []

    # --- 外部レプリカ（ゲート通過済みの2体。訓練にはs3、評価にはs1/s2を温存） ---
        # **訓練相手は忠実度ゲート（本番の個体/アーキタイプ値と±5pt）を通った枠だけ**。
    # 不忠実な相手を訓練分布に入れると、RLがその偽の弱点にbest-responseする（EXP-095）。
    # 重みは 07-30〜08-03 メタのシェア比（合計58.2%をカバー）。
    #   101 Marnie   48.3%  乖離 +1.1pt  ✅
    #   131 Ogerpon   7.2%  乖離 +2.6/+1.2pt ✅（EXP-110で新規作成）
    #   098 Spidops   2.7%  乖離 +0.5pt  ✅
    # **除外**: 130 Lopunny(w9.7, 乖離+32.1pt) / 084 Alakazam(+39.1) /
    #          085 Kangaskhan(+19.9) / 041 Garchomp(+22.8) / 021 Dragapult(+30.3)
    #          ＝ 全て軽量逆設計で本番より20〜39pt弱い。訓練に入れてはいけない
    ext_spec = [("agents/101_marnie_luca", "feat_101", "model_101.npz",
                 "101_marnie", 35.0),
                ("agents/131_ogerpon_majkel", "feat_131", "model_131.npz",
                 "131_ogerpon", 5.0),
                ("agents/098_spidops_gen", "feat_098", "model_098.npz",
                 "098_spidops", 2.0)]
    externals = []
    for d, mod, mdl, nm, wt in ext_spec:
        ad = os.path.join(_ROOT, d)
        _o = Opponent(nm, "ext",
                      BatchLSTMPolicy(os.path.join(ad, mdl), args.batch),
                      load_ft(ad, mod), rm.read_deck(ad), wt, temp=args.opp_temp)
        _o.model_path = os.path.join(ad, mdl)   # 評価時に別インスタンスを作る用
        externals.append(_o)

    past: list[Opponent] = []

    ext_by_name = {}

    def build_league():
        """相手の抽選テーブル。過去snapshotが無い初期は自己に振り替える。"""
        if args.opp_mix:
            # 例 "101:100" / "self:50,101:50"。名前は self / sl / past / 101 / 098
            sel = []
            for part in args.opp_mix.split(","):
                nm, _, w = part.partition(":")
                nm, w = nm.strip(), float(w or 100)
                if nm == "self":
                    sel.append(Opponent("self", "self", main_pol, main_ft,
                                        main_deck, w, temp=args.temp))
                elif nm == "sl":
                    sel.append(Opponent("sl_base", "sl", sl_pol, main_ft,
                                        main_deck, w, temp=args.opp_temp))
                elif nm == "past":
                    if past:
                        for p_ in past:
                            p_.weight = w / len(past)
                        sel += past
                else:
                    hit = [e for e in externals if e.name.startswith(nm)]
                    assert hit, f"相手 {nm} が見つからない"
                    hit[0].weight = w
                    sel.append(hit[0])
            return sel
        # **self は main と同じ温度にする**。collect は is_main でない側に opp.temp を
        # 使うので、既定のままだと main(0.7) より鈍い自分と戦うことになり自己対戦が
        # 非対称になる（実測: 自己対戦の勝率が45〜48%に偏った）。
        ops = [Opponent("self", "self", main_pol, main_ft, main_deck, 30.0,
                        temp=args.temp),
               Opponent("sl_base", "sl", sl_pol, main_ft, main_deck, 10.0,
                        temp=args.opp_temp)]
        ops += externals
        if past:
            w = 20.0 / len(past)
            for p in past:
                p.weight = w
            ops += past
        else:
            ops[0].weight += 20.0
        return ops

    def snapshot(it):
        frozen = copy.deepcopy(net).to(dev)
        for p in frozen.parameters():
            p.requires_grad_(False)
        frozen.eval()
        pol = _TorchAsBatchPolicy(frozen, vocab, args.batch, dev)
        past.append(Opponent(f"past{it}", "past", pol, main_ft, main_deck, 0.0,
                             temp=args.opp_temp))
        while len(past) > args.max_past:
            past.pop(0)

    @torch.no_grad()
    def evaluate(seed_base, n_games):
        """**argmax同士**で評価する（探索温度は学習用であって評価用ではない）。"""
        ev_main = _TorchAsBatchPolicy(net, vocab, 2 * args.batch, dev)
        if args.eval_vs == "sl":
            ev_opp = _TorchAsBatchPolicy(ref, vocab, args.batch, dev)
            ops = [Opponent("sl_base", "sl", ev_opp, main_ft, main_deck, 1.0, temp=0.0)]
        else:
            hit = [e for e in externals if e.name.startswith(args.eval_vs)]
            assert hit, f"評価相手 {args.eval_vs} が見つからない"
            e0 = hit[0]
            ops = [Opponent(e0.name, "ext",
                            BatchLSTMPolicy(e0.model_path, args.batch),
                            e0.ft, e0.deck, 1.0, temp=0.0)]
        _, st = collect(ev_main, main_ft, main_deck, ops, n_games, seed_base,
                        batch=args.batch, temperature=0.0)
        return st["wins"] / max(1, st["games"]), st["games"]

    # 推論形式（main.pyが読める形）で保存するためのメタ情報を、初期モデルから引き継ぐ。
    # **state_dictだけ保存すると main.py が読めない**（vocab/feat_keys/classes/
    # trained_ctx が要る）。ここを忘れるとリーグの成果物をエージェントにできない。
    _z0 = np.load(model_path, allow_pickle=False)
    _meta = {k: _z0[k] for k in ("vocab_keys", "vocab_vals", "n_board", "n_logs",
                                 "n_optdim", "feat_keys", "classes", "trained_ctx")
             if k in _z0.files}
    # classes は古い学習で作ったモデルには入っていない。**特徴器から補える**
    # （順序はソート済みタプルなので学習時と同一。main.py側の突合が働くようにする）
    if "classes" not in _meta and getattr(main_ft, "CLASSES", None):
        _meta["classes"] = np.array([str(x) for x in main_ft.CLASSES], dtype="U")
        print(f"初期モデルに classes が無いので特徴器から補完（{len(main_ft.CLASSES)}個）",
              flush=True)
    for _k in ("feat_keys", "classes"):
        if _k not in _meta:
            print(f"警告: {_k} が無い。エクスポートしたnpzはmain.py側の整合チェックを"
                  "通らない", file=sys.stderr)

    def export(path):
        """純numpy推論用（main.py が読む形式）で保存する。"""
        sd = {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}
        np.savez_compressed(path, **_meta, **sd)

    def save_ckpt(path, it):
        """**継ぎ足せる形**で保存する。Criticとoptimizerと過去snapshotを含む。"""
        torch.save({"it": it,
                    "net": net.state_dict(), "opt": opt.state_dict(),
                    "vnet": vnet.state_dict(), "vopt": vopt.state_dict(),
                    "past": [p.policy.net.state_dict() for p in past]}, path)

    start_it = 0
    if args.init_critic and os.path.exists(args.init_critic):
        ck = torch.load(args.init_critic, map_location=dev, weights_only=False)
        vnet.load_state_dict(ck["vnet"])
        vopt.load_state_dict(ck["vopt"])
        print(f"Criticのみ復元（it={ck.get('it')}）。方策はSL初期から開始", flush=True)
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location=dev, weights_only=False)
        net.load_state_dict(ck["net"])
        opt.load_state_dict(ck["opt"])
        vnet.load_state_dict(ck["vnet"])
        vopt.load_state_dict(ck["vopt"])
        for sd_p in ck.get("past", []):
            frozen = copy.deepcopy(net).to(dev)
            frozen.load_state_dict(sd_p)
            for q in frozen.parameters():
                q.requires_grad_(False)
            frozen.eval()
            past.append(Opponent(f"past_r{len(past)}", "past",
                                 _TorchAsBatchPolicy(frozen, vocab, args.batch, dev),
                                 main_ft, main_deck, 0.0, temp=args.opp_temp))
        start_it = int(ck.get("it", -1)) + 1
        print(f"再開: it{start_it} から / 過去snapshot {len(past)}体 復元", flush=True)

    # --no-privileged: 相手の隠れゾーン（手札・山∪サイド）をゼロにする。
    # 自分側と相手の公開ゾーン（場・トラッシュ）はそのまま＝Actorが見えている情報。
    _enc = encode
    if args.no_privileged:
        def _enc(ptr, obs, _base=encode):
            # **ゼロ埋めにしない**。ゼロは「空」を意味し、相手の山が空＝山札切れ＝
            # こちらの勝ち という偽の手掛かりになる（実測でAUC 0.203まで逆転した）。
            # state_encoder が「未知カードN枚」で表現する経路を使う。
            return _base(ptr, obs, privileged=False)
        print("Criticから相手の隠れ情報を伏せる（未知トークンN枚で表現）", flush=True)

    print(f"device={dev} / main={args.main_dir} / {args.iters}イテレーション"
          f"×{args.games}試合", flush=True)
    for it in range(start_it, args.iters):
        ops = build_league()
        trajs, st = collect(main_pol, main_ft, main_deck, ops, args.games,
                            args.seed * 100000 + it * 1000, batch=args.batch,
                            temperature=args.temp, encode_state=_enc,
                            group=args.group)
        if not trajs:
            print("軌跡が空。中断", flush=True)
            break
        if it == start_it:
            # **特権ブロックが実際に埋まっているかを1回だけ検査する**。
            # GetHiddenData が無い環境では state_encoder が例外を出さずに
            # 公開情報だけになる（無言で劣化する）ので、必ず可視化する。
            _z = np.asarray(trajs[0].zone)
            _cards = [(int((_z[:, p, k] > 0).sum(1).mean())) for p in (0, 1)
                      for k in range(_z.shape[2])]
            print(f"      [特権検査] 1決定あたりのカード枚数 "
                  f"p0(手/山/場/捨)={_cards[:4]} p1={_cards[4:]} "
                  f"→ 手札が0なら特権情報が入っていない", flush=True)
        # 1) **更新前のV**でadvantage（更新後だとA=R-Vが縮む）
        if args.group > 1 and args.no_critic:
            # **criticなし**: 同一CRN群の平均を baseline にする（配札分散を厳密に除去）
            advs = group_advantage(trajs)
        else:
            advs = gae_v2(vnet, trajs, dev, lam=args.lam, mb=args.mb)
        # 2) Criticを更新（直近vbufイテレーションのバッファ）
        if args.no_critic:
            vloss, flat = 0.0, []   # **criticを一切学習しない**（計算もメモリも省く）
        else:
            vbuf.append(to_value_trajs(trajs))
            vbuf = vbuf[-args.vbuf:]
            flat = [r for rs in vbuf for r in rs]
            vloss = value_update_v2(vnet, vopt, flat, dev, mb_seqs=args.mb)
        # 3) Actorを更新（PPO + SL初期へのKL）。ウォームアップ中は方策を動かさない
        if it < args.critic_warmup:
            s = {"pg": 0.0, "kl": 0.0, "ent": 0.0, "gnorm": 0.0}
            print(f"it{it}: [Criticウォームアップ] Vloss={vloss:.4f} "
                  f"勝率{st['wins']}/{st['games']} 生成{st['sec']:.0f}秒 "
                  f"buf={len(flat)}", flush=True)
        else:
            s = ppo_update(net, ref, opt, trajs, vocab, dev, kl_coef=args.kl,
                           temperature=args.temp, epochs=2, mb_seqs=args.mb,
                           advs=advs, kl_break=args.kl_break)
        by = " ".join(f"{k}:{v[1]}/{v[0]}" for k, v in sorted(st["by_opp"].items()))
        if it >= args.critic_warmup:
            print(f"it{it}: pg={s['pg']:+.4f} KL={s['kl']:.4f} ent={s['ent']:.4f} "
                  f"klstep={s.get('kl_step', 0.0):+.4f} 打切{int(s.get('broke', 0))} "
                  f"|g|={s['gnorm']:.2f} Vloss={vloss:.4f} "
                  f"勝率{st['wins']}/{st['games']} 生成{st['sec']:.0f}秒 "
                  f"buf={len(flat)}", flush=True)
        print(f"      相手別 {by}", flush=True)
        if (it + 1) % args.snap_every == 0:
            snapshot(it)
        if (it + 1) % args.eval_every == 0:
            wr, n = evaluate(args.eval_seed, args.eval_games)
            print(f"      [評価] vs {args.eval_vs} {wr:.1%} (n={n})", flush=True)
            export(os.path.join(args.out, f"main_it{it}.npz"))
            save_ckpt(os.path.join(args.out, "ckpt.pt"), it)
    export(os.path.join(args.out, "main_final.npz"))
    save_ckpt(os.path.join(args.out, "ckpt.pt"), args.iters - 1)
    print("完了", flush=True)


if __name__ == "__main__":
    main()
