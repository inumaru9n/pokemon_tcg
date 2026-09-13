"""観測のみの価値関数をスクラッチ学習する（価値誘導探索の土台）。

**なぜ要るか**: 推論時探索に載せるVは、提出物で使える情報だけで動く必要がある
（`GetHiddenData` は自前ビルド専用。本番では相手の手札は見えない）。
リーグ用のCriticは特権情報つきで学習しており、そのまま伏せると性能が落ちる。

**なぜスクラッチか**: 特権ありCriticを初期値に観測のみで微調整した版（E2のB条件）は、
turn5+ AUC 0.7192、しかも**ターン20以降でAUC 0.203（予測が逆向き）**だった。
原因は当時のマスクがゼロ埋めで「相手の山が空」という偽の手掛かりを作っていたこと。
未知トークン方式に直したうえで、特権ありの重みを引きずらないようスクラッチで学習する。

**合否（事前登録）**:
  - turn5以降の勝敗AUC ≥ 0.75
  - **全ターン帯でAUC ≥ 0.5**（特定区間の逆転を検出する。前回はこれが無くて見逃した）
  - ターン帯でおおむね単調増加

usage:
  python selfplay/train_critic_obs.py --games 20000 --epochs 6 --out value_obs.npz
"""

from __future__ import annotations

import argparse
import importlib.util as ilu
import os
import sys
import time

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "arena")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def load_ft(agent_dir: str, mod: str):
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
    ap.add_argument("--games", type=int, default=20000,
                    help="生成試合数。EXP-095の実測ではVは2万試合規模で飽和する")
    ap.add_argument("--chunk", type=int, default=2000, help="1回の生成単位")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--mb", type=int, default=24)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=20260801)
    ap.add_argument("--out", default="/kaggle/working/value_obs.npz")
    args = ap.parse_args()

    from selfplay.batch_policy import BatchLSTMPolicy
    from selfplay.eval_critic import auc
    from selfplay.league_rollout import Opponent, collect
    from selfplay.state_encoder import encode
    from selfplay.value_net import (ValueNetV2, _pack, to_value_trajs,
                                    value_update_v2)
    import run_match as rm

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    main_dir = os.path.join(_ROOT, args.main_dir)
    main_ft = load_ft(main_dir, args.main_feat)
    main_deck = rm.read_deck(main_dir)
    pol = BatchLSTMPolicy(os.path.join(main_dir, args.main_model), 2 * args.batch)

    # **訓練分布は本番の相手分布に近づける**。Vは探索の葉を評価するので、
    # 実戦で出会う盤面の上で正確であってほしい。
    ops = [Opponent("self", "self", pol, main_ft, main_deck, 30.0, temp=args.temp)]
    for d, mod, mdl, nm, wt in (
            ("agents/101_marnie_luca", "feat_101", "model_101.npz", "101_marnie", 60.0),
            ("agents/098_spidops_gen", "feat_098", "model_098.npz", "098_spidops", 10.0)):
        ad = os.path.join(_ROOT, d)
        ops.append(Opponent(nm, "ext",
                            BatchLSTMPolicy(os.path.join(ad, mdl), args.batch),
                            load_ft(ad, mod), rm.read_deck(ad), wt, temp=args.temp))

    def enc_obs(ptr, obs):
        return encode(ptr, obs, privileged=False)

    print(f"device={dev} / 生成 {args.games:,}試合（観測のみV・スクラッチ）", flush=True)
    trajs = []
    t0 = time.perf_counter()
    done = 0
    while done < args.games:
        k = min(args.chunk, args.games - done)
        tj, st = collect(pol, main_ft, main_deck, ops, k,
                         args.seed + done, batch=args.batch,
                         temperature=args.temp, encode_state=enc_obs)
        trajs += to_value_trajs(tj)
        done += k
        print(f"  {done:,}/{args.games:,}試合 系列{len(trajs):,} "
              f"{time.perf_counter()-t0:.0f}秒 勝率{st['wins']}/{st['games']}",
              flush=True)

    # **ゲーム単位で train/val 分割**（決定単位だと同じ試合が両方に入りリークする）
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(trajs))
    n_val = max(1, int(len(trajs) * args.val_frac))
    val = [trajs[i] for i in idx[:n_val]]
    tr = [trajs[i] for i in idx[n_val:]]
    print(f"train {len(tr):,}系列 / val {len(val):,}系列", flush=True)

    vnet = ValueNetV2().to(dev)
    vopt = torch.optim.Adam(vnet.parameters(), lr=args.lr)

    def evaluate():
        ys, ps, tn = [], [], []
        vnet.eval()
        with torch.no_grad():
            for s0 in range(0, len(val), args.mb):
                sub = val[s0:s0 + args.mb]
                b = _pack(sub, dev)
                lg, _ = vnet(b["zone"], b["attr"], b["prog"], b["logs"],
                             b["ctx"], b["prev_cidx"], b["prev_num"])
                p = torch.sigmoid(lg).float().cpu().numpy()
                for i, r in enumerate(sub):
                    n = len(r.zone)
                    ys.extend([1 if r.reward > 0 else 0] * n)
                    ps.extend(p[i, :n].tolist())
                    tn.extend(list(np.asarray(r.prog)[:n, 0].astype(int)))
        vnet.train()
        return np.array(ys), np.array(ps), np.array(tn)

    for ep in range(args.epochs):
        loss = value_update_v2(vnet, vopt, tr, dev, epochs=1, mb_seqs=args.mb)
        ys, ps, tn = evaluate()
        a5 = auc(ys[tn >= 5], ps[tn >= 5])
        print(f"epoch{ep}: loss={loss:.4f} 全体AUC={auc(ys, ps):.4f} "
              f"turn5+AUC={a5:.4f}", flush=True)

    ys, ps, tn = evaluate()
    print(f"\n{'ターン帯':<12}{'n':>8}{'AUC':>8}{'平均予測':>10}{'実勝率':>8}")
    bad = []
    for lo, hi in ((1, 4), (5, 9), (10, 14), (15, 19), (20, 99)):
        m = (tn >= lo) & (tn <= hi)
        if m.sum() < 50:
            continue
        a = auc(ys[m], ps[m])
        print(f"{f'{lo}-{hi}':<12}{int(m.sum()):>8}{a:>8.4f}{ps[m].mean():>10.3f}"
              f"{ys[m].mean():>8.3f}")
        if a < 0.5:
            bad.append((lo, hi, a))
    a5 = auc(ys[tn >= 5], ps[tn >= 5])
    print(f"\n**turn5以降 AUC {a5:.4f}**（合格 ≥0.75）")
    print(f"**全ターン帯 AUC ≥0.5**: {'OK' if not bad else f'NG {bad}'}")
    sd = {k: v.detach().cpu().numpy() for k, v in vnet.state_dict().items()}
    np.savez_compressed(args.out, **sd)
    print(f"保存 -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
