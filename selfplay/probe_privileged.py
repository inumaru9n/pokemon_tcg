"""特権情報がCriticの出力をどれだけ振らせるかを測る（仮説④の直接検証）。

**仮説④**: 特権ベースラインは不偏だが、**分散削減の意味で最適とは限らない**。
分散削減の最適ベースラインは観測条件付き E[Q|o] であって状態条件付き V(s) ではない。
Criticが相手の手札を見ていると、**同じ観測 o に対して隠れ情報次第で V が振れ**、
Actorが条件付けられない成分（相手の事故手札で勝った/負けた）が advantage に乗る。

**測り方**: ある決定の状態から、相手の手札を「観測と矛盾しない別の手札」に差し替える。
相手の未公開領域（手札 ∪ 山∪サイド）は、我々から見れば区別がつかないので、
そこから同じ枚数を引き直した手札は**同じ観測に対応する別の隠れ状態**になる。
その K 通りで V を計算し、ばらつきを見る。

  σ_hidden = 同一観測内での V の標準偏差（＝Actorが条件付けられないノイズ）
  σ_total  = 状態間での V の標準偏差（＝Vが持つ情報の全体）

**σ_hidden が σ_total に対して大きいほど、advantage は Actor の行動と無関係に振れる**。

usage:
  python selfplay/probe_privileged.py --ckpt ckpt.pt --games 40 --resample 8
"""

from __future__ import annotations

import argparse
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
    if agent_dir not in sys.path:
        sys.path.insert(0, agent_dir)
    spec = ilu.spec_from_file_location(mod, os.path.join(agent_dir, mod + ".py"))
    m = ilu.module_from_spec(spec)
    sys.modules[mod] = m
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--main-dir", default="agents/099_alakazam_gen")
    ap.add_argument("--main-feat", default="feat_099")
    ap.add_argument("--main-model", default="model_099.npz")
    ap.add_argument("--opp-dir", default="agents/101_marnie_luca")
    ap.add_argument("--opp-feat", default="feat_101")
    ap.add_argument("--opp-model", default="model_101.npz")
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--resample", type=int, default=8, help="1状態あたりの隠れ状態の数")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=4242)
    args = ap.parse_args()

    from selfplay.batch_policy import BatchLSTMPolicy
    from selfplay.league_rollout import Opponent, collect
    from selfplay.state_encoder import encode
    from selfplay.state_spec import ZONE_CAP
    from selfplay.value_net import ValueNetV2, _pack
    import run_match as rm

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    main_dir = os.path.join(_ROOT, args.main_dir)
    opp_dir = os.path.join(_ROOT, args.opp_dir)
    main_ft = load_ft(main_dir, args.main_feat)
    opp_ft = load_ft(opp_dir, args.opp_feat)
    main_deck, opp_deck = rm.read_deck(main_dir), rm.read_deck(opp_dir)

    pol = BatchLSTMPolicy(os.path.join(main_dir, args.main_model), 2 * args.batch)
    ops = [Opponent("101", "ext",
                    BatchLSTMPolicy(os.path.join(opp_dir, args.opp_model), args.batch),
                    opp_ft, opp_deck, 100.0, temp=0.0)]
    trajs, st = collect(pol, main_ft, main_deck, ops, args.games, args.seed,
                        batch=args.batch, temperature=0.7, encode_state=encode)
    print(f"生成 {st['games']}試合 / 決定 {st['decisions']:,}", flush=True)

    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    vnet = ValueNetV2().to(dev)
    vnet.load_state_dict(ck["vnet"])
    vnet.eval()

    rng = np.random.default_rng(7)

    def resample_hidden(zone, opp_idx):
        """相手の手札を、未公開領域（手札∪山∪サイド）から同枚数引き直す。

        我々の観測からは手札と山の区別がつかないので、これは**同じ観測に対応する
        別の隠れ状態**になる（枚数は保存する）。
        """
        z = zone.copy()
        hand = z[opp_idx, 0]
        deck = z[opp_idx, 1]
        nh = int((hand > 0).sum())
        pool = np.concatenate([hand[hand > 0], deck[deck > 0]])
        if len(pool) <= nh:
            return z
        perm = rng.permutation(len(pool))
        z[opp_idx, 0] = 0
        z[opp_idx, 1] = 0
        z[opp_idx, 0, :nh] = pool[perm[:nh]]
        rest = pool[perm[nh:]][:ZONE_CAP]
        z[opp_idx, 1, :len(rest)] = rest
        return z

    # 1系列ずつ、K通りの隠れ状態でVを計算する（LSTMがあるので系列単位で処理）
    sig_hidden, means = [], []
    n_used = 0
    with torch.no_grad():
        for r in trajs[:120]:
            n = len(r.zone)
            if n < 5:
                continue
            base = np.asarray(r.zone)
            # main の席は試合ごとに交替する。prog[1] は「いま手番のプレイヤー」＝main。
            me = int(np.asarray(r.prog)[0][1])
            variants = []
            for k in range(args.resample):
                zz = np.stack([resample_hidden(base[t], 1 - me) for t in range(n)])
                variants.append(zz)

            class _V:
                pass
            vs = []
            for zz in variants:
                o = _V()
                o.zone, o.attr, o.prog = zz, r.attr, r.prog
                o.logs, o.prev, o.ctx, o.reward = r.logs, r.prev, r.ctx, r.reward
                b = _pack([o], dev)
                logit, _ = vnet(b["zone"], b["attr"], b["prog"], b["logs"],
                                b["ctx"], b["prev_cidx"], b["prev_num"])
                vs.append((2.0 * torch.sigmoid(logit) - 1.0)[0, :n].cpu().numpy())
            vs = np.stack(vs)                      # (K, n)
            sig_hidden.append(vs.std(0))           # 各決定での隠れ状態による振れ
            means.append(vs.mean(0))
            n_used += n

    sh = np.concatenate(sig_hidden)
    mu = np.concatenate(means)
    print(f"\n評価した決定 {n_used:,}（{len(sig_hidden)}系列 × {args.resample}通りの隠れ状態）")
    print(f"σ_hidden（同一観測内でのVの標準偏差）  平均 {sh.mean():.4f} / 中央値 {np.median(sh):.4f}")
    print(f"σ_total （状態間でのVの標準偏差）      {mu.std():.4f}")
    print(f"**比率 σ_hidden/σ_total = {sh.mean()/max(1e-9, mu.std()):.3f}**")
    print()
    print("読み方: この比率が大きいほど、Vの変動のうち**Actorが条件付けられない成分**が")
    print("大きい。advantage A=R-V にそのノイズが乗るので、方策は無関係な方向に押される。")
    print("0.3を超えるようなら特権情報は分散削減として逆効果の可能性が高い。")


if __name__ == "__main__":
    main()
