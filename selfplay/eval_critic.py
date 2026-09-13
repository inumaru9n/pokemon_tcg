"""リーグで学習したCritic（ValueNetV2）の予測性能を測る（I-109 Stage 3の検証）。

**なぜ要るか**: Stage 1は「turn5+ の勝敗AUC ≥ 0.75 かつターン帯で単調増加」という
事前登録キルゲートでCriticを検証した。V2（対称・特権情報つき）ではリーグを回す前に
同等の検証をしておらず、`Vloss` の数字しか見ていない。**Vが無情報ならadvantageが
実質±1のコインフリップになり方策勾配の信号が消える**ので、方策側の測定を解釈する前に
ここを固める必要がある。

usage（torchが要るのでKaggle側）:
  python selfplay/eval_critic.py --ckpt league/ckpt.pt --games 300
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


def auc(y, s):
    """ROC-AUC（sklearn非依存。順位和で計算）。"""
    y = np.asarray(y, np.int8)
    s = np.asarray(s, np.float64)
    n1, n0 = int(y.sum()), int((1 - y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    # 同点は平均順位に均す
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt))
    np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True, help="league の ckpt.pt")
    ap.add_argument("--main-dir", default="agents/099_alakazam_gen")
    ap.add_argument("--main-feat", default="feat_099")
    ap.add_argument("--main-model", default="model_099.npz")
    ap.add_argument("--games", type=int, default=300)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=777001)
    ap.add_argument("--no-privileged", action="store_true",
                    help="相手の隠れゾーンを伏せて評価する。**提出物では GetHiddenData が"
                         "使えない**ので、推論時探索に載せるVはこの条件で測らないと意味がない")
    args = ap.parse_args()

    from selfplay.batch_policy import BatchLSTMPolicy
    from selfplay.league_rollout import Opponent, collect
    from selfplay.ppo import PolicyTorch
    from selfplay.ppo_probe import _TorchAsBatchPolicy
    from selfplay.state_encoder import encode
    from selfplay.value_net import ValueNetV2, _pack
    import run_match as rm

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    main_dir = os.path.join(_ROOT, args.main_dir)
    main_ft = load_ft(main_dir, args.main_feat)
    main_deck = rm.read_deck(main_dir)
    net, vocab = PolicyTorch.from_npz(
        os.path.join(main_dir, args.main_model), device=dev)

    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    net.load_state_dict(ck["net"])
    vnet = ValueNetV2().to(dev)
    vnet.load_state_dict(ck["vnet"])
    vnet.eval()
    print(f"device={dev} / ckpt it={ck.get('it')} / 過去snapshot {len(ck.get('past', []))}体",
          flush=True)

    # **訓練と同じ相手構成で評価する**（Vは訓練分布上で使われるため）
    pol = _TorchAsBatchPolicy(net, vocab, 2 * args.batch, dev)
    ops = [Opponent("self", "self", pol, main_ft, main_deck, 30.0)]
    for d, mod, mdl, nm, wt in (
            ("agents/101_marnie_luca", "feat_101", "model_101.npz", "101_marnie", 35.0),
            ("agents/098_spidops_gen", "feat_098", "model_098.npz", "098_spidops", 5.0)):
        ad = os.path.join(_ROOT, d)
        ops.append(Opponent(nm, "ext", BatchLSTMPolicy(os.path.join(ad, mdl), args.batch),
                            load_ft(ad, mod), rm.read_deck(ad), wt, temp=0.5))
    _enc = encode
    if args.no_privileged:
        def _enc(ptr, obs, _b=encode):
            return _b(ptr, obs, privileged=False)
        print("観測のみ（未知トークンN枚で表現）で評価", flush=True)
    trajs, st = collect(pol, main_ft, main_deck, ops, args.games, args.seed,
                        batch=args.batch, temperature=args.temp, encode_state=_enc)
    print(f"生成 {st['games']}試合 / 決定 {st['decisions']:,} / "
          f"勝率 {st['wins']}/{st['games']}", flush=True)

    ys, ps, turns = [], [], []
    with torch.no_grad():
        for s0 in range(0, len(trajs), 24):
            sub = trajs[s0:s0 + 24]
            b = _pack(sub, dev)
            logits, _ = vnet(b["zone"], b["attr"], b["prog"], b["logs"], b["ctx"],
                             b["prev_cidx"], b["prev_num"])
            p = torch.sigmoid(logits).float().cpu().numpy()
            for i, r in enumerate(sub):
                n = len(r.zone)
                ys.extend([1 if r.reward > 0 else 0] * n)
                ps.extend(p[i, :n].tolist())
                turns.extend(list(r.turn[:n]))
    ys, ps, turns = np.array(ys), np.array(ps), np.array(turns)
    print(f"\n全体 AUC {auc(ys, ps):.4f} (n={len(ys):,})")
    print(f"**turn5以降 AUC {auc(ys[turns >= 5], ps[turns >= 5]):.4f}** "
          f"(n={int((turns>=5).sum()):,})  ← Stage 1のキルゲートは ≥0.75")
    print(f"\n{'ターン帯':<12}{'n':>8}{'AUC':>8}{'平均予測':>10}{'実勝率':>8}")
    for lo, hi in ((1, 4), (5, 9), (10, 14), (15, 19), (20, 99)):
        m = (turns >= lo) & (turns <= hi)
        if m.sum() < 50:
            continue
        print(f"{f'{lo}-{hi}':<12}{int(m.sum()):>8}{auc(ys[m], ps[m]):>8.4f}"
              f"{ps[m].mean():>10.3f}{ys[m].mean():>8.3f}")
    print("\n**単調増加**していれば健全（終盤ほど勝敗が読めるはず）")


if __name__ == "__main__":
    main()
