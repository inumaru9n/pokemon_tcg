"""Stage 2b: KL係数を振って「方策が動くが壊れない」帯を探す。

**強さは測らない**。見るのは3点だけ:
  1. **動くか**  — 094からの行動一致率が下がるか（下がらなければPPOは何もしていない）
  2. **壊れないか** — 自己対戦の勝率が崩壊しないか、エントロピーが潰れ/発散しないか
  3. **KLが暴れないか** — SL方策からの乖離が制御下にあるか

**この検査が要る理由**（EXP-095）: 094は非常に尖った方策（選択肢平均8個でT=1.0でも
エントロピー0.406）。**KLアンカーと低エントロピーが両方「変わるな」方向に働く**ので、
KL係数を強くしすぎるとPPOがほとんど動かない。逆に弱すぎるとSL方策から離れて壊れる。

usage:
  uv run python selfplay/ppo_probe.py --kl 0.0 0.01 0.05 0.2 --iters 4 --games 400
"""

from __future__ import annotations

import argparse
import copy
import os
import sys

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "arena"),
           os.path.join(_ROOT, "agents/094_yushin_nn")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

MODEL = os.path.join(_ROOT, "agents/094_yushin_nn/model_094.npz")


class _TorchAsBatchPolicy:
    """PolicyTorch を rollout.collect が期待する scores() 形式で使う薄いラッパ。"""

    def __init__(self, net, vocab, n_slots, device):
        self.net, self.dev = net, device
        lo = min(vocab)
        self.lo = lo
        self.lut = np.ones(max(vocab) - lo + 1, np.int64)
        for k, v in vocab.items():
            self.lut[k - lo] = v
        d_model = net.lstm.hidden_size
        self.n_layers = net.lstm.num_layers
        self.n_slots = n_slots
        self.d_model = d_model
        self.reset_all()

    def reset_all(self):
        z = torch.zeros(self.n_layers, self.n_slots, self.d_model, device=self.dev)
        self.h, self.c = z.clone(), z.clone()

    def _cidx(self, a):
        return self.lut[np.clip(np.asarray(a, np.int64) - self.lo, 0, len(self.lut) - 1)]

    @torch.no_grad()
    def scores(self, slots, boards, logs, prevs, ctxs, opt_cids, opt_vecs, opt_clss):
        B = len(slots)
        O = max(len(c) for c in opt_cids)
        D = len(opt_vecs[0][0])
        ocid = np.zeros((B, O), np.int64)
        ovec = np.zeros((B, O, D), np.float32)
        ocls = np.full((B, O), -1, np.int64)
        mask = np.zeros((B, O), bool)
        for i in range(B):
            k = len(opt_cids[i])
            ocid[i, :k] = self._cidx(opt_cids[i])
            ovec[i, :k] = opt_vecs[i]
            if opt_clss[i] is not None:
                ocls[i, :k] = opt_clss[i]
            mask[i, :k] = True
        prevs = np.asarray(prevs, np.float32)
        t = lambda a: torch.as_tensor(a, device=self.dev)  # noqa: E731
        sl = torch.as_tensor(np.asarray(slots, np.int64), device=self.dev)
        st = (self.h[:, sl].contiguous(), self.c[:, sl].contiguous())
        sc, (h2, c2) = self.net(
            t(np.asarray(boards, np.float32))[:, None], t(np.asarray(logs, np.float32))[:, None],
            t(self._cidx(prevs[:, 0]))[:, None], t(prevs[:, None, 1:3]),
            t(np.asarray(ctxs, np.int64))[:, None], t(ocid)[:, None], t(ovec)[:, None],
            t(mask)[:, None], t(ocls)[:, None], st)
        self.h[:, sl] = h2
        self.c[:, sl] = c2
        sc = sc[:, 0].float().cpu().numpy()
        return [sc[i, :len(opt_cids[i])] for i in range(B)]


def agreement(net_a, net_b, vocab, deck, n_games, batch, device, seed, temp):
    """**argmax(net_a) vs argmax(net_b)** を同一状態列の上で比べる。

    誤りやすい点（実際に一度踏んだ）: 「net_aのサンプリング結果 vs net_bのargmax」を
    比べると、net_a==net_b でも T=0.7 のargmax一致率（約89%）が出てしまい、
    方策が動いたのかサンプリングのゆらぎなのか区別できない。**両方argmaxで比べる**。

    またLSTM状態は系列に沿って進める必要があるので、1決定ずつ呼ぶのでなく
    pack() した (B,T,...) を両ネットに1回ずつ通す。
    """
    from selfplay.ppo import pack
    from selfplay.rollout import collect
    pa = _TorchAsBatchPolicy(net_a, vocab, 2 * batch, device)
    rolls, _ = collect(pa, (deck, deck), n_games, seed, batch=batch, temperature=temp)
    # **ミニバッチで回す**（全系列を1回forwardするとCUDA OOM。実測で踏んだ）
    same = tot = 0
    for s0 in range(0, len(rolls), 32):
        b = pack(rolls[s0:s0 + 32], vocab, device)
        with torch.no_grad():
            sa, _ = net_a(b["board"], b["logs"], b["prev_cidx"], b["prev_num"],
                          b["ctx"], b["opt_cidx"], b["opt_vec"], b["mask"], b["opt_cls"])
            sb, _ = net_b(b["board"], b["logs"], b["prev_cidx"], b["prev_num"],
                          b["ctx"], b["opt_cidx"], b["opt_vec"], b["mask"], b["opt_cls"])
        m = (b["grad"] & b["valid"])
        same += int(((sa.argmax(-1) == sb.argmax(-1)) & m).sum())
        tot += int(m.sum())
    return same / max(1, tot), tot


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kl", type=float, nargs="+", default=[0.0, 0.01, 0.05, 0.2])
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, nargs="+", default=[1e-4],
                    help="複数指定可（KL係数との2軸スイープ）")
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--mb", type=int, default=24, help="更新のミニバッチ系列数")
    ap.add_argument("--vlr", type=float, default=3e-4, help="Criticの学習率")
    ap.add_argument("--lam", type=float, default=0.95, help="GAEのλ")
    ap.add_argument("--vbuf", type=int, default=6,
                    help="Criticが使う直近イテレーション数（Vは2万試合規模で飽和）")
    ap.add_argument("--vinit", default="", help="事前学習Vのnpz（省略でスクラッチ）")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    from selfplay.ppo import (PolicyTorch, ValueTorch, ppo_update,
                              value_update, gae, to_value_rollouts)
    from selfplay.rollout import collect
    import run_match as rm

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev} / KL係数 {args.kl} / {args.iters}イテレーション×{args.games}試合",
          flush=True)
    deck = rm.read_deck(os.path.join(_ROOT, "agents/094_yushin_nn"))

    rows = []
    for lr in args.lr:
      for kl_coef in args.kl:
          torch.manual_seed(args.seed)
          np.random.seed(args.seed)
          net, vocab = PolicyTorch.from_npz(MODEL, device=dev)
          ref, _ = PolicyTorch.from_npz(MODEL, device=dev)
          for p in ref.parameters():
              p.requires_grad_(False)
          ref.eval()
          opt = torch.optim.Adam(net.parameters(), lr=lr)
          # Critic（完全情報V）。Actorとは別ネットワーク・別optimizer
          z = np.load(MODEL, allow_pickle=False)
          vnet = ValueTorch(int(z["n_board"]), int(z["n_logs"]),
                            z["card_emb.weight"].shape[0]).to(dev)
          if args.vinit and os.path.exists(args.vinit):
              zz = np.load(args.vinit, allow_pickle=False)
              sd = vnet.state_dict()
              for k in sd:
                  if k in zz.files and tuple(zz[k].shape) == tuple(sd[k].shape):
                      sd[k].copy_(torch.as_tensor(np.asarray(zz[k], np.float32)))
              vnet.load_state_dict(sd)
              print(f"  Critic: {os.path.basename(args.vinit)} から初期化", flush=True)
          vopt = torch.optim.Adam(vnet.parameters(), lr=args.vlr)
          vbuf = []
          print(f"\n=== lr={lr} / KL係数 {kl_coef} ===", flush=True)
          last = {}
          for it in range(args.iters):
              pol = _TorchAsBatchPolicy(net, vocab, 2 * args.batch, dev)
              rolls, st = collect(pol, (deck, deck), args.games,
                                  args.seed * 100000 + it * 1000,
                                  batch=args.batch, temperature=args.temp)
              # 1) **更新前のV**でadvantageを作る（更新後だと A=R-V が縮む）
              advs = gae(vnet, rolls, vocab, dev, lam=args.lam, mb=args.mb)
              # 2) Criticを更新（直近 vbuf イテレーションのバッファ）
              vbuf.append(to_value_rollouts(rolls))  # 軽量化して溜める
              vbuf = vbuf[-args.vbuf:]
              flat = [r for rs in vbuf for r in rs]
              vloss = value_update(vnet, vopt, flat, vocab, dev, mb_seqs=args.mb)
              a_all = np.concatenate([a for a in advs if len(a)])
              # 3) Actor を更新
              s = ppo_update(net, ref, opt, rolls, vocab, dev, kl_coef=kl_coef,
                             temperature=args.temp, epochs=2, mb_seqs=args.mb,
                             advs=advs)
              last = s
              print(f"  it{it}: pg={s['pg']:+.4f} KL={s['kl']:.4f} ent={s['ent']:.4f} "
                    f"|g|={s['gnorm']:.2f} | Vloss={vloss:.4f} "
                    f"adv(sd)={a_all.std():.3f} buf={len(flat):,}系列 "
                    f"生成{st['sec']:.0f}秒", flush=True)
          ag, n_ag = agreement(net, ref, vocab, deck, 60, 32, dev, args.seed + 777, args.temp)
          rows.append((lr, kl_coef, last.get("kl", float("nan")),
                       last.get("ent", float("nan")), ag))
          print(f"  → 094とのargmax一致率 {ag:.1%} (n={n_ag:,})", flush=True)

    print("\n=== まとめ（動くか / 壊れないか）===")
    print(f"{'lr':>9}{'KL係数':>8}{'最終KL':>10}{'エントロピー':>14}{'094一致率':>12}")
    for l, k, kl, e, a in rows:
        print(f"{l:>9.0e}{k:>8}{kl:>10.4f}{e:>14.4f}{a:>12.1%}")
    print("\n読み方: 一致率が100%のままならPPOは何もしていない。"
          "80%を大きく割り込むなら離れすぎ。エントロピーが0付近に潰れるのも危険。")


if __name__ == "__main__":
    main()
