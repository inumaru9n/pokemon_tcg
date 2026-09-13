"""PPO + KLアンカー（I-109 Phase RL Stage 2）。

**設計**（EXP-095の実測に基づく）:
- 初期値 = 094（模倣で得たSL方策）。**SL方策へのKL正則化を最後まで掛ける**（AlphaStar同様）
- 探索温度 **T=0.7**（強さのコスト−3.7ptでT=0.5と同等、探索量は1.6倍。EXP-095 Stage 2a）
- **単一選択(minCount=maxCount=1)の決定のみ勾配対象**（全決定の84.5%）。複数選択は上位k個を
  決定的に取る仕様で joint action の確率分布になっていないため、無理に確率化せず除外する
- baseline = 完全情報のV。**スクラッチ初期化でよい**（094からの暖機は効果ゼロと実証済み）
- advantage = GAE(γ=1, λ)。報酬は終端の勝敗±1のみ
- 相手 = **現状は純ミラー自己対戦のみ（過去自己スナップショットは未実装）**。
  計画では「自己60%/過去自己40%」だが、Stage 2c（動くかの診断）では純ミラーで足りる。
  **Stage 2d（本番ラン）の前に必ず実装すること**——BACKLOG I-123 が警告する
  「純ミラー自己対戦はミラー強度だけを最適化する」問題への対処であり、
  凍結094（＝最古の過去自己）に対する勝率が循環で伸びない恐れがある。
  **プールv4は訓練に混ぜない**（忠実度監査により）

**最初に潰すべき懸念**: 094は非常に尖った方策（選択肢平均8個でT=1.0でもエントロピー0.406）。
**KLアンカーと低エントロピーが両方「変わるな」に働く**ので、KL係数を強くしすぎると
PPOがほとんど動かない。2bではKL係数を振って「方策が動くが壊れない」帯を探す。
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "arena"),
           os.path.join(_ROOT, "agents/094_yushin_nn")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# **MAIN階層の合成はSL学習側と同一実装を使う**（別々に書くと片方だけ直してズレる）
from selfplay.train_nn_kaggle import combine_scores  # noqa: E402


class PolicyTorch(nn.Module):
    """094と同一構造の方策（学習可能）。npzから重みを読み込める。"""

    def __init__(self, n_board, n_logs, n_optdim, n_cards, n_ctx, n_cls,
                 d_card=32, d_opt=64, d_model=256, n_layers=2, dropout=0.0,
                 head="additive"):
        super().__init__()
        self.card_emb = nn.Embedding(n_cards, d_card, padding_idx=0)
        self.prev_card_emb = nn.Embedding(n_cards, d_card, padding_idx=0)
        self.ctx_emb = nn.Embedding(n_ctx, 16)
        # **SL側(train_nn_kaggle.PolicyNet)と同じ Sequential 構成にする**。
        # SLがDropout入りで学習されていると重みは enc.0/enc.3 として保存されており、
        # ここが enc.0/enc.2 だと enc.3 が読み捨てられ enc.2 が乱数のまま残る
        # （load_state_dict を通さない copy_ ループなので例外も出ない）。
        self.enc = nn.Sequential(
            nn.Linear(n_board + n_logs + 16 + d_card + 2, d_model), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Dropout(dropout))
        self.lstm = nn.LSTM(d_model, d_model, num_layers=n_layers, batch_first=True,
                            dropout=dropout if n_layers > 1 else 0.0)
        self.opt_enc = nn.Sequential(
            nn.Linear(d_card + n_optdim, d_opt), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_opt, d_opt), nn.ReLU(), nn.Dropout(dropout))
        self.proj = nn.Linear(d_model, d_opt)
        self.ctx_scale = nn.Embedding(n_ctx, d_opt)
        self.ctx_bias = nn.Embedding(n_ctx, 1)
        self.cls_head = nn.Linear(d_model, n_cls) if n_cls else None
        self.n_cls = n_cls
        self.head = head
        # **特徴の標準化はネット内で行う**。呼び出し側（ロールアウト/学習/評価）が
        # それぞれ掛ける設計にすると、どれか1つで掛け忘れても例外が出ずに壊れる。
        self.register_buffer("board_mu", torch.zeros(n_board or 1))
        self.register_buffer("board_sd", torch.ones(n_board or 1))
        self.register_buffer("opt_mu", torch.zeros(n_optdim))
        self.register_buffer("opt_sd", torch.ones(n_optdim))

    def forward(self, board, logs, prev_cidx, prev_num, ctx,
                opt_cidx, opt_vec, mask, opt_cls=None, state=None):
        """(B,T,...) → scores (B,T,O)。マスク外は -1e9。"""
        board = (board - self.board_mu) / self.board_sd
        opt_vec = (opt_vec - self.opt_mu) / self.opt_sd
        x = torch.cat([board, logs, self.ctx_emb(ctx),
                       self.prev_card_emb(prev_cidx), prev_num], dim=-1)
        h, state = self.lstm(self.enc(x), state)
        e_card = self.card_emb(opt_cidx)
        e_opt = self.opt_enc(torch.cat([e_card, opt_vec], dim=-1))
        q = self.proj(h) * self.ctx_scale(ctx)
        sc = (e_opt * q.unsqueeze(2)).sum(-1) + self.ctx_bias(ctx)
        if self.cls_head is not None and opt_cls is not None:
            # **SL側と同一の合成式を使う**（別々に書くと片方だけ直して無言でズレる）
            sc = combine_scores(sc, self.cls_head(h), opt_cls, mask,
                                self.n_cls, self.head)
        return sc.masked_fill(~mask, -1e9), state

    @staticmethod
    def from_npz(path, device="cpu"):
        z = np.load(path, allow_pickle=False)
        n_cards, d_card = z["card_emb.weight"].shape
        n_ctx = z["ctx_emb.weight"].shape[0]
        d_model = z["lstm.weight_hh_l0"].shape[1]
        n_layers = sum(1 for k in z.files if k.startswith("lstm.weight_ih_l"))
        d_opt = z["opt_enc.0.weight"].shape[0]
        n_optdim = z["opt_enc.0.weight"].shape[1] - d_card
        n_board = int(z["n_board"]) if "n_board" in z.files else None
        n_logs = int(z["n_logs"]) if "n_logs" in z.files else None
        n_cls = z["cls_head.weight"].shape[0] if "cls_head.weight" in z.files else 0
        head = str(z["head"]) if "head" in z.files else "additive"
        # Dropout入りで学習された重みは enc.0/enc.3 に入る。受け側の Sequential を
        # 揃えないと enc.3 が読み捨てられ enc.2 が乱数のまま残る（例外なし）
        drop = 0.0 if "enc.2.weight" in z.files else 1e-9
        net = PolicyTorch(n_board, n_logs, n_optdim, n_cards, n_ctx, n_cls,
                          d_card=d_card, d_opt=d_opt, d_model=d_model,
                          n_layers=n_layers, dropout=drop, head=head)
        sd = net.state_dict()
        # 標準化統計は既定（0/1）のままだと素通し。npzにあれば必ず載せる
        norm_keys = {"board_mu", "board_sd", "opt_mu", "opt_sd"}
        missing = []
        for k in sd:
            if k in z.files:
                sd[k].copy_(torch.as_tensor(np.asarray(z[k], np.float32)))
            elif k not in norm_keys:
                missing.append(k)
        if missing:
            # **黙って乱数初期値を使わない**。構成不一致は必ず落とす
            raise KeyError(f"npzに無い重み: {missing[:8]}"
                           f"{'...' if len(missing) > 8 else ''} ({path})")
        net.load_state_dict(sd)
        vocab = {int(a): int(b) for a, b in zip(z["vocab_keys"], z["vocab_vals"])}
        return net.to(device), vocab


def pack(rollouts, vocab, device="cpu", max_opt=None):
    """Rollout のリスト → (B,T,...) のテンソル群。1系列=1試合1プレイヤー。"""
    B = len(rollouts)
    T = max(len(r.chosen) for r in rollouts)
    O = max_opt or max(len(c) for r in rollouts for c in r.opt_cid)
    F_ = len(rollouts[0].board[0])
    L_ = len(rollouts[0].logs[0])
    D_ = len(rollouts[0].opt_vec[0][0])
    lo = min(vocab)
    lut = np.ones(max(vocab) - lo + 1, np.int64)
    for k, v in vocab.items():
        lut[k - lo] = v

    def cidx(a):
        return lut[np.clip(np.asarray(a, np.int64) - lo, 0, len(lut) - 1)]

    board = np.zeros((B, T, F_), np.float32)
    logs = np.zeros((B, T, L_), np.float32)
    pc = np.zeros((B, T), np.int64)
    pn = np.zeros((B, T, 2), np.float32)
    ctx = np.zeros((B, T), np.int64)
    ocid = np.zeros((B, T, O), np.int64)
    ovec = np.zeros((B, T, O, D_), np.float32)
    ocls = np.full((B, T, O), -1, np.int64)
    mask = np.zeros((B, T, O), bool)
    act = np.zeros((B, T), np.int64)
    oldlp = np.zeros((B, T), np.float32)
    grad = np.zeros((B, T), bool)
    valid = np.zeros((B, T), bool)
    ret = np.zeros((B, T), np.float32)
    for b, r in enumerate(rollouts):
        n = len(r.chosen)
        board[b, :n] = r.board
        logs[b, :n] = r.logs
        p = np.asarray(r.prev, np.float32)
        pc[b, :n] = cidx(p[:, 0])
        pn[b, :n] = p[:, 1:3]
        ctx[b, :n] = r.ctx
        for t in range(n):
            k = len(r.opt_cid[t])
            ocid[b, t, :k] = cidx(r.opt_cid[t])
            ovec[b, t, :k] = r.opt_vec[t]
            if r.opt_cls[t] is not None:
                ocls[b, t, :k] = r.opt_cls[t]
            mask[b, t, :k] = True
        act[b, :n] = r.chosen
        oldlp[b, :n] = r.logp
        grad[b, :n] = r.grad
        valid[b, :n] = True
        ret[b, :n] = r.reward          # γ=1 なので全ステップのリターン=最終報酬
    t = lambda a: torch.as_tensor(a, device=device)  # noqa: E731
    return dict(board=t(board), logs=t(logs), prev_cidx=t(pc), prev_num=t(pn),
                ctx=t(ctx), opt_cidx=t(ocid), opt_vec=t(ovec), opt_cls=t(ocls),
                mask=t(mask), act=t(act), old_logp=t(oldlp), grad=t(grad),
                valid=t(valid), ret=t(ret))


def ppo_update(net, ref, opt, rollouts, vocab, device, *, clip=0.2, kl_coef=0.05,
               ent_coef=0.0, temperature=0.7, epochs=2, max_grad_norm=1.0,
               mb_seqs=32, advs=None, kl_break=0.5):
    """PPO更新。**系列のミニバッチ単位で pack して逐次更新する**。

    バッチ全体を1回のforwardに載せるとCUDA OOMになる（実測: 600試合=1,200系列で
    opt_encの中間活性だけで1GB超、LSTMのbackward込みで14GBを使い切った）。
    系列ごとに長さが違うので**パディングもミニバッチ単位**にする（全体最大Tに揃えない）。

    advs: 各 rollout の advantage 配列（GAEの出力）。**None ならリターンの平均を引くだけ**
    だが、それでは自己対戦の advantage が実質±1のコインフリップになり信号が消える
    （2b初回で実証: KL係数0でも8イテレーションで一致率93.2%までしか動かなかった）。
    **必ず V を baseline として渡すこと**。

    kl_break: **安全装置**。1更新あたりの KL(π_old‖π_new) がこの値を超えたら、その
    イテレーションの残りミニバッチを打ち切る（制御ではなく回路遮断器）。
    **閾値を絞りすぎると学習を黙って抑制し「効かない」と誤読する**ので、通常運転では
    絶対に発火しない緩い値にし、実測値を必ずログに出す（stats["kl_step"]）。
    なお SL方策への KL(π‖π_SL) とは別物（あちらは累積のアンカー、こちらは1更新の移動量）。
    """
    import numpy as _np
    # **正規化はイテレーション全体で1回**。ミニバッチ(約2,000決定)内で標準化すると
    # 各決定のadvantageが縮み、方策勾配がゼロに張り付く（2b二回目で実証: pg≈-0.0002）。
    # **正規化の対象は勾配対象(単一選択)のステップのみ**。複数選択の決定はゲーム中の
    # 別局面で起きるためadvantageの分布が系統的に違い、混ぜると平均・標準偏差が歪む。
    _mu, _sd = 0.0, 1.0
    if advs is not None:
        _sel = [a[np.asarray(r.grad, bool)] for a, r in zip(advs, rollouts) if len(a)]
        _cat = _np.concatenate([x for x in _sel if len(x)]) if _sel else _np.zeros(0)
        if len(_cat) > 1:
            _mu = float(_cat.mean())
            # **下限を置く**。advantageの分散がほぼ0のとき（全系列が同じ勝敗等）
            # 1e-6で割ると値が爆発して勾配が壊れる
            _sd = max(float(_cat.std()), 1e-3)
    agg = {}
    n_mb = 0
    for _ in range(max(1, epochs)):
        order = _np.random.permutation(len(rollouts))
        for s0 in range(0, len(order), mb_seqs):
            idx = order[s0:s0 + mb_seqs]
            sub = [rollouts[i] for i in idx]
            b = pack(sub, vocab, device)
            if advs is not None:
                T = b["ret"].shape[1]
                a_np = np.zeros((len(sub), T), np.float32)
                for j, i in enumerate(idx):
                    a = advs[i]
                    a_np[j, :len(a)] = (a - _mu) / _sd    # **イテレーション全体で正規化**
                adv = torch.as_tensor(a_np, device=device)
            else:
                adv = b["ret"] - b["ret"].mean()
            sc, _ = net(b["board"], b["logs"], b["prev_cidx"], b["prev_num"],
                        b["ctx"], b["opt_cidx"], b["opt_vec"], b["mask"], b["opt_cls"])
            logp_all = torch.log_softmax(sc / temperature, dim=-1)
            lp = torch.gather(logp_all, 2, b["act"].unsqueeze(-1)).squeeze(-1)
            m = (b["grad"] & b["valid"]).float()
            if float(m.sum()) < 1:
                continue
            # KL(π_old‖π_new) ≈ E_old[log π_old − log π_new]（1更新の移動量）
            kl_step = float((((b["old_logp"] - lp) * m).sum()
                             / m.sum()).detach())
            ratio = torch.exp((lp - b["old_logp"]) * m)
            s1 = ratio * adv
            s2 = torch.clamp(ratio, 1 - clip, 1 + clip) * adv
            pg = -(torch.min(s1, s2) * m).sum() / m.sum()
            with torch.no_grad():
                rsc, _ = ref(b["board"], b["logs"], b["prev_cidx"], b["prev_num"],
                             b["ctx"], b["opt_cidx"], b["opt_vec"], b["mask"],
                             b["opt_cls"])
                rlogp = torch.log_softmax(rsc / temperature, dim=-1)
            p_new = logp_all.exp()
            kl = ((p_new * (logp_all - rlogp)).sum(-1) * m).sum() / m.sum()
            ent = (-(p_new * logp_all).sum(-1) * m).sum() / m.sum()
            loss = pg + kl_coef * kl - ent_coef * ent
            opt.zero_grad()
            loss.backward()
            gn = nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
            opt.step()
            cur = dict(kl_step=kl_step,
                       pg=float(pg.detach()), kl=float(kl.detach()), ent=float(ent.detach()),
                       ratio=float(((ratio * m).sum() / m.sum()).detach()),
                       gnorm=float(gn), n=int(m.sum()))
            for k, v in cur.items():
                agg[k] = agg.get(k, 0.0) + v
            n_mb += 1
            if abs(kl_step) > kl_break:
                print(f"    [安全装置] 1更新のKL {kl_step:.3f} > {kl_break} "
                      f"→ 残りミニバッチを打ち切り", flush=True)
                agg["broke"] = agg.get("broke", 0.0) + 1
                break
    if n_mb:
        for k in agg:
            agg[k] = agg[k] / n_mb if k != "n" else int(agg[k])
    return agg


# ============================================================================
# 価値関数（Critic）と GAE
# ============================================================================

class ValueTorch(nn.Module):
    """完全情報のV。**Actorとは別ネットワーク**（入力に相手盤面を含むため）。

    train_value.ValueNet と同一構造。入力の並びも方策と揃えてあるが、
    **暖機は効果ゼロと実証済み**（EXP-095）なのでスクラッチ初期化で使う。
    """

    def __init__(self, n_board, n_logs, n_cards, n_ctx=49, d_card=32,
                 d_model=256, n_layers=2):
        super().__init__()
        self.n_board = n_board
        self.ctx_emb = nn.Embedding(n_ctx, 16)
        self.prev_card_emb = nn.Embedding(n_cards, d_card, padding_idx=0)
        d_in = n_board * 2 + n_logs + 16 + d_card + 2
        self.enc = nn.Sequential(
            nn.Linear(d_in, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model), nn.ReLU())
        self.lstm = nn.LSTM(d_model, d_model, num_layers=n_layers, batch_first=True)
        self.head = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.ReLU(),
                                  nn.Linear(d_model // 2, 1))

    def forward(self, board, oppb, logs, ctx, prev_cidx, prev_num, state=None):
        x = torch.cat([board, logs, self.ctx_emb(ctx),
                       self.prev_card_emb(prev_cidx), prev_num, oppb], dim=-1)
        h, state = self.lstm(self.enc(x), state)
        return self.head(h).squeeze(-1), state       # (B,T) ロジット


def pack_value(rollouts, vocab, device="cpu"):
    """Critic用のテンソル（相手盤面を含む）。"""
    B = len(rollouts)
    T = max(len(r.chosen) for r in rollouts)
    F_ = len(rollouts[0].board[0])
    L_ = len(rollouts[0].logs[0])
    lo = min(vocab)
    lut = np.ones(max(vocab) - lo + 1, np.int64)
    for k, v in vocab.items():
        lut[k - lo] = v
    board = np.zeros((B, T, F_), np.float32)
    oppb = np.zeros((B, T, F_), np.float32)
    logs = np.zeros((B, T, L_), np.float32)
    pc = np.zeros((B, T), np.int64)
    pn = np.zeros((B, T, 2), np.float32)
    ctx = np.zeros((B, T), np.int64)
    valid = np.zeros((B, T), bool)
    win = np.zeros((B, T), np.float32)
    for b, r in enumerate(rollouts):
        n = len(r.chosen)
        board[b, :n] = r.board
        oppb[b, :n] = r.oppb
        logs[b, :n] = r.logs
        p = np.asarray(r.prev, np.float32)
        pc[b, :n] = lut[np.clip(p[:, 0].astype(np.int64) - lo, 0, len(lut) - 1)]
        pn[b, :n] = p[:, 1:3]
        ctx[b, :n] = r.ctx
        valid[b, :n] = True
        win[b, :n] = 1.0 if r.reward > 0 else 0.0   # 勝ち=1 の二値
    t = lambda a: torch.as_tensor(a, device=device)  # noqa: E731
    return dict(board=t(board), oppb=t(oppb), logs=t(logs), prev_cidx=t(pc),
                prev_num=t(pn), ctx=t(ctx), valid=t(valid), win=t(win))


@torch.no_grad()
def gae(vnet, rollouts, vocab, device, lam=0.95, mb=32):
    """GAE(γ=1) を計算して各 Rollout に adv を返す（リストのリスト）。

    報酬は終端のみ（±1）。γ=1 なので δ_t = V(s_{t+1}) - V(s_t)（終端は R - V(s_T)）。
    λ=1 ならモンテカルロ（R - V(s_t)）に一致する。
    """
    out = [None] * len(rollouts)
    for s0 in range(0, len(rollouts), mb):
        sub = rollouts[s0:s0 + mb]
        b = pack_value(sub, vocab, device)
        logits, _ = vnet(b["board"], b["oppb"], b["logs"], b["ctx"],
                         b["prev_cidx"], b["prev_num"])
        # **Criticは BCEWithLogitsLoss で学習している**ので sigmoid(logits)=P(勝利)。
        # 報酬と同じ [-1,1] に写す正しい変換は 2σ(x)-1 = tanh(x/2) であって tanh(x) ではない
        # （tanh(x)=2σ(2x)-1 なので係数2ぶん過大になり、非線形な自信過剰を生む。
        #  例: logits=0.5 で 0.462 vs 正しくは 0.245）。
        # この歪みはadvantageの正規化では相殺できない（状態ごとに不均一なため）。
        v = 2.0 * torch.sigmoid(logits) - 1.0
        v = v.float().cpu().numpy()
        for i, r in enumerate(sub):
            n = len(r.chosen)
            vv = v[i, :n]
            adv = np.zeros(n, np.float32)
            gae_t = 0.0
            for t in range(n - 1, -1, -1):
                nxt = r.reward if t == n - 1 else vv[t + 1]
                delta = nxt - vv[t]
                gae_t = delta + lam * gae_t
                adv[t] = gae_t
            out[s0 + i] = adv
    return out


def value_update(vnet, vopt, rollouts, vocab, device, *, epochs=1, mb_seqs=32,
                 max_grad_norm=1.0):
    """Criticの更新。目標は**実際の勝敗**なのでオフポリシーでも正しい

    （＝直近数イテレーションのバッファから学習してよい。Stage 1の曲線測定で
    Vは2万試合規模で飽和すると判明しており、1イテレーション数百〜数千試合では足りない）。
    """
    import numpy as _np
    bce = nn.BCEWithLogitsLoss(reduction="none")
    tot, n_mb = 0.0, 0
    for _ in range(max(1, epochs)):
        order = _np.random.permutation(len(rollouts))
        for s0 in range(0, len(order), mb_seqs):
            sub = [rollouts[i] for i in order[s0:s0 + mb_seqs]]
            b = pack_value(sub, vocab, device)
            logits, _ = vnet(b["board"], b["oppb"], b["logs"], b["ctx"],
                             b["prev_cidx"], b["prev_num"])
            m = b["valid"].float()
            loss = (bce(logits, b["win"]) * m).sum() / m.sum().clamp(min=1)
            vopt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(vnet.parameters(), max_grad_norm)
            vopt.step()
            tot += float(loss)
            n_mb += 1
    return tot / max(1, n_mb)


class ValueRollout:
    """Critic学習に必要な分だけを numpy(float16) で持つ軽量版。

    **なぜ要るか**: Criticのバッファに Rollout をそのまま溜めると、Criticが一切使わない
    選択肢の特徴（opt_vec/opt_cid/opt_cls）がPythonリストのまま蓄積し、実測で
    **1系列あたり1.27MB**＝4,800系列で約6.0GBになる（Kaggleのメモリは約13GB）。
    Stage 0で同種のメモリ破綻を1度起こしているので、同じ轍を踏まない。
    必要なのは board / oppb / logs / prev / ctx / 勝敗のみ。
    """

    __slots__ = ("board", "oppb", "logs", "prev", "ctx", "chosen", "reward")

    def __init__(self, r):
        n = len(r.chosen)
        self.board = np.asarray(r.board, np.float16)
        self.oppb = np.asarray(r.oppb, np.float16)
        self.logs = np.asarray(r.logs, np.float16)
        self.prev = np.asarray(r.prev, np.float32)
        self.ctx = np.asarray(r.ctx, np.int16)
        self.chosen = np.zeros(n, np.int8)      # pack_value は長さしか見ない
        self.reward = r.reward


def to_value_rollouts(rollouts):
    """Rollout列 → ValueRollout列（Criticバッファ用）。"""
    return [ValueRollout(r) for r in rollouts]
