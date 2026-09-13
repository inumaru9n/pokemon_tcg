"""対称な privileged value network（I-109 Stage 3）。

    V(s) = head( LSTM( enc( [ E(p0), E(p1), prog, logs, ctx, prev ] ) ) )
    E(p) = shared( [ ゾーンごとのカード埋め込みの和 (4×d_card), 汎用属性 (14) ] )

**両プレイヤーに同じ E を使う**（対称）。V(s) は状態の勝率であって視点を持つ理由がない。
カードは**IDをそのまま埋め込み索引に使う**ので語彙も和集合も不要 —
リーグにデッキを足しても次元は変わらず、mainを差し替えてもCriticは作り直し不要。

旧 ValueNet（train_value.py）との違い:
  旧: [board(自分), logs, ctx, prev, oppb=ft_main.feat_vector(state,1-me)]
      → 相手側は「我々のデッキのID体系を相手に当てた」もので、非ミラーでは
        158次元中88が常にゼロ。相手の手札は観測に無いので**特権情報は元々ゼロ**だった。
  新: 両者を同じ関数で符号化し、隠れゾーンは GetHiddenData で実際に埋める。

**方策と同じ入力レイアウトにする設計（暖機用）は捨てた**。EXP-095で暖機の効果は
ゼロと実測済みで、その制約を維持する理由がもう無い。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from selfplay.state_spec import DEFAULT_N_CARDS, N_ATTR, N_PROG, N_ZONE


class ValueNetV2(nn.Module):
    def __init__(self, n_cards: int = DEFAULT_N_CARDS, d_card: int = 32,
                 n_logs: int = 48, n_ctx: int = 49, d_player: int = 128,
                 d_model: int = 256, n_layers: int = 2):
        super().__init__()
        self.card_emb = nn.Embedding(n_cards, d_card, padding_idx=0)
        # **両プレイヤー共有**のプレイヤー符号化器
        self.player_enc = nn.Sequential(
            nn.Linear(N_ZONE * d_card + N_ATTR, d_player), nn.ReLU(),
            nn.Linear(d_player, d_player), nn.ReLU())
        self.ctx_emb = nn.Embedding(n_ctx, 16)
        self.prev_card_emb = nn.Embedding(n_cards, d_card, padding_idx=0)
        d_in = 2 * d_player + N_PROG + n_logs + 16 + d_card + 2
        self.enc = nn.Sequential(
            nn.Linear(d_in, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model), nn.ReLU())
        self.lstm = nn.LSTM(d_model, d_model, num_layers=n_layers, batch_first=True)
        self.head = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.ReLU(),
                                  nn.Linear(d_model // 2, 1))

    def encode_players(self, zone, attr):
        """zone (B,T,2,Z,C) int64 / attr (B,T,2,A) → (B,T,2,d_player)"""
        e = self.card_emb(zone)              # (B,T,2,Z,C,d_card)
        bag = e.sum(dim=-2)                  # ゾーンごとに和（bag-of-embeddings）
        B, T, P, Z, D = bag.shape
        x = torch.cat([bag.reshape(B, T, P, Z * D), attr], dim=-1)
        return self.player_enc(x)

    def forward(self, zone, attr, prog, logs, ctx, prev_cidx, prev_num, state=None):
        pe = self.encode_players(zone, attr)                 # (B,T,2,d_player)
        B, T = pe.shape[0], pe.shape[1]
        x = torch.cat([pe.reshape(B, T, -1), prog, logs, self.ctx_emb(ctx),
                       self.prev_card_emb(prev_cidx), prev_num], dim=-1)
        h, state = self.lstm(self.enc(x), state)
        return self.head(h).squeeze(-1), state               # (B,T) ロジット


if __name__ == "__main__":
    torch.manual_seed(0)
    net = ValueNetV2()
    B, T, Z, C = 2, 5, N_ZONE, 64
    out, st = net(torch.randint(0, 100, (B, T, 2, Z, C)),
                  torch.randn(B, T, 2, N_ATTR), torch.randn(B, T, N_PROG),
                  torch.randn(B, T, 48), torch.randint(0, 49, (B, T)),
                  torch.randint(0, 100, (B, T)), torch.randn(B, T, 2))
    n_par = sum(p.numel() for p in net.parameters())
    print(f"出力 {tuple(out.shape)} / パラメータ {n_par:,}")
    # **対称性の検査**: プレイヤーを入れ替えたら勝率ロジットが反転する…わけではない
    # （progに手番が入るため）。ここでは「同じ入力なら同じ出力」の決定性だけ見る。
    out2, _ = net(torch.randint(0, 100, (B, T, 2, Z, C)),
                  torch.randn(B, T, 2, N_ATTR), torch.randn(B, T, N_PROG),
                  torch.randn(B, T, 48), torch.randint(0, 49, (B, T)),
                  torch.randint(0, 100, (B, T)), torch.randn(B, T, 2))
    print(f"forward OK（別入力で別出力: {not torch.allclose(out, out2)}）")


class ValueTraj:
    """Critic学習に必要な分だけを持つ軽量版（Traj からの抽出）。

    **なぜ要るか**: Criticのバッファに Traj をそのまま溜めると、Criticが一切使わない
    選択肢の特徴（opt_vec/opt_cid/opt_cls）が蓄積してメモリを食う（Stage 0/2で
    同種の破綻を経験済み）。zoneはint16のまま持つ（1決定1,024B、5,000試合で0.27GB）。
    """

    __slots__ = ("zone", "attr", "prog", "logs", "prev", "ctx", "reward")

    def __init__(self, t):
        self.zone = np.asarray(t.zone, np.int16)
        self.attr = np.asarray(t.attr, np.float16)
        self.prog = np.asarray(t.prog, np.float16)
        self.logs = np.asarray(t.logs, np.float16)
        self.prev = np.asarray(t.prev, np.float16)
        self.ctx = np.asarray(t.ctx, np.int16)
        self.reward = float(t.reward)


def to_value_trajs(trajs):
    return [ValueTraj(t) for t in trajs if len(t.zone)]


def _pack(sub, device):
    """ValueTraj / Traj のリストを (B,T,...) テンソルへ。"""
    B = len(sub)
    T = max(len(r.zone) for r in sub)
    z0 = np.asarray(sub[0].zone)          # (T, 2, N_ZONE, ZONE_CAP)
    # **末尾2軸から取る**。shape[1],shape[2] だと (2, N_ZONE) を掴んでしまう
    # （プレイヤー軸をゾーン軸と取り違える。実際にKaggleで踏んだ）
    Z, C = z0.shape[-2], z0.shape[-1]
    n_logs = np.asarray(sub[0].logs).shape[1]
    zone = np.zeros((B, T, 2, Z, C), np.int64)
    attr = np.zeros((B, T, 2, N_ATTR), np.float32)
    prog = np.zeros((B, T, N_PROG), np.float32)
    logs = np.zeros((B, T, n_logs), np.float32)
    ctx = np.zeros((B, T), np.int64)
    pc = np.zeros((B, T), np.int64)
    pn = np.zeros((B, T, 2), np.float32)
    win = np.zeros((B, T), np.float32)
    valid = np.zeros((B, T), bool)
    for b, r in enumerate(sub):
        n = len(r.zone)
        zone[b, :n] = np.asarray(r.zone, np.int64)
        attr[b, :n] = np.asarray(r.attr, np.float32)
        prog[b, :n] = np.asarray(r.prog, np.float32)
        logs[b, :n] = np.asarray(r.logs, np.float32)
        ctx[b, :n] = np.asarray(r.ctx, np.int64)
        pv = np.asarray(r.prev, np.float32)
        # prev のカードIDも **IDそのまま+1** で埋め込む（語彙不要）。
        # 負の値（初期値-1 / NULL_CID=-3）は 0（padding）に落とす。
        pc[b, :n] = np.clip(pv[:, 0].astype(np.int64) + 1, 0, None)
        pn[b, :n] = pv[:, 1:3]
        win[b, :n] = 1.0 if r.reward > 0 else 0.0
        valid[b, :n] = True
    t = lambda a: torch.as_tensor(a, device=device)  # noqa: E731
    return dict(zone=t(zone), attr=t(attr), prog=t(prog), logs=t(logs),
                ctx=t(ctx), prev_cidx=t(pc), prev_num=t(pn),
                win=t(win), valid=t(valid))


@torch.no_grad()
def group_advantage(trajs):
    """**Criticを使わない advantage**（CRN群相対、GRPO/RLOO と同型）。

    同じ `gkey` を持つ Traj は **配札・相手・先後が完全に同一**（CRNで複製した群）。
    そこで A_i = R_i − mean(R_群) とすれば、カードゲーム最大のノイズ源である
    **配札の分散が全ターンで厳密に消える**。criticの近似（序盤AUC 0.599）より強い。

    失うのは per-step の credit assignment（A は試合内で一定になる）。ただし
    EXP-104 で「教師の良手ですら ΔV>0 は52.2%＝コイン投げ」と実測しており、
    GAEが返す per-step の差は我々のVではノイズの可能性が高い。

    **leave-one-out にはしない**: 群平均に自分を含めると A の和がちょうど0になり
    （群内で相殺）、群サイズが小さいときの分散が素直に小さくなる。
    戻り: trajs と同じ順の list[np.ndarray]（各要素は決定数ぶんの定数列）。
    """
    import collections
    by = collections.defaultdict(list)
    for i, t in enumerate(trajs):
        by[getattr(t, "gkey", -1)].append(i)
    out = [None] * len(trajs)
    for _, idx in by.items():
        rs = [float(trajs[i].reward) for i in idx]
        mu = sum(rs) / len(rs)
        for i in idx:
            n = len(trajs[i].grad) if trajs[i].grad else len(trajs[i].board)
            out[i] = np.full(n, float(trajs[i].reward) - mu, np.float32)
    return out


def gae_v2(vnet, trajs, device, lam=0.95, mb=24):
    """GAE(γ=1)。報酬は終端のみ（±1）。**Criticを更新する前に呼ぶこと**
    （更新後だと A = R - V が縮んで勾配が痩せる。Stage 2で実測した罠）。"""
    out = [None] * len(trajs)
    for s0 in range(0, len(trajs), mb):
        sub = trajs[s0:s0 + mb]
        b = _pack(sub, device)
        logits, _ = vnet(b["zone"], b["attr"], b["prog"], b["logs"], b["ctx"],
                         b["prev_cidx"], b["prev_num"])
        # BCEWithLogitsLoss で学習しているので sigmoid(logits)=P(勝利)。
        # 報酬と同じ [-1,1] への正しい変換は 2σ(x)-1（tanh(x) は係数2ぶん過大）。
        v = (2.0 * torch.sigmoid(logits) - 1.0).float().cpu().numpy()
        for i, r in enumerate(sub):
            n = len(r.zone)
            vv = v[i, :n]
            adv = np.zeros(n, np.float32)
            g = 0.0
            for t in range(n - 1, -1, -1):
                nxt = r.reward if t == n - 1 else vv[t + 1]
                g = (nxt - vv[t]) + lam * g
                adv[t] = g
            out[s0 + i] = adv
    return out


def value_update_v2(vnet, vopt, vtrajs, device, *, epochs=1, mb_seqs=24,
                    max_grad_norm=1.0):
    """Criticの更新。目標は**実際の勝敗**なのでオフポリシーでも正しい
    （＝直近数イテレーションのバッファから学習してよい）。"""
    bce = nn.BCEWithLogitsLoss(reduction="none")
    tot, n_mb = 0.0, 0
    for _ in range(max(1, epochs)):
        order = np.random.permutation(len(vtrajs))
        for s0 in range(0, len(order), mb_seqs):
            sub = [vtrajs[i] for i in order[s0:s0 + mb_seqs]]
            b = _pack(sub, device)
            logits, _ = vnet(b["zone"], b["attr"], b["prog"], b["logs"],
                             b["ctx"], b["prev_cidx"], b["prev_num"])
            m = b["valid"].float()
            loss = (bce(logits, b["win"]) * m).sum() / m.sum().clamp(min=1)
            vopt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(vnet.parameters(), max_grad_norm)
            vopt.step()
            tot += float(loss.detach())
            n_mb += 1
    return tot / max(1, n_mb)
