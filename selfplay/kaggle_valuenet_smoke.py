"""ValueNetV2 の単体スモーク（Kaggle GPU）。ローカルはmacOS x86_64でtorch wheelが無い。

見るもの:
  1. forward の形状が通るか / パラメータ数
  2. **学習が回るか**（合成データで数十ステップ、lossが下がるか）
  3. **メモリ実測**（zoneテンソルが支配的なので、バッファ設計の判断材料にする）
  4. 勾配が全パラメータに流れるか（player_encが両プレイヤー共有で死んでいないか）
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

for p in ("/kaggle/input", "/kaggle/working", os.getcwd()):
    if p not in sys.path:
        sys.path.insert(0, p)

# --- state_spec / value_net を単一ファイルで完結させる（Kaggleに依存物を置かない） ---
N_ZONE, ZONE_CAP, N_ATTR, N_PROG = 4, 64, 14, 8
DEFAULT_N_CARDS = 2048


class ValueNetV2(nn.Module):
    def __init__(self, n_cards=DEFAULT_N_CARDS, d_card=32, n_logs=48, n_ctx=49,
                 d_player=128, d_model=256, n_layers=2):
        super().__init__()
        self.card_emb = nn.Embedding(n_cards, d_card, padding_idx=0)
        self.player_enc = nn.Sequential(
            nn.Linear(N_ZONE * d_card + N_ATTR, d_player), nn.ReLU(),
            nn.Linear(d_player, d_player), nn.ReLU())
        self.ctx_emb = nn.Embedding(n_ctx, 16)
        self.prev_card_emb = nn.Embedding(n_cards, d_card, padding_idx=0)
        d_in = 2 * d_player + N_PROG + n_logs + 16 + d_card + 2
        self.enc = nn.Sequential(nn.Linear(d_in, d_model), nn.ReLU(),
                                 nn.Linear(d_model, d_model), nn.ReLU())
        self.lstm = nn.LSTM(d_model, d_model, num_layers=n_layers, batch_first=True)
        self.head = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.ReLU(),
                                  nn.Linear(d_model // 2, 1))

    def encode_players(self, zone, attr):
        bag = self.card_emb(zone).sum(dim=-2)
        B, T, P, Z, D = bag.shape
        return self.player_enc(torch.cat([bag.reshape(B, T, P, Z * D), attr], -1))

    def forward(self, zone, attr, prog, logs, ctx, prev_cidx, prev_num, state=None):
        pe = self.encode_players(zone, attr)
        B, T = pe.shape[0], pe.shape[1]
        x = torch.cat([pe.reshape(B, T, -1), prog, logs, self.ctx_emb(ctx),
                       self.prev_card_emb(prev_cidx), prev_num], dim=-1)
        h, state = self.lstm(self.enc(x), state)
        return self.head(h).squeeze(-1), state


dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", dev, flush=True)
torch.manual_seed(0)
net = ValueNetV2().to(dev)
n_par = sum(p.numel() for p in net.parameters())
print(f"パラメータ {n_par:,}")

# --- 1. 形状 ---
B, T = 16, 120
mk = lambda: (torch.randint(1, 1200, (B, T, 2, N_ZONE, ZONE_CAP), device=dev),  # noqa
              torch.randn(B, T, 2, N_ATTR, device=dev),
              torch.randn(B, T, N_PROG, device=dev),
              torch.randn(B, T, 48, device=dev),
              torch.randint(0, 49, (B, T), device=dev),
              torch.randint(0, 1200, (B, T), device=dev),
              torch.randn(B, T, 2, device=dev))
args = mk()
out, _ = net(*args)
print(f"出力 {tuple(out.shape)}（期待 ({B}, {T})）")

# --- 2. 学習が回るか（合成タスク: p0の手札枚数が多いほど勝ち） ---
opt = torch.optim.Adam(net.parameters(), lr=3e-4)
bce = nn.BCEWithLogitsLoss()
zone, attr, prog, logs, ctx, pc, pn = args
target = (attr[:, :, 0, 0] > 0).float()          # attrの1列目で決まる合成ラベル
t0 = time.perf_counter()
for it in range(60):
    logits, _ = net(zone, attr, prog, logs, ctx, pc, pn)
    loss = bce(logits, target)
    opt.zero_grad()
    loss.backward()
    if it == 0:
        dead = [n for n, p in net.named_parameters()
                if p.grad is None or float(p.grad.abs().sum()) == 0.0]
        print(f"勾配が流れないパラメータ: {dead if dead else 'なし'}")
    opt.step()
    if it % 20 == 0:
        print(f"  it{it:>3} loss={loss.item():.4f}", flush=True)
print(f"  最終 loss={loss.item():.4f} / {time.perf_counter()-t0:.1f}秒")

# --- 3. メモリ実測（バッファ設計の判断材料） ---
per_dec_int16 = 2 * N_ZONE * ZONE_CAP * 2          # bytes
per_dec_other = (2 * N_ATTR + N_PROG + 48 + 3) * 4
print(f"\n1決定あたり: zone {per_dec_int16}B + その他 {per_dec_other}B "
      f"= {per_dec_int16+per_dec_other}B")
for games in (2000, 5000):
    dec = games * 100
    gb = dec * (per_dec_int16 + per_dec_other) / 1e9
    print(f"  {games}試合/イテレーション（約{dec:,}決定）: {gb:.2f} GB/イテレーション")
if dev == "cuda":
    print(f"GPUピーク {torch.cuda.max_memory_allocated()/1e9:.2f} GB "
          f"(B={B}, T={T})")
