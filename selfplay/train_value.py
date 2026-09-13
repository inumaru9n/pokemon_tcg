"""自己対戦データから**完全情報のLSTM価値関数**を学習し、キルゲートを判定する（Stage 1）。

**この実験が答える問い**: EXP-055 は「観察データ（本番episodes）から学んだ勝率回帰は交絡で
使えない（勝者の盤面と相関するものを選ぶだけで、勝ちに向かう行動を選ばない）。**根治は
オンポリシー自己対戦のみ**」と機序まで特定した。その処方をここで初めて実行し、
**自己対戦なら本当に価値関数が立つのか**を測る。

**設計**: 方策（PolicyNet）と**同一アーキテクチャのトランク**（MLPエンコーダ → LSTM 2層×256）。
違いは2点だけ:
  1. 入力に**相手側の盤面特徴**を連結する（AlphaStarのprivileged value baseline。
     学習時のみ使い、提出物には載らないので合法）
  2. 出力はpointerスコアでなく**スカラー価値**（勝率）

**方策とは重みを共有しない**（別ネットワーク）。理由:
  - 特権情報が方策に漏れる配線ミスが原理的に起こらない
  - 094からのKLアンカーと価値勾配がトランクを取り合わない
  - Vは学習時にしか使わない（advantage計算のみ）ので共有してもコストが浮かない

**事前登録した合否**:
  PASS  ターン5以降の勝敗AUC >= 0.75 かつ ターン帯で単調増加
  FAIL  AUC < 0.65 または単調性が壊れる → AlphaStar型を撤退

usage:
  uv run --python .venv_train/bin/python selfplay/train_value.py --npz data/rl/sp_seq_40k.npz
  （Kaggleでは push_kaggle.sh 経由でGPU実行）
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn as nn


class ValueNet(nn.Module):
    """方策と同じトランク（MLP → LSTM）+ スカラー価値ヘッド。

    **入力の並びは方策(PolicyNet)と完全に一致させる**:
        [board, logs, ctx_emb, prev_card_emb, prev_num]  ← 方策と同一（224次元）
        ++ [opp_board]                                    ← 特権情報（末尾に追加）
    こうしておくと `--init-from` で方策の重みを**そのままコピー**でき、
    追加した特権ブロックをゼロ初期化すれば **初期時点のトランク出力が方策と完全一致**する。
    """

    def __init__(self, n_board, n_logs, n_cards, n_ctx=49, d_card=32,
                 d_model=256, n_layers=2):
        super().__init__()
        self.n_board = n_board
        self.ctx_emb = nn.Embedding(n_ctx, 16)
        self.prev_card_emb = nn.Embedding(n_cards, d_card, padding_idx=0)
        self.d_shared = n_board + n_logs + 16 + d_card + 2   # 方策と同じ入力幅
        d_in = self.d_shared + n_board                        # + 相手盤面（特権）
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
        return self.head(h).squeeze(-1), state          # (B,T) ロジット

    def trunk(self, board, oppb, logs, ctx, prev_cidx, prev_num, state=None):
        """価値ヘッド手前まで（暖機の一致検証用）。"""
        x = torch.cat([board, logs, self.ctx_emb(ctx),
                       self.prev_card_emb(prev_cidx), prev_num, oppb], dim=-1)
        h, _ = self.lstm(self.enc(x), state)
        return h


def warm_start(net: ValueNet, policy_npz: str) -> dict:
    """方策(model_094.npz)の重みを価値ネットにコピーする。

    追加した特権ブロック（enc.0 の末尾 n_board 列）は**ゼロ初期化**するので、
    初期時点のトランク出力は方策と厳密に一致する（特権情報の寄与は0から育つ）。
    価値ヘッドのみ新規初期化のまま。
    """
    w = np.load(policy_npz, allow_pickle=False)
    sd = net.state_dict()
    nb = net.n_board
    with torch.no_grad():
        sd["ctx_emb.weight"].copy_(torch.as_tensor(w["ctx_emb.weight"]))
        sd["prev_card_emb.weight"].copy_(torch.as_tensor(w["prev_card_emb.weight"]))
        e0 = torch.as_tensor(w["enc.0.weight"])          # (d_model, d_shared)
        assert e0.shape[1] == net.d_shared, \
            f"方策の入力幅 {e0.shape[1]} != 想定 {net.d_shared}"
        sd["enc.0.weight"].zero_()
        sd["enc.0.weight"][:, :net.d_shared].copy_(e0)   # 末尾 nb 列は 0 のまま
        sd["enc.0.bias"].copy_(torch.as_tensor(w["enc.0.bias"]))
        sd["enc.2.weight"].copy_(torch.as_tensor(w["enc.2.weight"]))
        sd["enc.2.bias"].copy_(torch.as_tensor(w["enc.2.bias"]))
        for k in list(sd):
            if k.startswith("lstm.") and k in w.files:
                sd[k].copy_(torch.as_tensor(w[k]))
    net.load_state_dict(sd)
    vocab = {int(k): int(v) for k, v in zip(w["vocab_keys"], w["vocab_vals"])}
    return vocab


def auc(y: np.ndarray, s: np.ndarray) -> float:
    """ROC AUC（Mann-Whitney U）。"""
    if len(np.unique(y)) < 2:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    r = np.empty(len(s), float)
    sv = s[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    n1 = float(y.sum())
    n0 = float(len(y) - n1)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", nargs="+", default=["data/rl/sp_seq_40k.npz"],
                    help="複数指定可（シャードを連結する）。データ量スケーリング曲線用")
    ap.add_argument("--limit-games", type=int, default=0,
                    help=">0 なら先頭からこのゲーム数だけ使う（曲線の測定用）")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--val-frac", type=float, default=0.12)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--policy", default="agents/094_yushin_nn/model_094.npz",
                    help="カードID→indexの語彙をここから読む（暖機の有無に関わらず必要）")
    ap.add_argument("--warm", action="store_true",
                    help="方策の重みを暖機コピーする（特権ブロックは0初期化）")
    ap.add_argument("--out", default="data/rl/value_lstm.npz")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = "cpu"
    if torch.cuda.is_available():
        try:
            _l = torch.nn.LSTM(4, 4).cuda()
            _o, _ = _l(torch.zeros(2, 1, 4, device="cuda"))
            _ = float(_o.sum())
            dev = "cuda"
        except Exception as e:  # noqa: BLE001
            print(f"CUDA使用不可（{type(e).__name__}）→ CPU")
    print("device:", dev, flush=True)

    paths = [p for p in args.npz if os.path.exists(p)]
    if not paths:
        import glob
        paths = sorted(glob.glob("/kaggle/input/**/sp_seq*.npz", recursive=True))
        print("Kaggle入力から検出:", paths)
    parts = []
    g_off, s_off = 0, 0
    for p in paths:
        z = np.load(p)
        parts.append(dict(board=z["board"], oppb=z["opp_board"], logs=z["logs"],
                          prev=z["prev"], ctx=z["ctx"], turn=z["turn"],
                          sid=z["seq_id"] + s_off, label=z["label"],
                          gid=z["game_id"] + g_off))
        s_off += len(z["label"])
        g_off += int(z["game_id"].max()) + 1     # シャード間でgame_idが衝突しないようずらす
    cat = lambda k: np.concatenate([p[k] for p in parts])  # noqa: E731
    board, oppb, logs = cat("board"), cat("oppb"), cat("logs")
    prev, ctx, turn, sid = cat("prev"), cat("ctx"), cat("turn"), cat("sid")
    label, gid = cat("label"), cat("gid")
    # **valは全データから先に固定して取る**。limit-games の後に分割すると
    # データ量ごとにval集合が変わり、「学習データ量の効果」と「val集合の違い」が混ざる
    # （1倍化あたり0.01の判定に対し、val由来の揺れが同程度になる）。
    _all_g = np.unique(gid)
    _rng = np.random.default_rng(42)
    _perm = _rng.permutation(_all_g)
    _n_val = max(1, int(len(_all_g) * args.val_frac))
    VAL_GAMES = set(int(x) for x in _perm[:_n_val])
    TRAIN_POOL = [int(x) for x in _perm[_n_val:]]

    if args.limit_games:
        # 学習側だけを絞る。**valは触らない**ので3点で同一の評価集合になる
        keep_g = set(TRAIN_POOL[:args.limit_games]) | VAL_GAMES
        ks = np.fromiter((int(g) in keep_g for g in gid), bool, len(gid))
        keep_seq = np.where(ks)[0]
        rowmask = np.isin(sid, keep_seq)
        remap = -np.ones(len(label), np.int64)
        remap[keep_seq] = np.arange(len(keep_seq))
        board, oppb, logs = board[rowmask], oppb[rowmask], logs[rowmask]
        prev, ctx, turn = prev[rowmask], ctx[rowmask], turn[rowmask]
        sid = remap[sid[rowmask]]
        label, gid = label[keep_seq], gid[keep_seq]
    n_seq = len(label)
    print(f"シャード {len(paths)}本 / ゲーム {len(np.unique(gid)):,} / 系列 {n_seq:,} / "
          f"決定 {len(board):,} / 盤面{board.shape[1]} logs{logs.shape[1]}")

    # 系列の開始位置
    starts = np.searchsorted(sid, np.arange(n_seq))
    ends = np.append(starts[1:], len(sid))
    lens = ends - starts

    # **分割はゲーム単位**（同一ゲームの2系列=両プレイヤーは強く相関するので、
    # 系列単位で割るとtrain/valにまたがってAUCが偽に高く出る）
    ug = np.unique(gid)
    is_val = np.fromiter((int(g) in VAL_GAMES for g in gid), bool, n_seq)
    tr_idx = np.where(~is_val)[0]
    va_idx = np.where(is_val)[0]
    n_vg = len(set(int(g) for g in gid[is_val]))
    print(f"分割: train系列 {len(tr_idx):,} / val系列 {len(va_idx):,}"
          f"（学習ゲーム {len(ug)-n_vg:,} / **val {n_vg:,}=全データ量で共通**）")

    # 語彙（カードID→index）は暖機の有無に関わらず方策npzから読む。これは学習でなく
    # 前処理なので、スクラッチ実行でも同じ写像を使うのが正しい（埋め込みの重みだけ乱数）
    pol_path = args.policy
    if not os.path.exists(pol_path):
        import glob
        c = sorted(glob.glob("/kaggle/input/**/model_094*.npz", recursive=True))
        if c:
            pol_path = c[0]
    pw = np.load(pol_path, allow_pickle=False)
    vocab = {int(k): int(v) for k, v in zip(pw["vocab_keys"], pw["vocab_vals"])}
    n_cards = int(pw["card_emb.weight"].shape[0])
    d_card = int(pw["card_emb.weight"].shape[1])
    lut = np.full(int(max(vocab)) + 4, 1, np.int64)      # 既定=unknown(1)
    for k, v in vocab.items():
        lut[k + 3] = v                                   # NULL_CID=-3 に合わせて+3
    prev_cidx_all = lut[np.clip(prev[:, 0].astype(np.int64) + 3, 0, len(lut) - 1)]

    net = ValueNet(board.shape[1], logs.shape[1], n_cards=n_cards, d_card=d_card,
                   d_model=args.d_model, n_layers=args.layers).to(dev)
    if args.warm:
        warm_start(net, pol_path)
        print(f"暖機: 方策 {os.path.basename(pol_path)} からコピー"
              f"（特権ブロック{board.shape[1]}列はゼロ初期化）")
    else:
        print("スクラッチ初期化")
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    print("params:", sum(p.numel() for p in net.parameters()))

    def make_batch(idx):
        T = int(lens[idx].max())
        B = len(idx)
        b = np.zeros((B, T, board.shape[1]), np.float32)
        o = np.zeros((B, T, board.shape[1]), np.float32)
        g = np.zeros((B, T, logs.shape[1]), np.float32)
        c = np.zeros((B, T), np.int64)
        pc = np.zeros((B, T), np.int64)
        pn = np.zeros((B, T, 2), np.float32)
        y = np.zeros((B, T), np.float32)
        m = np.zeros((B, T), bool)
        tn = np.zeros((B, T), np.int16)
        for r, s in enumerate(idx):
            a, e = starts[s], ends[s]
            L = e - a
            b[r, :L] = board[a:e]
            o[r, :L] = oppb[a:e]
            g[r, :L] = logs[a:e]
            c[r, :L] = ctx[a:e]
            pc[r, :L] = prev_cidx_all[a:e]
            pn[r, :L] = prev[a:e, 1:3]
            y[r, :L] = label[s]
            m[r, :L] = True
            tn[r, :L] = turn[a:e]
        t = lambda a: torch.as_tensor(a, device=dev)  # noqa: E731
        return t(b), t(o), t(g), t(c), t(pc), t(pn), t(y), t(m), tn

    bce = nn.BCEWithLogitsLoss(reduction="none")

    def run(idx_all, train):
        net.train(train)
        order = np.random.permutation(idx_all) if train else idx_all
        tot, n = 0.0, 0
        preds, ys, tns = [], [], []
        for s in range(0, len(order), args.batch):
            idx = order[s:s + args.batch]
            b, o, g, c, pc, pn, y, m, tn = make_batch(idx)
            with torch.set_grad_enabled(train):
                logit, _ = net(b, o, g, c, pc, pn)
                loss = (bce(logit, y) * m).sum() / m.sum().clamp(min=1)
            if train:
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                opt.step()
            tot += float(loss) * int(m.sum())
            n += int(m.sum())
            if not train:
                mm = m.cpu().numpy()
                preds.append(torch.sigmoid(logit).detach().cpu().numpy()[mm])
                ys.append(y.cpu().numpy()[mm])
                tns.append(tn[mm])
        if train:
            return tot / max(1, n), None
        return tot / max(1, n), (np.concatenate(preds), np.concatenate(ys),
                                 np.concatenate(tns))

    best, bad, best_state, best_val = 1e9, 0, None, None
    for ep in range(args.epochs):
        trl, _ = run(tr_idx, True)
        val, pack = run(va_idx, False)
        sched.step()
        a = auc(pack[1], pack[0])
        star = ""
        if val < best - 1e-5:
            best, bad = val, 0
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            best_val = pack
            star = " *"
        else:
            bad += 1
        print(f"ep{ep:03d} train {trl:.4f} | val {val:.4f} AUC {a:.4f}{star}", flush=True)
        if bad >= args.patience:
            print(f"  early stop (patience {args.patience})")
            break

    p, y, tn = best_val
    print(f"\n=== キルゲート判定 ===")
    print(f"全体AUC: {auc(y, p):.4f}")
    print(f"{'ターン帯':<12}{'n':>10}{'AUC':>9}{'勝率':>8}")
    bands = [(0, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 14), (15, 99)]
    late = []
    for lo, hi in bands:
        mm = (tn >= lo) & (tn <= hi)
        if mm.sum() < 200:
            print(f"  t{lo}-{hi:<9}{int(mm.sum()):>10}{'n<200':>9}")
            continue
        a = auc(y[mm], p[mm])
        print(f"  t{lo}-{hi:<9}{int(mm.sum()):>10}{a:>9.4f}{y[mm].mean():>8.3f}")
        if lo >= 5:
            late.append(a)
    mono = all(late[i] <= late[i + 1] + 0.02 for i in range(len(late) - 1))
    la = float(np.mean(late)) if late else float("nan")
    print(f"\nターン5以降の平均AUC: {la:.4f} / 単調増加: {'はい' if mono else 'いいえ'}")
    if la >= 0.75 and mono:
        print("判定: **PASS** → Stage 2 (PPO) へ進む")
    elif la < 0.65:
        print("判定: **FAIL** → AlphaStar型は撤退（事前登録の基準）")
    else:
        print("判定: **保留**（0.65-0.75 or 非単調）→ データ量/特徴を増やして再測定")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    sd = {k: v.numpy() for k, v in best_state.items()}
    np.savez_compressed(args.out, n_board=np.int32(board.shape[1]),
                        n_logs=np.int32(logs.shape[1]), **sd)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
