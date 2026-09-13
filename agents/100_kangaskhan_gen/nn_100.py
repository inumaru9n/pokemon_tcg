"""EXP-094: 全選択LSTM方策の純numpy推論。

Kaggle評価環境にtorchは無いため、学習済み重み（model_094.npz）をnumpyだけで前向き計算する。
PyTorchのLSTM重みレイアウトに合わせてある:
  weight_ih_l{k} (4H, in), weight_hh_l{k} (4H, H), bias_ih_l{k}/bias_hh_l{k} (4H,)
  ゲート順は i, f, g, o

推論コストの目安: LSTM 2層×256 で 1決定あたり 4行列積×2層 ≈ 0.1-0.5 ms。
"""

from __future__ import annotations

import numpy as np


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


class LSTMPolicy:
    """model_094.npz を読み、1決定ずつ状態を進めながらオプションをスコアする。"""

    def __init__(self, path: str):
        z = np.load(path, allow_pickle=True)
        self.w = {k: z[k] for k in z.files}
        # カードID → 埋め込みindex
        keys = self.w["vocab_keys"].astype(np.int64)
        vals = self.w["vocab_vals"].astype(np.int64)
        self.vocab = {int(k): int(v) for k, v in zip(keys, vals)}
        self.n_layers = sum(1 for k in self.w if k.startswith("lstm.weight_ih_l"))
        self.h_dim = self.w["lstm.weight_hh_l0"].shape[1]
        self.reset()

    def reset(self):
        """ゲーム開始時にLSTM状態を初期化する。"""
        self.h = [np.zeros(self.h_dim, np.float32) for _ in range(self.n_layers)]
        self.c = [np.zeros(self.h_dim, np.float32) for _ in range(self.n_layers)]

    def _cidx(self, cid: int) -> int:
        return self.vocab.get(int(cid), 1)  # 1 = unknown

    def _mlp(self, x, prefix: str, n_layers: int = 2):
        """nn.Sequential(Linear, ReLU, Linear, ReLU) 相当。"""
        for i in range(n_layers):
            W = self.w[f"{prefix}.{2*i}.weight"]
            b = self.w[f"{prefix}.{2*i}.bias"]
            x = x @ W.T + b
            x = np.maximum(x, 0.0)
        return x

    def _lstm_step(self, x):
        """1ステップ分のLSTM前向き（状態を破壊的に更新）。"""
        inp = x
        for k in range(self.n_layers):
            Wi = self.w[f"lstm.weight_ih_l{k}"]
            Wh = self.w[f"lstm.weight_hh_l{k}"]
            bi = self.w[f"lstm.bias_ih_l{k}"]
            bh = self.w[f"lstm.bias_hh_l{k}"]
            g = inp @ Wi.T + bi + self.h[k] @ Wh.T + bh
            H = self.h_dim
            i_g = _sigmoid(g[0:H])
            f_g = _sigmoid(g[H:2 * H])
            g_g = np.tanh(g[2 * H:3 * H])
            o_g = _sigmoid(g[3 * H:4 * H])
            self.c[k] = f_g * self.c[k] + i_g * g_g
            self.h[k] = o_g * np.tanh(self.c[k])
            inp = self.h[k]
        return inp

    def scores(self, board, logs, prev, ctx: int, opt_cids, opt_vecs,
               opt_cls=None):
        """1決定 → 各オプションのスコア（LSTM状態を1ステップ進める）。

        board: (F,) / logs: (L,) / prev: (3,) / opt_vecs: (O, D)
        opt_cls: MAINの行動クラスID列（非MAINはNone）。階層ヘッドの加算に使う。
        """
        e_ctx = self.w["ctx_emb.weight"][ctx]
        e_prev = self.w["prev_card_emb.weight"][self._cidx(prev[0])]
        x = np.concatenate([
            np.asarray(board, np.float32), np.asarray(logs, np.float32),
            e_ctx, e_prev, np.asarray(prev[1:3], np.float32)]).astype(np.float32)
        z = self._mlp(x, "enc")
        h = self._lstm_step(z)

        E = self.w["card_emb.weight"]
        e_card = np.stack([E[self._cidx(c)] for c in opt_cids])     # (O, d_card)
        ov = np.asarray(opt_vecs, np.float32)                        # (O, D)
        e_opt = self._mlp(np.concatenate([e_card, ov], axis=1), "opt_enc")  # (O, d_opt)
        q = (h @ self.w["proj.weight"].T + self.w["proj.bias"]) \
            * self.w["ctx_scale.weight"][ctx]
        sc = e_opt @ q + float(self.w["ctx_bias.weight"][ctx][0])

        # MAIN階層ヘッド: score_i += class_logit[cls(i)]
        # （MAINは「行動タイプ×対象」が1selectに圧縮されておりエンジンが階層を出さない）
        if opt_cls is not None and "cls_head.weight" in self.w:
            cl = h @ self.w["cls_head.weight"].T + self.w["cls_head.bias"]
            for i, c in enumerate(opt_cls):
                if c is not None and 0 <= c < len(cl):
                    sc[i] += cl[c]
        return sc
