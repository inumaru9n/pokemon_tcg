"""全選択LSTM方策の純numpy推論（tools/gen_featurizer.py が各エージェントへ複製する）。

元は agents/094_yushin_nn/nn_094.py。**生成器がフリーズ済みエージェントを参照しない**
よう tools/ 側を正本にした（094 は EXP-094 の記録として不変に保つ）。

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


def _combine_np(pointer, cls_logit, opt_cls, n_cls, mode):
    """pointer とクラスlogitを合成（numpy・1決定ぶん）。学習側 combine_scores と同一の式。

    クラス内分布はどの形でも softmax_{j∈c}(pointer) で同じ。違うのは P(クラス) の logit:
      additive   : a_c + logsumexp_{j∈c}(s_j)   （従来）
      factorized : a_c
      maxpool    : a_c + max_{j∈c}(s_j)
    **学習時と違う形で合成すると例外を出さずに別方策になる**ので、モデルnpzに
    記録された head をそのまま使う。
    """
    NEG = -1e9
    s = np.asarray(pointer, np.float64)
    ci = np.asarray([c if (c is not None and 0 <= c < n_cls) else -1
                     for c in opt_cls], np.int64)
    valid = ci >= 0
    if mode == "additive":
        out = s.copy()
        out[valid] += cls_logit[ci[valid]]
        return out
    if not valid.any():
        return s
    cj = ci[valid]
    sv = s[valid]
    m_c = np.full(n_cls, -np.inf)
    np.maximum.at(m_c, cj, sv)
    sum_c = np.zeros(n_cls)
    np.add.at(sum_c, cj, np.exp(sv - m_c[cj]))
    present = sum_c > 0
    lse_c = np.full(n_cls, NEG)
    lse_c[present] = m_c[present] + np.log(sum_c[present])
    pool = np.where(present, m_c, NEG) if mode == "maxpool" else (
        lse_c if mode == "logsumexp" else np.zeros(n_cls))
    cl = np.asarray(cls_logit, np.float64) + (0.0 if mode == "factorized" else pool)
    cl = np.where(present, cl, NEG)
    mx = cl.max()
    logp_c = cl - (mx + np.log(np.exp(cl - mx).sum()))
    out = s.copy()
    out[valid] = logp_c[cj] + (sv - lse_c[cj])
    return out



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
        self._lin_cache: dict[str, list] = {}
        # **特徴の標準化**。学習時の train 分割だけで求めた平均/標準偏差を npz に
        # 焼いてあるので、推論でも同じ変換を掛ける。掛け忘れると特徴のスケールが
        # 学習時と全く違う値になり、例外は出ないまま方策が壊れる。
        self.b_mu = self.w.get("board_mu")
        self.b_sd = self.w.get("board_sd")
        self.o_mu = self.w.get("opt_mu")
        self.o_sd = self.w.get("opt_sd")
        _hd = self.w.get("head")
        self.head = str(_hd) if _hd is not None else "additive"
        self.reset()

    def reset(self):
        """ゲーム開始時にLSTM状態を初期化する。"""
        self.h = [np.zeros(self.h_dim, np.float32) for _ in range(self.n_layers)]
        self.c = [np.zeros(self.h_dim, np.float32) for _ in range(self.n_layers)]

    def _cidx(self, cid: int) -> int:
        return self.vocab.get(int(cid), 1)  # 1 = unknown

    def _lin_idx(self, prefix: str):
        """{prefix}.N.weight として保存されている線形層の N を昇順に返す。

        **番号を 0,2 と決め打ちしない**。学習側に Dropout を足すと nn.Sequential の
        番号が 0,3 のようにずれるが、Dropout は推論では恒等なので重みだけ拾えば
        同じ計算になる。決め打ちだと KeyError で無言フォールバックに落ちる。
        """
        key = self._lin_cache.get(prefix)
        if key is None:
            key = sorted(int(k.split(".")[1]) for k in self.w
                         if k.startswith(prefix + ".") and k.endswith(".weight"))
            self._lin_cache[prefix] = key
        return key

    def _mlp(self, x, prefix: str, n_layers: int = 2):
        """Linear→ReLU を線形層の数だけ繰り返す（Dropoutは推論では恒等）。"""
        for i in self._lin_idx(prefix):
            x = x @ self.w[f"{prefix}.{i}.weight"].T + self.w[f"{prefix}.{i}.bias"]
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
        bd = np.asarray(board, np.float32)
        if self.b_mu is not None:
            bd = (bd - self.b_mu) / self.b_sd
        x = np.concatenate([
            bd, np.asarray(logs, np.float32),
            e_ctx, e_prev, np.asarray(prev[1:3], np.float32)]).astype(np.float32)
        z = self._mlp(x, "enc")
        h = self._lstm_step(z)

        E = self.w["card_emb.weight"]
        e_card = np.stack([E[self._cidx(c)] for c in opt_cids])     # (O, d_card)
        ov = np.asarray(opt_vecs, np.float32)                        # (O, D)
        if self.o_mu is not None:
            ov = (ov - self.o_mu) / self.o_sd
        e_opt = self._mlp(np.concatenate([e_card, ov], axis=1), "opt_enc")  # (O, d_opt)
        q = (h @ self.w["proj.weight"].T + self.w["proj.bias"]) \
            * self.w["ctx_scale.weight"][ctx]
        sc = e_opt @ q + float(self.w["ctx_bias.weight"][ctx][0])

        # MAIN階層ヘッド（MAINは「行動タイプ×対象」が1selectに圧縮されており
        # エンジンが階層を出さないので、ここだけ自前で階層を入れる）
        if opt_cls is not None and "cls_head.weight" in self.w:
            cl = h @ self.w["cls_head.weight"].T + self.w["cls_head.bias"]
            sc = _combine_np(sc, cl, opt_cls, len(cl), self.head)
        return sc
