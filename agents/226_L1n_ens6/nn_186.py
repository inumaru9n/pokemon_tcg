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
        _oa = self.w.get("opt_attn")
        self.opt_attn = bool(_oa) and "a_q.weight" in self.w
        _nc = self.w.get("norm_clip")
        self.nclip = float(_nc) if _nc is not None else 0.0
        _hd = self.w.get("head")
        self.head = str(_hd) if _hd is not None else "additive"
        self.reset()

    def get_state(self):
        """LSTM状態のスナップショット。**探索で仮の手を試す前に退避する**。"""
        return ([x.copy() for x in self.h], [x.copy() for x in self.c])

    def set_state(self, st):
        """退避した状態に戻す。探索が本番の状態を汚さないために必須。"""
        self.h = [x.copy() for x in st[0]]
        self.c = [x.copy() for x in st[1]]

    def value(self, board, logs, prev, ctx: int) -> float:
        """盤面の勝率推定（-1〜+1）。**LSTM状態を1ステップ進める**ので、
        探索中は get_state/set_state で括ること。v_head が無いモデルでは 0 を返す。"""
        if "v_head.weight" not in self.w and "v_head.0.weight" not in self.w:
            return 0.0
        h = self._advance(board, logs, prev, ctx)
        if "v_head.0.weight" in self.w:      # 隠れ層あり
            x = h @ self.w["v_head.0.weight"].T + self.w["v_head.0.bias"]
            x = np.maximum(x, 0.0)
            v = x @ self.w["v_head.2.weight"].T + self.w["v_head.2.bias"]
        else:
            v = h @ self.w["v_head.weight"].T + self.w["v_head.bias"]
        return float(np.tanh(v)[0])

    def _advance(self, board, logs, prev, ctx: int):
        """board/logs/ctx/prev から h を計算し、LSTM状態を進める（scores と共通）。"""
        e_ctx = self.w["ctx_emb.weight"][ctx]
        e_prev = self.w["prev_card_emb.weight"][self._cidx(prev[0])]
        bd = np.asarray(board, np.float32)
        if self.b_mu is not None:
            bd = (bd - self.b_mu) / self.b_sd
            if self.nclip:            # 分布外の相手で σ が爆発するのを防ぐ（EXP-111）
                bd = np.clip(bd, -self.nclip, self.nclip)
        x = np.concatenate([bd, np.asarray(logs, np.float32), e_ctx, e_prev,
                            np.asarray(prev[1:3], np.float32)]).astype(np.float32)
        return self._lstm_step(self._mlp(x, "enc"))

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
        h = self._advance(board, logs, prev, ctx)

        E = self.w["card_emb.weight"]
        e_card = np.stack([E[self._cidx(c)] for c in opt_cids])     # (O, d_card)
        ov = np.asarray(opt_vecs, np.float32)                        # (O, D)
        if self.o_mu is not None:
            ov = (ov - self.o_mu) / self.o_sd
            if self.nclip:
                ov = np.clip(ov, -self.nclip, self.nclip)
        e_opt = self._mlp(np.concatenate([e_card, ov], axis=1), "opt_enc")  # (O, d_opt)
        if self.opt_attn:
            # **選択肢間 self-attention**（EXP-112 軸B）。torch側と厳密に一致させる:
            #   att = QK^T / sqrt(d) → softmax → e_opt += a_o(att @ V)
            # 推論時は実オプションだけを渡されるので padding マスクは不要。
            def _lin(x, nm):
                return x @ self.w[f"{nm}.weight"].T + self.w[f"{nm}.bias"]
            aq, ak, av = _lin(e_opt, "a_q"), _lin(e_opt, "a_k"), _lin(e_opt, "a_v")
            att = (aq @ ak.T) / np.sqrt(aq.shape[-1], dtype=np.float32)
            att -= att.max(axis=-1, keepdims=True)        # softmax の数値安定化
            w_att = np.exp(att)
            w_att /= w_att.sum(axis=-1, keepdims=True)
            e_opt = e_opt + _lin(w_att @ av, "a_o")
        q = (h @ self.w["proj.weight"].T + self.w["proj.bias"]) \
            * self.w["ctx_scale.weight"][ctx]
        sc = e_opt @ q + float(self.w["ctx_bias.weight"][ctx][0])

        # MAIN階層ヘッド（MAINは「行動タイプ×対象」が1selectに圧縮されており
        # エンジンが階層を出さないので、ここだけ自前で階層を入れる）
        if opt_cls is not None and "cls_head.weight" in self.w:
            cl = h @ self.w["cls_head.weight"].T + self.w["cls_head.bias"]
            sc = _combine_np(sc, cl, opt_cls, len(cl), self.head)
        return sc


class EnsemblePolicy:
    """同一特徴器で学習した複数モデルの**確率平均**アンサンブル（EXP-117）。

    LSTMPolicy と同じインタフェースを持つので main.py 側は `_NET` を差し替えるだけでよい。

    **各メンバーは独自のLSTM状態を持つ**。決定ごとに全メンバーが1ステップ進み、
    実際に選ばれた手（`prev`）は全メンバーに共通で流れる＝「1本の行動軌跡を共有する
    方策のアンサンブル」。状態を共有すると層の意味が個体ごとに違うので混ぜてはいけない。

    合成は**決定ごとの softmax を平均**してから log を取る:
        score_i = log( mean_m softmax(s^m)_i )
    生スコアの平均ではない。`scores` の戻り値はモデルごとにスケールが違う
    （ctx_bias/ctx_scale が個体ごとに学習される）ので、そのまま平均すると
    確信の強い個体でなく**スケールの大きい個体**が勝ってしまう。
    argsort しか使わないので log を取る意味は無いが、単体モデルと同じ
    「大きいほど良い実数スコア」の規約に揃えておく（CLASS_BIAS の加算先でもある）。
    """

    def __init__(self, paths):
        paths = list(paths)
        assert paths, "モデルが1つも指定されていない"
        self.nets = [LSTMPolicy(p) for p in paths]
        self.w = self.nets[0].w
        # **メンバー間の整合を起動時に確認する**。特徴器・行動クラス・語彙・標準化統計が
        # 違うモデルを混ぜると例外は出ないまま別方策になる（無言劣化の典型）。
        a = self.nets[0]
        for i, n in enumerate(self.nets[1:], 1):
            for key in ("feat_keys", "classes", "vocab_keys", "vocab_vals"):
                x, y = a.w.get(key), n.w.get(key)
                assert (x is None) == (y is None) and (
                    x is None or [str(v) for v in x] == [str(v) for v in y]), \
                    f"メンバー{i}の {key} が0番目と違う"
            assert a.w["enc.0.weight"].shape[1] == n.w["enc.0.weight"].shape[1], \
                f"メンバー{i}の入力次元が違う"
            assert a.head == n.head, f"メンバー{i}の head が違う"
            for key in ("board_mu", "board_sd", "opt_mu", "opt_sd"):
                x, y = a.w.get(key), n.w.get(key)
                assert (x is None) == (y is None), f"メンバー{i}の {key} の有無が違う"
        self.head = a.head
        self.opt_attn = a.opt_attn
        self.nclip = a.nclip
        self.b_mu, self.b_sd = a.b_mu, a.b_sd
        self.o_mu, self.o_sd = a.o_mu, a.o_sd

    def reset(self):
        for n in self.nets:
            n.reset()

    def get_state(self):
        return [n.get_state() for n in self.nets]

    def set_state(self, st):
        for n, s in zip(self.nets, st):
            n.set_state(s)

    def value(self, board, logs, prev, ctx: int) -> float:
        """メンバーの V の平均。**全メンバーの状態を1ステップ進める**点は単体と同じ。"""
        vs = [n.value(board, logs, prev, ctx) for n in self.nets]
        return float(np.mean(vs))

    def scores(self, board, logs, prev, ctx: int, opt_cids, opt_vecs,
               opt_cls=None):
        p = None
        for n in self.nets:
            s = np.asarray(n.scores(board, logs, prev, ctx, opt_cids, opt_vecs,
                                    opt_cls), np.float64)
            e = np.exp(s - s.max())
            e /= e.sum()
            p = e if p is None else p + e
        return np.log(np.maximum(p / len(self.nets), 1e-300))
