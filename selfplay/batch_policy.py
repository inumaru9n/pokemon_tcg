"""094のLSTM方策を**Bバトル分まとめて**前向き計算する（I-109 Phase RL Stage 0）。

agents/094_yushin_nn/nn_094.py は提出物用の1決定ずつの実装で、そこは触らない。
こちらは自己対戦生成専用のバッチ版で、行列積を (B, ...) に束ねてNN forwardの
per-decision コストを潰す（EXP-094実測で生成コストの62%がここ）。

**状態の持ち方**: 1バトルにつきプレイヤー2人ぶんのLSTM状態を持つ（自己対戦でも
両者は別々の観測列を見るため状態は共有できない）。スロットは (battle_idx, player) で識別。
"""

from __future__ import annotations

import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "agents/094_yushin_nn")):
    if _p not in sys.path:
        sys.path.insert(0, _p)




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



def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


class BatchLSTMPolicy:
    """model_094.npz を読み、B個のスロットの状態を並行に進める。"""

    def __init__(self, path: str, n_slots: int):
        z = np.load(path, allow_pickle=False)
        self.w = {k: z[k] for k in z.files}
        keys = self.w["vocab_keys"].astype(np.int64)
        vals = self.w["vocab_vals"].astype(np.int64)
        self.vocab = {int(k): int(v) for k, v in zip(keys, vals)}
        self.n_layers = sum(1 for k in self.w if k.startswith("lstm.weight_ih_l"))
        self.h_dim = int(self.w["lstm.weight_hh_l0"].shape[1])
        self.trained_ctx = ({int(c) for c in self.w["trained_ctx"]}
                            if "trained_ctx" in self.w else None)
        self.n_slots = n_slots
        self._lin_cache: dict[str, list] = {}
        # 学習時の標準化統計（無い旧モデルは None のまま＝素通し）
        self.b_mu = self.w.get("board_mu")
        self.b_sd = self.w.get("board_sd")
        self.o_mu = self.w.get("opt_mu")
        self.o_sd = self.w.get("opt_sd")
        _hd = self.w.get("head")
        self.head = str(_hd) if _hd is not None else "additive"
        self.reset_all()

    def reset_all(self):
        self.h = [np.zeros((self.n_slots, self.h_dim), np.float32)
                  for _ in range(self.n_layers)]
        self.c = [np.zeros((self.n_slots, self.h_dim), np.float32)
                  for _ in range(self.n_layers)]

    def reset(self, slots):
        for k in range(self.n_layers):
            self.h[k][slots] = 0.0
            self.c[k][slots] = 0.0

    def _cidx(self, cid) -> int:
        return self.vocab.get(int(cid), 1)

    def _lin_idx(self, prefix: str):
        """線形層の番号を npz のキーから発見する（Dropout でずれても壊れない）。"""
        k = self._lin_cache.get(prefix)
        if k is None:
            k = sorted(int(j.split(".")[1]) for j in self.w
                       if j.startswith(prefix + ".") and j.endswith(".weight"))
            self._lin_cache[prefix] = k
        return k

    def _mlp(self, x, prefix: str, n_layers: int = 2):
        for i in self._lin_idx(prefix):
            x = x @ self.w[f"{prefix}.{i}.weight"].T + self.w[f"{prefix}.{i}.bias"]
            x = np.maximum(x, 0.0)
        return x

    def _lstm_step(self, slots, x):
        """slots (B,) のLSTM状態を x (B, d) で1ステップ進め、最終層の h を返す。"""
        inp = x
        H = self.h_dim
        for k in range(self.n_layers):
            Wi, Wh = self.w[f"lstm.weight_ih_l{k}"], self.w[f"lstm.weight_hh_l{k}"]
            bi, bh = self.w[f"lstm.bias_ih_l{k}"], self.w[f"lstm.bias_hh_l{k}"]
            hprev, cprev = self.h[k][slots], self.c[k][slots]
            g = inp @ Wi.T + bi + hprev @ Wh.T + bh
            i_g = _sigmoid(g[:, 0:H])
            f_g = _sigmoid(g[:, H:2 * H])
            g_g = np.tanh(g[:, 2 * H:3 * H])
            o_g = _sigmoid(g[:, 3 * H:4 * H])
            cnew = f_g * cprev + i_g * g_g
            hnew = o_g * np.tanh(cnew)
            self.c[k][slots] = cnew
            self.h[k][slots] = hnew
            inp = hnew
        return inp

    def scores(self, slots, boards, logs, prevs, ctxs, opt_cids, opt_vecs, opt_clss):
        """B決定ぶんのスコアを返す。

        slots: (B,) LSTMスロット番号 / boards,logs: (B,F),(B,L)
        prevs: (B,3) / ctxs: (B,)
        opt_cids / opt_vecs / opt_clss: 長さBのリスト（各要素は可変長）
        戻り: 長さBのリスト（各要素は (n_opt,) のスコア配列）
        """
        B = len(slots)
        slots = np.asarray(slots, np.int64)
        ctxs = np.asarray(ctxs, np.int64)
        e_ctx = self.w["ctx_emb.weight"][ctxs]
        prevs = np.asarray(prevs, np.float32)
        pc = np.array([self._cidx(p) for p in prevs[:, 0]], np.int64)
        e_prev = self.w["prev_card_emb.weight"][pc]
        bd = np.asarray(boards, np.float32)
        if self.b_mu is not None:
            bd = (bd - self.b_mu) / self.b_sd
        x = np.concatenate([bd, np.asarray(logs, np.float32),
                            e_ctx, e_prev, prevs[:, 1:3]], axis=1).astype(np.float32)
        h = self._lstm_step(slots, self._mlp(x, "enc"))          # (B, d_model)

        q = (h @ self.w["proj.weight"].T + self.w["proj.bias"]) \
            * self.w["ctx_scale.weight"][ctxs]                   # (B, d_opt)
        cbias = self.w["ctx_bias.weight"][ctxs][:, 0]            # (B,)
        has_cls = "cls_head.weight" in self.w
        if has_cls:
            cl = h @ self.w["cls_head.weight"].T + self.w["cls_head.bias"]  # (B, n_cls)

        # オプション側は全バトルぶんを1本に連結して1回のMLPで処理する
        flat_cid, flat_vec, owner = [], [], []
        for b in range(B):
            for cid in opt_cids[b]:
                flat_cid.append(self._cidx(cid))
            flat_vec.extend(opt_vecs[b])
            owner.extend([b] * len(opt_cids[b]))
        if not flat_cid:
            return [np.zeros(0, np.float32) for _ in range(B)]
        e_card = self.w["card_emb.weight"][np.asarray(flat_cid, np.int64)]
        fv = np.asarray(flat_vec, np.float32)
        if self.o_mu is not None:
            fv = (fv - self.o_mu) / self.o_sd
        e_opt = self._mlp(np.concatenate([e_card, fv], axis=1), "opt_enc")
        owner = np.asarray(owner, np.int64)
        sc = np.einsum("od,od->o", e_opt, q[owner]) + cbias[owner]
        # **クラスの正規化は決定単位**なので、平坦化した配列のまま合成できない。
        # バトルごとに切り出してから _combine_np に渡す。
        out, pos = [], 0
        for b in range(B):
            k = len(opt_cids[b])
            s_b = sc[pos:pos + k]
            if has_cls and opt_clss[b] is not None:
                s_b = _combine_np(s_b, cl[b], opt_clss[b], cl.shape[1], self.head)
            out.append(s_b)
            pos += k
        return out


class TorchBatchPolicy:
    """BatchLSTMPolicy の torch 版（GPUで行列積を回す）。

    PPOではオンポリシー生成が毎イテレーション走るため、生成コストが全体を支配する。
    NN前向きをGPUに逃がすための実装。**インターフェースはnumpy版と同一**にしてあり、
    `gen_selfplay.py --policy-backend torch` で差し替えられる。

    注意: エンジン/JSON/特徴抽出/選択肢構築はバトルごとの逐次Pythonなので速くならない。
    GPU化で効くのは行列積だけであり、同時進行バトル数を上げないと
    カーネル起動のオーバーヘッドに食われる（1.2Mパラメータの小さいLSTMのため）。
    """

    def __init__(self, path: str, n_slots: int, device: str | None = None):
        import torch
        self.torch = torch
        z = np.load(path, allow_pickle=False)
        self.dev = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.w = {k: torch.as_tensor(np.asarray(z[k], np.float32), device=self.dev)
                  for k in z.files if z[k].dtype.kind == "f"}
        keys = z["vocab_keys"].astype(np.int64)
        vals = z["vocab_vals"].astype(np.int64)
        self.vocab = {int(k): int(v) for k, v in zip(keys, vals)}
        # カードID→index を GPU 上の LUT にしておく（Python dict 引きを避ける）
        lo = min(self.vocab)
        self.lut_off = -lo
        lut = np.ones(max(self.vocab) - lo + 1, np.int64)
        for k, v in self.vocab.items():
            lut[k - lo] = v
        self.lut = torch.as_tensor(lut, device=self.dev)
        self.n_layers = sum(1 for k in z.files if k.startswith("lstm.weight_ih_l"))
        self.h_dim = int(z["lstm.weight_hh_l0"].shape[1])
        self.trained_ctx = ({int(c) for c in z["trained_ctx"]}
                            if "trained_ctx" in z.files else None)
        self.n_slots = n_slots
        self._lin_cache: dict[str, list] = {}
        g = (lambda k: self.w[k] if k in self.w else None)
        self.b_mu, self.b_sd = g("board_mu"), g("board_sd")
        self.o_mu, self.o_sd = g("opt_mu"), g("opt_sd")
        self.head = str(z["head"]) if "head" in z.files else "additive"
        self.reset_all()

    def reset_all(self):
        t = self.torch
        self.h = [t.zeros((self.n_slots, self.h_dim), device=self.dev)
                  for _ in range(self.n_layers)]
        self.c = [t.zeros((self.n_slots, self.h_dim), device=self.dev)
                  for _ in range(self.n_layers)]

    def reset(self, slots):
        idx = self.torch.as_tensor(list(slots), device=self.dev, dtype=self.torch.long)
        for k in range(self.n_layers):
            self.h[k][idx] = 0.0
            self.c[k][idx] = 0.0

    def _cidx_t(self, ids):
        t = self.torch
        x = t.as_tensor(ids, device=self.dev, dtype=t.long) + self.lut_off
        x = x.clamp_(0, self.lut.numel() - 1)
        return self.lut[x]

    def _lin_idx(self, prefix: str):
        k = self._lin_cache.get(prefix)
        if k is None:
            k = sorted(int(j.split(".")[1]) for j in self.w
                       if j.startswith(prefix + ".") and j.endswith(".weight"))
            self._lin_cache[prefix] = k
        return k

    def _mlp(self, x, prefix: str, n_layers: int = 2):
        t = self.torch
        for i in self._lin_idx(prefix):
            x = t.addmm(self.w[f"{prefix}.{i}.bias"], x,
                        self.w[f"{prefix}.{i}.weight"].T).relu_()
        return x

    @staticmethod
    def _lstm_cell(t, g, cprev, H):
        i_g = g[:, 0:H].sigmoid()
        f_g = g[:, H:2 * H].sigmoid()
        g_g = g[:, 2 * H:3 * H].tanh()
        o_g = g[:, 3 * H:].sigmoid()
        cnew = f_g * cprev + i_g * g_g
        return o_g * cnew.tanh(), cnew

    def scores(self, slots, boards, logs, prevs, ctxs, opt_cids, opt_vecs, opt_clss):
        t = self.torch
        with t.inference_mode():
            sl = t.as_tensor(np.asarray(slots, np.int64), device=self.dev)
            ct = t.as_tensor(np.asarray(ctxs, np.int64), device=self.dev)
            pv = t.as_tensor(np.asarray(prevs, np.float32), device=self.dev)
            e_ctx = self.w["ctx_emb.weight"][ct]
            e_prev = self.w["prev_card_emb.weight"][self._cidx_t(
                np.asarray(prevs, np.float32)[:, 0].astype(np.int64))]
            bd = t.as_tensor(np.asarray(boards, np.float32), device=self.dev)
            if self.b_mu is not None:
                bd = (bd - self.b_mu) / self.b_sd
            x = t.cat([bd,
                       t.as_tensor(np.asarray(logs, np.float32), device=self.dev),
                       e_ctx, e_prev, pv[:, 1:3]], dim=1)
            z = self._mlp(x, "enc")
            H = self.h_dim
            inp = z
            for k in range(self.n_layers):
                g = (t.addmm(self.w[f"lstm.bias_ih_l{k}"], inp,
                             self.w[f"lstm.weight_ih_l{k}"].T)
                     + t.addmm(self.w[f"lstm.bias_hh_l{k}"], self.h[k][sl],
                               self.w[f"lstm.weight_hh_l{k}"].T))
                hnew, cnew = self._lstm_cell(t, g, self.c[k][sl], H)
                self.c[k][sl] = cnew
                self.h[k][sl] = hnew
                inp = hnew
            h = inp
            q = (t.addmm(self.w["proj.bias"], h, self.w["proj.weight"].T)
                 * self.w["ctx_scale.weight"][ct])
            cbias = self.w["ctx_bias.weight"][ct][:, 0]
            has_cls = "cls_head.weight" in self.w
            if has_cls:
                cl = t.addmm(self.w["cls_head.bias"], h, self.w["cls_head.weight"].T)

            flat_cid, flat_vec, owner = [], [], []
            for b in range(len(slots)):
                flat_cid.extend(opt_cids[b])
                flat_vec.extend(opt_vecs[b])
                owner.extend([b] * len(opt_cids[b]))
            if not flat_cid:
                return [np.zeros(0, np.float32) for _ in slots]
            ow = t.as_tensor(np.asarray(owner, np.int64), device=self.dev)
            e_card = self.w["card_emb.weight"][self._cidx_t(
                np.asarray(flat_cid, np.int64))]
            fv = t.as_tensor(np.asarray(flat_vec, np.float32), device=self.dev)
            if self.o_mu is not None:
                fv = (fv - self.o_mu) / self.o_sd
            e_opt = self._mlp(t.cat([e_card, fv], dim=1), "opt_enc")
            sc = (e_opt * q[ow]).sum(1) + cbias[ow]
            out_np = sc.float().cpu().numpy()
            cl_np = cl.float().cpu().numpy() if has_cls else None
        # クラス合成は決定単位。numpy版と同じ _combine_np を使い、実装を1本に保つ
        res, pos = [], 0
        for b in range(len(slots)):
            k = len(opt_cids[b])
            s_b = out_np[pos:pos + k]
            if has_cls and opt_clss[b] is not None:
                s_b = _combine_np(s_b, cl_np[b], opt_clss[b], cl_np.shape[1], self.head)
            res.append(s_b)
            pos += k
        return res
