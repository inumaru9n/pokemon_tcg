"""EXP-015: カード埋め込み方策ネット（純NumPy、依存はnumpyのみ）。

構成: 選択肢トークン（カードID/対象ID/攻撃ID埋め込み+種別+スカラー）×盤面コンテキスト
→ ポイントワイズMLP → 選択肢ごとのロジット → softmax。
教師（islet方策）の選択をcross-entropyで模倣する。推論はforward()のみ使用。
"""

from __future__ import annotations

import numpy as np

import features_016 as F

D_CARD = 16
D_ATTACK = 8
D_TOKEN = D_CARD + D_CARD + D_ATTACK + F.N_OPT_TYPE + F.N_OPT_SCALAR   # 70
D_CTX = F.N_CONTEXT + D_CARD + D_CARD + F.N_CTX_SCALAR                 # 110
D_IN = D_TOKEN + D_CTX                                                 # 180
H1, H2 = 128, 64
P_MAX = 80  # 1 selectあたりの選択肢数上限（超過分は学習からスキップ）


def init_params(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    def lin(n_in, n_out):
        return (rng.standard_normal((n_in, n_out)) * np.sqrt(2.0 / n_in)).astype(np.float32)
    return {
        "card_emb": (rng.standard_normal((F.N_CARD, D_CARD)) * 0.05).astype(np.float32),
        "attack_emb": (rng.standard_normal((F.N_ATTACK, D_ATTACK)) * 0.05).astype(np.float32),
        "W1": lin(D_IN, H1), "b1": np.zeros(H1, np.float32),
        "W2": lin(H1, H2), "b2": np.zeros(H2, np.float32),
        "w3": lin(H2, 1)[:, 0], "b3": np.float32(0.0),
    }


def save_params(path: str, p: dict) -> None:
    np.savez_compressed(path, **p)


def load_params(path: str) -> dict:
    z = np.load(path)
    return {k: z[k] for k in z.files}


def build_batch(p: dict, ctxs, opt_arrays, pad_to: int | None = None):
    """ctxs: (ctx_id, my_act, op_act, ctx_scalars)のリスト。
    opt_arrays: selectごとの (opt_type, card, target, attack, scalars) 行列のリスト。
    Returns: X (B,P,D_IN), mask (B,P)
    """
    B = len(ctxs)
    P = pad_to or max(len(o) for o in opt_arrays)
    X = np.zeros((B, P, D_IN), np.float32)
    mask = np.zeros((B, P), bool)
    idx = {"card": np.zeros((B, P), np.int32), "target": np.zeros((B, P), np.int32),
           "attack": np.zeros((B, P), np.int32),
           "my_act": np.zeros(B, np.int32), "op_act": np.zeros(B, np.int32)}
    for b, (ctx, opts) in enumerate(zip(ctxs, opt_arrays)):
        ctx_id, my_act, op_act, ctx_sc = ctx
        idx["my_act"][b] = my_act % F.N_CARD
        idx["op_act"][b] = op_act % F.N_CARD
        ctx_vec = np.zeros(D_CTX, np.float32)
        ctx_vec[ctx_id % F.N_CONTEXT] = 1.0
        ctx_vec[F.N_CONTEXT:F.N_CONTEXT + D_CARD] = p["card_emb"][my_act % F.N_CARD]
        ctx_vec[F.N_CONTEXT + D_CARD:F.N_CONTEXT + 2 * D_CARD] = p["card_emb"][op_act % F.N_CARD]
        ctx_vec[F.N_CONTEXT + 2 * D_CARD:] = np.asarray(ctx_sc, np.float32)
        for j, (ot, cid, tid, aid, sc) in enumerate(opts[:P]):
            mask[b, j] = True
            idx["card"][b, j] = cid % F.N_CARD
            idx["target"][b, j] = tid % F.N_CARD
            idx["attack"][b, j] = aid % F.N_ATTACK
            tok = np.zeros(D_TOKEN, np.float32)
            tok[:D_CARD] = p["card_emb"][cid % F.N_CARD]
            tok[D_CARD:2 * D_CARD] = p["card_emb"][tid % F.N_CARD]
            tok[2 * D_CARD:2 * D_CARD + D_ATTACK] = p["attack_emb"][aid % F.N_ATTACK]
            tok[2 * D_CARD + D_ATTACK + (ot % F.N_OPT_TYPE)] = 1.0
            tok[2 * D_CARD + D_ATTACK + F.N_OPT_TYPE:] = np.asarray(sc, np.float32)
            X[b, j, :D_TOKEN] = tok
            X[b, j, D_TOKEN:] = ctx_vec
    return X, mask, idx


def forward(p: dict, X, mask):
    """Returns logits (B,P) と中間値（学習用）"""
    h1 = np.maximum(X @ p["W1"] + p["b1"], 0.0)
    h2 = np.maximum(h1 @ p["W2"] + p["b2"], 0.0)
    logits = h2 @ p["w3"] + p["b3"]
    logits = np.where(mask, logits, -1e9)
    return logits, (X, h1, h2)


def loss_and_grads(p: dict, X, mask, idx, labels, sample_w=None):
    """sample_w=None でBC（CE損失）。sample_w=advantage でREINFORCE勾配
    （∇logπ×A はCE勾配のサンプル重み付けと同形）。"""
    B = X.shape[0]
    logits, (X_, h1, h2) = forward(p, X, mask)
    logits = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(logits) * mask
    probs = e / e.sum(axis=1, keepdims=True)
    loss = -np.log(probs[np.arange(B), labels] + 1e-9).mean()
    acc = float((logits.argmax(axis=1) == labels).mean())

    dlogit = probs.copy()
    dlogit[np.arange(B), labels] -= 1.0
    if sample_w is not None:
        dlogit = dlogit * np.asarray(sample_w, np.float32)[:, None]
    dlogit = (dlogit * mask) / B                              # (B,P)

    g = {}
    g["w3"] = np.einsum("bp,bph->h", dlogit, h2)
    g["b3"] = dlogit.sum()
    dh2 = dlogit[..., None] * p["w3"] * (h2 > 0)              # (B,P,H2)
    g["W2"] = np.einsum("bph,bpk->hk", h1, dh2)
    g["b2"] = dh2.sum(axis=(0, 1))
    dh1 = (dh2 @ p["W2"].T) * (h1 > 0)                        # (B,P,H1)
    g["W1"] = np.einsum("bpd,bph->dh", X_, dh1)
    g["b1"] = dh1.sum(axis=(0, 1))
    dX = dh1 @ p["W1"].T                                      # (B,P,D_IN)

    # 埋め込み勾配（token側 + context側）
    g["card_emb"] = np.zeros_like(p["card_emb"])
    g["attack_emb"] = np.zeros_like(p["attack_emb"])
    m = mask[..., None]
    np.add.at(g["card_emb"], idx["card"].ravel(),
              (dX[:, :, :D_CARD] * m).reshape(-1, D_CARD))
    np.add.at(g["card_emb"], idx["target"].ravel(),
              (dX[:, :, D_CARD:2 * D_CARD] * m).reshape(-1, D_CARD))
    np.add.at(g["attack_emb"], idx["attack"].ravel(),
              (dX[:, :, 2 * D_CARD:2 * D_CARD + D_ATTACK] * m).reshape(-1, D_ATTACK))
    base = D_TOKEN + F.N_CONTEXT
    d_my = (dX[:, :, base:base + D_CARD] * m).sum(axis=1)
    d_op = (dX[:, :, base + D_CARD:base + 2 * D_CARD] * m).sum(axis=1)
    np.add.at(g["card_emb"], idx["my_act"], d_my)
    np.add.at(g["card_emb"], idx["op_act"], d_op)
    return loss, acc, g


class Adam:
    def __init__(self, params: dict, lr: float = 1e-3):
        self.lr = lr
        self.t = 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params: dict, grads: dict) -> None:
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for k in params:
            gk = grads[k]
            self.m[k] = b1 * self.m[k] + (1 - b1) * gk
            self.v[k] = b2 * self.v[k] + (1 - b2) * gk * gk
            mhat = self.m[k] / (1 - b1 ** self.t)
            vhat = self.v[k] / (1 - b2 ** self.t)
            params[k] = (params[k] - self.lr * mhat / (np.sqrt(vhat) + eps)).astype(np.float32)
