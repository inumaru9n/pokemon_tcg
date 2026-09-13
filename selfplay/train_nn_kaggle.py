"""EXP-094 Step3: 全選択LSTM方策の教師あり学習（Kaggle Notebook / GPU で実行）。

AlphaStar型の構造を我々のエンジンに合わせたもの:
  - 自己回帰はエンジンが提供（MAIN→サーチ先→捨て札が別々の select() として順に来る）
  - 共有バックボーン + LSTM(記憶) + pointer型スコアリング（内積）+ ctxアダプタ
  - 全コンテキストを1つのモデルでカバー（低頻度ctxは共有パラメタで融通）

入力データ = selfplay/pack_dataset.py が作る npz。
出力 = model_094.npz（純numpy推論用にすべての重みをフラットに保存）。

Kaggle Notebookでの使い方:
  1. データセット（train.npz）をNotebookのInputに追加
  2. このファイルの中身を貼り付けて実行（GPU推奨）
  3. /kaggle/working/model_094.npz をダウンロード

ローカル検証（torchがある環境）:
  python selfplay/train_nn_kaggle.py --npz data/nn/train.npz --out model_094.npz --epochs 2
"""

from __future__ import annotations

import argparse
import collections
import os

import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- カードID語彙 ----
# opt_cid は生のカードID（1000超もある）。埋め込みのためインデックスに圧縮する。
# 標準化後の値をこの範囲に丸める（分布外の相手で σ が爆発するのを防ぐ）
NORM_CLIP = 10.0   # 既定。--norm-clip 0 で無効化できる（104と条件を揃える用）
_NCLIP = NORM_CLIP

PAD_CID = -2
UNK_CID = -1


def combine_scores(pointer, cls_logit, opt_cls, mask, n_cls, mode="additive"):
    """pointer と クラスlogit を合成して、選択肢ごとの最終スコアを返す（torch版）。

    どの形も **クラス内分布は同一**（softmax_{j∈c}(pointer)）で、違うのは
    `P(クラス)` の logit だけ:

      additive   : a_c + logsumexp_{j∈c}(s_j)   選択肢の質を見るが**個数でも膨らむ**（従来）
      factorized : a_c                          個数に無関係だが**選択肢を全く見ない**
      maxpool    : a_c + max_{j∈c}(s_j)         個数に無関係で質は見る

    実測（EXP-097）: MAIN決定内のクラスは **77.3%が選択肢1個**。純factorizedだと
    そこで P(i|c)=1 となり pointer が効かなくなる＝選択肢側特徴が死ぬ。

    additive は非正規化スコア、他2つは正規化済み log P を返すが、呼び出し側が
    `log_softmax` を掛けても log P は不変（logsumexp=0）なので、下流は分岐しなくてよい。

    非MAIN（opt_cls<0）の決定は pointer をそのまま返す。
    """
    NEG = -1e9
    valid = mask & (opt_cls >= 0) & (opt_cls < n_cls)
    ci = opt_cls.clamp(min=0, max=n_cls - 1)
    a = torch.gather(cls_logit, 2, ci)                 # (B,T,O) 各選択肢のクラスlogit
    if mode == "additive":
        return torch.where(valid, pointer + a, pointer)

    B, T, _ = pointer.shape
    s = pointer.masked_fill(~valid, float("-inf"))
    # クラスごとの max（無効な選択肢は -inf なので効かない）
    m_c = torch.full((B, T, n_cls), float("-inf"), device=s.device, dtype=s.dtype)
    m_c = m_c.scatter_reduce(2, ci, s, reduce="amax", include_self=True)
    # **不在クラス（選択肢が1つも属さない）の m_c は -inf**。非MAINの決定では
    # opt_cls=-1 が ci=0 にクランプされるため、そこで exp(-inf −(−inf))=nan が出る。
    # forward は torch.where で潰せるが、**where は選ばれなかった枝のnanも逆伝播で
    # 伝播させる**ので、指数の中に -inf を入れないよう先に 0 へ丸める。
    m_safe = torch.nan_to_num(m_c, neginf=0.0)
    m_i = torch.gather(m_safe, 2, ci)                  # (B,T,O)
    e = torch.where(valid, torch.exp(s - m_i), torch.zeros_like(s))
    sum_c = torch.zeros((B, T, n_cls), device=s.device, dtype=s.dtype)
    sum_c = sum_c.scatter_add(2, ci, e)
    present = sum_c > 0
    lse_c = torch.where(present, m_safe + torch.log(sum_c.clamp(min=1e-30)),
                        torch.full_like(sum_c, NEG))
    pool = lse_c if mode == "logsumexp" else torch.where(
        present, m_safe, torch.full_like(m_safe, NEG))  # maxpool は max を使う
    cl = cls_logit if mode == "factorized" else cls_logit + pool
    logp_c = torch.log_softmax(cl.masked_fill(~present, NEG), dim=-1)
    # クラス内: log softmax_{j∈c}(s_j) = s_i - LSE_c
    logp_i = s - torch.gather(lse_c, 2, ci)
    out = torch.gather(logp_c, 2, ci) + logp_i
    return torch.where(valid, out, pointer)


class PolicyNet(nn.Module):
    """共有バックボーン + LSTM + pointer型スコアリング。"""

    def __init__(self, n_board, n_logs, n_optdim, n_cards, n_ctx,
                 d_card=32, d_opt=64, d_model=256, n_layers=2, n_cls=0,
                 dropout=0.0, head="additive", v_detach=False, v_hidden=0,
                 opt_attn=False, prev_opp=False, layer_norm=False,
                 target_emb=False, post_mlp=0):
        super().__init__()
        # MAINは「行動タイプ×対象」が1回のselectに圧縮されており、エンジンは階層を
        # 提供しない（サーチ先や捨て札は別selectとして分解されるのとは対照的）。
        # そこでAlphaStarのaction_type→target自己回帰ヘッドに相当する階層を
        # ここだけ自前で入れる: score_i = class_logit[cls(i)] + pointer_score_i
        self.n_cls = n_cls
        self.head = head
        # 補助: 勝敗予測（tanh で ±1 に収める）。v_w=0 なら勾配が流れないので無害
        self.v_head = (nn.Linear(d_model, 1) if not v_hidden else nn.Sequential(
            nn.Linear(d_model, v_hidden), nn.ReLU(), nn.Linear(v_hidden, 1)))
        # **True なら value head はトランクに勾配を返さない**。方策を汚さずに V だけ得る。
        # 共有の利点（マルチタスク正則化）は失うが、AWR と価値誘導探索には V があれば足りる。
        self.v_detach = v_detach
        # **joint では確保しない**。使わない重みを持たせると weight decay が掛かり、
        # npz にも載る（提出物が太る）。head で構造そのものを切り替える。
        self._two_tower = head not in ("joint",)
        if n_cls and head != "joint":
            self.cls_head = nn.Linear(d_model, n_cls)
        self.card_emb = nn.Embedding(n_cards, d_card, padding_idx=0)
        self.ctx_emb = nn.Embedding(n_ctx, 16)
        # 前の選択（card_id, opt_type, area）→ 埋め込み
        self.prev_card_emb = nn.Embedding(n_cards, d_card, padding_idx=0)

        # **Dropout は Sequential の途中に入れる**。推論(numpy)側は線形層の番号を
        # npzのキーから発見するので、ここで番号が 0,3 にずれても壊れない。
        # **相手の直前の選択**（EXP-133）: card埋め込み(共有) + 数値2 を連結する。
        # prev が6要素の npz でのみ有効。3要素なら従来と厳密に同じ次元になる。
        self.prev_opp = bool(prev_opp)
        d_in = n_board + n_logs + 16 + d_card + 2 + ((d_card + 2) if prev_opp else 0)
        # **内部正規化**（EXP-134）。EXP-121 は layers 3 が明確に悪化したことを
        # 「ネットワークに内部正規化が一切ないため深くできない」と診断したが、
        # **その対処は一度も試していなかった**。並びは Linear→ReLU→LayerNorm→Dropout。
        # numpy推論側は重みの次元（Linear=2次元 / LayerNorm=1次元）で分岐して
        # 保存順に適用するので、この並びなら torch と同じ計算になる。
        self.layer_norm = bool(layer_norm)
        if layer_norm:
            self.enc = nn.Sequential(
                nn.Linear(d_in, d_model), nn.ReLU(), nn.LayerNorm(d_model),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model), nn.ReLU(), nn.LayerNorm(d_model),
                nn.Dropout(dropout))
        else:
            self.enc = nn.Sequential(
                nn.Linear(d_in, d_model), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(d_model, d_model), nn.ReLU(), nn.Dropout(dropout))
        # **n_layers=0 で再帰を完全に外す**（EXP-140）。深さの実測は
        # L1 0.8718 > L2 0.8703 > L3 0.8696 > L4 0.8682 と**単調**だったが、
        # **L0 は一度も測っていない**。LSTM は全パラメータの80%（1,052,672）を占める。
        # 「もう Rare Candy を使い切った」等の履歴は pool_*/dis_* が既に持っているので、
        # 再帰が本当に要るかは自明でない（DeNA/HandyRL も**ステートレス**Transformer）。
        #
        # **post_mlp** は「再帰だけ外し、深さは保つ」変種。
        # LSTMの除去は *再帰* と *深さ2層分* を同時に消すので、
        # どちらが効いたのか分離できない。post_mlp=k で d_model の MLP を k 層挟む。
        self.n_layers = int(n_layers)
        self.lstm = (nn.LSTM(d_model, d_model, num_layers=n_layers,
                             batch_first=True,
                             dropout=dropout if n_layers > 1 else 0.0)
                     if n_layers > 0 else None)
        self.post = None
        if n_layers == 0 and post_mlp > 0:
            _l = []
            for _ in range(int(post_mlp)):
                _l += [nn.Linear(d_model, d_model), nn.ReLU()]
                if layer_norm:
                    _l += [nn.LayerNorm(d_model)]
                _l += [nn.Dropout(dropout)]
            self.post = nn.Sequential(*_l)
        # LSTM出力にも正規化を入れる（深い段の勾配を通しやすくする）
        self.h_norm = nn.LayerNorm(d_model) if layer_norm else None

        # オプション埋め込み: カード埋め込み + 数値特徴
        # **貼り先/進化先の正体**（EXP-139）。opt_vec が対象について持つのは
        # inPlayArea/inPlayIndex（位置）と 残HP比/エネ数 の4次元だけで、
        # **種族は入っていない**。MAINでは cls_head の ATTACH_BUCKET が代役をするが
        # バケットが粗く（Abra 741 と Kadabra 742 が同一クラス 'abra_p'）、
        # 実測で **MAIN決定の22.1%** が「同一クラスに2個以上の選択肢」＝クラス頭が
        # 同点で、位置番号とエネ数だけで割っている状態になる。
        # ctx37 EVOLVE は非MAINなので cls_head 自体が効かない（11,836選択肢）。
        # card_emb を**共有**する（同じカードなので1行あたりの勾配が増える。
        # EXP-133 で prev_opp を prev_card_emb 共有にしたのと同じ判断）。
        self.target_emb = bool(target_emb)
        self.opt_enc = (nn.Sequential(
            nn.Linear(d_card + n_optdim + (d_card if target_emb else 0), d_opt),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_opt, d_opt), nn.ReLU(), nn.Dropout(dropout))
            if head not in ("joint", "joint_cls") else None)
        # ctxアダプタ: 文脈ごとの双線形行列（低ランク: d_opt × d_model を ctx別バイアスで代用）
        # **選択肢間 self-attention**（EXP-112 軸B）。
        # 従来は score_i = dot(e_opt_i, q) で**各選択肢を独立に採点**していた。
        # カードゲームでは「他に何が打てるか」で1枚の価値が変わる
        # （同じ Rare Candy でも、手札に Alakazam があるかで意味が違う）。
        # pointer network に集合上の attention を入れるのは標準的な拡張。
        # 単ヘッド1層・残差にしたのは、numpy 推論を素直に書けて検証しやすいため。
        # opt_attn=False なら重みを持たず、従来と**厳密に**同じ計算になる。
        self.opt_attn = opt_attn
        if opt_attn:
            self.a_q = nn.Linear(d_opt, d_opt)
            self.a_k = nn.Linear(d_opt, d_opt)
            self.a_v = nn.Linear(d_opt, d_opt)
            self.a_o = nn.Linear(d_opt, d_opt)
            nn.init.zeros_(self.a_o.weight)      # **恒等から始める**（残差が最初は素通り）
            nn.init.zeros_(self.a_o.bias)
        # two-tower の合流部。**joint 系では作らない**（内積を使わないため）
        if head not in ("joint", "joint_cls"):
            self.proj = nn.Linear(d_model, d_opt)          # 共有
            self.ctx_scale = nn.Embedding(n_ctx, d_opt)    # 文脈別の重み付け
            self.ctx_bias = nn.Embedding(n_ctx, 1)
            nn.init.ones_(self.ctx_scale.weight)
            nn.init.zeros_(self.ctx_bias.weight)

        # ==== joint（cross-encoder）ヘッド（EXP-142）====
        # two-tower は score = dot(q(h), e(o)) なので、**状態と選択肢の相互作用が
        # d_opt=64 本の軸に制限される**（内積は双線形形式）。検索の分野では
        # bi-encoder（速い/表現力が低い）と cross-encoder（遅い/表現力が高い）と
        # 呼び分けられ、two-tower を選ぶ動機は**アイテムを事前計算してインデックス化
        # できること**にある。**我々にはその動機が無い**（選択肢は5〜30個で毎回の
        # 盤面から作られるので再利用が効かない）＝対価だけ払っている。
        #
        # 同時に、two-tower を採るために生じた負債をまとめて消す:
        #   ctx_bias   出力に一切影響しない死重（softmax/argmax はシフト不変）
        #   ctx_scale  proj に対して 64 パラメータ分のゲージ自由度がある
        #   cls_head   65クラスが**デッキ固有**（CLS_PLAY/CLS_EVO/CLS_AB/CLS_ATK/
        #              ATTACH_BUCKET を deck.csv から生成）。しかも ATTACH_BUCKET は
        #              Abra(741) と Kadabra(742) を同一クラスに潰す
        #   式が2種類  cls_head は MAIN のみ（opt_cls>=0）。非MAINは内積だけ
        #   生の序数   opt.type / opt.area を enum の整数値のまま MLP に入れている
        #
        # **第1層を因数分解する**: J([h,o]) の第1層は W_h·h + W_o·o なので、
        # h 側を (B,T,128) で1回、o 側を (B,T,O,128) で計算して broadcast 加算する。
        # (B,T,O,371) を実体化しないので、メモリは e_opt と同オーダーに収まる。
        if head in ("joint", "joint_cls"):
            self.type_emb = nn.Embedding(20, 16)   # OptionType 0..16 と NULL(-1)
            self.area_emb = nn.Embedding(16, 8)    # AreaType 1..12 と NULL(-1)
            _od = (d_card + 16 + 8 + (d_card if target_emb else 0)
                   + (n_optdim - 2))               # type/area は埋め込みに移すので除く
            self.j_h = nn.Linear(d_model, 128)
            self.j_o = nn.Linear(_od, 128, bias=False)
            self.j_out = nn.Linear(128, 1)
            self.j_drop = nn.Dropout(dropout)

    def forward(self, board, logs, prev_cidx, prev_num, ctx, opt_cidx, opt_vec,
                mask, state=None, opt_cls=None, prevo_cidx=None, prevo_num=None,
                opt_tcid=None, opt_type=None, opt_area=None):
        """board (B,T,F) / opt_cidx (B,T,O) / mask (B,T,O) → scores (B,T,O)"""
        B, T, _ = board.shape
        e_ctx = self.ctx_emb(ctx)                    # (B,T,16)
        e_prev = self.prev_card_emb(prev_cidx)       # (B,T,d_card)
        _p = [board, logs, e_ctx, e_prev, prev_num]
        if self.prev_opp:
            _p += [self.prev_card_emb(prevo_cidx), prevo_num]
        x = torch.cat(_p, dim=-1)
        z = self.enc(x)                              # (B,T,d_model)
        if self.lstm is not None:
            h, state = self.lstm(z, state)           # (B,T,d_model)
        else:                                        # 再帰なし＝決定ごとに独立
            h, state = (self.post(z) if self.post is not None else z), None
        if self.h_norm is not None:
            h = self.h_norm(h)

        e_card = self.card_emb(opt_cidx)             # (B,T,O,d_card)

        # ==== joint ヘッド: 内積も cls_head も ctx_scale も通さない単一の式 ====
        if self.head in ("joint", "joint_cls"):
            # 並びは [自カード, 行動タイプ, エリア, (対象カード), 数値]。
            # **推論側(nn_*.py)と厳密に一致させること**
            _p = [e_card, self.type_emb(opt_type), self.area_emb(opt_area)]
            if self.target_emb:
                _p.append(self.card_emb(opt_tcid))
            _p.append(opt_vec[..., 2:])              # type/area は埋め込みへ移した
            o = torch.cat(_p, dim=-1)                # (B,T,O,_od)
            # **第1層を因数分解**（(B,T,O,371) を作らない）
            z = torch.relu(self.j_h(h).unsqueeze(2) + self.j_o(o))
            scores = self.j_out(self.j_drop(z)).squeeze(-1)      # (B,T,O)
            cls_logits = None
            if self.head == "joint_cls" and self.n_cls and opt_cls is not None:
                cls_logits = self.cls_head(h)
                _v = mask & (opt_cls >= 0) & (opt_cls < self.n_cls)
                _a = torch.gather(cls_logits, 2, opt_cls.clamp(0, self.n_cls - 1))
                scores = torch.where(_v, scores + _a, scores)
            scores = scores.masked_fill(~mask, -1e9)
            v = torch.tanh(self.v_head(h.detach() if self.v_detach else h)).squeeze(-1)
            return scores, state, cls_logits, v

        # 並びは [自カード, **対象カード**, 数値]。推論側(nn_*.py)と厳密に一致させる
        _oin = ([e_card, self.card_emb(opt_tcid), opt_vec] if self.target_emb
                else [e_card, opt_vec])
        e_opt = self.opt_enc(torch.cat(_oin, dim=-1))  # (B,T,O,d_opt)
        if self.opt_attn:
            # 選択肢集合の中で互いを見る。**マスク外（padding）は参照しない**
            aq, ak, av = self.a_q(e_opt), self.a_k(e_opt), self.a_v(e_opt)
            d = aq.shape[-1] ** 0.5
            att = torch.matmul(aq, ak.transpose(-1, -2)) / d          # (B,T,O,O)
            att = att.masked_fill(~mask.unsqueeze(2), float("-inf"))
            # 全部マスクの行（実オプション0個）で softmax が nan になるのを防ぐ
            att = torch.where(mask.unsqueeze(2).any(-1, keepdim=True),
                              att, torch.zeros_like(att))
            w_att = torch.softmax(att, dim=-1)
            e_opt = e_opt + self.a_o(torch.matmul(w_att, av))          # 残差
        q = self.proj(h) * self.ctx_scale(ctx)       # (B,T,d_opt)
        scores = (e_opt * q.unsqueeze(2)).sum(-1) + self.ctx_bias(ctx)  # (B,T,O)
        cls_logits = None
        if self.n_cls and opt_cls is not None:
            cls_logits = self.cls_head(h)                      # (B,T,n_cls)
            scores = combine_scores(scores, cls_logits, opt_cls, mask,
                                    self.n_cls, self.head)
        scores = scores.masked_fill(~mask, -1e9)
        # (B,T) の勝敗予測。方策側の勾配は cls_logits と同じくトランク経由で共有
        v = torch.tanh(self.v_head(h.detach() if self.v_detach else h)).squeeze(-1)
        return scores, state, cls_logits, v


D_CARD_FOR_VOCAB = 32          # 埋め込み1行の次元（②の閾値 k×d_card に使う）


def _card_names():
    """cardId → 名前。cg が読めない環境（Kaggle）でも落ちないようにする。"""
    try:
        from cg.api import all_card_data
        return {c.cardId: c.name for c in all_card_data()}
    except Exception:      # noqa: BLE001
        return {}


def build_vocab(opt_cid, prev):
    ids = set(np.unique(opt_cid).tolist()) | set(np.unique(prev[:, 0]).tolist())
    if prev.shape[1] >= 6:
        # **相手の直前カード**（EXP-133）。無条件に足すと +99語 になり、その大半は
        # 基本エネ（色は os*_et* に既出＝冗長）と数百回未満の稀少札で、
        # **教師データに乏しい札を足すと他の学習を薄める純損失**になる（EXP-120: −2.29pt）。
        # 埋め込み語彙の基準を2つ置く:
        #   ① 非冗長性   identity が opt_vec の数値特徴を超える情報を持つか
        #                → META_CARDS と同じ "Basic " 除外
        #   ② 学習可能性 出現数 >= k × **d_card**（=320, k=10）
        #                board列の k×d_model=2,560 とは**別の閾値**。
        #                埋め込み1行は d_card=32 パラメータであって256ではない。
        _po = prev[:, 3].astype(np.int64)
        _cnt = collections.Counter(int(x) for x in _po if x >= 0)
        _names = _card_names()
        _add = {c for c, n in _cnt.items()
                if n >= 10 * D_CARD_FOR_VOCAB
                and not str(_names.get(c, "")).startswith("Basic ")}
        print(f"相手カード語彙: 候補{len(_cnt)}種 → **{len(_add & (set(_cnt) - ids))}種を追加**"
              f"（①非冗長 ②出現>={10*D_CARD_FOR_VOCAB}）")
        ids |= _add
    ids.discard(PAD_CID)
    ids = sorted(int(i) for i in ids)
    # 0 = padding, 1 = unknown(-1), 以降が実カード
    vocab = {PAD_CID: 0, UNK_CID: 1}
    for i in ids:
        if i not in vocab:
            vocab[i] = len(vocab)
    return vocab


def to_seq(data, vocab, max_T=128):
    """決定の平坦配列 → 系列テンソル群。戻り値は (chunks, game_of_chunk)。

    長いゲームはmax_Tで分割するが、**同一ゲームのchunkがtrain/valにまたがると
    リークする**ため、どのゲーム由来かを返して分割時にゲーム単位でまとめる。
    """
    gidx = data["gidx"]
    seqs, owner = [], []
    for g in np.unique(gidx):
        idx = np.where(gidx == g)[0]
        for s in range(0, len(idx), max_T):
            seqs.append(idx[s:s + max_T])
            owner.append(int(g))
    return seqs, owner


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", default="/kaggle/input/ptcg-yushin-nn/train.npz")
    ap.add_argument("--out", default="/kaggle/working/model_094.npz")
    ap.add_argument("--epochs", type=int, default=12)  # Kaggleの12h制限に対する安全側
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    # **選択肢側の幅**。d_model を上げても d_opt は64のままだったので、
    # pointer採点の帯域だけが未検証だった（EXP-117の監査で判明）。
    # **±10σクリップの無効化**。104はクリップ無しで学習されているので、
    # 「データだけ増やす」等の統制実験では 0 にして条件を揃える。
    ap.add_argument("--norm-clip", type=float, default=10.0, dest="norm_clip")
    ap.add_argument("--d-opt", type=int, default=64, dest="d_opt")
    ap.add_argument("--d-card", type=int, default=32, dest="d_card")
    ap.add_argument("--val-frac", type=float, default=0.1)
    # **1試合=1系列にする**（最長240決定、max_game_len=300で足切り済み）。
    # 128だと2.1%の試合が分割され、その先頭がLSTM状態ゼロで学習される＝推論と食い違う。
    ap.add_argument("--max-T", type=int, default=300)
    ap.add_argument("--cosine", action="store_true", help="コサイン学習率減衰")
    ap.add_argument("--patience", type=int, default=15, help="早期終了")
    # **optimizer step 数を条件間で揃える**（EXP-114）。混合比を変えるとデータ量が
    # 大きく変わり、エポック固定だと更新量が桁で違ってしまう（実測: 元データ4,254系列で
    # 133step/epoch に対し、40%混合は207系列で7step/epoch＝1/19）。
    # 「混合比の効果」でなく「学習量の差」を測ってしまうのを防ぐ。
    # 0 なら従来どおりエポック固定。
    ap.add_argument("--stop-metric", default="acc", choices=("acc", "loss"),
                    dest="stop_metric",
                    help="早期終了/ベスト保存の基準。SILでは loss（accは確率変化に鈍感）")
    ap.add_argument("--save-final", action="store_true", dest="save_final",
                    help="最終step時点も別ファイルで保存する（step数を揃えた比較用）")
    ap.add_argument("--max-steps", type=int, default=0, dest="max_steps",
                    help="総optimizer step数。データ量からエポック数を逆算する")
    ap.add_argument("--seed", type=int, default=0, help="重み初期化/バッチ順のシード")
    # **過学習対策**（実測: 訓練0.879 vs 検証0.754 の12.5pt差）。特徴を増やすと悪化するので
    # 特徴拡張とセットで入れる。
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--wd", type=float, default=0.0, help="AdamWの重み減衰")
    # **特徴の標準化**。従来は生の値がそのまま nn.Linear に入っており、0/1フラグと
    # my_best_dmg(最大540)・act_hp(最大340) が同じ層を共有していた。
    ap.add_argument("--no-norm", action="store_true", help="標準化を無効にする")
    # MAIN階層の合成形（EXP-098）。詳細は combine_scores のdocstring
    ap.add_argument("--awr-beta", type=float, default=0.0,
                    help="AWRの温度β。0で無効（＝素の模倣）。小さいほど強く重み付ける")
    ap.add_argument("--awr-lambda", type=float, default=0.95, help="GAEのλ")
    ap.add_argument("--awr-wmax", type=float, default=20.0, help="重みの上限")
    ap.add_argument("--awr-warmup", type=int, default=10,
                    help="このエポック数は素の模倣で学習してから AWR を有効化する"
                         "（V が育つ前の advantage は雑音）")
    ap.add_argument("--init-from", default=None,
                    help="既存モデルのnpzから方策の重みを読み込む")
    ap.add_argument("--freeze-policy", action="store_true",
                    help="v_head 以外を凍結する（方策を1ステップも動かさない）")
    ap.add_argument("--v-hidden", type=int, default=0,
                    help="value head の隠れ層サイズ。0なら線形1枚")
    ap.add_argument("--v-detach", action="store_true",
                    help="value head からトランクへ勾配を返さない（方策を汚さない）")
    ap.add_argument("--v-weight", type=float, default=0.0,
                    help="value head（勝敗予測）の補助損失の重み。0で無効")
    ap.add_argument("--head", default="additive",  # additive/joint/joint_cls ほか
                    choices=["additive", "factorized", "maxpool", "logsumexp",
                             "joint", "joint_cls"])
    ap.add_argument("--max-game-len", type=int, default=300,
                    help="この決定数を超えるゲームを除外（ループ/膠着の外れ値）")
    ap.add_argument("--sweep", action="store_true",
                    help="複数構成を順に学習して比較（Kaggle 1実行で完結させる）")
    ap.add_argument("--opt-attn", action="store_true", dest="opt_attn",
                    help="選択肢間 self-attention を有効にする（EXP-112 軸B）")
    ap.add_argument("--focal-gamma", type=float, default=0.0, dest="focal_gamma",
                    help="focal loss の γ。0で従来と厳密一致（EXP-111 軸C）")
    ap.add_argument("--no-sweep", action="store_true")
    ap.add_argument("--post-mlp", type=int, default=0, dest="post_mlp",
                    help="--layers 0 のとき、LSTMの代わりに挟む d_model のMLP層数")
    ap.add_argument("--eval-only", action="store_true", dest="eval_only",
                    help="学習せず --init-from の重みで保持データの指標だけ出す")
    ap.add_argument("--no-target-emb", action="store_true", dest="target_emb_off",
                    help="opt_tcid があっても対象カード埋め込みを使わない（EXP-139のベース）")
    args = ap.parse_args()
    global _NCLIP
    _NCLIP = float(getattr(args, 'norm_clip', 10.0))
    # Kaggle Notebookは引数なしで実行されるため、そこではスイープを既定にする
    if os.path.isdir("/kaggle/input") and not args.no_sweep:
        args.sweep = True

    if args.sweep:
        # EXP-094は12epochで打ち切っており（val曲線はまだ上昇中）過少学習だった。
        # 容量とエポック数をスケールさせて到達点を測る。
        # **同一構成×3シード**。版間の差が学習ラン間のばらつきに埋もれていないかを
        # 測るため（d384は劣ることが確定済みなので外した）。
        # **EXP-097 のスイープ**: 特徴を158→384次元に増やしたので、
        #   (a) 標準化が効くか、(b) 正則化がどの強さで最良か
        # を切り分けて測る。実測の過学習は 訓練0.879 vs 検証0.754 の12.5pt差。
        # base は標準化なし・正則化なし＝従来と同じ学習条件（特徴だけが違う）。
        # **EXP-098 のスイープ**: MAIN階層の合成形の比較（EXP-097 で dropout0.2 を採用済み）。
        # 同一シード×2本で、階層差が学習ラン間のばらつきに埋もれていないかも見る。
        # **EXP-099 のスイープ**: 特徴が384→768次元になったので正則化の最適点を測り直す。
        # EXP-098 で階層の合成形は additive を維持と決定済み（差はノイズ床以下）。
        # d20 は 102（プールv4 78.74%）と同一設定なので、**特徴の変化だけを切り出せる**。
        # **EXP-100 のスイープ**: 104(506次元)。d20 は 102(384次元, プールv4 78.74%) と
        # 完全に同一設定なので、**特徴の差だけを切り出せる**アンカーになる。
        # 103(768次元)で val 横ばい・プール -4.8pt だったため、d_model の増減も1本ずつ見る
        # （ただし訓練精度は既に102より高く、容量不足の兆候は無い＝期待は薄い）。
        # **EXP-100 のスイープ**: 105(384次元) は 102 と盤面キーが完全一致で、
        # **違いは打点計算の中身だけ**（弱点×2 / 効果無効化 / チェックアップ）。
        # 102 と同一設定で4シード回し、次元増加と正しさを分離する。
        # 注: logs 48→53、選択肢 28→29 の6次元だけは統制できていない（1.5%）。
        # **EXP-101 のスイープ**: AWR（勝敗ラベルの活用）。105(384次元・打点99.0%)がベース。
        # β=0（AWR無効）が105そのもの＝アンカー。βが小さいほど強く重み付ける。
        # **val一致率は下がりうる**（教師の再現が目的ではない）。合否は直接対戦とプールで。
        # **EXP-101 v2**: v_weight=1.0 は補助タスクが強すぎて val を −1.7pt 落とした
        # （方策CEが決定あたり0.4-0.5、valueのMSEが0.5-1.0で同等の重みになるため）。
        # 重みを振り直し、**stop-gradient 版**（value headがトランクを汚さない）も測る。
        # **EXP-104**: 104（提出済み・プールv5首位）の方策を**凍結**して value head だけを学習。
        # 価値誘導探索の葉評価に使う V を、方策と同じトランク h から読み出す。
        # 隠れ層サイズと重みを振り、value AUC で選ぶ（方策は1ステップも動かないので
        # val一致率は全構成で同じになる）。
        # **EXP-110 のスイープ**: 素の模倣学習に戻す。用途が3つあるので構成は共通:
        #   (a) 主力候補の対面別教師混合（Yushin 非Spidops + Majkel Spidops、506次元）
        #   (b) プール枠 Mega Lopunny ex（Majkel 5c1c56479b、465次元、重み9.7%）
        #   (c) プール枠 Teal Mask Ogerpon ex（Majkel 69a3d48d59、420次元、重み7.2%）
        # 設定は EXP-097 で確定した最良（標準化あり + dropout0.2 + wd1e-4 + additive）。
        # **学習シードを3本振る**（EXP-094 v9: 同一構成でも val の sd が 1.81pt あるので
        # 単発の差を改善と読んではいけない。最良シードを採用しつつ振れ幅も記録する）。
        # value head と AWR と init-from は**全て無効**（EXP-101/104でいずれも利得なし）。
        # **EXP-111 のスイープ**: 軸A（相手の隠れ情報推論、563次元）と
        # 軸C（focal loss）を同時に測る。sl_s* が軸Aのみ、fc* が A+C。
        # **EXP-112 のスイープ**: 軸B（選択肢間 self-attention）。
        # base_s* がアンカー（attnなし＝133と同一構成）、at_s* が attn あり。
        # a_o を零初期化しているので学習開始時点は厳密にアンカーと一致する。
        # **EXP-114 のスイープ**: reward-conditioned self-imitation（SIL）。
        # 117の本番リプレイの**勝ち試合だけ**を元データに混ぜ、117を初期値に継続学習する。
        # argmax一致は0.9995だが教師手の確率は0.849（CE 0.2157）＝**勾配は十分にある**。
        # 勝ち試合だけを選ぶので、勝ちと相関する行動の確率が選択的に上がる。
        # 混合比は npz 側（--new-ratio）で作り、ここでは step数を揃えて2シード回す。
        # **117の重みを初期値にする**。Kaggle側では入力データセットに同梱した
        # model_117.npz を探す（_train_one が /kaggle/input を再帰検索する実装）。
        # **EXP-116 のスイープ**: 勝ち試合だけで学習すると強くなるか。
        # 104 と同一条件（スクラッチ・全設定同じ）で、データだけが違う。
        #   w104_win  = Yushinの勝ち試合のみ 183,934決定
        #   w104_ctrl = 全体からランダム間引き 183,988決定（**データ量を揃えた対照**）
        # 2シードずつ回して、シード幅（val sd 約1.8pt）と区別できるかを見る。
        # **EXP-123 のスイープ**: 165 = 162から**軸A（相手の隠れ情報推論）47特徴を除外**。
        # 軸Aは EXP-113 で「プール −0.12/−0.23pt・対Lopunny −3.2pt」と**棄却済み**なのに、
        # tools/featgen_core.py にコードが残っていたため 156〜164 の全特徴器へ自動混入した。
        # `oppbelief` グループを切って既定から外し、588→**541次元**にした。
        # 教師・期間・デッキは162と完全に同一（Yushin+AlphaStarmie / 156952a871 /
        # 07-17〜08-07 / 5,309試合）なので、**軸Aの寄与だけを分離できる**。
        # **EXP-122 のスイープ**: 主力 104 を最新手法で再構築（162/163）。
        # 教師 = **Yushin Ito + AlphaStarmie**（08-03に両名義が併存＝リネーム）
        # / sig 156952a871（104と同一デッキ）/ 07-17〜08-07 / 5,309試合・435,150決定。
        # 現行104は4,254試合・355,621決定なので +25%。
        # **作り直す主因はデータ量ではなく語彙の陳腐化**:
        #   104 の META_ARCH は7系統で、**Mega Lopunny(直近14.3%) と Ogerpon(6.0%) が無い**
        #   ＝メタの20.3%の対面を識別できていない。META_CARDS も 52→81枚（33枚欠落）。
        #   さらに 104 は norm_clip=False で分布外の値がそのまま流れる（EXP-111の機序）。
        # 2条件:
        #   alakazam162    全試合    435,150決定
        #   alakazam163win **勝ちのみ** 222,994決定（2,956試合）
        # Marnie(160/161)では勝ち選別が9対面すべてで負けた（加重-2.0pt）。別デッキ・
        # 別教師で再現するかを見る（EXP-116のYushinでは+6.5ptだった）。
        # **EXP-121 のスイープ**: プール最大枠 Marnie を最新手法で再構築（160/161）。
        # 教師 = Luca / sig 1ec0f47981 / 2026-07-16〜08-06 / **5,458試合・483,798決定**
        # （現行101は833試合・76,572決定なので **6.3倍**）。47クラス・562次元。
        # エンジン変更(07-17)前のデータも使う: 新旧直接比較で **399/400ゲーム一致**
        # と実測済み（knowledge/discussions/727094）。
        # 2条件:
        #   marnie160    全試合      483,798決定
        #   marnie161win **勝ちのみ**  304,262決定（3,180試合）
        # EXP-116 は「勝ち選別 +6.5pt / データ半減 −9.5pt = 差引 −3.6pt」で、
        # 勝ちのみが 183,934決定しか無かった。今回はその **1.65倍**あるので、
        # 「データが十分なら勝ち選別が勝つ」を直接検証できる。
        # **EXP-120 のスイープ**: Mega特徴を入れた再挑戦（158 Lucario / 159 Kangaskhan）。
        # 157/156 は忠実度 −32.7/−36.2pt で落ちた。原因はエンジンの
        # `State.h::getPrizeCount`（Mega ex は**サイド3枚**、ex は2枚、通常1枚）が
        # 特徴に無かったこと。**cg/api.py のドキュメントは「ex は Mega を含む」と
        # 書いているが、実装(Api.h:243)は排他**で、Mega Lucario は ex=False。
        # よって従来の is_ex ではデッキの主役が 0.0 になっていた。
        # 追加: 盤面 my_act_prize / my_bench_prize_max / my_prize_exposure /
        #       opp_bench_prize_max、選択肢 is_mega_ex / opt_prize（OPT_DIM 29→31）。
        # **156/157 が同一条件の前測定**なので、この修正の効果を純粋に切り出せる。
        # **EXP-119 のスイープ**: 主力候補 Mega Lucario ex（157）。
        # 教師 = Majkel1337 / sig 0c0f140f4f / 08-02〜08-07 / 865試合・50,538決定。
        # 直近1週間の個体勝率 **69.2%（メタ加重58.2%）** で全個体中トップ。
        # 現行主力の教師 Yushin(Alakazam) はメタ加重54.4% なので +3.8pt の上積みを狙う。
        # 550次元・52クラス。設定は 104 と同一（標準化 + ±10σクリップ + dropout0.2
        # + wd1e-4 + additive + 60ep cosine）。
        # **決定数が130 Lopunny(失敗, 53,524)とほぼ同水準**なので、学習後は必ず
        # vs random と tools/act_freq.py の end膨張で早期スクリーニングする。
        # **EXP-118 のスイープ**: プール枠 Mega Kangaskhan ex のSLレプリカ（156）。
        # 教師 = James Cox / James Cox & Henry Chao（同一チームの改名。sig c64b28ca9d）。
        # 625次元・96クラス・1,764試合112,331決定。設定は EXP-097 で確定した最良
        # （標準化 + ±10σクリップ + dropout0.2 + wd1e-4 + additive）で 104 と同一。
        # **EXP-096 の 100 が同じ教師・1,265試合・283次元で失敗している**（`end` の
        # 過剰選択で random 相手 70.5%）ので、学習後に必ず行動頻度を教師と突合する。
        # 3シード振る（val の sd が約1.8pt あるので単発の差を読まない）。
        # **EXP-124: 容量スイープ**（165のデータ 541次元・435,150決定で測り直す）。
        # 前回の監査（EXP-117）では d_model 384 が 256 と同値だったが、それは
        # 104のデータ（506次元・355,621決定）での1本のみ。165はデータが+22%あり、
        # **layers と d_opt は一度も振っていない**（d_opt は引数化すらされていなかった）。
        # アンカーは 165 の s1=0.8671 / s2=0.8637（シード幅 0.34pt）。
        # **0.5pt以上のval改善がなければ差なしと判定する**。
        # **EXP-122 の語彙アブレーション**（104と完全に同一の教師データ 4,254試合・
        # 355,621決定で、**特徴の語彙だけを削る**）:
        #   104  opp_seen 52枚 + oparch 7系統   506次元  → プールv7 70.8%
        #   166  opp_seen 81枚 + oparch 9系統   541次元  → **68.5%（−2.29pt）**
        #   **167  語彙なし                      451次元  → ?**
        #   **168  oparch のみ（opp_seen除去）   460次元  → ?**
        # opp_seen は「相手の場・付随カード・**トラッシュ**・スタジアムの累積枚数」なので
        # 試合が進むほど単調増加し**ターン進行の代理**にもなる。oparch はその要約
        # （看板カードを何割見たか）。要約だけで足りるなら生の52次元は冗長。
        # **EXP-125**: 166 = 165の特徴器 × **104と完全に同一の教師データ**
        # （Yushin Ito / 156952a871 / 07-17〜29 / 4,254試合・355,621決定。
        #  欠落2件[no decisions]まで104と一致）。
        # 変数を**特徴器だけ**に絞り、166−104 で「特徴更新の純粋な効果」、
        # 165−166 で「データ追加（07-30〜08-07）の純粋な効果」を分離する。
        # 追加データは対Marnieで教師の勝率が 52.0%→43.9% に落ちた期間なので、
        # **165−166 は負になりうる**（その場合166が最良の候補になる）。
        # **EXP-122 の到達点**: 170 = **教師データ基準で選んだ語彙** × 165のデータ。
        #   opp_seen 73枚（候補85枚から board_mu ≥0.02 のもの。**死に次元ゼロ**）
        #   oparch **0個**（168の実測で寄与+0.0pt。冗長性測定でも F1 0.85〜0.96 で
        #                  既存特徴から予測可能＝情報として重複）
        #   524次元 / クリップなし（104と条件を揃える）
        # 語彙は「メタシェア」ではなく「**教師データでの観測頻度**」で選ぶ。
        # メタシェアで選んだ166は +33枚中18枚が死に次元で **-2.29pt** だった。
        # **EXP-122 の段階適用**: 104 → 165 の変更を1つずつ当てる。
        #   **169 = 104の特徴器（506次元・opp_seen 52・oparch 7・OPT_DIM 29・クリップ無し）
        #          × 165の教師データ（5,309試合・435,150決定、07-17〜08-07）**
        #   → 「データだけ増やす」純粋効果。166（特徴だけ変える）の対になる統制。
        # **EXP-122: 172 = 171 と完全に同一の特徴器・語彙・clip で、勝ち試合のみ**。
        #   171 の抽出pklから `--only win` でパック（2,956試合 / 222,994決定、-49%）。
        #   **特徴器を1ビットも変えない**ことで勝ち選別の効果だけを分離する。
        #   勝ち選別はこれまで3例すべて負け（161 Marnie -2.0pt / 163 Alakazam -2.4pt /
        #   163 vs104 直接 -1.2pt）。171は語彙最適化+clipで条件が違うので4例目として測る。
        # **EXP-122: 171 = 170 + clip 10.0 + 冗長prize 4個を除去**（520次元）。
        # clip は実対戦の **28.4%の決定**で |z|>10 が発生しており（訓練時は0.04%）、
        # 挙動に実質的に関わる。170(clipなし)との差でクリップの純粋効果を測る。
        # 冗長prize（my_act_prize / my_bench_prize_max / my_prize_exposure /
        # opp_bench_prize_max）は ms*_prz / os*_prz の要約で完全に冗長と判明したため除去。
        # 選択肢側の is_mega_ex / opt_prize は残す（is_ex が Mega ex で 0.0 になる穴を埋める）。
        # **EXP-122: 174 = この系統の設計を原理で揃えた最終形**（449次元）。
        #   170から: 完全重複55列 / 定数21列 / 語彙の逸脱4枚 を削除、
        #   対称なサイド特徴2つ と 脅威追跡3枚（Fezandipiti/Unfair Stamp/Xerosic's）を追加。
        #   **clip は 0.0**。clipは定数列(sd=1e-3)の1,000シグマを止める対症療法だったが、
        #   174はその定数列を除去したので役目が無い。10.0のままだと生きた列の33%に掛かり
        #   本物の信号を潰す（171 vs 170 の -0.54pt の主因と考えられる）。
        #   ※ s1/s2 を norm_clip=0.0、s3/s4 を 10.0 にすると**特徴を完全に固定したまま
        #     clipの純効果**が測れる（171 vs 170 では特徴削除と交絡していた）。
        # **EXP-124 のスイープ**: ExIt（Expert Iteration）。**186を初期値**に、
        # 決定化ロールアウトが見つけた補正 1,089件を**試合系列に埋め込んで**継続学習する。
        #   補正の真の利得 = **不偏gap +0.0945 ±0.0120**（前半でbestを選び後半で測った値。
        #   素の平均 +0.16 の 60% は選択バイアスなので、それを抜いた実数）。
        # 決定ごとの損失重み dw は npz 側で作る（補正 44.4 / 既存 1.0 / **文脈専用 0.0**）。
        # dw=0 の決定は勾配も評価指標も受け取らないが、**LSTMに履歴を運ばせる**ために
        # 系列に残してある。長さ1の系列で学習するとゼロ状態になり、実状態と
        # **22.2%の決定で判断が変わる**＝教師そのものが壊れる。
        # **どこまで動かすかは val では決まらない**（元データ側は劣化の検知器、
        # 補正側は単調に下がる）。EXP-114 と同じ運用で
        # **step 200/400/800/1600 と最終を全部残し、対戦で選ぶ**。
        # lr は 1e-3 → **1e-4**（186の重みを壊さないため）。cosine は T_max=エポック数。
        # **EXP-134 のスイープ**: 内部正規化と深さ。
        # EXP-121 は容量スイープで **layers 3 が明確に悪化**したことを
        # 「ネットワークに内部正規化が一切ないため深くできない」と診断したが、
        # **その対処（LayerNormを入れる）は一度も試していなかった**。診断の検定。
        # データは 206（=186 + 相手の直前の選択、prev 6列）で固定し、
        # **正規化の有無 × 深さ 2/3 の2×2**にする。ln2 が 206 と同一構成の対照。
        #   ln0_l2 : 正規化なし・layers2 = **206 と同一（アンカー）**
        #   ln1_l2 : 正規化あり・layers2
        #   ln1_l3 : 正規化あり・layers3  ← 診断が正しければ **ここが伸びる**
        #   ln0_l3 : 正規化なし・layers3 = EXP-121 の再現（悪化するはず）
        # **EXP-134 v2**: 2×2 の残り1セルを埋める。
        #   186    = LayerNormなし・prev_oppなし  71.25%（既知）
        #   206    = なし・あり                   70.94 / 70.26（既知）
        #   208    = あり・あり                   71.76%（既知）
        #   **本run = あり・なし**  ← 186のnpz（prev 3列）で学習するので
        #   _HAS_PO=False となり、**「186 + LayerNorm」ちょうど**になる。
        #   これで LayerNorm 単体の効果と、prev_opp との交互作用が分離できる。
        # **EXP-134 v3**: 186データ（prev 3列）上で **LayerNormあり × 深さ**を振る。
        # 深さはこれまで 206データ上でしか測っていない（ln1_l3 = 0.8687 で L2 に届かず）。
        # 186データでも同じかを確認する。**layers 以外は 186 と完全に同一**。
        # 参考: 186 = layers2・正規化なし  val 0.8648/0.8662  プール 71.25%
        #       186+LN・layers2            val 0.8703/0.8704  プール **70.31%**
        # **EXP-134 v4**: layers=1 が val 最良（0.8718 / MAIN 0.8411）かつ
        # パラメータは186の60%（790,531 vs 1,316,867）。深さは単調に悪化した
        # （L1 0.8718 > L2 0.8703 > L3 0.8696 > L4 0.8682）。
        # **容量は飽和ではなく過剰**だった可能性があるので、シードを足して確認する。
        # **EXP-134 v5**: 比較を**同一シード群**に揃える。
        # これまでベースライン186は1シード（=s1）なのに候補は3シードで、不公平だった。
        # 主対比は **L2 vs L1（どちらも正規化なし＝186からの最小変更）** を3シードずつ。
        #   既存: L2 s1=71.25%(=186), L2 s2=未評価, L1n s1=71.47%
        # **EXP-135**: 2×2 の残りセル。**L1 × 相手の直前の選択（prev_opp）**。
        # 既知（すべて同一デッキ・同一教師・正規化なし）:
        #   L2 / prev_opp なし  val 0.8661  プール **70.93%**（3seed）
        #   L2 / prev_opp あり  val 0.8655  プール  70.60%（2seed）
        #   L1 / prev_opp なし  val 0.8695  プール **71.45%**（3seed）
        #   **L1 / prev_opp あり  ← 本run**
        # 206データ（prev 6列）で学習するので _HAS_PO=True になり自動で有効化される。
        # **EXP-136**: L1（layers=1・正規化なし）の**確率平均アンサンブル**が
        # 3本で **プール 71.45%（単体平均）→ 73.26%（+1.81pt）** と大きく効いた。
        # EXP-117 は L2（容量過剰）の3本で「val +0.86pt → プール −0.03pt」と
        # 棄却していたが、**容量が適正だと3本の誤差が独立になり効く**。
        # 重み平均は cos類似度 +0.033（ほぼ直交）で完全に壊れた（val 0.53）＝
        # 3本は本当に別の解にいる。
        # **シードを5本に増やして収穫逓減の位置を測る**（s4/s5 を追加）。
        # **EXP-137**: 教師の悪手を学習対象から外す（DeNA の Speed Up 対策と同型）。
        # 実測: 教師は `my_dmg_nullified`（Powerful Hand がダメカン配置＝効果なので
        # Articuno Repelling Veil / Mist・Rock Fighting Energy で無効化される状態）で
        # **Powerful Hand を 5,265回撃っている**（全 Powerful Hand の 21%）。
        # しかも **5,265件すべてで他に手があり**（選択肢中央13、打つ手なしは0件）、
        # Dudunsparce（Land Crush = 素のダメージで通る）は**選択肢に一度も現れない**。
        # ＝ 時間稼ぎでも打つ手なしでもない、教師の明確な機械的欠陥。
        # 対処は**負の報酬でなく損失重み dw=0**（DeNAは負報酬だとカード自体を避けると警告）。
        # 土台は **L1（layers=1・正規化なし）**＝プール71.45%（3seed）で、186のL2(70.93%)より上。
        # **【EXP-139 結果: 棄却】** 狙い所（正解クラスが同点の決定）で **−0.03pt**
        # （3シード sd 0.59pt, t=−0.10）、全体val +0.013pt、MAIN −0.013pt。
        # 機序: **学習ラベルを board の ms{slot}_{種族} one-hot + inPlayIndex から
        # 100.0000% 復元できた**＝情報は既に入力に在り、内積が合成できていた。
        # プールは事前登録どおり回していない。**再開する前に EXP-139.md を読むこと。**
        # 以下は当時の設計メモ（`target_emb` は opt_tcid.npz が無ければ既定off）。
        #
        # **EXP-139**: 貼り先/進化先の**正体**を opt_enc に入れる。
        # 現行の opt_vec が対象について持つのは inPlayArea/inPlayIndex（位置）と
        # 残HP比/エネ数の4次元だけで、種族は入っていない。MAINでは cls_head の
        # ATTACH_BUCKET が代役をするが **Abra(741) と Kadabra(742) が同一クラス
        # 'abra_p'** という粗さがある。実測（教師435,150決定）:
        #   対象を持つ選択肢            953,577 = 全実選択肢の 25.9%
        #   うち対象が一意でない(MAIN)  244,524 = MAIN対象ありの 26.0%
        #   ctx37 EVOLVE（非MAIN=cls_head が効かない） 11,836
        #   → **対象の正体が未表現の選択肢 256,360 = 全体の 7.0%**
        #   **MAIN決定の 22.1%** が「同一クラスに2個以上の選択肢」＝クラス頭が同点で、
        #   位置番号とエネ数だけで割っている。教師がそこから選んだのは 2.6%。
        # コストは opt_enc.0 が 64x62→64x94 の **+2,048** のみ（L1全体の +0.26%）。
        # card_emb は**共有**（EXP-133 の prev_opp と同じ判断）。
        #
        # **ベースと実験を同一runで交互に回す**。同じ train.npz / 同じ opt_tcid.npz を
        # 読ませたまま `target_emb_off` だけを反転するので、データ・前処理・系列順・
        # ハイパラがすべて一致し、**差は当該フラグのみ**になる。過去のL1 3seed
        # （val 0.8695 / プール71.45%）を基準にすると当時のハイパラ差が交絡しうるため、
        # 基準側も引き直す。**交互順**なので途中で12h上限に当たっても対の比較が残る。
        # **【EXP-140 結果: 棄却】** L0（再帰なし）は val +0.62pt（t=12.4）だが
        # プール 72.10% vs 222の73.26% = **−1.16pt**。再開前に EXP-140.md を読むこと。
        #
        # **【EXP-142 結果: 棄却】** joint（cross-encoder のみ）は val **−0.54pt**
        # （3シードとも同符号、t=−7.8）/ MAIN **−1.02pt**。cls_head を戻した jcls は
        # base 並み（−0.18pt, n.s.）なので**落ちた原因は65クラスの枚挙の消失で確定**。
        # ただし**非MAINでは cross が明確に勝つ**（ctx15 +12.8 / ctx30 +11.1 / ctx8 +4.4）＝
        # two-tower の64軸制約は実在するが、効く範囲は val 決定の1.2%しかない。
        # プールは未実施（理由は EXP-142.md 7節）。**再開前に EXP-142.md を読むこと。**
        #
        # **EXP-142: two-tower を捨てて cross-encoder（joint）にする**
        # 動機は性能でなく**構造の負債の一括返済**。two-tower のために生じたものが消える:
        #   ctx_bias   出力に一切影響しない死重（softmax/argmax はシフト不変）
        #   ctx_scale  proj に対し 64 パラメータ分のゲージ自由度
        #   cls_head   65クラスが**デッキ固有**（deck.csv から生成）。しかも
        #              ATTACH_BUCKET は Abra(741)/Kadabra(742) を同一クラスに潰す
        #   式が2種類  cls_head は MAIN のみ。非MAINは内積だけ
        #   生の序数   opt.type/opt.area を enum の整数値のまま入れていた → 埋め込みへ
        # 表現力の面でも、内積は**双線形＝相互作用が d_opt=64 本の軸に制限される**。
        # 検索分野の bi-encoder / cross-encoder と同じ対立で、two-tower の動機は
        # 「アイテムを事前計算してインデックス化できる」ことだが、
        # **我々の選択肢は5〜30個で毎回作り直すので、その利益を1つも受けていない**。
        #
        # **容量はほぼ据え置き**（788,995 → 792,786、+0.48%）。EXP-140 で容量が
        # 効くと分かったので、ここは動かさない。
        #
        # 3水準。**cls_head の枚挙が必要かを切り分ける**のが要点:
        #   base       現行 L1（=222の構成要素）
        #   joint      cross-encoder のみ（cls_head なし）  ← 美しい版
        #   joint_cls  joint + cls_head 併存
        # 両方 base 並み  → 枚挙は不要＝joint 採用。joint_cls だけ → 枚挙は必要。
        # 両方負ける      → 内積+cls_head の分業が本質＝棄却。
        #
        # **注意**: joint 系は対象カード埋め込み（EXP-139 の opt_tcid）も使う。
        # EXP-139 単独では −0.03pt なので交絡は無視できるが、設計上は
        # 「対象の正体を1箇所で解決する」という負債返済の一部。
        # base は 222 と厳密に同一にするため target_emb_off を立てる。
        cfgs = [dict(tag="joint_s1", seed=1, head="joint"),
                dict(tag="jcls_s1", seed=1, head="joint_cls"),
                dict(tag="base_s1", seed=1, target_emb_off=True),
                dict(tag="joint_s2", seed=2, head="joint"),
                dict(tag="jcls_s2", seed=2, head="joint_cls"),
                dict(tag="base_s2", seed=2, target_emb_off=True),
                dict(tag="joint_s3", seed=3, head="joint"),
                dict(tag="jcls_s3", seed=3, head="joint_cls"),
                dict(tag="base_s3", seed=3, target_emb_off=True)]
        for c in cfgs:
            c.update(layers=1, layer_norm=False)
        for c in cfgs:
            c.update(d_model=256, d_opt=64, norm_clip=0.0)
        for c in cfgs:
            c.update(dropout=0.2, d_model=256, v_weight=0.0, v_detach=False,
                     awr_beta=0.0, freeze_policy=False, v_hidden=0,
                     focal_gamma=0.0, opt_attn=False, init_from=None,
                     lr=1e-3, max_steps=0, stop_metric="acc", save_final=False)
        for c in cfgs:
            c.setdefault("head", "additive")
            c.setdefault("no_norm", False)
            c.setdefault("wd", 1e-4)
        for c in cfgs:
            c.setdefault("layers", 2)
            c.update(epochs=60, cosine=True)
        results = []
        for c in cfgs:
            print(f"\n{'='*60}\n### {c['tag']}\n{'='*60}", flush=True)
            a2 = argparse.Namespace(**vars(args))
            a2.sweep = False
            a2.d_model, a2.layers = c["d_model"], c["layers"]
            a2.d_opt = c.get("d_opt", 64); a2.d_card = c.get("d_card", 32)
            # `global` は main() の冒頭で1回だけ宣言済み。ここで再宣言すると
            # **SyntaxError: name '_NCLIP' is assigned to before global declaration**
            # になる（ast.parse は通るが compile で落ちるので、検査は compile で行う）
            globals()["_NCLIP"] = float(
                c.get("norm_clip", getattr(args, "norm_clip", 10.0)))
            a2.epochs, a2.cosine = c["epochs"], c["cosine"]
            a2.seed = c.get("seed", 0)
            a2.no_norm = c.get("no_norm", False)
            a2.dropout = c.get("dropout", 0.0)
            a2.wd = c.get("wd", 1e-4)
            a2.head = c.get("head", "additive")
            a2.v_weight = c.get("v_weight", 0.0)
            a2.awr_beta = c.get("awr_beta", 0.0)
            a2.awr_warmup = c.get("awr_warmup", 10)
            a2.v_detach = c.get("v_detach", False)
            a2.v_hidden = c.get("v_hidden", 0)
            a2.focal_gamma = c.get("focal_gamma", 0.0)
            a2.opt_attn = c.get("opt_attn", False)
            a2.layer_norm = c.get("layer_norm", False)
            a2.freeze_policy = c.get("freeze_policy", False)
            a2.target_emb_off = c.get("target_emb_off", False)
            a2.post_mlp = c.get("post_mlp", 0)
            a2.lr = c.get("lr", getattr(args, "lr", None))
            a2.max_steps = c.get("max_steps", getattr(args, "max_steps", None))
            a2.stop_metric = c.get("stop_metric", getattr(args, "stop_metric", None))
            a2.save_final = c.get("save_final", getattr(args, "save_final", None))
            a2.init_from = c.get("init_from", None)
            a2.out = args.out.replace(".npz", f"_{c['tag']}.npz")
            best, ctx = _train_one(a2)
            results.append((c["tag"], best, ctx))
        print(f"\n{'='*60}\n### スイープ結果\n{'='*60}")
        for tag, best, ctx in results:
            m = ctx.get(0, (0, 1))
            gap = ctx.get("_gap", 0.0)
            auc = ctx.get("_v_auc")
            print(f"{tag:<12} val={best:.4f}  MAIN={m[0]/max(1,m[1]):.4f}"
                  f"  訓練-検証差={gap:+.4f}"
                  + (f"  valueAUC={auc:.4f}" if auc else ""))
        return
    _train_one(args)


def _train_one(args):

    # **乱数シードを固定する**。固定しないと重み初期化もバッチ順も毎回変わり、
    # 「版Aと版Bの差」が学習ラン間のばらつきと区別できない（v8で判明した穴）。
    seed = int(getattr(args, "seed", 0))
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # KaggleのP100(sm_60)は現行イメージのPyTorch(sm_70+)と非互換でLSTMのcudnnカーネルが無い。
    # 実際に小さなLSTMを走らせて検証し、駄目ならCPUへ落とす。
    dev = "cpu"
    if torch.cuda.is_available():
        try:
            _l = torch.nn.LSTM(4, 4).cuda()
            _o, _ = _l(torch.zeros(2, 1, 4, device="cuda"))
            _ = float(_o.sum())
            dev = "cuda"
        except Exception as e:  # noqa: BLE001
            print(f"CUDA使用不可（{type(e).__name__}）→ CPUで学習します")
    print("device:", dev, flush=True)

    # Kaggleのマウントパスはデータセットのtitle由来slugになることがあるため探索する
    npz_path = args.npz
    if not os.path.exists(npz_path):
        import glob
        cands = sorted(glob.glob("/kaggle/input/**/*.npz", recursive=True))
        print(f"{npz_path} が無い。/kaggle/input 配下の候補: {cands}")
        if not cands:
            for root, dirs, files in os.walk("/kaggle/input"):
                print(" ", root, dirs[:5], files[:5])
            raise FileNotFoundError("train.npz not found under /kaggle/input")
        npz_path = next((c for c in cands if c.endswith("train.npz")), cands[0])
        print("使用:", npz_path)
    z = np.load(npz_path, allow_pickle=True)
    data = {k: z[k] for k in z.files}
    print({k: data[k].shape for k in ("board", "opt_cid", "opt_vec", "chosen")})

    # **対象カードID（EXP-139）は別ファイルで合流させる**。train.npz 本体を書き換えると
    # ベース(219系)と実験(EXP-139)が別ファイルを読むことになり、データ差が交絡しうる。
    # 同一の train.npz + 有無だけが違う添付ファイル、という形にして変数を1つに保つ。
    # **max_game_len の外れ値除去より前**に合流させること（後だと行がずれる）。
    if "opt_tcid" not in data:
        import glob as _g
        _c = ([os.path.join(os.path.dirname(npz_path), "opt_tcid.npz")]
              + sorted(_g.glob("/kaggle/input/**/opt_tcid.npz", recursive=True)))
        _tp = next((p for p in _c if os.path.exists(p)), None)
        if _tp:
            _tz = np.load(_tp)
            _t = _tz["opt_tcid"]
            if _t.shape != data["opt_cid"].shape:
                raise SystemExit(
                    f"opt_tcid の形が opt_cid と違う: {_t.shape} vs "
                    f"{data['opt_cid'].shape}（{_tp}）")
            # **どの train.npz から作ったかを指紋で照合する**。別データで作った
            # opt_tcid を黙って載せると、対象が全く別の局面のものになる。
            _fp = int(_tz["fingerprint"]) if "fingerprint" in _tz else None
            _cur = int(data["opt_cid"].sum()) ^ int(data["nopt"].sum())
            if _fp is not None and _fp != _cur:
                raise SystemExit(
                    f"opt_tcid の指紋不一致: {_fp} != {_cur}（{_tp}）"
                    " — 別の train.npz から作られている")
            data["opt_tcid"] = _t
            print(f"**opt_tcid を合流**: {_tp}")

    # 異常に長いゲーム（ループ/膠着で決定数が数百〜数千）は代表性が無く、決定数ベースの
    # 損失では過剰な重みを持つ（15件で全決定の9.3%）。**派生配列を作る前に**除外する。
    if getattr(args, "max_game_len", 0):
        lens = np.bincount(data["gidx"])
        bad = np.where(lens > args.max_game_len)[0]
        if len(bad):
            keep = ~np.isin(data["gidx"], bad)
            n0 = len(data["gidx"])
            for k in ("board", "logs", "prev", "ctx", "turn", "nopt",
                      "opt_cid", "opt_tcid", "opt_vec", "opt_cls", "chosen",
                      "first", "gidx", "dw", "is_new"):
                if k in data and len(data[k]) == n0:
                    data[k] = data[k][keep]
            print(f"外れ値ゲーム除外: {len(bad)}件 / 決定 {n0}→{len(data['gidx'])}"
                  f" ({1 - len(data['gidx']) / n0:.1%}削減)")

    # 勝敗ラベル（ゲーム単位）。**これまで一度も学習に使っていなかった**。
    # 負け試合1844件(43.3%)の全決定を等重みで正解として学習している状態の解消は
    # AWR（EXP-101予定）で行うが、まず value head の baseline として使えるか測る。
    # **決定ごとの損失重み dw**（EXP-124）。ExItの補正決定に大きな重みを、
    # 「LSTMの文脈としてだけ要る決定」に 0 を与える。0 の決定は勾配も評価指標も
    # 受け取らないが、**系列の一部として状態を運ぶ**。無い npz では全1になり従来と一致。
    _dw = data.get("dw")
    if _dw is None:
        dw_all = np.ones(len(data["gidx"]), np.float32)
    else:
        dw_all = np.asarray(_dw, np.float32)
        assert len(dw_all) == len(data["gidx"]), "dw の長さが決定数と違う"
        _nz = dw_all > 0
        print(f"決定重み dw: >0 が {int(_nz.sum()):,}/{len(dw_all):,} / "
              f"総和 {dw_all.sum():,.0f} / 最大 {dw_all.max():.1f} "
              f"（dw>1 の重み占有 {100*dw_all[dw_all>1].sum()/dw_all.sum():.1f}%）")

    gres_of = None
    v_w = float(getattr(args, "v_weight", 0.0))
    awr_beta = float(getattr(args, "awr_beta", 0.0))
    _lam = float(getattr(args, "awr_lambda", 0.95))
    _wmax = float(getattr(args, "awr_wmax", 20.0))
    if awr_beta > 0 and v_w <= 0:
        # **V なしのAWRは交絡そのもの**なので、無言で劣化させず落とす
        raise SystemExit("--awr-beta を使うには --v-weight > 0 が必須（Vがadvantageの基準）")
    if v_w > 0:
        if "gres" not in data:
            raise SystemExit("npz に gres が無い（pack_dataset を新しくして作り直す）")
        gres_of = data["gres"].astype(np.float32)
        print(f"value head: 有効 (v_weight={v_w}) / 勝敗の内訳 "
              f"{dict(zip(*np.unique(gres_of, return_counts=True)))}")

    vocab = build_vocab(data["opt_cid"], data["prev"])
    # **--init-from のときは元モデルの vocab をそのまま使う**。データから作り直すと
    # カードID→埋め込みindex の対応がずれ、サイズが同じ場合は例外も出ずに壊れる。
    _if = getattr(args, "init_from", None)
    if _if:
        import glob as _g
        _p = _if if os.path.exists(_if) else next(
            iter(sorted(_g.glob("/kaggle/input/**/" + os.path.basename(_if),
                                recursive=True))), None)
        if _p is None:
            raise SystemExit(f"--init-from が見つからない: {_if}")
        _zz = np.load(_p, allow_pickle=False)
        vocab = {int(k): int(v) for k, v in zip(_zz["vocab_keys"], _zz["vocab_vals"])}
        args.init_from = _p
        print(f"vocab を {_p} から継承（{len(vocab)}語）")
    print("card vocab:", len(vocab))
    # 最小のIDは NULL_CID=-3（棄権オプション）なので **+3** ずらす。
    # +2 だと -3 が負インデックスになり、最大カードIDと同じ末尾スロットを共有して
    # 「棄権と特定カードが同一埋め込み」になる（例外は出ないので発見しにくい）。
    OFF = 3
    lut = np.zeros(int(max(vocab)) + OFF + 1, np.int64)
    for k, v in vocab.items():
        lut[k + OFF] = v          # -3→0, -2→1, -1→2 の位置にずらす
    opt_cidx = lut[data["opt_cid"] + OFF]
    prev_cidx = lut[data["prev"][:, 0].astype(np.int64) + OFF]
    # **相手の直前の選択**（EXP-133）: prev が6列の npz でのみ有効になる
    _HAS_PO = data["prev"].shape[1] >= 6
    prevo_cidx = (lut[data["prev"][:, 3].astype(np.int64) + OFF] if _HAS_PO else None)
    if _HAS_PO:
        print(f"**相手の直前の選択を使用**（prev 6列）: 定義済み "
              f"{100*(data['prev'][:,3]>=0).mean():.0f}% / 種類 "
              f"{len(np.unique(data['prev'][:,3]))}")
    # **貼り先/進化先の埋め込み**（EXP-139）。npz に opt_tcid があるときだけ有効。
    # 無いデータでは _HAS_TE=False となり、従来と**厳密に同じ**計算になる。
    # **--no-target-emb で明示的に切れる**。同一の train.npz + opt_tcid.npz を読ませたまま
    # フラグだけを反転できるので、ベースと実験で**データも順序も完全に同一**にできる。
    _HAS_TE = ("opt_tcid" in data
               and not bool(getattr(args, "target_emb_off", False)))
    opt_tcidx = lut[data["opt_tcid"] + OFF] if _HAS_TE else None
    if _HAS_TE:
        _tv = data["opt_tcid"]
        print(f"**対象カードの埋め込みを使用**: 対象あり選択肢 "
              f"{int((_tv > 0).sum()):,} / 種類 {len(np.unique(_tv[_tv > 0]))}")
    # **joint ヘッド用のカテゴリ列（EXP-142）**。opt_vec[:,:,0]=opt.type /
    # [:,:,1]=opt.area は enum の整数値だが、**この後 _apply で標準化されて実数になる**
    # ので、ここで整数のまま切り出しておく。NULL は -1 なので +1 して 0 に寄せる。
    _EMB_OFF = 1
    opt_type_idx = np.clip(data["opt_vec"][:, :, 0].astype(np.int64) + _EMB_OFF, 0, 19)
    opt_area_idx = np.clip(data["opt_vec"][:, :, 1].astype(np.int64) + _EMB_OFF, 0, 15)

    # ctx埋め込みは **SelectContextの全種類分**を確保する。
    # データ最大値+1で作ると、学習データに無いctx（COIN_HEAD=46等は実戦で普通に来る）が
    # 範囲外→例外→フォールバックになる。範囲内でも未学習ctxは乱数埋め込みのまま使われ、
    # 例外にならない分こちらの方が発見しにくい（v4までの穴D）。
    n_ctx = max(int(data["ctx"].max()) + 1, 49)
    O = data["opt_cid"].shape[1]
    mask = np.arange(O)[None, :] < data["nopt"][:, None]
    has_cls = "opt_cls" in data
    # **クラス総数はデータセットに記録された値を優先する**。観測max+1で決めると、
    # 教師データに現れないクラスが末尾にある場合ヘッド幅が足りず、推論時に
    # 範囲外indexで落ちる（生成特徴器はクラスが疎になるので現実的な穴）。
    n_cls = 0
    if has_cls:
        n_cls = max(int(data["opt_cls"].max()) + 1,
                    int(data["n_cls"]) if "n_cls" in data else 0)
    print(f"MAIN階層ヘッド: {'有効' if has_cls else '無効(opt_cls無し)'} n_cls={n_cls}"
          f" (観測max={int(data['opt_cls'].max()) if has_cls else -1})")


    # **ゲーム単位**で train/val 分割（chunk単位で割ると長いゲームが両方に入りリークする）
    seqs, owner = to_seq(data, vocab, args.max_T)
    games = np.unique(owner)
    rng = np.random.default_rng(42)
    gperm = rng.permutation(games)
    # **新規/元データで層化して val を作る**（EXP-114）。
    # 訓練と同じ混合比の val にすることで、**最適化している目的そのもの**を
    # 保持データで測れる。混合val lossは
    #   Yushin部分（SILで離れるほど上がる）+ 新規部分（勝ちパターンを汎化するほど下がる）
    # の和なので、**得失が逆転した時点で最小になる**＝早期終了の正しい基準。
    _gnew = data.get("g_new")
    if _gnew is not None and int(np.asarray(_gnew).sum()) > 0:
        _gn = np.asarray(_gnew)
        _new_g = [int(x) for x in gperm if _gn[int(x)] == 1]
        _old_g = [int(x) for x in gperm if _gn[int(x)] == 0]
        val_games = set(_new_g[:max(1, int(len(_new_g) * args.val_frac))]
                        + _old_g[:max(1, int(len(_old_g) * args.val_frac))])
        n_vg = len(val_games)
        print(f"  val層化: 新規{max(1,int(len(_new_g)*args.val_frac))}試合 "
              f"+ 元{max(1,int(len(_old_g)*args.val_frac))}試合")
    else:
        n_vg = max(1, int(len(games) * args.val_frac))
        val_games = set(int(x) for x in gperm[:n_vg])
    val_seqs = [s for s, g in zip(seqs, owner) if g in val_games]
    tr_seqs = [s for s, g in zip(seqs, owner) if g not in val_games]
    # **元データと新規データの val を分けて持つ**（EXP-114）。
    # 混ぜた val loss は「Yushinから離れる分（上がる）」と「勝ちパターンを覚える分
    # （下がる）」の和で、どちらが動いたのか分からない。**別々に記録して両方見る**。
    val_old = val_new = []
    if _gnew is not None and int(np.asarray(_gnew).sum()) > 0:
        _gn2 = np.asarray(_gnew)
        val_old = [s for s, g in zip(seqs, owner)
                   if g in val_games and _gn2[int(g)] == 0]
        val_new = [s for s, g in zip(seqs, owner)
                   if g in val_games and _gn2[int(g)] == 1]
        print(f"  val内訳: 元{len(val_old)}系列 / 新規{len(val_new)}系列")
    print(f"sequences: train={len(tr_seqs)} val={len(val_seqs)} "
          f"(games train={len(games)-n_vg} val={n_vg})")

    # **標準化の統計は train 側の決定だけで取る**（val決定を含めると評価がリークする）。
    norm = None
    if not getattr(args, "no_norm", False):
        # **チャンクで集計し、その場で書き換える**。opt_vec は (355k,49,28)=1.95GB あり、
        # `(x - mu) / sd` を一括で書くと一時配列2本で +3.9GB になってKaggleのRAMを割る。
        tr_idx = np.sort(np.concatenate(tr_seqs))
        CH = 20000

        def _stats(key, sel=None):
            """train分割だけで (平均, 標準偏差) を1パス集計する。"""
            n = 0
            s1 = s2 = None
            for i in range(0, len(tr_idx), CH):
                blk = data[key][tr_idx[i:i + CH]].astype(np.float64)
                if sel is not None:               # opt_vec は実オプション行だけ使う
                    blk = blk[sel[tr_idx[i:i + CH]]]
                blk = blk.reshape(-1, blk.shape[-1])
                s1 = blk.sum(0) if s1 is None else s1 + blk.sum(0)
                s2 = (blk * blk).sum(0) if s2 is None else s2 + (blk * blk).sum(0)
                n += blk.shape[0]
            mu = s1 / max(1, n)
            var = np.maximum(s2 / max(1, n) - mu * mu, 0.0)
            return mu.astype(np.float32), np.maximum(np.sqrt(var), 1e-3).astype(np.float32)

        def _apply(key, mu, sd):
            a = data[key]
            for i in range(0, len(a), CH):
                blk = a[i:i + CH]
                blk -= mu          # in-place（一時配列を作らない）
                blk /= sd
                # **標準化後のクリップ**（EXP-111）。教師データに現れないカードの特徴は
                # 訓練中ほぼ常に0で sd が下限1e-3 に張り付く。本番で新メタの相手が出ると
                # その特徴に値が入り、**744σ**（実測: Ogerpon の Lively Stadium）という
                # 値がネットワークに流れて方策が壊れる（対Ogerpon 78.2%→30.7%）。
                # 学習時にも掛けることで、推論側のクリップと分布を一致させる。
                if _NCLIP:
                    np.clip(blk, -_NCLIP, _NCLIP, out=blk)
        # **padding行を平均に入れない**（nopt超のスロットは全ゼロで、実オプションの
        # 分布をゼロ側に引っ張る）
        bmu, bsd = _stats("board")
        omu, osd = _stats("opt_vec", sel=mask)

        # ==== 有界カウント列は**標準化しない**（EXP-122）====
        # 標準化は z=(x-mu)/sd なので、稀なカウント特徴では sd≈√p となり
        # **z(1枚) ≈ 1/√p** が発散する。実測(180): 追加した稀な語彙11枚は
        # z(1枚) 4.6〜23.7σ（既存69枚は中央2.3σ・最大5.2σ）で、重みは中央値並みのため
        # **発火した瞬間に既存語彙の5倍の寄与**を注入していた。
        # os*_resist_vs_me に至っては 199.7σ。これは EXP-111 の744σと同じ病理の弱い版。
        #
        # 上限が意味的に決まる列は **x/上限** に写す。単調な線形写像なので
        # **情報は1ビットも失われず**、clipのように値を潰さずに暴れだけが消える。
        #   opp_seen_{cid}  → 4（同名4枚まで。基本エネは語彙の①で除外済み）
        #                     **実測maxを使わない**のは、稀な札(実測max=1)と4枚積みで
        #                     「1枚見えた」の符号化が4倍ずれてしまうため
        #   その他の小さな非負整数列（pool_/left_/hand_/dis_/n_/line_/*_prz 等）
        #                   → 実測max（pool_/left_ はこれが正確に DECK_COUNTS になる）
        #   HP・打点・山札枚数など上限が意味で決まらない列 → **標準化のまま**
        # **上限は npz の feat_cap（特徴の定義側が持つ既知の上限）を最優先する**。
        # 自デッキ由来の列（pool_/left_/hand_/dis_）は DECK_COUNTS が上限そのものなので、
        # データから推定してはいけない（実測 hand_741 は 3 だが真の上限は 4。
        # 「4枚同時に手札へ来る局面が観測されなかった」だけで上限を誤認する）。
        # feat_cap が無い場合だけ、下のデータ駆動規則にフォールバックする。
        _cap_arr = data.get("feat_cap")
        _fk_n = [str(x) for x in data.get("feat_keys", [])]
        if _cap_arr is not None and len(_cap_arr) == len(bmu):
            _fixed = 0
            for _j in range(len(bmu)):
                _c = float(_cap_arr[_j])
                if _c > 0:
                    bmu[_j] = 0.0; bsd[_j] = _c; _fixed += 1
            print(f"**固定尺度**: {_fixed}/{len(bmu)}列を x/上限 に（feat_cap 由来）")
        elif len(_fk_n) == len(bmu):
            _B = data["board"]
            _mx = _B.max(0); _mn = _B.min(0)
            _isint = np.all(np.abs(_B - np.round(_B)) < 1e-6, axis=0)
            _fixed = 0
            for _j, _nm in enumerate(_fk_n):
                if re.match(r"^opp_seen_\d+$", _nm):
                    _cap = 4.0
                elif _isint[_j] and _mn[_j] >= 0 and 0 < _mx[_j] <= 20:
                    _cap = float(_mx[_j])
                else:
                    continue
                bmu[_j] = 0.0; bsd[_j] = _cap; _fixed += 1
            print(f"**固定尺度**: {_fixed}/{len(bmu)}列を x/上限 に（残りは標準化）")

        _apply("board", bmu, bsd)
        _apply("opt_vec", omu, osd)
        norm = dict(board_mu=bmu, board_sd=bsd, opt_mu=omu, opt_sd=osd)
        norm["norm_clip"] = np.float32(_NCLIP)
        # **--init-from のときは元モデルの標準化統計をそのまま使う**。
        # 統計が変われば入力が変わり、重みが同一でも**方策の出力が変わってしまう**
        # （凍結の意味が消える）。実測で board_mu が最大13.8ずれた。
        _if2 = getattr(args, "init_from", None)
        if _if2:
            _z2 = np.load(_if2, allow_pickle=False)
            if "board_mu" in _z2.files:
                bmu, bsd = _z2["board_mu"], _z2["board_sd"]
                omu, osd = _z2["opt_mu"], _z2["opt_sd"]
                norm = dict(board_mu=bmu, board_sd=bsd, opt_mu=omu, opt_sd=osd)
                print("標準化統計も元モデルから継承（方策の出力を完全に保つ）")
        print(f"標準化: board {len(bmu)}次元 (|平均|最大 {np.abs(bmu).max():.1f}, "
              f"sd最大 {bsd.max():.1f}) / opt {len(omu)}次元 "
              f"(|平均|最大 {np.abs(omu).max():.1f}, sd最大 {osd.max():.1f})")

    net = PolicyNet(
        n_board=data["board"].shape[1], n_logs=data["logs"].shape[1],
        n_optdim=data["opt_vec"].shape[2], n_cards=len(vocab), n_ctx=n_ctx,
        d_model=args.d_model, n_layers=args.layers, n_cls=n_cls,
        d_opt=getattr(args, "d_opt", 64), d_card=getattr(args, "d_card", 32),
        dropout=getattr(args, "dropout", 0.0),
        head=getattr(args, "head", "additive"),
        v_detach=bool(getattr(args, "v_detach", False)),
        v_hidden=int(getattr(args, "v_hidden", 0)),
        opt_attn=bool(getattr(args, "opt_attn", False)),
        prev_opp=_HAS_PO,
        target_emb=_HAS_TE,
        post_mlp=int(getattr(args, "post_mlp", 0) or 0),
        layer_norm=bool(getattr(args, "layer_norm", False))).to(dev)
    # **既存モデルの方策をそのまま載せる**（--init-from）。--freeze-policy なら
    # v_head 以外の勾配を止めるので、方策は1ステップも動かない＝提出済みと同一のまま。
    _init = getattr(args, "init_from", None)
    if _init and not os.path.exists(_init):
        # **Kaggle では /kaggle/input 配下のどこかにある**。ファイル名で再帰検索する
        # （データセットのマウント経路がランによって変わるため）。
        import glob as _g
        _pat = os.path.basename(_init)
        _hit = [x for x in _g.glob(f"/kaggle/input/**/{_pat}", recursive=True)
                if "_step" not in x and "_final" not in x]
        if _hit:
            print(f"--init-from を解決: {_hit[0]}", flush=True)
            _init = _hit[0]
    if _init:
        _z = np.load(_init, allow_pickle=False)
        _sd = net.state_dict()
        _miss = [k for k in _sd if k not in _z.files and not k.startswith("v_head")]
        if _miss:
            raise SystemExit(f"--init-from のモデルに無い重み: {_miss[:6]}")
        for k in _sd:
            if k in _z.files:
                _sd[k].copy_(torch.as_tensor(np.asarray(_z[k], np.float32)))
        net.load_state_dict(_sd)
        print(f"方策を {_init} から読み込んだ", flush=True)
    if getattr(args, "freeze_policy", False):
        _n = 0
        for k, prm in net.named_parameters():
            if not k.startswith("v_head"):
                prm.requires_grad_(False)
                _n += prm.numel()
        print(f"方策を凍結: {_n:,} パラメータを固定 / "
              f"学習するのは v_head の "
              f"{sum(p.numel() for p in net.v_head.parameters()):,} のみ", flush=True)

    wd = float(getattr(args, "wd", 0.0))
    _prm = [p for p in net.parameters() if p.requires_grad]
    opt = (torch.optim.AdamW(_prm, lr=args.lr, weight_decay=wd)
           if wd > 0 else torch.optim.Adam(_prm, lr=args.lr))
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
             if getattr(args, "cosine", False) else None)
    print("params:", sum(p.numel() for p in net.parameters()))
    if awr_beta > 0:
        print(f"AWR: β={awr_beta} λ={_lam} wmax={_wmax} "
              f"warmup={getattr(args, 'awr_warmup', 10)}ep / "
              f"**val一致率は下がりうる（教師の再現が目的ではない）**", flush=True)

    def _awr_weights(v, z, valid):
        """GAE(λ) で advantage を出し、exp(A/β) の重みを返す（平均1に正規化）。

        報酬は終端のみ（勝敗 z）なので γ=1。**系列は1試合1本**（max_T=300）なので
        行をまたぐ漏れは無い。最後の有効ステップが終端。
        """
        vf = valid.float()
        # 各行の最後の有効ステップ
        cum = vf.cumsum(1)
        last = valid & (cum == vf.sum(1, keepdim=True))
        v_next = torch.zeros_like(v)
        v_next[:, :-1] = v[:, 1:]
        delta = torch.where(last, z - v, v_next - v) * vf
        adv = torch.zeros_like(v)
        run = torch.zeros(v.shape[0], device=v.device, dtype=v.dtype)
        for t in range(v.shape[1] - 1, -1, -1):
            # 終端では蓄積をリセット（試合をまたがない）
            run = delta[:, t] + _lam * run * (~last[:, t]).to(v.dtype)
            adv[:, t] = run
        wt = torch.exp(adv / awr_beta).clamp(max=_wmax) * vf
        return wt / wt.sum().clamp(min=1e-6) * vf.sum().clamp(min=1.0)

    def make_batch(batch_seqs):
        T = max(len(s) for s in batch_seqs)
        B = len(batch_seqs)
        out = {}
        shapes = {"board": data["board"].shape[1], "logs": data["logs"].shape[1]}
        b_board = np.zeros((B, T, shapes["board"]), np.float32)
        b_logs = np.zeros((B, T, shapes["logs"]), np.float32)
        b_prevc = np.zeros((B, T), np.int64)
        b_prevn = np.zeros((B, T, 2), np.float32)
        b_poc = np.zeros((B, T), np.int64)
        b_pon = np.zeros((B, T, 2), np.float32)
        b_ctx = np.zeros((B, T), np.int64)
        b_ocid = np.zeros((B, T, O), np.int64)
        b_otc = np.zeros((B, T, O), np.int64)     # 対象カード（EXP-139。無効時は全0=PAD）
        b_oty = np.zeros((B, T, O), np.int64)     # opt.type（EXP-142 joint用）
        b_oar = np.zeros((B, T, O), np.int64)     # opt.area（EXP-142 joint用）
        b_ovec = np.zeros((B, T, O, data["opt_vec"].shape[2]), np.float32)
        b_mask = np.zeros((B, T, O), bool)
        b_lab = np.full((B, T), -100, np.int64)
        b_valid = np.zeros((B, T), bool)
        b_ocls = np.full((B, T, O), -1, np.int64)
        b_multi = np.zeros((B, T, O), np.float32)  # 複数選択のマルチホット
        b_k = np.zeros((B, T), np.float32)
        b_z = np.zeros((B, T), np.float32)      # ゲームの勝敗（±1、決定ごとに同値）
        b_dw = np.zeros((B, T), np.float32)     # 決定ごとの損失重み（0=文脈専用）
        for bi, s in enumerate(batch_seqs):
            L = len(s)
            b_board[bi, :L] = data["board"][s]
            b_logs[bi, :L] = data["logs"][s]
            b_prevc[bi, :L] = prev_cidx[s]
            b_prevn[bi, :L] = data["prev"][s][:, 1:3]
            if _HAS_PO:
                b_poc[bi, :L] = prevo_cidx[s]
                b_pon[bi, :L] = data["prev"][s][:, 4:6]
            b_ctx[bi, :L] = data["ctx"][s]
            b_ocid[bi, :L] = opt_cidx[s]
            if _HAS_TE:
                b_otc[bi, :L] = opt_tcidx[s]
            b_oty[bi, :L] = opt_type_idx[s]
            b_oar[bi, :L] = opt_area_idx[s]
            b_ovec[bi, :L] = data["opt_vec"][s]
            b_mask[bi, :L] = mask[s]
            if has_cls:
                b_ocls[bi, :L] = data["opt_cls"][s]
            ch = data["chosen"][s]
            lab = np.where(ch.any(1), ch.argmax(1), -100)
            b_lab[bi, :L] = lab
            b_valid[bi, :L] = ch.any(1)
            b_multi[bi, :L] = ch
            b_k[bi, :L] = ch.sum(1)
            b_dw[bi, :L] = dw_all[s]
            if gres_of is not None:
                b_z[bi, :L] = gres_of[data["gidx"][s]]
        t = lambda a, d=None: torch.as_tensor(a, device=dev)  # noqa: E731
        return (t(b_board), t(b_logs), t(b_prevc), t(b_prevn), t(b_ctx),
                t(b_ocid), t(b_ovec), t(b_mask), t(b_lab), t(b_valid),
                t(b_ocls), t(b_multi), t(b_k), t(b_z), t(b_dw), t(b_poc),
                t(b_pon), t(b_otc), t(b_oty), t(b_oar))

    def run_epoch(seq_list, train: bool, epoch: int = 0):
        net.train(train)
        idx = np.random.permutation(len(seq_list)) if train else np.arange(len(seq_list))
        tot_loss, tot_n, tot_ok = 0.0, 0, 0
        ctx_ok, ctx_n = {}, {}
        v_all, z_all = [], []
        for s in range(0, len(idx), args.batch):
            bs = [seq_list[i] for i in idx[s:s + args.batch]]
            (bb, bl, bpc, bpn, bc, boc, bov, bm, blab, bv,
             bocls, bmulti, bk, bz, bdw, bpoc, bpon, botc,
             boty, boar) = make_batch(bs)
            # **dw=0 の決定は損失にも評価指標にも入れない**。系列の一部として
            # LSTMに状態を運ばせるためだけに存在する。
            bvw = bv & (bdw > 0)
            with torch.set_grad_enabled(train):
                scores, _, _cl, vpred = net(bb, bl, bpc, bpn, bc, boc, bov, bm,
                                            opt_cls=bocls if has_cls else None,
                                            prevo_cidx=bpoc, prevo_num=bpon,
                                            opt_tcid=botc, opt_type=boty,
                                            opt_area=boar)
                # 単一選択はCE。複数選択（全体の6%、DISCARDは平均7.3枚）は
                # 「最初の1枚」に潰さず、選んだk枚すべてを正例とする（listwise）。
                logp = torch.log_softmax(scores.masked_fill(~bm, -1e9), dim=-1)
                multi = bk > 1
                ce = F.cross_entropy(scores.reshape(-1, O), blab.reshape(-1),
                                     ignore_index=-100, reduction="none"
                                     ).reshape(blab.shape)
                lw = -(logp * bmulti).sum(-1) / bk.clamp(min=1)
                per = torch.where(multi, lw, ce)
                w = bvw.float() * bdw
                # **focal loss**（EXP-111 軸C）: 正解の予測確率が高い決定＝簡単な決定の
                # 重みを下げ、間違える決定に容量を割く。
                # 強制手（選択肢1個）は既に CE=0 で勾配ゼロなので、対象は
                # 「複数選択肢があるのに当てられない決定」に絞られる。
                # γ=0 で従来と厳密に一致する。
                _fg = float(getattr(args, "focal_gamma", 0.0) or 0.0)
                if _fg > 0:
                    with torch.no_grad():
                        _p = torch.exp(-per.detach()).clamp(0.0, 1.0)
                    w = w * (1.0 - _p) ** _fg
                use_awr = (awr_beta > 0 and train
                           and epoch >= int(getattr(args, "awr_warmup", 10)))
                if use_awr:
                    w = w * _awr_weights(vpred.detach(), bz, bv)
                loss = (per * w).sum() / w.sum().clamp(min=1e-6)
                vloss = torch.zeros((), device=loss.device)
                if v_w > 0:
                    # 勝敗ラベル z は決定ごとに同じ値（ゲーム単位）。有効な決定だけで平均
                    vloss = (((vpred - bz) ** 2) * bv).sum() / bv.sum().clamp(min=1)
                    loss = loss + v_w * vloss
            if train:
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                opt.step()
            with torch.no_grad():
                # **複数選択は「上位k枚の集合一致」で採点する**。
                # argmax1枚が正解集合に入るかで測ると、k/nが大きいctxでchanceが跳ね上がり
                # （ctx34は常に1.000、ctx9 0.847、ctx8 0.667）指標が甘くなる（穴B-4）。
                pred = scores.argmax(-1)
                hit1 = torch.gather(bmulti, 2, pred.unsqueeze(-1)).squeeze(-1) > 0
                kk = bk.long().clamp(min=1)
                topk = scores.topk(min(O, int(kk.max().item()) if kk.numel() else 1),
                                   dim=-1).indices
                rank = torch.zeros_like(bmulti)
                rank.scatter_(2, topk, 1.0)
                # 上位k枚だけを1にする（kは決定ごとに異なるので順位でマスク）
                ar = torch.arange(topk.shape[-1], device=scores.device)
                sel_mask = (ar[None, None, :] < kk.unsqueeze(-1)).float()
                rank = torch.zeros_like(bmulti).scatter_(
                    2, topk, sel_mask)
                inter = (rank * bmulti).sum(-1)
                exact = inter >= bk                      # 集合として完全一致
                okm = torch.where(bk > 1, exact, hit1) & bvw
                tot_ok += int(okm.sum())
                tot_n += int(bvw.sum())
                tot_loss += float(loss) * int(bvw.sum())
                if not train and v_w > 0:
                    v_all.append(vpred[bv].detach().cpu().numpy())
                    z_all.append(bz[bv].detach().cpu().numpy())
                if not train:
                    for c in torch.unique(bc[bvw]):
                        m = bvw & (bc == c)
                        ci = int(c)
                        ctx_ok[ci] = ctx_ok.get(ci, 0) + int((okm & m).sum())
                        ctx_n[ci] = ctx_n.get(ci, 0) + int(m.sum())
                # **クラス頭が同点になる決定**（EXP-139の狙い所）だけの一致率。
                # cls_head は「行動タイプ×対象バケット」しか見ないので、同一クラスに
                # 2個以上の選択肢がある決定では**pointer内積が勝負を決める**。
                # 対象カード埋め込みが効くならここが先に動く。全体valでは
                # 該当が MAIN決定の 22.1%（全決定では約14%）なので差が埋もれる。
                if not train and has_cls:
                    _cnt = torch.zeros(bocls.shape[0], bocls.shape[1], n_cls,
                                       device=scores.device)
                    _cnt.scatter_add_(2, bocls.clamp(min=0),
                                      (bm & (bocls >= 0)).float())
                    _tie = (_cnt >= 2).any(-1) & bvw
                    ctx_ok["_tie_ok"] = ctx_ok.get("_tie_ok", 0) + int((okm & _tie).sum())
                    ctx_ok["_tie_n"] = ctx_ok.get("_tie_n", 0) + int(_tie.sum())
                    # **教師が選んだ手のクラスが同点だった決定**。上の _tie は
                    # 「決定内のどこかに同点クラスがある」なのでMAINの7割が該当して薄まる。
                    # こちらは**同点の解消そのものが正解を決めた**決定に絞る。
                    _lc = torch.gather(bocls, 2, blab.clamp(min=0).unsqueeze(-1)
                                       ).squeeze(-1)
                    _cc = torch.gather(_cnt, 2, _lc.clamp(min=0).unsqueeze(-1)
                                       ).squeeze(-1)
                    _tiec = (_lc >= 0) & (_cc >= 2) & bvw
                    ctx_ok["_tiec_ok"] = ctx_ok.get("_tiec_ok", 0) + int((okm & _tiec).sum())
                    ctx_ok["_tiec_n"] = ctx_ok.get("_tiec_n", 0) + int(_tiec.sum())
        if not train and v_w > 0 and v_all:
            import numpy as _np
            _v = _np.concatenate(v_all)
            _z = _np.concatenate(z_all)
            m1, m0 = _z > 0, _z < 0
            if m1.any() and m0.any():
                # AUC = ランダムに選んだ勝ちと負けで、勝ちの方が高い予測になる確率
                r = _np.argsort(_np.argsort(_v))
                n1_, n0_ = int(m1.sum()), int(m0.sum())
                auc = (r[m1].sum() - n1_ * (n1_ - 1) / 2) / (n1_ * n0_)
                ctx_ok["_v_auc"] = float(auc)
        return tot_loss / max(1, tot_n), tot_ok / max(1, tot_n), ctx_ok, ctx_n

    # **学習せずに保持データの指標だけを出す**（--eval-only、--init-from と併用）。
    # 分割・標準化・語彙は学習時と同一コードを通るので、手で再現する必要がない。
    if getattr(args, "eval_only", False):
        vl, va, _cok, _cn = run_epoch(val_seqs, False, 0)
        vctx = {c: (_cok[c], _cn[c]) for c in _cn}
        for _k in ("_tie_ok", "_tie_n", "_tiec_ok", "_tiec_n"):
            vctx[_k] = _cok.get(_k, 0)
        print(f"\n=== eval-only: {getattr(args,'init_from',None)} ===")
        print(f"val loss {vl:.4f} / val acc **{va:.4f}**")
        _m = vctx.get(0, (0, 1))
        print(f"MAIN一致率 {_m[0]}/{_m[1]} = **{_m[0]/max(1,_m[1]):.4f}**")
        _to, _tn = vctx.get("_tie_ok", 0), vctx.get("_tie_n", 0)
        if _tn:
            print(f"クラス同点を含む決定   {_to}/{_tn} = **{_to/_tn:.4f}**")
        _co, _cnn = vctx.get("_tiec_ok", 0), vctx.get("_tiec_n", 0)
        if _cnn:
            print(f"**正解クラスが同点の決定** {_co}/{_cnn} = **{_co/_cnn:.4f}**  ← EXP-139の狙い所")
        for c in sorted(k for k in vctx if isinstance(k, int)):
            ok, n = vctx[c]
            print(f"  ctx{c:>3}: {ok}/{n} = {ok/max(1,n):.4f}")
        return va, vctx

    best = -1.0
    best_auc = -1.0
    best_ctx = {}
    bad = 0
    # step数を揃える指定があれば、エポック数をデータ量から逆算する
    _n_ep = args.epochs
    _spe = 0
    if getattr(args, "max_steps", 0) > 0:
        _spe = max(1, (len(tr_seqs) + args.batch - 1) // args.batch)
        _n_ep = max(1, int(round(args.max_steps / _spe)))
        print(f"**step数を固定**: {args.max_steps} step / {_spe} step-per-epoch "
              f"→ {_n_ep} エポック（訓練系列 {len(tr_seqs)}）", flush=True)
        if sched is not None:
            sched.T_max = _n_ep
    for ep in range(_n_ep):
        trl, tra, _, _ = run_epoch(tr_seqs, True, ep)
        vll, vla, cok, cn = run_epoch(val_seqs, False, ep)
        _lo = _ln = float("nan")
        if val_old and val_new:
            _lo = run_epoch(val_old, False, ep)[0]
            _ln = run_epoch(val_new, False, ep)[0]
        if sched is not None:
            sched.step()
        _auc = cok.get("_v_auc")
        if _auc is not None:
            best_auc = max(best_auc, _auc)
            if ep % 5 == 4 or ep == _n_ep - 1:
                print(f"       value AUC={_auc:.4f} (best {best_auc:.4f})", flush=True)
        if True:      # **固定step運用では全エポックを出す**（EXP-114）
            _sp = f" | 元{_lo:.4f} 新規{_ln:.4f}" if _lo == _lo else ""
        # **節目のstepでチェックポイントを保存**（EXP-114）。
        # val では最適点が決まらない（元データは劣化の検知器、新規は単調に下がる）ので、
        # 「117からどれだけ動かすか」を**対戦で**選べるようにする。
        if _spe:
            _done = (ep + 1) * _spe
            for _ms in (200, 400, 800, 1600):
                if _done - _spe < _ms <= _done:
                    _cp = args.out.replace(".npz", f"_step{_ms}.npz")
                    save(net, vocab, data, _cp, norm)
                    print(f"  checkpoint step~{_ms} -> {os.path.basename(_cp)}", flush=True)
            print(f"ep{ep:03d} step{(ep + 1) * _spe if getattr(args, 'max_steps', 0) else 0:>4} "
                  f"train {trl:.4f}/{tra:.4f} | val {vll:.4f}/{vla:.4f}{_sp}"
                  f"{'  *' if vla > best else ''}", flush=True)
        # **停止基準**: accuracy は argmax の一致率なので、SILのように「確率だけが動く」
        # 変更にはほぼ反応しない（実測: 117 vs 自分のリプレイで argmax一致 0.9995 だが
        # 教師手の確率は 0.849）。loss は −log p を直接測るので変化が見える。
        _key = (-vll if getattr(args, "stop_metric", "acc") == "loss"
                else (_auc if (getattr(args, "freeze_policy", False) and _auc) else vla))
        if _key > best:
            best = _key
            best_ctx = {c: (cok[c], cn[c]) for c in cn}
            best_ctx["_gap"] = tra - vla        # best時点の過学習量
            save(net, vocab, data, args.out, norm)
            bad = 0
        else:
            bad += 1
            # **--max-steps 指定時は早期終了しない**（EXP-114）。
            # 「step数を揃えて、その中の最良を採る」設計なので、
            # patienceで打ち切ると条件ごとにstep数が変わり比較が崩れる。
            # bestは別途保存済みなので、最後まで回しても最良点は失われない。
            if getattr(args, "max_steps", 0) > 0:
                pass
            elif bad >= getattr(args, "patience", 15):
                print(f"  early stop (patience {args.patience}, best {best:.4f})")
                break
    if getattr(args, "save_final", False):
        _fin = args.out.replace(".npz", "_final.npz")
        save(net, vocab, data, _fin, norm)
        print(f"最終step時点を保存 -> {_fin}")
    print("\nctx別 val一致率（best時点）:")
    for c in sorted(k for k in best_ctx if isinstance(k, int)):
        ok, n = best_ctx[c]
        print(f"  ctx{c:>3}: {ok}/{n} = {ok/max(1,n):.4f}")
    print("best val acc:", best)
    _to, _tn = best_ctx.get("_tie_ok", 0), best_ctx.get("_tie_n", 0)
    if _tn:
        print(f"**クラス同点決定の一致率**: {_to}/{_tn} = {_to/_tn:.4f}"
              f"（val決定の {_tn/max(1,sum(n for k,(o,n) in best_ctx.items() if isinstance(k,int))):.1%}）")
    if best_auc > 0:
        print(f"best value AUC: {best_auc:.4f}")
    return best, best_ctx


def save(net, vocab, data, out, norm=None):
    """全重みを純numpy推論用にフラット保存。

    **特徴の次元も一緒に保存する**。推論側で突合しないと、特徴を変えたのに古い重みを
    使っている状態が「errors=0で動くが毎決定フォールバック」に無言劣化する（穴B-1）。
    """
    sd = {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}
    sd["opt_attn"] = np.int8(1 if getattr(net, "opt_attn", False) else 0)
    # **対象カード埋め込みの有無**（EXP-139）。opt_enc.0 の入力次元だけでも判別できるが、
    # 推論側が黙って別の並びで連結しないよう明示フラグを持たせる。
    sd["target_emb"] = np.int8(1 if getattr(net, "target_emb", False) else 0)
    keys = sorted(vocab)
    np.savez_compressed(
        out,
        vocab_keys=np.array(keys, np.int64),
        vocab_vals=np.array([vocab[k] for k in keys], np.int64),
        n_board=np.int32(data["board"].shape[1]),
        n_logs=np.int32(data["logs"].shape[1]),
        n_optdim=np.int32(data["opt_vec"].shape[2]),
        # **特徴キーの並びそのもの**も保存する。次元数が同じまま順序だけ変わる変更
        # （キーのリネーム等）は長さチェックを素通りして無言で壊れるため。
        # **object dtypeにしない**（読み込みにpickleが必要になり、提出先のnumpyが
        # ローカルより古いと壊れうる）。固定長unicodeならバージョン非依存で読める。
        feat_keys=np.array([str(x) for x in data.get("feat_keys", [])], dtype="U"),
        # **行動クラスの名前列**も同様に保存する。クラス数が同じでも名前が変われば
        # sorted順が変わり、分類ヘッドのIDが全体的にずれる（EXP-096で実際に踏んだ:
        # カード名の短縮規則を変えただけで use_boss_order→use_orders となりID崩壊）。
        classes=np.array([str(x) for x in data.get("classes", [])], dtype="U"),
        # **学習に現れたctxの集合**。未学習ctxの埋め込みは乱数初期値のままなので、
        # 推論側はそれを検出してフォールバックする（LSTMも進めない＝状態を汚さない）。
        trained_ctx=np.array(sorted(set(int(c) for c in np.unique(data["ctx"]))),
                             np.int32),
        # **MAIN階層の合成形**。推論側が違う形で合成すると例外を出さずに別方策になる
        head=np.array(str(getattr(net, "head", "additive")), dtype="U"),
        # **特徴の標準化統計**。推論側で同じ変換を掛けないと方策が壊れる
        # （例外は出ないので無言劣化になる）。nn_*.py が npz から読んで適用する。
        **(norm or {}),
        **sd)


if __name__ == "__main__":
    main()
