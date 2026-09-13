"""117の設計特徴を使う Critic 用の入力ビルダ（EXP-113）。

**なぜ作るか**: リーグの現行 Critic（ValueNetV2）は カードIDの bag-of-embeddings で、
`my_best_dmg` / `ko_opp_active` / `lethal_for_me` といった**検証済みの設計特徴を一切見ない**。
一方 EXP-095 Stage 1 の V1 は「方策と同じ特徴 × 2人分」で **t5以降 AUC 0.857** を実測している。
V2 の AUC は未測定のまま、実績のある表現から未検証の表現に置き換わっていた。

  V1（実測 0.857）: [feat(state, me)] ++ [feat(state, 1-me)]  ← 同じ特徴器を相手indexで呼ぶ
  V2（未測定）    : zone(カードIDの袋) + attr14 + prog8

**異種相手でも動く根拠**: 117の特徴器の相手側ブロック（opp_seen_* / oparch_* / opp_act_hp …）は
任意の相手を符号化できる設計で、実際にアーキタイプを96%識別している。
相手indexで呼んだときデッキ固有の項（left_{cid} 等）は無意味な定数になるが、
構造的な項（HP・打点・KO判定・サイド・エネ）は意味を持つ。

**特権情報**: `GetHiddenData` が返す {hand, deck, prize} を obs に注入してから
feat_vector を呼ぶ。提出物では使えないが、Criticは提出物に含まれないので合法
（AlphaStar の privileged value baseline と同じ）。
"""
from __future__ import annotations

import numpy as np


def inject_hidden(obs: dict, ptr: int, pi: int) -> dict:
    """obs の players[pi] に隠れ領域の実カードを注入した**浅いコピー**を返す。

    元の obs は書き換えない（方策側が同じ dict を見ているため）。
    GetHiddenData が使えない場合は obs をそのまま返す（= 非特権と同じ）。
    """
    from selfplay.state_encoder import _hidden
    h = _hidden(ptr, pi)
    if not h:
        return obs
    cur = dict(obs["current"])
    players = list(cur["players"])
    ps = dict(players[pi])
    # **Card は id/serial/playerIndex の3つが必須**（dataclass）。id だけの dict を入れると
    # to_observation_class が TypeError で落ちる。serial は特徴器が使わないので連番でよい。
    def _cards(ids, base):
        return [{"id": int(c), "serial": -(i + 1), "playerIndex": pi}
                for i, c in enumerate(ids)]
    if h.get("hand"):
        ps["hand"] = _cards(h["hand"], ps.get("hand"))
    if h.get("prize"):
        ps["prize"] = _cards(h["prize"], ps.get("prize"))
    players[pi] = ps
    cur["players"] = players
    out = dict(obs)
    out["current"] = cur
    return out


def opp_board(ft, obs: dict, me: int, ptr: int, privileged: bool,
              to_obs) -> np.ndarray:
    """相手視点の board 特徴（Criticの特権ブロック）。

    privileged=True なら相手の手札・サイドの実カードを注入してから符号化する。
    False なら観測どおり（handCount だけ分かる状態）＝提出物と同じ情報量。
    """
    src = inject_hidden(obs, ptr, 1 - me) if privileged else obs
    st = to_obs(src).current
    b, _ = ft.feat_vector(st, 1 - me)
    return np.asarray(b, np.float32)
