"""Critic用の対称な状態符号化（I-109 Stage 3）。

**設計の意図**: これまでCriticの相手側入力は `ft_main.feat_vector(state, 1-me)` ＝
**Actorの特徴器（＝我々のデッキのID体系）を相手に当てたもの**だった。相手が同じデッキの
ときだけ偶然機能する形で、実測ではミラーで158次元中121が有効、対Marnieでは70しか有効でない
（EXP-096）。病気は「Actorの特徴器をStateの符号化器に流用したこと」そのもの。

ここでは **両プレイヤーを同じ関数 E(p) で符号化する**。V(s) は状態の勝率であって視点を
持つ理由がないので、これが定義通りの形になる。

    E(p) = [ ゾーンごとのカードID列（手札 / 山∪サイド / 場 / トラッシュ）]
         + [ 汎用属性（枚数・HP・ダメージ・エネ数・状態異常など）]

カードは**IDをそのまま埋め込み索引に使う**ので、語彙も和集合も要らない。
リーグにデッキを足しても次元は変わらず、mainを差し替えてもCriticは作り直し不要。

隠れゾーン（手札・山・サイドの中身）は `GetHiddenData`（cg/Export_seeded.cpp、
副作用なし・CRN決定性維持を検証済み）で埋める。Criticは訓練専用で提出物に載らないため、
特権情報を使っても本番との乖離は生じない。方策勾配は不偏に保たれる
（π(a|o) は o にしか依存せず、隠れ状態 s は o の下で a と条件付き独立）。
"""

from __future__ import annotations

import ctypes
import json
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from cg.sim import HAS_HIDDEN_DATA, lib  # noqa: E402

# ゾーンの並び・次元は selfplay/state_spec.py が唯一の定義元（学習側と共有するため）
from selfplay.state_spec import (CARD_ID_OFFSET, N_ATTR, N_PROG,  # noqa: E402
                                 N_ZONE, UNKNOWN_CARD, ZONE_CAP)


def _hidden(ptr: int, pi: int) -> dict:
    if not HAS_HIDDEN_DATA or not ptr:
        return {}
    sd = lib.GetHiddenData(ctypes.c_void_p(ptr), ctypes.c_int(pi))
    if not sd.json:
        return {}
    return json.loads(sd.json.decode())


def _pack(ids, out):
    """カードID列を固定長スロットへ。0はpadding（埋め込みのpadding_idx=0と対応）。"""
    n = min(len(ids), ZONE_CAP)
    for i in range(n):
        out[i] = int(ids[i]) + CARD_ID_OFFSET   # 0はpadding専用にする
    return out


def _field_ids(ps: dict) -> list[int]:
    """場のカードID（ポケモン本体 + 付いているエネ/ツール/進化元）。"""
    ids = []
    for slot in ("active", "bench"):
        for p in (ps.get(slot) or []):
            if not p:
                continue
            ids.append(p["id"])
            for key in ("energyCards", "tools", "preEvolution"):
                for c in (p.get(key) or []):
                    if c:
                        ids.append(c["id"])
    return ids


def attrs(cur: dict, pi: int) -> list[float]:
    """デッキ非依存の汎用属性（両プレイヤーに同じ関数を当てる）。"""
    ps = cur["players"][pi]
    act = (ps.get("active") or [None])[0]
    bench = [b for b in (ps.get("bench") or []) if b]
    hp = float(act["hp"]) if act else 0.0
    mx = float(act["maxHp"]) if act else 0.0
    return [
        float(ps.get("handCount") or 0),
        float(ps.get("deckCount") or 0),
        float(len(ps.get("prize") or [])),
        float(len(ps.get("discard") or [])),
        float(len(bench)),
        float(ps.get("benchMax") or 0),
        hp, mx, (mx - hp),
        (hp / mx) if mx > 0 else 0.0,
        float(len(act.get("energyCards") or [])) if act else 0.0,
        1.0 if ps.get("asleep") else 0.0,
        1.0 if ps.get("paralyzed") else 0.0,
        1.0 if ps.get("confused") else 0.0,
    ]


def _pack_unknown(n: int, out):
    """未知カードをn枚ぶん置く。**ゼロ埋めにしない**（ゼロは「空」を意味し、
    相手の山が空＝山札切れ＝こちらの勝ち という偽の手掛かりになる）。"""
    for i in range(min(n, ZONE_CAP)):
        out[i] = UNKNOWN_CARD
    return out


def encode(ptr: int, obs: dict, privileged: bool = True
           ) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """1決定ぶんの状態を符号化する。

    戻り値:
      zone_idx (2, N_ZONE, ZONE_CAP) int16 — プレイヤー×ゾーン×カード
      attr     (2, N_ATTR) float32
      prog     進行情報（手番・先攻・result前の共通量）
    **視点を持たない**（player 0 / 1 の順で固定）。どちらが自分かは prog に入れる。
    """
    cur = obs["current"]
    zone = np.zeros((2, N_ZONE, ZONE_CAP), np.int16)
    attr = np.zeros((2, N_ATTR), np.float32)
    me = cur.get("yourIndex", 0)
    for pi in (0, 1):
        ps = cur["players"][pi]
        # privileged=False のときは**相手の隠れ領域を「未知カードN枚」で表す**。
        # 提出物では GetHiddenData が使えないので、推論時に使うVはこの条件で学習する。
        if not privileged and pi != me:
            _pack_unknown(int(ps.get("handCount") or 0), zone[pi, 0])
            _pack_unknown(int(ps.get("deckCount") or 0)
                          + len(ps.get("prize") or []), zone[pi, 1])
        else:
            h = _hidden(ptr, pi)
            _pack(h.get("hand") or [c["id"] for c in (ps.get("hand") or [])],
                  zone[pi, 0])
            _pack(list(h.get("deck") or []) + list(h.get("prize") or []),
                  zone[pi, 1])
        _pack(_field_ids(ps), zone[pi, 2])
        _pack([c["id"] for c in (ps.get("discard") or []) if c], zone[pi, 3])
        attr[pi] = attrs(cur, pi)
    stad = cur.get("stadium") or []
    prog = [
        float(cur.get("turn") or 0),
        float(cur.get("yourIndex") or 0),          # いま手番なのはどちらか
        float(cur.get("firstPlayer") or 0),
        1.0 if cur.get("supporterPlayed") else 0.0,
        1.0 if cur.get("energyAttached") else 0.0,
        1.0 if cur.get("retreated") else 0.0,
        1.0 if cur.get("stadiumPlayed") else 0.0,
        float(stad[0]["id"] + CARD_ID_OFFSET) if stad else 0.0,
    ]
    return zone, attr, prog


if __name__ == "__main__":
    import random
    sys.path.insert(0, os.path.join(_ROOT, "arena"))
    import run_match as rm
    import cg.game as cgg
    from cg.game import Battle

    A = os.path.join(_ROOT, "agents/099_alakazam_gen")
    B = os.path.join(_ROOT, "agents/101_marnie_luca")
    a, b = rm.load_agent_module(A, "A"), rm.load_agent_module(B, "B")
    da, db = rm.read_deck(A), rm.read_deck(B)
    random.seed(0)
    obs, _ = cgg.battle_start_seeded(da, db, 4242)
    ptr = Battle.battle_ptr
    n = 0
    tot = np.zeros(2)
    while obs is not None and n < 200:
        if obs["current"]["result"] != -1:
            break
        z, at, pg = encode(ptr, obs)
        n += 1
        tot += [(z[0] > 0).sum(), (z[1] > 0).sum()]
        me = obs["current"]["yourIndex"]
        obs = cgg.battle_select((a if me == 0 else b)(obs))
    cgg.battle_finish()
    print(f"符号化 {n} 回 / 非ゼロカード枠 平均 p0={tot[0]/n:.1f} p1={tot[1]/n:.1f}"
          f"（各プレイヤー最大 {N_ZONE*ZONE_CAP}）")
    print(f"次元: zone {2*N_ZONE*ZONE_CAP} (int16) / attr {2*N_ATTR} / prog {N_PROG}")
