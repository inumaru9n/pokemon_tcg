"""Critic（privileged value baseline）用の相手側特権特徴（I-109 Stage 3）。

**なぜ必要か**: これまでCriticの相手側ブロックは `ft.feat_vector(state, 1-me)` ＝
**手番プレイヤーの観測に我々の特徴器を当てただけ**だった。エンジンの観測は視点付きで
相手の手札が `null` なので、**真の特権情報は最初から入っていなかった**
（EXP-095の「完全情報critic」という記述は誤り。EXP-096で訂正）。
さらに相手のデッキが違うと、我々のカードID体系に依存する特徴（hand_/dis_/pool_/種族one-hot）
が全滅する。実測: ミラーでは158次元中121が有効、対Marnieでは70しか有効でない。

**どう取るか**: `cg/Export_seeded.cpp` に追加した `GetHiddenData(ptr, playerIndex)` で
相手の hand / deck / prize の cardId 列を直接取得する（副作用なし・CRN決定性維持を検証済み）。
Criticは訓練専用で提出物に載らないため、この情報を使っても本番との乖離は生じない。

**理論的な正当性**: 方策 π(a|o) は観測 o にしか依存せず、隠れ状態 s は o の下で a と
条件付き独立なので E[∇log π(a|o)·b(s)] = E[∇log π(a|o)]·E[b(s)|o] = 0。
すなわち**特権ベースラインは方策勾配を不偏に保つ**（MAPPO/COMA/AlphaStarと同じ根拠）。

usage:
    from selfplay.privileged import PrivilegedReader, UNION_IDS
    pr = PrivilegedReader([deck_main, deck_101, deck_098])
    vec = pr.block(battle_ptr, opp_index)      # list[float]
"""

from __future__ import annotations

import ctypes
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from cg.sim import HAS_HIDDEN_DATA as _HAS_HIDDEN  # noqa: E402
from cg.sim import lib  # noqa: E402  シグネチャ登録は cg/sim.py に一元化


def read_deck_ids(agent_dir: str) -> list[int]:
    path = os.path.join(agent_dir, "deck.csv")
    with open(path) as f:
        return [int(x) for x in f.read().split()]


class PrivilegedReader:
    """リーグ登場デッキの和集合IDで相手の隠れ領域を符号化する。

    Criticの入力次元を固定するため、**和集合は最初に確定させる**
    （後からデッキを足すと次元が変わり、学習済みCriticが使えなくなる）。
    """

    def __init__(self, decks: list[list[int]]):
        ids = set()
        for d in decks:
            ids |= set(d)
        self.union_ids = sorted(ids)
        self.index = {cid: i for i, cid in enumerate(self.union_ids)}
        self.n_ids = len(self.union_ids)
        # hand / deck+prize（＝相手が今後引ける集合）/ 枚数3つ
        self.dim = self.n_ids * 2 + 3
        self.available = _HAS_HIDDEN
        self._unknown: dict[int, int] = {}

    def _counts(self, cards: list[int]) -> list[float]:
        v = [0.0] * self.n_ids
        for cid in cards:
            k = self.index.get(int(cid))
            if k is not None:
                v[k] += 1.0
            else:
                # **和集合外のIDは無言で落ちる**。リーグにデッキを足したのに
                # PrivilegedReaderを作り直していない状態がこれで、Criticの入力が
                # 静かに欠落する（次元エラーにならないので気付けない）。必ず可視化する。
                self._unknown[int(cid)] = self._unknown.get(int(cid), 0) + 1
                if len(self._unknown) <= 5 and self._unknown[int(cid)] == 1:
                    print(f"[privileged] 警告: 和集合外のcardId {cid} を無視した。"
                          f"リーグ構成が変わったなら PrivilegedReader を作り直すこと",
                          file=sys.stderr)
        return v

    @property
    def unknown_ids(self) -> dict[int, int]:
        """和集合外で落としたIDと回数（学習後の健全性チェック用）。"""
        return dict(self._unknown)

    def block(self, battle_ptr: int, opp_index: int) -> list[float]:
        """相手の (手札構成, 山∪サイド構成, 枚数3) を並べたベクトル。

        libcgに GetHiddenData が無い場合はゼロベクトルを返す（配布dylibでも落ちない）。
        """
        if not _HAS_HIDDEN or not battle_ptr:
            return [0.0] * self.dim
        sd = lib.GetHiddenData(ctypes.c_void_p(battle_ptr), ctypes.c_int(opp_index))
        if not sd.json:
            return [0.0] * self.dim
        d = json.loads(sd.json.decode())
        hand = d.get("hand") or []
        deck = d.get("deck") or []
        prize = d.get("prize") or []
        # **山とサイドは合わせて1つの分布にする**。どちらも相手が今後引きうる
        # 未公開の集合であり、価値評価上は「どちらにあるか」より「残っているか」が効く。
        return (self._counts(hand) + self._counts(list(deck) + list(prize))
                + [float(len(hand)), float(len(deck)), float(len(prize))])


def build_reader(agent_dirs: list[str]) -> PrivilegedReader:
    return PrivilegedReader([read_deck_ids(a) for a in agent_dirs])


if __name__ == "__main__":
    import random
    sys.path.insert(0, os.path.join(_ROOT, "arena"))
    import run_match as rm
    import cg.game as cgg
    from cg.game import Battle

    dirs = [os.path.join(_ROOT, a) for a in
            ("agents/099_alakazam_gen", "agents/101_marnie_luca",
             "agents/098_spidops_gen")]
    pr = build_reader(dirs)
    print(f"和集合 {pr.n_ids}種 / 特権ブロック {pr.dim}次元 / "
          f"GetHiddenData={'あり' if pr.available else 'なし'}")

    a = rm.load_agent_module(dirs[0], "A")
    b = rm.load_agent_module(dirs[1], "B")
    da, db = rm.read_deck(dirs[0]), rm.read_deck(dirs[1])
    random.seed(0)
    obs, _ = cgg.battle_start_seeded(da, db, 4242)
    ptr = Battle.battle_ptr
    n = nz = 0
    while obs is not None and n < 400:
        if obs["current"]["result"] != -1:
            break
        me = obs["current"]["yourIndex"]
        if me == 0:
            v = pr.block(ptr, 1)
            n += 1
            nz += sum(1 for x in v if x != 0.0)
        obs = cgg.battle_select((a if me == 0 else b)(obs))
    cgg.battle_finish()
    print(f"我々の決定 {n} 回 / 特権ブロックの平均非ゼロ次元 {nz/max(1,n):.1f}"
          f" / {pr.dim}")
