"""複数バトルを1プロセスで同時進行させるベクトル化環境（I-109 Phase RL Stage 0）。

**なぜ作るか**: 自己対戦RLの生成コストの62%はNN forward（1決定2.4ms、EXP-094実測）。
1決定ずつnumpyで回している限りここは減らない。N試合を同時に「次の決定待ち」まで進めて
まとめて1回forwardすれば、この62%がほぼ消える。

**なぜ可能か**: エンジンのC APIは全て battlePtr を引数で受け取る（cg/sim.py の argtypes 参照）。
  lib.Select(ptr, ...) / lib.GetBattleData(ptr) / lib.BattleFinish(ptr)
1プロセス1バトルに縛っているのは cg/game.py の `Battle.battle_ptr`（クラス属性）だけなので、
**エンジンを改造せずに**並行化できる。

**cg/game.py は変更しない**（提出物が通る経路を壊さないため）。ここは自己対戦専用の別経路。

usage:
  uv run python selfplay/vecenv.py --selftest      # 逐次実行と一致するか検証
  uv run python selfplay/vecenv.py --bench -n 64   # スループット計測
"""

from __future__ import annotations

import ctypes
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from cg.sim import HAS_SEEDED_START, lib  # noqa: E402


class VecBattles:
    """N個のバトルを同時に保持し、決定待ちの盤面をまとめて返す。

    使い方:
        vb = VecBattles(decks0, decks1, seeds)
        while vb.live:
            obs = vb.observations()          # [(idx, obs_dict), ...]
            acts = policy_batch(obs)         # {idx: [option index, ...]}
            vb.step(acts)
        results = vb.results                 # [0|1|2(draw)|-1(未完), ...]
    """

    def __init__(self, decks0: list[list[int]], decks1: list[list[int]],
                 seeds: list[int] | None = None):
        n = len(decks0)
        assert len(decks1) == n
        if seeds is not None:
            assert len(seeds) == n
            if not HAS_SEEDED_START:
                raise RuntimeError("libcg に BattleStartSeeded が無い"
                                   "（cg/Export_seeded.cpp でビルドし直す）")
        self.n = n
        self.ptrs: list[int | None] = []
        self.obs: list[dict | None] = []
        self.results: list[int] = [-1] * n
        self.steps: list[int] = [0] * n
        for i in range(n):
            d0, d1 = decks0[i], decks1[i]
            if len(d0) != 60 or len(d1) != 60:
                raise ValueError("デッキは60枚")
            arg = (ctypes.c_int * 120)(*(d0 + d1))
            if seeds is None:
                sd = lib.BattleStart(arg)
            else:
                sd = lib.BattleStartSeeded(arg, ctypes.c_uint(seeds[i] & 0xFFFFFFFF))
            ptr = sd.battlePtr
            if not ptr:
                self.ptrs.append(None)
                self.obs.append(None)
                self.results[i] = 2          # 開始失敗は引き分け扱い
                continue
            self.ptrs.append(ptr)
            self.obs.append(self._read(ptr))
        self._sync_finished()

    # ---- エンジン呼び出し ----

    @staticmethod
    def _read(ptr) -> dict:
        """GetBattleData の薄いラッパ。

        cg/game.py の `_get_battle_data` と違い **search_begin_input を作らない**
        （探索用のバイナリblobを毎決定 ascii デコードするのは自己対戦では純粋な無駄）。
        """
        sd = lib.GetBattleData(ptr)
        return json.loads(sd.json.decode())

    def _sync_finished(self):
        for i, o in enumerate(self.obs):
            if o is None or self.results[i] != -1:
                continue
            r = o["current"]["result"]
            if r != -1:
                self.results[i] = r
                self._close_one(i)

    def _close_one(self, i: int):
        if self.ptrs[i] is not None:
            lib.BattleFinish(self.ptrs[i])
            self.ptrs[i] = None
        self.obs[i] = None

    # ---- 外向きAPI ----

    @property
    def live(self) -> bool:
        return any(p is not None for p in self.ptrs)

    def observations(self) -> list[tuple[int, dict]]:
        """決定待ちのバトルの (index, observation) を全部返す。"""
        return [(i, o) for i, o in enumerate(self.obs) if o is not None]

    def step(self, acts: dict[int, list[int]], step_cap: int = 3000):
        """{index: 選択} をまとめて適用する。不正選択はそのバトルを相手の勝ちで終了。"""
        for i, sel in acts.items():
            ptr = self.ptrs[i]
            if ptr is None:
                continue
            me = self.obs[i]["current"]["yourIndex"]
            arg = (ctypes.c_int * len(sel))(*sel) if sel else (ctypes.c_int * 0)()
            err = lib.Select(ptr, arg, len(sel))
            if err != 0:
                self.results[i] = 1 - me     # 不正選択は即負け（arenaと同じ規約）
                self._close_one(i)
                continue
            self.steps[i] += 1
            if self.steps[i] >= step_cap:
                self.results[i] = 2
                self._close_one(i)
                continue
            self.obs[i] = self._read(ptr)
        self._sync_finished()

    def close(self):
        for i in range(self.n):
            self._close_one(i)


def _cli():
    import argparse
    import os
    import sys
    import time

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, root)
    sys.path.insert(0, os.path.join(root, "arena"))

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true",
                    help="逐次実行と結果が一致するか検証（並行実行の安全性の確認）")
    ap.add_argument("--bench", action="store_true", help="スループット計測")
    ap.add_argument("-n", type=int, default=32, help="同時バトル数")
    ap.add_argument("--games", type=int, default=64)
    ap.add_argument("--agent", default="agents/094_yushin_nn")
    args = ap.parse_args()

    import run_match as rm
    deck = rm.read_deck(os.path.join(root, args.agent))
    seeds = [rm.derive_engine_seed(777777, g) for g in range(args.games)]

    def always_first(obs):
        """決定的な参照方策（常に先頭を選ぶ）。エンジンの並行安全性だけを見るため。"""
        sel = obs["select"]
        k = min(sel["maxCount"] or sel["minCount"], len(sel["option"]))
        k = max(k, sel["minCount"])
        return list(range(min(max(0, k), len(sel["option"]))))

    if args.selftest:
        # (1) 1バトルずつ逐次に回す
        seq = []
        for s in seeds:
            vb = VecBattles([deck], [deck], [s])
            while vb.live:
                vb.step({i: always_first(o) for i, o in vb.observations()})
            seq.append((vb.results[0], vb.steps[0]))
            vb.close()
        # (2) N個を同時に回す
        vb = VecBattles([deck] * args.games, [deck] * args.games, seeds)
        while vb.live:
            vb.step({i: always_first(o) for i, o in vb.observations()})
        par = list(zip(vb.results, vb.steps))
        vb.close()
        same = sum(1 for a, b in zip(seq, par) if a == b)
        print(f"逐次 vs 並行({args.games}同時): 勝敗+手数が一致 {same}/{len(seq)}")
        if same != len(seq):
            for j, (a, b) in enumerate(zip(seq, par)):
                if a != b:
                    print(f"  不一致 game{j}: 逐次{a} / 並行{b}")
                    break
            print("→ エンジンにバトル跨ぎの共有状態があり、並行化は不可")
            return 1
        print("→ エンジンは1プロセス複数バトルで安全")
        return 0

    if args.bench:
        t0 = time.perf_counter()
        done = 0
        total_steps = 0
        while done < args.games:
            batch = min(args.n, args.games - done)
            sd = [rm.derive_engine_seed(999999, done + j) for j in range(batch)]
            vb = VecBattles([deck] * batch, [deck] * batch, sd)
            while vb.live:
                vb.step({i: always_first(o) for i, o in vb.observations()})
            total_steps += sum(vb.steps)
            vb.close()
            done += batch
        dt = time.perf_counter() - t0
        print(f"{args.games}試合 / {dt:.1f}秒 = {args.games/dt:.1f}試合/秒 "
              f"({args.games/dt*3600:,.0f}試合/時, 1プロセス)")
        print(f"  総ステップ {total_steps}, {dt/max(1,total_steps)*1000:.3f} ms/決定"
              f"（参照方策のためNN forwardは含まない＝エンジン+JSONの下限）")
        return 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
