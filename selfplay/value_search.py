"""価値誘導探索（I-109 後段）。方策の上位候補を展開し、葉を観測のみVで評価する。

**なぜこれが残る道か**: 094は 2ms/手 = 0.22秒/ゲームで、上限600秒の **0.04%** しか
使っていない（LB上位のYushin 81f1は1.58秒/手 = 174秒/ゲーム）。**許された計算の
1000分の1しか使っていない**。模倣の天井を超える手段として、リーグRLが行き詰まった今、
未使用の推論時計算がいちばん大きい未開拓資源。

**EXP-053で探索が5連敗した2つの死因と、その回避**:

| 死因 | 回避 |
|---|---|
| 決定化した相手が受動的で「手番を渡す」価値を過大評価 | **ロールアウトしない**。自ターン内だけ展開し、葉をVで評価する（相手の手を模擬しない） |
| 真のコミット差±2〜5%WRがN=8サンプルの解像度外 | **Vは連続値**なので1サンプルで差が出る |

**設計上の注意（実装前に潰した点）**:
- Vは**観測のみ**で学習したものを使う（提出物では `GetHiddenData` が使えない）
- 葉は「自ターン終了時点の盤面」で、Vの学習分布（自分の決定時点）とずれる。
  **この分布ずれは実測で確認する**必要がある（`--check-drift`）
- 時間予算は1手と1ゲームの二重管理。超過時は素の方策へ即フォールバック（084と同型）
- `search_begin`/`search_step` が本番で動く保証は無い（CLAUDE.md）。
  **例外が出たら黙って素の方策に落ちる**構造にする
"""

from __future__ import annotations

import time

import numpy as np


class ValueSearch:
    """方策の上位k候補を1ply展開し、葉をVで評価して選ぶ。

    policy_scores: (obs) -> np.ndarray  各選択肢のスコア（素の方策）
    eval_leaf:     (obs) -> float       葉の盤面の勝率（観測のみV）
    """

    def __init__(self, policy_scores, eval_leaf, *, k=4, move_budget=1.0,
                 game_budget=450.0, min_margin=0.0, blend=0.0):
        self.policy_scores = policy_scores
        self.eval_leaf = eval_leaf
        self.k = k
        self.move_budget = move_budget
        self.game_budget = game_budget
        # **方策スコアとVの併用**（piKL型）。blend=0 なら純粋にV最大、
        # 大きくすると方策から離れにくくなる。模倣方策は既に強いので、
        # Vの誤差で大きく外すリスクを抑える保険。
        self.blend = blend
        # Vの差がこの値未満なら方策の1位を採る（**Vのノイズで無意味に手を変えない**）
        self.min_margin = min_margin
        self.used = 0.0
        self.stats = {"calls": 0, "searched": 0, "changed": 0, "timeout": 0,
                      "error": 0, "sec": 0.0}

    def new_game(self):
        self.used = 0.0

    def choose(self, obs, search_begin, search_step, search_end, determinize):
        """探索して選択肢indexを返す。使えないときは None（呼び出し側が素の方策へ）。"""
        sel = obs.select
        self.stats["calls"] += 1
        if sel is None or sel.minCount != 1 or sel.maxCount != 1:
            return None
        n = len(sel.option)
        if n < 2:
            return None
        if self.used > self.game_budget:
            self.stats["timeout"] += 1
            return None

        t0 = time.perf_counter()
        deadline = t0 + self.move_budget
        try:
            sc = np.asarray(self.policy_scores(obs), np.float64)
            order = list(np.argsort(-sc))[:self.k]
            if len(order) < 2:
                return None
            root = search_begin(obs, *determinize(obs))
            vals = {}
            for idx in order:
                if time.perf_counter() > deadline:
                    break
                child = search_step(root.searchId, [int(idx)])
                # **自ターン内だけ進める**。相手の手は模擬しない（EXP-053の死因1）。
                leaf = self._advance_own_turn(child, search_step, deadline)
                vals[int(idx)] = float(self.eval_leaf(leaf))
            self.stats["searched"] += 1
        except Exception:
            self.stats["error"] += 1
            return None
        finally:
            try:
                search_end()
            except Exception:
                pass
            dt = time.perf_counter() - t0
            self.used += dt
            self.stats["sec"] += dt

        if len(vals) < 2:
            return None
        base = int(order[0])
        # 方策スコアを混ぜる（blend=0 なら純V）
        rank = {int(i): float(sc[int(i)]) for i in order}
        mx = max(rank.values()) or 1.0
        score = {i: v + self.blend * (rank[i] / abs(mx)) for i, v in vals.items()}
        best = max(score, key=score.get)
        if score[best] - score.get(base, -1e9) < self.min_margin:
            return base
        if best != base:
            self.stats["changed"] += 1
        return best

    @staticmethod
    def _advance_own_turn(state, search_step, deadline, cap=40):
        """自分のターンが終わるまで（または上限まで）決定を進める。

        **相手の手番に入ったら止める**。ここから先は相手の方策が要るが、それを
        決定化で模擬するとEXP-053の死因1（受動的な相手）を踏む。
        """
        cur = state
        for _ in range(cap):
            if time.perf_counter() > deadline:
                break
            sel = getattr(cur, "select", None)
            if sel is None or not getattr(sel, "option", None):
                break
            # 相手の手番になったら終了（葉はここ）
            if getattr(cur.current, "yourIndex", None) != getattr(
                    state.current, "yourIndex", None):
                break
            k = max(min(sel.maxCount or sel.minCount, len(sel.option)), sel.minCount)
            cur = search_step(cur.searchId, list(range(max(1, k))))
        return cur

    def summary(self):
        s = self.stats
        c = max(1, s["calls"])
        return (f"探索: 呼出{s['calls']} 実行{s['searched']}({s['searched']/c:.0%}) "
                f"手変更{s['changed']} 予算切れ{s['timeout']} 例外{s['error']} "
                f"総{s['sec']:.1f}秒 平均{s['sec']/max(1,s['searched'])*1000:.0f}ms/探索")
