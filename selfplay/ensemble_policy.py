"""複数の学習シードのモデルを推論時に平均する（アンサンブル）。

**なぜ効くか**: 同一教師・同一データで学習シードだけ違うモデルは、実測で
**行動一致率82.7%**（EXP-096）＝6決定に1回は違う手を打つ。互いに独立な誤差を持つので、
スコアを平均すると誤差が打ち消し合う。SL方策の一致率は一般に1〜3pt上がる。

**なぜ今できるか**: 我々は常に3シードのスイープで学習しており、s1/s2/s3 が手元にある。
これまでは最良シードだけ採用して2つを捨てていた。

**コスト**: 1.2ms → 3.6ms/手。**1ゲーム600秒の予算に対して無視できる**
（094は0.22秒/ゲーム＝予算の0.04%しか使っていない）。

**平均するもの**: 各モデルの**スコア**（logit相当）を平均する。確率で平均するより
pointer networkの構造に素直で、温度の扱いも変わらない。
"""

from __future__ import annotations

import numpy as np


class EnsembleLSTMPolicy:
    """BatchLSTMPolicy と同じインタフェースで、複数モデルのスコアを平均する。"""

    def __init__(self, paths: list[str], n_slots: int, weights=None):
        from selfplay.batch_policy import BatchLSTMPolicy
        self.members = [BatchLSTMPolicy(p, n_slots) for p in paths]
        n = len(self.members)
        self.weights = np.ones(n) / n if weights is None else (
            np.asarray(weights, np.float64) / np.sum(weights))
        # 語彙・trained_ctx は同一構成なので先頭のものを代表として持つ
        self.w = self.members[0].w
        self.vocab = self.members[0].vocab
        self.trained_ctx = self.members[0].trained_ctx
        self.n_slots = n_slots

    def reset_all(self):
        for m in self.members:
            m.reset_all()

    def reset(self, slots):
        for m in self.members:
            m.reset(slots)

    def scores(self, slots, boards, logs, prevs, ctxs, opt_cids, opt_vecs, opt_clss):
        acc = None
        for wgt, m in zip(self.weights, self.members):
            sc = m.scores(slots, boards, logs, prevs, ctxs, opt_cids, opt_vecs,
                          opt_clss)
            if acc is None:
                acc = [wgt * np.asarray(s, np.float64) for s in sc]
            else:
                for i, s in enumerate(sc):
                    acc[i] += wgt * np.asarray(s, np.float64)
        return acc


if __name__ == "__main__":
    import os
    import sys
    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for _p in (_ROOT, os.path.join(_ROOT, "arena")):
        if _p not in sys.path:
            sys.path.insert(0, _p)
    import argparse
    import importlib.util as ilu
    import run_match as rm
    from selfplay.batch_policy import BatchLSTMPolicy
    from selfplay.league_rollout import Opponent, collect

    ap = argparse.ArgumentParser(description="アンサンブル vs 単体 を直接対戦で比較")
    ap.add_argument("--agent-dir", default="agents/101_marnie_luca")
    ap.add_argument("--feat", default="feat_101")
    ap.add_argument("--models", nargs="+", required=True, help="npzを複数")
    ap.add_argument("--single", required=True, help="比較対象の単体npz")
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=555001)
    a = ap.parse_args()

    d = os.path.join(_ROOT, a.agent_dir)
    if d not in sys.path:
        sys.path.insert(0, d)
    sp = ilu.spec_from_file_location(a.feat, os.path.join(d, a.feat + ".py"))
    ft = ilu.module_from_spec(sp)
    sys.modules[a.feat] = ft
    sp.loader.exec_module(ft)
    deck = rm.read_deck(d)

    ens = EnsembleLSTMPolicy([os.path.join(d, m) if not os.path.isabs(m) else m
                              for m in a.models], 2 * a.batch)
    one = BatchLSTMPolicy(os.path.join(d, a.single) if not os.path.isabs(a.single)
                          else a.single, a.batch)
    ops = [Opponent("single", "ext", one, ft, deck, 100.0, temp=0.0)]
    _, st = collect(ens, ft, deck, ops, a.games, a.seed, batch=a.batch,
                    temperature=0.0)
    n, w = st["games"], st["wins"]
    p = w / max(1, n)
    se = (p * (1 - p) / max(1, n)) ** 0.5
    print(f"アンサンブル({len(a.models)}モデル) vs 単体: {w}/{n} = {p:.1%} "
          f"[{p-1.96*se:.1%}-{p+1.96*se:.1%}]")
    print("50%を有意に超えればアンサンブルが強い（同一デッキ・同一教師なので純粋な方策差）")
