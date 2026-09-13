"""EXP-015: カード埋め込みBC方策（教師=islet方策のbehavior cloning、純NumPy推論）。"""

from __future__ import annotations

import os
import sys

from cg.api import to_observation_class

# 同ディレクトリのモジュール/重みを解決（Kaggleはexecロードで __file__ が無い）
try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = "/kaggle_simulations/agent"
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import features_015 as F  # noqa: E402
import model_015 as M     # noqa: E402

_deck_path = os.path.join(_HERE, "deck.csv")
if not os.path.exists(_deck_path):
    _deck_path = "deck.csv"
if not os.path.exists(_deck_path):
    _deck_path = "/kaggle_simulations/agent/deck.csv"
with open(_deck_path, "r", encoding="utf-8") as f:
    my_deck = [int(line) for line in f.read().splitlines() if line.strip()]

_weights_path = os.path.join(_HERE, "weights_015.npz")
PARAMS = M.load_params(_weights_path) if os.path.exists(_weights_path) else None


def agent(obs_dict: dict) -> list[int]:
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        return my_deck

    select = obs.select
    n = len(select.option)
    if n == 0 or select.maxCount == 0:
        return []
    if PARAMS is None or n == 1:
        return list(range(n))[: select.maxCount]  # エンジン順フォールバック

    try:
        ctx, opts = F.extract(obs)
        X, mask, _ = M.build_batch(PARAMS, [ctx], [opts], pad_to=min(len(opts), M.P_MAX))
        logits, _ = M.forward(PARAMS, X, mask)
        order = list(map(int, logits[0, :len(opts)].argsort()[::-1]))
        # P_MAX超過分（未スコア）は末尾にエンジン順で付ける
        order += [i for i in range(n) if i not in order]
        return order[: select.maxCount]
    except Exception:
        return list(range(n))[: select.maxCount]
