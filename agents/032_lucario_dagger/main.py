"""EXP-032: 007のDAgger蒸留エージェント = カード埋め込みNN方策 + lethal限定探索。

EXP-031（plain BC）の分布シフト対策として、生徒がプレイ→訪れた状態を教師007が
ラベル→追加して再学習、を反復（DAgger）。main.pyの構造は031と同一。

二層構造は007と同一:
  - 通常の全判断: NN（007の全挙動を教師に蒸留した方策）の argmax
  - 自分の残りサイド<=3のMAIN選択のみ: 決定化lethal探索（全サンプル勝利のみ上書き）
lethal層の候補ランキングとロールアウトもNNで行う（純NN+探索の自己完結構成）。
環境変数 DISTILL_LETHAL=0 で探索を無効化できる（M2評価: 純NN vs 004用）。
"""

from __future__ import annotations

import os
import random
import sys
import time
from collections import Counter

from cg.api import (
    Observation,
    SelectContext,
    all_card_data,
    to_observation_class,
    search_begin,
    search_step,
    search_end,
)

try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = "/kaggle_simulations/agent"
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import features_032 as F  # noqa: E402
import model_032 as M     # noqa: E402

_deck_path = os.path.join(_HERE, "deck.csv")
if not os.path.exists(_deck_path):
    _deck_path = "/kaggle_simulations/agent/deck.csv"
with open(_deck_path, "r", encoding="utf-8") as f:
    my_deck = [int(line) for line in f.read().splitlines() if line.strip()]

_weights_path = os.path.join(_HERE, "weights_032.npz")
PARAMS = M.load_params(_weights_path) if os.path.exists(_weights_path) else None

USE_LETHAL = os.environ.get("DISTILL_LETHAL", "1") != "0"

all_card = all_card_data()
card_table = {c.cardId: c for c in all_card}
my_deck_counts = Counter(my_deck)

BASIC_FIGHTING_ENERGY = 6


def _nn_logits(obs: Observation):
    """select中の全選択肢のNNスコアを返す（失敗時None）。"""
    if PARAMS is None:
        return None
    opts = obs.select.option
    if not (2 <= len(opts) <= M.P_MAX):
        return None
    try:
        ctx, toks = F.extract(obs)
        X, mask, _ = M.build_batch(PARAMS, [ctx], [toks], pad_to=len(toks))
        logits, _ = M.forward(PARAMS, X, mask)
        return logits[0, : len(toks)]
    except Exception:
        return None


def _nn_choose(obs: Observation) -> list[int]:
    select = obs.select
    n = len(select.option)
    if n == 0 or select.maxCount == 0:
        return []
    logits = _nn_logits(obs)
    if logits is None:
        order = list(range(n))
    else:
        order = list(map(int, logits.argsort()[::-1]))
        order += [i for i in range(n) if i not in order]
    return order[: select.maxCount]


# ---------------------------------------------------------------------------
# lethal限定探索（007と同一の枠組み。ランキング/ロールアウトはNN）
# ---------------------------------------------------------------------------

SEARCH_CANDIDATES = 8
SEARCH_SAMPLES = 3
SEARCH_MOVE_BUDGET = 1.5
SEARCH_GAME_BUDGET = 450.0
ROLLOUT_STEP_CAP = 80

PLACEHOLDER_MON = 1072
if not (card_table.get(PLACEHOLDER_MON) and card_table[PLACEHOLDER_MON].basic):
    PLACEHOLDER_MON = next(c.cardId for c in all_card if c.basic)

_search_time_used = 0.0
_pre_turn = -1


def _fallback_selection(select) -> list[int]:
    count = max(select.minCount, min(1, select.maxCount))
    return list(range(count))


def _my_visible_counts(obs: Observation) -> Counter:
    me = obs.current.players[obs.current.yourIndex]
    seen: Counter = Counter()
    for card in (me.hand or []):
        seen[card.id] += 1
    for card in me.discard:
        seen[card.id] += 1
    for pokemon in me.active + me.bench:
        if pokemon is None:
            continue
        seen[pokemon.id] += 1
        for card in pokemon.preEvolution:
            seen[card.id] += 1
        for card in pokemon.energyCards:
            seen[card.id] += 1
        for card in pokemon.tools:
            seen[card.id] += 1
    for card in obs.current.stadium:
        if seen[card.id] < my_deck_counts[card.id]:
            seen[card.id] += 1
    return seen


def _determinize(obs: Observation):
    state = obs.current
    me = state.players[state.yourIndex]
    op = state.players[1 - state.yourIndex]

    remaining = my_deck_counts - _my_visible_counts(obs)
    unknown = list(remaining.elements())
    random.shuffle(unknown)
    need = me.deckCount + len(me.prize)
    while len(unknown) < need:
        unknown.append(BASIC_FIGHTING_ENERGY)
    your_deck = unknown[: me.deckCount]
    your_prize = unknown[me.deckCount : me.deckCount + len(me.prize)]

    opponent_deck = [PLACEHOLDER_MON] * op.deckCount
    opponent_prize = [PLACEHOLDER_MON] * len(op.prize)
    opponent_hand = [BASIC_FIGHTING_ENERGY] * op.handCount
    opponent_active = []
    if op.active and op.active[0] is None:
        opponent_active = [PLACEHOLDER_MON]
    return your_deck, your_prize, opponent_deck, opponent_prize, opponent_hand, opponent_active


def _rollout(state, root_turn: int, deadline: float):
    """自分のターンが終わるまでNN方策でgreedyに進める。"""
    for _ in range(ROLLOUT_STEP_CAP):
        obs = state.observation
        if obs.current.result != -1 or obs.current.turn != root_turn:
            break
        if time.perf_counter() > deadline:
            break
        try:
            sel = _nn_choose(obs)
            if not sel and obs.select.minCount > 0:
                sel = _fallback_selection(obs.select)
        except Exception:
            sel = _fallback_selection(obs.select)
        state = search_step(state.searchId, sel)
    return state.observation


def _lethal_search(obs: Observation) -> list[int] | None:
    global _search_time_used

    select = obs.select
    if select.minCount != 1 or select.maxCount != 1 or len(select.option) < 2:
        return None

    my_index = obs.current.yourIndex
    me = obs.current.players[my_index]
    if len(me.prize) > 3:
        return None
    if _search_time_used > SEARCH_GAME_BUDGET:
        return None

    t0 = time.perf_counter()
    deadline = t0 + SEARCH_MOVE_BUDGET
    root_turn = obs.current.turn

    logits = _nn_logits(obs)
    if logits is None:
        return None
    scores = list(map(float, logits))
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    candidates = ranked[:SEARCH_CANDIDATES]

    always_wins = {i: True for i in candidates}
    tried = {i: 0 for i in candidates}
    try:
        for _ in range(SEARCH_SAMPLES):
            if time.perf_counter() > deadline:
                break
            root = search_begin(obs, *_determinize(obs))
            for idx in candidates:
                if not always_wins[idx] or time.perf_counter() > deadline:
                    continue
                try:
                    child = search_step(root.searchId, [idx])
                    final_obs = _rollout(child, root_turn, deadline)
                    if final_obs.current.result != my_index:
                        always_wins[idx] = False
                    tried[idx] += 1
                except Exception:
                    always_wins[idx] = False
    finally:
        try:
            search_end()
        except Exception:
            pass
        _search_time_used += time.perf_counter() - t0

    winners = [i for i in candidates if always_wins[i] and tried[i] == SEARCH_SAMPLES]
    if not winners:
        return None
    return [max(winners, key=lambda i: scores[i])]


def agent(obs_dict: dict) -> list[int]:
    global _search_time_used, _pre_turn
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        return my_deck

    if obs.current.turn <= 2 and _pre_turn > 2:
        _search_time_used = 0.0  # 新しいゲーム（ローカルarenaはモジュール使い回し）
    _pre_turn = obs.current.turn

    if (
        USE_LETHAL
        and obs.select.context == SelectContext.MAIN
        and obs.current.turn >= 2
    ):
        try:
            choice = _lethal_search(obs)
        except Exception:
            choice = None
        if choice is not None:
            return choice

    return _nn_choose(obs)
