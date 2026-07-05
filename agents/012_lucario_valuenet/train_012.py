"""EXP-012: 価値関数の学習スクリプト（収集→学習→main.pyへ焼き込み）。

リポジトリルートから:
  uv run agents/012_lucario_valuenet/train_012.py

- 012の方策（=007ヒューリスティック、VALUE_WEIGHTS空なので探索上書きなし）で
  ミラー/対002/対010 の対戦を行い、各ターン終了時点の盤面特徴と最終勝敗を収集
- 純Python SGDでロジスティック回帰を学習（ホールドアウトで精度を報告）
- main.py の VALUE_* 定数を学習結果で書き換える
"""

from __future__ import annotations

import math
import random
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "arena"))

from run_match import load_agent_module, read_deck  # noqa: E402

AGENT_DIR = Path(__file__).resolve().parent
GAMES = {"mirror": 800, "002": 600, "010": 600}

me_agent = load_agent_module(str(AGENT_DIR), "V")
mod_me = sys.modules["agent_V"]
opponents = {
    "mirror": (me_agent, read_deck(str(AGENT_DIR))),
    "002": (load_agent_module(str(REPO / "agents/002_marnie_grimmsnarl"), "O2"),
            read_deck(str(REPO / "agents/002_marnie_grimmsnarl"))),
    "010": (load_agent_module(str(REPO / "agents/010_alakazam"), "O10"),
            read_deck(str(REPO / "agents/010_alakazam"))),
}
my_deck = read_deck(str(AGENT_DIR))

from cg.game import battle_start, battle_select, battle_finish  # noqa: E402
from cg.api import to_observation_class  # noqa: E402


def collect() -> tuple[list[list[float]], list[int]]:
    xs: list[list[float]] = []
    ys: list[int] = []
    random.seed(20260704)
    game_no = 0
    for tag, n in GAMES.items():
        opp_agent, opp_deck = opponents[tag]
        for g in range(n):
            game_no += 1
            me_seat = game_no % 2
            deck0 = my_deck if me_seat == 0 else opp_deck
            deck1 = opp_deck if me_seat == 0 else my_deck
            agents = {me_seat: me_agent, 1 - me_seat: opp_agent}

            obs_dict, _ = battle_start(deck0, deck1)
            samples: list[tuple[list[float], int]] = []  # (features, player)
            try:
                if obs_dict is None:
                    continue
                last_turn = 0
                steps = 0
                while obs_dict["current"]["result"] == -1 and steps < 30000:
                    cur_turn = obs_dict["current"]["turn"]
                    if cur_turn > last_turn and last_turn >= 2:
                        # プレイヤー last_turn の手番が終わった直後の状態
                        prev_player = obs_dict["current"]["yourIndex"] ^ 1
                        if tag == "mirror" or prev_player == me_seat:
                            o = to_observation_class(obs_dict)
                            samples.append((mod_me.value_features(o, prev_player), prev_player))
                    last_turn = max(last_turn, cur_turn)
                    player = obs_dict["current"]["yourIndex"]
                    sel = agents[player](obs_dict)
                    obs_dict = battle_select(sel)
                    steps += 1
                result = obs_dict["current"]["result"]
            finally:
                battle_finish()
            if result not in (0, 1):
                continue
            for feats, pl in samples:
                xs.append(feats)
                ys.append(1 if pl == result else 0)
            if game_no % 200 == 0:
                print(f"  {game_no} games, {len(xs)} samples")
    return xs, ys


def train(xs: list[list[float]], ys: list[int]):
    n = len(xs)
    d = len(xs[0])
    idx = list(range(n))
    random.seed(7)
    random.shuffle(idx)
    cut = int(n * 0.9)
    tr, te = idx[:cut], idx[cut:]

    means = [sum(xs[i][j] for i in tr) / len(tr) for j in range(d)]
    stds = []
    for j in range(d):
        v = sum((xs[i][j] - means[j]) ** 2 for i in tr) / len(tr)
        stds.append(math.sqrt(v) if v > 1e-12 else 1.0)

    def norm(row):
        return [(row[j] - means[j]) / stds[j] for j in range(d)]

    w = [0.0] * d
    b = 0.0
    lr, l2, epochs = 0.05, 1e-4, 40
    for ep in range(epochs):
        random.shuffle(tr)
        for i in tr:
            x = norm(xs[i])
            z = b + sum(w[j] * x[j] for j in range(d))
            p = 1.0 / (1.0 + math.exp(-max(-30, min(30, z))))
            g = p - ys[i]
            b -= lr * g
            for j in range(d):
                w[j] -= lr * (g * x[j] + l2 * w[j])

    def acc_auc(indices):
        pairs = []
        correct = 0
        for i in indices:
            x = norm(xs[i])
            z = b + sum(w[j] * x[j] for j in range(d))
            p = 1.0 / (1.0 + math.exp(-max(-30, min(30, z))))
            pairs.append((p, ys[i]))
            correct += int((p >= 0.5) == (ys[i] == 1))
        pos = sorted(p for p, y in pairs if y == 1)
        neg = sorted(p for p, y in pairs if y == 0)
        if not pos or not neg:
            return correct / len(indices), float("nan")
        import bisect
        wins = sum(bisect.bisect_left(neg, p) + 0.5 * (bisect.bisect_right(neg, p) - bisect.bisect_left(neg, p)) for p in pos)
        return correct / len(indices), wins / (len(pos) * len(neg))

    tr_acc, tr_auc = acc_auc(tr)
    te_acc, te_auc = acc_auc(te)
    print(f"train acc={tr_acc:.3f} auc={tr_auc:.3f} | test acc={te_acc:.3f} auc={te_auc:.3f} (n={n}, d={d})")
    return w, b, means, stds


def bake(w, b, means, stds):
    path = AGENT_DIR / "main.py"
    src = path.read_text()

    def fmt(v):
        return "[" + ", ".join(f"{x:.6f}" for x in v) + "]"

    src = re.sub(r"VALUE_WEIGHTS: list\[float\] = \[.*?\]", f"VALUE_WEIGHTS: list[float] = {fmt(w)}", src, count=1, flags=re.S)
    src = re.sub(r"VALUE_MEANS: list\[float\] = \[.*?\]", f"VALUE_MEANS: list[float] = {fmt(means)}", src, count=1, flags=re.S)
    src = re.sub(r"VALUE_STDS: list\[float\] = \[.*?\]", f"VALUE_STDS: list[float] = {fmt(stds)}", src, count=1, flags=re.S)
    src = re.sub(r"VALUE_BIAS: float = .*", f"VALUE_BIAS: float = {b:.6f}", src, count=1)
    path.write_text(src)
    print(f"baked into {path}")


if __name__ == "__main__":
    print("collecting ...")
    xs, ys = collect()
    print(f"total samples: {len(xs)}, win rate in labels: {sum(ys)/len(ys):.3f}")
    w, b, means, stds = train(xs, ys)
    bake(w, b, means, stds)
