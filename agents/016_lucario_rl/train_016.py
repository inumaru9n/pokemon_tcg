"""EXP-016: BC初期化+REINFORCEによる自己対戦RL（リポジトリルートから実行）。

  uv run python -u agents/016_lucario_rl/train_016.py run --iters 20

各イテレーション:
  1. 現在の方策（温度サンプリング）で対戦を収集
     （ミラー40% + vs 004/002/010 各20%。libcgリーク対策でワーカー使い捨て）
  2. REINFORCE更新: advantage = 勝敗報酬 − 相手タグ別平均（分散削減）
  3. 3イテレーションごとに argmax方策で vs 004 のH2H評価（400戦）、ベスト重みを保存

出力: weights_016.npz（最新）/ weights_016_best.npz（対004ベスト）/ 学習ログはstdout
"""

from __future__ import annotations

import os

# numpy import前にBLASをシングルスレッド化する。理由: マルチスレッドBLAS実行後に
# fork するとロック済みミューテックスを子が継承してデッドロックする（EXP-016初回の
# ハング原因）。ワーカー4並列とのスレッド過剰化も同時に防げる。
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import multiprocessing as mp
import random
import sys
import time
from pathlib import Path

import numpy as np

AGENT_DIR = Path(__file__).resolve().parent
REPO = AGENT_DIR.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "arena"))
sys.path.insert(0, str(AGENT_DIR))

import features_016 as F  # noqa: E402
import model_016 as M     # noqa: E402

WEIGHTS = AGENT_DIR / "weights_016.npz"
BEST = AGENT_DIR / "weights_016_best.npz"

OPP_MIX = [("mirror", 0.4), ("004", 0.2), ("002", 0.2), ("010", 0.2)]
TEMP = 0.7          # 収集時のsoftmax温度
GAMES_PER_ITER = 600
LR = 1e-4
EVAL_EVERY = 3
STEP_CAP = 30000

# ---------------------------------------------------------------- worker

_W: dict = {}


def _init_worker(weights_path: str) -> None:
    global _W
    from run_match import load_agent_module, read_deck
    _W["p"] = M.load_params(weights_path)
    _W["deck"] = read_deck(str(AGENT_DIR))
    _W["opps"] = {
        "004": (load_agent_module(str(REPO / "agents/004_lucario_islet"), "W4"),
                read_deck(str(REPO / "agents/004_lucario_islet"))),
        "002": (load_agent_module(str(REPO / "agents/002_marnie_grimmsnarl"), "W2"),
                read_deck(str(REPO / "agents/002_marnie_grimmsnarl"))),
        "010": (load_agent_module(str(REPO / "agents/010_alakazam"), "W10"),
                read_deck(str(REPO / "agents/010_alakazam"))),
    }
    _W["rng"] = np.random.default_rng()


def _policy_select(obs_dict, record: list):
    """現方策で温度サンプリングし、(特徴, 選択index) をrecordに積む。"""
    from cg.api import to_observation_class
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        return _W["deck"]
    n = len(obs.select.option)
    if n == 0 or obs.select.maxCount == 0:
        return []
    if n == 1:
        return [0]
    ctx, opts = F.extract(obs)
    if len(opts) > M.P_MAX:
        return list(range(n))[: obs.select.maxCount]
    X, mask, _ = M.build_batch(_W["p"], [ctx], [opts])
    logits, _ = M.forward(_W["p"], X, mask)
    z = logits[0, :len(opts)] / TEMP
    z = z - z.max()
    prob = np.exp(z)
    prob /= prob.sum()
    a = int(_W["rng"].choice(len(opts), p=prob))
    record.append((ctx, opts, a))
    order = [a] + [i for i in np.argsort(-logits[0, :len(opts)]) if i != a]
    return [int(i) for i in order[: obs.select.maxCount]]


def play_game(args) -> tuple:
    """1ゲーム実行。Returns (tag, reward, trajectory)"""
    game_no, tag = args
    from cg.game import battle_start, battle_select, battle_finish

    if tag == "mirror":
        opp_agent, opp_deck = None, _W["deck"]  # 相手側も現方策（ただし勾配は片側のみ）
    else:
        opp_agent, opp_deck = _W["opps"][tag]

    my_seat = game_no % 2
    deck0 = _W["deck"] if my_seat == 0 else opp_deck
    deck1 = opp_deck if my_seat == 0 else _W["deck"]

    record: list = []
    dummy: list = []
    obs_dict, _ = battle_start(deck0, deck1)
    try:
        if obs_dict is None:
            return tag, 0.0, []
        steps = 0
        while obs_dict["current"]["result"] == -1 and steps < STEP_CAP:
            player = obs_dict["current"]["yourIndex"]
            if player == my_seat:
                sel = _policy_select(obs_dict, record)
            elif tag == "mirror":
                sel = _policy_select(obs_dict, dummy)  # 記録しない側
            else:
                sel = opp_agent(obs_dict)
            obs_dict = battle_select(sel)
            steps += 1
        result = obs_dict["current"]["result"]
    finally:
        battle_finish()
    reward = 1.0 if result == my_seat else (-1.0 if result == (1 - my_seat) else 0.0)
    return tag, reward, record


# ---------------------------------------------------------------- trainer

def collect(weights_path: str, n_games: int, workers: int = 4):
    tags = []
    for tag, w in OPP_MIX:
        tags += [tag] * int(n_games * w)
    random.shuffle(tags)
    args = list(enumerate(tags))
    # forkコンテキストを明示（macOS Python3.13のデフォルトspawnはmain再importで壊れる、
    # かつspawnはlibcgの再ロードで遅い）。maxtasksperchildでlibcgリークを封じる。
    ctx = mp.get_context("fork")
    with ctx.Pool(workers, initializer=_init_worker, initargs=(weights_path,),
                  maxtasksperchild=80) as pool:
        results = pool.map(play_game, args, chunksize=4)
    return results


def update(p: dict, opt: M.Adam, results, batch: int = 256) -> dict:
    # 相手タグ別ベースライン
    by_tag: dict = {}
    for tag, r, _ in results:
        by_tag.setdefault(tag, []).append(r)
    base = {t: float(np.mean(v)) for t, v in by_tag.items()}

    samples = []
    for tag, r, traj in results:
        adv = r - base[tag]
        for ctx, opts, a in traj:
            samples.append((ctx, opts, a, adv))
    if not samples:
        return {}
    advs = np.array([s[3] for s in samples], np.float32)
    std = advs.std() + 1e-6
    random.shuffle(samples)

    n_upd = 0
    for i in range(0, len(samples), batch):
        chunk = samples[i:i + batch]
        ctxs = [c[0] for c in chunk]
        optss = [c[1] for c in chunk]
        labels = np.array([c[2] for c in chunk], np.int32)
        w = np.array([c[3] for c in chunk], np.float32) / std
        X, mask, idx = M.build_batch(p, ctxs, optss)
        _, _, g = M.loss_and_grads(p, X, mask, idx, labels, sample_w=w)
        opt.step(p, g)
        n_upd += 1
    stats = {t: (len(v), float(np.mean(v))) for t, v in by_tag.items()}
    stats["n_samples"] = len(samples)
    return stats


def eval_vs(agent_dir: str, opponent: str, n: int, seed: int) -> float:
    import subprocess, json
    out = subprocess.run(
        ["uv", "run", str(REPO / "arena/run_match.py"), agent_dir,
         str(REPO / f"agents/{opponent}"), "-n", str(n), "-w", "4", "--seed", str(seed)],
        capture_output=True, text=True, cwd=str(REPO))
    return json.loads(out.stdout)["a_score_rate"]


def run(iters: int) -> None:
    p = M.load_params(str(WEIGHTS))
    opt = M.Adam(p, lr=LR)
    best = eval_vs(str(AGENT_DIR), "004_lucario_islet", 400, seed=999)
    print(f"initial eval vs 004 (argmax): {best:.3f}", flush=True)

    for it in range(1, iters + 1):
        t0 = time.time()
        M.save_params(str(WEIGHTS), p)
        results = collect(str(WEIGHTS), GAMES_PER_ITER)
        stats = update(p, opt, results)
        wr = {t: f"{v[1]:+.2f}({v[0]})" for t, v in stats.items() if t != "n_samples"}
        print(f"iter {it}: reward {wr} samples={stats.get('n_samples')} "
              f"({time.time() - t0:.0f}s)", flush=True)

        if it % EVAL_EVERY == 0:
            M.save_params(str(WEIGHTS), p)
            score = eval_vs(str(AGENT_DIR), "004_lucario_islet", 400, seed=999 + it)
            mark = ""
            if score > best:
                best = score
                M.save_params(str(BEST), p)
                mark = "  ** new best"
            print(f"  eval vs 004 (argmax, 400g): {score:.3f} (best {best:.3f}){mark}",
                  flush=True)

    M.save_params(str(WEIGHTS), p)
    print(f"done. best vs 004 = {best:.3f} (weights_016_best.npz)", flush=True)


def smoke() -> None:
    """24ゲーム収集+1更新の疎通確認（本番前）。"""
    p = M.load_params(str(WEIGHTS))
    opt = M.Adam(p, lr=LR)
    M.save_params(str(WEIGHTS), p)
    results = collect(str(WEIGHTS), 24, workers=4)
    print("games:", len(results), "traj samples:", sum(len(r[2]) for r in results), flush=True)
    stats = update(p, opt, results)
    print("update ok:", stats, flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    iters = int(sys.argv[sys.argv.index("--iters") + 1]) if "--iters" in sys.argv else 20
    if cmd == "run":
        run(iters)
    elif cmd == "smoke":
        smoke()
    else:
        sys.exit("usage: train_016.py run [--iters N] | smoke")
