"""SPSA optimizer for 079 class-logit biases (I-122 / EXP-079).

CRN改造エンジン（cg/Export_seeded.cpp）上で、058のGBTクラスlogitへの
バイアス b[class][bucket] (37クラス×3山残量バケット=111次元) を
プール対戦の重み付き勝率を目的関数としてSPSAで直接最適化する。

- 目的関数: プールv3から 038(Marnie) を除外し正規化した重み付きスコア率。
  038は 036 と並ぶ転移ホールドアウト（学習に使わない）。
- CRNペア: f(θ+cΔ) と f(θ-cΔ) は同一エンジンシードブロックで評価
  （初期手札・シャッフル・コインが一致し、差分推定の分散が対消滅で減る）。
  シードブロックはイテレーションごとに更新（単一シード集合への過適合防止）。
- 各イテレーションの θ を jsonl にチェックポイント（中断・再開可能）。

実行: uv run agents/079_058_crnopt/spsa_079.py --iters 40 --games 240
再開: --resume out/spsa_log.jsonl があれば最後のθから続行
"""

import argparse
import json
import math
import os
import random
import sys
import time

REPO = "/Users/akira/kaggle/pokemon_tcg"
AGENT_DIR = os.path.join(REPO, "agents/079_058_crnopt")
sys.path.insert(0, REPO)
sys.path.insert(0, AGENT_DIR)

from feat_058 import CLASSES  # noqa: E402

sys.path.insert(0, os.path.join(REPO, "arena"))
import run_match  # noqa: E402

BUCKETS = 3  # 山残量 >=30 / 15-29 / <=14

# 目的関数プール: pool.json v3 から 038(ホールドアウト) を除外
OBJECTIVE_POOL = [
    ("agents/045_alakazam_full", 40.4),
    ("agents/047_kangaskhan_replica", 13.0),
    ("agents/041_garchomp_replica", 7.4),
    ("agents/042_crustle_replica", 4.3),
    ("agents/007_lucario_lethal", 3.7),
    ("agents/035_starmie_replica", 3.6),
    ("agents/046_archaludon_replica", 3.0),
    ("agents/021_dragapult_sample", 2.1),
    ("agents/000_random", 10.2),
]


def theta_to_bias(theta: list[float]) -> dict:
    return {c: [theta[i * BUCKETS + b] for b in range(BUCKETS)]
            for i, c in enumerate(CLASSES)}


def alloc_games(total: int) -> list[int]:
    wsum = sum(w for _, w in OBJECTIVE_POOL)
    ns = [max(8, round(total * w / wsum)) for _, w in OBJECTIVE_POOL]
    # 偶数化（先後入替のバランス維持）
    return [n + (n % 2) for n in ns]


def eval_weighted(bias: dict, seed_base: int, games_alloc: list[int],
                  workers: int, tmp_dir: str, tag: str) -> tuple[float, dict]:
    """重み付きスコア率と対面別詳細を返す。CRN: engine seedはseed_baseから導出。"""
    bp = os.path.join(tmp_dir, f"bias_{tag}.json")
    with open(bp, "w") as f:
        json.dump(bias, f)
    os.environ["F079_BIAS"] = bp

    num = 0.0
    den = 0.0
    detail = {}
    for k, ((opp, w), n) in enumerate(zip(OBJECTIVE_POOL, games_alloc)):
        r = run_match.run_series(
            AGENT_DIR, os.path.join(REPO, opp), n, workers,
            step_cap=30000, seed=seed_base + k * 131071,
            engine_seed_base=seed_base + k * 65537)
        if r["errors"] > 0:
            raise RuntimeError(f"errors vs {opp}: {r['error_details'][:2]}")
        sr = r["a_score_rate"]
        num += w * sr
        den += w
        detail[os.path.basename(opp)] = {"n": n, "sr": sr}
    return num / den, detail


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--games", type=int, default=240,
                    help="games per function eval (± each)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--c0", type=float, default=0.2, help="SPSA probe size")
    ap.add_argument("--a0", type=float, default=0.8, help="SPSA step size")
    ap.add_argument("--A", type=float, default=6.0)
    ap.add_argument("--clip", type=float, default=0.6,
                    help="per-component |theta| clip")
    ap.add_argument("--out", default=os.path.join(AGENT_DIR, "spsa_log.jsonl"))
    ap.add_argument("--master-seed", type=int, default=20260720)
    args = ap.parse_args()

    dim = len(CLASSES) * BUCKETS
    theta = [0.0] * dim
    start_iter = 0
    if os.path.exists(args.out):
        with open(args.out) as f:
            lines = [json.loads(x) for x in f if x.strip()]
        if lines:
            theta = lines[-1]["theta"]
            start_iter = lines[-1]["iter"] + 1
            print(f"resume from iter {start_iter}", file=sys.stderr)

    tmp_dir = os.path.join(AGENT_DIR, "spsa_tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    games_alloc = alloc_games(args.games)
    total_games = sum(games_alloc)
    print(f"dim={dim} games/eval={total_games} alloc={games_alloc}",
          file=sys.stderr)

    # 摂動列はイテレーションkごとにマスターシードから導出（再開時も同一列を再現）
    for k in range(start_iter, args.iters):
        it_rng = random.Random(args.master_seed + 7 * k)
        ck = args.c0 / ((k + 1) ** 0.101)
        ak = args.a0 / ((k + 1 + args.A) ** 0.602)
        delta = [it_rng.choice([-1.0, 1.0]) for _ in range(dim)]
        seed_base = 1_000_000 + k * 977  # イテレーションごとに新しいCRNブロック

        t0 = time.time()
        tp = [theta[i] + ck * delta[i] for i in range(dim)]
        tm = [theta[i] - ck * delta[i] for i in range(dim)]
        fp, dp = eval_weighted(theta_to_bias(tp), seed_base, games_alloc,
                               args.workers, tmp_dir, "plus")
        fm, dm = eval_weighted(theta_to_bias(tm), seed_base, games_alloc,
                               args.workers, tmp_dir, "minus")

        g_scalar = (fp - fm) / (2.0 * ck)
        theta = [max(-args.clip, min(args.clip,
                 theta[i] + ak * g_scalar * delta[i])) for i in range(dim)]

        rec = {"iter": k, "ck": round(ck, 4), "ak": round(ak, 4),
               "fp": round(fp, 4), "fm": round(fm, 4),
               "diff": round(fp - fm, 4),
               "theta_norm": round(math.sqrt(sum(t * t for t in theta)), 4),
               "sec": round(time.time() - t0, 1),
               "seed_base": seed_base,
               "detail_plus": dp, "detail_minus": dm,
               "theta": [round(t, 5) for t in theta]}
        with open(args.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"iter {k}: f+={fp:.4f} f-={fm:.4f} diff={fp-fm:+.4f} "
              f"|θ|={rec['theta_norm']:.3f} ({rec['sec']}s)", file=sys.stderr)

    # 最終θを平均化なしでbiasファイルへ（評価は別途、run_matchのF079_BIASで）
    final = theta_to_bias(theta)
    with open(os.path.join(AGENT_DIR, "bias_spsa_final.json"), "w") as f:
        json.dump(final, f, indent=0)
    print("final bias written to bias_spsa_final.json", file=sys.stderr)


if __name__ == "__main__":
    main()
