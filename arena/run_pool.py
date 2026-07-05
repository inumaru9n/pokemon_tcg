"""候補エージェントを標準相手プール（arena/pool.json）と対戦させ、加重総合勝率で評価するCLI。

使い方（リポジトリルートから）:
  uv run arena/run_pool.py agents/003_xxx -n 200 --json arena/results/EXP-003.json

- プール内の各相手と n 戦ずつ対戦（候補自身がプールにいる場合はスキップ）
- 総合勝率はプール定義の weight（本番メタシェア）で加重平均し、95%CIを正規近似で算出
- 判定目安: 加重CI下限 > 50% で validated、上限 < 50% で rejected（詳細はCLAUDE.mdの統計基準）
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_match import run_series  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_POOL = os.path.join(REPO_ROOT, "arena", "pool.json")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("candidate", help="directory of the candidate agent")
    ap.add_argument("-n", "--games", type=int, default=200, help="games per opponent")
    ap.add_argument("-w", "--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--step-cap", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pool", default=DEFAULT_POOL)
    ap.add_argument("--json", dest="json_path", default=None)
    args = ap.parse_args()

    with open(args.pool) as f:
        pool = json.load(f)
    cand_abs = os.path.abspath(args.candidate)
    opponents = [o for o in pool["opponents"]
                 if os.path.abspath(os.path.join(REPO_ROOT, o["dir"])) != cand_abs]
    if not opponents:
        sys.exit("プールに候補以外の相手がいません")

    matchups = []
    for o in opponents:
        opp_dir = os.path.join(REPO_ROOT, o["dir"])
        s = run_series(args.candidate, opp_dir, args.games, args.workers,
                       args.step_cap, args.seed)
        s["opponent"] = o["dir"]
        s["weight"] = o["weight"]
        s["archetype"] = o.get("archetype", "")
        matchups.append(s)
        lo, hi = s["a_score_rate_ci95"]
        name = os.path.basename(os.path.normpath(o["dir"]))
        print(f"  vs {name}: {s['a_wins']}W-{s['b_wins']}L-{s['draws']}D "
              f"| {s['a_score_rate']:.1%} (CI {lo:.1%}-{hi:.1%}) "
              f"| errors={s['errors']}", file=sys.stderr)

    # 加重平均と正規近似CI（分散は各対面の二項分散の加重和）
    total_w = sum(m["weight"] for m in matchups)
    p = sum(m["weight"] / total_w * m["a_score_rate"] for m in matchups)
    var = sum((m["weight"] / total_w) ** 2 * m["a_score_rate"] * (1 - m["a_score_rate"])
              / m["games"] for m in matchups)
    margin = 1.96 * math.sqrt(var)
    lo, hi = max(0.0, p - margin), min(1.0, p + margin)
    uniform = sum(m["a_score_rate"] for m in matchups) / len(matchups)
    errors = sum(m["errors"] for m in matchups)
    verdict = ("validated" if lo > 0.5 else
               "rejected" if hi < 0.5 else "inconclusive")

    result = {
        "candidate": cand_abs,
        "pool": args.pool,
        "games_per_opponent": args.games,
        "weighted_score_rate": round(p, 4),
        "weighted_ci95": [round(lo, 4), round(hi, 4)],
        "uniform_score_rate": round(uniform, 4),
        "errors": errors,
        "verdict": verdict,
        "seed": args.seed,
        "matchups": matchups,
    }
    if args.json_path:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_path)), exist_ok=True)
        with open(args.json_path, "w") as f:
            json.dump(result, f, indent=1)

    brief = {k: v for k, v in result.items() if k != "matchups"}
    brief["matchups"] = [{k: v for k, v in m.items() if k != "records"}
                         for m in result["matchups"]]
    print(json.dumps(brief, indent=1))
    name = os.path.basename(os.path.normpath(args.candidate))
    print(f"\n{name} vs pool: weighted {p:.1%} (95% CI {lo:.1%}-{hi:.1%}) "
          f"| uniform {uniform:.1%} | errors={errors} | verdict={verdict}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
