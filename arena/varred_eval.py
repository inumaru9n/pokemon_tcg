"""CUPED型の制御変量による評価分散削減（AIVAT arXiv 1612.06915 の軽量版）。

  uv run arena/varred_eval.py <agentA> <agentB> -n 400

狙い: CRN不可の本環境で、勝敗(0/1)の高分散を序盤盤面の共変量で減らす。
共変量は **turn<=2 の盤面価値**（ほぼ運=処置前変数なので、両アームで期待値を共有でき
差分推定を不偏に保てる）。中盤以降の盤面はエージェント依存(処置後)なのでバイアスの元になり使わない。

出力: 生の勝率±CI と、CUPED調整後の勝率±CI（分散削減率と相関ρ）。
検証: 同一エージェント同士(A=B)で調整後差分≈0かつCIが縮めば無バイアス＆有効。
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from run_match import load_agent_module, read_deck  # noqa: E402

CAPTURE_TURN = 2  # 共変量を取る局面（<=2はほぼ運＝処置前）


def early_features(obs_dict, my_index):
    """turn<=2時点の、勝敗と相関しそうな盤面スカラー（運で決まる序盤状況）。"""
    cur = obs_dict["current"]
    me = cur["players"][my_index]
    op = cur["players"][1 - my_index]

    def board(ps):
        n = hp = en = 0
        for p in (ps.get("active") or []) + (ps.get("bench") or []):
            if p is None:
                continue
            n += 1
            hp += p.get("hp", 0)
            en += len(p.get("energies") or [])
        return n, hp, en

    m_n, m_hp, m_en = board(me)
    o_n, o_hp, o_en = board(op)
    return [
        1.0,
        m_n - o_n,
        (m_hp - o_hp) / 100.0,
        m_en - o_en,
        me.get("handCount", 0) - op.get("handCount", 0),
        1.0 if cur.get("firstPlayer") == my_index else 0.0,
    ]


def play(agent_a, deck_a, agent_b, deck_b, game_index, step_cap=30000):
    from cg.game import battle_start, battle_select, battle_finish
    from cg.api import to_observation_class  # noqa: F401 (kept for parity)

    random.seed(1234 + game_index)
    a_seat = game_index % 2
    seat = {a_seat: "A", 1 - a_seat: "B"}
    agents = {"A": agent_a, "B": agent_b}
    deck0 = deck_a if a_seat == 0 else deck_b
    deck1 = deck_b if a_seat == 0 else deck_a

    obs, _ = battle_start(deck0, deck1)
    feat = None
    try:
        if obs is None:
            return None
        steps = 0
        while obs["current"]["result"] == -1 and steps < step_cap:
            if feat is None and obs["current"]["turn"] >= CAPTURE_TURN:
                feat = early_features(obs, a_seat)  # A視点
            tag = seat[obs["current"]["yourIndex"]]
            sel = agents[tag](obs)
            obs = battle_select(sel)
            steps += 1
        result = obs["current"]["result"]
    finally:
        battle_finish()
    if feat is None:
        feat = [1.0, 0, 0, 0, 0, 1.0 if a_seat == 0 else 0.0]
    y = 1.0 if result == a_seat else 0.0 if result == (1 - a_seat) else 0.5
    return y, feat


def wilson(p, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - m, c + m)


def cuped(ys, feats):
    """共変量Cを序盤特徴の最小二乗予測とし、CUPED調整。E[C]は標本平均（序盤=処置前近似）。"""
    n = len(ys)
    d = len(feats[0])
    # 正規方程式 (X^T X) w = X^T y で outcome の線形予測 C を得る
    XtX = [[sum(feats[i][a] * feats[i][b] for i in range(n)) for b in range(d)] for a in range(d)]
    Xty = [sum(feats[i][a] * ys[i] for i in range(n)) for a in range(d)]
    w = _solve(XtX, Xty)
    C = [sum(w[a] * feats[i][a] for a in range(d)) for i in range(n)]
    Cbar = sum(C) / n
    varC = sum((c - Cbar) ** 2 for c in C) / n
    if varC < 1e-12:
        return sum(ys) / n, 0.0, 0.0
    covYC = sum((ys[i] - sum(ys) / n) * (C[i] - Cbar) for i in range(n)) / n
    theta = covYC / varC
    adj = [ys[i] - theta * (C[i] - Cbar) for i in range(n)]
    mean_adj = sum(adj) / n
    var_adj = sum((a - mean_adj) ** 2 for a in adj) / (n - 1)
    ybar = sum(ys) / n
    var_y = sum((y - ybar) ** 2 for y in ys) / (n - 1)
    rho2 = 1 - (var_adj / var_y) if var_y > 0 else 0.0
    return mean_adj, var_adj, rho2


def _solve(A, b):
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[piv] = M[piv], M[c]
        if abs(M[c][c]) < 1e-12:
            M[c][c] = 1e-12
        for r in range(n):
            if r != c:
                f = M[r][c] / M[c][c]
                for k in range(c, n + 1):
                    M[r][k] -= f * M[c][k]
    return [M[i][n] / M[i][i] for i in range(n)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("agent_a")
    ap.add_argument("agent_b")
    ap.add_argument("-n", "--games", type=int, default=400)
    args = ap.parse_args()

    a = load_agent_module(args.agent_a, "A")
    b = load_agent_module(args.agent_b, "B")
    da, db = read_deck(args.agent_a), read_deck(args.agent_b)

    ys, feats = [], []
    for g in range(args.games):
        r = play(a, da, b, db, g)
        if r is None:
            continue
        ys.append(r[0])
        feats.append(r[1])

    n = len(ys)
    raw = sum(ys) / n
    lo, hi = wilson(raw, n)
    mean_adj, var_adj, rho2 = cuped(ys, feats)
    se_adj = math.sqrt(var_adj / n)
    alo, ahi = mean_adj - 1.96 * se_adj, mean_adj + 1.96 * se_adj

    print(f"games: {n}")
    print(f"raw   : {raw:.4f}  CI [{lo:.4f}, {hi:.4f}]  width {hi-lo:.4f}")
    print(f"CUPED : {mean_adj:.4f}  CI [{alo:.4f}, {ahi:.4f}]  width {ahi-alo:.4f}")
    print(f"共変量相関 rho^2 = {rho2:.3f}  → 分散削減率 {rho2*100:.1f}% (CI幅 {(1-math.sqrt(max(0,1-rho2)))*100:.1f}% 縮小の理論値)")


if __name__ == "__main__":
    main()
