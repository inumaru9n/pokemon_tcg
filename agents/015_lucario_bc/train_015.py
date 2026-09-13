"""EXP-015: BCデータ収集と学習（リポジトリルートから実行）。

  uv run agents/015_lucario_bc/train_015.py collect   # 教師(004)の対戦から教師データ収集 → data_015.npz
  uv run agents/015_lucario_bc/train_015.py train     # 学習 → weights_015.npz（ホールドアウト精度を報告）
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

AGENT_DIR = Path(__file__).resolve().parent
REPO = AGENT_DIR.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "arena"))
sys.path.insert(0, str(AGENT_DIR))

import features_015 as F  # noqa: E402
import model_015 as M     # noqa: E402

GAMES = {"mirror": 500, "002": 500, "010": 500}
DATA_PATH = AGENT_DIR / "data_015.npz"
WEIGHTS_PATH = AGENT_DIR / "weights_015.npz"


def collect() -> None:
    from run_match import load_agent_module, read_deck
    from cg.game import battle_start, battle_select, battle_finish
    from cg.api import to_observation_class

    teacher = load_agent_module(str(REPO / "agents/004_lucario_islet"), "T")
    teacher_deck = read_deck(str(REPO / "agents/004_lucario_islet"))
    opponents = {
        "mirror": (teacher, teacher_deck),
        "002": (load_agent_module(str(REPO / "agents/002_marnie_grimmsnarl"), "O2"),
                read_deck(str(REPO / "agents/002_marnie_grimmsnarl"))),
        "010": (load_agent_module(str(REPO / "agents/010_alakazam"), "O10"),
                read_deck(str(REPO / "agents/010_alakazam"))),
    }

    ctx_id, my_act, op_act, labels = [], [], [], []
    ctx_sc = []
    off = [0]
    o_type, o_card, o_target, o_attack = [], [], [], []
    o_sc = []

    random.seed(20260705)
    game_no = 0
    for tag, n in GAMES.items():
        opp_agent, opp_deck = opponents[tag]
        for _ in range(n):
            game_no += 1
            t_seat = game_no % 2
            deck0 = teacher_deck if t_seat == 0 else opp_deck
            deck1 = opp_deck if t_seat == 0 else teacher_deck
            agents = {t_seat: teacher, 1 - t_seat: opp_agent}
            obs_dict, _ = battle_start(deck0, deck1)
            try:
                if obs_dict is None:
                    continue
                steps = 0
                while obs_dict["current"]["result"] == -1 and steps < 30000:
                    player = obs_dict["current"]["yourIndex"]
                    sel = agents[player](obs_dict)
                    is_teacher = (player == t_seat) or tag == "mirror"
                    n_opt = len(obs_dict["select"]["option"])
                    if is_teacher and 2 <= n_opt <= M.P_MAX and sel:
                        obs = to_observation_class(obs_dict)
                        (cid, ma, oa, csc), opts = F.extract(obs)
                        ctx_id.append(cid); my_act.append(ma); op_act.append(oa)
                        ctx_sc.append(csc); labels.append(sel[0])
                        for ot, c, t, a, s in opts:
                            o_type.append(ot); o_card.append(c)
                            o_target.append(t); o_attack.append(a); o_sc.append(s)
                        off.append(len(o_type))
                    obs_dict = battle_select(sel)
                    steps += 1
            finally:
                battle_finish()
            if game_no % 100 == 0:
                print(f"  {game_no} games, {len(labels)} selects")

    np.savez_compressed(
        DATA_PATH,
        ctx_id=np.array(ctx_id, np.int32), my_act=np.array(my_act, np.int32),
        op_act=np.array(op_act, np.int32), ctx_sc=np.array(ctx_sc, np.float32),
        labels=np.array(labels, np.int32), off=np.array(off, np.int64),
        o_type=np.array(o_type, np.int32), o_card=np.array(o_card, np.int32),
        o_target=np.array(o_target, np.int32), o_attack=np.array(o_attack, np.int32),
        o_sc=np.array(o_sc, np.float32),
    )
    print(f"saved: {DATA_PATH} ({len(labels)} selects)")


def _select_slice(z, i):
    s, e = z["off"][i], z["off"][i + 1]
    ctx = (int(z["ctx_id"][i]), int(z["my_act"][i]), int(z["op_act"][i]), z["ctx_sc"][i])
    opts = list(zip(z["o_type"][s:e], z["o_card"][s:e], z["o_target"][s:e],
                    z["o_attack"][s:e], z["o_sc"][s:e]))
    return ctx, opts


def train(epochs: int = 4, batch: int = 256, lr: float = 1e-3) -> None:
    z = np.load(DATA_PATH)
    n = len(z["labels"])
    rng = np.random.default_rng(7)
    order = rng.permutation(n)
    cut = int(n * 0.95)
    tr, te = order[:cut], order[cut:]
    print(f"train={len(tr)} holdout={len(te)}")

    p = M.init_params(seed=0)
    opt = M.Adam(p, lr=lr)
    n_batches = (len(tr) + batch - 1) // batch
    for ep in range(epochs):
        rng.shuffle(tr)
        tot_loss = tot_acc = nb = 0
        for i in range(0, len(tr), batch):
            ids = tr[i:i + batch]
            ctxs, optss = zip(*(_select_slice(z, j) for j in ids))
            X, mask, idx = M.build_batch(p, list(ctxs), list(optss))
            loss, acc, g = M.loss_and_grads(p, X, mask, idx, z["labels"][ids])
            opt.step(p, g)
            tot_loss += loss; tot_acc += acc; nb += 1
            if nb % 50 == 0:
                print(f"  ep{ep + 1} {nb}/{n_batches} loss={tot_loss / nb:.4f} acc={tot_acc / nb:.4f}",
                      flush=True)
        print(f"epoch {ep + 1}: loss={tot_loss / nb:.4f} acc={tot_acc / nb:.4f}", flush=True)

    # ホールドアウト評価（全体+コンテキスト別）
    correct = {}
    total = {}
    hit = 0
    for i in range(0, len(te), batch):
        ids = te[i:i + batch]
        ctxs, optss = zip(*(_select_slice(z, j) for j in ids))
        X, mask, _ = M.build_batch(p, list(ctxs), list(optss))
        logits, _ = M.forward(p, X, mask)
        pred = logits.argmax(axis=1)
        lab = z["labels"][ids]
        hit += int((pred == lab).sum())
        for j, ok in zip(ids, pred == lab):
            c = int(z["ctx_id"][j])
            total[c] = total.get(c, 0) + 1
            correct[c] = correct.get(c, 0) + int(ok)
    print(f"holdout acc = {hit / len(te):.4f}")
    for c in sorted(total, key=lambda c: -total[c])[:10]:
        print(f"  ctx {c}: acc={correct[c] / total[c]:.3f} (n={total[c]})")

    M.save_params(str(WEIGHTS_PATH), p)
    print(f"saved: {WEIGHTS_PATH}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "collect":
        collect()
    elif cmd == "train":
        train()
    else:
        sys.exit("usage: train_015.py collect|train")
