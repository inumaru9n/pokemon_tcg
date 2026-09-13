"""EXP-033: 007蒸留のデータ収集と学習（リポジトリルートから実行）。

  uv run agents/033_lucario_distill2/train_033.py smoke     # 小規模収集で特徴抽出+1バッチ学習の動作確認
  uv run agents/033_lucario_distill2/train_033.py collect   # 教師(007)の対戦から教師データ収集 → data_033.npz
  uv run agents/033_lucario_distill2/train_033.py train     # 学習 → weights_033.npz（ホールドアウト精度を報告）

EXP-031との違い: 特徴量拡充（features_033: 盤面スロット48+参照ポケモン/攻撃8）と
モデル容量増（model_033: D_CARD 24 / 256-128）、8エポック。
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

import features_033 as F  # noqa: E402
import model_033 as M     # noqa: E402

# プール構成に合わせた対面分布（メタシェア荷重を粗く反映）
GAMES = {"mirror": 1200, "002": 800, "010": 800, "017": 400, "021": 400}
TEACHER_DIR = "agents/007_lucario_lethal"
DATA_PATH = AGENT_DIR / "data_033.npz"
WEIGHTS_PATH = AGENT_DIR / "weights_033.npz"


def collect(games: dict[str, int] | None = None, out_path: Path | None = None) -> None:
    games = games or GAMES
    out_path = out_path or DATA_PATH
    from run_match import load_agent_module, read_deck
    from cg.game import battle_start, battle_select, battle_finish
    from cg.api import to_observation_class

    teacher = load_agent_module(str(REPO / TEACHER_DIR), "T")
    teacher_deck = read_deck(str(REPO / TEACHER_DIR))
    opp_dirs = {
        "002": "agents/002_marnie_grimmsnarl",
        "010": "agents/010_alakazam",
        "017": "agents/017_starmie",
        "021": "agents/021_dragapult_sample",
    }
    opponents = {"mirror": (teacher, teacher_deck)}
    for tag, d in opp_dirs.items():
        opponents[tag] = (load_agent_module(str(REPO / d), f"O{tag}"), read_deck(str(REPO / d)))

    ctx_id, my_act, op_act, labels = [], [], [], []
    ctx_sc = []
    off = [0]
    o_type, o_card, o_target, o_attack = [], [], [], []
    o_sc = []

    random.seed(20260709)
    game_no = 0
    for tag, n in games.items():
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
                print(f"  {game_no} games, {len(labels)} selects", flush=True)

    np.savez_compressed(
        out_path,
        ctx_id=np.array(ctx_id, np.int32), my_act=np.array(my_act, np.int32),
        op_act=np.array(op_act, np.int32), ctx_sc=np.array(ctx_sc, np.float32),
        labels=np.array(labels, np.int32), off=np.array(off, np.int64),
        o_type=np.array(o_type, np.int32), o_card=np.array(o_card, np.int32),
        o_target=np.array(o_target, np.int32), o_attack=np.array(o_attack, np.int32),
        o_sc=np.array(o_sc, np.float32),
    )
    print(f"saved: {out_path} ({len(labels)} selects)")


def _select_slice(z, i):
    s, e = z["off"][i], z["off"][i + 1]
    ctx = (int(z["ctx_id"][i]), int(z["my_act"][i]), int(z["op_act"][i]), z["ctx_sc"][i])
    opts = list(zip(z["o_type"][s:e], z["o_card"][s:e], z["o_target"][s:e],
                    z["o_attack"][s:e], z["o_sc"][s:e]))
    return ctx, opts


def _holdout_eval(p, z, te, batch: int = 256):
    """ホールドアウト精度（全体+コンテキスト別）を返す。"""
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
    return hit / len(te), correct, total


def train(epochs: int = 8, batch: int = 256, lr: float = 1e-3,
          data_path: Path | None = None) -> None:
    # NpzFileはアクセス毎にzip解凍が走る（1バッチ84秒の主犯）→ 最初に全部メモリへ
    z = {k: v for k, v in np.load(data_path or DATA_PATH).items()}
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
            if nb % 200 == 0:
                print(f"  ep{ep + 1} {nb}/{n_batches} loss={tot_loss / nb:.4f} acc={tot_acc / nb:.4f}",
                      flush=True)
        acc_te, _, _ = _holdout_eval(p, z, te, batch)
        print(f"epoch {ep + 1}: loss={tot_loss / nb:.4f} acc={tot_acc / nb:.4f} "
              f"holdout={acc_te:.4f}", flush=True)
        M.save_params(str(WEIGHTS_PATH), p)  # 途中停止に備えたエポック毎チェックポイント

    acc_te, correct, total = _holdout_eval(p, z, te, batch)
    print(f"holdout acc = {acc_te:.4f}")
    for c in sorted(total, key=lambda c: -total[c])[:12]:
        print(f"  ctx {c}: acc={correct[c] / total[c]:.3f} (n={total[c]})")

    M.save_params(str(WEIGHTS_PATH), p)
    print(f"saved: {WEIGHTS_PATH}")


def smoke() -> None:
    """小規模収集→1エポック学習で特徴抽出・学習系の動作確認（重みは保存しない）。"""
    path = AGENT_DIR / "data_033_smoke.npz"
    collect(games={"mirror": 6, "002": 4, "010": 4, "017": 2, "021": 2}, out_path=path)
    z = {k: v for k, v in np.load(path).items()}
    n = len(z["labels"])
    assert z["ctx_sc"].shape[1] == F.N_CTX_SCALAR, z["ctx_sc"].shape
    assert z["o_sc"].shape[1] == F.N_OPT_SCALAR, z["o_sc"].shape
    p = M.init_params(seed=0)
    opt = M.Adam(p, lr=1e-3)
    ids = np.arange(min(256, n))
    ctxs, optss = zip(*(_select_slice(z, j) for j in ids))
    import time
    t0 = time.perf_counter()
    X, mask, idx = M.build_batch(p, list(ctxs), list(optss))
    loss, acc, g = M.loss_and_grads(p, X, mask, idx, z["labels"][ids])
    opt.step(p, g)
    dt = time.perf_counter() - t0
    print(f"smoke OK: {n} selects, batch256 loss={loss:.4f} acc={acc:.4f} "
          f"1step={dt:.2f}s, D_IN={M.D_IN}", flush=True)
    path.unlink(missing_ok=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "collect":
        collect()
    elif cmd == "train":
        train()
    elif cmd == "smoke":
        smoke()
    else:
        sys.exit("usage: train_033.py smoke|collect|train")
