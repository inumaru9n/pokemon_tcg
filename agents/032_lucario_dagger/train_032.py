"""EXP-032: DAgger蒸留（EXP-031 plain BCの続き）。リポジトリルートから実行。

  uv run agents/032_lucario_dagger/train_032.py collect 1   # 生徒(現weights_032の純NN)がプレイ→教師007がラベル → dagger_iter1.npz
  uv run agents/032_lucario_dagger/train_032.py train 1     # data_031 + dagger_iter1..N を統合しゼロから再学習 → weights_032.npz

DAggerの核心: 実際に指す手は生徒の選択（教師の手を指してはいけない）。
教師007はSEARCH_GAME_BUDGET=-1.0でlethal探索を無効化してラベル付けする
（蒸留対象はヒューリスティック部分。lethal層は生徒main.pyに残っている）。
"""

from __future__ import annotations

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

import features_032 as F  # noqa: E402
import model_032 as M     # noqa: E402

# 1イテレーションあたりの収集ゲーム数（プール構成の粗い反映、EXP-031の半分）
GAMES = {"mirror": 600, "002": 400, "010": 400, "017": 200, "021": 200}
TEACHER_DIR = "agents/007_lucario_lethal"
BASE_DATA = AGENT_DIR / "data_031.npz"   # EXP-031の教師データ（DAggerイテレーション0相当）
WEIGHTS_PATH = AGENT_DIR / "weights_032.npz"


def _iter_path(it: int) -> Path:
    return AGENT_DIR / f"dagger_iter{it}.npz"


def collect(it: int) -> None:
    from run_match import load_agent_module, read_deck
    from cg.game import battle_start, battle_select, battle_finish
    from cg.api import to_observation_class

    params = M.load_params(str(WEIGHTS_PATH))
    student_deck = read_deck(str(AGENT_DIR))
    teacher_agent = load_agent_module(str(REPO / TEACHER_DIR), "T")
    teacher_mod = sys.modules["agent_T"]
    # 探索を常にスキップ（_search_time_used=0.0 > -1.0 で発火しない）。収集高速化
    teacher_mod.SEARCH_GAME_BUDGET = -1.0

    opp_dirs = {
        "002": "agents/002_marnie_grimmsnarl",
        "010": "agents/010_alakazam",
        "017": "agents/017_starmie",
        "021": "agents/021_dragapult_sample",
    }
    opponents = {}
    for tag, d in opp_dirs.items():
        opponents[tag] = (load_agent_module(str(REPO / d), f"O{tag}"), read_deck(str(REPO / d)))

    ctx_id, my_act, op_act, ctx_sc, labels = [], [], [], [], []
    off = [0]
    o_type, o_card, o_target, o_attack, o_sc = [], [], [], [], []
    n_agree = 0   # 生徒と教師の一致数（オンポリシー分布での実効クローン精度）

    random.seed(20260710 + it)
    total = sum(GAMES.values())
    t_start = time.time()
    game_no = 0
    skipped = 0
    for tag, n_games in GAMES.items():
        if tag == "mirror":
            opp_agent, opp_deck = None, student_deck
        else:
            opp_agent, opp_deck = opponents[tag]
        for _ in range(n_games):
            game_no += 1
            s_seat = game_no % 2   # 生徒の席（mirrorでは両席とも生徒）
            deck0 = student_deck if s_seat == 0 else opp_deck
            deck1 = opp_deck if s_seat == 0 else student_deck
            obs_dict, _ = battle_start(deck0, deck1)
            try:
                if obs_dict is None:
                    skipped += 1
                    continue
                steps = 0
                while obs_dict["current"]["result"] == -1 and steps < 30000:
                    player = obs_dict["current"]["yourIndex"]
                    is_student = (tag == "mirror") or (player == s_seat)
                    if is_student:
                        obs = to_observation_class(obs_dict)
                        select = obs.select
                        n_opt = len(select.option)
                        feats = None
                        if 2 <= n_opt <= M.P_MAX:
                            try:
                                feats = F.extract(obs)
                            except Exception:
                                feats = None
                        # --- 生徒の選択（純NN、main.pyの_nn_choose相当） ---
                        order = None
                        if feats is not None:
                            try:
                                X, mask, _ = M.build_batch(
                                    params, [feats[0]], [feats[1]], pad_to=n_opt)
                                logits, _ = M.forward(params, X, mask)
                                order = list(map(int, logits[0, :n_opt].argsort()[::-1]))
                            except Exception:
                                order = None
                        if order is None:
                            order = list(range(n_opt))
                        sel = order[: select.maxCount] if (n_opt and select.maxCount) else []
                        # --- 教師ラベル（DAgger: 状態は生徒のオンポリシー） ---
                        if feats is not None and sel:
                            try:
                                t_sel = teacher_agent(obs_dict)
                            except Exception:
                                t_sel = None
                            if t_sel and isinstance(t_sel, list) and 0 <= t_sel[0] < n_opt:
                                (cid, ma, oa, csc), opts = feats
                                ctx_id.append(cid); my_act.append(ma); op_act.append(oa)
                                ctx_sc.append(csc); labels.append(t_sel[0])
                                for ot, c, t, a, s in opts:
                                    o_type.append(ot); o_card.append(c)
                                    o_target.append(t); o_attack.append(a); o_sc.append(s)
                                off.append(len(o_type))
                                if t_sel[0] == sel[0]:
                                    n_agree += 1
                    else:
                        sel = opp_agent(obs_dict)
                    obs_dict = battle_select(sel)
                    steps += 1
            except Exception as e:
                skipped += 1
                print(f"  iter{it}: game {game_no} ({tag}) aborted: "
                      f"{type(e).__name__}: {e}", flush=True)
            finally:
                battle_finish()
            if game_no % 100 == 0:
                agree = n_agree / max(1, len(labels))
                print(f"  iter{it}: {game_no}/{total} games ({tag}), {len(labels)} selects, "
                      f"on-policy agree={agree:.3f}, {time.time() - t_start:.0f}s", flush=True)

    np.savez_compressed(
        _iter_path(it),
        ctx_id=np.array(ctx_id, np.int32), my_act=np.array(my_act, np.int32),
        op_act=np.array(op_act, np.int32), ctx_sc=np.array(ctx_sc, np.float32),
        labels=np.array(labels, np.int32), off=np.array(off, np.int64),
        o_type=np.array(o_type, np.int32), o_card=np.array(o_card, np.int32),
        o_target=np.array(o_target, np.int32), o_attack=np.array(o_attack, np.int32),
        o_sc=np.array(o_sc, np.float32),
    )
    agree = n_agree / max(1, len(labels))
    print(f"saved: {_iter_path(it)} ({len(labels)} selects, on-policy agree={agree:.4f}, "
          f"skipped={skipped})", flush=True)


def _load_all(up_to: int) -> dict:
    """data_031 + dagger_iter1..up_to を統合。
    NpzFileはアクセス毎にzip解凍が走る（1バッチ84秒の主犯）→ 最初に全部メモリへ展開。"""
    paths = [BASE_DATA] + [_iter_path(k) for k in range(1, up_to + 1)]
    parts = []
    for p in paths:
        z = {k: v for k, v in np.load(p).items()}
        parts.append(z)
        print(f"  loaded {p.name}: {len(z['labels'])} selects", flush=True)
    keys = ["ctx_id", "my_act", "op_act", "ctx_sc", "labels",
            "o_type", "o_card", "o_target", "o_attack", "o_sc"]
    merged = {k: np.concatenate([z[k] for z in parts]) for k in keys}
    offs = [parts[0]["off"]]
    shift = parts[0]["off"][-1]
    for z in parts[1:]:
        offs.append(z["off"][1:] + shift)
        shift += z["off"][-1]
    merged["off"] = np.concatenate(offs)
    assert merged["off"][-1] == len(merged["o_type"])
    assert len(merged["off"]) == len(merged["labels"]) + 1
    return merged


def _select_slice(z, i):
    s, e = z["off"][i], z["off"][i + 1]
    ctx = (int(z["ctx_id"][i]), int(z["my_act"][i]), int(z["op_act"][i]), z["ctx_sc"][i])
    opts = list(zip(z["o_type"][s:e], z["o_card"][s:e], z["o_target"][s:e],
                    z["o_attack"][s:e], z["o_sc"][s:e]))
    return ctx, opts


def train(up_to: int, epochs: int = 4, batch: int = 256, lr: float = 1e-3) -> None:
    z = _load_all(up_to)
    n = len(z["labels"])
    rng = np.random.default_rng(7)
    order = rng.permutation(n)
    cut = int(n * 0.95)
    tr, te = order[:cut], order[cut:]
    print(f"train={len(tr)} holdout={len(te)}", flush=True)

    p = M.init_params(seed=0)   # 毎イテレーションゼロから再学習
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
            if nb % 400 == 0:
                print(f"  ep{ep + 1} {nb}/{n_batches} loss={tot_loss / nb:.4f} "
                      f"acc={tot_acc / nb:.4f}", flush=True)
        print(f"epoch {ep + 1}: loss={tot_loss / nb:.4f} acc={tot_acc / nb:.4f}", flush=True)
        M.save_params(str(WEIGHTS_PATH), p)  # 途中停止に備えたエポック毎チェックポイント

    correct, total = {}, {}
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
    for c in sorted(total, key=lambda c: -total[c])[:12]:
        print(f"  ctx {c}: acc={correct[c] / total[c]:.3f} (n={total[c]})")

    M.save_params(str(WEIGHTS_PATH), p)
    print(f"saved: {WEIGHTS_PATH}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    it = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    if cmd == "collect":
        collect(it)
    elif cmd == "train":
        train(it)
    else:
        sys.exit("usage: train_032.py collect|train <iter>")
