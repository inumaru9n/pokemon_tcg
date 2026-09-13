"""学習した torch モデルと、提出物が使う numpy 推論が **同じスコアを出すか** を検証する。

**なぜ必要か**: 推論経路は3本ある（torch=学習/PPO、numpy単発=`nn_*.py`、numpy batch=
`batch_policy.py`）。特徴の標準化や Dropout の追加は、どれか1本に入れ忘れても例外を
出さずに黙って別の方策になる。EXP-096 では「次元が同じなら同じもの」と思い込んで
クラスIDのずれを見逃し、無効な測定値（68.4%）を作った。**数値一致で確かめる**。

usage:
  uv run python selfplay/check_export.py --npz data/nn/train102.npz \\
      --model agents/102_alakazam_feat2/model_102.npz --agent-dir agents/102_alakazam_feat2
"""

from __future__ import annotations

import argparse
import importlib.util as ilu
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO, os.path.join(REPO, "selfplay")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", required=True, help="学習に使ったデータセット")
    ap.add_argument("--model", required=True)
    ap.add_argument("--agent-dir", required=True)
    ap.add_argument("--nn-mod", default=None, help="既定は agent-dir 内の nn_*.py")
    ap.add_argument("-n", type=int, default=64, help="照合する決定数")
    a = ap.parse_args()

    import torch

    from selfplay.batch_policy import BatchLSTMPolicy
    from selfplay.train_nn_kaggle import PolicyNet, build_vocab

    nnp = a.nn_mod or next(f for f in sorted(os.listdir(a.agent_dir))
                           if f.startswith("nn_") and f.endswith(".py"))
    sp = ilu.spec_from_file_location("nn_x", os.path.join(a.agent_dir, nnp))
    nn_mod = ilu.module_from_spec(sp)
    sp.loader.exec_module(nn_mod)

    z = np.load(a.npz, allow_pickle=True)
    data = {k: z[k] for k in z.files}
    w = np.load(a.model, allow_pickle=False)
    wd = {k: w[k] for k in w.files}
    has_norm = "board_mu" in wd
    print(f"標準化統計: {'あり' if has_norm else 'なし'} / "
          f"線形層 enc: {sorted(int(k.split('.')[1]) for k in wd if k.startswith('enc.') and k.endswith('.weight'))}")

    # ---- torch 側を npz から復元 ----
    vocab = {int(k): int(v) for k, v in zip(wd["vocab_keys"], wd["vocab_vals"])}
    n_cards = wd["card_emb.weight"].shape[0]
    d_model = wd["lstm.weight_hh_l0"].shape[1]
    n_layers = sum(1 for k in wd if k.startswith("lstm.weight_ih_l"))
    d_card = wd["card_emb.weight"].shape[1]
    enc_idx = sorted(int(k.split(".")[1]) for k in wd
                     if k.startswith("opt_enc.") and k.endswith(".weight"))
    d_opt = wd[f"opt_enc.{enc_idx[0]}.weight"].shape[0]
    n_optdim = wd[f"opt_enc.{enc_idx[0]}.weight"].shape[1] - d_card
    n_cls = wd["cls_head.weight"].shape[0] if "cls_head.weight" in wd else 0
    # Dropout を入れて学習したかは Sequential の番号から判る（0,3 ならDropout有り）
    dropout = 0.0 if enc_idx == [0, 2] else 1e-9
    head = str(wd["head"]) if "head" in wd else "additive"
    print(f"MAIN階層の合成形: {head}")
    net = PolicyNet(n_board=int(wd["n_board"]), n_logs=int(wd["n_logs"]),
                    n_optdim=n_optdim, n_cards=n_cards,
                    n_ctx=wd["ctx_emb.weight"].shape[0], d_card=d_card,
                    d_opt=d_opt, d_model=d_model, n_layers=n_layers, n_cls=n_cls,
                    dropout=dropout, head=head)
    sd = net.state_dict()
    miss = [k for k in sd if k not in wd and k not in
            ("board_mu", "board_sd", "opt_mu", "opt_sd")]
    if miss:
        raise SystemExit(f"npzに無い重み: {miss}")
    net.load_state_dict({k: torch.as_tensor(np.asarray(wd[k], np.float32))
                         for k in sd})
    net.eval()

    # ---- 同じ決定列を3経路に通す ----
    OFF = 3
    lut = np.zeros(int(max(vocab)) + OFF + 1, np.int64)
    for k, v in vocab.items():
        lut[k + OFF] = v
    gid = data["gidx"]
    g0 = gid[0]
    idx = np.where(gid == g0)[0][:a.n]
    O = data["opt_cid"].shape[1]

    one = nn_mod.LSTMPolicy(a.model)
    bat = BatchLSTMPolicy(a.model, 1)
    one.reset()
    bat.reset_all()

    bmu = wd.get("board_mu")
    bsd = wd.get("board_sd")
    omu = wd.get("opt_mu")
    osd = wd.get("opt_sd")

    st = None
    d_one, d_bat = [], []
    for i in idx:
        no = int(data["nopt"][i])
        board = data["board"][i]
        ov = data["opt_vec"][i][:no]
        cids = data["opt_cid"][i][:no]
        cls = data["opt_cls"][i][:no] if "opt_cls" in data else None
        ctx = int(data["ctx"][i])

        # torch: 標準化は呼び出し側で（学習時に data を正規化して食わせているため）
        tb = (board - bmu) / bsd if bmu is not None else board
        to = (ov - omu) / osd if omu is not None else ov
        t = lambda x, dt=torch.float32: torch.as_tensor(  # noqa: E731
            np.asarray(x)[None, None], dtype=dt)
        pad_c = np.zeros(O, np.int64)
        pad_c[:no] = lut[cids + OFF]
        pad_v = np.zeros((O, n_optdim), np.float32)
        pad_v[:no] = to
        pad_m = np.zeros(O, bool)
        pad_m[:no] = True
        pad_l = np.full(O, -1, np.int64)
        if cls is not None:
            pad_l[:no] = cls
        with torch.no_grad():
            sc_t, st, _, _v = net(
                t(tb), t(data["logs"][i]), t(lut[int(data["prev"][i, 0]) + OFF],
                                             torch.long),
                t(data["prev"][i, 1:3]), t(ctx, torch.long),
                t(pad_c, torch.long), t(pad_v), t(pad_m, torch.bool),
                state=st, opt_cls=t(pad_l, torch.long) if n_cls else None)
        ref = sc_t[0, 0, :no].numpy()

        s1 = one.scores(board, data["logs"][i], data["prev"][i], ctx,
                        cids.tolist(), ov,
                        cls.tolist() if cls is not None else None)
        s2 = bat.scores([0], [board], [data["logs"][i]], [data["prev"][i]], [ctx],
                        [cids.tolist()], [ov],
                        [cls.tolist() if cls is not None else None])[0]
        d_one.append(np.abs(np.asarray(s1) - ref).max())
        d_bat.append(np.abs(np.asarray(s2) - ref).max())

    d1, d2 = float(np.max(d_one)), float(np.max(d_bat))
    print(f"照合 {len(idx)} 決定")
    print(f"  torch vs nn_*.py(単発numpy)   最大差 {d1:.3e}")
    print(f"  torch vs batch_policy(batch)  最大差 {d2:.3e}")
    tol = 2e-3
    if d1 > tol or d2 > tol:
        raise SystemExit(f"不一致（許容 {tol}）。推論経路のどれかが学習と違う計算をしている")
    print("OK: 3経路が一致")


if __name__ == "__main__":
    main()
