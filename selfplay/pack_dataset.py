"""EXP-094: extract_yushin.py の pickle を Kaggle アップロード用の npz に詰め直す。

pickle（可変長のPythonオブジェクト）→ 固定長パディング済みの numpy 配列群。
Kaggle Notebook 側は numpy だけで読めるようにする（torch非依存のロード）。

レイアウト（N=全決定数、O=最大オプション数）:
  board   (N, F)      float32   盤面特徴
  logs    (N, 48)     float32   前回決定以降のlogs要約
  prev    (N, 3)      float32   前の選択 (card_id, opt_type, area)
  ctx     (N,)        int16     コンテキストID
  turn    (N,)        int16
  nopt    (N,)        int16     実オプション数
  opt_cid (N, O)      int32     オプションのカードID（-1=解決不能, パディングは-2）
  opt_vec (N, O, D)   float32   オプション数値特徴
  chosen  (N, O)      int8      教師が選んだか（複数選択ctxは複数1）
  first   (N,)        int8      そのゲームの最初の決定なら1（LSTM状態リセット位置）
  gidx    (N,)        int32     ゲーム番号（系列の切り出し用）
  gres    (G,)        int8      ゲームの勝敗（1=win, 0=draw, -1=loss）

usage:
  uv run python selfplay/pack_dataset.py --pkl data/nn/train.pkl --out data/nn/train.npz
"""

from __future__ import annotations

import argparse
import os
import pickle

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    # **複数指定で結合できる**（対面別の教師混合。EXP-110）。
    # 同一デッキ・同一特徴器で抽出した pickle 同士に限る。
    # ゲーム単位で連結するのでLSTMの系列は壊れない。
    ap.add_argument("--pkl", required=True, nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-opt", type=int, default=0,
                    help="最大オプション数（0=データから決定）")
    ap.add_argument("--n-cls", type=int, default=0,
                    help="行動クラス総数（0=pickleの記録、無ければ観測max+1）")
    # **新規データ（本番リプレイの勝ち試合）の混合比**（EXP-114）。
    # 元データが 19:1 と多いので、目標比になるよう**元データをランダムに間引く**。
    # 間引きは**ゲーム単位**（LSTMの系列を壊さない）。決定数ベースで比率を合わせる。
    ap.add_argument("--new-ratio", type=float, default=0.0,
                    help="新規データが占める目標割合（決定数ベース）。例 0.2 / 0.4。0で無効")
    ap.add_argument("--sample-seed", type=int, default=0)
    # **勝敗による選別**（EXP-116）。勝ち試合だけで学習すると強くなるかを測る。
    # 注意: 試合数が減るので「勝ちで選別した効果」と「データ量が減った効果」が
    # 交絡する。--match-size で全体データを同数までランダムに間引いた対照を作れる。
    ap.add_argument("--only", default=None, choices=("win", "loss"),
                    help="この結果の試合だけを使う")
    ap.add_argument("--match-decisions", type=int, default=0,
                    help="この決定数までランダムに間引く（データ量を揃えた対照用）")
    args = ap.parse_args()

    games = []
    for p in args.pkl:
        with open(p, "rb") as f:
            d = pickle.load(f)
        g = d["games"]
        # **特徴の形が違う pickle を混ぜると黙って壊れる**ので次元を突合する
        if games:
            a, b = games[0]["decisions"][0], g[0]["decisions"][0]
            assert len(a["board"]) == len(b["board"]), \
                f"board次元が違う: {len(a['board'])} vs {len(b['board'])}"
            assert len(a["logs"]) == len(b["logs"]), "logs次元が違う"
            assert len(a["opt_vec"][0]) == len(b["opt_vec"][0]), "opt_vec次元が違う"
        print(f"  {p}: games={len(g)} "
              f"decisions={sum(len(x['decisions']) for x in g)} "
              f"teacher={d.get('teacher')}")
        games += g

    # ---- 新規データの混合比を作る（元データをランダムに間引く） ----
    if args.new_ratio > 0:
        rng = np.random.default_rng(args.sample_seed)
        new_g = [g for g in games if str(g.get("date", "")).startswith("mine:")]
        old_g = [g for g in games if not str(g.get("date", "")).startswith("mine:")]
        n_new = sum(len(g["decisions"]) for g in new_g)
        n_old = sum(len(g["decisions"]) for g in old_g)
        assert n_new > 0, "新規データ（mine:）が1件も無い"
        # 目標: n_new / (n_new + keep) = ratio  →  keep = n_new * (1-ratio) / ratio
        want = n_new * (1.0 - args.new_ratio) / args.new_ratio
        if want >= n_old:
            print(f"  元データ({n_old}決定)が目標({want:.0f})以下なので間引かない")
            keep_g = old_g
        else:
            idx = rng.permutation(len(old_g))
            keep_g, acc = [], 0
            for j in idx:
                if acc >= want:
                    break
                keep_g.append(old_g[j])
                acc += len(old_g[j]["decisions"])
            print(f"  元データを {len(old_g)}→{len(keep_g)}試合に間引き "
                  f"({n_old}→{acc}決定)")
        games = keep_g + new_g
        tot = sum(len(g["decisions"]) for g in games)
        print(f"  **混合比: 新規 {n_new}/{tot} = {n_new/tot:.1%}"
              f"（目標 {args.new_ratio:.0%}）** / 新規{len(new_g)}試合 + 元{len(keep_g)}試合")

    if args.only:
        before = len(games)
        games = [g for g in games if g.get("result") == args.only]
        print(f"  結果で選別: {before} → {len(games)}試合（{args.only} のみ）")
        assert games, f"result=={args.only} の試合が無い"
    if args.match_decisions > 0:
        rng2 = np.random.default_rng(args.sample_seed)
        idx = rng2.permutation(len(games))
        keep, acc = [], 0
        for j in idx:
            if acc >= args.match_decisions:
                break
            keep.append(games[j]); acc += len(games[j]["decisions"])
        print(f"  決定数を揃える: {len(games)}→{len(keep)}試合 "
              f"({sum(len(g['decisions']) for g in games)}→{acc}決定)")
        games = keep

    n_dec = sum(len(g["decisions"]) for g in games)
    max_opt = args.max_opt or max(len(dd["opt_cid"])
                                  for g in games for dd in g["decisions"])
    F = len(games[0]["decisions"][0]["board"])
    L = len(games[0]["decisions"][0]["logs"])
    D = len(games[0]["decisions"][0]["opt_vec"][0])
    print(f"games={len(games)} decisions={n_dec} board={F} logs={L} "
          f"opt_dim={D} max_opt={max_opt}")

    board = np.zeros((n_dec, F), np.float32)
    logs = np.zeros((n_dec, L), np.float32)
    # prev は3列（自分のみ）または**6列（+相手の直前選択、EXP-133）**
    _PW = len(games[0]["decisions"][0]["prev"])
    prev = np.zeros((n_dec, _PW), np.float32)
    ctx = np.zeros(n_dec, np.int16)
    turn = np.zeros(n_dec, np.int16)
    nopt = np.zeros(n_dec, np.int16)
    opt_cid = np.full((n_dec, max_opt), -2, np.int32)
    opt_vec = np.zeros((n_dec, max_opt, D), np.float32)
    opt_cls = np.full((n_dec, max_opt), -1, np.int16)  # MAINの行動クラス（非MAINは-1）
    chosen = np.zeros((n_dec, max_opt), np.int8)
    first = np.zeros(n_dec, np.int8)
    gidx = np.zeros(n_dec, np.int32)
    is_new = np.zeros(n_dec, np.int8)
    gres = np.zeros(len(games), np.int8)
    # **新規データ（本番リプレイの勝ち試合）かどうか**。学習時の重み付けに使う（EXP-114）
    g_new = np.zeros(len(games), np.int8)

    i = 0
    for gi, g in enumerate(games):
        gres[gi] = {"win": 1, "loss": -1}.get(g["result"], 0)
        g_new[gi] = 1 if str(g.get("date", "")).startswith("mine:") else 0
        for k, dd in enumerate(g["decisions"]):
            m = len(dd["opt_cid"])
            if m > max_opt:      # --max-opt指定で切り詰めた場合、教師選択を含む範囲のみ採用
                if max(dd["chosen"], default=0) >= max_opt:
                    continue
                m = max_opt
            board[i] = dd["board"]
            logs[i] = dd["logs"]
            prev[i] = dd["prev"]
            ctx[i] = dd["ctx"]
            turn[i] = dd["turn"]
            nopt[i] = m
            opt_cid[i, :m] = dd["opt_cid"][:m]
            opt_vec[i, :m] = np.asarray(dd["opt_vec"][:m], np.float32)
            opt_cls[i, :m] = (dd.get("opt_cls") or [-1] * m)[:m]
            for c in dd["chosen"]:
                if c < m:
                    chosen[i, c] = 1
            first[i] = 1 if k == 0 else 0
            gidx[i] = gi
            is_new[i] = g_new[gi]
            i += 1
    n_dec = i  # 切り詰め分を反映

    # 行動クラス総数。**観測max+1では足りないことがある**（末尾クラスが教師データに
    # 現れないと分類ヘッドが狭くなり、推論で範囲外indexになる）ので特徴器の値を優先する。
    n_cls = args.n_cls or int(d.get("n_cls") or 0) or int(opt_cls[:n_dec].max()) + 1
    if n_cls < int(opt_cls[:n_dec].max()) + 1:
        raise SystemExit(f"n_cls={n_cls} が観測クラス最大 "
                         f"{int(opt_cls[:n_dec].max())} より小さい")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(
        args.out,
        n_cls=np.int32(n_cls),
        board=board[:n_dec], logs=logs[:n_dec], prev=prev[:n_dec],
        ctx=ctx[:n_dec], turn=turn[:n_dec], nopt=nopt[:n_dec],
        opt_cid=opt_cid[:n_dec], opt_vec=opt_vec[:n_dec],
        opt_cls=opt_cls[:n_dec],
        chosen=chosen[:n_dec], first=first[:n_dec], gidx=gidx[:n_dec],
        gres=gres,
        feat_keys=np.array(d.get("feat_keys") or [], dtype=object),
        # **行動クラスの名前列**。IDでなく名前で持つことで、推論時に特徴器の
        # クラス分類が学習時とズレていないかを検証できる（クラス数の一致だけでは
        # 名前の変化によるID全体のずれを見逃す。EXP-096で実際に踏んだ）
        classes=np.array([str(x) for x in (d.get("classes") or [])], dtype=object),
        meta=np.array([str(d.get("teacher")), str(d.get("dates"))], dtype=object),
        is_new=is_new[:n_dec], g_new=g_new)
    mb = os.path.getsize(args.out) / 1e6
    print(f"saved -> {args.out} ({mb:.1f} MB), decisions={n_dec}, n_cls={n_cls} "
          f"(観測max={int(opt_cls[:n_dec].max())})")

    # 健全性チェック
    ok = int((chosen[:n_dec].sum(1) > 0).sum())
    print(f"教師選択が範囲内の決定: {ok}/{n_dec} ({ok/n_dec:.1%})")
    print(f"ctx別: {dict(zip(*np.unique(ctx[:n_dec], return_counts=True)))}")


if __name__ == "__main__":
    main()
