"""EXP-094 Step2: Yushin Ito (sig 156952a871) の全コンテキスト決定を抽出する。

AlphaStar型の全選択NN方策（LSTM）の教師データを作る。EXP-057系の「MAINのクラスだけ」と違い、
**全ての select() 決定**（MAIN/サーチ先/捨て札/交代/ダメカン配分…）を対象にする。

1決定 = (盤面特徴, logs要約, 前の選択, 各オプションの特徴, 教師が選んだindex)。
LSTMのため**ゲーム単位で決定列の順序を保つ**（1ゲーム=1系列）。

usage:
  uv run python selfplay/extract_yushin.py --dates 2026-07-17 ... --out data/nn/train.pkl -w 4
  uv run python selfplay/extract_yushin.py --dates 2026-07-22 --census   # ctx別決定数だけ出す
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pickle
import sys
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

REPO_ROOT = "/Users/akira/kaggle/pokemon_tcg"
# **既定値を置かない**。以前は 094（126次元）が既定だったため、指定を忘れると
# 黙って古い特徴器で抽出され、506次元のモデルを作るつもりが126次元になっていた
# （例外は出ず、学習も通る）。EXP-097 の +10.5pt は特徴器の拡張によるものなので、
# ここを取り違えると成果がまるごと消える。**必ず明示させる**。
AGENT_DIR = os.environ.get("EXTRACT_AGENT_DIR")
FEAT_MOD = os.environ.get("EXTRACT_FEAT_MOD")   # 特徴の唯一の定義元
if not AGENT_DIR or not FEAT_MOD:
    raise SystemExit(
        "EXTRACT_AGENT_DIR と EXTRACT_FEAT_MOD を必ず指定すること。\n"
        "  例: EXTRACT_AGENT_DIR=$PWD/agents/104_alakazam_feat4 EXTRACT_FEAT_MOD=feat_104\n"
        "（既定値を置くと、指定忘れが古い特徴器での抽出として黙って通ってしまう）")
for _p in (REPO_ROOT, AGENT_DIR, os.path.join(REPO_ROOT, "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

TEACHER_TEAM = "Yushin Ito"
TEACHER_SIG = "156952a871"

_W: dict = {}


def _ensure_worker():
    if _W.get("loaded"):
        return
    # 特徴は agents/094_yushin_nn/feat_094.py を唯一の定義元とする
    # （抽出と推論で完全に同一のコードを使うため）
    if AGENT_DIR not in sys.path:
        sys.path.insert(0, AGENT_DIR)
    spec = importlib.util.spec_from_file_location(
        FEAT_MOD, os.path.join(AGENT_DIR, f"{FEAT_MOD}.py"))
    ft = importlib.util.module_from_spec(spec)
    sys.modules[FEAT_MOD] = ft
    spec.loader.exec_module(ft)
    from cg.api import to_observation_class
    _W["ft"] = ft
    _W["to_obs"] = to_observation_class
    _W["zips"] = {}
    _W["loaded"] = True


def _zip_for(date: str) -> zipfile.ZipFile:
    if date not in _W["zips"]:
        d = os.path.join(REPO_ROOT, "data", "episodes", date)
        zpath = next(os.path.join(d, x) for x in os.listdir(d) if x.endswith(".zip"))
        _W["zips"][date] = zipfile.ZipFile(zpath)
    return _W["zips"][date]


def process_episode(task: tuple) -> dict:
    (date, ep_id, me, meta) = task
    _ensure_worker()
    ft = _W["ft"]
    try:
        if date.startswith("mine:"):
            # **自分の提出のリプレイ**（data/my_episodes/<sid>/<ep>.json、個別ファイル）。
            # 日次データセットと違いレートに関係なく全対戦が取れる（EXP-114）。
            _p = os.path.join(REPO_ROOT, "data", "my_episodes",
                              date.split(":", 1)[1], f"{ep_id}.json")
            steps = json.load(open(_p))["steps"]
        else:
            steps = json.loads(_zip_for(date).read(f"{ep_id}.json"))["steps"]
    except (KeyError, OSError) as e:
        return {"game": None, "err": f"{ep_id}: {e}"}

    decisions = []
    # 前半3 = 自分の前の選択 (card_id, opt_type, area)
    # **後半3 = 相手の直前の選択 (card_id, log_type, toArea)**（EXP-133）。
    # 生ログには cardId があるのに logs_vector は LogType のカウントしか取らず、
    # **相手側だけカードの識別が無い**という非対称があった。持ち越し方式で密度93%。
    # **main.py（Observationオブジェクト・属性アクセス）と同じ結果になること**。
    prev = [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
    errs = []
    for t, st in enumerate(steps):
        s = st[me]
        if s.get("status") != "ACTIVE":
            continue
        obs_d = s.get("observation") or {}
        sel = obs_d.get("select")
        cur = obs_d.get("current")
        if not sel or not cur:
            continue
        if t + 1 >= len(steps):
            continue
        actual = steps[t + 1][me].get("action")
        if not isinstance(actual, list):
            continue
        nopt = len(sel.get("option") or [])
        # **空action（棄権）も教師例として残す**。minCount=0の決定は「してもしなくてよい」で、
        # 教師はctx2 SETUP_BENCHの36.3%で「置かない」を選ぶ（ベンチを増やしてボスの指令の
        # 的を作らない判断）。捨てると方策がこの行動を表現できなくなる。
        is_decline = (len(actual) == 0)
        can_decline = int(sel.get("minCount", 1)) == 0
        if is_decline and not can_decline:
            continue                      # minCount>0で空actionは異常データ
        # 選択の余地が無い（強制手）決定は学習対象外。ただし **minCount=0 なら
        # 選択肢1個でも「やる/やらない」の実質2択**なので必ず残す。
        # ここを nopt<2 で切ると「教師が実行した1択」だけが落ち、残るのは棄権例のみ＝
        # 「選択肢1個で棄権可能なら必ず棄権する」という嘘を学ぶ（v7の穴。実測で
        # 400試合中 実行729件が脱落し棄権75件だけが残っていた）。
        if nopt < 1 or (nopt < 2 and not can_decline):
            continue
        try:
            obs = _W["to_obs"](obs_d)
            state = obs.current
            my_i = state.yourIndex
            board, _keys = ft.feat_vector(state, my_i)
            logs = ft.logs_vector(obs, my_i)
            # **相手の直前の選択を持ち越す**（EXP-133）。main.py と**同一の走査**:
            # logs を末尾から見て、相手の playerIndex かつ cardId を持つ最初のログ。
            for _lg in reversed(list(getattr(obs, "logs", None) or [])):
                _pi = getattr(_lg, "playerIndex", None)
                _cd = getattr(_lg, "cardId", None)
                if _pi is not None and _pi != my_i and _cd is not None:
                    _t = getattr(_lg, "type", None)
                    _a = getattr(_lg, "toArea", None)
                    prev[3] = float(_cd)
                    prev[4] = float(int(_t)) if _t is not None else -1.0
                    prev[5] = float(int(_a)) if _a is not None else -1.0
                    break
            hand_counts, pool_left = ft.pool_counts(state, my_i)
            cids, ovecs = [], []
            for i in range(nopt):
                cid, v = ft.option_vector(obs, obs.select.option[i], my_i,
                                          hand_counts, pool_left)
                cids.append(cid)
                ovecs.append(v)
            # MAINは「行動タイプ×対象」が1回のselectに圧縮されているため、
            # エンジンは階層を提供してくれない。EXP-057の37クラスを付与して
            # モデル側で階層化できるようにする（非MAINは-1）。
            ocls = [-1] * nopt
            if int(sel.get("context", -1)) == 0:
                for i in range(nopt):
                    try:
                        ocls[i] = ft.CLASS_ID.get(
                            ft.option_class(state, obs.select, i, my_i), -1)
                    except Exception:  # noqa: BLE001
                        ocls[i] = -1
            # **棄権(NULL)オプションを末尾に追加**（minCount==0 のときのみ）。
            # pointer networkの標準的なstop action。これが無いと「選ばない」を表現できず、
            # 学習にも推論にも棄権が存在しなくなる。
            if can_decline:
                cids.append(-3)                       # NULL専用のID
                ovecs.append(ft.null_option_vector())
                ocls.append(-1)
                nopt += 1
            chosen_idx = [int(x) for x in actual if 0 <= int(x) < nopt]
            if is_decline:
                chosen_idx = [nopt - 1]               # NULLを選んだ扱い
            decisions.append({
                "ctx": int(sel.get("context", -1)),
                "turn": int(cur.get("turn", 0)),
                "mn": int(sel.get("minCount", 1)),
                "mx": int(sel.get("maxCount", 1)),
                "can_decline": 1 if can_decline else 0,
                "board": board,
                "logs": logs,
                "prev": list(prev),   # 6要素（後半3が相手の直前選択）
                "opt_cid": cids,
                "opt_vec": ovecs,
                "opt_cls": ocls,
                "chosen": chosen_idx,
            })
            if chosen_idx:
                c0 = chosen_idx[0]
                prev = [float(cids[c0]), ovecs[c0][0], ovecs[c0][1]] + prev[3:6]
        except Exception as e:  # noqa: BLE001
            errs.append(f"{ep_id}@{t}: {type(e).__name__}: {e}")
            continue
    if not decisions:
        return {"game": None, "err": errs[0] if errs else f"{ep_id}: no decisions"}
    return {"game": {"ep": ep_id, "date": date, "result": meta["result"],
                     "opp_arch": meta.get("opp_arch", "?"),
                     "decisions": decisions},
            "keys": ft.FEAT_KEYS,   # 特徴順の記録（親では確定しないのでワーカーから返す）
            # **クラス総数を特徴器から取る**。学習側で「観測されたクラスの最大+1」から
            # 推定すると、末尾のクラスが教師データに現れないときヘッド幅が足りず、
            # 推論時に範囲外indexで落ちる（生成特徴器はクラスが疎になるので現実的な穴）
            "n_cls": len(getattr(ft, "CLASSES", ()) or ()),
            "classes": list(getattr(ft, "CLASSES", ()) or ()),  # ID→名前（診断用）
            "err": errs[0] if errs else None}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dates", nargs="+", default=[])
    # **自分の提出リプレイ**（data/my_episodes/<sid>/）。レートに関係なく全対戦が取れる。
    # 継続学習の教師にするので既定は勝ち試合のみ（EXP-114）。
    ap.add_argument("--mine", nargs="+", default=[],
                    help="submission_id を並べる。勝ち試合だけを採用する")
    ap.add_argument("--mine-all", action="store_true", help="負け試合も含める")
    ap.add_argument("--my-team", default="haruto")
    # **複数team指定可**。チーム統合/改名で同一デッキ(sig)の稼働が別teamに分かれることがある
    # （例: James Cox → James Cox & Henry Chao、sig c64b28ca9d、07-28前後）。
    # 同一sigかつ期間が連続なら同一系統とみなして結合できる。
    ap.add_argument("--team", nargs="+", default=[TEACHER_TEAM])
    ap.add_argument("--sig", default=TEACHER_SIG)
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--census", action="store_true", help="ctx別決定数だけ出して終了")
    # **対戦相手のアーキタイプで絞る**（対面別の教師混合のため。EXP-110）。
    # 同一デッキでも対面ごとに最良の個体は違う（実測: sig 156952a871 の対Spidopsは
    # Yushin 17.9% に対し Majkel 53.4%＝35.5pt差）。対面ごとに最良の教師のデータだけを
    # 集めれば、**どの単一個体よりも強い方策**を1つのモデルに学習させられる。
    # 特徴器に相手アーキタイプ（oparch_*）が入っているので条件付けは学習側で解ける。
    ap.add_argument("--opp-arch", nargs="+", default=None, dest="opp_arch",
                    help="この相手アーキタイプの試合**だけ**を使う（部分一致）")
    ap.add_argument("--exclude-opp-arch", nargs="+", default=None, dest="excl_arch",
                    help="この相手アーキタイプの試合を**除く**（部分一致）")
    ap.add_argument("-w", "--workers", type=int, default=4)
    args = ap.parse_args()

    from agreement import find_episodes
    dates = [d for d in args.dates
             if os.path.isdir(os.path.join(REPO_ROOT, "data", "episodes", d))]
    missing = sorted(set(args.dates) - set(dates))
    if missing:
        print(f"WARN: 未取得の日付をスキップ: {missing}")
    eps = []
    if args.mine:
        from list_my_games import list_games
        for sid in args.mine:
            g, st = list_games(sid, args.my_team, not args.mine_all)
            print(f"  自分の提出 {sid}: {dict(st)} → 採用 {len(g)}件")
            eps.extend(g)
    for tm in (args.team if args.dates else []):
        got = find_episodes(dates, tm, args.sig)
        print(f"  team={tm!r}: {len(got)} episodes")
        eps.extend(got)
    if args.opp_arch or args.excl_arch:
        def _keep(meta):
            a = (meta or {}).get("opp_arch", "?")
            if args.opp_arch and not any(k.lower() in a.lower() for k in args.opp_arch):
                return False
            if args.excl_arch and any(k.lower() in a.lower() for k in args.excl_arch):
                return False
            return True
        before = len(eps)
        seen = Counter(e[3].get("opp_arch", "?") for e in eps)
        eps = [e for e in eps if _keep(e[3])]
        print(f"  相手アーキタイプで絞り込み: {before} → {len(eps)} episodes"
              f"（include={args.opp_arch} exclude={args.excl_arch}）")
        kept = Counter(e[3].get("opp_arch", "?") for e in eps)
        for a, n in seen.most_common(8):
            print(f"     {a:<28}{n:>5} → {kept.get(a, 0):>5}")
        assert eps, "絞り込み後に0件。アーキタイプ名を確認すること"
    import re as _re
    def _epkey(x):
        m = _re.search(r"\d+", str(x[1]))
        return int(m.group()) if m else 0
    eps.sort(key=_epkey)
    if args.limit:
        eps = eps[:args.limit]
    print(f"{len(eps)} episodes ({dict(Counter(e[0] for e in eps))})")
    if not eps:
        return

    if args.workers <= 1:
        results = [process_episode(e) for e in eps]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            results = list(ex.map(process_episode, eps, chunksize=4))

    games = [r["game"] for r in results if r.get("game")]
    errs = [r["err"] for r in results if r.get("err")]
    ctx_n = Counter()
    ctx_opts = defaultdict(list)
    n_dec = 0
    for g in games:
        for d in g["decisions"]:
            ctx_n[d["ctx"]] += 1
            ctx_opts[d["ctx"]].append(len(d["opt_cid"]))
            n_dec += 1
    from cg.api import SelectContext
    name = {}
    for e in SelectContext:
        name[int(e)] = e.name
    print(f"\ngames={len(games)} decisions={n_dec} "
          f"avg_dec/game={n_dec/max(1,len(games)):.1f} errors={len(errs)}")
    print(f"{'ctx':>4} {'name':<18} {'決定数':>8} {'割合':>7} {'平均選択肢':>10}")
    for c, n in ctx_n.most_common():
        o = ctx_opts[c]
        print(f"{c:>4} {name.get(c,'?'):<18} {n:>8} {n/n_dec:>6.1%} "
              f"{sum(o)/len(o):>10.1f}")
    w = Counter(g["result"] for g in games)
    print(f"教師の成績: {dict(w)}")
    for e in errs[:3]:
        print("  ERR", e)

    if args.census or not args.out:
        return
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    # feat_keys はワーカーが確定させた値を使う
    # （-w 2以上だと親で _ensure_worker() が走らず None のまま保存されていた＝穴E。
    #  特徴順が記録されないと、学習時と推論時のズレを検出できない）
    fk = next((r.get("keys") for r in results if r.get("keys")), None)
    if fk is None:
        raise RuntimeError("feat_keysが取得できない（特徴順の記録が必須）")
    n_cls = max((r.get("n_cls") or 0) for r in results)
    print(f"feat_keys: {len(fk)}個 / 行動クラス: {n_cls}個 記録")
    with open(args.out, "wb") as f:
        classes = next((r.get("classes") for r in results if r.get("classes")), [])
        pickle.dump({"games": games, "feat_keys": fk, "n_cls": n_cls,
                     "classes": classes,
                     "teacher": {"team": args.team, "sig": args.sig},
                     "dates": dates}, f, protocol=4)
    mb = os.path.getsize(args.out) / 1e6
    print(f"saved -> {args.out} ({mb:.1f} MB)")


if __name__ == "__main__":
    main()
