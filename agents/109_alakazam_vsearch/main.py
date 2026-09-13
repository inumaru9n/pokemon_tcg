"""109_alakazam_feat4 — 全選択NN方策（LSTM）の模倣学習版。tools/gen_featurizer.py が生成。

教師: （未記入）

設計（agents/094_yushin_nn と同一。EXP-094の到達形）:
- **自己回帰はエンジンが提供**: MAIN → サーチ先 → 捨て札 が別々の select() として順に来る。
  LSTM を決定列に沿って進めることで「前の選択に条件付けた次の選択」になる
- **共有バックボーン + LSTM(記憶) + pointer型スコアリング**: オプション埋め込みと LSTM 状態の
  内積でスコア（可変長の選択肢を自然に扱う）。ctxアダプタで文脈差を吸収
- **前の選択の埋め込み**と**前回決定以降の logs 要約**（相手が何をしたか）を入力に明示投入
- **ルール・GBT・lethal探索は一切使わない**（NN単独で全選択）

model_109.npz が無い場合はエンジン列挙順の先頭を返す（＝方策として無力だが落ちない）。
"""

from __future__ import annotations

import os
import random
import time
import sys

try:
    _AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:  # Kaggle評価環境は exec() ロードで __file__ 未定義
    _AGENT_DIR = ("/kaggle_simulations/agent"
                  if os.path.exists("/kaggle_simulations/agent") else os.getcwd())
if _AGENT_DIR not in sys.path:
    sys.path.insert(0, _AGENT_DIR)

from cg.api import to_observation_class  # noqa: E402
from cg.api import OptionType as _OptionType  # noqa: E402
from cg.api import search_begin as _search_begin  # noqa: E402
from cg.api import search_end as _search_end  # noqa: E402
from cg.api import search_step as _search_step  # noqa: E402


def read_deck_csv() -> list[int]:
    for path in (os.path.join(_AGENT_DIR, "deck.csv"), "deck.csv",
                 "/kaggle_simulations/agent/deck.csv"):
        if os.path.exists(path):
            with open(path) as f:
                return [int(line.split(",")[0]) for line in f
                        if line.strip() and line.split(",")[0].strip().isdigit()]
    raise FileNotFoundError("deck.csv not found")


my_deck = read_deck_csv()

# ---- 行動クラスのスコア補正（実測頻度での較正）----
# 模倣方策は自分の下手な手で教師が到達しない盤面に入り、そこで「常に選べる安全な選択」
# （多くは end）を過剰に選ぶ（covariate shift）。実プレイの行動頻度を教師の頻度に
# 合わせる1クラス1スカラーの補正で矯正する。EXP-052の「実測頻度での較正」と同型。
# ファイルが無ければ補正なし（= 素のNN）。
CLASS_BIAS: dict[str, float] = {}
try:
    import json as _json
    _bp = os.path.join(_AGENT_DIR, "class_bias.json")
    if os.path.exists(_bp):
        with open(_bp) as _f:
            CLASS_BIAS = {str(k): float(v) for k, v in _json.load(_f).items()}
except Exception as _e:  # noqa: BLE001
    print(f"[109_alakazam_feat4] class_bias.json を読めない: {_e}", file=sys.stderr)

# ---- モデルと特徴のロード（失敗しても落ちない） ----
_NET = None
_FT = None
if os.environ.get("NN_109", "on") != "off":
    try:
        import feat_109 as _ft
        from nn_109 import LSTMPolicy

        _mp = os.path.join(_AGENT_DIR, "model_109.npz")
        if os.path.exists(_mp):
            _NET = LSTMPolicy(_mp)
            _FT = _ft
            # 特徴次元とモデルの入力次元が一致するか起動時に検証する。
            # 不一致を握り潰すと「動くが毎決定フォールバックする弱いエージェント」に
            # 無言で劣化し、errors=0なので検証でも捕捉できない。
            # **board側も必ず検証する**（opt側だけ見ていると board 変更を見逃す）
            _d_card = _NET.w["card_emb.weight"].shape[1]
            _exp_opt = _NET.w["opt_enc.0.weight"].shape[1]
            if _exp_opt != _d_card + _ft.OPT_DIM:
                raise RuntimeError(
                    f"option特徴の次元不一致: model={_exp_opt} "
                    f"expect={_d_card}+{_ft.OPT_DIM}")
            if "n_board" in _NET.w:
                _nb = int(_NET.w["n_board"])
                _nl = int(_NET.w["n_logs"])
                _exp_enc = _NET.w["enc.0.weight"].shape[1]
                if _exp_enc != _nb + _nl + 16 + _d_card + 2:
                    raise RuntimeError(f"enc入力次元の不整合: {_exp_enc}")
                if _nl != _ft.LOGS_DIM:
                    raise RuntimeError(
                        f"logs次元の不一致: model={_nl} feat={_ft.LOGS_DIM}")
                globals()["_EXPECT_BOARD"] = _nb  # 初回決定時にFEAT_KEYS確定後に検証
            # **標準化統計があるのに推論側が適用しない組み合わせを弾く**。
            # モデルは標準化された特徴で学習されているので、掛け忘れると例外を
            # 出さずに全く別の方策になる（無言劣化の典型）。
            if "board_mu" in _NET.w and getattr(_NET, "b_mu", None) is None:
                raise RuntimeError(
                    "モデルに標準化統計があるのに nn_109.py が適用していない"
                    "（古い nn_*.py が同梱されている）")
            if "trained_ctx" in _NET.w:
                globals()["_TRAINED_CTX"] = {int(c) for c in _NET.w["trained_ctx"]}
            # **行動クラスの名前列を突合する**。クラス数が同じでも名前が変われば
            # sorted順が変わって分類ヘッドのIDが全体的にずれ、全MAIN決定が誤った
            # クラスバイアスを受ける。次元数・クラス数の一致チェックでは検出できない
            # （EXP-096で実際に踏み、直接対戦で18pt失った）。
            _mc = _NET.w.get("classes")
            if _mc is not None and len(_mc):
                _fc = [str(x) for x in getattr(_ft, "CLASSES", ())]
                if [str(x) for x in _mc] != _fc:
                    raise RuntimeError(
                        f"行動クラスがモデルと不一致: model={len(_mc)}個 "
                        f"feat={len(_fc)}個")
            elif getattr(_ft, "CLASSES", None) and "cls_head.weight" in _NET.w:
                print("[109_alakazam_feat4] 警告: モデルにクラス名が記録されていないため"
                      "分類の整合を検証できない（古い学習で作られたモデル）",
                      file=sys.stderr)
    except Exception as _e:  # noqa: BLE001
        # 握り潰すと「動くが毎決定フォールバックする弱いエージェント」に無言で劣化し、
        # errors=0 なので検証でも捕捉できない。必ず可視化する（_NN_ACTIVE で判定可能）。
        import traceback
        print(f"[109_alakazam_feat4] モデル無効化: {type(_e).__name__}: {_e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        _NET = None
        _FT = None

# スモークテストや外部から「NNが実際に効いているか」を確認するためのフラグ
_NN_ACTIVE = _NET is not None and _FT is not None

_prev = [-1.0, -1.0, -1.0]   # 前の選択 (card_id, opt_type, area)
_last_turn = -1              # 直前に見たターン（ゲーム跨ぎの検出用・保険）
_prev_ctx = -1               # 直前の決定のcontext（ゲーム境界マーカー用）
_BOARD_CHECKED = False       # board次元の検証を初回決定時に1度だけ行う


def _new_game():
    """LSTM状態と履歴をゲーム開始状態に戻す。"""
    global _prev, _last_turn
    if _NET is not None:
        _NET.reset()
    _game_sec[0] = 0.0
    _prev = [-1.0, -1.0, -1.0]
    _last_turn = -1


# ---- 価値誘導探索 ----
# 方策の上位候補を展開し、**自分のターンを打ち切った盤面**を value head で評価して
# 並べ替える。相手は動かさない。詳細は tools/search_probe.py の docstring。
_SEARCH_MARGIN = float(os.environ.get("SEARCH_MARGIN", "1e9"))
# 対象にする SelectContext（カンマ区切り。空なら全ての単一選択ctx）
_SEARCH_CTX = {int(x) for x in os.environ.get("SEARCH_CTX", "0").split(",") if x.strip()}
_SEARCH_K = int(os.environ.get("SEARCH_K", "12"))
_SEARCH_MS = float(os.environ.get("SEARCH_MS", "200"))
_SEARCH_GAME_S = float(os.environ.get("SEARCH_GAME_S", "450"))
_sstat = {"calls": 0, "ran": 0, "changed": 0, "err": 0, "sec": 0.0}
_game_sec = [0.0]


def _determinize(obs):
    """隠れ情報をカードIDのリストで埋める。

    **自分側は厳密**（デッキ構成から観測済みを引いた残り）。山とサイドの振り分けだけが
    推測。相手側はプレースホルダで良い（相手を動かさないため）。
    """
    st = obs.current
    me = st.players[st.yourIndex]
    op = st.players[1 - st.yourIndex]
    seen = {}
    for c in (me.hand or []):
        seen[c.id] = seen.get(c.id, 0) + 1
    for c in me.discard:
        seen[c.id] = seen.get(c.id, 0) + 1
    fld = ([me.active[0]] if me.active and me.active[0] else []) \
        + [p for p in me.bench if p is not None]
    for p in fld:
        seen[p.id] = seen.get(p.id, 0) + 1
        for g in ("energyCards", "tools", "preEvolution"):
            for c in (getattr(p, g, None) or []):
                seen[c.id] = seen.get(c.id, 0) + 1
    for c in st.stadium:
        if c.playerIndex == st.yourIndex:
            seen[c.id] = seen.get(c.id, 0) + 1
    unseen = []
    for cid, n in _FT.DECK_COUNTS.items():
        unseen += [cid] * max(0, n - seen.get(cid, 0))
    need = me.deckCount + len(me.prize)
    pad = sorted(_FT.DECK_COUNTS)[0]
    while len(unseen) < need:
        unseen.append(pad)
    random.shuffle(unseen)
    return (unseen[:me.deckCount], unseen[me.deckCount:need],
            [pad] * op.deckCount, [pad] * len(op.prize), [pad] * op.handCount,
            [pad] if (op.active and op.active[0] is None) else [])


def _leaf1(ch, last_own_hand, flags):
    """**1手だけ進めた盤面**から特徴を作る。

    非MAIN（サーチ先・ベンチ出し等）や MAIN の攻撃は、1手進めても**まだ自分の手番**
    なので観測がそのまま使える（学習データに303,365件ある「選択直後かつ自分の決定点」
    と同じ分布）。MAINのENDだけは即座に相手の手番になるので、呼び出し側で除外する。
    保険として、相手の手番になっていた場合は手札とフラグを復元する。
    """
    ob = ch.observation
    st = ob.current
    if st.yourIndex != 0:
        if st.players[0].hand is None and last_own_hand is not None:
            try:
                st.players[0].hand = last_own_hand
            except Exception:  # noqa: BLE001
                pass
        try:
            st.energyAttached, st.supporterPlayed, st.retreated = flags
        except Exception:  # noqa: BLE001
            pass
    b, _k = _FT.feat_vector(st, 0)
    ctx = int(ob.select.context) if (ob.select and ob.select.option) else 0
    return b, _FT.logs_vector(ob, 0), ctx


def _rollout_leaf(ch, prev0):
    """自分のターンが終わるまで方策で進め、葉の (board, logs, prev, ctx) を返す。

    **葉は相手視点の観測になるので、自分の手札が None になる**（相手からは見えない）。
    そのままだと hand_* 22次元と left_* 22次元が全部ゼロになり、
    「手札を1枚も持っていない自分」という実在しない盤面をVに渡すことになる。
    実測でこれが原因で探索が −20〜−49pt の壊滅的失敗をした。
    自分の手番だった最後の観測から手札を移植して補う（実測で76%は1〜2枚ずれるが、
    全欠損よりは遥かに小さい誤差）。
    """
    pv = list(prev0)
    ctx = 0
    last_own = None
    for _ in range(30):
        ob = ch.observation
        s = ob.select
        if s is None or not s.option or ob.current.yourIndex != 0:
            break
        last_own = ob
        b, lg, cx, cids, vecs, cls = _feats(ob, 0)
        sc = _NET.scores(b, lg, pv, cx, cids, vecs, cls)
        k = max(1, min(s.maxCount or s.minCount or 1, len(s.option)))
        idx = sorted(range(len(sc)), key=lambda i: -sc[i])[:k]
        pv = [float(cids[idx[0]]), float(s.option[idx[0]].type or -1),
              float(s.option[idx[0]].area or -1)]
        ch = _search_step(ch.searchId, [int(x) for x in idx])
    ob = ch.observation
    st = ob.current
    if st.players[0].hand is None and last_own is not None:
        try:
            st.players[0].hand = last_own.current.players[0].hand
        except Exception:  # noqa: BLE001
            pass
    b, _k = _FT.feat_vector(st, 0)
    return b, _FT.logs_vector(ob, 0), pv, ctx


def _feats(o, me):
    """1決定ぶんの (board, logs, ctx, 選択肢cid, 選択肢vec, クラス) を作る。"""
    st = o.current
    b, _ = _FT.feat_vector(st, me)
    lg = _FT.logs_vector(o, me)
    ctx = int(o.select.context)
    hc, pl = _FT.pool_counts(st, me)
    cids, vecs = [], []
    for opt in o.select.option:
        c, v = _FT.option_vector(o, opt, me, hc, pl)
        cids.append(c)
        vecs.append(v)
    cls = ([_FT.CLASSES.index(_FT.option_class(st, o.select, i, me))
            for i in range(len(o.select.option))] if ctx == 0 else None)
    return b, lg, ctx, cids, vecs, cls


def _value_search(obs, scores, state_after):
    """探索して選択肢indexを返す。使えない/変更不要なら None。"""
    if _SEARCH_MARGIN >= 1e8 or _NET is None:
        return None
    sel = obs.select
    if sel is None or len(sel.option) < 2:
        return None
    if sel.maxCount and sel.maxCount > 1:
        return None                      # 複数選択（DISCARD等）は展開できない
    if _SEARCH_CTX and int(sel.context) not in _SEARCH_CTX:
        return None
    _sstat["calls"] += 1
    if _game_sec[0] > _SEARCH_GAME_S:
        return None
    t0 = time.perf_counter()
    deadline = t0 + _SEARCH_MS / 1000.0
    order = sorted(range(len(scores)), key=lambda i: -scores[i])[:_SEARCH_K]
    base = scores[order[0]]
    vals = {}
    try:
        st0 = obs.current
        me0 = st0.players[st0.yourIndex]
        hand0 = list(me0.hand) if me0.hand is not None else None
        flags0 = (st0.energyAttached, st0.supporterPlayed, st0.retreated)
        is_main = int(sel.context) == 0
        root = _search_begin(obs, *_determinize(obs))
        for cand in order:
            if time.perf_counter() > deadline:
                break
            # **MAIN の END は1手で相手の手番になり比較できない**ので飛ばす
            if is_main and sel.option[cand].type == _OptionType.END:
                continue
            _NET.set_state(state_after)      # 各候補を同じ状態から出発させる
            ch = _search_step(root.searchId, [int(cand)])
            b, lg, cx = _leaf1(ch, hand0, flags0)
            vals[int(cand)] = _NET.value(b, lg, _prev, cx)
    except Exception:  # noqa: BLE001
        _sstat["err"] += 1
        vals = {}
    finally:
        try:
            _search_end()
        except Exception:  # noqa: BLE001
            pass
        _NET.set_state(state_after)          # 本番の状態を必ず戻す
        dt = time.perf_counter() - t0
        _sstat["sec"] += dt
        _game_sec[0] += dt
    if len(vals) < 2:
        return None
    _sstat["ran"] += 1
    top = order[0]
    if top not in vals:
        return None                      # 方策の1位が評価できていない（END等）なら触らない
    best = max(vals, key=vals.get)
    if best == top or vals[best] - vals.get(top, -9.9) < _SEARCH_MARGIN:
        return None                      # 差が小さければ方策の1位を尊重する
    _sstat["changed"] += 1
    return best


def agent(obs_dict: dict) -> list[int]:
    global _prev, _last_turn
    obs = to_observation_class(obs_dict)

    if obs.select is None:      # 本番環境のデッキ提出時（ローカルarenaでは発生しない）
        _new_game()
        return my_deck

    # ゲーム跨ぎの検出: ワーカープロセスは複数ゲームで使い回されるが、ローカルarenaでは
    # select is None が来ないため自前で境界を見つける必要がある（検出漏れ＝前ゲームの
    # LSTM状態が漏れる）。**ターン後退だけでは不十分**で、前ゲームの終了ターンが
    # 新ゲームの開始ターン以下だと検出できない（実測200戦中1戦）。
    # 確実な境界マーカー: ゲームは必ず SETUP_ACTIVE_POKEMON(1) を1回だけ通り、
    # IS_FIRST(41) が来る場合は必ずその直前（実測30/30試合）。
    _ctx0 = int(obs.select.context)
    global _prev_ctx
    if _ctx0 == 41 or (_ctx0 == 1 and _prev_ctx != 41):
        _new_game()
    _prev_ctx = _ctx0
    _t = obs.current.turn if obs.current is not None else -1
    if _t >= 0:                 # 保険: マーカーを取り逃してもターン後退で拾う
        if _t < _last_turn:
            _new_game()
        _last_turn = _t

    n = len(obs.select.option)
    # **選ぶ枚数は maxCount を既定にする**。教師の実選択枚数との一致は
    # maxCount=100.0% に対し max(1,minCount)=96.3%（EXP-094で実測5,060決定）。
    k = min(obs.select.maxCount if obs.select.maxCount else obs.select.minCount, n)
    k = max(k, obs.select.minCount)
    if _NET is None or _FT is None or n == 0:
        return list(range(min(max(1, k), n))) if n else []

    # 選択肢が1個以下＝強制手。**学習系列にはこれらが含まれない**（抽出時に除外）ため、
    # 推論でもLSTMを進めない。進めると学習に無いステップが入り train/serving skew になる。
    # ただし minCount==0 なら「やる/やらない」の実質2択なので学習に含まれる（棄権対応）。
    if n < 2 and obs.select.minCount != 0:
        return list(range(min(max(1, k), n)))

    # **学習に現れなかったcontextは埋め込みが乱数初期値のまま**なので、スコアは
    # 意味を持たず、LSTMを進めれば以降の状態まで汚れる。被害をこの1決定に閉じる。
    _tc = globals().get("_TRAINED_CTX")
    if _tc is not None and int(obs.select.context) not in _tc:
        print(f"[109_alakazam_feat4] 未学習context {int(obs.select.context)} → この決定のみ"
              f"フォールバック", file=sys.stderr)
        return list(range(min(max(1, k), n)))

    try:
        state = obs.current
        my = state.yourIndex
        board, _keys = _FT.feat_vector(state, my)
        # board次元の検証は FEAT_KEYS が確定する初回決定時に行う。ここを黙って通すと、
        # 特徴を変えたのに古い重みのまま「毎決定index0を返すエージェント」になり、
        # errors=0・_NN_ACTIVE=True のまま気付けない。
        global _BOARD_CHECKED
        if not _BOARD_CHECKED:
            _BOARD_CHECKED = True
            _eb = globals().get("_EXPECT_BOARD")
            _bad = None
            if _eb is not None and len(board) != _eb:
                _bad = f"board次元不一致 model={_eb} feat={len(board)}"
            # **キーの並びも突合する**。次元数が同じまま順序だけ変わる変更は
            # 長さチェックを素通りして無言で壊れる。
            _mk = _NET.w.get("feat_keys") if _NET is not None else None
            if _bad is None and _mk is not None and len(_mk):
                if [str(x) for x in _mk] != list(_keys):
                    _bad = "feat_keysの並びがモデルと不一致"
            if _bad:
                print(f"[109_alakazam_feat4] {_bad} → NN無効化", file=sys.stderr)
                globals()["_NET"] = None
                globals()["_NN_ACTIVE"] = False
                return list(range(min(max(1, k), n)))
        logs = _FT.logs_vector(obs, my)
        hand_counts, pool_left = _FT.pool_counts(state, my)
        cids, ovecs = [], []
        for i in range(n):
            cid, v = _FT.option_vector(obs, obs.select.option[i], my,
                                       hand_counts, pool_left)
            cids.append(cid)
            ovecs.append(v)
        ctx_i = int(obs.select.context)
        ocls = None
        bias = None
        if ctx_i == 0:   # MAINのみ階層ヘッドを使う（学習時と同一条件）
            ocls = []
            if CLASS_BIAS:
                bias = []
            for i in range(n):
                try:
                    _nm = _FT.option_class(state, obs.select, i, my)
                    ocls.append(_FT.CLASS_ID.get(_nm, -1))
                except Exception:  # noqa: BLE001
                    _nm = None
                    ocls.append(-1)
                if bias is not None:
                    bias.append(CLASS_BIAS.get(_nm, 0.0) if _nm else 0.0)
        # **棄権(NULL)オプションを末尾に足す**（minCount==0のときのみ。学習と同一条件）。
        n_real = n
        can_decline = obs.select.minCount == 0
        if can_decline:
            cids = cids + [_FT.NULL_CID]
            ovecs = ovecs + [_FT.null_option_vector()]
            if ocls is not None:
                ocls = ocls + [-1]
            if bias is not None:
                bias = bias + [0.0]
            n = n + 1
        sc = _NET.scores(board, logs, _prev, ctx_i, cids, ovecs, ocls)
        if bias is not None:
            sc = [s + b for s, b in zip(sc, bias)]
        # **探索は実オプションのみを対象にする**（棄権NULLは展開できない）
        _sw = _value_search(obs, sc[:n_real], _NET.get_state())
        if _sw is not None:
            sc = [(1e6 if i == _sw else s) for i, s in enumerate(sc)]
        order = sorted(range(n), key=lambda i: -sc[i])
        if can_decline and order[0] == n_real:
            _prev = [float(_FT.NULL_CID), -1.0, -1.0]
            return []                       # 棄権
        order = [i for i in order if i < n_real]
        kk = max(1, min(k, n_real))
        pick = sorted(order[:kk])
        _prev = [float(cids[pick[0]]), ovecs[pick[0]][0], ovecs[pick[0]][1]]
        return pick
    except Exception:  # noqa: BLE001  推論に失敗しても不正選択で負けない
        return list(range(max(1, k)))[:n]
