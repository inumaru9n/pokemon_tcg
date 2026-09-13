"""174_alakazam_clean — 全選択NN方策（LSTM）の模倣学習版。tools/gen_featurizer.py が生成。

教師: Yushin Ito + AlphaStarmie / 156952a871 / 07-17〜08-07（170と同一データ）

**EXP-122: この系統の設計を原理で揃えた最終形**（170から board 524→449）:
- 完全重複55列を削除（学習データ435,150決定で厳密一致を確認。各群1本残す）
- 定数21列を削除（sd=1e-3に張り付き、本番1.1%の試合で1,000シグマを噴いていた）
- 語彙を69枚に厳格化（識別=ARCH_DECK 2系統以下 ∧ 教師観測mu>=0.02）
- サイド系の非対称2つを追加 / 汎用札の脅威追跡3枚を追加（干渉性で選抜）

設計（agents/094_yushin_nn と同一。EXP-094の到達形）:
- **自己回帰はエンジンが提供**: MAIN → サーチ先 → 捨て札 が別々の select() として順に来る。
  LSTM を決定列に沿って進めることで「前の選択に条件付けた次の選択」になる
- **共有バックボーン + LSTM(記憶) + pointer型スコアリング**: オプション埋め込みと LSTM 状態の
  内積でスコア（可変長の選択肢を自然に扱う）。ctxアダプタで文脈差を吸収
- **前の選択の埋め込み**と**前回決定以降の logs 要約**（相手が何をしたか）を入力に明示投入
- **ルール・GBT・lethal探索は一切使わない**（NN単独で全選択）

model_182.npz が無い場合はエンジン列挙順の先頭を返す（＝方策として無力だが落ちない）。
"""

from __future__ import annotations

import os
import sys

try:
    _AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:  # Kaggle評価環境は exec() ロードで __file__ 未定義
    _AGENT_DIR = ("/kaggle_simulations/agent"
                  if os.path.exists("/kaggle_simulations/agent") else os.getcwd())
if _AGENT_DIR not in sys.path:
    sys.path.insert(0, _AGENT_DIR)

from cg.api import to_observation_class  # noqa: E402


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
    print(f"[174_alakazam_clean] class_bias.json を読めない: {_e}", file=sys.stderr)

# ---- モデルと特徴のロード（失敗しても落ちない） ----
_NET = None
_FT = None
if os.environ.get("NN_170", "on") != "off":
    try:
        import feat_182 as _ft
        from nn_182 import LSTMPolicy

        _mp = os.path.join(_AGENT_DIR, "model_182.npz")
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
                    "モデルに標準化統計があるのに nn_182.py が適用していない"
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
                print("[174_alakazam_clean] 警告: モデルにクラス名が記録されていないため"
                      "分類の整合を検証できない（古い学習で作られたモデル）",
                      file=sys.stderr)
    except Exception as _e:  # noqa: BLE001
        # 握り潰すと「動くが毎決定フォールバックする弱いエージェント」に無言で劣化し、
        # errors=0 なので検証でも捕捉できない。必ず可視化する（_NN_ACTIVE で判定可能）。
        import traceback
        print(f"[174_alakazam_clean] モデル無効化: {type(_e).__name__}: {_e}", file=sys.stderr)
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
    _prev = [-1.0, -1.0, -1.0]
    _last_turn = -1


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
        print(f"[174_alakazam_clean] 未学習context {int(obs.select.context)} → この決定のみ"
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
                print(f"[174_alakazam_clean] {_bad} → NN無効化", file=sys.stderr)
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
