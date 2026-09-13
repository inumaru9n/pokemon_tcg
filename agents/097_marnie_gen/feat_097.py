"""自動生成された特徴器（tools/gen_featurizer.py）。手で編集しないこと。

デッキ: agents/097_marnie_gen/deck.csv
ポケモン 6種 / 進化ライン 3本 / エネ 1種 / スタジアム 1種
行動クラス 47個

汎用コア（後半）は tools/featgen_core.py と同一。デッキ依存はこのヘッダの定数のみ。
"""

from __future__ import annotations

import os

from cg.api import CardType, all_card_data

# ---- 攻撃テーブル（EN_Card_Data.csv由来の自動生成モジュール。全1056種を収録） ----
try:
    from attack_table_056 import ATTACKS_056
except ImportError:
    import importlib.util as _ilu

    try:
        _dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        _dir = ("/kaggle_simulations/agent"
                if os.path.exists("/kaggle_simulations/agent/main.py")
                else os.getcwd())
    _p = os.path.join(_dir, "attack_table_056.py")
    _sp = _ilu.spec_from_file_location("attack_table_056", _p)
    _m = _ilu.module_from_spec(_sp)
    _sp.loader.exec_module(_m)
    ATTACKS_056 = _m.ATTACKS_056

card_table = {c.cardId: c for c in all_card_data()}

# ---- 相手側の判定に使う固定ID（自デッキと無関係。094から継承） ----
BOSS_ORDERS = 1182
TR_ARTICUNO = 414
MIST_IDS = (11, 20)          # Mist Energy / Rock Fighting Energy（PH無効化）

# ---- 自デッキから導出した定数 ----
SPECIES = (104, 112, 646, 647, 648, 860)
SPECIES_NAME = {104: 'froslass', 112: 'munkidori', 646: 'marnie_impid', 647: 'marnie_morgr', 648: 'marnie_grimm', 860: 'snorunt'}
LINE_OF = {104: 'snorunt', 112: 'munkidori', 646: 'marnie_impid', 647: 'marnie_impid', 648: 'marnie_impid', 860: 'snorunt'}        # cardId -> 進化ライン名（最下段の種の名前）
LINE_NAMES = ('marnie_impid', 'munkidori', 'snorunt')
PRE_EVO = (646, 647, 860)        # 同デッキ内に進化先を持つ種
ENERGY_NAME = {7: 'basic_d_en'}      # 全エネ（基本+特殊）の cardId -> 短縮名
BASIC_ENERGY_NAME = {7: 'basic_d_en'}
STADIUM_NAME = {1259: 'spikemuth_'}
ATTACH_BUCKET = {104: 'snorunt_f', 112: 'munkidori_f', 646: 'marnie_impid_p', 647: 'marnie_impid_p', 648: 'marnie_impid_f', 860: 'snorunt_p'}   # エネ貼り先の種 -> クラス分類のバケット
CLS_PLAY = {104: 'play_froslass', 112: 'play_munkidori', 646: 'play_marnie_impid', 647: 'play_marnie_morgr', 648: 'play_marnie_grimm', 860: 'play_snorunt', 1079: 'use_rare_candy', 1080: 'use_unfair_sta', 1086: 'use_buddy_budd', 1097: 'use_night_stre', 1122: 'use_pokégear_3', 1137: 'use_tool_scrap', 1152: 'use_poké_pad', 1182: 'use_boss_order', 1219: 'use_team_rocke', 1227: 'use_lillie_det', 1231: 'use_dawn', 1259: 'use_spikemuth_'}
CLS_EVO = {104: 'evo_froslass', 647: 'evo_marnie_morgr', 648: 'evo_marnie_grimm'}
CLS_AB = {104: 'ab_froslass', 112: 'ab_munkidori', 646: 'ab_marnie_impid', 647: 'ab_marnie_morgr', 648: 'ab_marnie_grimm', 860: 'ab_snorunt', 1259: 'ab_spikemuth_'}        # ポケモンの特性 + 効果を持つスタジアム
CLS_ATK = {131: 'atk_froslass_0', 141: 'atk_munkidori_0', 934: 'atk_marnie_impid_0', 935: 'atk_marnie_impid_1', 936: 'atk_marnie_morgr_0', 937: 'atk_marnie_grimm_0', 1239: 'atk_snorunt_0'}      # attackId -> クラス名（技の打ち分け）
CLASSES = ('ab_froslass', 'ab_marnie_grimm', 'ab_marnie_impid', 'ab_marnie_morgr', 'ab_munkidori', 'ab_other', 'ab_snorunt', 'ab_spikemuth_', 'ab_stadium', 'atk_froslass_0', 'atk_marnie_grimm_0', 'atk_marnie_impid_0', 'atk_marnie_impid_1', 'atk_marnie_morgr_0', 'atk_munkidori_0', 'atk_other', 'atk_snorunt_0', 'attach_basic_d_en_marnie_impid_f', 'attach_basic_d_en_marnie_impid_p', 'attach_basic_d_en_munkidori_f', 'attach_basic_d_en_oth', 'attach_basic_d_en_snorunt_f', 'attach_basic_d_en_snorunt_p', 'end', 'evo_froslass', 'evo_marnie_grimm', 'evo_marnie_morgr', 'other', 'play_froslass', 'play_marnie_grimm', 'play_marnie_impid', 'play_marnie_morgr', 'play_munkidori', 'play_snorunt', 'retreat', 'use_boss_order', 'use_buddy_budd', 'use_dawn', 'use_lillie_det', 'use_night_stre', 'use_poké_pad', 'use_pokégear_3', 'use_rare_candy', 'use_spikemuth_', 'use_team_rocke', 'use_tool_scrap', 'use_unfair_sta')


def _load_deck_counts() -> dict[int, int]:
    try:
        _d = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        _d = os.getcwd()
    for path in (os.path.join(_d, "deck.csv"),
                 "deck.csv", "/kaggle_simulations/agent/deck.csv"):
        if os.path.exists(path):
            counts: dict[int, int] = {}
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line or not line.split(",")[0].isdigit():
                        continue
                    cid = int(line.split(",")[0])
                    counts[cid] = counts.get(cid, 0) + 1
            return counts
    raise FileNotFoundError("deck.csv not found")


DECK_COUNTS = _load_deck_counts()
DECK_IDS = sorted(DECK_COUNTS)  # 特徴キーの固定順


# ==== 汎用コア（tools/gen_featurizer.py が生成ファイルの後半へ丸ごと連結する） ====
# ここはデッキに依存しない。デッキ依存の情報は前半の生成ヘッダが定義する定数
#   SPECIES / SPECIES_NAME / LINE_OF / LINE_NAMES / PRE_EVO / ENERGY_NAME /
#   BASIC_ENERGY_NAME / STADIUM_NAME / ATTACH_BUCKET / CLS_PLAY / CLS_EVO /
#   CLS_AB / CLASSES / DECK_COUNTS / DECK_IDS
# だけを参照する。
#
# セクション1/2/3/5/7/8とオプション側特徴は agents/094_yushin_nn/feat_094.py の
# **忠実移植**（EXP-094 v12時点。A-1/A-2・playerIndex・toolIndex/energyIndex・
# NULLオプションの修正込み）。セクション4/6の導出項/9のスタジアムone-hotは、
# 094と092の手作り述語を「種族×エネ量」「進化ライン数」の機械展開に置き換えた版。


def _prz_of_id(cid: int) -> int:
    cd = card_table.get(cid)
    if cd is None:
        return 1
    return 3 if cd.megaEx else (2 if cd.ex else 1)


def _maxdmg_next(p, opph: int) -> tuple[int, int]:
    """(現エネ+1で撃てる最大固定打点, 可変攻撃フラグ)。相手PHは手札依存で特別計算。"""
    e_next = len(p.energyCards) + 1
    mx, var = 0, 0
    for (cn, dmg, mv) in ATTACKS_056.get(p.id, ()):
        if cn > e_next:
            continue
        if dmg is None:
            if mv == "Powerful Hand":
                mx = max(mx, 20 * (opph + 1))
            else:
                var = 1
        elif dmg > mx:
            mx = dmg
    return mx, var


def _e_count(p, ids) -> int:
    """ポケモンpに付いている、id集合idsのエネ枚数。"""
    return sum(1 for e in (p.energyCards or []) if e.id in ids)


def featurize(state, my_index: int) -> dict[str, float]:
    """State（obs.current）→ 特徴dict。全キー常に出力（0含む）。"""
    me = state.players[my_index]
    op = state.players[1 - my_index]
    f: dict[str, float] = {}

    # ---- 1. ゲーム進行 ----
    f["turn"] = state.turn
    f["first"] = 1.0 if state.firstPlayer == my_index else 0.0
    prz_me, prz_op = len(me.prize), len(op.prize)
    f["prz_me"] = prz_me
    f["prz_op"] = prz_op
    f["prz_diff"] = prz_op - prz_me  # 正=自分リード

    # ---- 2. 枚数カウント ----
    my_hand = me.hand or []
    f["n_hand_me"] = len(my_hand) if me.hand is not None else me.handCount
    f["n_deck_me"] = me.deckCount
    f["n_hand_op"] = op.handCount
    f["n_deck_op"] = op.deckCount
    my_bench = [p for p in me.bench if p is not None]
    op_bench = [p for p in op.bench if p is not None]
    f["n_bench_me"] = len(my_bench)
    f["n_bench_op"] = len(op_bench)
    f["n_dis_me"] = len(me.discard)
    f["n_dis_op"] = len(op.discard)

    # ---- 3. 自分の手札の中身（ID別） ----
    for cid in DECK_IDS:
        f[f"hand_{cid}"] = 0.0
    for c in my_hand:
        k = f"hand_{c.id}"
        if k in f:
            f[k] += 1

    # ---- 4. 自分の場 ----
    # 手作り版の述語（094: ready_zam/loaded_pre/dudun3、092: ready_grim/charged_pre/
    # munki_online/grimline_field/snow_field）は全て「種族またはラインの、場の数 ×
    # エネ枚数の閾値」だった。固定閾値を焼くのでなく **枚数そのもの（合計と最大）** を
    # 出す方が情報量が多く、閾値はMLP側が学べる。
    act = me.active[0] if me.active else None
    for cid in SPECIES:
        f[f"act_{SPECIES_NAME[cid]}"] = 0.0
    f["act_hp"] = 0.0
    f["act_hp_ratio"] = 0.0
    f["act_dmg"] = 0.0
    f["act_e"] = 0.0
    f["act_n_tool"] = 0.0
    for nm in ENERGY_NAME.values():
        f[f"act_e_{nm}"] = 0.0
    f["act_cant"] = 1.0 if (me.asleep or me.paralyzed) else 0.0
    f["act_fresh"] = 0.0
    if act is not None:
        k = SPECIES_NAME.get(act.id)
        if k:
            f[f"act_{k}"] = 1.0
        f["act_hp"] = act.hp
        f["act_hp_ratio"] = act.hp / act.maxHp if act.maxHp else 0.0
        f["act_dmg"] = act.maxHp - act.hp
        f["act_e"] = len(act.energyCards or [])
        f["act_n_tool"] = len(getattr(act, "tools", []) or [])
        f["act_fresh"] = 1.0 if act.appearThisTurn else 0.0
        for e in (act.energyCards or []):
            nm = ENERGY_NAME.get(e.id)
            if nm:
                f[f"act_e_{nm}"] += 1

    field_all = ([act] if act is not None else []) + my_bench
    for cid in SPECIES:
        nm = SPECIES_NAME[cid]
        cps = [p for p in field_all if p.id == cid]
        f[f"n_{nm}"] = len(cps)                    # 場（アクティブ込み）の数
        f[f"e_{nm}"] = sum(len(p.energyCards or []) for p in cps)
        f[f"emax_{nm}"] = max((len(p.energyCards or []) for p in cps), default=0)
        f[f"dmg_{nm}"] = sum(p.maxHp - p.hp for p in cps)
    for ln in LINE_NAMES:                           # 進化ライン単位の場の数
        f[f"line_{ln}"] = sum(1 for p in field_all if LINE_OF.get(p.id) == ln)
    pres = [p for p in field_all if p.id in PRE_EVO]
    f["n_pre"] = len(pres)
    f["e_pre"] = sum(len(p.energyCards or []) for p in pres)
    f["emax_pre"] = max((len(p.energyCards or []) for p in pres), default=0)
    f["bench_free"] = me.benchMax - len(my_bench)
    f["bench_dmg"] = sum(p.maxHp - p.hp for p in my_bench)
    f["field_dmg_me"] = f["bench_dmg"] + f["act_dmg"]

    # ---- 5. 自分のトラッシュ（ID別） ----
    for cid in DECK_IDS:
        f[f"dis_{cid}"] = 0.0
    for c in me.discard:
        k = f"dis_{c.id}"
        if k in f:
            f[k] += 1

    # ---- 6. 山∪サイドのプール（ID別残数） ----
    seen: dict[int, int] = {}

    def _see(cid):
        seen[cid] = seen.get(cid, 0) + 1

    for c in my_hand:
        _see(c.id)
    for c in me.discard:
        _see(c.id)
    for p in field_all:
        _see(p.id)
        for c in (p.energyCards or []):
            _see(c.id)
        for c in (getattr(p, "tools", []) or []):
            _see(c.id)
        for c in (getattr(p, "preEvolution", []) or []):
            _see(c.id)
    for c in state.stadium:
        if c.playerIndex == my_index:
            _see(c.id)
    for cid in DECK_IDS:
        f[f"pool_{cid}"] = max(0, DECK_COUNTS[cid] - seen.get(cid, 0))
    # 導出確率: 各基本エネが「残り全部サイド落ち」の確率（超幾何）。
    # 094の p_psy_all_prized の一般化（基本エネ種ごとに1つ）。
    pool_total = me.deckCount + prz_me
    for cid, nm in BASIC_ENERGY_NAME.items():
        c_left = f.get(f"pool_{cid}", 0.0)
        p_all = 0.0
        if 0 < c_left <= prz_me and pool_total > 0:
            p_all = 1.0
            for i in range(int(c_left)):
                p_all *= max(0.0, (prz_me - i)) / max(1, (pool_total - i))
        f[f"p_{nm}_all_prized"] = p_all

    # ---- 7. 相手の場（属性ベース。デッキ非依存） ----
    op_act = op.active[0] if op.active else None
    op_articuno = any(p.id == TR_ARTICUNO
                      for p in ([op_act] if op_act else []) + op_bench)
    f["opp_act_hp"] = op_act.hp if op_act else 0.0
    f["opp_act_dmg"] = (op_act.maxHp - op_act.hp) if op_act else 0.0
    f["opp_act_prize"] = _prz_of_id(op_act.id) if op_act else 0.0
    f["opp_act_energy"] = len(op_act.energyCards) if op_act else 0.0
    mx, var = _maxdmg_next(op_act, op.handCount) if op_act else (0, 0)
    f["opp_act_maxdmg"] = mx
    f["opp_act_vardmg"] = var
    ph_immune = 0.0
    if op_act is not None:
        mist = _e_count(op_act, MIST_IDS)
        cd = card_table.get(op_act.id)
        veil = (op_articuno and cd is not None and cd.basic
                and "Team Rocket" in cd.name)
        ph_immune = 1.0 if (mist > 0 or veil) else 0.0
    f["opp_act_ph_immune"] = ph_immune
    f["opp_bench_prz2"] = sum(1 for p in op_bench if _prz_of_id(p.id) >= 2)
    f["opp_bench_canatk"] = sum(
        1 for p in op_bench
        if any(cn <= len(p.energyCards) for (cn, _d, _m) in ATTACKS_056.get(p.id, ())))
    f["opp_field_maxdmg"] = max(
        [mx] + [_maxdmg_next(p, op.handCount)[0] for p in op_bench], default=0)
    f["opp_bench_maxhp"] = max([p.hp for p in op_bench], default=0)
    f["opp_bench_growing"] = sum(1 for p in op_bench
                                 if len(p.energyCards) == 0 and p.hp == p.maxHp)
    f["opp_cant"] = 1.0 if (op.asleep or op.paralyzed) else 0.0

    # ---- 8. 相手のトラッシュ（資源勘定） ----
    f["opp_used_boss"] = sum(1 for c in op.discard if c.id == BOSS_ORDERS)
    n_e = n_sup = n_poke = 0
    for c in op.discard:
        cd = card_table.get(c.id)
        if cd is None:
            continue
        if cd.cardType in (CardType.BASIC_ENERGY, CardType.SPECIAL_ENERGY):
            n_e += 1
        elif cd.cardType == CardType.SUPPORTER:
            n_sup += 1
        elif cd.cardType == CardType.POKEMON:
            n_poke += 1
    f["opp_e_disc"] = n_e
    f["opp_sup_used"] = n_sup
    f["opp_poke_lost"] = n_poke

    # ---- 9. スタジアム・ターンフラグ ----
    st_mine = st_op = 0.0
    for nm in STADIUM_NAME.values():
        f[f"stadium_{nm}"] = 0.0
    for c in state.stadium:
        if c.playerIndex == my_index:
            st_mine = 1.0
        else:
            st_op = 1.0
        nm = STADIUM_NAME.get(c.id)
        if nm:
            f[f"stadium_{nm}"] = 1.0
    f["stadium_mine"] = st_mine
    f["stadium_op"] = st_op
    f["f_eatt"] = 1.0 if state.energyAttached else 0.0
    f["f_sup"] = 1.0 if state.supporterPlayed else 0.0
    f["f_retreated"] = 1.0 if state.retreated else 0.0

    return f


CLASS_ID = {c: i for i, c in enumerate(CLASSES)}


def option_class(state, select, opt_i: int, my_index: int) -> str:
    """MAIN選択肢→行動クラス名。単一コードパス（収集・推論共用）。"""
    from cg.api import AreaType, OptionType
    o = select.option[opt_i]
    me = state.players[my_index]
    if o.type == OptionType.ATTACK:
        # **技ごとに分ける**（092で実装。どの技を撃つかはMAIN決定の中核なので、
        # 「atk」1クラスに潰すと打ち分けの情報がクラス頭に載らない）
        return CLS_ATK.get(o.attackId, "atk_other")
    if o.type == OptionType.END:
        return "end"
    if o.type == OptionType.RETREAT:
        return "retreat"
    if o.type == OptionType.ABILITY:
        try:
            # **STADIUMを必ず分岐に入れる**。スタジアムの効果発動もABILITY型で来る
            # （092のSpikemuth GymはMAIN決定の9.9%）。ACTIVE以外をbenchと決めつけると
            # 無関係なベンチのポケモンのIDを読み、例外も出ずに誤分類される。
            if o.area == AreaType.STADIUM:
                # **相手のスタジアムも来る**（Spikemuth Gym等は両者が効果を使える。
                # 094の教師データではMAIN決定の1.5%）。自デッキに無いカードは
                # CLS_ABに席が無いので、雑多な ab_other でなく専用クラスに寄せる。
                return CLS_AB.get(state.stadium[o.index].id, "ab_stadium")
            elif o.area == AreaType.ACTIVE:
                cid = me.active[o.index].id
            elif o.area == AreaType.BENCH:
                cid = me.bench[o.index].id
            else:
                return "ab_other"
        except (IndexError, TypeError, AttributeError):
            return "ab_other"
        return CLS_AB.get(cid, "ab_other")
    try:
        cid = me.hand[o.index].id
    except (IndexError, TypeError, AttributeError):
        return "other"
    if o.type == OptionType.PLAY:
        return CLS_PLAY.get(cid, "other")
    if o.type == OptionType.EVOLVE:
        return CLS_EVO.get(cid, "other")
    if o.type == OptionType.ATTACH:
        et = ENERGY_NAME.get(cid)
        if et is None:
            return "other"
        try:
            src = me.active if o.inPlayArea == AreaType.ACTIVE else me.bench
            tid = src[o.inPlayIndex].id
        except (IndexError, TypeError, AttributeError):
            return f"attach_{et}_oth"
        return f"attach_{et}_{ATTACH_BUCKET.get(tid, 'oth')}"
    return "other"


FEAT_KEYS = None  # 初回featurize時に確定


def feat_vector(state, my_index: int):
    """特徴dictを固定順のリストにして返す（学習・推論共用）。"""
    global FEAT_KEYS
    f = featurize(state, my_index)
    if FEAT_KEYS is None:
        FEAT_KEYS = sorted(f)
    return [f[k] for k in FEAT_KEYS], FEAT_KEYS


# ==== 全選択NN方策（LSTM）用の追加特徴（feat_094.py から忠実移植・デッキ非依存） ====

N_LOGTYPE = 24          # cg.api.LogType の種類数
LOGS_DIM = N_LOGTYPE * 2  # 自分/相手で分ける
OPT_DIM = 24            # option_vector が返す数値特徴の次元
N_CTX = 49              # cg.api.SelectContext の種類数（0..48）


def _num094(x, default=-1.0):
    """None/enum混在の安全な数値化（END/ATTACK等は area/index が None）。"""
    if x is None:
        return default
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def logs_vector(obs, my_index: int) -> list[float]:
    """前回決定以降の logs 要約: LogType別カウント（自分/相手）。"""
    v = [0.0] * LOGS_DIM
    for lg in (getattr(obs, "logs", None) or []):
        t = _num094(getattr(lg, "type", None), -1.0)
        ti = int(t)
        if not (0 <= ti < N_LOGTYPE):
            continue
        pi = getattr(lg, "playerIndex", my_index)
        off = 0 if pi == my_index else N_LOGTYPE
        v[off + ti] += 1.0
    return v


def resolve_option(obs, opt, my_index: int):
    """option → (card_id, 属性dict)。解決不能なら card_id=-1（裏向きサイド等）。

    **opt.playerIndex を尊重する**（ボスの指令のガスト先など、相手の場を対象にする
    選択が実在する。ctx3 SWITCHの約6割がこれ）。
    """
    from cg.api import AreaType, OptionType
    st = obs.current
    owner = opt.playerIndex if opt.playerIndex is not None else my_index
    ps = st.players[owner]
    card = None
    try:
        if opt.area is None:
            # **PLAY は area=None で index が手札index**（cardIdも空）。
            if opt.type == OptionType.PLAY and opt.index is not None:
                card = (ps.hand or [])[opt.index]
        elif opt.area == AreaType.DECK:
            card = (obs.select.deck or [])[opt.index]
        elif opt.area == AreaType.HAND:
            card = (ps.hand or [])[opt.index]
        elif opt.area == AreaType.DISCARD:
            card = ps.discard[opt.index]
        elif opt.area == AreaType.ACTIVE:
            card = ps.active[opt.index]
        elif opt.area == AreaType.BENCH:
            card = ps.bench[opt.index]
        elif opt.area == AreaType.STADIUM:
            card = st.stadium[opt.index]
        elif opt.area == AreaType.LOOKING:
            card = st.looking[opt.index]
    except (IndexError, TypeError, AttributeError):
        card = None
    at = {"is_opp": 1.0 if owner != my_index else 0.0}
    if card is None:
        return -1, at
    cid = getattr(card, "id", -1)
    if hasattr(card, "energyCards"):     # ポケモン
        cd = card_table.get(cid)
        mx = float(getattr(card, "maxHp", 0) or 0)
        hp = float(getattr(card, "hp", 0) or 0)   # hp=残HP、maxHp=最大HP
        at["is_pokemon"] = 1.0
        at["hp"] = hp / 300.0
        at["max_hp"] = mx / 300.0
        at["dmg"] = (mx - hp) / 300.0
        at["hurt_ratio"] = (1.0 - hp / mx) if mx > 0 else 0.0
        at["n_energy"] = float(len(card.energyCards or []))
        at["n_tool"] = float(len(getattr(card, "tools", []) or []))
        at["n_pre"] = float(len(getattr(card, "preEvolution", []) or []))
        at["is_ex"] = 1.0 if (cd is not None and getattr(cd, "ex", False)) else 0.0
    else:
        at["is_pokemon"] = 0.0
    return cid, at


def pool_counts(state, my_index: int):
    """(手札内の同カード枚数, 山∪サイド残枚数) の2 dict。"""
    me = state.players[my_index]
    hand = me.hand or []
    hand_counts: dict[int, int] = {}
    for c in hand:
        hand_counts[c.id] = hand_counts.get(c.id, 0) + 1
    seen = dict(hand_counts)
    for c in me.discard:
        seen[c.id] = seen.get(c.id, 0) + 1
    fld = ([me.active[0]] if me.active and me.active[0] else []) \
        + [b for b in me.bench if b is not None]
    for p in fld:
        seen[p.id] = seen.get(p.id, 0) + 1
        for c in (p.energyCards or []):
            seen[c.id] = seen.get(c.id, 0) + 1
        for c in (getattr(p, "preEvolution", []) or []):
            seen[c.id] = seen.get(c.id, 0) + 1
    left = {cid: max(0, n - seen.get(cid, 0)) for cid, n in DECK_COUNTS.items()}
    return hand_counts, left


def option_vector(obs, opt, my_index: int, hand_counts, pool_left):
    """オプション1つ → (card_id, 数値特徴list[OPT_DIM])。"""
    cid, at = resolve_option(obs, opt, my_index)
    _idx = opt.index if opt.index is not None else -1
    return cid, [
        _num094(opt.type),
        _num094(opt.area),
        float(min(_idx, 20)),
        at.get("is_pokemon", 0.0),
        at.get("hp", 0.0),
        at.get("dmg", 0.0),
        at.get("n_energy", 0.0),
        at.get("n_tool", 0.0),
        at.get("n_pre", 0.0),
        at.get("is_ex", 0.0),
        float(hand_counts.get(cid, 0)),
        float(pool_left.get(cid, 0)),
        _num094(getattr(opt, "number", None), 0.0),
        _num094(getattr(opt, "attackId", None), 0.0),
        at.get("is_opp", 0.0),        # 相手所有の選択肢か（ガスト先など）
        at.get("max_hp", 0.0),
        at.get("hurt_ratio", 0.0),
        # 同一ポケモンに付いた複数のツール/エネを区別する
        _num094(getattr(opt, "toolIndex", None), -1.0),
        _num094(getattr(opt, "energyIndex", None), -1.0),
        _num094(getattr(opt, "count", None), 0.0),
        # **貼り先/進化先**（inPlayArea/inPlayIndex）。無いとMAIN決定の約15%が
        # 原理的に判別不能になる（094 v5までの穴 A-2）。
        _num094(getattr(opt, "inPlayArea", None), -1.0),
        _num094(getattr(opt, "inPlayIndex", None), -1.0),
        *_inplay_target_feats(obs, opt, my_index),
    ]


def _inplay_target_feats(obs, opt, my_index: int):
    """貼り先/進化先ポケモンの (残HP割合, エネ数)。取得できなければ0。"""
    from cg.api import AreaType
    ipa, ipi = getattr(opt, "inPlayArea", None), getattr(opt, "inPlayIndex", None)
    if ipa is None or ipi is None:
        return [0.0, 0.0]
    try:
        ps = obs.current.players[my_index]
        p = ps.active[ipi] if ipa == AreaType.ACTIVE else ps.bench[ipi]
        if p is None:
            return [0.0, 0.0]
        mx = float(getattr(p, "maxHp", 0) or 0)
        hp = float(getattr(p, "hp", 0) or 0)
        return [hp / mx if mx > 0 else 0.0, float(len(p.energyCards or []))]
    except (IndexError, TypeError, AttributeError):
        return [0.0, 0.0]


# 棄権(NULL)オプションの特徴。minCount==0 の決定で選択肢列の末尾に足し、
# pointer networkのstop actionとして機能させる（EXP-094 v7）。
NULL_CID = -3


def null_option_vector():
    """棄権オプションの数値特徴。実オプションと区別できる値にする。"""
    v = [0.0] * OPT_DIM
    v[0] = -1.0   # type: 実optionには無い値
    v[1] = -1.0   # area
    v[2] = -1.0   # index
    return v
