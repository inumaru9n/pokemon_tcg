"""EXP-057 ゾーン特徴器（knowledge/study/feature_inventory_v1.md コアセットの実装）。

学習（リプレイ収集）と推論（main.py）で共用する単一コードパス（パリティ原則 EXP-055）。
入力 = cg.api.Observation（current必須）、出力 = 特徴dict（キー→数値。0は省略しない）。
履歴・状態保持なし: observationスナップショット1つから関数的に計算できるものだけ。
"""

from __future__ import annotations

import os

from cg.api import CardType, all_card_data

# ---- 攻撃テーブル（EN_Card_Data.csv由来の自動生成モジュール） ----
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

# ---- カードID定数（main.pyと同値） ----
Abra, Kadabra, Alakazam = 741, 742, 743
Dunsparce, Dudunsparce = 305, 66
Fezandipiti_ex, Shaymin = 140, 343
Boss_Orders = 1182
Basic_Psychic_Energy, Telepath_Psychic_Energy, Enriching_Energy = 5, 19, 13
Mist_Energy, Rock_Fighting_Energy = 11, 20
TR_Articuno = 414
SPECIES = (Abra, Kadabra, Alakazam, Dunsparce, Dudunsparce, Fezandipiti_ex, Shaymin)
SPECIES_NAME = {Abra: "abra", Kadabra: "kadabra", Alakazam: "zam",
                Dunsparce: "dun", Dudunsparce: "dudun", Fezandipiti_ex: "fez",
                Shaymin: "shay"}

# ---- 自デッキ（60枚固定。deck.csvから採用枚数を読む） ----


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


def featurize(state, my_index: int) -> dict[str, float]:
    """State（obs.current）→ 特徴dict。~120キー、全キー常に出力（0含む）。"""
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
    act = me.active[0] if me.active else None
    for cid in SPECIES:
        f[f"act_{SPECIES_NAME[cid]}"] = 0.0
    f["act_hp"] = 0.0
    f["act_e_basic"] = f["act_e_telepath"] = f["act_e_enrich"] = 0.0
    f["act_cant"] = 1.0 if (me.asleep or me.paralyzed) else 0.0
    f["act_fresh"] = 0.0
    if act is not None:
        k = SPECIES_NAME.get(act.id)
        if k:
            f[f"act_{k}"] = 1.0
        f["act_hp"] = act.hp
        f["act_e_basic"] = sum(1 for e in act.energyCards
                               if e.id == Basic_Psychic_Energy)
        f["act_e_telepath"] = sum(1 for e in act.energyCards
                                  if e.id == Telepath_Psychic_Energy)
        f["act_e_enrich"] = sum(1 for e in act.energyCards
                                if e.id == Enriching_Energy)
        f["act_fresh"] = 1.0 if act.appearThisTurn else 0.0
    for cid in SPECIES:
        f[f"n_{SPECIES_NAME[cid]}"] = sum(1 for p in my_bench if p.id == cid)

    def _psy(p):
        return sum(1 for e in p.energyCards
                   if e.id in (Basic_Psychic_Energy, Telepath_Psychic_Energy))

    field_all = ([act] if act is not None else []) + my_bench
    f["ready_zam"] = sum(1 for p in field_all
                         if p.id == Alakazam and _psy(p) >= 1)
    f["loaded_pre"] = sum(1 for p in field_all
                          if p.id in (Abra, Kadabra) and len(p.energyCards) >= 1)
    f["dudun3"] = sum(1 for p in field_all
                      if p.id == Dudunsparce and len(p.energyCards) >= 3)
    f["bench_free"] = me.benchMax - len(my_bench)
    f["bench_dmg"] = sum(p.maxHp - p.hp for p in my_bench)

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
        for c in p.energyCards:
            _see(c.id)
        for c in p.tools:
            _see(c.id)
        for c in p.preEvolution:
            _see(c.id)
    for c in state.stadium:
        if c.playerIndex == my_index:
            _see(c.id)
    for cid in DECK_IDS:
        f[f"pool_{cid}"] = max(0, DECK_COUNTS[cid] - seen.get(cid, 0))
    # 導出確率: 基本超エネが全てサイド落ちしている確率（超幾何）
    pool_total = me.deckCount + prz_me
    c_psy = f[f"pool_{Basic_Psychic_Energy}"]
    p_all = 0.0
    if 0 < c_psy <= prz_me and pool_total > 0:
        p_all = 1.0
        for i in range(int(c_psy)):
            p_all *= max(0.0, (prz_me - i)) / max(1, (pool_total - i))
    f["p_psy_all_prized"] = p_all if c_psy > 0 else 0.0

    # ---- 7. 相手の場（属性ベース） ----
    op_act = op.active[0] if op.active else None
    op_articuno = any(p.id == TR_Articuno for p in ([op_act] if op_act else []) + op_bench)
    f["opp_act_hp"] = op_act.hp if op_act else 0.0
    f["opp_act_prize"] = _prz_of_id(op_act.id) if op_act else 0.0
    f["opp_act_energy"] = len(op_act.energyCards) if op_act else 0.0
    mx, var = _maxdmg_next(op_act, op.handCount) if op_act else (0, 0)
    f["opp_act_maxdmg"] = mx
    f["opp_act_vardmg"] = var
    ph_immune = 0.0
    if op_act is not None:
        mist = sum(1 for e in op_act.energyCards
                   if e.id in (Mist_Energy, Rock_Fighting_Energy))
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
    f["opp_used_boss"] = sum(1 for c in op.discard if c.id == Boss_Orders)
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
    for c in state.stadium:
        if c.playerIndex == my_index:
            st_mine = 1.0
        else:
            st_op = 1.0
    f["stadium_mine"] = st_mine
    f["stadium_op"] = st_op
    f["f_eatt"] = 1.0 if state.energyAttached else 0.0
    f["f_sup"] = 1.0 if state.supporterPlayed else 0.0
    f["f_retreated"] = 1.0 if state.retreated else 0.0

    return f


# ---- 行動クラス分類（学習の目的変数。選択肢→クラス名） ----
Buddy_Buddy_Poffin, Poke_Pad, Enhanced_Hammer = 1086, 1152, 1081
Rare_Candy, Night_Stretcher, Sacred_Ash = 1079, 1097, 1129
Dawn, Hilda, Xerosic, Lanas_Aid, Nighttime_Mine = 1231, 1225, 1197, 1184, 1266
Tool_Scrapper, Battle_Cage = 1137, 1264

_PLAY_CLASS = {Buddy_Buddy_Poffin: "poffin", Poke_Pad: "pad",
               Enhanced_Hammer: "hammer", Rare_Candy: "candy",
               Night_Stretcher: "stretcher", Sacred_Ash: "ash", Dawn: "dawn",
               Hilda: "hilda", Boss_Orders: "boss", Xerosic: "xero",
               Lanas_Aid: "lana", Nighttime_Mine: "mine",
               Tool_Scrapper: "scrapper", Battle_Cage: "cage",
               Abra: "put_abra", Dunsparce: "put_dun",
               Fezandipiti_ex: "put_fez", Shaymin: "put_shay"}
_EVO_CLASS = {Kadabra: "evo_kad", Alakazam: "evo_zam", Dudunsparce: "evo_dud"}
# EXP-058 (I-113①): attachは貼り先の種×状態バケットまで細分化
# （option一致0.457 vs クラス一致0.550の約10pt=対象ミスの回収が狙い）
_ATTACH_ETYPE = {Basic_Psychic_Energy: "basic", Telepath_Psychic_Energy: "tel"}
_ATTACH_BUCKETS = ("zam", "pre", "dun", "oth")


def _attach_bucket(pid: int) -> str:
    if pid == Alakazam:
        return "zam"
    if pid in (Abra, Kadabra):
        return "pre"
    if pid in (Dunsparce, Dudunsparce):
        return "dun"
    return "oth"


_ATTACH_CLASSES = {f"attach_{e}_{b}" for e in _ATTACH_ETYPE.values()
                   for b in _ATTACH_BUCKETS} | {"attach_enr"}

CLASSES = sorted(set(_PLAY_CLASS.values()) | set(_EVO_CLASS.values())
                 | _ATTACH_CLASSES
                 | {"ab_dud", "ab_fez", "ab_other", "retreat", "atk", "end",
                    "other"})
CLASS_ID = {c: i for i, c in enumerate(CLASSES)}


def option_class(state, select, opt_i: int, my_index: int) -> str:
    """MAIN選択肢→行動クラス名。単一コードパス（収集・推論共用）。"""
    from cg.api import AreaType, OptionType
    o = select.option[opt_i]
    me = state.players[my_index]
    if o.type == OptionType.ATTACK:
        return "atk"
    if o.type == OptionType.END:
        return "end"
    if o.type == OptionType.RETREAT:
        return "retreat"
    if o.type == OptionType.ABILITY:
        try:
            src = me.active if o.area == AreaType.ACTIVE else me.bench
            cid = src[o.index].id
        except (IndexError, TypeError, AttributeError):
            return "ab_other"
        return {Dudunsparce: "ab_dud", Fezandipiti_ex: "ab_fez"}.get(cid, "ab_other")
    try:
        cid = me.hand[o.index].id
    except (IndexError, TypeError, AttributeError):
        return "other"
    if o.type == OptionType.PLAY:
        return _PLAY_CLASS.get(cid, "other")
    if o.type == OptionType.EVOLVE:
        return _EVO_CLASS.get(cid, "other")
    if o.type == OptionType.ATTACH:
        if cid == Enriching_Energy:
            return "attach_enr"
        et = _ATTACH_ETYPE.get(cid)
        if et is None:
            return "other"
        try:
            src = me.active if o.inPlayArea == AreaType.ACTIVE else me.bench
            tid = src[o.inPlayIndex].id
        except (IndexError, TypeError, AttributeError):
            return f"attach_{et}_oth"
        return f"attach_{et}_{_attach_bucket(tid)}"
    return "other"


FEAT_KEYS = None  # 初回featurize時に確定


def feat_vector(state, my_index: int):
    """特徴dictを固定順のリストにして返す（学習・推論共用）。"""
    global FEAT_KEYS
    f = featurize(state, my_index)
    if FEAT_KEYS is None:
        FEAT_KEYS = sorted(f)
    return [f[k] for k in FEAT_KEYS], FEAT_KEYS


# ==== EXP-094: 全選択NN方策（LSTM）用の追加特徴 ====
# 抽出（selfplay/extract_yushin.py）と推論（main.py）で同一コードを使うため
# ここを唯一の定義とする。

N_LOGTYPE = 24          # cg.api.LogType の種類数
LOGS_DIM = N_LOGTYPE * 2  # 自分/相手で分ける
OPT_DIM = 24            # option_vector が返す数値特徴の次元
N_CTX = 49              # cg.api.SelectContext の種類数（0..48）。未学習ctxも埋め込みを確保


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
    選択が実在する。ctx3 SWITCHの約6割がこれ）。無視すると相手のポケモンを
    「自分の同indexのポケモン」として符号化してしまう。
    """
    from cg.api import AreaType, OptionType
    st = obs.current
    owner = opt.playerIndex if opt.playerIndex is not None else my_index
    ps = st.players[owner]
    card = None
    try:
        if opt.area is None:
            # **PLAY は area=None で index が手札index**（cardIdも空）。
            # ここを拾わないとMAIN選択肢の約42%がカード未解決(-1)になり、
            # カード同一性も hand_counts/pool_left も失われる（v5までの最大の穴 A-1）。
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
        at["dmg"] = (mx - hp) / 300.0                     # 負傷量（旧: damage属性は存在しない）
        at["hurt_ratio"] = (1.0 - hp / mx) if mx > 0 else 0.0
        at["n_energy"] = float(len(card.energyCards or []))
        at["n_tool"] = float(len(getattr(card, "tools", []) or []))  # 旧: toolCardsは存在しない
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
        # --- 以下 v5 で追加 ---
        at.get("is_opp", 0.0),        # 相手所有の選択肢か（ガスト先など）
        at.get("max_hp", 0.0),
        at.get("hurt_ratio", 0.0),    # 負傷割合（回復/回収/狙撃の判断に直結）
        # 同一ポケモンに付いた複数のツール/エネを区別する（旧: 全く同じ特徴になっていた）
        _num094(getattr(opt, "toolIndex", None), -1.0),
        _num094(getattr(opt, "energyIndex", None), -1.0),
        _num094(getattr(opt, "count", None), 0.0),
        # **貼り先/進化先**（inPlayArea/inPlayIndex）。これが無いと「同じエネを別の
        # ポケモンに貼る2択」「同じカダブラを別のアブラに乗せる3択」が完全に同一の
        # 特徴になり、MAIN決定の約15%が原理的に判別不能だった（v5までの穴 A-2）。
        _num094(getattr(opt, "inPlayArea", None), -1.0),
        _num094(getattr(opt, "inPlayIndex", None), -1.0),
        # 貼り先ポケモンの状態（誰に貼るかの判断材料）
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
