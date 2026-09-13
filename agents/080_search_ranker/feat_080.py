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


# ==== EXP-059: DISCARD二値分類器の特徴（カード×盤面 → 捨てるか） ====
# 盤面ゾーン特徴（featurizeを流用）+ そのカード自身の属性。自デッキ60枚固定=外挿なし。

DISCARD_FEAT_KEYS = None


def _card_feats(state, my_index: int, cid: int) -> dict[str, float]:
    """捨て候補カード自身の属性（盤面から算出できるもののみ）。"""
    me = state.players[my_index]
    hand = me.hand or []
    field_all = ([me.active[0]] if me.active and me.active[0] else []) \
        + [p for p in me.bench if p is not None]
    d: dict[str, float] = {}
    # カードID one-hot（自デッキの全ID）
    for c in DECK_IDS:
        d[f"c_is_{c}"] = 1.0 if c == cid else 0.0
    # そのカードの手札内の残枚数（=何枚持っているか。1枚しかないものは残しやすい）
    d["c_in_hand"] = sum(1 for c in hand if c.id == cid)
    # 盤面での有用性（今このカードが機能する状態か）
    cd = card_table.get(cid)
    d["c_is_energy"] = 1.0 if cid in (Basic_Psychic_Energy, Telepath_Psychic_Energy,
                                      Enriching_Energy) else 0.0
    d["c_is_pokemon"] = 1.0 if (cd is not None and cd.cardType == CardType.POKEMON) else 0.0
    d["c_is_supporter"] = 1.0 if (cd is not None and cd.cardType == CardType.SUPPORTER) else 0.0
    # プール残（山∪サイドに同カードが何枚残っているか。残ってれば捨てても引き直せる）
    seen = 0
    for c in hand:
        seen += 1 if c.id == cid else 0
    for c in me.discard:
        seen += 1 if c.id == cid else 0
    for p in field_all:
        seen += 1 if p.id == cid else 0
        for c in p.energyCards:
            seen += 1 if c.id == cid else 0
        for c in p.preEvolution:
            seen += 1 if c.id == cid else 0
    d["c_pool_left"] = max(0, DECK_COUNTS.get(cid, 0) - seen)
    # 進化ライン系: そのポケモンが今場で必要か（進化元/進化先が場にいるか）
    field_ids = {p.id for p in field_all}
    need = 0.0
    if cid == Kadabra and Abra in field_ids:
        need = 1.0
    elif cid == Alakazam and (Kadabra in field_ids or Abra in field_ids):
        need = 1.0
    elif cid == Rare_Candy and Abra in field_ids and Alakazam in {c.id for c in hand}:
        need = 1.0
    elif cid == Dudunsparce and Dunsparce in field_ids:
        need = 1.0
    d["c_evo_useful"] = need
    return d


def discard_feat_vector(state, my_index: int, cid: int):
    """DISCARD分類器用の (盤面特徴 + カード特徴) の固定順ベクトル。"""
    global DISCARD_FEAT_KEYS
    board = featurize(state, my_index)
    card = _card_feats(state, my_index, cid)
    merged = {**{f"b_{k}": v for k, v in board.items()}, **card}
    if DISCARD_FEAT_KEYS is None:
        DISCARD_FEAT_KEYS = sorted(merged)
    return [merged[k] for k in DISCARD_FEAT_KEYS], DISCARD_FEAT_KEYS


# ==== EXP-080: TO_HAND（サーチ先）二値ランカーの特徴（カード×盤面 → 取るか） ====
# DISCARD(059)と同じ「盤面ゾーン特徴 + 候補カード属性」の二値形式。
# 追加属性: サーチ元エリア（山/トラッシュ）と、取った場合の即効性（今ターン使えるか）

TOHAND_FEAT_KEYS = None


def to_hand_feat_vector(state, my_index: int, cid: int, from_discard: float = 0.0):
    """TO_HANDランカー用の (盤面特徴 + カード特徴 + サーチ文脈) の固定順ベクトル。"""
    global TOHAND_FEAT_KEYS
    board = featurize(state, my_index)
    card = _card_feats(state, my_index, cid)
    card["c_from_discard"] = from_discard
    # 即効性: エネ未添付ターンならエネ、進化可能ならその進化先は「今ターン使える」
    me = state.players[my_index]
    field_all = ([me.active[0]] if me.active and me.active[0] else []) \
        + [p for p in me.bench if p is not None]
    field_ids = {p.id for p in field_all}
    usable = 0.0
    if card["c_is_energy"] and not state.energyAttached:
        usable = 1.0
    elif cid == Kadabra and Abra in field_ids:
        usable = 1.0
    elif cid == Alakazam and Kadabra in field_ids:
        usable = 1.0
    card["c_usable_now"] = usable
    merged = {**{f"b_{k}": v for k, v in board.items()}, **card}
    if TOHAND_FEAT_KEYS is None:
        TOHAND_FEAT_KEYS = sorted(merged)
    return [merged[k] for k in TOHAND_FEAT_KEYS], TOHAND_FEAT_KEYS
