"""EXP-091 Starmie(491b8bfb26)版ゾーン特徴器 + MAIN行動クラス分類 (I-136)。

EXP-057定式化のStarmie適用: 目的変数=「今この盤面で教師(Yushin Ito/491b)が
取る行動クラス」。feat_089の構成法（手札ID別/山∪サイドプール/トラッシュID別/
資源勘定/両者盤面属性/ターンフラグ）を491b8bfb26の60枚に対応させて再設計。
学習（リプレイ収集）と推論（main.py）で共用する単一コードパス（パリティ原則）。
"""

from __future__ import annotations

import os

from cg.api import CardType, EnergyType, all_card_data

# ---- 攻撃テーブル（EN_Card_Data.csv由来の自動生成モジュール。056のコピー） ----
try:
    from attack_table_093 import ATTACKS_056 as ATTACKS
except ImportError:
    import importlib.util as _ilu

    try:
        _dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        _dir = ("/kaggle_simulations/agent"
                if os.path.exists("/kaggle_simulations/agent/main.py")
                else os.getcwd())
    _p = os.path.join(_dir, "attack_table_093.py")
    _sp = _ilu.spec_from_file_location("attack_table_093", _p)
    _m = _ilu.module_from_spec(_sp)
    _sp.loader.exec_module(_m)
    ATTACKS = _m.ATTACKS_056

card_table = {c.cardId: c for c in all_card_data()}

# ---- カードID定数（main.py/EXP-035系と同値） ----
Basic_Water = 3
Ignition = 17
Cinderace = 666
Staryu = 1030
Mega_Starmie = 1031
Poffin = 1086
Night_Stretcher = 1097
Hammer = 1120
Ultra_Ball = 1121
Pokegear = 1122
Mega_Signal = 1145
Cape = 1159
Boss_Orders = 1182
Salvatore = 1189
Harlequin = 1223
Hilda = 1225
Lillie = 1227
Wally = 1229

Spikemuth_Gym = 1259
Munkidori_ids = (112, 139)
Marnie_ids = (646, 647, 648)
Crustle_Wall = 345
Mega_Kangaskhan = 756
Dunsparce_line = (65, 66, 305, 306, 996, 997)
Evolution_seeds = (109, 741, 742, 119, 120, 379, 380, 333, 677, 974,
                   344, 532, 103, 860, 646, 647)

SPECIES = (Cinderace, Staryu, Mega_Starmie)
SPECIES_NAME = {Cinderace: "cind", Staryu: "star", Mega_Starmie: "mega"}

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
DECK_IDS = sorted(DECK_COUNTS)  # 特徴キーの固定順（18種）


def _prz_of_id(cid: int) -> int:
    cd = card_table.get(cid)
    if cd is None:
        return 1
    return 3 if cd.megaEx else (2 if cd.ex else 1)


def _has_ability(cid: int) -> bool:
    cd = card_table.get(cid)
    return cd is not None and len(cd.skills) > 0


def _maxdmg_next(p, opph: int) -> tuple[int, int]:
    """(現エネ+1で撃てる最大固定打点, 可変攻撃フラグ)。相手PHは手札依存で特別計算。"""
    e_next = len(p.energyCards) + 1
    mx, var = 0, 0
    for (cn, dmg, mv) in ATTACKS.get(p.id, ()):
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
    """State（obs.current）→ 特徴dict。~115キー、全キー常に出力（0含む）。"""
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
    f["act_dmg"] = 0.0
    f["act_e"] = 0.0
    f["act_w"] = 0.0
    f["act_tool"] = 0.0
    f["act_cant"] = 1.0 if (me.asleep or me.paralyzed) else 0.0
    f["act_fresh"] = 0.0
    if act is not None:
        k = SPECIES_NAME.get(act.id)
        if k:
            f[f"act_{k}"] = 1.0
        f["act_hp"] = act.hp
        f["act_dmg"] = act.maxHp - act.hp
        f["act_e"] = len(act.energyCards)
        f["act_w"] = sum(1 for e in act.energies if e == EnergyType.WATER)
        f["act_tool"] = len(act.tools)
        f["act_fresh"] = 1.0 if act.appearThisTurn else 0.0
    for cid in SPECIES:
        f[f"n_{SPECIES_NAME[cid]}"] = sum(1 for p in my_bench if p.id == cid)

    field_all = ([act] if act is not None else []) + my_bench
    # 資源勘定・結合述語（Starmie版）
    megas = [p for p in field_all if p.id == Mega_Starmie]
    f["n_mega_field"] = len(megas)
    f["line_on_field"] = len(megas) + sum(1 for p in field_all if p.id == Staryu)
    f["mega_dmg_max"] = max([p.maxHp - p.hp for p in megas], default=0)  # Wally弾
    f["mega_w_total"] = sum(
        sum(1 for e in p.energies if e == EnergyType.WATER) for p in megas)
    f["mega_charged"] = sum(
        1 for p in megas if any(e == EnergyType.WATER for e in p.energies))
    f["cape_on_line"] = sum(1 for p in field_all
                            if p.id in (Staryu, Mega_Starmie) and p.tools)
    f["staryu_naked"] = sum(1 for p in field_all
                            if p.id == Staryu and not p.energies and not p.tools)
    hand_ids = [c.id for c in my_hand]
    dis_ids = [c.id for c in me.discard]
    f["deck_megas"] = max(0, 3 - len(megas)
                          - hand_ids.count(Mega_Starmie)
                          - dis_ids.count(Mega_Starmie))
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

    # ---- 7. 相手の場(属性ベース + 対面述語) ----
    op_act = op.active[0] if op.active else None
    f["opp_act_hp"] = op_act.hp if op_act else 0.0
    f["opp_act_dmg"] = (op_act.maxHp - op_act.hp) if op_act else 0.0
    f["opp_act_prize"] = _prz_of_id(op_act.id) if op_act else 0.0
    f["opp_act_energy"] = len(op_act.energyCards) if op_act else 0.0
    mx, var = _maxdmg_next(op_act, op.handCount) if op_act else (0, 0)
    f["opp_act_maxdmg"] = mx
    f["opp_act_vardmg"] = var
    weak = res = 0.0
    abil = 0.0
    if op_act is not None:
        cd = card_table.get(op_act.id)
        if cd is not None:
            weak = 1.0 if cd.weakness == EnergyType.WATER else 0.0
            res = 1.0 if cd.resistance == EnergyType.WATER else 0.0
        abil = 1.0 if _has_ability(op_act.id) else 0.0
    f["opp_act_weak_water"] = weak
    f["opp_act_res_water"] = res
    f["opp_act_ability"] = abil
    # Nebula 210 / Jetting 120 の当てはまり（R15の意思決定変数）
    f["opp_act_hp_jetting"] = 1.0 if (op_act is not None
                                      and op_act.hp <= 120) else 0.0
    f["opp_act_hp_nebula"] = 1.0 if (op_act is not None
                                     and 120 < op_act.hp <= 210) else 0.0
    f["opp_wall_active"] = 1.0 if (op_act is not None
                                   and op_act.id == Crustle_Wall) else 0.0
    op_field = ([op_act] if op_act is not None else []) + op_bench
    f["opp_big_tank"] = 1.0 if any(
        card_table.get(p.id) is not None and card_table[p.id].hp >= 280
        for p in op_field) else 0.0
    f["opp_marnie"] = 1.0 if (
        any(p.id in Marnie_ids for p in op_field)
        or any(c.id in Marnie_ids for c in op.discard)) else 0.0
    f["opp_munki_field"] = sum(1 for p in op_field if p.id in Munkidori_ids)
    f["opp_megakang_bench"] = sum(1 for p in op_bench if p.id == Mega_Kangaskhan)
    f["opp_n_ability"] = sum(1 for p in op_field if _has_ability(p.id))
    f["opp_bench_prz2"] = sum(1 for p in op_bench if _prz_of_id(p.id) >= 2)
    f["opp_bench_le50"] = sum(1 for p in op_bench if p.hp <= 50)  # snipe即KO圏
    f["opp_bench_seeds"] = sum(1 for p in op_bench if p.id in Evolution_seeds)
    f["opp_bench_dun"] = sum(1 for p in op_bench if p.id in Dunsparce_line)
    f["opp_bench_canatk"] = sum(
        1 for p in op_bench
        if any(cn <= len(p.energyCards) for (cn, _d, _m) in ATTACKS.get(p.id, ())))
    f["opp_field_maxdmg"] = max(
        [mx] + [_maxdmg_next(p, op.handCount)[0] for p in op_bench], default=0)
    f["opp_bench_maxhp"] = max([p.hp for p in op_bench], default=0)
    f["opp_bench_dmg"] = sum(p.maxHp - p.hp for p in op_bench)
    f["opp_bench_growing"] = sum(1 for p in op_bench
                                 if len(p.energyCards) == 0 and p.hp == p.maxHp)
    f["opp_cant"] = 1.0 if (op.asleep or op.paralyzed) else 0.0
    f["opp_tools"] = sum(len(p.tools) for p in op_field)
    f["opp_act_e_strip"] = 1.0 if (op_act is not None
                                   and op_act.energyCards) else 0.0  # Hammer弾

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
    st_mine = st_op = st_spike = 0.0
    for c in state.stadium:
        if c.playerIndex == my_index:
            st_mine = 1.0
        else:
            st_op = 1.0
        if c.id == Spikemuth_Gym:
            st_spike = 1.0
    f["stadium_mine"] = st_mine
    f["stadium_op"] = st_op
    f["stadium_spike"] = st_spike
    f["f_eatt"] = 1.0 if state.energyAttached else 0.0
    f["f_sup"] = 1.0 if state.supporterPlayed else 0.0
    f["f_retreated"] = 1.0 if state.retreated else 0.0

    return f


# ---- 行動クラス分類（学習の目的変数。選択肢→クラス名) ----
# クラス設計は教師データのctx0選択の頻度集計（07-03 Yushin Ito, 11,597決定）で確定:
#   atk_jetting 14.7% / attach_w_mega 9.6% / ab_gym(Spikemuth効果) 6.3%
#   / evo_mega 5.7% / gear 5.5% / attach_ign_mega 5.2% / atk_nebula 5.2%
#   / end 4.9% / signal 4.8% / poffin 4.4% / wally 4.4% / hammer 4.3%
#   / lillie 4.2% / attach_w_staryu 2.9% / hilda 2.5% / retreat 2.5%
#   / salvatore 2.3% / put_staryu 2.1% / harlequin 1.6% / attach_cape_mega 1.3%
#   / stretcher 1.3% / boss 0.9% / atk_turbo 0.7% / attach_w_cind 0.7%
#   / ball 0.6% / atk_gun 0.6% / attach_cape_staryu 0.5% / attach_ign_oth 0.1%
# 1%未満でも意味の異なる行動（boss/atk_turbo/ball等）は独立クラスとして保持。
# 発見(EXP-089踏襲): ABILITY optionのarea=7はSTADIUM（Spikemuth Gym効果=相手側
# スタジアムでも自分が起動可）。計33クラス。

JETTING_BLOW = 1487
NEBULA_BEAM = 1488
TURBO_FLARE = 965
WATER_GUN = 1486

_PLAY_TRAINER_CLASS = {
    Poffin: "poffin", Night_Stretcher: "stretcher", Hammer: "hammer",
    Ultra_Ball: "ball", Pokegear: "gear", Mega_Signal: "signal",
    Boss_Orders: "boss", Salvatore: "salvatore", Harlequin: "harlequin",
    Hilda: "hilda", Lillie: "lillie", Wally: "wally",
}
_PLAY_POKE_CLASS = {Staryu: "put_staryu"}
_EVO_CLASS = {Mega_Starmie: "evo_mega"}
_ATK_CLASS = {JETTING_BLOW: "atk_jetting", NEBULA_BEAM: "atk_nebula",
              TURBO_FLARE: "atk_turbo", WATER_GUN: "atk_gun"}

_ATTACH_CLASSES = {"attach_w_mega", "attach_w_staryu", "attach_w_cind",
                   "attach_ign_mega", "attach_ign_oth",
                   "attach_cape_mega", "attach_cape_staryu", "attach_oth"}

CLASSES = sorted(
    set(_PLAY_TRAINER_CLASS.values()) | set(_PLAY_POKE_CLASS.values())
    | set(_EVO_CLASS.values()) | set(_ATK_CLASS.values()) | _ATTACH_CLASSES
    | {"atk_other", "ab_gym", "ab_other", "retreat", "end", "other"})
CLASS_ID = {c: i for i, c in enumerate(CLASSES)}


def _attach_class(cid: int, tid: int) -> str:
    if cid == Basic_Water:
        if tid == Mega_Starmie:
            return "attach_w_mega"
        if tid == Staryu:
            return "attach_w_staryu"
        if tid == Cinderace:
            return "attach_w_cind"
        return "attach_oth"
    if cid == Ignition:
        return "attach_ign_mega" if tid == Mega_Starmie else "attach_ign_oth"
    if cid == Cape:
        if tid == Mega_Starmie:
            return "attach_cape_mega"
        if tid == Staryu:
            return "attach_cape_staryu"
        return "attach_oth"
    return "attach_oth"


def option_class(state, select, opt_i: int, my_index: int) -> str:
    """MAIN選択肢→行動クラス名。単一コードパス（収集・推論共用）。"""
    from cg.api import AreaType, OptionType
    o = select.option[opt_i]
    me = state.players[my_index]
    if o.type == OptionType.ATTACK:
        return _ATK_CLASS.get(o.attackId, "atk_other")
    if o.type == OptionType.END:
        return "end"
    if o.type == OptionType.RETREAT:
        return "retreat"
    if o.type == OptionType.ABILITY:
        try:
            if o.area == AreaType.STADIUM:
                cid = state.stadium[o.index].id
            elif o.area == AreaType.ACTIVE:
                cid = me.active[o.index].id
            elif o.area == AreaType.BENCH:
                cid = me.bench[o.index].id
            else:
                return "ab_other"
        except (IndexError, TypeError, AttributeError):
            return "ab_other"
        if cid == Spikemuth_Gym:
            return "ab_gym"
        return "ab_other"
    try:
        cid = me.hand[o.index].id
    except (IndexError, TypeError, AttributeError):
        return "other"
    if o.type == OptionType.PLAY:
        cd = card_table.get(cid)
        if cd is not None and cd.cardType == CardType.POKEMON:
            return _PLAY_POKE_CLASS.get(cid, "other")
        return _PLAY_TRAINER_CLASS.get(cid, "other")
    if o.type == OptionType.EVOLVE:
        return _EVO_CLASS.get(cid, "other")
    if o.type == OptionType.ATTACH:
        try:
            src = me.active if o.inPlayArea == AreaType.ACTIVE else me.bench
            tid = src[o.inPlayIndex].id
        except (IndexError, TypeError, AttributeError):
            return "attach_oth"
        return _attach_class(cid, tid)
    return "other"


FEAT_KEYS = None  # 初回featurize時に確定


def feat_vector(state, my_index: int):
    """特徴dictを固定順のリストにして返す（学習・推論共用）。"""
    global FEAT_KEYS
    f = featurize(state, my_index)
    if FEAT_KEYS is None:
        FEAT_KEYS = sorted(f)
    return [f[k] for k in FEAT_KEYS], FEAT_KEYS
