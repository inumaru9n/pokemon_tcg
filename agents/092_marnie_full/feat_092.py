"""EXP-089 Marnie(1ec0f47981)版ゾーン特徴器 + MAIN行動クラス分類 (I-114)。

EXP-057定式化のMarnie適用: 目的変数=「今この盤面で教師(Luca/1ec0)が取る行動クラス」。
feat_058の構成法（手札ID別/山∪サイドプール/トラッシュID別/資源勘定/両者盤面属性/
ターンフラグ）を1ec0f47981の60枚に対応させて再設計。
学習（リプレイ収集）と推論（main.py）で共用する単一コードパス（パリティ原則）。
"""

from __future__ import annotations

import os

from cg.api import CardType, EnergyType, all_card_data

# ---- 攻撃テーブル（EN_Card_Data.csv由来の自動生成モジュール。056のコピー） ----
try:
    from attack_table_092 import ATTACKS_056 as ATTACKS
except ImportError:
    import importlib.util as _ilu

    try:
        _dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        _dir = ("/kaggle_simulations/agent"
                if os.path.exists("/kaggle_simulations/agent/main.py")
                else os.getcwd())
    _p = os.path.join(_dir, "attack_table_092.py")
    _sp = _ilu.spec_from_file_location("attack_table_092", _p)
    _m = _ilu.module_from_spec(_sp)
    _sp.loader.exec_module(_m)
    ATTACKS = _m.ATTACKS_056

card_table = {c.cardId: c for c in all_card_data()}

# ---- カードID定数（main.py/EXP-082と同値） ----
Basic_Dark_Energy = 7
Froslass = 104
Munkidori = 112
Impidimp = 646
Morgrem = 647
Grimmsnarl = 648
Snorunt = 860
Rare_Candy = 1079
Unfair_Stamp = 1080
Buddy_Poffin = 1086
Night_Stretcher = 1097
Pokegear = 1122
Tool_Scrapper = 1137
Poke_Pad = 1152
Boss_Orders = 1182
Petrel = 1219
Lillie_Determination = 1227
Dawn = 1231
Spikemuth_Gym = 1259

Crustle_Wall = 345
Mega_Kangaskhan = 756

SPECIES = (Impidimp, Morgrem, Grimmsnarl, Munkidori, Snorunt, Froslass)
SPECIES_NAME = {Impidimp: "imp", Morgrem: "mor", Grimmsnarl: "grim",
                Munkidori: "munki", Snorunt: "sno", Froslass: "fro"}

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
DECK_IDS = sorted(DECK_COUNTS)  # 特徴キーの固定順（19種）


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
    """State（obs.current）→ 特徴dict。~110キー、全キー常に出力（0含む）。"""
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
    f["act_cant"] = 1.0 if (me.asleep or me.paralyzed) else 0.0
    f["act_fresh"] = 0.0
    if act is not None:
        k = SPECIES_NAME.get(act.id)
        if k:
            f[f"act_{k}"] = 1.0
        f["act_hp"] = act.hp
        f["act_dmg"] = act.maxHp - act.hp
        f["act_e"] = len(act.energyCards)
        f["act_fresh"] = 1.0 if act.appearThisTurn else 0.0
    for cid in SPECIES:
        f[f"n_{SPECIES_NAME[cid]}"] = sum(1 for p in my_bench if p.id == cid)

    field_all = ([act] if act is not None else []) + my_bench
    # 資源勘定・結合述語（Marnie版）
    f["ready_grim"] = sum(1 for p in field_all
                          if p.id == Grimmsnarl and len(p.energyCards) >= 2)
    f["charged_pre"] = sum(1 for p in field_all
                           if p.id in (Impidimp, Morgrem)
                           and len(p.energyCards) >= 2)
    f["munki_online"] = sum(1 for p in field_all
                            if p.id == Munkidori and len(p.energyCards) >= 1)
    f["grimline_field"] = sum(1 for p in field_all
                              if p.id in (Impidimp, Morgrem, Grimmsnarl))
    f["snow_field"] = sum(1 for p in field_all
                          if p.id in (Snorunt, Froslass))
    f["bench_free"] = me.benchMax - len(my_bench)
    f["bench_dmg"] = sum(p.maxHp - p.hp for p in my_bench)
    f["field_dmg_me"] = f["bench_dmg"] + f["act_dmg"]  # Adrena-Brainの弾

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

    # ---- 7. 相手の場(属性ベース + Marnie対面述語) ----
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
            weak = 1.0 if cd.weakness == EnergyType.DARKNESS else 0.0
            res = 1.0 if cd.resistance == EnergyType.DARKNESS else 0.0
        abil = 1.0 if _has_ability(op_act.id) else 0.0
    f["opp_act_weak_dark"] = weak
    f["opp_act_res_dark"] = res
    f["opp_act_ability"] = abil
    f["opp_wall_active"] = 1.0 if (op_act is not None
                                   and op_act.id == Crustle_Wall) else 0.0
    op_field = ([op_act] if op_act is not None else []) + op_bench
    f["opp_wall_play"] = 1.0 if any(p.id == Crustle_Wall for p in op_field) else 0.0
    f["opp_megakang_bench"] = sum(1 for p in op_bench if p.id == Mega_Kangaskhan)
    f["opp_n_ability"] = sum(1 for p in op_field if _has_ability(p.id))
    f["opp_munki_field"] = sum(1 for p in op_field if p.id == Munkidori)
    f["opp_bench_prz2"] = sum(1 for p in op_bench if _prz_of_id(p.id) >= 2)
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
# クラス設計は教師データのctx0選択の頻度集計（07-16 Luca, 4359決定）で確定:
#   ab_munki(Adrena) 12.7% / ab_gym(Spikemuth効果) 9.9% / atk_bullet 7.4%
#   / pad 6.3% / attach_munki 5.7% / evo_mor 5.0% / put_munki 4.8%
#   / stretcher 4.4% / lillie 4.3% / put_imp 4.0% / petrel 4.0% / poffin 3.6%
#   / gym 3.4% / end 3.1% / evo_fro 3.0% / evo_grim 2.3% / attach_pre 2.3%
#   / retreat 1.8% / candy 1.7% / stamp 1.7% / gear 1.5% / boss 1.4%
#   / attach_snow 1.3% / put_sno 1.2% / atk_cork 1.1% / atk_filch 0.6%
#   / dawn 0.6% / attach_grim 0.4% / scrapper 0.4% / ab_other 0.07% / atk_other 0.02%
# 1%未満でも意味の異なる行動（atk_filch/dawn/attach_grim/scrapper）は独立クラス
# として保持（otherに混ぜると背骨の別行動と衝突するため）。計32クラス。

SHADOW_BULLET = 937
CORKSCREW_PUNCH = 936
FILCH = 934

_PLAY_TRAINER_CLASS = {
    Rare_Candy: "candy", Unfair_Stamp: "stamp", Buddy_Poffin: "poffin",
    Night_Stretcher: "stretcher", Pokegear: "gear", Tool_Scrapper: "scrapper",
    Poke_Pad: "pad", Boss_Orders: "boss", Petrel: "petrel",
    Lillie_Determination: "lillie", Dawn: "dawn", Spikemuth_Gym: "gym",
}
_PLAY_POKE_CLASS = {Impidimp: "put_imp", Munkidori: "put_munki",
                    Snorunt: "put_sno"}
_EVO_CLASS = {Morgrem: "evo_mor", Grimmsnarl: "evo_grim", Froslass: "evo_fro"}
_ATK_CLASS = {SHADOW_BULLET: "atk_bullet", CORKSCREW_PUNCH: "atk_cork",
              FILCH: "atk_filch"}


def _attach_bucket(pid: int) -> str:
    if pid == Munkidori:
        return "munki"
    if pid == Grimmsnarl:
        return "grim"
    if pid in (Impidimp, Morgrem):
        return "pre"
    if pid in (Snorunt, Froslass):
        return "snow"
    return "oth"


_ATTACH_CLASSES = {"attach_munki", "attach_grim", "attach_pre", "attach_snow",
                   "attach_oth"}

CLASSES = sorted(
    set(_PLAY_TRAINER_CLASS.values()) | set(_PLAY_POKE_CLASS.values())
    | set(_EVO_CLASS.values()) | set(_ATK_CLASS.values()) | _ATTACH_CLASSES
    | {"atk_other", "ab_munki", "ab_grim", "ab_gym", "ab_other", "retreat",
       "end", "other"})
CLASS_ID = {c: i for i, c in enumerate(CLASSES)}


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
        if cid == Munkidori:
            return "ab_munki"
        if cid == Grimmsnarl:
            return "ab_grim"
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
        return f"attach_{_attach_bucket(tid)}"
    return "other"


FEAT_KEYS = None  # 初回featurize時に確定


def feat_vector(state, my_index: int):
    """特徴dictを固定順のリストにして返す（学習・推論共用）。"""
    global FEAT_KEYS
    f = featurize(state, my_index)
    if FEAT_KEYS is None:
        FEAT_KEYS = sorted(f)
    return [f[k] for k in FEAT_KEYS], FEAT_KEYS
