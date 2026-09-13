"""EXP-096: Marnie(1ec0f47981) の全選択NN方策用 特徴器。

**構成**（EXP-094と同水準にするための組み立て）:
  - 盤面特徴（セクション1〜9）と行動クラス(CLASS_ID) = agents/092_marnie_full/feat_092.py
    と同一。092はEXP-057系の同じ設計テンプレートで作られており、9セクションが094と対応する。
    Marnie固有述語（ready_grim / grimline_field / munki_online / opp_wall_* 等）を含む
  - 選択肢特徴（option_vector / resolve_option / logs_vector / pool_counts / NULL）
    = agents/094_yushin_nn/feat_094.py からの移植。**デッキ非依存**であり、
    EXP-094で見つけた以下の修正がすべて入っている:
      A-1 PLAYは area=None で index が手札index（無いとMAIN選択肢の42%が未解決）
      A-2 inPlayArea/inPlayIndex（無いとMAIN決定の14.6%が判別不能）
      opt.playerIndex の尊重 / hp・maxHp・tools の属性名 / toolIndex・energyIndex
      minCount==0 の棄権(NULL)オプション

教師: Dries @ Tufa Labs、署名 1ec0f47981、07-26〜28（833戦・勝率60.1%）。
デッキは092のdeck.csvと**完全一致を検証済み**。
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
