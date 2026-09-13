"""自動生成された特徴器（tools/gen_featurizer.py）。手で編集しないこと。

デッキ: agents/102_alakazam_feat2/deck.csv
ポケモン 7種 / 進化ライン 4本 / エネ 3種 / スタジアム 1種
行動クラス 65個

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

# ---- 効果文から復元した打点（tools/gen_atk_dmg.py 生成。デッキ非依存） ----
# attack_table_056 の Damage 列は「n/a」の技を全て打点不明にしており、その中に
# Alakazam の Powerful Hand（手札1枚につき20）のような主砲が含まれる。
ATK_FIX = {
    (40, 'Mirage Barrage'): (120, 1),
    (45, 'Pinpoint Dive'): (60, 0),
    (80, 'Twin Shotels'): (50, 1),
    (94, 'Cursed Drop'): (40, 1),
    (140, 'Cruel Arrow'): (100, 1),
    (144, 'Trifrost'): (110, 1),
    (153, 'Garnet Volley'): (180, 1),
    (189, 'Sonic Peridot'): (100, 1),
    (215, 'Painful Memories'): (20, 1),
    (219, 'Law of the Underworld'): (60, 1),
    (241, 'Severe Squall'): (60, 1),
    (247, 'Sneaky Placement'): (20, 1),
    (266, 'Thump-Thump Boom'): (100, 1),
    (302, 'Wide Blast'): (50, 0),
    (372, 'Dual Bolt'): (50, 1),
    (377, 'Thunder Raid'): (210, 0),
    (438, 'Drag Off'): (30, 1),
    (449, 'Spinning Tail'): (30, 1),
    (508, 'Drag Off'): (20, 1),
    (591, 'Telekinesis'): (70, 1),
    (638, 'Sonic Double'): (50, 1),
    (665, 'Jumping Kick'): (40, 1),
    (730, 'Chilling Wings'): (20, 1),
    (759, 'Dashing Kick'): (50, 0),
    (774, 'Sniping Feathers'): (120, 1),
    (805, 'Targeted Dive'): (70, 0),
    (817, 'Sneaky Placement'): (10, 1),
    (844, 'Dual Tail'): (60, 1),
    (868, 'Split Bomb'): (60, 1),
    (880, 'Phantasmal Barrage'): (120, 1),
    (889, 'Tar Cannon'): (140, 1),
    (928, 'Explosion Y'): (280, 1),
    (946, 'Spit Shot'): (120, 1),
    (984, 'Bone Shot'): (50, 1),
    (1021, 'Feather Shot'): (90, 1),
    (1058, 'Haunt'): (30, 1),
    (1064, 'Sonic Ripper'): (220, 1),
}
ATK_VAR = {
    (171, 'Thunderburst Storm'): ('energy', 30, 1),
    (620, 'Power Whip'): ('energy', 20, 1),
    (743, 'Powerful Hand'): ('hand', 20, 1),
}


# ---- 相手側の判定に使う固定ID（自デッキと無関係。094から継承） ----
BOSS_ORDERS = 1182
TR_ARTICUNO = 414
MIST_IDS = (11, 20)          # Mist Energy / Rock Fighting Energy（PH無効化）

# ---- 自デッキから導出した定数 ----
SPECIES = (66, 140, 305, 343, 741, 742, 743)
SPECIES_NAME = {66: 'dudunsparce', 140: 'fezandipiti', 305: 'dunsparce', 343: 'shaymin', 741: 'abra', 742: 'kadabra', 743: 'alakazam'}
LINE_OF = {66: 'dunsparce', 140: 'fezandipiti', 305: 'dunsparce', 343: 'shaymin', 741: 'abra', 742: 'abra', 743: 'abra'}        # cardId -> 進化ライン名（最下段の種の名前）
LINE_NAMES = ('abra', 'dunsparce', 'fezandipiti', 'shaymin')
PRE_EVO = (305, 741, 742)        # 同デッキ内に進化先を持つ種
ENERGY_NAME = {5: 'basic_p_en', 13: 'enriching_', 19: 'telepath_p'}      # 全エネ（基本+特殊）の cardId -> 短縮名
BASIC_ENERGY_NAME = {5: 'basic_p_en'}
TOOL_NAME = {}     # ポケモンに付けるツール（ATTACHで来る）
STADIUM_NAME = {1266: 'nighttime_'}
ATTACH_BUCKET = {66: 'dunsparce_f', 140: 'fezandipiti_f', 305: 'dunsparce_p', 343: 'shaymin_f', 741: 'abra_p', 742: 'abra_p', 743: 'abra_f'}   # エネ貼り先の種 -> クラス分類のバケット
CLS_PLAY = {66: 'play_dudunsparce', 140: 'play_fezandipiti', 305: 'play_dunsparce', 343: 'play_shaymin', 741: 'play_abra', 742: 'play_kadabra', 743: 'play_alakazam', 1079: 'use_rare_candy', 1081: 'use_enhanced_h', 1086: 'use_buddy_budd', 1097: 'use_night_stre', 1129: 'use_sacred_ash', 1152: 'use_poké_pad', 1182: 'use_orders', 1184: 'use_aid', 1197: 'use_machinatio', 1225: 'use_hilda', 1231: 'use_dawn', 1266: 'use_nighttime_'}
CLS_EVO = {66: 'evo_dudunsparce', 742: 'evo_kadabra', 743: 'evo_alakazam'}
CLS_AB = {66: 'ab_dudunsparce', 140: 'ab_fezandipiti', 305: 'ab_dunsparce', 343: 'ab_shaymin', 741: 'ab_abra', 742: 'ab_kadabra', 743: 'ab_alakazam', 1266: 'ab_nighttime_'}        # ポケモンの特性 + 効果を持つスタジアム
CLS_ATK = {76: 'atk_dudunsparce_0', 183: 'atk_fezandipiti_0', 423: 'atk_dunsparce_0', 424: 'atk_dunsparce_1', 477: 'atk_shaymin_0', 1070: 'atk_abra_0', 1071: 'atk_kadabra_0', 1072: 'atk_alakazam_0'}      # attackId -> クラス名（技の打ち分け）
CLASSES = ('ab_abra', 'ab_alakazam', 'ab_dudunsparce', 'ab_dunsparce', 'ab_fezandipiti', 'ab_kadabra', 'ab_nighttime_', 'ab_other', 'ab_shaymin', 'ab_stadium', 'atk_abra_0', 'atk_alakazam_0', 'atk_dudunsparce_0', 'atk_dunsparce_0', 'atk_dunsparce_1', 'atk_fezandipiti_0', 'atk_kadabra_0', 'atk_other', 'atk_shaymin_0', 'attach_basic_p_en_abra_f', 'attach_basic_p_en_abra_p', 'attach_basic_p_en_dunsparce_f', 'attach_basic_p_en_dunsparce_p', 'attach_basic_p_en_fezandipiti_f', 'attach_basic_p_en_oth', 'attach_basic_p_en_shaymin_f', 'attach_enriching__abra_f', 'attach_enriching__abra_p', 'attach_enriching__dunsparce_f', 'attach_enriching__dunsparce_p', 'attach_enriching__fezandipiti_f', 'attach_enriching__oth', 'attach_enriching__shaymin_f', 'attach_telepath_p_abra_f', 'attach_telepath_p_abra_p', 'attach_telepath_p_dunsparce_f', 'attach_telepath_p_dunsparce_p', 'attach_telepath_p_fezandipiti_f', 'attach_telepath_p_oth', 'attach_telepath_p_shaymin_f', 'end', 'evo_alakazam', 'evo_dudunsparce', 'evo_kadabra', 'other', 'play_abra', 'play_alakazam', 'play_dudunsparce', 'play_dunsparce', 'play_fezandipiti', 'play_kadabra', 'play_shaymin', 'retreat', 'use_aid', 'use_buddy_budd', 'use_dawn', 'use_enhanced_h', 'use_hilda', 'use_machinatio', 'use_night_stre', 'use_nighttime_', 'use_orders', 'use_poké_pad', 'use_rare_candy', 'use_sacred_ash')


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



# ==== 対面識別・脅威判定に使う定数（メタ知識。全レプリカ共通なのでコア側に置く） ====
# 各アーキタイプを一意に特定できる看板カード（直近メタの個体デッキから弁別力で自動選定）。
# **相手がどのデッキかを表す特徴が1つも無かった**のが従来の最大の穴。教師は対面ごとに
# 全く違う打ち方をするのに、モデルはそれを条件付けられなかった
# （実測: 同一デッキ156952a871でも、対Spidopsの勝率がパイロット間で17.9% vs 51.0%）。
META_CARDS = [2, 6, 11, 13, 14, 15, 18, 19, 89, 90, 92, 93, 104, 119, 120, 121, 235, 341, 342, 344, 345, 379, 380, 381, 387, 400, 401, 414, 431, 646, 647, 648, 741, 742, 743, 756, 860, 1071, 1081, 1134, 1142, 1147, 1173, 1198, 1216, 1217, 1218, 1219, 1220, 1245, 1259, 1261]
META_ARCH = {"Marnie's Grimmsnarl ex": [104, 646, 647, 648, 860, 1219, 1259], 'Mega Kangaskhan ex': [11, 14, 18, 344, 345, 756, 1147], 'Alakazam': [13, 19, 741, 742, 743, 1081], "Team Rocket's Spidops": [15, 400, 401, 414, 431, 1134, 1216, 1217, 1218, 1220], 'Dragapult ex': [2, 119, 120, 121, 235, 1071, 1198], "Cynthia's Garchomp ex": [6, 341, 342, 379, 380, 381, 387, 1142, 1173, 1261], 'Thwackey': [89, 90, 92, 93, 1245]}
ARCH_SLUG = {a: "".join(ch for ch in a.lower().replace(" ", "_")
                        if ch.isalnum() or ch == "_")[:14]
             for a in META_ARCH}
# 自デッキの各カードから進化できる先（選択肢の「取ると進化がつながるか」判定用）
META_ARCH_NAMES = ['Alakazam', "Cynthia's Garchomp ex", 'Dragapult ex', "Marnie's Grimmsnarl ex", 'Mega Kangaskhan ex', "Team Rocket's Spidops", 'Thwackey']


def _resolve_dmg(cid: int, mv: str, dmg, ctx):
    """1つの技の (アクティブに通る打点, 打点が解決できたか) を返す。

    **打点解決はここ1か所に集約する**。`attack_table_056.py` は EN_Card_Data.csv の
    Damage 列しか見ておらず、打点が効果文にしか書かれていない技（全体の約26%）を
    すべて None にしている。その中には **Alakazam の唯一の攻撃 Powerful Hand**
    （手札1枚につき20）が含まれる。ATK_FIX/ATK_VAR はその効果文を解析した表
    （tools/gen_atk_dmg.py 生成）。

    ベンチ限定の技（Shaymin の Pinpoint Dive 等）はアクティブに通らないので 0 を返す。
    """
    if dmg is not None:
        return dmg, True
    key = (cid, (mv or "").strip())
    t = ATK_FIX.get(key)
    if t is not None:
        return (t[0] if t[1] else 0), True
    t = ATK_VAR.get(key)
    if t is not None:
        kind, unit, to_act = t
        return (unit * max(0, ctx.get(kind, 0)) if to_act else 0), True
    return 0, False


def _atk_scan(p, n_energy: int, ctx):
    """p が n_energy 個のエネで出せる (最大打点, 未解決可変フラグ, 最小要求エネ)。

    ctx は打点が状態依存の技のための盤面数値: hand / bench / energy / prize_taken /
    discard_poke。**p の視点**（自分の技なら自分の手札枚数）で渡すこと。
    """
    mx, var, need = 0, 0, 99
    for (cn, dmg, mv) in ATTACKS_056.get(p.id, ()):
        d, ok = _resolve_dmg(p.id, mv, dmg, ctx)
        if not ok:
            if cn <= n_energy:
                var = 1          # 打点不明だが撃てる技がある
            continue
        if d > 0 and cn < need:
            need = cn
        if cn <= n_energy and d > mx:
            mx = d
    return mx, var, (0 if need == 99 else need)


def _atk_ctx(ps, p, extra_energy: int = 0, extra_hand: int = 0):
    """_atk_scan に渡す ctx を、プレイヤー ps・ポケモン p から組み立てる。"""
    return {
        "hand": (ps.handCount if hasattr(ps, "handCount") else len(ps.hand or []))
        + extra_hand,
        "bench": sum(1 for b in ps.bench if b is not None),
        "energy": len(p.energyCards or []) + extra_energy,
        "prize_taken": 6 - len(ps.prize or []),
        "discard_poke": sum(1 for c in ps.discard
                            if getattr(card_table.get(c.id), "cardType", None)
                            == CardType.POKEMON),
    }


def _clock(hp: float, dmg: float) -> float:
    """残HPを打点で割った「あと何回殴られるか」。打点0なら大きな値。"""
    if dmg <= 0:
        return 9.0
    import math
    return min(9.0, math.ceil(hp / dmg))


def _prz_of_id(cid: int) -> int:
    cd = card_table.get(cid)
    if cd is None:
        return 1
    return 3 if cd.megaEx else (2 if cd.ex else 1)


def _maxdmg_next(p, opp_player) -> tuple[int, int]:
    """相手ポケモン p が次のターン（エネ+1・ドロー+1）に出せる (最大打点, 可変フラグ)。

    以前は Powerful Hand だけを名指しで特別扱いしていたが、`_resolve_dmg` に統一した。
    """
    e_next = len(p.energyCards or []) + 1
    ctx = _atk_ctx(opp_player, p, extra_energy=1, extra_hand=1)
    mx, var, _ = _atk_scan(p, e_next, ctx)
    return mx, var


EVO_TARGETS = {
    c: tuple(t for t in DECK_IDS
             if getattr(card_table.get(t), "evolvesFrom", None)
             == getattr(card_table.get(c), "name", None))
    for c in DECK_IDS}
EVO_TARGETS = {k: v for k, v in EVO_TARGETS.items() if v}


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
    mx, var = _maxdmg_next(op_act, op) if op_act else (0, 0)
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
        [mx] + [_maxdmg_next(p, op)[0] for p in op_bench], default=0)
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

    # ---- 10. 対面識別（相手がどのデッキか） ----
    # **従来ここが完全に欠落していた**。教師は対面ごとに全く違う打ち方をするのに、
    # モデルはそれを条件付けられなかった。相手が公開した領域（場・付随カード・
    # トラッシュ・スタジアム）から、アーキタイプの看板カードの観測数を数える。
    op_field = ([op_act] if op_act is not None else []) + op_bench
    op_seen: dict[int, int] = {}
    for p in op_field:
        op_seen[p.id] = op_seen.get(p.id, 0) + 1
        for grp in ("energyCards", "tools", "preEvolution"):
            for c in (getattr(p, grp, None) or []):
                op_seen[c.id] = op_seen.get(c.id, 0) + 1
    for c in op.discard:
        op_seen[c.id] = op_seen.get(c.id, 0) + 1
    for c in state.stadium:
        if c.playerIndex != my_index:
            op_seen[c.id] = op_seen.get(c.id, 0) + 1
    for cid in META_CARDS:
        f[f"opp_seen_{cid}"] = float(op_seen.get(cid, 0))
    # 系統ごとの「看板カードを何割見たか」。1枚でも見えれば強い証拠になる
    for a in META_ARCH_NAMES:
        ids = META_ARCH[a]
        hit = sum(1 for cid in ids if op_seen.get(cid))
        f[f"oparch_{ARCH_SLUG[a]}"] = hit / max(1, len(ids))

    # ---- 11. 脅威とクロック（ダメージレースの算術） ----
    # **これらは1つも無かった**。「今KOできるか」「次にKOされるか」は決定に直結する。
    my_dmg, my_var, my_need = (
        _atk_scan(act, len(act.energyCards or []), _atk_ctx(me, act))
        if act is not None else (0, 0, 0))
    op_dmg_now, op_var, _ = (
        _atk_scan(op_act, len(op_act.energyCards or []), _atk_ctx(op, op_act))
        if op_act is not None else (0, 0, 0))
    op_dmg_next, _ = _maxdmg_next(op_act, op) if op_act else (0, 0)
    f["my_best_dmg"] = float(my_dmg)
    f["my_var_atk"] = float(my_var)
    f["my_energy_need"] = float(max(0, my_need - (len(act.energyCards or [])
                                                  if act else 0)))
    f["opp_best_dmg_now"] = float(op_dmg_now)
    f["opp_best_dmg_next"] = float(op_dmg_next)
    opp_hp = float(op_act.hp) if op_act else 0.0
    my_hp = float(act.hp) if act else 0.0
    f["ko_opp_active"] = 1.0 if (op_act is not None and my_dmg >= opp_hp > 0) else 0.0
    f["they_ko_me"] = 1.0 if (act is not None and op_dmg_next >= my_hp > 0) else 0.0
    f["clock_them"] = _clock(opp_hp, my_dmg)         # 相手を倒すのに何回
    f["clock_me"] = _clock(my_hp, op_dmg_next)       # 自分が倒れるまで何回
    f["clock_diff"] = f["clock_me"] - f["clock_them"]   # 正=こちらが速い
    f["dmg_margin"] = float(my_dmg) - opp_hp         # 打点の過不足
    # サイドレース: KOで何枚動くか / それで決着するか
    f["prz_if_my_act_ko"] = float(_prz_of_id(act.id)) if act else 0.0
    f["prz_if_opp_act_ko"] = float(_prz_of_id(op_act.id)) if op_act else 0.0
    f["lethal_for_me"] = 1.0 if (f["ko_opp_active"] and
                                 prz_me <= f["prz_if_opp_act_ko"]) else 0.0
    f["lethal_for_opp"] = 1.0 if (f["they_ko_me"] and
                                  prz_op <= f["prz_if_my_act_ko"]) else 0.0

    # ---- 12. 盤面のスロット単位（集約をやめる） ----
    # ベンチを合計値でしか持っていなかったが、「どのスロットの誰が傷んでいるか」は
    # ボスの指令の的・進化先・エネの貼り先の判断に直結する。選択肢側の inPlayIndex と
    # 対応が取れる形で、スロットごとに出す。
    # bench は **詰めずに** 使う。空きスロットを詰めると選択肢側の inPlayIndex と
    # 番号が食い違い、「どのスロットを指しているか」を突き合わせられなくなる。
    for side, lst, tag in ((me, [act] + list(me.bench), "ms"),
                           (op, [op_act] + list(op.bench), "os")):
        for i in range(6):
            p = lst[i] if i < len(lst) else None
            pre = f"{tag}{i}_"
            f[pre + "on"] = 1.0 if p is not None else 0.0
            f[pre + "hpr"] = (p.hp / p.maxHp) if (p and p.maxHp) else 0.0
            f[pre + "dmg"] = float(p.maxHp - p.hp) if p else 0.0
            f[pre + "e"] = float(len(p.energyCards or [])) if p else 0.0
            f[pre + "tool"] = float(len(getattr(p, "tools", []) or [])) if p else 0.0
            f[pre + "pre"] = float(len(getattr(p, "preEvolution", []) or [])) if p else 0.0
            f[pre + "prz"] = float(_prz_of_id(p.id)) if p else 0.0
            f[pre + "fresh"] = 1.0 if (p and p.appearThisTurn) else 0.0
            if tag == "ms":
                for cid in SPECIES:      # 自分側は種族まで持つ（相手は自デッキ外なので不可）
                    f[pre + SPECIES_NAME[cid]] = 1.0 if (p and p.id == cid) else 0.0
            else:
                f[pre + "dmgnext"] = float(_maxdmg_next(p, op)[0]) if p else 0.0

    # ---- 13. 資源とデッキアウト ----
    n_by_type = {"poke": 0, "sup": 0, "item": 0, "tool": 0, "energy": 0, "stad": 0}
    for c in my_hand:
        cd = card_table.get(c.id)
        t = getattr(cd, "cardType", None)
        if t == CardType.POKEMON:
            n_by_type["poke"] += 1
        elif t == CardType.SUPPORTER:
            n_by_type["sup"] += 1
        elif t == CardType.ITEM:
            n_by_type["item"] += 1
        elif t == CardType.TOOL:
            n_by_type["tool"] += 1
        elif t in (CardType.BASIC_ENERGY, CardType.SPECIAL_ENERGY):
            n_by_type["energy"] += 1
        elif t == CardType.STADIUM:
            n_by_type["stad"] += 1
    for k, v in n_by_type.items():
        f[f"hand_n_{k}"] = float(v)
    f["deckout_me"] = float(me.deckCount)      # 1ターン1ドローなので枚数がそのままターン数
    f["deckout_op"] = float(op.deckCount)

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
        # **エネだけでなくツールも ATTACH で来る**（Hero's Cape / Brave Bangle 等）。
        # エネしか見ないとツール装着が全て "other" に落ちる
        # （Spidopsの教師データではMAIN決定の1.9%）。貼り先の選択も伴う行動なので
        # エネと同じ「対象バケット付き」のクラスにする。
        et = ENERGY_NAME.get(cid) or TOOL_NAME.get(cid)
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
OPT_DIM = 28            # option_vector が返す数値特徴の次元（24 + 可能化4）
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
        # **「そのカードが何を可能にするか」**（ここまでは「何であるか」しか無かった）。
        # サーチ先の選択(ctx7)は全決定の15%を占め、一致率が0.62と最も低い区間だった。
        # カードIDの埋め込みだけから「今このカードが要るか」を学ばせるのは無理がある。
        *_enable_feats(obs, opt, my_index, cid, pool_left),
    ]


def _enable_feats(obs, opt, my_index: int, cid: int, pool_left):
    """選択肢のカードが何を可能にするか（4次元）。

      enables_evo   : そのカードの進化元が自分の場にいるか（＝取れば進化できる）
      is_pre_needed : そのカードの進化先を自分が持っている（場or手札）か
      completes_atk : エネなら、アクティブの最小要求エネを満たすか
      left_after    : 取った後に山∪サイドに残る枚数
    """
    try:
        st = obs.current
        ps = st.players[my_index]
        cd = card_table.get(cid)
        if cd is None:
            return [0.0, 0.0, 0.0, 0.0]
        actv = ps.active[0] if (ps.active and ps.active[0]) else None
        field = ([actv] if actv else []) + [b for b in ps.bench if b is not None]
        src = getattr(cd, "evolvesFrom", None)
        enables = 0.0
        if src:
            enables = 1.0 if any(
                getattr(card_table.get(p.id), "name", None) == src for p in field) else 0.0
        # 自分がこのカードから進化する先を持っているか（EVO_TARGETS は事前計算）
        tgt = EVO_TARGETS.get(cid)
        need_pre = 0.0
        if tgt:
            have = {c.id for c in (ps.hand or [])} | {p.id for p in field}
            need_pre = 1.0 if any(t in have for t in tgt) else 0.0
        completes = 0.0
        if cid in ENERGY_NAME and actv is not None:
            _mx, _v, need = _atk_scan(actv, len(actv.energyCards or []),
                                      _atk_ctx(ps, actv))
            completes = 1.0 if (
                need and len(actv.energyCards or []) + 1 >= need) else 0.0
        return [enables, need_pre, completes,
                float(max(0, pool_left.get(cid, 0) - 1))]
    except (IndexError, TypeError, AttributeError):
        return [0.0, 0.0, 0.0, 0.0]


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
