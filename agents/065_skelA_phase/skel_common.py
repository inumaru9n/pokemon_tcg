"""ルール骨格A/B/Cの共有基盤（EXP-065/066/067）。

盤面事実の抽出・選択肢のセマンティックラベリング・カード/ポケモンの基礎価値を提供する。
A/B/C はこの基盤の上で「全選択をどう組織するか」だけが異なる。
モデル無し・純ルール。Majkelデッキ156952a871前提。

注意: main.py 側から必要な定数（Abra等のカードID、AreaType/OptionType/SelectContext,
PSYCHIC_ENERGY_IDS 等）を setup() で受け取る（各エージェントで定数定義が同じため）。
"""

from __future__ import annotations

_C = {}  # 定数・enum のバインド


def setup(consts: dict):
    _C.update(consts)


# ---- 盤面事実 ----
def facts(state, my_index):
    C = _C
    me = state.players[my_index]
    op = state.players[1 - my_index]
    field = ([me.active[0]] if me.active and me.active[0] else []) \
        + [b for b in me.bench if b is not None]

    def psy(p):
        return sum(1 for e in p.energyCards if e.id in C["PSYCHIC_ENERGY_IDS"])

    ready_zam = any(p.id == C["Alakazam"] and psy(p) >= 1 for p in field)
    loaded_pre = sum(1 for p in field
                     if p.id in (C["Abra"], C["Kadabra"]) and len(p.energyCards) >= 1)
    hand_ids = [c.id for c in (me.hand or [])]
    from collections import Counter
    hc = Counter(hand_ids)
    field_ids = Counter(p.id for p in field)
    op_field = ([op.active[0]] if op.active and op.active[0] else []) \
        + [b for b in op.bench if b is not None]
    return {
        "me": me, "op": op, "field": field, "field_ids": field_ids,
        "hand_ids": hand_ids, "hc": hc, "psy": psy,
        "ready_zam": ready_zam, "loaded_pre": loaded_pre,
        "hand": me.handCount, "deck": me.deckCount,
        "opp_hand": op.handCount, "prz_me": len(me.prize), "prz_op": len(op.prize),
        "turn": state.turn, "op_field": op_field,
        "op_active": (op.active[0] if op.active else None),
        "bench_free": me.benchMax - len([b for b in me.bench if b is not None]),
    }


# ---- 選択肢ラベリング: (kind, cid_or_pokemon) ----
def label(state, select, i, my_index):
    C = _C
    o = select.option[i]
    me = state.players[my_index]
    op = state.players[1 - my_index]
    OT = C["OptionType"]
    AT = C["AreaType"]

    def hand_id(idx):
        try:
            return me.hand[idx].id
        except (IndexError, TypeError, AttributeError):
            return None

    def pk(area, idx, player):
        try:
            return player.active[idx] if area == AT.ACTIVE else player.bench[idx]
        except (IndexError, TypeError, AttributeError):
            return None

    t = o.type
    if t == OT.END:
        return ("end", None)
    if t == OT.ATTACK:
        return ("attack", o.attackId)
    if t == OT.RETREAT:
        return ("retreat", None)
    if t == OT.ABILITY:
        src = pk(o.area, o.index, me)
        return ("ability", src.id if src else None)
    if t == OT.PLAY:
        return ("play", hand_id(o.index))
    if t == OT.EVOLVE:
        return ("evolve", hand_id(o.index))
    if t == OT.ATTACH:
        tp = pk(o.inPlayArea, o.inPlayIndex, me)
        return ("attach", (hand_id(o.index), tp.id if tp else None,
                           o.inPlayArea == AT.ACTIVE))
    if t == OT.CARD:
        a = o.area
        if a in (AT.ACTIVE, AT.BENCH):
            tp = pk(a, o.index, me)
            if tp is None:
                tp = pk(a, o.index, op)
                return ("select_opp_pokemon", tp.id if tp else None)
            return ("select_my_pokemon", tp.id)
        if a == AT.HAND:
            return ("discard_hand", hand_id(o.index))
        if a == AT.DISCARD:
            try:
                return ("from_discard", me.discard[o.index].id)
            except (IndexError, TypeError, AttributeError):
                return ("from_discard", None)
        # DECK/looking からのサーチ
        try:
            return ("search", select.deck[o.index].id if select.deck else None)
        except (IndexError, TypeError, AttributeError):
            return ("search", None)
    return ("other", None)


# ---- 基礎価値（Alakazamデッキ156952a871）----
# keep_value: 手札に残す価値（DISCARDで下位を捨てる）。高い=残す
def keep_value(cid, f):
    C = _C
    fid = f["field_ids"]
    # 進化ラインが場で必要なら残す
    if cid == C["Alakazam"]:
        return 55 if (fid[C["Kadabra"]] or fid[C["Abra"]]) else 40
    if cid == C["Kadabra"]:
        return 50 if fid[C["Abra"]] else 35
    if cid == C["Abra"]:
        return 30  # 供給多い、最も捨ててよい寄りだが種は要る
    if cid == C["Rare_Candy"]:
        return 60 if (fid[C["Abra"]] and C["Alakazam"] in f["hand_ids"]) else 45
    if cid == C["Dudunsparce"]:
        return 58 if fid[C["Dunsparce"]] else 40
    if cid == C["Dunsparce"]:
        return 42
    if cid == C["Boss_Orders"]:
        return 75  # 締めの必須札、温存
    if cid == C["Xerosic"]:
        return 62  # 妨害、温存寄り
    if cid == C["Enhanced_Hammer"]:
        return 55
    if cid in (C["Hilda"], C["Dawn"]):
        return 66  # ドローサポーター、温存
    if cid == C["Lanas_Aid"]:
        return 48
    if cid in C["PSYCHIC_ENERGY_IDS"]:
        return 50  # エネは山6枚のみ、そこそこ温存
    if cid == C["Enriching_Energy"]:
        return 52
    if cid in (C["Buddy_Buddy_Poffin"], C["Poke_Pad"]):
        return 44
    if cid == C["Nighttime_Mine"]:
        return 46
    if cid == C["Night_Stretcher"]:
        return 54
    if cid == C["Sacred_Ash"]:
        return 50
    if cid in (C["Fezandipiti_ex"], C["Shaymin"]):
        return 36
    return 45


# search_value: サーチで取ってくる価値（進化ライン前進を重視）
def search_value(cid, f):
    C = _C
    fid = f["field_ids"]
    hid = f["hand_ids"]
    if cid == C["Kadabra"] and fid[C["Abra"]] and C["Kadabra"] not in hid:
        return 95
    if cid == C["Alakazam"] and (fid[C["Kadabra"]] or fid[C["Abra"]]) and not f["ready_zam"]:
        return 92
    if cid == C["Dudunsparce"] and fid[C["Dunsparce"]] and not fid[C["Dudunsparce"]]:
        return 85
    if cid == C["Abra"] and f["bench_free"] > 0 and fid[C["Abra"]] < 2:
        return 70
    if cid == C["Dunsparce"] and f["bench_free"] > 0:
        return 68
    if cid == C["Rare_Candy"] and fid[C["Abra"]]:
        return 66
    if cid == C["Boss_Orders"] and f["prz_me"] <= 3:
        return 72
    if cid in (C["Hilda"], C["Dawn"]):
        return 55
    if cid in C["PSYCHIC_ENERGY_IDS"]:
        return 50
    return 40


# attack_value: そのポケモンをアクティブにした時の攻撃態勢（SWITCH/TO_ACTIVE用）
# 精読120戦+全数検証: 昇格先の選択がミラー/対壁の勝敗を分ける最大要因(L1/L2)
def attack_value(p, f):
    C = _C
    psy = f["psy"](p)
    # L2: Fez ex(2プライズ)を前線に出すのは厳禁(負け40.2% vs 勝ち18.8%)
    if p.id == C["Fezandipiti_ex"]:
        return -100
    if p.id == C["Shaymin"]:
        return -50   # Shaymin先頭も弱い(対アグロ裏目、精読ep85309304)
    if p.id == C["Alakazam"] and psy >= 1:
        return 100   # L1: ready Alakazam最優先(終盤これを保てるかが勝敗)
    if p.id == C["Dudunsparce"] and len(p.energyCards) >= 3:
        return 80
    if p.id == C["Kadabra"] and psy >= 1:
        return 62
    if p.id == C["Alakazam"]:
        return 50    # 未装填でも進化先として前線維持の価値
    if p.id == C["Kadabra"]:
        return 44
    if p.id == C["Dunsparce"]:
        return 42    # Trading Placesで無償帰還
    if p.id == C["Abra"]:
        return 30
    return 20


# 精読spec由来の共有シグナル（A/B/C共通の状況判断）
def spec_signals(f):
    C = _C
    fid = f["field_ids"]
    hid = f["hand_ids"]
    # 2体目ready Alakazam確保状況（S3: KO後即交代のため）
    ready_zams = sum(1 for p in f["field"]
                     if p.id == C["Alakazam"] and f["psy"](p) >= 1)
    backup_ready = ready_zams >= 2
    # Alakazam完成の速さ（S2）: 場にKadabra/Abraがいて手札にZam/Candy
    can_build_zam = (fid[C["Kadabra"]] or (fid[C["Abra"]] and C["Rare_Candy"] in hid)) \
        and C["Alakazam"] in hid
    # 山切れ危険（K1: 全数検証で負けの36.4% vs 勝ち6.2%。閾値を精読に合わせ拡大）
    deckout_risk = f["deck"] <= 10 and f["hand"] >= 8
    # 対アグロ（相手が高打点で自分が押されている: サイド負け進行）
    under_pressure = f["prz_me"] > f["prz_op"]
    return {"ready_zams": ready_zams, "backup_ready": backup_ready,
            "can_build_zam": can_build_zam, "deckout_risk": deckout_risk,
            "under_pressure": under_pressure}


# ============================================================================
# 全コンテキスト・ディスパッチャ: pol（骨格A/B/C）のメソッドを呼んで全選択を処理
#   pol が実装するメソッド（各骨格が phase/goal/economy でこれらを実装）:
#     main_priority(cls, f) -> float     # MAIN行動クラスの優先度
#     discard_keep(cid, f)  -> float     # 手札カードの残す価値（下位を捨てる）
#     search_pref(cid, f)   -> float     # サーチで取る価値
#     switch_pref(p, f)     -> float     # アクティブ昇格の価値
#     activate_yes(f, aid)  -> bool      # 能力YesNo
# ============================================================================

def main_class(state, select, i, my_index):
    """MAIN選択肢 → クラス名（feat系と同粒度の粗い分類）。"""
    C = _C
    o = select.option[i]
    me = state.players[my_index]
    OT = C["OptionType"]
    AT = C["AreaType"]
    if o.type == OT.ATTACK:
        return "atk"
    if o.type == OT.END:
        return "end"
    if o.type == OT.RETREAT:
        return "retreat"
    if o.type == OT.ABILITY:
        try:
            src = (me.active[o.index] if o.area == AT.ACTIVE else me.bench[o.index])
            return {C["Dudunsparce"]: "ab_dud",
                    C["Fezandipiti_ex"]: "ab_fez"}.get(src.id, "ab_other")
        except (IndexError, TypeError, AttributeError):
            return "ab_other"
    try:
        cid = me.hand[o.index].id
    except (IndexError, TypeError, AttributeError):
        return "other"
    play_map = {C["Buddy_Buddy_Poffin"]: "poffin", C["Poke_Pad"]: "pad",
                C["Enhanced_Hammer"]: "hammer", C["Rare_Candy"]: "candy",
                C["Boss_Orders"]: "boss", C["Xerosic"]: "xero",
                C["Dawn"]: "dawn", C["Hilda"]: "hilda", C["Lanas_Aid"]: "lana",
                C["Nighttime_Mine"]: "mine", C["Night_Stretcher"]: "stretcher",
                C["Sacred_Ash"]: "ash", C["Abra"]: "put_abra",
                C["Dunsparce"]: "put_dun", C["Fezandipiti_ex"]: "put_fez",
                C["Shaymin"]: "put_shay"}
    evo_map = {C["Kadabra"]: "evo_kad", C["Alakazam"]: "evo_zam",
               C["Dudunsparce"]: "evo_dud"}
    if o.type == OT.PLAY:
        return play_map.get(cid, "other")
    if o.type == OT.EVOLVE:
        return evo_map.get(cid, "other")
    if o.type == OT.ATTACH:
        return "attach"
    return "other"


def decide(obs, pol):
    """骨格pol で全コンテキストを処理し選択indexリストを返す。不明はNone→fallback。"""
    C = _C
    state = obs.current
    select = obs.select
    ctx = select.context
    my = state.yourIndex
    n = len(select.option)
    if n < 1:
        return None
    minc, maxc = select.minCount, select.maxCount
    f = facts(state, my)
    SC = C["SelectContext"]

    # 強制（選ぶ余地なし）
    if n < 2 or minc >= n:
        return list(range(max(minc, min(1, maxc))))

    # --- MAIN ---
    if ctx == SC.MAIN and minc == 1 and maxc == 1:
        cls = [main_class(state, select, i, my) for i in range(n)]
        best = max(range(n), key=lambda i: (pol.main_priority(cls[i], f),
                                            _tiebreak(state, select, i, my, f)))
        return [best]

    # --- DISCARD（手札からk枚捨てる。残す価値の下位k枚）---
    if ctx == SC.DISCARD:
        vals = []
        for i in range(n):
            cid = _opt_hand_id(state, select, i, my)
            vals.append((pol.discard_keep(cid, f) if cid is not None else 0, i))
        vals.sort()  # 昇順=残す価値が低い順
        k = minc
        return sorted(i for _v, i in vals[:k])

    # --- サーチ（TO_HAND: 山/lookingから取る。任意枚数ならmaxまで高価値）---
    if ctx == SC.TO_HAND:
        vals = []
        for i in range(n):
            cid = _opt_search_id(state, select, i, my)
            vals.append((pol.search_pref(cid, f) if cid is not None else 0, i))
        vals.sort(reverse=True)
        k = minc if minc >= 1 else min(maxc, sum(1 for v, _ in vals if v >= 55) or 1)
        k = max(minc, min(k, maxc))
        return sorted(i for _v, i in vals[:k])

    # --- セットアップのアクティブ選択（S1: Dunsparce/Abra先頭、Shaymin/Fez回避）---
    if ctx == SC.SETUP_ACTIVE_POKEMON and maxc == 1:
        me = state.players[my]

        def setup_active_val(i):
            cid = _opt_hand_id(state, select, i, my)
            return {C["Dunsparce"]: 90, C["Abra"]: 85, C["Kadabra"]: 70,
                    C["Shaymin"]: 20, C["Fezandipiti_ex"]: 10}.get(cid, 40)
        return [max(range(n), key=setup_active_val)]

    # --- 昇格（SWITCH/TO_ACTIVE/TO_BENCH: 自ポケ選択）---
    if ctx in (SC.SWITCH, SC.TO_ACTIVE, SC.TO_BENCH) and maxc == 1:
        best = max(range(n), key=lambda i: _pol_switch(pol, state, select, i, my, f))
        return [best]

    # --- 能力YesNo（ACTIVATE）---
    if ctx == SC.ACTIVATE:
        # option[0]=Yes 相当。activate_yes で決める
        aid = getattr(select.contextCard, "id", None)
        yes = pol.activate_yes(f, aid)
        # YesNo は通常 option 2つ（YES/NO）。YESなら該当index
        return [0] if yes else ([1] if n >= 2 else [0])

    # --- 先攻後攻（IS_FIRST: Majkel実測で決定、全案共通）---
    if ctx == SC.IS_FIRST:
        return [0]  # 後述の集計で確定（暫定: 先攻）

    # --- 枚数（DRAW_COUNT: Majkel実測=最小(1相当)。全案共通）---
    if ctx == SC.DRAW_COUNT:
        return [0]

    # それ以外（ダメカン配置等の低頻度）→ fallback
    return None


def _tiebreak(state, select, i, my, f):
    """MAIN同クラス内: 攻撃準備済みへの装填/ライン前進を微加点。"""
    C = _C
    o = select.option[i]
    AT = C["AreaType"]
    if o.type == C["OptionType"].ATTACH and o.inPlayArea == AT.ACTIVE:
        return 3
    return 0


def _opt_hand_id(state, select, i, my):
    C = _C
    me = state.players[my]
    o = select.option[i]
    try:
        return me.hand[o.index].id
    except (IndexError, TypeError, AttributeError):
        return None


def _opt_search_id(state, select, i, my):
    C = _C
    me = state.players[my]
    o = select.option[i]
    AT = C["AreaType"]
    try:
        if o.area == AT.DISCARD:
            return me.discard[o.index].id
        if select.deck:
            return select.deck[o.index].id
        return me.hand[o.index].id
    except (IndexError, TypeError, AttributeError):
        return None


def _pol_switch(pol, state, select, i, my, f):
    C = _C
    me = state.players[my]
    o = select.option[i]
    AT = C["AreaType"]
    try:
        p = (me.active[o.index] if o.area == AT.ACTIVE else me.bench[o.index])
    except (IndexError, TypeError, AttributeError):
        p = None
    return pol.switch_pref(p, f) if p is not None else -1
