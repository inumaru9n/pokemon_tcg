"""EXP-033: 特徴量拡充版の特徴抽出（収集・学習・推論で共有）。

EXP-031/032の診断「ボトルネックは1判断の表現力」を受けた拡充:
- コンテキスト側: 両者の盤面各スロット（アクティブ+ベンチ5）の
  [HP割合, エネルギー数, ex/megaEx価値, ダメージ有無] を明示的に並べる（48次元）。
  （山札残・サイド・ターンは既存28次元に含まれる）
- オプション側: 選択肢が参照するポケモンの詳細
  [HP割合, エネ数, 闘エネ数, ツール有無, ex価値, ダメージ有無] と、
  攻撃オプションのコスト充足度・ダメージ量（8次元）。
全て防御的に実装（属性欠落・型不一致は0埋め、例外で落とさない）。
"""

from __future__ import annotations

from cg.api import (
    AreaType,
    EnergyType,
    Observation,
    OptionType,
    all_attack,
    all_card_data,
)

N_CARD = 1300      # カードID埋め込みテーブルサイズ（実IDは1..1267）
N_ATTACK = 1400    # 攻撃ID埋め込みテーブルサイズ
N_OPT_TYPE = 16
N_CONTEXT = 50
N_OPT_SCALAR = 22  # 14(EXP-031) + 8(参照ポケモン詳細+攻撃情報)
N_CTX_SCALAR = 76  # 28(EXP-031) + 48(盤面スロット 2P×6スロット×4値)

try:
    _CARD_TABLE = {c.cardId: c for c in all_card_data()}
except Exception:
    _CARD_TABLE = {}
try:
    _ATTACK_TABLE = {a.attackId: a for a in all_attack()}
except Exception:
    _ATTACK_TABLE = {}


def _card_at(obs: Observation, area, index: int, player_index: int):
    ps = obs.current.players[player_index]
    try:
        if area == AreaType.DECK:
            return obs.select.deck[index]
        if area == AreaType.HAND:
            return ps.hand[index] if ps.hand else None
        if area == AreaType.DISCARD:
            return ps.discard[index]
        if area == AreaType.ACTIVE:
            return ps.active[index]
        if area == AreaType.BENCH:
            return ps.bench[index]
        if area == AreaType.PRIZE:
            return ps.prize[index]
        if area == AreaType.STADIUM:
            return obs.current.stadium[index]
        if area == AreaType.LOOKING:
            return obs.current.looking[index]
    except (IndexError, TypeError):
        return None
    return None


def _ex_value(card_id) -> float:
    """きぜつ時のサイド価値を連想させるスカラー: 通常0 / ex0.5 / megaEx1.0"""
    cd = _CARD_TABLE.get(int(card_id or 0))
    if cd is None:
        return 0.0
    if getattr(cd, "megaEx", False):
        return 1.0
    if getattr(cd, "ex", False):
        return 0.5
    return 0.0


def _mon_scalars(mon) -> list[float]:
    """ポケモン1体 → [HP割合, エネ数/4, 闘エネ数/4, ツール有無, ex価値, ダメージ有無]"""
    if mon is None or not hasattr(mon, "energies"):
        return [0.0] * 6
    try:
        hp = float(getattr(mon, "hp", 0) or 0)
        max_hp = float(getattr(mon, "maxHp", 0) or 0)
        ratio = hp / max_hp if max_hp > 0 else 0.0
        energies = getattr(mon, "energies", None) or []
        n_fight = sum(
            1 for e in energies
            if e in (EnergyType.FIGHTING, EnergyType.RAINBOW)
        )
        tools = getattr(mon, "tools", None) or []
        return [
            ratio,
            len(energies) / 4.0,
            n_fight / 4.0,
            1.0 if tools else 0.0,
            _ex_value(getattr(mon, "id", 0)),
            1.0 if (max_hp > 0 and hp < max_hp) else 0.0,
        ]
    except Exception:
        return [0.0] * 6


def _attack_scalars(option, obs: Observation, my_index: int) -> list[float]:
    """攻撃オプション → [コスト充足度, ダメージ/300]"""
    try:
        atk = _ATTACK_TABLE.get(int(getattr(option, "attackId", 0) or 0))
        if atk is None:
            return [0.0, 0.0]
        me = obs.current.players[my_index]
        mon = me.active[0] if me.active else None
        have = list(getattr(mon, "energies", None) or []) if mon is not None else []
        req = list(getattr(atk, "energies", None) or [])
        matched = 0
        for t in req:
            if t == EnergyType.COLORLESS:
                continue
            for i, h in enumerate(have):
                if h == t or h == EnergyType.RAINBOW:
                    have.pop(i)
                    matched += 1
                    break
        n_colorless = sum(1 for t in req if t == EnergyType.COLORLESS)
        matched += min(n_colorless, len(have))
        sat = matched / len(req) if req else 1.0
        return [min(sat, 1.0), float(getattr(atk, "damage", 0) or 0) / 300.0]
    except Exception:
        return [0.0, 0.0]


def option_token(obs: Observation, option) -> tuple[int, int, int, int, list[float]]:
    """1選択肢 → (opt_type, card_id, target_id, attack_id, スカラー22)"""
    my_index = obs.current.yourIndex
    opt_type = int(option.type)
    card_id = 0
    target_id = 0
    attack_id = 0

    card = None
    target = None
    if option.type == OptionType.CARD:
        card = _card_at(obs, option.area, option.index, option.playerIndex)
    elif option.type in (OptionType.PLAY, OptionType.ATTACH, OptionType.EVOLVE):
        card = _card_at(obs, AreaType.HAND, option.index, my_index)
        in_area = getattr(option, "inPlayArea", None)
        if in_area is not None and option.type != OptionType.PLAY:
            target = _card_at(obs, in_area, getattr(option, "inPlayIndex", 0), my_index)
    elif option.type == OptionType.ABILITY:
        card = _card_at(obs, option.area, option.index, my_index)
    elif option.type == OptionType.ATTACK:
        attack_id = int(getattr(option, "attackId", 0) or 0)

    if card is not None:
        card_id = int(getattr(card, "id", 0) or 0)
    if target is not None:
        target_id = int(getattr(target, "id", 0) or 0)

    def poke_scalar(p, attr, div):
        v = getattr(p, attr, None) if p is not None else None
        if v is None:
            return 0.0
        if isinstance(v, list):
            return len(v) / div
        return float(v) / div

    # 選択肢が参照するポケモン: 攻撃なら自アクティブ、次点で対象、最後に選択カード自身
    ref = None
    try:
        if option.type == OptionType.ATTACK:
            me = obs.current.players[my_index]
            ref = me.active[0] if me.active else None
        elif target is not None and hasattr(target, "energies"):
            ref = target
        elif card is not None and hasattr(card, "energies"):
            ref = card
        elif getattr(option, "inPlayArea", None) is not None:
            pidx = option.playerIndex if option.playerIndex is not None else my_index
            cand = _card_at(obs, option.inPlayArea,
                            getattr(option, "inPlayIndex", 0) or 0, pidx)
            if cand is not None and hasattr(cand, "energies"):
                ref = cand
    except Exception:
        ref = None

    scalars = [
        float(getattr(option, "number", 0) or 0) / 10.0,
        1.0 if getattr(option, "playerIndex", my_index) == my_index else 0.0,
        float(int(getattr(option, "area", 0) or 0)) / 10.0,
        float(int(getattr(option, "inPlayArea", 0) or 0)) / 10.0,
        float(getattr(option, "inPlayIndex", 0) or 0) / 5.0,
        float(getattr(option, "index", 0) or 0) / 10.0,
        poke_scalar(card, "hp", 300.0),
        poke_scalar(card, "energies", 4.0),
        poke_scalar(card, "tools", 2.0),
        poke_scalar(target, "hp", 300.0),
        poke_scalar(target, "energies", 4.0),
        1.0 if option.type == OptionType.YES else 0.0,
        1.0 if option.type == OptionType.NO else 0.0,
        1.0,
    ]
    scalars += _mon_scalars(ref)
    if option.type == OptionType.ATTACK:
        scalars += _attack_scalars(option, obs, my_index)
    else:
        scalars += [0.0, 0.0]
    return opt_type, card_id, target_id, attack_id, scalars


def _board_slots(ps) -> list[float]:
    """1プレイヤーの盤面 → 6スロット×[HP割合, エネ数/4, ex価値, ダメージ有無]=24次元"""
    out: list[float] = []
    slots = list(ps.active or []) + list(ps.bench or [])
    for i in range(6):
        mon = slots[i] if i < len(slots) else None
        if mon is None or not hasattr(mon, "energies"):
            out += [0.0, 0.0, 0.0, 0.0]
            continue
        try:
            hp = float(getattr(mon, "hp", 0) or 0)
            max_hp = float(getattr(mon, "maxHp", 0) or 0)
            out += [
                hp / max_hp if max_hp > 0 else 0.0,
                len(getattr(mon, "energies", None) or []) / 4.0,
                _ex_value(getattr(mon, "id", 0)),
                1.0 if (max_hp > 0 and hp < max_hp) else 0.0,
            ]
        except Exception:
            out += [0.0, 0.0, 0.0, 0.0]
    return out


def context_features(obs: Observation) -> tuple[int, int, int, list[float]]:
    """盤面 → (ctx_id, my_active_id, op_active_id, スカラー76)"""
    state = obs.current
    my_index = state.yourIndex
    me = state.players[my_index]
    op = state.players[1 - my_index]

    def board_stats(ps):
        count = hp = en = ready = 0.0
        for p in ps.active + ps.bench:
            if p is None:
                continue
            count += 1
            hp += p.hp
            en += len(p.energies)
            if len(p.energies) >= 2:
                ready += 1
        return count, hp, en, ready

    m_cnt, m_hp, m_en, m_rdy = board_stats(me)
    o_cnt, o_hp, o_en, o_rdy = board_stats(op)
    m_active = me.active[0] if me.active else None
    o_active = op.active[0] if op.active else None

    scalars = [
        len(me.prize) / 6.0, len(op.prize) / 6.0,
        (len(op.prize) - len(me.prize)) / 6.0,
        m_cnt / 6.0, m_hp / 600.0, m_en / 8.0, m_rdy / 4.0,
        o_cnt / 6.0, o_hp / 600.0, o_en / 8.0, o_rdy / 4.0,
        (m_active.hp / 300.0) if m_active is not None else 0.0,
        (len(m_active.energies) / 4.0) if m_active is not None else 0.0,
        (o_active.hp / 300.0) if o_active is not None else 0.0,
        (len(o_active.energies) / 4.0) if o_active is not None else 0.0,
        me.handCount / 10.0, op.handCount / 10.0,
        me.deckCount / 60.0, op.deckCount / 60.0,
        min(state.turn, 30) / 30.0,
        1.0 if state.energyAttached else 0.0,
        1.0 if state.supporterPlayed else 0.0,
        obs.select.minCount / 5.0,
        obs.select.maxCount / 5.0,
        1.0 if (me.asleep or me.paralyzed or me.confused) else 0.0,
        1.0 if (op.asleep or op.paralyzed or op.confused) else 0.0,
        1.0 if state.firstPlayer == my_index else 0.0,
        1.0,
    ]
    try:
        scalars += _board_slots(me)
    except Exception:
        scalars += [0.0] * 24
    try:
        scalars += _board_slots(op)
    except Exception:
        scalars += [0.0] * 24
    my_active_id = int(m_active.id) if m_active is not None else 0
    op_active_id = int(o_active.id) if o_active is not None else 0
    return int(obs.select.context), my_active_id, op_active_id, scalars


def extract(obs: Observation):
    """1 select → (ctx_tuple, [option_token, ...])"""
    ctx = context_features(obs)
    opts = [option_token(obs, o) for o in obs.select.option]
    return ctx, opts
