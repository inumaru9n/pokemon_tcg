"""EXP-015: カード埋め込みBC方策の特徴抽出（収集・学習・推論で共有）。

設計方針（Discussion 713608の教訓）: 「カードの同一性」を特徴に入れることが学習の鍵。
各選択肢を (card_id, target_id, attack_id, スカラー) のトークンにし、
盤面コンテキスト（集計スカラー+SelectContext+両アクティブのカードID）と合わせてスコアリングする。
"""

from __future__ import annotations

from cg.api import AreaType, Observation, OptionType

N_CARD = 1300      # カードID埋め込みテーブルサイズ（実IDは1..1267）
N_ATTACK = 1400    # 攻撃ID埋め込みテーブルサイズ
N_OPT_TYPE = 16
N_CONTEXT = 50
N_OPT_SCALAR = 14
N_CTX_SCALAR = 28


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


def option_token(obs: Observation, option) -> tuple[int, int, int, int, list[float]]:
    """1選択肢 → (opt_type, card_id, target_id, attack_id, スカラー14)"""
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
    return opt_type, card_id, target_id, attack_id, scalars


def context_features(obs: Observation) -> tuple[int, int, int, list[float]]:
    """盤面 → (ctx_id, my_active_id, op_active_id, スカラー28)"""
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
    my_active_id = int(m_active.id) if m_active is not None else 0
    op_active_id = int(o_active.id) if o_active is not None else 0
    return int(obs.select.context), my_active_id, op_active_id, scalars


def extract(obs: Observation):
    """1 select → (ctx_tuple, [option_token, ...])"""
    ctx = context_features(obs)
    opts = [option_token(obs, o) for o in obs.select.option]
    return ctx, opts
