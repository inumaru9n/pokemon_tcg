from __future__ import annotations

import os
from collections import defaultdict

from cg.api import (AreaType, CardType, EnergyType, Observation, SelectContext,
                    OptionType, Card, Pokemon, all_card_data, to_observation_class)

"""
Archaludon ex metal-tempo deck (ladder recipe from our own submission replays; sig of the
most common Archaludon opponent at rating ~895). Built as a POOL OPPONENT so we can iterate
007 against the matchup that beats us in production (35% win rate, 31% of opponents).

Engine: Cinderace (Explosiveness starter, Turbo Flare ramps 3 basic energy to bench) ->
evolve Duraludon -> Archaludon ex (HP300, Metal Defender 220 for 3 metal). Full Metal Lab
gives metal Pokemon -30 damage taken; Jumbo Ice Cream heals 80 from a 3+-energy active.
The healing + damage reduction is the key behavior that grinds Lucario down.
"""

file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deck.csv") \
    if "__file__" in globals() else "deck.csv"
if not os.path.exists(file_path):
    file_path = "/kaggle_simulations/agent/deck.csv"
with open(file_path, "r", encoding="utf-8") as f:
    my_deck = [int(line) for line in f.read().splitlines() if line.strip()]

all_card = all_card_data()
card_table = {c.cardId: c for c in all_card}

Metal_Energy = 8
Relicanth = 57
Duraludon = 169
Archaludon = 190
Cinderace = 666
Night_Stretcher = 1097
Ultra_Ball = 1121
Pokegear = 1122
Jumbo_Ice_Cream = 1147
Poke_Pad = 1152
Heros_Cape = 1159
Boss_Orders = 1182
Explorers_Guidance = 1185
Lillie_Determination = 1227
Full_Metal_Lab = 1244

METAL_DEFENDER = 253
TURBO_FLARE = 965


class AttackPlan:
    attacker = -1
    target = -1
    attack_id = -1
    remain_hp = -1
    energy = False


plan = AttackPlan()
pre_turn = 0


def get_card(obs, area, index, player_index):
    ps = obs.current.players[player_index]
    match area:
        case AreaType.DECK:
            return obs.select.deck[index]
        case AreaType.HAND:
            return ps.hand[index]
        case AreaType.DISCARD:
            return ps.discard[index]
        case AreaType.ACTIVE:
            return ps.active[index]
        case AreaType.BENCH:
            return ps.bench[index]
        case AreaType.PRIZE:
            return ps.prize[index]
        case AreaType.STADIUM:
            return obs.current.stadium[index]
        case AreaType.LOOKING:
            return obs.current.looking[index]
        case _:
            return None


def prize_count(pokemon):
    data = card_table[pokemon.id]
    count = 3 if data.megaEx else 2 if data.ex else 1
    return max(0, count)


def pokemon_score(pokemon):
    data = card_table[pokemon.id]
    score = prize_count(pokemon) * 1000
    score += len(pokemon.energies) * 150
    if data.stage2:
        score += 250
    elif data.stage1:
        score += 130
    score += pokemon.hp
    return score


def agent(obs_dict):
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        return my_deck

    state = obs.current
    select = obs.select
    context = select.context
    my_index = state.yourIndex
    me = state.players[my_index]
    op = state.players[1 - my_index]

    global plan, pre_turn
    if pre_turn != state.turn:
        pre_turn = state.turn
        plan = AttackPlan()

    field_counts = defaultdict(int)
    hand_counts = defaultdict(int)
    for c in me.active + me.bench:
        if c is not None:
            field_counts[c.id] += 1
    for c in me.hand:
        hand_counts[c.id] += 1

    stadium_id = state.stadium[0].id if state.stadium else 0
    active = me.active[0] if me.active else None

    # Should we heal the active Archaludon? (damaged and has 3+ energy)
    want_heal = (active is not None and card_table[active.id].hp - active.hp >= 60
                 and len(active.energies) >= 3)

    if context == SelectContext.MAIN:
        can_op_switch = any(o.type == OptionType.PLAY and
                            get_card(obs, AreaType.HAND, o.index, my_index).id == Boss_Orders
                            for o in select.option)
        my_cards = [me.active[0]] + list(me.bench)
        op_cards = [op.active[0]] + list(op.bench)
        if state.turn >= 2:
            best = -1
            for i, mine in enumerate(my_cards):
                if mine is None or i != 0:  # only the active attacks (no free switch engine)
                    continue
                cands = []
                if mine.id == Archaludon:
                    cands.append((METAL_DEFENDER, 3, 220, 60))
                elif mine.id == Cinderace:
                    cands.append((TURBO_FLARE, 0, 50, -40))  # ramp, not really attacking
                else:
                    continue
                for attack_id, need, base_damage, base_score in cands:
                    ec = len(mine.energies)
                    more = False
                    if ec < need:
                        if hand_counts[Metal_Energy] >= 1 and not state.energyAttached:
                            ec += 1
                            if ec < need:
                                continue
                            more = True
                        else:
                            continue
                    for j, opp in enumerate(op_cards):
                        if opp is None:
                            continue
                        if j != 0 and not can_op_switch:
                            break
                        damage = base_damage
                        prize = 0
                        score = pokemon_score(opp)
                        if opp.hp <= damage:
                            prize = prize_count(opp)
                        else:
                            score *= damage / opp.hp
                        score += base_score
                        if len(op.prize) <= prize:
                            score = 50000
                        if j == 0:
                            score += 300
                        if best < score:
                            best = score
                            plan.attacker = i
                            plan.target = j
                            plan.attack_id = attack_id
                            plan.remain_hp = opp.hp - damage
                            plan.energy = more

    def energy_score(pokemon, is_active):
        ec = len(pokemon.energies)
        score = 8000 + (10 if is_active else 0)
        if pokemon.id == Archaludon:
            score += 5
            if ec < 3:
                score += 120
        elif pokemon.id == Duraludon:
            score += 3
            if ec < 1:
                score += 40
        else:
            score -= 60
        return score

    scores = []
    for o in select.option:
        score = 0
        if o.type == OptionType.NUMBER:
            score = o.number
        elif o.type == OptionType.YES:
            score = 1
        elif o.type == OptionType.CARD:
            card = get_card(obs, o.area, o.index, o.playerIndex)
            if card is not None:
                ec = len(card.energies) if isinstance(card, Pokemon) else 0
                if context in (SelectContext.SWITCH, SelectContext.TO_ACTIVE):
                    if o.playerIndex == my_index:
                        score += ec * 2
                        if card.id == Archaludon:
                            score += 25
                        elif card.id == Cinderace:
                            score += 12
                        elif card.id == Duraludon:
                            score += 6
                    else:
                        if o.index == plan.target - 1:
                            score += 100
                elif context == SelectContext.SETUP_ACTIVE_POKEMON:
                    score = 4 if card.id == Cinderace else 2 if card.id == Duraludon else 1
                elif context == SelectContext.SETUP_BENCH_POKEMON:
                    score = 4 if card.id == Duraludon else 3 if card.id == Cinderace else 1
                elif context == SelectContext.TO_HAND:
                    score = 200 - hand_counts[card.id] * 100
                    if card.id == Archaludon:
                        score += 60 if field_counts[Duraludon] >= 1 else 20
                    elif card.id == Duraludon:
                        need = 2 - (field_counts[Duraludon] + field_counts[Archaludon])
                        score += 40 * max(0, need)
                    elif card.id == Cinderace:
                        score += -50 if field_counts[Cinderace] >= 1 else 30
                    elif card.id == Metal_Energy:
                        score += 25 if not state.energyAttached else -1
                    elif card.id == Jumbo_Ice_Cream:
                        score += 40 if want_heal else 5
                    elif card.id == Boss_Orders:
                        score += 15
                elif context == SelectContext.ATTACH_FROM:
                    score = energy_score(card, o.area == AreaType.ACTIVE)
                elif context == SelectContext.HEAL or context == SelectContext.REMOVE_DAMAGE_COUNTER:
                    score = (card_table[card.id].hp - card.hp) * 10 if o.playerIndex == my_index else -100
                elif context == SelectContext.EVOLVES_TO:
                    score = 100 if card.id == Archaludon else 10
        elif o.type == OptionType.PLAY:
            card = get_card(obs, AreaType.HAND, o.index, my_index)
            data = card_table[card.id]
            if data.cardType == CardType.POKEMON:
                score = 20000
                if card.id == Cinderace and field_counts[Cinderace] >= 1:
                    score = -1
                elif card.id == Duraludon and field_counts[Duraludon] + field_counts[Archaludon] >= 2:
                    score = -1
            else:
                score = 10000
                if card.id == Jumbo_Ice_Cream:
                    score = 12000 if want_heal else -1
                elif card.id == Full_Metal_Lab:
                    score = -1 if stadium_id == Full_Metal_Lab else 6000
                elif card.id == Boss_Orders:
                    score = 3200 if plan.target >= 1 else -1
                elif card.id == Lillie_Determination:
                    score = 3100
                elif card.id == Explorers_Guidance:
                    score = 3000
                elif card.id == Ultra_Ball:
                    score = 8500
                elif card.id == Pokegear:
                    score = 8000
                elif card.id == Poke_Pad:
                    score = 7500
                elif card.id == Night_Stretcher:
                    score = 2400
        elif o.type == OptionType.ATTACH:
            card = get_card(obs, AreaType.HAND, o.index, my_index)
            pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
            if card.id == Heros_Cape:
                score = 7000 + (200 if pokemon.id == Archaludon else 0)
            else:
                score = energy_score(pokemon, o.inPlayArea == AreaType.ACTIVE)
                if plan.energy and ((o.inPlayArea == AreaType.ACTIVE and plan.attacker == 0)
                                    or plan.attacker == 1 + o.inPlayIndex):
                    score += 200
        elif o.type == OptionType.EVOLVE:
            pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
            score = 9000 + len(pokemon.energies)
        elif o.type == OptionType.ABILITY:
            score = 30000
        elif o.type == OptionType.RETREAT:
            score = -1
        elif o.type == OptionType.ATTACK:
            score = 1000
            if o.attackId == plan.attack_id:
                score += 100
            elif o.attackId == METAL_DEFENDER:
                score += 60
        scores.append(score)

    desc = [i for i, _ in sorted(enumerate(scores), key=lambda x: x[1], reverse=True)]
    return desc[: select.maxCount]
