from __future__ import annotations

import os
from collections import defaultdict

from cg.api import (AreaType, CardType, EnergyType, Observation, SelectContext,
                    OptionType, Card, Pokemon, all_card_data, to_observation_class)

"""
Cynthia's Garchomp ex deck (07-06 ladder recipe, sig fc15d0c926, 896 games 56.4%).
Built as candidate + pool opponent (new 9.1% archetype our Lucario can't hit for weakness).

Engine: evolve Gible->Gabite (Champion's Call searches Cynthia's Pokemon)->Garchomp ex.
Roserade's Cheer On to Glory gives Cynthia's attacks +30 to the active. Garchomp ex swings
Draconic Buster 260 (+30 = 290) for 2 Fighting (discards all energy after -> recharge), or
Corkscrew Dive 100 (+30) for 1 Fighting with a draw. Power Weight tool = +70 HP (330->400).
Policy modeled on the islet AttackPlan + option-scoring skeleton.
"""

file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deck.csv") \
    if "__file__" in globals() else "deck.csv"
if not os.path.exists(file_path):
    file_path = "/kaggle_simulations/agent/deck.csv"
with open(file_path, "r", encoding="utf-8") as f:
    my_deck = [int(line) for line in f.read().splitlines() if line.strip()]

all_card = all_card_data()
card_table = {c.cardId: c for c in all_card}

Fighting_Energy = 6
Rock_Fighting_Energy = 20
Roselia = 341
Roserade = 342
Gible = 379
Gabite = 380
Garchomp = 381
Spiritomb = 387
Unfair_Stamp = 1080
Buddy_Poffin = 1086
Night_Stretcher = 1097
Fighting_Gong = 1142
Poke_Pad = 1152
Power_Weight = 1173
Boss_Orders = 1182
Xerosic = 1197
Surfer = 1203
Hilda = 1225
Lillie_Determination = 1227
Forest_Vitality = 1261

DRACONIC_BUSTER = 532   # 260 [2 fighting], discard all energy after
CORKSCREW_DIVE = 531    # 100 [1 fighting] + draw to 6
DRAGONSLICE = 530       # Gabite 40 [1 fighting]
LEAF_STEP = 476         # Roserade 80

FIGHTING = (EnergyType.FIGHTING, EnergyType.COLORLESS)


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
    if pokemon.id == 112 and len(pokemon.energies) >= 1:  # opposing Munkidori
        score += 300
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

    roserade_boost = 30 if field_counts[Roserade] >= 1 else 0  # Cheer On to Glory
    hand_fighting = hand_counts[Fighting_Energy] + hand_counts[Rock_Fighting_Energy]

    if context == SelectContext.MAIN:
        can_op_switch = any(o.type == OptionType.PLAY and
                            get_card(obs, AreaType.HAND, o.index, my_index).id == Boss_Orders
                            for o in select.option)
        my_cards = [me.active[0]] + list(me.bench)
        op_cards = [op.active[0]] + list(op.bench)
        if state.turn >= 2:
            best = -1
            for i, mine in enumerate(my_cards):
                if mine is None or i != 0:  # active-only attacker (no free switch)
                    continue
                cands = []
                if mine.id == Garchomp:
                    cands.append((DRACONIC_BUSTER, 2, 260, 60))
                    cands.append((CORKSCREW_DIVE, 1, 100, 10))
                elif mine.id == Gabite:
                    cands.append((DRAGONSLICE, 1, 40, -40))
                elif mine.id == Roserade:
                    cands.append((LEAF_STEP, 1, 80, -30))
                else:
                    continue
                for attack_id, need, base_damage, base_score in cands:
                    ec = len(mine.energies)
                    more = False
                    if ec < need:
                        if hand_fighting >= 1 and not state.energyAttached:
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
                        # Roserade +30 only to the active target; Leaf Step is Grass
                        atk_type = EnergyType.GRASS if attack_id == LEAF_STEP else EnergyType.FIGHTING
                        if j == 0:
                            damage += roserade_boost
                        data = card_table[opp.id]
                        if data.weakness == atk_type:
                            damage *= 2
                        elif data.resistance == atk_type:
                            damage -= 30
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
        if pokemon.id == Garchomp:
            score += 5
            if ec < 2:
                score += 120
        elif pokemon.id in (Gabite, Gible):
            score += 2
            if ec < 1:
                score += 40
        elif pokemon.id in (Roserade, Roselia):
            if ec < 1:
                score += 20
        else:
            score -= 60
        return score

    stadium_id = state.stadium[0].id if state.stadium else 0
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
                        if card.id == Garchomp:
                            score += 25
                        elif card.id == Gabite:
                            score += 10
                        elif card.id == Roserade:
                            score += 6
                        elif card.id == Gible:
                            score += 4
                    else:
                        if o.index == plan.target - 1:
                            score += 100
                elif context == SelectContext.SETUP_ACTIVE_POKEMON:
                    score = 4 if card.id == Gible else 3 if card.id == Roselia else 2 if card.id == Spiritomb else 1
                elif context == SelectContext.SETUP_BENCH_POKEMON:
                    score = 4 if card.id == Gible else 3 if card.id == Roselia else 1
                elif context == SelectContext.TO_HAND:
                    score = 200 - hand_counts[card.id] * 100
                    if card.id == Garchomp:
                        score += 60 if field_counts[Gabite] >= 1 else 20
                    elif card.id == Gabite:
                        score += 40 if field_counts[Gible] >= 1 else 10
                    elif card.id == Gible:
                        need = 2 - (field_counts[Gible] + field_counts[Gabite] + field_counts[Garchomp])
                        score += 40 * max(0, need)
                    elif card.id == Roselia:
                        score += 30 if field_counts[Roselia] + field_counts[Roserade] == 0 else -30
                    elif card.id in (Fighting_Energy, Rock_Fighting_Energy):
                        score += 25 if not state.energyAttached else -1
                    elif card.id == Boss_Orders:
                        score += 15
                elif context == SelectContext.ATTACH_FROM:
                    score = energy_score(card, o.area == AreaType.ACTIVE)
                elif context in (SelectContext.DAMAGE, SelectContext.DAMAGE_COUNTER):
                    if o.playerIndex != my_index:
                        score = 5000 + pokemon_score(card) if card.hp <= 30 else pokemon_score(card)
                    else:
                        score = -100
                elif context == SelectContext.EVOLVES_TO:
                    score = 100 if card.id == Garchomp else 60 if card.id == Gabite else 40 if card.id == Roserade else 10
        elif o.type == OptionType.PLAY:
            card = get_card(obs, AreaType.HAND, o.index, my_index)
            data = card_table[card.id]
            if data.cardType == CardType.POKEMON:
                score = 20000
                if card.id == Gible and field_counts[Gible] + field_counts[Gabite] + field_counts[Garchomp] >= 3:
                    score = -1
                elif card.id == Roselia and field_counts[Roselia] + field_counts[Roserade] >= 1:
                    score = -1
            else:
                score = 10000
                if card.id == Boss_Orders:
                    score = 3200 if plan.target >= 1 else -1
                elif card.id == Lillie_Determination:
                    score = 3100
                elif card.id == Hilda:
                    score = 3000
                elif card.id == Surfer:
                    score = 2900
                elif card.id == Xerosic:
                    score = 2800
                elif card.id == Buddy_Poffin:
                    score = 9000
                elif card.id == Poke_Pad:
                    score = 8000
                elif card.id == Night_Stretcher:
                    score = 2400
                elif card.id == Forest_Vitality:
                    score = -1 if stadium_id == Forest_Vitality else 5000
        elif o.type == OptionType.ATTACH:
            card = get_card(obs, AreaType.HAND, o.index, my_index)
            pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
            if card.id == Power_Weight:
                score = 7000 + (200 if pokemon.id == Garchomp else 0)
            else:
                score = energy_score(pokemon, o.inPlayArea == AreaType.ACTIVE)
                if plan.energy and ((o.inPlayArea == AreaType.ACTIVE and plan.attacker == 0)
                                    or plan.attacker == 1 + o.inPlayIndex):
                    score += 200
        elif o.type == OptionType.EVOLVE:
            pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
            score = 9000 + len(pokemon.energies)
        elif o.type == OptionType.ABILITY:
            score = 30000  # Champion's Call (search) always worth it
        elif o.type == OptionType.RETREAT:
            score = -1
        elif o.type == OptionType.ATTACK:
            score = 1000
            if o.attackId == plan.attack_id:
                score += 100
            elif o.attackId == DRACONIC_BUSTER:
                score += 60
        scores.append(score)

    desc = [i for i, _ in sorted(enumerate(scores), key=lambda x: x[1], reverse=True)]
    return desc[: select.maxCount]
