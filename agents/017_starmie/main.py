from __future__ import annotations

import os
from collections import defaultdict

from cg.api import (AreaType, CardType, EnergyType, Observation, SelectContext,
                    OptionType, Card, Pokemon, all_card_data, to_observation_class)

"""
Mega Starmie ex Deck (ladder recipe, sig 491b8bfb26, 509 games 53.8% on 2026-07-03).
Policy modeled on the islet Lucario agent (AttackPlan + option scoring + matchup guards).

Game plan:
- Cinderace is the Explosiveness starter; Turbo Flare (free) ramps 3 basic energy onto the bench.
- Evolve Staryu -> Mega Starmie ex (from hand / Salvatore / after Mega Signal fetch).
- Attack with Nebula Beam 210 (3 any energy, ignores Weakness/Resistance AND effects on the
  opponent's active -> beats walls like Crustle) or Jetting Blow 120 + 50 bench snipe (1 water).
- Crushing Hammer disrupts the opponent's charged attacker; Wally's Compassion heals a dying Starmie.
"""

file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deck.csv") \
    if "__file__" in globals() else "deck.csv"
if not os.path.exists(file_path):
    file_path = "/kaggle_simulations/agent/deck.csv"
with open(file_path, "r", encoding="utf-8") as f:
    my_deck = [int(line) for line in f.read().splitlines() if line.strip()]

all_card = all_card_data()
card_table = {c.cardId: c for c in all_card}

# Decklist
Water_Energy = 3
Ignition_Energy = 17
Cinderace = 666
Staryu = 1030
Mega_Starmie = 1031
Buddy_Poffin = 1086
Night_Stretcher = 1097
Crushing_Hammer = 1120
Ultra_Ball = 1121
Pokegear = 1122
Mega_Signal = 1145
Heros_Cape = 1159
Boss_Orders = 1182
Salvatore = 1189
Harlequin = 1223
Hilda = 1225
Lillie_Determination = 1227
Wallys_Compassion = 1229

JETTING_BLOW = 1487   # 120 + 50 bench snipe, cost 1 water
NEBULA_BEAM = 1488    # 210, ignores weakness/resistance/effects, cost 3 any
TURBO_FLARE = 965     # 50 + ramp 3 basic energy, free
WATER_GUN = 1486      # 20, cost 1 water


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
    for card in pokemon.energyCards:
        if card.id == 12:  # Legacy Energy
            count -= 1
    return max(0, count)


def pokemon_score(pokemon):
    data = card_table[pokemon.id]
    score = prize_count(pokemon) * 1000
    score += len(pokemon.energies) * 150
    score += len(pokemon.tools) * 100
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

    # ready Starmie = evolved Mega Starmie with >=1 energy (can at least Jetting Blow with a water)
    ready_starmie = sum(1 for c in me.active + me.bench
                        if c is not None and c.id == Mega_Starmie and len(c.energies) >= 1)

    def energy_count_any(pokemon):
        return len(pokemon.energies)

    def water_count(pokemon):
        return sum(1 for e in pokemon.energies if e in (EnergyType.WATER, EnergyType.COLORLESS)) \
            + sum(1 for e in pokemon.energies if e not in (EnergyType.WATER, EnergyType.COLORLESS))

    can_attack = False
    if context == SelectContext.MAIN:
        can_switch = any(o.type == OptionType.RETREAT for o in select.option) or \
            any(o.type == OptionType.PLAY and get_card(obs, AreaType.HAND, o.index, my_index).id
                in (Night_Stretcher,) for o in select.option)
        can_op_switch = any(o.type == OptionType.PLAY and
                            get_card(obs, AreaType.HAND, o.index, my_index).id == Boss_Orders
                            for o in select.option)
        for o in select.option:
            if o.type == OptionType.ATTACK:
                can_attack = True

        my_cards = [me.active[0]] + list(me.bench)
        op_cards = [op.active[0]] + list(op.bench)

        if state.turn >= 2:
            best = -1
            for i, mine in enumerate(my_cards):
                if mine is None:
                    continue
                if i != 0 and not can_switch:
                    break
                # candidate attacks for this attacker
                cands = []  # (attack_id, energy_required, base_damage, base_score, energy_type)
                if mine.id == Mega_Starmie:
                    cands.append((NEBULA_BEAM, 3, 210, 40, None))     # ignores weakness/effects
                    cands.append((JETTING_BLOW, 1, 120, 30, EnergyType.WATER))  # +50 snipe
                elif mine.id == Cinderace:
                    cands.append((TURBO_FLARE, 0, 50, -40, None))     # prefer ramp over KO usually
                elif mine.id == Staryu:
                    cands.append((WATER_GUN, 1, 20, -60, EnergyType.WATER))
                for attack_id, need, base_damage, base_score, atk_type in cands:
                    more_energy = False
                    ec = energy_count_any(mine)
                    if ec < need:
                        if hand_counts[Water_Energy] + hand_counts[Ignition_Energy] >= 1 \
                                and not state.energyAttached:
                            ec += 1
                            if ec < need:
                                continue
                            more_energy = True
                        else:
                            continue
                    for j, opp in enumerate(op_cards):
                        if opp is None:
                            continue
                        if j != 0 and not can_op_switch:
                            break
                        damage = base_damage
                        data = card_table[opp.id]
                        if atk_type is not None:  # Nebula Beam ignores weakness
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
                        if i == 0:
                            score += 220
                        if j == 0:
                            score += 300
                        score += ec
                        if best < score:
                            best = score
                            plan.attacker = i
                            plan.target = j
                            plan.attack_id = attack_id
                            plan.remain_hp = opp.hp - damage
                            plan.energy = more_energy

    def energy_score(pokemon, active):
        ec = len(pokemon.energies)
        score = 8000 + (10 if active else 0)
        if pokemon.id == Mega_Starmie:
            score += 5
            if ec < 3:
                score += 120
            if ready_starmie >= 1 and ec >= 1:
                score -= 40
        elif pokemon.id == Staryu:
            score += 3            # pre-load Staryu so it can attack right after evolving
            if ec < 1:
                score += 60
        elif pokemon.id == Cinderace:
            score -= 60           # Cinderace attacks for free; don't waste energy on it
        else:
            score -= 100
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
                        if o.index == plan.attacker - 1:
                            score += 100
                        if card.id == Mega_Starmie:
                            score += 25
                        elif card.id == Cinderace:
                            score += 12
                        elif card.id == Staryu:
                            score += 4
                    else:
                        if o.index == plan.target - 1:
                            score += 100
                elif context == SelectContext.SETUP_ACTIVE_POKEMON:
                    # Cinderace is the Explosiveness starter
                    score = 4 if card.id == Cinderace else 2 if card.id == Staryu else 1
                elif context == SelectContext.SETUP_BENCH_POKEMON:
                    score = 4 if card.id == Staryu else 3 if card.id == Cinderace else 1
                elif context == SelectContext.TO_HAND:
                    score = 200 - hand_counts[card.id] * 100
                    if card.id == Mega_Starmie:
                        score += 60 if field_counts[Staryu] >= 1 else 20
                    elif card.id == Staryu:
                        need = 2 - (field_counts[Staryu] + field_counts[Mega_Starmie])
                        score += 40 * max(0, need)
                    elif card.id == Cinderace:
                        score += -50 if field_counts[Cinderace] >= 1 else 30
                    elif card.id in (Water_Energy, Ignition_Energy):
                        score += 25 if not state.energyAttached else -1
                    elif card.id == Salvatore:
                        score += 30 if field_counts[Staryu] >= 1 and hand_counts[Mega_Starmie] == 0 else 10
                    elif card.id == Mega_Signal:
                        score += 25 if hand_counts[Mega_Starmie] == 0 else -20
                    elif card.id == Boss_Orders:
                        score += 15
                elif context == SelectContext.ATTACH_FROM:
                    score = energy_score(card, o.area == AreaType.ACTIVE)
                elif context in (SelectContext.DAMAGE, SelectContext.DAMAGE_COUNTER):
                    # Jetting Blow bench snipe: hit the most valuable / KO-able bench mon
                    if o.playerIndex != my_index:
                        score = 5000 + pokemon_score(card) if card.hp <= 50 else pokemon_score(card)
                    else:
                        score = -100
                elif context == SelectContext.HEAL or context == SelectContext.REMOVE_DAMAGE_COUNTER:
                    if o.playerIndex == my_index:
                        score = (card_table[card.id].hp - card.hp) * 10
                    else:
                        score = -100
                elif context == SelectContext.EVOLVES_TO:
                    score = 100 if card.id == Mega_Starmie else 10
        elif o.type == OptionType.PLAY:
            card = get_card(obs, AreaType.HAND, o.index, my_index)
            data = card_table[card.id]
            if data.cardType == CardType.POKEMON:
                score = 20000
                if card.id == Cinderace and field_counts[Cinderace] >= 1:
                    score = -1
                elif card.id == Staryu and field_counts[Staryu] + field_counts[Mega_Starmie] >= 2:
                    score = -1
            else:
                score = 10000
                if card.id == Mega_Signal:
                    score = 11000 if hand_counts[Mega_Starmie] == 0 else -1
                elif card.id == Salvatore:
                    score = 6000 if field_counts[Staryu] >= 1 else 2600
                elif card.id == Crushing_Hammer:
                    # try to strip energy off a charged opposing attacker
                    opp_act = op.active[0] if op.active else None
                    score = 4500 if (opp_act is not None and len(opp_act.energies) >= 2) else -1
                elif card.id == Wallys_Compassion:
                    worst = 0
                    for c in me.active + me.bench:
                        if c is not None and card_table[c.id].megaEx:
                            worst = max(worst, card_table[c.id].hp - c.hp)
                    score = 5500 if worst >= 120 else -1
                elif card.id == Boss_Orders:
                    score = 3200 if plan.target >= 1 else -1
                elif card.id == Lillie_Determination:
                    score = 3100
                elif card.id == Salvatore:
                    score = 3000
                elif card.id == Hilda:
                    score = 2950
                elif card.id == Harlequin:
                    score = -1 if me.handCount >= 6 else 2500  # avoid dumping a good hand
                elif card.id == Buddy_Poffin:
                    score = 9000
                elif card.id == Ultra_Ball:
                    score = 8500
                elif card.id == Pokegear:
                    score = 8000
                elif card.id == Night_Stretcher:
                    score = 2400
        elif o.type == OptionType.ATTACH:
            card = get_card(obs, AreaType.HAND, o.index, my_index)
            pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
            if card.id == Heros_Cape:
                score = 7000 + (200 if pokemon.id == Mega_Starmie else 0)
            else:
                score = energy_score(pokemon, o.inPlayArea == AreaType.ACTIVE)
                if o.inPlayArea == AreaType.ACTIVE and plan.attacker == 0 and plan.energy:
                    score += 200
                elif plan.attacker == 1 + o.inPlayIndex and plan.energy:
                    score += 200
        elif o.type == OptionType.EVOLVE:
            pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
            score = 9000 + len(pokemon.energies)
        elif o.type == OptionType.ABILITY:
            score = 30000
        elif o.type == OptionType.RETREAT:
            score = 2000 if plan.attacker >= 1 else -1
        elif o.type == OptionType.ATTACK:
            score = 1000
            if o.attackId == plan.attack_id:
                score += 100
            elif o.attackId == NEBULA_BEAM:
                score += 60
        scores.append(score)

    desc = [i for i, _ in sorted(enumerate(scores), key=lambda x: x[1], reverse=True)]
    return desc[: select.maxCount]
