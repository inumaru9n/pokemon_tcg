import os
from collections import defaultdict

from cg.api import AreaType, CardType, EnergyType, Observation, SelectContext, OptionType, Card, Pokemon, all_card_data, to_observation_class

"""
Crustle wall + Mega Kangaskhan ex Deck (ladder recipe, sig 89d834e4d4)
Simple fixed-priority policy (I-003: test whether the deck carries a simple brain):
- Crustle's Mysterious Rock Inn blanks all damage from opponent's {ex} attackers.
- Mega Kangaskhan ex tanks (HP300), draws 2/turn with Run Errand, attacks for 200 on 3 any energy.
- Keep Crustle active vs ex attackers, Kangaskhan otherwise. Heal with Jumbo Ice Cream.
"""

try:  # Kaggle評価環境はexecロードのため __file__ が無い
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deck.csv")
except NameError:
    file_path = "deck.csv"
if not os.path.exists(file_path):
    file_path = "/kaggle_simulations/agent/deck.csv"
with open(file_path, "r") as file:
    csv = file.read().split("\n")
my_deck = []
for i in range(60):
    my_deck.append(int(csv[i]))

all_card = all_card_data()
card_table = {c.cardId: c for c in all_card}

# Decklist
Grass_Energy = 1        # x1
Mist_Energy = 11        # x4
Spiky_Energy = 14       # x4
Grow_Grass_Energy = 18  # x4
Dwebble = 344           # x3
Crustle = 345           # x3
Kangaskhan = 756        # x4 (Mega Kangaskhan ex)
Buddy_Poffin = 1086     # x3
Hand_Trimmer = 1087     # x1
Ultra_Ball = 1121       # x1
Pokegear = 1122         # x3
Switch = 1123           # x2
Jumbo_Ice_Cream = 1147  # x4
Heros_Cape = 1159       # x1
Handheld_Fan = 1161     # x1
Boss_Orders = 1182      # x4
Eri = 1186              # x2
Xerosic = 1197          # x1
Lisia = 1204            # x1
Petrel = 1219           # x4
Hilda = 1225            # x3
Lillie_Determination = 1227  # x4
Community_Center = 1242  # x1
TR_Factory = 1257       # x1

ASCENSION = 478
SUPERB_SCISSORS = 479
RAPID_FIRE = 1092

ENERGY_IDS = (Grass_Energy, Mist_Energy, Spiky_Energy, Grow_Grass_Energy)


class AttackPlan:
    attacker = -1
    target = -1
    attack_id = -1
    remain_hp = -1
    energy = False


plan = AttackPlan()
pre_turn = 0


def get_card(obs: Observation, area: AreaType, index: int, player_index: int) -> Pokemon | Card | None:
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


def prize_count(pokemon: Pokemon) -> int:
    data = card_table[pokemon.id]
    count = 3 if data.megaEx else 2 if data.ex else 1
    for card in pokemon.energyCards:
        if card.id == 12:  # Legacy Energy
            count -= 1
    for card in pokemon.tools:
        if card.id == 1172 and "Lillie" in data.name:
            count -= 1
    return max(0, count)


def pokemon_score(pokemon: Pokemon) -> int:
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


def agent(obs_dict: dict) -> list[int]:
    obs = to_observation_class(obs_dict)
    if obs.select == None:
        return my_deck

    state = obs.current
    select = obs.select
    context = select.context
    my_index = state.yourIndex
    my_state = state.players[my_index]
    op_state = state.players[1 - my_index]

    global plan
    global pre_turn
    if pre_turn != state.turn:
        pre_turn = state.turn
        plan = AttackPlan()

    field_counts = defaultdict(int)
    hand_counts = defaultdict(int)
    for card in my_state.active + my_state.bench:
        if card == None:
            continue
        field_counts[card.id] += 1
    for card in my_state.hand:
        hand_counts[card.id] += 1

    hand_energy = sum(hand_counts[e] for e in ENERGY_IDS)

    # Is the opponent's main threat an {ex} Pokémon? (Crustle walls those)
    op_active_ex = False
    if op_state.active and op_state.active[0] != None:
        op_active_ex = card_table[op_state.active[0].id].ex or card_table[op_state.active[0].id].megaEx

    stadium_id = 0
    for card in state.stadium:
        stadium_id = card.id

    if context == SelectContext.MAIN:
        can_switch = False
        can_op_switch = False
        for o in select.option:
            if o.type == OptionType.PLAY:
                card = get_card(obs, AreaType.HAND, o.index, my_index)
                if card.id == Boss_Orders:
                    can_op_switch = True
                elif card.id == Switch:
                    can_switch = True
            elif o.type == OptionType.RETREAT:
                can_switch = True

        my_cards = [my_state.active[0]]
        for pokemon in my_state.bench:
            my_cards.append(pokemon)
        op_cards = [op_state.active[0]]
        for pokemon in op_state.bench:
            op_cards.append(pokemon)

        if state.turn >= 2:
            best_score = -1
            for i, my_pokemon in enumerate(my_cards):
                if my_pokemon == None:
                    continue
                if i != 0 and not can_switch:
                    break
                energy_required = 0
                base_damage = 0
                base_score = 0
                attack_id = -1
                atk_type = EnergyType.COLORLESS
                if my_pokemon.id == Kangaskhan:
                    energy_required = 3
                    base_damage = 200  # + coin flips, treat expectation conservatively
                    attack_id = RAPID_FIRE
                elif my_pokemon.id == Crustle:
                    energy_required = 3
                    base_damage = 120
                    base_score += 40  # damage ignores opposing effects
                    attack_id = SUPERB_SCISSORS
                    atk_type = EnergyType.GRASS
                elif my_pokemon.id == Dwebble:
                    # Ascension: free evolve from deck; valuable while Crustle line unfinished
                    if i == 0 and field_counts[Crustle] < 2:
                        if best_score < 30:
                            best_score = 30
                            plan.attacker = 0
                            plan.target = 0
                            plan.attack_id = ASCENSION
                            plan.remain_hp = 999
                            plan.energy = False
                    continue
                else:
                    continue

                more_energy = False
                energy_count = len(my_pokemon.energies)
                if energy_count < energy_required:
                    if hand_energy >= 1 and not state.energyAttached:
                        energy_count += 1
                        if energy_count < energy_required:
                            continue
                        more_energy = True
                    else:
                        continue

                for j, op_pokemon in enumerate(op_cards):
                    if op_pokemon == None:
                        continue
                    if j != 0 and not can_op_switch:
                        break
                    damage = base_damage
                    data = card_table[op_pokemon.id]
                    if data.weakness == atk_type:
                        damage *= 2
                    elif data.resistance == atk_type:
                        damage -= 30
                    prize = 0
                    score = pokemon_score(op_pokemon)
                    if op_pokemon.hp <= damage:
                        prize = prize_count(op_pokemon)
                    else:
                        score *= damage / op_pokemon.hp
                    score += base_score

                    if len(op_state.prize) <= prize:
                        score = 50000

                    if i == 0:
                        score += 220
                    if j == 0:
                        score += 300
                    score += energy_count
                    if best_score < score:
                        best_score = score
                        plan.attacker = i
                        plan.target = j
                        plan.attack_id = attack_id
                        plan.remain_hp = op_pokemon.hp - damage
                        plan.energy = more_energy

    def energy_score(pokemon: Pokemon, active: bool, energy_id: int = -1) -> int:
        energy_count = len(pokemon.energies)
        score = 8000
        if active:
            score += 10
        if pokemon.id == Kangaskhan:
            score += 3
            if energy_count < 3:
                score += 100
        elif pokemon.id == Crustle:
            score += 2
            if energy_count < 3:
                score += 90
            # grass-providing energies belong on Crustle (Superb Scissors cost)
            if energy_id in (Grass_Energy, Grow_Grass_Energy):
                score += 30
        elif pokemon.id == Dwebble:
            if energy_count < 3 and field_counts[Crustle] + hand_counts[Crustle] >= 1:
                score += 50
        else:
            score -= 100
        # Mist Energy protects the holder from attack effects: best on the wall
        if energy_id == Mist_Energy and pokemon.id in (Crustle, Kangaskhan):
            score += 10
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
            if card != None:
                energy_count = 0
                if isinstance(card, Pokemon):
                    energy_count = len(card.energies)
                if context == SelectContext.SWITCH or context == SelectContext.TO_ACTIVE:
                    if o.playerIndex == my_index:
                        score += energy_count * 2
                        if o.index == plan.attacker - 1:
                            score += 100
                        if card.id == Crustle:
                            score += 30 if op_active_ex else 10
                        elif card.id == Kangaskhan:
                            score += 20 if not op_active_ex else 8
                        elif card.id == Dwebble:
                            score += 3
                    else:
                        if o.index == plan.target - 1:
                            score += 100
                elif context == SelectContext.SETUP_ACTIVE_POKEMON:
                    if card.id == Kangaskhan:
                        score = 4
                    elif card.id == Dwebble:
                        score = 3
                elif context == SelectContext.SETUP_BENCH_POKEMON:
                    if card.id == Dwebble:
                        score = 4
                    elif card.id == Kangaskhan:
                        score = 3
                elif context == SelectContext.TO_HAND:
                    score = 200 - hand_counts[card.id] * 100
                    if card.id == Crustle:
                        if field_counts[Dwebble] >= 1:
                            score += 80
                        else:
                            score -= 10
                    elif card.id == Dwebble:
                        need = 2 - (field_counts[Dwebble] + field_counts[Crustle])
                        score += 40 * need
                    elif card.id == Kangaskhan:
                        if field_counts[card.id] >= 2:
                            score -= 50
                        else:
                            score += 45
                    elif card.id in ENERGY_IDS:
                        if hand_energy == 0 and not state.energyAttached:
                            score += 30
                        else:
                            score -= 1
                    elif card.id == Boss_Orders:
                        score += 15
                    elif card.id == Jumbo_Ice_Cream:
                        score += 10
                elif context == SelectContext.ATTACH_FROM:
                    score = energy_score(card, o.area == AreaType.ACTIVE)
                elif context == SelectContext.HEAL or context == SelectContext.REMOVE_DAMAGE_COUNTER:
                    if o.playerIndex == my_index:
                        data = card_table[card.id]
                        score = (data.hp - card.hp) * 10
                        if card.id in (Crustle, Kangaskhan):
                            score += 30
                    else:
                        score = -100
                elif context == SelectContext.DAMAGE_COUNTER or context == SelectContext.DAMAGE:
                    if o.playerIndex != my_index:
                        if card.hp <= 30:
                            score = 5000 + pokemon_score(card)
                        else:
                            score = pokemon_score(card)
                    else:
                        score = -100
                elif context == SelectContext.EVOLVES_TO:
                    if card.id == Crustle:
                        score = 100
        elif o.type == OptionType.PLAY:
            card = get_card(obs, AreaType.HAND, o.index, my_index)
            data = card_table[card.id]
            if data.cardType == CardType.POKEMON:
                score = 20000
                if card.id == Dwebble:
                    if field_counts[Dwebble] + field_counts[Crustle] >= 3:
                        score = -1
                elif card.id == Kangaskhan:
                    if field_counts[card.id] >= 2:
                        score = -1
            else:
                score = 10000
                if card.id == Switch:
                    if plan.attacker <= 0:
                        score = -1
                    else:
                        score = 6000
                elif card.id == Jumbo_Ice_Cream:
                    worst = 0
                    for p in my_state.active + my_state.bench:
                        if p != None:
                            worst = max(worst, card_table[p.id].hp - p.hp)
                    score = 8000 if worst >= 60 else -1
                elif card.id == Boss_Orders:
                    if plan.target >= 1:
                        score = 3200
                    else:
                        score = -1
                elif card.id == Lillie_Determination:
                    score = 3100
                elif card.id == Hilda:
                    score = 3050
                elif card.id == Petrel:
                    score = 3000
                elif card.id == Eri:
                    score = 2900
                elif card.id == Lisia:
                    score = 2850
                elif card.id == Xerosic:
                    score = 2800
                elif card.id == Community_Center or card.id == TR_Factory:
                    if stadium_id == card.id:
                        score = -1
                    else:
                        score = 5000
        elif o.type == OptionType.ATTACH:
            card = get_card(obs, AreaType.HAND, o.index, my_index)
            pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
            if card.id == Heros_Cape or card.id == Handheld_Fan:
                score = 7000
                if pokemon.id == Kangaskhan:
                    score += 200
                elif pokemon.id == Crustle:
                    score += 150
            else:
                score = energy_score(pokemon, o.inPlayArea == AreaType.ACTIVE, card.id)
                if o.inPlayArea == AreaType.ACTIVE:
                    if plan.attacker == 0 and plan.energy:
                        score += 200
                else:
                    if plan.attacker == 1 + o.inPlayIndex and plan.energy:
                        score += 200
        elif o.type == OptionType.EVOLVE:
            pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
            score = 9000 + len(pokemon.energies)
        elif o.type == OptionType.ABILITY:
            score = 30000  # Run Errand: draw 2, always
        elif o.type == OptionType.RETREAT:
            if plan.attacker >= 1:
                score = 2000
            else:
                score = -1
        elif o.type == OptionType.ATTACK:
            score = 1000
            if o.attackId == plan.attack_id:
                score += 100

        scores.append(score)

    desc_indices = [i for i, _ in sorted(enumerate(scores), key=lambda x: x[1], reverse=True)]
    return desc_indices[:select.maxCount]
