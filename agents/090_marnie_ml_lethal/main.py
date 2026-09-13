import os
import random
import time
from collections import Counter, defaultdict

from cg.api import (
    AreaType,
    Card,
    CardType,
    EnergyType,
    Observation,
    OptionType,
    Pokemon,
    SelectContext,
    all_card_data,
    to_observation_class,
    search_begin,
    search_step,
    search_end,
)

"""
090_marnie_ml_lethal — EXP-090 (I-135): 089_marnie_ml + lethal-only determinized
search layer (the 075-style 3-layer stack: lethal -> ML -> rules).

Base: agents/089_marnie_ml (EXP-089, verbatim copy; helpers renamed feat_090/
model_090 to avoid module collisions when 089 and 090 are loaded side by side).
Added: the lethal search layer ported from agents/043_marnie_full (EXP-043),
including the mandatory plan VALUE-snapshot isolation (038-family policies
mutate the global plan in place; search rollouts replaying MAINs of the same
turn would otherwise corrupt the real-trajectory sub-selections: -16pt in the
wall matchup, EXP-043). SEARCH_SAMPLES=8 (Marnie's in-turn draw lines need 8,
EXP-043). _search_time_used is reset at the top of agent() each game
(INSIGHTS: latent 045 accumulation bug). F090_LETHAL=off disables the layer;
with search unavailable the agent naturally falls back to 089 behavior.

--- Original 089 header below ---

089_marnie_ml — EXP-089 (I-114): EXP-057-style direct action-class policy learning
applied to the Marnie's Grimmsnarl deck (sig 1ec0f47981, teacher = Luca, the
originator team of the recipe).

Structure: 082_marnie_1ec0 rule policy as the backbone (EXP-061: ML must sit on a
competitive rule spine, MAIN only) + a LightGBM multiclass layer that predicts the
teacher's action class on MAIN decisions (turn>=2, single-pick). Within the chosen
class, the concrete option is picked by the 082 policy scores. With no model file
or F090_ML=off, behavior is identical to 082.

Backbone below is a verbatim copy of agents/082_marnie_1ec0/main.py (EXP-082),
refactored only to expose policy_scores()/policy().

Base: agents/038_marnie_replica (sig 7308938c4d replica). Deck is different:
drops Dunsparce/Dudunsparce, Budew, Yveltal, Fezandipiti, Hero's Cape, Xerosic,
Risky Ruins; adds Snorunt/Froslass 2-2 (Freezing Shroud passive chip), Team Rocket's
Petrel x4, Night Stretcher x3, Unfair Stamp (ACE SPEC), Pokegear, Poke Pad x4.

Behavior baked in from action-frequency aggregation over 100 sampled replays
(seed 1; totals below are per-100-games):
 1. Always goes first (55/55).
 2. Setup active Impidimp 53 > Munkidori 30 > Snorunt 17; bench Munkidori 30 >
    Snorunt 14 > Impidimp 7.
 3. Hand energy: Munkidori(0en) 224 >> Impidimp(0en) 48 > Grimmsnarl(1en) 16;
    Snorunt/Froslass only get a retreat energy when stuck active (13/12).
 4. Only real attack is Shadow Bullet (324; Filch 29, Morgrem Corkscrew 15,
    Froslass Frost Smash 0, Mind Bend 0). Froslass is a pure passive chipper.
 5. Supporters: Petrel 145 ~ Lillie 130 (Lillie at small hand, mean 4.4; Petrel 4.9)
    > Boss 43 (kill tool) > Dawn 32. Petrel's top fetch is Unfair Stamp 55 >
    Night Stretcher 36 > Poke Pad 15.
 6. Unfair Stamp played 67, at opp hand mean 10.0 (disruption when opponent is fat).
 7. Up to 3 Munkidori fielded (max simultaneous: 2 in 46 games, 3 in 30, 4 in 11);
    Snorunt/Froslass line up to 2.
 8. Adrena-Brain destination: opposing Munkidori at ANY hp (49 placements at full HP);
    source: own Munkidori 250 > Grimmsnarl 188 (Freezing Shroud self-chip is ammo).
 9. Night Stretcher: energy 100 >> Impidimp 21 > Munkidori 16 (energy recycling).
"""

# Load deck.csv in the dataset
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

# Decklist (sig 1ec0f47981)
Basic_Dark_Energy = 7      # x10
Froslass = 104             # x2 (Freezing Shroud: checkup chip on all ability Pokémon)
Munkidori = 112            # x4
Impidimp = 646             # x4
Morgrem = 647              # x3
Grimmsnarl = 648           # x3
Snorunt = 860              # x2
Rare_Candy = 1079          # x3
Unfair_Stamp = 1080        # x1 (ACE SPEC: both shuffle hands, I draw 5, opp draws 2)
Buddy_Poffin = 1086        # x4
Night_Stretcher = 1097     # x3 (Pokémon or basic energy from discard to hand)
Pokegear = 1122            # x1
Tool_Scrapper = 1137       # x1
Poke_Pad = 1152            # x4 (search non-rule-box Pokémon)
Boss_Orders = 1182         # x2
Petrel = 1219              # x4 (search any Trainer)
Lillie_Determination = 1227  # x4
Dawn = 1231                # x1
Spikemuth_Gym = 1259       # x4

SHADOW_BULLET = 937
CORKSCREW_PUNCH = 936
FILCH = 934

Crustle_Wall = 345         # opp wall: Mysterious Rock Inn zeroes ex attack damage
Mega_Kangaskhan = 756

GRIMM_LINE = (Impidimp, Morgrem, Grimmsnarl)
SNOW_LINE = (Snorunt, Froslass)


class AttackPlan:
    attacker = -1
    target = -1
    target_id = -1
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
        if card.id == 1172 and "Lillie" in data.name:  # Lillie's Pearl
            count -= 1
    return max(0, count)


def pokemon_score(pokemon: Pokemon) -> int:
    """Value of targeting an opponent Pokémon."""
    data = card_table[pokemon.id]
    score = prize_count(pokemon) * 1000
    score += len(pokemon.energies) * 150
    score += len(pokemon.tools) * 100
    if data.stage2:
        score += 250
    elif data.stage1:
        score += 130
    if pokemon.id == Munkidori:
        # The individual funnels Adrena counters into opposing Munkidori at any HP
        # (49 placements at full 110 HP in 100 games): break the mirror engine first.
        score += 250
        if len(pokemon.energies) >= 1:
            score += 150
    score += pokemon.hp
    return score


def policy_scores(obs: Observation) -> list[int]:
    """082ルール方策: 全選択肢のスコア（MAINでは副作用としてplanを更新）。"""
    state = obs.current
    select = obs.select
    context = select.context
    my_index = state.yourIndex
    my_state = state.players[my_index]
    op_state = state.players[1 - my_index]
    my_prize = len(my_state.prize)

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
    bench_used = sum(1 for p in my_state.bench if p != None)
    snow_count = field_counts[Snorunt] + field_counts[Froslass]

    op_tool_count = 0
    for pokemon in op_state.active + op_state.bench:
        if pokemon != None:
            op_tool_count += len(pokemon.tools)

    # Number of ready attackers (Grimmsnarl with 2+ energies)
    ready_grimm = 0
    for card in my_state.active + my_state.bench:
        if card != None and card.id == Grimmsnarl and len(card.energies) >= 2:
            ready_grimm += 1

    stadium_id = 0
    for card in state.stadium:
        stadium_id = card.id

    # Wall matchup flags (Crustle 345: Mysterious Rock Inn zeroes ex attack damage)
    wall_in_play = any(p is not None and p.id == Crustle_Wall
                       for p in op_state.active + op_state.bench)
    wall_active = bool(op_state.active and op_state.active[0] is not None
                       and op_state.active[0].id == Crustle_Wall)

    can_attack = False
    if context == SelectContext.MAIN:
        can_switch = False
        can_op_switch = False
        for o in select.option:
            if o.type == OptionType.PLAY:
                card = get_card(obs, AreaType.HAND, o.index, my_index)
                if card.id == Boss_Orders:
                    can_op_switch = True
            elif o.type == OptionType.RETREAT:
                can_switch = True
            elif o.type == OptionType.ATTACK:
                can_attack = True

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
                attacker_is_ex = False
                if my_pokemon.id == Grimmsnarl:
                    energy_required = 2
                    base_damage = 180
                    base_score += 60  # bench snipe 30 is nearly always useful
                    attack_id = SHADOW_BULLET
                    attacker_is_ex = True
                elif my_pokemon.id == Morgrem:
                    energy_required = 2
                    base_damage = 60
                    base_score -= 50  # would rather evolve than attack with Morgrem
                    attack_id = CORKSCREW_PUNCH
                elif my_pokemon.id == Impidimp:
                    energy_required = 0
                    base_damage = 0
                    base_score = 10  # Filch: draw 1, better than nothing
                    attack_id = FILCH
                else:
                    # Munkidori / Snorunt / Froslass never attack in the replays
                    # (Frost Smash 0 uses, Mind Bend 0 uses)
                    continue
                if attack_id == FILCH and my_state.deckCount <= 2:
                    continue  # deck guard: never draw ourselves to death

                more_energy = False
                energy_count = len(my_pokemon.energies)
                if energy_count < energy_required:
                    if hand_counts[Basic_Dark_Energy] >= 1 and not state.energyAttached:
                        energy_count += 1
                        if energy_count < energy_required:
                            continue
                        more_energy = True
                    else:
                        continue

                if base_damage <= 0:
                    # Filch plan: only as fallback on the active spot
                    if i == 0 and best_score < 10:
                        best_score = 10
                        plan.attacker = 0
                        plan.target = 0
                        plan.attack_id = FILCH
                        plan.remain_hp = 999
                        plan.energy = False
                    continue

                for j, op_pokemon in enumerate(op_cards):
                    if op_pokemon == None:
                        continue
                    if j != 0 and not can_op_switch:
                        break
                    damage = base_damage
                    data = card_table[op_pokemon.id]
                    if data.weakness == EnergyType.DARKNESS:
                        damage *= 2
                    elif data.resistance == EnergyType.DARKNESS:
                        damage -= 30
                    if op_pokemon.id == Crustle_Wall and attacker_is_ex:
                        # Mysterious Rock Inn: ex attack damage is zeroed (study K1).
                        # Never plan kills into the wall; Morgrem's 60 does go through.
                        damage = 0
                    prize = 0
                    wall_breaker_bonus = 0
                    if op_pokemon.id == Crustle_Wall and my_pokemon.id == Morgrem:
                        # Wall games: Morgrem is the individual's wall-breaker
                        # (338 active-turns / 27 Corkscrew uses vs Kangaskhan decks;
                        # a cheap 1-prize body that actually damages Crustle)
                        wall_breaker_bonus = 250
                    score = pokemon_score(op_pokemon)
                    if op_pokemon.hp <= damage:
                        prize = prize_count(op_pokemon)
                        # Gust/attack to KILL (038 rule, confirmed here: Boss targets are
                        # low-HP evolvers / opposing Munkidori)
                        score += 1200
                    else:
                        score *= damage / op_pokemon.hp
                    score += base_score + wall_breaker_bonus

                    if len(op_state.prize) <= prize:
                        score = 50000

                    if i == 0:
                        # Tank rotation: a damaged active Grimmsnarl steps back when a
                        # charged replacement waits on the bench (retreats observed at
                        # 140 HP; Adrena-Brain heals it later).
                        if (my_pokemon.id == Grimmsnarl and my_pokemon.hp <= 140
                                and any(p is not None and p.id == Grimmsnarl and len(p.energies) >= 2
                                        for p in my_state.bench)):
                            score -= 50
                        else:
                            score += 220
                    if j == 0:
                        score += 300
                    score += energy_count
                    if best_score < score:
                        best_score = score
                        plan.attacker = i
                        plan.target = j
                        plan.target_id = op_pokemon.id
                        plan.attack_id = attack_id
                        plan.remain_hp = op_pokemon.hp - damage
                        plan.energy = more_energy

    def energy_score(pokemon: Pokemon, active: bool) -> int:
        """Where to attach a {D} energy.

        Replay stats (hand attach dest): Munkidori(0en) 224 >> Impidimp(0en) 48 >
        Grimmsnarl(1en) 16 > Snorunt 13 ~ Froslass 12 > Morgrem 8. Punk Up charges
        the Grimmsnarl line; the manual attach switches on Adrena-Brain, then
        pre-charges the next Grimm line body.
        """
        energy_count = len(pokemon.energies)
        score = 8000
        if active:
            score += 10
        if pokemon.id == Munkidori:
            if energy_count < 1:
                score += 120  # top priority: Adrena-Brain online
            else:
                return -5  # saturated: decline (keep energy in hand/deck, only 10 total)
        elif pokemon.id == Grimmsnarl:
            score += 3
            if energy_count < 2:
                score += 100
            else:
                # Never overstack a Grimmsnarl past 2 (Shadow Bullet cost): the 10
                # deck energies are the scarcest resource in long games, and a KO'd
                # 5-energy Grimmsnarl is an instant energy drought. Negative score
                # lets "up to N" Punk Up selections decline the surplus.
                return -5
            if ready_grimm >= 1:
                score -= 50
        elif pokemon.id == Morgrem or pokemon.id == Impidimp:
            score += 2
            if energy_count < 2:
                score += 80
            else:
                return -5
            if ready_grimm >= 1:
                score -= 50
        elif pokemon.id in SNOW_LINE:
            if active and energy_count < 1:
                score += 12  # retreat money when stuck in front; never charge otherwise
            else:
                return -5
        else:
            return -5
        return score

    scores = []
    for o in select.option:
        score = 0
        if o.type == OptionType.NUMBER:
            score = o.number
        elif o.type == OptionType.YES:
            score = 1  # Punk Up / Adrena-Brain / Spikemuth search / go first: default yes (196:1)
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
                        if card.id == Grimmsnarl:
                            # In wall games the individual promotes cheap non-ex bodies
                            # instead of the grass-weak 2-prize Grimmsnarl (promotions
                            # vs Kangaskhan: Morgrem(0en) 34 / Impidimp(0en) 33)
                            score += 6 if wall_active else 20
                        elif card.id == Morgrem:
                            score += 12 if wall_active else 10
                        elif card.id == Impidimp:
                            score += 8  # most-promoted body after a KO (53 promotions)
                        elif card.id == Froslass:
                            score += 4  # 1-prize wall that keeps chipping
                        elif card.id == Snorunt:
                            score += 3
                        elif card.id == Munkidori:
                            score += 2  # keep it benched moving counters
                    else:
                        if o.index == plan.target - 1:
                            score += 100
                elif context == SelectContext.SETUP_ACTIVE_POKEMON:
                    # Setup active: Impidimp 53 > Munkidori 30 > Snorunt 17
                    if card.id == Impidimp:
                        score = 6
                    elif card.id == Munkidori:
                        score = 5
                    elif card.id == Snorunt:
                        score = 4
                elif context == SelectContext.SETUP_BENCH_POKEMON:
                    # Setup bench: Munkidori 30 > Snorunt 14 > Impidimp 7
                    if card.id == Munkidori:
                        score = 5
                    elif card.id == Snorunt:
                        score = 4
                    elif card.id == Impidimp:
                        score = 3
                elif context in (SelectContext.TO_HAND, SelectContext.TO_BENCH, SelectContext.TO_FIELD):
                    # Deck/discard/looking searches: Spikemuth Gym, Poke Pad, Poffin,
                    # Petrel, Night Stretcher, Dawn, Pokegear all land here.
                    score = 200 - hand_counts[card.id] * 100
                    if card.id == Impidimp:
                        need = 3 - (field_counts[Impidimp] + field_counts[Morgrem] + field_counts[Grimmsnarl])
                        score += 50 * need
                    elif card.id == Grimmsnarl:
                        # Spikemuth search: Grimmsnarl to hand 167 (top fetch overall)
                        if field_counts[Impidimp] + field_counts[Morgrem] >= 1:
                            score += 80
                        else:
                            score -= 10
                    elif card.id == Morgrem:
                        # Poke Pad top fetch (138): grab it whenever the line needs it
                        if field_counts[Impidimp] >= 1 and hand_counts[Rare_Candy] == 0:
                            score += 40
                        else:
                            score -= 20
                    elif card.id == Munkidori:
                        if field_counts[card.id] >= 3:
                            score -= 100
                        else:
                            score += 45
                    elif card.id == Froslass:
                        if field_counts[Snorunt] >= 1 and field_counts[Froslass] == 0:
                            score += 40
                        else:
                            score -= 30
                    elif card.id == Snorunt:
                        if snow_count < 2:
                            score += 25
                        else:
                            score -= 50
                    elif card.id == Unfair_Stamp:
                        # Petrel's #1 fetch (55/145): pre-stock the disruption bomb
                        score += 90
                    elif card.id == Tool_Scrapper:
                        # Petrel fetch #2 in wall games (19): strip Cape / healing tools
                        score += 60 if op_tool_count >= 1 else -20
                    elif card.id == Night_Stretcher:
                        has_dark_in_discard = any(c.id == Basic_Dark_Energy for c in my_state.discard)
                        score += 55 if has_dark_in_discard else 5
                    elif card.id == Poke_Pad:
                        score += 30
                    elif card.id == Rare_Candy:
                        if field_counts[Impidimp] >= 1 and hand_counts[Grimmsnarl] >= 1:
                            score += 70
                        else:
                            score += 25
                    elif card.id == Basic_Dark_Energy:
                        # Night Stretcher discard pick: energy 100 >> Pokémon 55
                        if hand_counts[Basic_Dark_Energy] == 0:
                            score += 60
                        else:
                            score -= 1
                    elif card.id == Lillie_Determination:
                        score += 25  # Pokegear pick
                    elif card.id == Petrel:
                        score += 20
                    elif card.id == Boss_Orders:
                        # Wall games are won by dragging up the benched Mega Kangaskhan
                        # for Shadow Bullet 180x2 (3 prizes): fetch the gust actively
                        if (wall_in_play and hand_counts[Boss_Orders] == 0 and ready_grimm >= 1
                                and any(p is not None and p.id == Mega_Kangaskhan for p in op_state.bench)):
                            score += 85
                        else:
                            score += 15
                    elif card.id == Spikemuth_Gym:
                        score += 10 if stadium_id != Spikemuth_Gym else -20
                elif context == SelectContext.ATTACH_FROM:
                    score = energy_score(card, o.area == AreaType.ACTIVE)
                elif context == SelectContext.DAMAGE_COUNTER or context == SelectContext.DAMAGE:
                    # Shadow Bullet snipe / Adrena-Brain destination
                    if o.playerIndex != my_index:
                        if context == SelectContext.DAMAGE and card.id == Crustle_Wall:
                            # Shadow Bullet's bench snipe is also zeroed by the wall
                            score = -50
                        elif card.hp <= 30:
                            score = 5000 + pokemon_score(card)
                        elif card.hp <= 60:
                            # kill it over two moves (low-HP evolving basics: Abra etc.)
                            score = 400 + pokemon_score(card)
                        else:
                            score = pokemon_score(card)
                    else:
                        score = -100  # never place counters on my own side if avoidable
                elif context == SelectContext.REMOVE_DAMAGE_COUNTER:
                    # Adrena-Brain source: own Munkidori 250 > Grimmsnarl 188
                    # (Freezing Shroud self-chip on Munkidori is the ammo supply)
                    if o.playerIndex == my_index:
                        data = card_table[card.id]
                        score = (data.hp - card.hp) * 10
                        if card.id == Munkidori:
                            score += 50  # heal the fragile engine first
                        elif card.id == Grimmsnarl:
                            score += 30  # then protect the attacker
                    else:
                        score = -100
                elif context == SelectContext.EVOLVES_TO:
                    if card.id == Grimmsnarl:
                        score = 100
                    elif card.id == Morgrem:
                        score = 50
                    elif card.id == Froslass:
                        score = 40
        elif o.type == OptionType.TOOL_CARD:
            # Tool Scrapper: discard opponent tools, never our own
            if o.playerIndex != my_index:
                score = 300
            else:
                score = -200
        elif o.type == OptionType.PLAY:
            card = get_card(obs, AreaType.HAND, o.index, my_index)
            data = card_table[card.id]
            if data.cardType == CardType.POKEMON:
                score = 20000
                if card.id == Impidimp:
                    if field_counts[Impidimp] + field_counts[Morgrem] + field_counts[Grimmsnarl] >= 3:
                        score = -1
                elif card.id == Munkidori:
                    # Individual fields up to 3 (max simultaneous 3 in 30/100 games)
                    if field_counts[card.id] >= 3:
                        score = -1
                    elif field_counts[card.id] == 2:
                        score = 13000 if bench_used <= 3 else -1
                    elif field_counts[card.id] == 1:
                        score = 15000 if bench_used <= 4 else -1
                    else:
                        score = 18000
                elif card.id == Snorunt:
                    if snow_count >= 2 or bench_used > 4:
                        score = -1
                    else:
                        score = 14000
            else:
                score = 10000
                if card.id == Rare_Candy:
                    score = 12000  # engine only offers it when playable
                elif card.id == Tool_Scrapper:
                    score = 11000 if op_tool_count >= 1 else -1
                elif card.id == Unfair_Stamp:
                    # Played at opp hand mean 10.0 / own hand mean 4.1 (67 plays):
                    # a disruption bomb, dropped before the supporter for the turn.
                    if op_state.handCount >= 5:
                        score = 3400
                    elif len(my_state.hand) <= 4:
                        score = 3000
                    else:
                        score = 300
                elif card.id == Pokegear:
                    score = 10500
                elif card.id == Boss_Orders:
                    # Kill tool (038 rule; this individual's gust targets are killable
                    # evolvers / opposing Munkidori). Wall games add a second mode:
                    # drag up Mega Kangaskhan past the Crustle wall and Shadow Bullet
                    # it for 3 prizes over 2 turns (gusts vs Kangaskhan: 756 x23).
                    if plan.target >= 1 and plan.remain_hp <= 0:
                        score = 3250
                    elif plan.target >= 1 and plan.target_id == Mega_Kangaskhan:
                        score = 3200
                    elif plan.target >= 1:
                        score = 400  # drag-up: only when nothing better
                    else:
                        score = -1
                elif card.id == Petrel:
                    # Petrel 145 ~ Lillie 130; Lillie preferred on small hands
                    # (mean 4.4 vs 4.9), Petrel preferred over Boss 29:10
                    if len(my_state.hand) >= 5:
                        score = 3150
                    else:
                        score = 3050
                elif card.id == Lillie_Determination:
                    # Deck guard (study K9): wall games last 16+ turns and two of the
                    # source individual's study losses were deck-outs. Lillie shrinks
                    # the deck when the hand is small.
                    if my_state.deckCount + len(my_state.hand) <= 9:
                        score = 200
                    else:
                        score = 3100
                elif card.id == Dawn:
                    score = 2900 if my_state.deckCount > 6 else 200
                elif card.id == Spikemuth_Gym:
                    if stadium_id == Spikemuth_Gym:
                        score = -1
                    else:
                        score = 5000
        elif o.type == OptionType.ATTACH:
            card = get_card(obs, AreaType.HAND, o.index, my_index)
            pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
            score = energy_score(pokemon, o.inPlayArea == AreaType.ACTIVE)
            if o.inPlayArea == AreaType.ACTIVE:
                if plan.attacker == 0 and plan.energy:
                    score += 200
            else:
                if plan.attacker == 1 + o.inPlayIndex and plan.energy:
                    score += 200
        elif o.type == OptionType.EVOLVE:
            pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
            score = 9000 + len(pokemon.energies)
            card = get_card(obs, o.area, o.index, my_index)
            if card != None and card.id == Froslass:
                score = 8900  # evolve the attacker line first, then the chipper
        elif o.type == OptionType.ABILITY:
            score = 30000  # Adrena-Brain: default use (496 uses/100 games)
        elif o.type == OptionType.RETREAT:
            if plan.attacker >= 1:
                score = 2000
            else:
                score = -1
        elif o.type == OptionType.ATTACK:
            score = 1000
            if o.attackId == plan.attack_id:
                score += 100
            elif o.attackId == SHADOW_BULLET:
                score += 90

        scores.append(score)

    return scores


def policy(obs: Observation) -> list[int]:
    """082ルール方策の最終選択（フォールバック層）。"""
    select = obs.select
    scores = policy_scores(obs)
    desc_indices = [i for i, _ in sorted(enumerate(scores), key=lambda x: x[1], reverse=True)]
    chosen = desc_indices[:select.maxCount]
    # For "up to N" selections, drop deliberately-avoided (negative) picks
    while len(chosen) > select.minCount and scores[chosen[-1]] < 0:
        chosen.pop()
    return chosen


# ---- EXP-089: 選択クラス予測モデル（GBT、EXP-057定式化のMarnie適用） ----
# ロード順: lightgbm+model_090.txt（ローカル高速）→ model_090.npz（純Python、Kaggle用。
# argmax一致検証済み）→ どちらも無ければ082ルールへ自然フォールバック
try:  # Kaggle評価環境はexecロードのため __file__ が無い
    _AGENT_DIR_090 = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _AGENT_DIR_090 = ("/kaggle_simulations/agent"
                      if os.path.exists("/kaggle_simulations/agent/main.py")
                      else os.getcwd())

_M090 = None
if os.environ.get("F090_ML", "on") != "off":
    try:
        try:
            import feat_090 as _ft090
        except ImportError:
            import importlib.util as _ilu090

            _fp = os.path.join(_AGENT_DIR_090, "feat_090.py")
            _sp = _ilu090.spec_from_file_location("feat_090", _fp)
            _ft090 = _ilu090.module_from_spec(_sp)
            _sp.loader.exec_module(_ft090)
        try:
            import lightgbm as _lgb090

            _mp = os.path.join(_AGENT_DIR_090, "model_090.txt")
            if not os.path.exists(_mp):
                raise FileNotFoundError(_mp)
            _M090 = {"mode": "lgb", "bst": _lgb090.Booster(model_file=_mp),
                     "ft": _ft090}
        except Exception:  # noqa: BLE001  lightgbm無し → npz純Python推論
            import numpy as _np090

            _z = _np090.load(os.path.join(_AGENT_DIR_090, "model_090.npz"))
            _M090 = {"mode": "npz", "ft": _ft090,
                     "feat": _z["feat"], "thr": _z["thr"],
                     "left": _z["left"], "right": _z["right"],
                     "val": _z["val"], "roots": _z["roots"],
                     "tcls": _z["tree_cls"],
                     "ncls": int(_z["num_class"])}
    except Exception:  # noqa: BLE001
        _M090 = None


def _m090_scores(x: list) -> list:
    """行動クラスごとの生スコア（softmax前。argmax用途なので正規化不要）。"""
    if _M090["mode"] == "lgb":
        return list(_M090["bst"].predict([x])[0])
    feat, thr = _M090["feat"], _M090["thr"]
    left, right, val = _M090["left"], _M090["right"], _M090["val"]
    scores = [0.0] * _M090["ncls"]
    for r, c in zip(_M090["roots"], _M090["tcls"]):
        i = int(r)
        while feat[i] >= 0:
            i = int(left[i]) if x[feat[i]] <= thr[i] else int(right[i])
        scores[int(c)] += float(val[i])
    return scores


def _ml_layer(obs: Observation) -> list[int] | None:
    """MAIN決定: GBTで行動クラスをargmax→クラス内は082policyスコアで具体option化。"""
    select = obs.select
    if select.minCount != 1 or select.maxCount != 1 or len(select.option) < 2:
        return None
    state = obs.current
    my = state.yourIndex
    ft = _M090["ft"]
    feats, _keys = ft.feat_vector(state, my)
    proba = _m090_scores(feats)
    cls_of = [ft.option_class(state, select, i, my)
              for i in range(len(select.option))]
    best_cls = max(set(cls_of), key=lambda c: proba[ft.CLASS_ID[c]])
    cands = [i for i, c in enumerate(cls_of) if c == best_cls]
    ps = policy_scores(obs)  # planの副作用も082と同一に保つ
    if len(cands) > 1:
        cands = [max(cands, key=lambda i: ps[i])]
    return [cands[0]]


# ---------------------------------------------------------------------------
# EXP-090: Lethal-only determinized search layer (MAIN only), ported from
# 043_marnie_full (itself from 036/007). Runs only when our remaining prizes
# are <=3. Candidate MAIN options are ranked by the 082 rule heuristic; a
# candidate overrides the lower layers only if EVERY determinization sample
# ends the greedy rollout of our own turn with the game already won. No board
# evaluation is involved, so a bad eval can never overrule the tuned layers.
# Any failure (incl. search API unavailable) falls back to ML -> rules.

SEARCH_CANDIDATES = 8
SEARCH_SAMPLES = 8           # EXP-043: Marnie's draw-order dependent lethal
                             # lines (Lillie/Unfair Stamp mid-turn draws) need
                             # 8 samples (5 gave 3 losing overrides / 300 games)
SEARCH_MOVE_BUDGET = 1.5     # seconds per MAIN decision
SEARCH_GAME_BUDGET = 450.0   # total seconds of search per game (600s limit)
ROLLOUT_STEP_CAP = 80

PLACEHOLDER_MON = 1072  # basic Pokémon used for unknown opponent cards
if not (card_table.get(PLACEHOLDER_MON) and card_table[PLACEHOLDER_MON].basic):
    PLACEHOLDER_MON = next(c.cardId for c in all_card if c.basic)

my_deck_counts = Counter(my_deck)

_search_time_used = 0.0

F090_LETHAL = os.environ.get("F090_LETHAL", "on") != "off"
F090_STATS = {"search_calls": 0, "fires": 0, "overrides": 0}


def _fallback_selection(select) -> list[int]:
    count = max(select.minCount, min(1, select.maxCount))
    return list(range(count))


def _my_visible_counts(obs: Observation) -> Counter:
    """Every card of mine whose location is known (for elimination counting)."""
    me = obs.current.players[obs.current.yourIndex]
    seen: Counter = Counter()
    for card in (me.hand or []):
        seen[card.id] += 1
    for card in me.discard:
        seen[card.id] += 1
    for pokemon in me.active + me.bench:
        if pokemon is None:
            continue
        seen[pokemon.id] += 1
        for card in pokemon.preEvolution:
            seen[card.id] += 1
        for card in pokemon.energyCards:
            seen[card.id] += 1
        for card in pokemon.tools:
            seen[card.id] += 1
    for card in obs.current.stadium:
        if seen[card.id] < my_deck_counts[card.id]:
            seen[card.id] += 1
    return seen


def _determinize(obs: Observation):
    state = obs.current
    me = state.players[state.yourIndex]
    op = state.players[1 - state.yourIndex]

    remaining = my_deck_counts - _my_visible_counts(obs)
    unknown = list(remaining.elements())
    random.shuffle(unknown)
    need = me.deckCount + len(me.prize)
    while len(unknown) < need:  # counting slack: pad with basic Dark energy
        unknown.append(Basic_Dark_Energy)
    your_deck = unknown[: me.deckCount]
    your_prize = unknown[me.deckCount : me.deckCount + len(me.prize)]

    opponent_deck = [PLACEHOLDER_MON] * op.deckCount
    opponent_prize = [PLACEHOLDER_MON] * len(op.prize)
    opponent_hand = [Basic_Dark_Energy] * op.handCount
    opponent_active = []
    if op.active and op.active[0] is None:
        opponent_active = [PLACEHOLDER_MON]
    return your_deck, your_prize, opponent_deck, opponent_prize, opponent_hand, opponent_active


def _rollout(state, root_turn: int, deadline: float):
    """Greedy rollout with the rule policy until our turn ends (or cap/deadline)."""
    for _ in range(ROLLOUT_STEP_CAP):
        obs = state.observation
        if obs.current.result != -1 or obs.current.turn != root_turn:
            break
        if time.perf_counter() > deadline:
            break
        try:
            sel = policy(obs)
            if not sel and obs.select.minCount > 0:
                sel = _fallback_selection(obs.select)
        except Exception:  # noqa: BLE001
            sel = _fallback_selection(obs.select)
        state = search_step(state.searchId, sel)
    return state.observation


def _lethal_search(obs: Observation, heuristic_scores: list) -> list[int] | None:
    """Override the MAIN choice only when a line certainly wins THIS turn."""
    global _search_time_used

    select = obs.select
    if select.minCount != 1 or select.maxCount != 1 or len(select.option) < 2:
        return None

    my_index = obs.current.yourIndex
    me = obs.current.players[my_index]
    if len(me.prize) > 3:  # cannot finish this turn: skip search entirely
        return None

    if _search_time_used > SEARCH_GAME_BUDGET:
        return None

    F090_STATS["search_calls"] += 1
    t0 = time.perf_counter()
    deadline = t0 + SEARCH_MOVE_BUDGET
    root_turn = obs.current.turn

    ranked = sorted(range(len(heuristic_scores)), key=lambda i: heuristic_scores[i], reverse=True)
    candidates = ranked[:SEARCH_CANDIDATES]

    always_wins = {i: True for i in candidates}
    tried = {i: 0 for i in candidates}
    try:
        for _ in range(SEARCH_SAMPLES):
            if time.perf_counter() > deadline:
                break
            root = search_begin(obs, *_determinize(obs))
            for idx in candidates:
                if not always_wins[idx] or time.perf_counter() > deadline:
                    continue
                try:
                    child = search_step(root.searchId, [idx])
                    final_obs = _rollout(child, root_turn, deadline)
                    if final_obs.current.result != my_index:
                        always_wins[idx] = False
                    tried[idx] += 1
                except Exception:  # noqa: BLE001
                    always_wins[idx] = False
    finally:
        try:
            search_end()
        except Exception:  # noqa: BLE001
            pass
        _search_time_used += time.perf_counter() - t0

    winners = [i for i in candidates if always_wins[i] and tried[i] == SEARCH_SAMPLES]
    if not winners:
        return None
    return [max(winners, key=lambda i: heuristic_scores[i])]


def agent(obs_dict: dict) -> list[int]:
    global plan, pre_turn, _search_time_used
    obs = to_observation_class(obs_dict)
    if obs.select == None:
        return my_deck

    if obs.current is not None and obs.current.turn <= 2:
        # Per-game search budget reset at the top of agent() (INSIGHTS: the
        # 045-style in-search reset never fires because the prize gate returns
        # first; local arena workers keep the module loaded across games).
        _search_time_used = 0.0

    is_main = (obs.select.context == SelectContext.MAIN
               and obs.current is not None and obs.current.turn >= 2)

    if F090_LETHAL and is_main:
        # Layer 1 (lethal): score once with the rule heuristic (this also sets
        # the attack plan for the real trajectory), then let the search
        # override only the MAIN choice.
        scores = policy_scores(obs)
        # Snapshot the plan VALUES (not the reference): rollouts inside the
        # search replay MAINs of the same turn and mutate the same global plan
        # object in place, which would otherwise corrupt the sub-selections
        # (Boss gust target, energy attach) that follow the real MAIN choice
        # (EXP-043: -16pt in the wall matchup).
        snap = AttackPlan()
        snap.attacker = plan.attacker
        snap.target = plan.target
        snap.target_id = plan.target_id
        snap.attack_id = plan.attack_id
        snap.remain_hp = plan.remain_hp
        snap.energy = plan.energy
        saved_turn = pre_turn
        try:
            choice = _lethal_search(obs, scores)
        except Exception:  # noqa: BLE001
            choice = None
        plan, pre_turn = snap, saved_turn
        if choice is not None:
            F090_STATS["fires"] += 1
            try:  # instrumentation only: would the lower layers differ?
                cf = None
                if _M090 is not None:
                    cf = _ml_layer(obs)
                if cf is None:
                    cf = policy(obs)
                if cf and cf[0] != choice[0]:
                    F090_STATS["overrides"] += 1
                plan, pre_turn = snap, saved_turn  # re-restore after cf scoring
            except Exception:  # noqa: BLE001
                plan, pre_turn = snap, saved_turn
            return choice

    # Layer 2 (ML): identical to 089
    if _M090 is not None and is_main:
        try:
            choice = _ml_layer(obs)
        except Exception:  # noqa: BLE001
            choice = None
        if choice is not None:
            return choice

    # Layer 3 (082 rules)
    return policy(obs)
