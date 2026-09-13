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
043_marnie_full — 038_marnie_replica + full replay study rules + lethal search.
EXP-043 / I-079.

Base: agents/038_marnie_replica (top ladder Marnie sig 7308938c4d, 7 frequency
rules). Added from the full-precision replay study (knowledge/study/
marnie_7308_study.md, 18 games incl. 6 losses vs Mega Kangaskhan):

Wall rules (opponent plays Crustle 345 / Cornerstone Ogerpon 117, whose
abilities nullify our ex / ability-holder attacks):
  K1  attack plan treats walls as untargetable by the blocked attackers
      (plus an explicit defensive rotation plan when nothing can be damaged,
      replacing the rotation 038 got implicitly from its nullified plan)
  K2  split DAMAGE (attack snipe, blocked by the wall) from DAMAGE_COUNTER
      (Adrena-Brain, bypasses it). Counter priority: killable(<=30) >
      non-wall ex (Kangaskhan/Cornerstone = the convertible prize target) >
      energized opposing Munkidori > lone wall concentration
  K5  Boss drags a benched Mega Kangaskhan past the wall (180x2 = 3 prizes)
  K6  never promote Munkidori after a KO in the wall matchup
  K7  Budew Itchy Pollen (item lock) bonus vs active wall (blocks Ice Cream)
Deck management (2 of 6 studied losses were self deck-outs):
  K9  stop optional draws when the deck is low (Run Away Draw / Flip the
      Script / Lillie / Dawn). Gated on wall_present: the ungated version
      cost ~4pt vs Alakazam (lost draw power) in the EXP-043 A/B, while
      deck-outs only occur in wall/stall matchups.
Mirror:
  M1  Adrena/counter destination: opposing energized Munkidori is a priority
      target even at full HP (break their heal+chip engine first)
Rejected in local falsification vs the optimized replicas (see EXP-043):
  K3 (never rotate vs walls) and K4 (3rd Munkidori) from the study made
  things worse vs 039/042 and were reverted to 038 behaviour.

Lethal-only determinized search (ported from 036/EXP-036): MAIN choices are
overridden only when our remaining prizes <=3 and a candidate wins the game in
ALL determinization samples. Any failure falls back to the pure heuristic.
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

# Decklist (sig 7308938c4d)
Basic_Dark_Energy = 7      # x10
Dudunsparce = 66           # x3 (Run Away Draw: draw 3, shuffle self back)
Munkidori = 112            # x3
Fezandipiti = 140          # x1 (ex, Flip the Script)
Budew = 235                # x1 (Itchy Pollen: item lock, 0 energy)
Dunsparce = 305            # x3
Impidimp = 646             # x4
Morgrem = 647              # x2
Grimmsnarl = 648           # x4
Yveltal = 689              # x1 (Dark Feather 110, non-ex)
Rare_Candy = 1079          # x4
Buddy_Poffin = 1086        # x4
Tool_Scrapper = 1137       # x1
Poke_Pad = 1152            # x4
Heros_Cape = 1159          # x1 (ACE SPEC, +100 HP)
Boss_Orders = 1182         # x2
Xerosic = 1197             # x1 (opponent discards)
Lillie_Determination = 1227  # x4
Dawn = 1231                # x3
Spikemuth_Gym = 1259       # x3
Risky_Ruins = 1260         # x1

SHADOW_BULLET = 937
CORKSCREW_PUNCH = 936
FILCH = 934
ITCHY_POLLEN = 323
CLUTCH = 997  # Yveltal 20, 1 energy (the individual never used Dark Feather 998)
TRADING_PLACES = 423  # Dunsparce 0-cost: switch with a Benched Pokémon (free pivot)

# Opposing key cards (study: marnie_7308_study.md)
CRUSTLE_WALL = 345   # Mysterious Rock Inn: all damage from opposing ex/megaEx -> 0
CORNERSTONE = 117    # Cornerstone Stance: all damage from ability-holders -> 0
MEGA_KANGA = 756     # Mega Kangaskhan ex, 300HP, takes normal damage
OPP_MUNKIDORI = (112, 139)


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
    if pokemon.id == 112 and len(pokemon.energies) >= 1:  # opposing Munkidori
        score += 300
    score += pokemon.hp
    return score


GRIMM_LINE = (Impidimp, Morgrem, Grimmsnarl)


def _score_options(obs: Observation) -> list[float]:
    """Score every option of the current selection (038 policy + study rules).

    Side effect: updates the global attack plan on MAIN selections (sub-selects
    in the same turn rely on it), exactly like 038.
    """
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
    hand_size = len(my_state.hand)
    deck_n = my_state.deckCount

    op_tool_count = 0
    for pokemon in op_state.active + op_state.bench:
        if pokemon != None:
            op_tool_count += len(pokemon.tools)

    # Wall matchup detection (study K-rules)
    wall_present = any(p != None and p.id == CRUSTLE_WALL
                       for p in op_state.active + op_state.bench)
    wall_active = (len(op_state.active) > 0 and op_state.active[0] != None
                   and op_state.active[0].id == CRUSTLE_WALL)
    # Non-wall ex in the opponent's field: the target our counters/attacks
    # should convert into prizes (Kangaskhan, Cornerstone Ogerpon, ...)
    opp_ex_nonwall = any(
        p != None and p.id != CRUSTLE_WALL
        and (card_table[p.id].ex or card_table[p.id].megaEx)
        for p in op_state.active + op_state.bench)

    # Number of ready attackers (Grimmsnarl with 2+ energies)
    ready_grimm = 0
    for card in my_state.active + my_state.bench:
        if card != None and card.id == Grimmsnarl and len(card.energies) >= 2:
            ready_grimm += 1

    stadium_id = 0
    for card in state.stadium:
        stadium_id = card.id

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
                if my_pokemon.id == Grimmsnarl:
                    energy_required = 2
                    base_damage = 180
                    base_score += 60  # bench snipe 30 is nearly always useful
                    attack_id = SHADOW_BULLET
                elif my_pokemon.id == Morgrem:
                    energy_required = 2
                    base_damage = 60
                    base_score -= 50  # would rather evolve than attack with Morgrem
                    attack_id = CORKSCREW_PUNCH
                elif my_pokemon.id == Yveltal:
                    energy_required = 1
                    base_damage = 20
                    base_score -= 30  # wall/pivot; Clutch only as a last resort
                    attack_id = CLUTCH
                elif my_pokemon.id == Budew:
                    energy_required = 0
                    base_damage = 10
                    base_score += 40  # item lock is better than Filch
                    if wall_active:
                        base_score += 150  # K7: lock blocks Jumbo Ice Cream heals
                    attack_id = ITCHY_POLLEN
                elif my_pokemon.id == Impidimp:
                    energy_required = 0
                    base_damage = 0
                    base_score = 10  # Filch: draw 1, better than nothing
                    attack_id = FILCH
                elif my_pokemon.id == Dunsparce:
                    # Trading Places: free pivot into a charged bench Grimmsnarl
                    # (the individual used it 13 times to dodge retreat costs)
                    if i == 0 and ready_grimm >= 1 and best_score < 45:
                        best_score = 45
                        plan.attacker = 0
                        plan.target = 0
                        plan.target_id = -1
                        plan.attack_id = TRADING_PLACES
                        plan.remain_hp = 999
                        plan.energy = False
                    continue
                else:
                    continue

                my_data = card_table[my_pokemon.id]

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
                        plan.target_id = -1
                        plan.attack_id = FILCH
                        plan.remain_hp = 999
                        plan.energy = False
                    continue

                for j, op_pokemon in enumerate(op_cards):
                    if op_pokemon == None:
                        continue
                    if j != 0 and not can_op_switch:
                        break
                    if op_pokemon.id == CRUSTLE_WALL and (my_data.ex or my_data.megaEx):
                        # K1: Mysterious Rock Inn nullifies our ex attacks entirely
                        continue
                    if op_pokemon.id == CORNERSTONE and len(my_data.skills) > 0:
                        # Cornerstone Stance nullifies attacks from ability holders
                        # (Grimmsnarl has Punk Up; Budew/Impidimp/Yveltal get through)
                        continue
                    damage = base_damage
                    data = card_table[op_pokemon.id]
                    if data.weakness == EnergyType.DARKNESS:
                        damage *= 2
                    elif data.resistance == EnergyType.DARKNESS:
                        damage -= 30
                    prize = 0
                    score = pokemon_score(op_pokemon)
                    if op_pokemon.hp <= damage:
                        prize = prize_count(op_pokemon)
                        # The individual gusts/attacks to KILL (207 of 234 Boss targets
                        # were killable): deny evolution lines & opposing Munkidori.
                        score += 1200
                    else:
                        score *= damage / op_pokemon.hp
                    score += base_score

                    if len(op_state.prize) <= prize:
                        score = 50000

                    if i == 0:
                        # Tank rotation: a damaged active Grimmsnarl steps back when a
                        # charged replacement waits on the bench (heal it with Adrena-Brain).
                        if (my_pokemon.id == Grimmsnarl and my_pokemon.hp <= 130
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

            # Defensive rotation when nothing can be damaged (wall matchups):
            # K1 removes wall targets from the plan, but a damaged active
            # Grimmsnarl must still step back behind a fresh one (exactly what
            # 038 achieved implicitly by planning its nullified attack).
            if (can_switch and plan.attacker <= 0
                    and plan.attack_id in (-1, FILCH, ITCHY_POLLEN)):
                act = my_cards[0]
                if act != None and act.id == Grimmsnarl and act.hp <= 180:
                    for bi, p in enumerate(my_state.bench):
                        if (p != None and p.id == Grimmsnarl
                                and len(p.energies) >= 2 and p.hp > act.hp):
                            plan.attacker = 1 + bi
                            plan.target = 0
                            plan.target_id = -1
                            plan.attack_id = SHADOW_BULLET
                            plan.remain_hp = 999
                            plan.energy = False
                            break

    def energy_score(pokemon: Pokemon, active: bool) -> int:
        """Where to attach a {D} energy.

        Replay stats (hand attach dest): Munkidori(0en) 754 >> Grimmsnarl 148 >
        Impidimp(0en) 70. Punk Up charges the Grimmsnarl line; the manual attach
        exists to switch on Adrena-Brain. Punk Up selections (ATTACH_FROM) can only
        target Marnie's Pokémon, so the Munkidori branch never interferes there.
        """
        energy_count = len(pokemon.energies)
        score = 8000
        if active:
            score += 10
        if pokemon.id == Munkidori:
            if energy_count < 1:
                score += 120  # top priority: Adrena-Brain online
            else:
                score -= 100
        elif pokemon.id == Grimmsnarl:
            score += 3
            if energy_count < 2:
                score += 100
            if ready_grimm >= 1:
                score -= 50
        elif pokemon.id == Morgrem or pokemon.id == Impidimp:
            score += 2
            if energy_count < 2:
                score += 80
            if ready_grimm >= 1:
                score -= 50
        elif pokemon.id == Yveltal:
            if energy_count < 1:
                score += 15  # enough for Clutch pivot; never charge to 3
            else:
                score -= 100
        else:  # Dunsparce line / Budew / Fezandipiti: support only
            score -= 100
        return score

    scores = []
    for o in select.option:
        score = 0
        if o.type == OptionType.NUMBER:
            score = o.number
        elif o.type == OptionType.YES:
            score = 1  # Punk Up / Adrena-Brain / Run Away Draw / go first: default yes
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
                            score += 20
                        elif card.id == Morgrem:
                            score += 10
                        elif card.id == Yveltal:
                            score += 6
                        elif card.id == Impidimp:
                            score += 5
                        elif card.id == Budew:
                            score += 4  # free retreat pivot
                        elif card.id == Munkidori:
                            score += 2  # keep it benched moving counters
                        elif card.id == Dunsparce or card.id == Dudunsparce:
                            score += 1
                        if wall_present and isinstance(card, Pokemon):
                            # K6: protect the chip engine; prefer tooled (Cape) walls
                            if card.id == Munkidori:
                                score -= 40
                            score += len(card.tools) * 10
                    else:
                        if o.index == plan.target - 1:
                            score += 100
                elif context == SelectContext.SETUP_ACTIVE_POKEMON:
                    # Pairwise preferences from 371 replays: Budew ~ Yveltal > Impidimp
                    # > Dunsparce > Munkidori > Fezandipiti
                    if card.id == Budew:
                        score = 6  # item lock opener
                    elif card.id == Yveltal:
                        score = 5  # 110 HP non-ex wall
                    elif card.id == Impidimp:
                        score = 4
                    elif card.id == Dunsparce:
                        score = 3
                    elif card.id == Munkidori:
                        score = 2
                    elif card.id == Fezandipiti:
                        score = 1
                elif context == SelectContext.SETUP_BENCH_POKEMON:
                    # The individual benches only the engine pieces at setup and keeps
                    # Budew/Yveltal/Fezandipiti in hand (bench counts 117/79/73 vs 6/5/0).
                    if card.id == Munkidori:
                        score = 5
                    elif card.id == Impidimp:
                        score = 4
                    elif card.id == Dunsparce:
                        score = 3
                    elif card.id == Budew or card.id == Yveltal:
                        score = -1
                    elif card.id == Fezandipiti:
                        score = -2
                elif context in (SelectContext.TO_HAND, SelectContext.TO_BENCH, SelectContext.TO_FIELD):
                    score = 200 - hand_counts[card.id] * 100
                    if card.id == Impidimp:
                        need = 3 - (field_counts[Impidimp] + field_counts[Morgrem] + field_counts[Grimmsnarl])
                        score += 50 * need
                    elif card.id == Grimmsnarl:
                        if field_counts[Impidimp] + field_counts[Morgrem] >= 1:
                            score += 80
                        else:
                            score -= 10
                    elif card.id == Morgrem:
                        if field_counts[Impidimp] >= 1 and hand_counts[Rare_Candy] == 0:
                            score += 40
                        else:
                            score -= 20
                    elif card.id == Munkidori:
                        if field_counts[card.id] >= 2:
                            score -= 100
                        else:
                            score += 45
                    elif card.id == Dunsparce:
                        if field_counts[Dunsparce] + field_counts[Dudunsparce] >= 1:
                            score -= 50
                        else:
                            score += 25
                    elif card.id == Dudunsparce:
                        if field_counts[Dunsparce] >= 1 and field_counts[Dudunsparce] == 0:
                            score += 30
                        else:
                            score -= 50
                    elif card.id == Fezandipiti:
                        if field_counts[card.id] >= 1:
                            score -= 100
                        else:
                            score += 10
                    elif card.id == Budew:
                        score -= 20  # early-game only
                    elif card.id == Rare_Candy:
                        if field_counts[Impidimp] >= 1 and hand_counts[Grimmsnarl] >= 1:
                            score += 70
                        else:
                            score += 25
                    elif card.id == Basic_Dark_Energy:
                        if not state.energyAttached and hand_counts[Basic_Dark_Energy] == 0:
                            score += 30
                        else:
                            score -= 1
                    elif card.id == Boss_Orders:
                        score += 15
                elif context == SelectContext.ATTACH_FROM:
                    score = energy_score(card, o.area == AreaType.ACTIVE)
                elif context == SelectContext.DAMAGE:
                    # K2a: attack-based damage placement (Shadow Bullet bench snipe).
                    # Mysterious Rock Inn blocks it — never waste the snipe on the wall.
                    if o.playerIndex != my_index:
                        if card.id in (CRUSTLE_WALL, CORNERSTONE):
                            # the snipe comes from Grimmsnarl (ex + ability): blocked
                            score = -50
                        elif card.hp <= 30:
                            score = 5000 + pokemon_score(card)
                        elif card.hp <= 60:
                            score = 400 + pokemon_score(card)
                        else:
                            score = pokemon_score(card)
                    else:
                        score = -100  # never damage my own side if avoidable
                elif context == SelectContext.DAMAGE_COUNTER:
                    # K2b: counter placement (Adrena-Brain destination) bypasses walls.
                    if o.playerIndex != my_index:
                        card_data = card_table[card.id]
                        if card.hp <= 30:
                            score = 5000 + pokemon_score(card)
                        elif (wall_present and card.id != CRUSTLE_WALL
                                and (card_data.ex or card_data.megaEx)):
                            # chip the multi-prize target our attacks/counters can
                            # actually convert (Kangaskhan, Cornerstone Ogerpon);
                            # the stacked wall itself is not a prize source
                            score = 3600 + (card_data.hp - card.hp)
                        elif card.id in OPP_MUNKIDORI and energy_count >= 1:
                            # M1: break the opposing heal+chip engine first
                            # (mirror & Crustle decks that splash Munkidori)
                            score = 2200 + pokemon_score(card)
                        elif card.id == CRUSTLE_WALL:
                            if opp_ex_nonwall:
                                score = 300 - card.hp  # wall chip: last resort
                            else:
                                score = 3500 - card.hp  # lone wall: concentrate
                        elif card.hp <= 60:
                            # kill it over two moves: the individual piles counters on
                            # low-HP evolving basics (Abra 431 vs Kadabra 86 vs Alakazam 70)
                            score = 400 + pokemon_score(card)
                        else:
                            score = pokemon_score(card)
                    else:
                        score = -100  # never place counters on my own side if avoidable
                elif context == SelectContext.REMOVE_DAMAGE_COUNTER:
                    # Adrena-Brain source: prefer my own most damaged Pokémon
                    if o.playerIndex == my_index:
                        data = card_table[card.id]
                        score = (data.hp - card.hp) * 10
                        if card.id == Grimmsnarl:
                            score += 30  # protect the attacker first
                    else:
                        score = -100
                elif context == SelectContext.EVOLVES_TO:
                    if card.id == Grimmsnarl:
                        score = 100
                    elif card.id == Morgrem:
                        score = 50
                    elif card.id == Dudunsparce:
                        score = 40
        elif o.type == OptionType.TOOL_CARD:
            # Tool Scrapper: discard opponent tools, never our own (Hero's Cape)
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
                    if field_counts[card.id] >= 2:
                        score = -1
                    elif field_counts[card.id] == 1:
                        score = 15000 if bench_used <= 3 else -1
                    else:
                        score = 18000
                elif card.id == Dunsparce:
                    if field_counts[Dunsparce] + field_counts[Dudunsparce] >= 1:
                        score = -1
                    else:
                        score = 19000
                elif card.id == Fezandipiti:
                    if field_counts[card.id] >= 1 or bench_used > 3:
                        score = -1
                    else:
                        score = 14000
                elif card.id == Yveltal:
                    if field_counts[card.id] >= 1 or bench_used > 3:
                        score = -1
                    else:
                        score = 12000
                elif card.id == Budew:
                    if field_counts[card.id] >= 1 or bench_used > 3:
                        score = -1
                    else:
                        score = 11000
            else:
                score = 10000
                if card.id == Rare_Candy:
                    score = 12000  # engine only offers it when playable (Impidimp + Grimmsnarl in hand)
                elif card.id == Tool_Scrapper:
                    score = 11000 if op_tool_count >= 1 else -1
                elif card.id == Boss_Orders:
                    # Kill tool: 207/234 of the individual's gust targets died that turn
                    if plan.target >= 1 and plan.remain_hp <= 0:
                        score = 3150
                    elif wall_present and plan.target >= 1 and plan.target_id == MEGA_KANGA:
                        # K5: drag the 3-prize Kangaskhan past the wall (180x2 kills it)
                        score = 3120
                    elif plan.target >= 1:
                        score = 400  # tank drag-up: only when nothing better
                    else:
                        score = -1
                elif card.id == Lillie_Determination:
                    # K9: don't draw ourselves to death in wall/stall matchups
                    # (ungated it cost ~4pt vs Alakazam via lost draw power)
                    draw_n = 8 if my_prize >= 6 else 6
                    if wall_present and deck_n + hand_size - 1 - draw_n < 4:
                        score = -1
                    else:
                        score = 3100
                elif card.id == Xerosic:
                    # Pairwise-preferred over every other supporter (291:76);
                    # played down to opp hand = 4 in the replays, never below.
                    if op_state.handCount >= 4:
                        score = 3300
                    else:
                        score = 500
                elif card.id == Dawn:
                    score = -1 if (wall_present and deck_n <= 6) else 2900  # K9
                elif card.id == Spikemuth_Gym:
                    if stadium_id == Spikemuth_Gym:
                        score = -1
                    else:
                        score = 5000
                elif card.id == Risky_Ruins:
                    if stadium_id == Spikemuth_Gym or stadium_id == Risky_Ruins:
                        score = -1
                    else:
                        score = 4800
        elif o.type == OptionType.ATTACH:
            card = get_card(obs, AreaType.HAND, o.index, my_index)
            pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
            if card.id == Heros_Cape:
                # Cape belongs on the Grimmsnarl line (evolution keeps the tool:
                # study 84750337 t2 / 84751391 t1). Never waste it on support mons.
                if pokemon.id == Grimmsnarl:
                    score = 7300  # 320 -> 420 HP tank
                elif pokemon.id in (Impidimp, Morgrem):
                    score = 7100  # pre-evolution: becomes the 420 tank later
                elif pokemon.id == Fezandipiti:
                    score = 7000
                else:
                    score = 400  # hold it for the Grimmsnarl line
            else:
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
            if card != None and card.id == Dudunsparce:
                score = 8000  # evolve the attacker line first
        elif o.type == OptionType.ABILITY:
            score = 30000  # Adrena-Brain / Run Away Draw / Flip the Script: default use
            card = get_card(obs, o.area, o.index, my_index)
            if card != None and card.id == Dudunsparce:
                bench_n = sum(1 for p in my_state.bench if p != None)
                if o.area == AreaType.ACTIVE and bench_n == 0:
                    score = -1  # never shuffle away our only Pokémon
                elif wall_present and deck_n <= 4:
                    score = -1  # K9: stop milling when the deck is low
                else:
                    score = 25000  # use after other abilities
            elif card != None and card.id == Fezandipiti and wall_present and deck_n <= 5:
                score = -1  # K9
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


def _choose_from_scores(select, scores: list[float]) -> list[int]:
    desc_indices = [i for i, _ in sorted(enumerate(scores), key=lambda x: x[1], reverse=True)]
    chosen = desc_indices[:select.maxCount]
    # For "up to N" selections, drop deliberately-avoided (negative) picks
    while len(chosen) > select.minCount and scores[chosen[-1]] < 0:
        chosen.pop()
    return chosen


def policy_choose(obs: Observation) -> list[int]:
    return _choose_from_scores(obs.select, _score_options(obs))


# ---------------------------------------------------------------------------
# Lethal-only determinized search layer (MAIN only), ported from 036/007.
# Runs only when our remaining prizes are <=3. Candidate MAIN options are
# ranked by the heuristic; a candidate overrides the heuristic choice only if
# EVERY determinization sample ends the greedy rollout of our own turn with
# the game already won. No board evaluation is involved, so a bad eval can
# never overrule the tuned heuristic. Any failure falls back to the heuristic.

SEARCH_CANDIDATES = 8
SEARCH_SAMPLES = 8           # EXP-036: draw-order dependent lethal lines need
                             # more samples to keep the false-positive rate ~0;
                             # Marnie (Dudunsparce/Lillie mid-turn draws) needed
                             # 8 (5 gave 3 losing overrides / 300 games, EXP-043)
SEARCH_MOVE_BUDGET = 1.5     # seconds per MAIN decision
SEARCH_GAME_BUDGET = 450.0   # total seconds of search per game (600s limit)
ROLLOUT_STEP_CAP = 80

PLACEHOLDER_MON = 1072  # basic Pokémon used for unknown opponent cards
if not (card_table.get(PLACEHOLDER_MON) and card_table[PLACEHOLDER_MON].basic):
    PLACEHOLDER_MON = next(c.cardId for c in all_card if c.basic)

my_deck_counts = Counter(my_deck)

_search_time_used = 0.0


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
    """Greedy rollout with the heuristic until our turn ends (or cap/deadline)."""
    for _ in range(ROLLOUT_STEP_CAP):
        obs = state.observation
        if obs.current.result != -1 or obs.current.turn != root_turn:
            break
        if time.perf_counter() > deadline:
            break
        try:
            sel = policy_choose(obs)
            if not sel and obs.select.minCount > 0:
                sel = _fallback_selection(obs.select)
        except Exception:
            sel = _fallback_selection(obs.select)
        state = search_step(state.searchId, sel)
    return state.observation


def _lethal_search(obs: Observation, heuristic_scores: list[float]) -> list[int] | None:
    """Override the MAIN choice only when a line certainly wins THIS turn."""
    global _search_time_used

    select = obs.select
    if select.minCount != 1 or select.maxCount != 1 or len(select.option) < 2:
        return None

    my_index = obs.current.yourIndex
    me = obs.current.players[my_index]
    if len(me.prize) > 3:  # cannot finish this turn: skip search entirely
        return None

    # Reset the per-game search budget at the start of each game
    # (the module stays loaded across games in local arena workers).
    if obs.current.turn <= 2:
        _search_time_used = 0.0
    if _search_time_used > SEARCH_GAME_BUDGET:
        return None

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
                except Exception:
                    always_wins[idx] = False
    finally:
        try:
            search_end()
        except Exception:
            pass
        _search_time_used += time.perf_counter() - t0

    winners = [i for i in candidates if always_wins[i] and tried[i] == SEARCH_SAMPLES]
    if not winners:
        return None
    return [max(winners, key=lambda i: heuristic_scores[i])]


def agent(obs_dict: dict) -> list[int]:
    obs = to_observation_class(obs_dict)
    if obs.select == None:
        return my_deck

    global plan, pre_turn

    if obs.select.context == SelectContext.MAIN and obs.current.turn >= 2:
        # Score once (this also sets the attack plan for the real trajectory),
        # then let the search override only the MAIN choice.
        scores = _score_options(obs)
        # Snapshot the plan VALUES (not the reference): rollouts inside the
        # search replay MAINs of the same turn and mutate the same global plan
        # object in place, which would otherwise corrupt the sub-selections
        # (Boss gust target, energy attach) that follow the real MAIN choice.
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
        except Exception:
            choice = None
        plan, pre_turn = snap, saved_turn
        if choice is not None:
            return choice
        return _choose_from_scores(obs.select, scores)

    return policy_choose(obs)
