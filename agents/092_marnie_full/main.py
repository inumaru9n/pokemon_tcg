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

Base: agents/089_marnie_ml (EXP-089, verbatim copy; helpers renamed feat_092/
model_092 to avoid module collisions when 089 and 090 are loaded side by side).
Added: the lethal search layer ported from agents/043_marnie_full (EXP-043),
including the mandatory plan VALUE-snapshot isolation (038-family policies
mutate the global plan in place; search rollouts replaying MAINs of the same
turn would otherwise corrupt the real-trajectory sub-selections: -16pt in the
wall matchup, EXP-043). SEARCH_SAMPLES=8 (Marnie's in-turn draw lines need 8,
EXP-043). _search_time_used is reset at the top of agent() each game
(INSIGHTS: latent 045 accumulation bug). F092_LETHAL=off disables the layer;
with search unavailable the agent naturally falls back to 089 behavior.

--- Original 089 header below ---

089_marnie_ml — EXP-089 (I-114): EXP-057-style direct action-class policy learning
applied to the Marnie's Grimmsnarl deck (sig 1ec0f47981, teacher = Luca, the
originator team of the recipe).

Structure: 082_marnie_1ec0 rule policy as the backbone (EXP-061: ML must sit on a
competitive rule spine, MAIN only) + a LightGBM multiclass layer that predicts the
teacher's action class on MAIN decisions (turn>=2, single-pick). Within the chosen
class, the concrete option is picked by the 082 policy scores. With no model file
or F092_ML=off, behavior is identical to 082.

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
Fezandipiti_ex = 140       # opp support ex: Boss-farm target (R13a)

# EXP-092 ablation flags: F092_OFF=P,R,S,D,B（P=PunkUp配分 R=リトリート賃 S=サーチ表
# D=DISCARD表 B=Boss新モード。offで082相当へフォールバック）
_F092_OFF = set(x.strip().upper() for x in os.environ.get("F092_OFF", "").split(",") if x.strip())
ABL_P = "P" not in _F092_OFF
ABL_R = "R" not in _F092_OFF
ABL_S = "S" not in _F092_OFF
ABL_D = "D" not in _F092_OFF
ABL_B = "B" not in _F092_OFF

# R25: discard-rate-per-offer table from 13,556 offered cards under opponent
# Xerosic/Hand Trimmer (knowledge/study/marnie_1ec0_study.md). Higher = discard.
DISCARD_RATE = {
    Buddy_Poffin: 95, Impidimp: 91, Poke_Pad: 82, Morgrem: 81, Snorunt: 78,
    Pokegear: 78, Munkidori: 76, Dawn: 72, Rare_Candy: 70, Spikemuth_Gym: 66,
    Tool_Scrapper: 61, Night_Stretcher: 42, Boss_Orders: 39,
    Basic_Dark_Energy: 34, Froslass: 34, Petrel: 33, Lillie_Determination: 33,
    Grimmsnarl: 19, Unfair_Stamp: 4,
}

GRIMM_LINE = (Impidimp, Morgrem, Grimmsnarl)
SNOW_LINE = (Snorunt, Froslass)


class AttackPlan:
    attacker = -1
    target = -1
    target_id = -1
    attack_id = -1
    remain_hp = -1
    energy = False
    retreat_money = False  # R8: this turn's hand attach funds the active's retreat


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
    """092ルール方策: knowledge/study/marnie_1ec0_study.md（R1..R27）を正とする再構成。

    082（EXP-082の軽量頻度レプリカ）からの主変更（studyの差分優先順）:
    DISCARD捨て率表(R25) / Punk Up配分(R7) / Froslass2体目(R5) / Boss 3モード(R13) /
    リトリート賃の意図的貼り(R8) / Petrel等サーチのeffect別分岐(R17,R20-22)。
    MAINでは副作用としてplanを更新する（082と同じ契約）。
    """
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
    # wall DECK detection (Dwebble/Crustle/Mega Kangaskhan seen anywhere): the
    # Luca-calibrated search tables help in wall games (+6.5pt vs 085 CRN) but
    # hurt the Marnie matchup (-7.5pt vs 082 CRN) -> gate them on the deck type
    # (EXP-044: matchup tech must be gated by archetype-specific card ids)
    opp_wall_deck = any(
        p is not None and p.id in (344, Crustle_Wall, Mega_Kangaskhan)
        for p in op_state.active + op_state.bench)
    if not opp_wall_deck:
        opp_wall_deck = any(c.id in (344, Crustle_Wall, Mega_Kangaskhan)
                            for c in op_state.discard)
    # R13a: benched Fezandipiti ex with no energy = Boss farming target
    fez_bench = any(p is not None and p.id == Fezandipiti_ex and len(p.energies) == 0
                    for p in op_state.bench)

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

        # R8: a bench attacker is reachable without a legal RETREAT option when a
        # single hand attach to the active covers its retreat cost (the individual
        # attaches "retreat money" then pivots into the charged Grimmsnarl:
        # ep 87365211 t5/t7)
        retreat_fundable = False
        act0 = my_state.active[0] if my_state.active else None
        if (not can_switch and act0 is not None and hand_counts[Basic_Dark_Energy] >= 1
                and not state.energyAttached):
            data0 = card_table.get(act0.id)
            if ABL_R and data0 is not None and len(act0.energies) + 1 >= data0.retreatCost:
                # gate (dev mining): never divert the hand attach to retreat money
                # when the active itself can attack this turn
                act_en = len(act0.energies)
                active_can_hit = (
                    (act0.id == Grimmsnarl and act_en + 1 >= 2)
                    or (act0.id == Morgrem and act_en + 1 >= 2)
                    or act0.id == Impidimp)
                if not active_can_hit:
                    retreat_fundable = True

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
                via_retreat_money = False
                if i != 0 and not can_switch:
                    if not retreat_fundable:
                        break
                    via_retreat_money = True
                energy_required = 0
                base_damage = 0
                base_score = 0
                attack_id = -1
                attacker_is_ex = False
                if my_pokemon.id == Grimmsnarl:
                    energy_required = 2
                    base_damage = 180
                    base_score += 60  # bench snipe 30 is nearly always useful (R12)
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
                    # Munkidori / Snorunt / Froslass never attack (R10, 0/8,611)
                    continue
                if attack_id == FILCH and my_state.deckCount <= 2:
                    continue  # R11 deck guard: never draw ourselves to death

                more_energy = False
                energy_count = len(my_pokemon.energies)
                if energy_count < energy_required:
                    if via_retreat_money:
                        continue  # the one hand attach is spent on retreat money
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
                        plan.retreat_money = False
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
                        # R14: Morgrem Corkscrew 60 is the wall grinder (139 uses)
                        wall_breaker_bonus = 250
                    if wall_active and j == 0 and my_pokemon.id == Grimmsnarl:
                        # R14: Shadow Bullet into the wall active still delivers the
                        # 30 bench snipe (343 uses); keep it viable but below a
                        # Boss-enabled bench kill.
                        wall_breaker_bonus = 40
                    score = pokemon_score(op_pokemon)
                    if op_pokemon.hp <= damage:
                        prize = prize_count(op_pokemon)
                        # Gust/attack to KILL (R13b)
                        score += 1200
                    else:
                        score *= damage / op_pokemon.hp
                    score += base_score + wall_breaker_bonus

                    if len(op_state.prize) <= prize:
                        score = 50000

                    if i == 0:
                        # R26 tank rotation: a damaged active Grimmsnarl steps back
                        # when a charged replacement waits on the bench
                        if (my_pokemon.id == Grimmsnarl and my_pokemon.hp <= 140
                                and any(p is not None and p.id == Grimmsnarl and len(p.energies) >= 2
                                        for p in my_state.bench)):
                            score -= 50
                        else:
                            score += 220
                    elif via_retreat_money:
                        score -= 40  # costs the hand attach + retreat: slight tax
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
                        plan.retreat_money = via_retreat_money

    def energy_score(pokemon: Pokemon, active: bool) -> int:
        """Manual hand attach destination (R6).

        Replay stats (hand attach dest, n=8,953): Munkidori(en0) 5,242 >>
        Impidimp(en0) 1,188 > Morgrem(en0) 501 > Froslass(en0) 456 >
        Snorunt(en0) 411 > Grimmsnarl(en1) 222. Punk Up charges the Grimmsnarl
        line; the manual attach switches on Adrena-Brain, funds retreats, then
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
                score += 12  # retreat money when stuck in front
            else:
                return -5
        else:
            return -5
        return score

    def punkup_score(pokemon: Pokemon) -> int:
        """R7: Punk Up distribution (never decline; 13,565/13,565 taken).

        Fill order: evolved Grimmsnarl to 2 (Shadow Bullet online) > next
        Impidimp/Morgrem to 2 (pre-charge the next attacker) > Grimmsnarl
        3rd-5th (retreat currency + rotation stamina; en2:604 en3:438 en4:302) >
        line 3rd. Munkidori/Snow line never get Punk Up energy (0 in 13,565).
        """
        en = len(pokemon.energies)
        if pokemon.id == Grimmsnarl:
            if en < 2:
                return 300 - en
            if en < 5:
                return 80 - 5 * en   # 3rd-5th: after the next line is charged
            return 2
        if pokemon.id == Morgrem:
            if en < 2:
                return 210 - en      # next attacker first (dev mining: Morgrem > Imp)
            if en < 3:
                return 100
            return 1
        if pokemon.id == Impidimp:
            if en < 2:
                return 200 - en
            if en < 3:
                return 95
            return 1
        if pokemon.id == Munkidori:
            return 20 if en < 1 else -5
        return -5

    scores = []
    for o in select.option:
        score = 0
        if o.type == OptionType.NUMBER:
            score = o.number
        elif o.type == OptionType.YES:
            score = 1  # Punk Up / Adrena-Brain / Spikemuth search / go first: default yes
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
                            # R14: in wall games promote cheap non-ex bodies instead
                            # of the grass-weak 2-prize Grimmsnarl
                            score += 6 if wall_active else 20
                        elif card.id == Morgrem:
                            score += 12 if wall_active else 10
                        elif card.id == Impidimp:
                            score += 8  # most-promoted cheap body after a KO
                        elif card.id == Froslass:
                            score += 4  # 1-prize wall that keeps chipping
                        elif card.id == Snorunt:
                            score += 3
                        elif card.id == Munkidori:
                            score += 2  # keep it benched moving counters
                    else:
                        # Boss gust target (R13, n=1,203)
                        if o.index == plan.target - 1:
                            score += 100
                        if ABL_B:
                            if card.id == Fezandipiti_ex and energy_count == 0:
                                score += 90   # (a) farm the 2-prize sitting duck (497)
                            elif wall_in_play and card.id == 344:  # Dwebble
                                score += 85   # (c) cut the next wall's supply line (52)
                            elif wall_in_play and card.id == Mega_Kangaskhan:
                                score += 80 if ready_grimm >= 1 else 30  # 3-prize burst
                            elif card.id == Munkidori and energy_count >= 1:
                                score += 70   # (b) kill the mirror engine
                            elif isinstance(card, Pokemon) and card.hp <= 70:
                                score += 60   # killable evolver
                elif context == SelectContext.SETUP_ACTIVE_POKEMON:
                    # R2 setup active: Impidimp 1235 > Munkidori 635 > Snorunt 474
                    if card.id == Impidimp:
                        score = 6
                    elif card.id == Munkidori:
                        score = 5
                    elif card.id == Snorunt:
                        score = 4
                elif context == SelectContext.SETUP_BENCH_POKEMON:
                    # R2 setup bench: Munkidori 896 > Snorunt 321 > Impidimp 214
                    if card.id == Munkidori:
                        score = 5
                    elif card.id == Snorunt:
                        score = 4
                    elif card.id == Impidimp:
                        score = 3
                elif context in (SelectContext.TO_HAND, SelectContext.TO_BENCH, SelectContext.TO_FIELD):
                    eff_id = select.effect.id if select.effect is not None else -1
                    if not ABL_S or not opp_wall_deck:
                        eff_id = -1
                    score = 200 - hand_counts[card.id] * 100
                    if eff_id == Petrel:
                        # R17: Stamp 1093 top fetch; rest follows the generic table
                        # ordering that already matched best (round-2 order-trap
                        # lesson: per-effect flips regressed)
                        if card.id == Unfair_Stamp:
                            score += 90
                        elif card.id == Tool_Scrapper:
                            score += 80 if op_tool_count >= 1 else 10
                        elif card.id == Night_Stretcher:
                            has_dark_in_discard = any(c.id == Basic_Dark_Energy for c in my_state.discard)
                            score += 55 if has_dark_in_discard else 8
                        elif card.id == Poke_Pad:
                            score += 45
                        elif card.id == Lillie_Determination:
                            score += 40
                        elif card.id == Spikemuth_Gym:
                            score += 35 if stadium_id != Spikemuth_Gym else 5
                        elif card.id == Rare_Candy:
                            if field_counts[Impidimp] >= 1 and hand_counts[Grimmsnarl] >= 1:
                                score += 65
                            else:
                                score += 30
                        elif card.id == Boss_Orders:
                            if (wall_in_play and hand_counts[Boss_Orders] == 0 and ready_grimm >= 1
                                    and any(p is not None and p.id == Mega_Kangaskhan for p in op_state.bench)):
                                score += 85  # R13c: fetch the gust for the Kang burst
                            else:
                                score += 20
                        elif card.id == Buddy_Poffin:
                            score += 15
                        else:
                            score += 5
                    elif eff_id == Pokegear:
                        # R22: Lillie 424 ~ Petrel 422 > Boss 185 > Dawn 57
                        if card.id == Lillie_Determination:
                            score += 45 if len(my_state.hand) <= 4 else 38
                        elif card.id == Petrel:
                            score += 42
                        elif card.id == Boss_Orders:
                            score += 44 if (plan.target >= 1 and plan.remain_hp <= 0) else 25
                        elif card.id == Dawn:
                            score += 5
                    elif eff_id == Poke_Pad:
                        # R5 + dev mining round2: Froslass line first (x115 vs x15
                        # direction), Munkidori only while the board lacks it
                        if card.id == Froslass:
                            if field_counts[Froslass] + hand_counts[Froslass] < 2:
                                score += 48
                            else:
                                score -= 30
                        elif card.id == Munkidori:
                            if field_counts[Munkidori] == 0:
                                score += 45
                            elif field_counts[Munkidori] < 3:
                                score += 38
                            else:
                                score -= 100
                        elif card.id == Snorunt:
                            score += 30 if snow_count < 2 else -50
                        elif card.id == Impidimp:
                            need = 3 - (field_counts[Impidimp] + field_counts[Morgrem] + field_counts[Grimmsnarl])
                            score += 13 * max(0, need)
                        elif card.id == Morgrem:
                            if field_counts[Impidimp] >= 1 and hand_counts[Rare_Candy] == 0:
                                score += 10
                            else:
                                score -= 20
                    elif eff_id == Night_Stretcher:
                        # R21: energy 2528/4100 >> Impidimp 487 > Snorunt 281 > Munkidori
                        if card.id == Basic_Dark_Energy:
                            score += 60 if hand_counts[Basic_Dark_Energy] == 0 else 20
                        elif card.id == Impidimp:
                            need = 3 - (field_counts[Impidimp] + field_counts[Morgrem] + field_counts[Grimmsnarl])
                            score += 35 if need > 0 else 5
                        elif card.id == Snorunt:
                            score += 25 if snow_count < 2 else 2
                        elif card.id == Munkidori:
                            score += 20 if field_counts[Munkidori] < 3 else 2
                        elif card.id == Grimmsnarl:
                            score += 15
                        else:
                            score += 5
                    else:
                        # Spikemuth Gym search (R20) / Poké Pad (R5) / setup pulls:
                        # shared line-need table
                        if card.id == Impidimp:
                            need = 3 - (field_counts[Impidimp] + field_counts[Morgrem] + field_counts[Grimmsnarl])
                            score += 50 * need
                        elif card.id == Grimmsnarl:
                            # Gym top fetch (3,231): stepwise line logic (dev mining)
                            if field_counts[Morgrem] >= 1 or (
                                    field_counts[Impidimp] >= 1 and hand_counts[Rare_Candy] >= 1):
                                score += 80
                            elif field_counts[Impidimp] >= 1:
                                score += 35
                            else:
                                score -= 10
                        elif card.id == Morgrem:
                            if (field_counts[Impidimp] >= 1 and field_counts[Morgrem] == 0
                                    and hand_counts[Rare_Candy] == 0):
                                score += 70
                            elif field_counts[Impidimp] >= 1:
                                score += 40
                            else:
                                score -= 20
                        elif card.id == Munkidori:
                            if field_counts[card.id] >= 3:
                                score -= 100
                            else:
                                score += 45
                        elif card.id == Froslass:
                            # R5: the individual runs TWO Froslass (Pad fetch #2,
                            # 1,713) to double the passive chip
                            if field_counts[Snorunt] > field_counts[Froslass]:
                                score += 44
                            else:
                                score -= 30
                        elif card.id == Snorunt:
                            if snow_count < 2:
                                score += 25
                            else:
                                score -= 50
                        elif card.id == Unfair_Stamp:
                            score += 90
                        elif card.id == Tool_Scrapper:
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
                            if hand_counts[Basic_Dark_Energy] == 0:
                                score += 60
                            else:
                                score -= 1
                        elif card.id == Lillie_Determination:
                            score += 25
                        elif card.id == Petrel:
                            score += 20
                        elif card.id == Boss_Orders:
                            if (wall_in_play and hand_counts[Boss_Orders] == 0 and ready_grimm >= 1
                                    and any(p is not None and p.id == Mega_Kangaskhan for p in op_state.bench)):
                                score += 85
                            else:
                                score += 15
                        elif card.id == Spikemuth_Gym:
                            score += 10 if stadium_id != Spikemuth_Gym else -20
                elif context == SelectContext.ATTACH_FROM:
                    # Punk Up destination (R7); manual attach never uses this ctx
                    if ABL_P:
                        score = punkup_score(card)
                    else:
                        score = energy_score(card, o.area == AreaType.ACTIVE)
                elif context == SelectContext.DISCARD:
                    # R25: opponent Xerosic/Trimmer discard response. Discard rate
                    # per offer (n=13,556): dump searchable line bodies and items,
                    # keep Stamp/Grimmsnarl/draw engines/energy. Higher score =
                    # discard first (min=max selects take the top scores).
                    if ABL_D and o.area == AreaType.HAND and o.playerIndex == my_index:
                        score = DISCARD_RATE.get(card.id, 50)
                        if card.id == Basic_Dark_Energy and hand_counts[Basic_Dark_Energy] >= 3:
                            score = 65  # surplus energy is droppable
                        if card.id == Grimmsnarl and hand_counts[Grimmsnarl] >= 2:
                            score = 55  # keep one, the second is Gym-refetchable
                elif context == SelectContext.DAMAGE_COUNTER or context == SelectContext.DAMAGE:
                    # Shadow Bullet snipe / Adrena-Brain destination (R12/R24)
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
                            if context == SelectContext.DAMAGE_COUNTER and card.id == Crustle_Wall:
                                score += 600  # R24b: stack the wall at ANY hp
                    else:
                        score = -100  # never place counters on my own side if avoidable
                elif context == SelectContext.REMOVE_DAMAGE_COUNTER:
                    # R23 Adrena source: own Munkidori 7,330 > Grimmsnarl 3,497
                    # (heal the damaged tank/engine while making ammo)
                    if o.playerIndex == my_index:
                        data = card_table[card.id]
                        score = (data.hp - card.hp) * 10
                        if card.id == Munkidori:
                            score += 50
                        elif card.id == Grimmsnarl:
                            score += 30
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
            # Tool Scrapper: discard opponent tools, never our own (R15)
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
                    # R4: up to 3 Munkidori
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
                    # R19: disruption bomb, dropped before the supporter for the
                    # turn (fired at opp hand median 8; 12 cards = 45% of plays)
                    if op_state.handCount >= 5:
                        score = 3400
                    elif len(my_state.hand) <= 4:
                        score = 3000
                    else:
                        score = 300
                elif card.id == Pokegear:
                    score = 10500
                elif card.id == Boss_Orders:
                    # R13: (b) kill > (c) wall-game Kang burst > (a) Fez farming >
                    # generic drag-up
                    if plan.target >= 1 and plan.remain_hp <= 0:
                        score = 3250
                    elif plan.target >= 1 and plan.target_id == Mega_Kangaskhan:
                        score = 3200
                    elif ABL_B and fez_bench and ready_grimm >= 1:
                        score = 3075  # farm Fez ex: 180x2 for 2 prizes + tempo
                    elif ABL_B and wall_in_play and any(p is not None and p.id == 344 for p in op_state.bench) \
                            and ready_grimm >= 1 and not wall_active:
                        score = 3070  # gust Dwebble: deny the next wall
                    elif plan.target >= 1:
                        score = 400  # drag-up: only when nothing better
                    else:
                        score = -1
                elif card.id == Petrel:
                    # R16: Petrel 3,604 ~ Lillie 3,390; Lillie preferred on small hands
                    if len(my_state.hand) >= 5:
                        score = 3150
                    else:
                        score = 3050
                elif card.id == Lillie_Determination:
                    # R18 deck guard: two of the 7308 study losses were deck-outs
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
                if plan.attacker >= 1 and plan.retreat_money:
                    score = 8400  # R8: retreat money enables the pivot-burst
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
            score = 30000  # Adrena-Brain / Spikemuth search: default use
        elif o.type == OptionType.RETREAT:
            if plan.attacker >= 1 and not plan.retreat_money:
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
    """092ルール方策の最終選択（フォールバック層）。"""
    select = obs.select
    scores = policy_scores(obs)
    desc_indices = [i for i, _ in sorted(enumerate(scores), key=lambda x: x[1], reverse=True)]
    chosen = desc_indices[:select.maxCount]
    # For "up to N" selections, drop deliberately-avoided (negative) picks
    while len(chosen) > select.minCount and scores[chosen[-1]] < 0:
        chosen.pop()
    return chosen


# ---- EXP-089: 選択クラス予測モデル（GBT、EXP-057定式化のMarnie適用） ----
# ロード順: lightgbm+model_092.txt（ローカル高速）→ model_092.npz（純Python、Kaggle用。
# argmax一致検証済み）→ どちらも無ければ082ルールへ自然フォールバック
try:  # Kaggle評価環境はexecロードのため __file__ が無い
    _AGENT_DIR_092 = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _AGENT_DIR_092 = ("/kaggle_simulations/agent"
                      if os.path.exists("/kaggle_simulations/agent/main.py")
                      else os.getcwd())

_M092 = None
if os.environ.get("F092_ML", "on") != "off":
    try:
        try:
            import feat_092 as _ft092
        except ImportError:
            import importlib.util as _ilu092

            _fp = os.path.join(_AGENT_DIR_092, "feat_092.py")
            _sp = _ilu092.spec_from_file_location("feat_092", _fp)
            _ft092 = _ilu092.module_from_spec(_sp)
            _sp.loader.exec_module(_ft092)
        try:
            import lightgbm as _lgb092

            _mp = os.path.join(_AGENT_DIR_092, "model_092.txt")
            if not os.path.exists(_mp):
                raise FileNotFoundError(_mp)
            _M092 = {"mode": "lgb", "bst": _lgb092.Booster(model_file=_mp),
                     "ft": _ft092}
        except Exception:  # noqa: BLE001  lightgbm無し → npz純Python推論
            import numpy as _np092

            _z = _np092.load(os.path.join(_AGENT_DIR_092, "model_092.npz"))
            _M092 = {"mode": "npz", "ft": _ft092,
                     "feat": _z["feat"], "thr": _z["thr"],
                     "left": _z["left"], "right": _z["right"],
                     "val": _z["val"], "roots": _z["roots"],
                     "tcls": _z["tree_cls"],
                     "ncls": int(_z["num_class"])}
    except Exception:  # noqa: BLE001
        _M092 = None


def _m092_scores(x: list) -> list:
    """行動クラスごとの生スコア（softmax前。argmax用途なので正規化不要）。"""
    if _M092["mode"] == "lgb":
        return list(_M092["bst"].predict([x])[0])
    feat, thr = _M092["feat"], _M092["thr"]
    left, right, val = _M092["left"], _M092["right"], _M092["val"]
    scores = [0.0] * _M092["ncls"]
    for r, c in zip(_M092["roots"], _M092["tcls"]):
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
    ft = _M092["ft"]
    feats, _keys = ft.feat_vector(state, my)
    proba = _m092_scores(feats)
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

F092_LETHAL = os.environ.get("F092_LETHAL", "on") != "off"
F092_STATS = {"search_calls": 0, "fires": 0, "overrides": 0}


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

    F092_STATS["search_calls"] += 1
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

    if F092_LETHAL and is_main:
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
        snap.retreat_money = plan.retreat_money
        saved_turn = pre_turn
        try:
            choice = _lethal_search(obs, scores)
        except Exception:  # noqa: BLE001
            choice = None
        plan, pre_turn = snap, saved_turn
        if choice is not None:
            F092_STATS["fires"] += 1
            try:  # instrumentation only: would the lower layers differ?
                cf = None
                if _M092 is not None:
                    cf = _ml_layer(obs)
                if cf is None:
                    cf = policy(obs)
                if cf and cf[0] != choice[0]:
                    F092_STATS["overrides"] += 1
                plan, pre_turn = snap, saved_turn  # re-restore after cf scoring
            except Exception:  # noqa: BLE001
                plan, pre_turn = snap, saved_turn
            return choice

    # Layer 2 (ML): identical to 089
    if _M092 is not None and is_main:
        try:
            choice = _ml_layer(obs)
        except Exception:  # noqa: BLE001
            choice = None
        if choice is not None:
            return choice

    # Layer 3 (082 rules)
    return policy(obs)
