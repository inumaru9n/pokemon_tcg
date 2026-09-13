from __future__ import annotations

import os
from collections import defaultdict

from cg.api import (AreaType, CardType, EnergyType, SelectContext,
                    OptionType, Pokemon, all_card_data, to_observation_class)

"""
EXP-041: Lightweight reverse-engineered replica of the top Cynthia's Garchomp ex
individual (sig fc15d0c926, nasuo445, 596 games 58.4% on 07-08 ladder).
Same 60 cards as agents/022_garchomp; policy rebuilt from action-frequency
aggregation of all 596 replays (scratchpad aggregate_041*.py). Key findings baked in:

 1. Corkscrew Dive is the main attack (1747 uses vs 285 Draconic Buster).
    Rule: Corkscrew-kill > Buster-kill > Buster-chunk (2en no-kill: Buster 86%).
 2. Cheer On to Glory STACKS (+30 per Roserade in play, engine ClampShort adds).
    The pro keeps 2 Roserades on bench -> Corkscrew 160 / Buster 320.
 3. Spiritomb Raging Curse = 10 x damage counters on MY benched Cynthia's pokemon
    (all our pokemon are Cynthia's). Damaged Garchomps are retreated (free) to the
    bench to bank counters; Spiritomb finishes (535 uses, 82% kills).
 4. Bench NOTHING at setup (596/596 games); fill bench via Poffin/hand during turns.
    Setup active: Gible > Roselia > Spiritomb (strict).
 5. Supporters: Lillie > Xerosic(opp hand>=7) > Hilda > Boss(kill-only) > Surfer(rare).
 6. Energy: Rock Fighting to the Garchomp line, Basic F to Spiritomb; keep every
    Garchomp/Gabite/Gible/Spiritomb at >=1, active Garchomp to 2 for Buster.
 7. Power Weight on the Garchomp line (Gible/Gabite early, carries through evolution).
 8. Unfair Stamp when opponent hand >= 4; Xerosic-forced discards follow the pro's
    observed priority (Poffin/Boss/spare energy first, keep Garchomp/searchers).
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

CORKSCREW_DIVE = 531    # 100 [1F] + may draw to 6
DRACONIC_BUSTER = 532   # 260 [2F], discard all energy after
DRAGONSLICE = 530       # Gabite 40 [1F]
ROCK_HURL = 529         # Gible 20 [1F], ignores resistance
SPIKE_STING = 475       # Roselia 20 [1C]
RAGING_CURSE = 540      # Spiritomb 0 [1C], 10 x counters on my benched Cynthia's

LINE = (Gible, Gabite, Garchomp)
ENERGIES = (Fighting_Energy, Rock_Fighting_Energy)

# Discard priority when forced (opp Xerosic etc.): higher = discard first.
# From the pro's 596-game forced-discard distribution.
DISCARD_ORDER = {Buddy_Poffin: 17, Boss_Orders: 16, Fighting_Energy: 15, Roselia: 14,
                 Hilda: 13, Lillie_Determination: 12, Gabite: 11, Forest_Vitality: 10,
                 Surfer: 9, Roserade: 8, Rock_Fighting_Energy: 7, Gible: 6,
                 Garchomp: 5, Spiritomb: 4, Power_Weight: 3, Fighting_Gong: 2,
                 Poke_Pad: 1, Night_Stretcher: 1, Unfair_Stamp: 1, Xerosic: 1}


class AttackPlan:
    attacker = -1     # index into [active] + bench
    target = -1       # index into [opp active] + opp bench
    attack_id = -1
    remain_hp = -1
    energy = False    # attach an energy to the attacker this turn
    kill = False


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
    return 3 if data.megaEx else 2 if data.ex else 1


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


def counters(pokemon):
    return max(0, (pokemon.maxHp - pokemon.hp) // 10)


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
    my_field = []
    for c in (me.active or []) + list(me.bench):
        if c is not None:
            field_counts[c.id] += 1
            my_field.append(c)
    for c in me.hand or []:
        hand_counts[c.id] += 1

    roserade_boost = 30 * field_counts[Roserade]  # Cheer On to Glory stacks
    hand_fighting = hand_counts[Fighting_Energy] + hand_counts[Rock_Fighting_Energy]
    my_active = me.active[0] if me.active else None
    line_count = field_counts[Gible] + field_counts[Gabite] + field_counts[Garchomp]
    ros_count = field_counts[Roselia] + field_counts[Roserade]
    bench_free = me.benchMax - sum(1 for b in me.bench if b is not None)

    def curse_damage(attacker_index):
        """Raging Curse base damage if my pokemon #attacker_index attacks.
        Counters on MY bench after the (possible) switch: everyone except the
        attacker; the current active joins the bench when a bench pokemon attacks."""
        my_cards = [my_active] + list(me.bench)
        total = 0
        for k, pk in enumerate(my_cards):
            if pk is None or k == attacker_index:
                continue
            if k == 0 and attacker_index == 0:
                continue  # active stays active
            total += counters(pk)
        return total * 10

    if context == SelectContext.MAIN:
        can_op_switch = any(
            o.type == OptionType.PLAY and me.hand[o.index].id == Boss_Orders
            for o in select.option) and not state.supporterPlayed
        my_cards = [my_active] + list(me.bench)
        op_cards = ([op.active[0] if op.active else None] + list(op.bench))
        if state.turn >= 2:
            best = -1
            can_attach = hand_fighting >= 1 and not state.energyAttached
            can_retreat = not state.retreated and my_active is not None
            active_rc = card_table[my_active.id].retreatCost if my_active else 99
            for i, mine in enumerate(my_cards):
                if mine is None:
                    continue
                friction = 0
                bank = 0
                if i != 0:
                    if not can_retreat:
                        continue
                    if active_rc > len(my_active.energies):
                        continue
                    friction = 60 if active_rc == 0 else 260
                    bank = min(200, counters(my_active) * 10)  # counters feed Raging Curse
                cands = []
                if mine.id == Garchomp:
                    cands.append((CORKSCREW_DIVE, 1, 100, 70))
                    cands.append((DRACONIC_BUSTER, 2, 260, -30))
                elif mine.id == Gabite:
                    cands.append((DRAGONSLICE, 1, 40, -40))
                elif mine.id == Gible:
                    cands.append((ROCK_HURL, 1, 20, -60))
                elif mine.id == Roselia:
                    cands.append((SPIKE_STING, 1, 20, -60))
                elif mine.id == Spiritomb:
                    base = curse_damage(i)
                    if base > 0:
                        cands.append((RAGING_CURSE, 1, base, 20))
                else:
                    continue
                for attack_id, need, base_damage, base_score in cands:
                    ec = len(mine.energies)
                    more = False
                    if ec < need:
                        if can_attach and ec + 1 >= need:
                            ec += 1
                            more = True
                        else:
                            continue
                    for j, opp in enumerate(op_cards):
                        if opp is None:
                            continue
                        if j != 0 and not can_op_switch:
                            continue
                        damage = base_damage + roserade_boost
                        data = card_table[opp.id]
                        atk_type = card_table[mine.id].energyType
                        if attack_id != RAGING_CURSE and data.weakness == atk_type:
                            damage *= 2
                        if attack_id != ROCK_HURL and data.resistance == atk_type:
                            damage -= 30
                        if damage <= 0:
                            continue
                        kill = opp.hp <= damage
                        score = pokemon_score(opp)
                        prize = 0
                        if kill:
                            prize = prize_count(opp)
                        else:
                            score *= damage / opp.hp
                        score += base_score - friction + bank
                        if kill and len(op.prize) <= prize:
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
                            plan.kill = kill

    def energy_score(pokemon, is_active, energy_id, in_play_index):
        ec = len(pokemon.energies)
        score = 8000 + (10 if is_active else 0)
        pid = pokemon.id
        if pid == Garchomp:
            if ec < 1:
                score += 160
            elif ec < 2:
                score += 110
            else:
                score -= 60
        elif pid in (Gible, Gabite):
            score += 100 if ec < 1 else -60
        elif pid == Spiritomb:
            score += 90 if ec < 1 else -90
        elif pid in (Roselia, Roserade):
            score += 10 if (is_active and ec < 1) else -110
        else:
            score -= 120
        # Rock Fighting to the (Fighting) Garchomp line, Basic F to Spiritomb
        if energy_id == Rock_Fighting_Energy:
            score += 12 if pid in LINE else -12
        elif energy_id == Fighting_Energy and pid == Spiritomb:
            score += 12
        if plan.energy and ((is_active and plan.attacker == 0)
                            or (not is_active and plan.attacker == 1 + in_play_index)):
            score += 400
        return score

    def to_hand_score(cid):
        """Deck/discard search target value (Champion's Call, Gong, Hilda, Pad, ...)."""
        if cid == Garchomp:
            if field_counts[Gabite] >= 1 and hand_counts[Garchomp] == 0:
                return 300
            return max(40, 150 - 70 * hand_counts[Garchomp])
        if cid == Gabite:
            if field_counts[Gible] >= 1 and hand_counts[Gabite] == 0 \
                    and field_counts[Gabite] + field_counts[Garchomp] < 3:
                return 280
            return max(30, 120 - 60 * hand_counts[Gabite])
        if cid == Roserade:
            if field_counts[Roselia] >= 1 and hand_counts[Roserade] == 0 \
                    and field_counts[Roserade] < 2:
                return 260
            return 60
        if cid == Roselia:
            if ros_count + hand_counts[Roselia] < 2:
                return 220
            return 30
        if cid == Spiritomb:
            if field_counts[Spiritomb] + hand_counts[Spiritomb] == 0:
                return 200
            return 20
        if cid == Gible:
            if line_count + hand_counts[Gible] < 3:
                return 180
            return 25
        if cid in ENERGIES:
            base = 240 if hand_fighting == 0 else 35
            return base + (15 if cid == Rock_Fighting_Energy else 0)
        if cid == Boss_Orders:
            return 45
        return 10

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
                        score += ec * 8
                        if plan.attacker >= 1 and o.area == AreaType.BENCH \
                                and o.index == plan.attacker - 1:
                            score += 500
                        if card.id == Garchomp:
                            score += 40
                        elif card.id == Spiritomb:
                            op_active = op.active[0] if op.active else None
                            cd = curse_damage(0)  # everyone but current active... approx
                            if isinstance(card, Pokemon):
                                cd = 10 * sum(counters(b) for b in [my_active] + list(me.bench)
                                              if b is not None and b.serial != card.serial)
                            cd += roserade_boost if cd > 0 else 0
                            if op_active is not None and ec >= 1 and cd >= op_active.hp:
                                score += 55
                            else:
                                score += 8
                        elif card.id == Gabite:
                            score += 10
                        elif card.id == Gible:
                            score += 4
                        elif card.id == Roselia:
                            score += 3
                        else:  # Roserade: keep the boost on the bench
                            score += 2
                    else:
                        if o.index == plan.target - 1:
                            score += 100
                elif context == SelectContext.SETUP_ACTIVE_POKEMON:
                    score = 4 if card.id == Gible else 3 if card.id == Roselia else 2
                elif context == SelectContext.SETUP_BENCH_POKEMON:
                    score = -1  # pro benches nothing at setup (596/596)
                elif context == SelectContext.TO_HAND:
                    score = to_hand_score(card.id)
                elif context in (SelectContext.TO_BENCH, SelectContext.TO_FIELD):
                    # Buddy-Buddy Poffin: Gible > Roselia(if <2) > Spiritomb(if none)
                    if card.id == Gible:
                        score = 300 if line_count < 3 else 50
                    elif card.id == Roselia:
                        score = 250 if ros_count < 2 else 20
                    elif card.id == Spiritomb:
                        score = 200 if field_counts[Spiritomb] == 0 else 10
                    else:
                        score = 5
                elif context == SelectContext.DISCARD:
                    score = DISCARD_ORDER.get(card.id, 1)
                    # keep the last energy in hand if we still need to attach
                    if card.id in ENERGIES and hand_fighting <= 1 and not state.energyAttached:
                        score = 2
                elif context == SelectContext.ATTACH_FROM:
                    score = energy_score(card, o.area == AreaType.ACTIVE, 0,
                                         o.index if o.area == AreaType.BENCH else 0)
                elif context in (SelectContext.DAMAGE, SelectContext.DAMAGE_COUNTER):
                    if o.playerIndex != my_index:
                        score = 5000 + pokemon_score(card) if card.hp <= 30 else pokemon_score(card)
                    else:
                        score = -100
                elif context == SelectContext.EVOLVES_TO:
                    score = 100 if card.id == Garchomp else 60 if card.id == Gabite else 40
        elif o.type == OptionType.PLAY:
            card = get_card(obs, AreaType.HAND, o.index, my_index)
            data = card_table[card.id]
            if data.cardType == CardType.POKEMON:
                score = 20000
                if card.id == Gible:
                    if line_count >= 4 or (line_count >= 3 and bench_free < 2):
                        score = -1
                elif card.id == Roselia:
                    if ros_count >= 3 or (ros_count >= 2 and bench_free < 2):
                        score = -1
                elif card.id == Spiritomb:
                    if field_counts[Spiritomb] >= 1:
                        score = -1
            else:
                score = 10000
                if card.id == Boss_Orders:
                    score = 3150 if (plan.target >= 1 and plan.kill) else -1
                elif card.id == Lillie_Determination:
                    score = 3100
                elif card.id == Xerosic:
                    score = 3050 if op.handCount >= 7 else -1
                elif card.id == Hilda:
                    score = 3000
                elif card.id == Surfer:
                    score = 2500 if len(me.hand or []) <= 3 else -1
                elif card.id == Buddy_Poffin:
                    score = 9000 if bench_free >= 1 else -1
                elif card.id == Fighting_Gong:
                    score = 8500
                elif card.id == Poke_Pad:
                    score = 8300
                elif card.id == Night_Stretcher:
                    score = 6000
                elif card.id == Unfair_Stamp:
                    score = 9800 if op.handCount >= 4 else -1
                elif card.id == Forest_Vitality:
                    score = -1 if stadium_id == Forest_Vitality else 5000
        elif o.type == OptionType.ATTACH:
            card = get_card(obs, AreaType.HAND, o.index, my_index)
            pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
            if card.id == Power_Weight:
                if pokemon.id in LINE:
                    score = 7000 + (200 if pokemon.id == Garchomp else
                                    150 if pokemon.id == Gabite else 120)
                else:
                    score = 100
            else:
                score = energy_score(pokemon, o.inPlayArea == AreaType.ACTIVE,
                                     card.id, o.inPlayIndex)
        elif o.type == OptionType.EVOLVE:
            pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
            score = 9000 + len(pokemon.energies)
        elif o.type == OptionType.ABILITY:
            score = 30000  # Champion's Call: every Gabite, every turn (5705 uses)
        elif o.type == OptionType.RETREAT:
            score = 2600 if plan.attacker >= 1 else -1
        elif o.type == OptionType.ATTACK:
            score = 1000
            if o.attackId == plan.attack_id and plan.attacker == 0:
                score += 100
        scores.append(score)

    desc = [i for i, _ in sorted(enumerate(scores), key=lambda x: x[1], reverse=True)]
    if select.minCount < select.maxCount:
        picked = [i for i in desc if scores[i] >= 0]
        if len(picked) < select.minCount:
            picked = desc[: select.minCount]
        return picked[: select.maxCount]
    return desc[: select.maxCount]
