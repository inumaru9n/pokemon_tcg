from __future__ import annotations

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


class C:
    KYOGRE = 721
    SNOVER = 722
    MEGA_ABOMASNOW_EX = 723

    MAKUHITA = 673
    HARIYAMA = 674
    LUNATONE = 675
    SOLROCK = 676
    RIOLU = 677
    MEGA_LUCARIO_EX = 678

    BASIC_FIGHTING_ENERGY = 6
    DUSK_BALL = 1102
    SWITCH = 1123
    PREMIUM_POWER_PRO = 1141
    FIGHTING_GONG = 1142
    POKE_PAD = 1152
    HERO_CAPE = 1159
    BOSS_ORDERS = 1182
    CARMINE = 1192
    LILLIE_DETERMINATION = 1227
    GRAVITY_MOUNTAIN = 1252

    LUMIOSE_CITY = 1267
    LILLIES_PEARL = 1172
    LEGACY_ENERGY = 12


MEGA_BRAVE = 983
LOW_DECK_COUNT = 10


DECK_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deck.csv")
if not os.path.exists(DECK_PATH):
    DECK_PATH = "/kaggle_simulations/agent/deck.csv"
with open(DECK_PATH, "r", encoding="utf-8") as f:
    my_deck = [int(line) for line in f.read().splitlines() if line.strip()]


all_card = all_card_data()
card_table = {card.cardId: card for card in all_card}


class AttackPlan:
    def __init__(
        self,
        attacker: int = -1,
        target: int = -1,
        attack_index: int = -1,
        remain_hp: int = -1,
        needs_energy: bool = False,
    ):
        self.attacker = attacker
        self.target = target
        self.attack_index = attack_index
        self.remain_hp = remain_hp
        self.needs_energy = needs_energy


plan = AttackPlan()
pre_turn = -1
ability_used = False


def get_card(obs: Observation, area: AreaType, index: int, player_index: int) -> Pokemon | Card | None:
    player = obs.current.players[player_index]
    match area:
        case AreaType.DECK:
            return obs.select.deck[index]
        case AreaType.HAND:
            return player.hand[index]
        case AreaType.DISCARD:
            return player.discard[index]
        case AreaType.ACTIVE:
            return player.active[index]
        case AreaType.BENCH:
            return player.bench[index]
        case AreaType.PRIZE:
            return player.prize[index]
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
        if card.id == C.LEGACY_ENERGY:
            count -= 1
    for card in pokemon.tools:
        if card.id == C.LILLIES_PEARL and "Lillie" in data.name:
            count -= 1
    return max(0, count)


def target_score(pokemon: Pokemon) -> int:
    data = card_table[pokemon.id]
    score = prize_count(pokemon) * 1000
    score += len(pokemon.energies) * 150
    score += len(pokemon.tools) * 100
    if data.stage2:
        score += 250
    elif data.stage1:
        score += 130

    if pokemon.id in {144, 322, 323, 337}:  # low-value support Pokemon
        score -= 200
    if pokemon.id == C.SNOVER:
        score += 950
    elif pokemon.id == C.MEGA_ABOMASNOW_EX:
        score += 250
    if pokemon.id == C.RIOLU:
        score += 800
    elif pokemon.id == C.MEGA_LUCARIO_EX:
        score += 100
    if pokemon.id == 112 and len(pokemon.energies) >= 1:  # Munkidori
        score += 300
    score += pokemon.hp
    return score


class LucarioPolicy:
    def __init__(self, obs: Observation):
        self.obs = obs
        self.state = obs.current
        self.select = obs.select
        self.context = self.select.context
        self.my_index = self.state.yourIndex
        self.op_index = 1 - self.my_index
        self.me = self.state.players[self.my_index]
        self.opponent = self.state.players[self.op_index]
        self.my_prizes_left = len(self.me.prize)

        self.field_counts = defaultdict(int)
        self.hand_counts = defaultdict(int)
        self.discard_counts = defaultdict(int)
        self.has_ready_lucario_line = False
        self.has_ready_hariyama_line = False
        self.can_switch = False
        self.can_gust = False
        self.can_attack = False
        self.can_use_mega_brave = False
        self.stadium_id = self.state.stadium[0].id if self.state.stadium else 0

        self._count_cards()
        self._scan_main_options()

    def choose(self) -> list[int]:
        if not self.select.option or self.select.maxCount == 0:
            return []

        if self.context == SelectContext.MAIN:
            self._plan_attack()

        scores = [self._score_option(option) for option in self.select.option]
        ranked = [i for i, _ in sorted(enumerate(scores), key=lambda item: item[1], reverse=True)]
        self._remember_lunatone_ability(ranked)
        return ranked[: self.select.maxCount]

    def _count_cards(self) -> None:
        for pokemon in self.me.active + self.me.bench:
            if pokemon is None:
                continue
            self.field_counts[pokemon.id] += 1
            if pokemon.id in {C.MAKUHITA, C.HARIYAMA} and len(pokemon.energies) >= 3:
                self.has_ready_hariyama_line = True
            if pokemon.id in {C.RIOLU, C.MEGA_LUCARIO_EX} and len(pokemon.energies) >= 2:
                self.has_ready_lucario_line = True

        for card in self.me.hand:
            self.hand_counts[card.id] += 1
        for card in self.me.discard:
            self.discard_counts[card.id] += 1

    def _scan_main_options(self) -> None:
        if self.context != SelectContext.MAIN:
            return
        for option in self.select.option:
            if option.type == OptionType.PLAY:
                card = get_card(self.obs, AreaType.HAND, option.index, self.my_index)
                if card.id == C.SWITCH:
                    self.can_switch = True
                elif card.id == C.BOSS_ORDERS:
                    self.can_gust = True
            elif option.type == OptionType.EVOLVE:
                card = get_card(self.obs, AreaType.HAND, option.index, self.my_index)
                if card.id == C.HARIYAMA:
                    self.can_gust = True
            elif option.type == OptionType.RETREAT:
                self.can_switch = True
            elif option.type == OptionType.ATTACK:
                self.can_attack = True
                if option.attackId == MEGA_BRAVE:
                    self.can_use_mega_brave = True

    def _my_board(self) -> list[Pokemon | None]:
        return self.me.active + self.me.bench

    def _opponent_board(self) -> list[Pokemon | None]:
        return self.opponent.active + self.opponent.bench

    def _opponent_has(self, ids: set[int]) -> bool:
        return any(pokemon is not None and pokemon.id in ids for pokemon in self._opponent_board())

    def _opponent_is_water_deck(self) -> bool:
        return self._opponent_has({C.KYOGRE, C.SNOVER, C.MEGA_ABOMASNOW_EX})

    def _opponent_is_crustle_wall(self) -> bool:
        return self._opponent_has({344, 345})

    def _can_evolve_board_index(self, board_index: int) -> bool:
        for option in self.select.option:
            if option.type != OptionType.EVOLVE:
                continue
            target_index = option.inPlayIndex
            if option.inPlayArea == AreaType.BENCH:
                target_index += 1
            if target_index == board_index:
                return True
        return False

    def _base_attack(self, pokemon: Pokemon, attack_index: int) -> tuple[int, int, int] | None:
        energy_required = 0
        base_damage = 0
        base_score = 0

        if pokemon.id == C.MEGA_LUCARIO_EX:
            if attack_index == 0:
                energy_required = 1
                base_damage = 130
                base_score += 60 * min(3, self.discard_counts[C.BASIC_FIGHTING_ENERGY])
            else:
                energy_required = 2
                base_damage = 270
            if self._opponent_is_water_deck() and len(self.opponent.prize) <= 3:
                base_score -= 500
        elif attack_index == 1:
            return None
        elif pokemon.id == C.HARIYAMA:
            energy_required = 3
            base_damage = 210
        elif pokemon.id == C.MAKUHITA:
            return None
        elif pokemon.id == C.SOLROCK and self.field_counts[C.LUNATONE] >= 1:
            energy_required = 1
            base_damage = 70

        if base_damage <= 0:
            return None
        return energy_required, base_damage, base_score

    def _base_attack_after_evolution(self, pokemon: Pokemon, board_index: int, attack_index: int):
        if pokemon.id == C.MAKUHITA and attack_index == 0 and self._can_evolve_board_index(board_index):
            return 3, 210, -100
        return self._base_attack(pokemon, attack_index)

    def _plan_attack(self) -> None:
        global plan
        best_score = -1
        plan = AttackPlan()

        if self.state.turn < 2:
            return

        for attacker_index, my_pokemon in enumerate(self._my_board()):
            if my_pokemon is None:
                continue
            if attacker_index != 0 and not self.can_switch:
                break

            for attack_index in range(2):
                attack = self._base_attack_after_evolution(my_pokemon, attacker_index, attack_index)
                if attack is None:
                    continue
                energy_required, base_damage, base_score = attack

                energy_count = len(my_pokemon.energies)
                if attack_index == 1 and attacker_index == 0 and energy_count >= 2 and not self.can_use_mega_brave:
                    break

                needs_energy = False
                if energy_count < energy_required:
                    if self.hand_counts[C.BASIC_FIGHTING_ENERGY] >= 1 and not self.state.energyAttached:
                        energy_count += 1
                        needs_energy = energy_count >= energy_required
                    if not needs_energy:
                        continue

                for target_index, op_pokemon in enumerate(self._opponent_board()):
                    if op_pokemon is None:
                        continue
                    if target_index != 0 and not self.can_gust:
                        break
                    if (
                        self._opponent_is_crustle_wall()
                        and my_pokemon.id == C.MEGA_LUCARIO_EX
                        and op_pokemon.id == 345
                    ):
                        continue

                    damage = base_damage
                    op_data = card_table[op_pokemon.id]
                    if op_data.weakness == EnergyType.FIGHTING:
                        damage *= 2
                    elif op_data.resistance == EnergyType.FIGHTING:
                        damage -= 30

                    score = target_score(op_pokemon)
                    prize = prize_count(op_pokemon) if op_pokemon.hp <= damage else 0
                    if prize == 0:
                        score *= damage / op_pokemon.hp
                    if len(self.opponent.prize) <= prize:
                        score = 50000

                    score += base_score
                    score += 220 if attacker_index == 0 else 0
                    score += 300 if target_index == 0 else 0
                    score += energy_count

                    if score > best_score:
                        best_score = score
                        plan = AttackPlan(
                            attacker=attacker_index,
                            target=target_index,
                            attack_index=attack_index,
                            remain_hp=op_pokemon.hp - damage,
                            needs_energy=needs_energy,
                        )

    def _energy_target_score(self, pokemon: Pokemon, active: bool) -> int:
        energy_count = len(pokemon.energies)
        score = 8000 + (10 if active else 0)

        if pokemon.id in {C.MAKUHITA, C.HARIYAMA}:
            score += 1 if pokemon.id == C.HARIYAMA else 0
            if self._opponent_is_crustle_wall():
                score += 260 if energy_count < 3 else 30
            else:
                score += 100 if energy_count < 3 else 0
                score -= 50 if self.has_ready_hariyama_line else 0
        elif pokemon.id == C.LUNATONE:
            score -= 100
        elif pokemon.id == C.SOLROCK:
            score += 20 if energy_count < 1 else -100
        elif pokemon.id in {C.RIOLU, C.MEGA_LUCARIO_EX}:
            score += 1 if pokemon.id == C.MEGA_LUCARIO_EX else 0
            score += 100 if energy_count < 2 else 0
            score -= 50 if self.has_ready_lucario_line else 0
        return score

    def _score_option(self, option) -> float:
        if option.type == OptionType.NUMBER:
            return option.number
        if option.type == OptionType.YES:
            return 100 if self.context == SelectContext.IS_FIRST else 1
        if option.type == OptionType.NO:
            return 0
        if option.type == OptionType.CARD:
            return self._score_card_choice(option)
        if option.type == OptionType.PLAY:
            return self._score_play(option)
        if option.type == OptionType.ATTACH:
            return self._score_attach(option)
        if option.type == OptionType.EVOLVE:
            return self._score_evolve(option)
        if option.type == OptionType.ABILITY:
            return self._score_ability(option)
        if option.type == OptionType.RETREAT:
            return 2000 if plan.attacker >= 1 else -1
        if option.type == OptionType.ATTACK:
            if (
                self._opponent_is_crustle_wall()
                and self.me.active
                and self.opponent.active
                and self.me.active[0].id == C.MEGA_LUCARIO_EX
                and self.opponent.active[0].id == 345
                and plan.target < 0
            ):
                return -1
            return 1100 if (option.attackId == MEGA_BRAVE) == (plan.attack_index == 1) else 1000
        return 0

    def _score_card_choice(self, option) -> float:
        card = get_card(self.obs, option.area, option.index, option.playerIndex)
        if card is None:
            return 0

        if self.context in {SelectContext.SWITCH, SelectContext.TO_ACTIVE}:
            return self._score_active_choice(option, card)
        if self.context == SelectContext.SETUP_ACTIVE_POKEMON:
            return self._score_setup_active(card)
        if self.context == SelectContext.TO_HAND:
            return self._score_to_hand(card)
        if self.context == SelectContext.ATTACH_FROM and isinstance(card, Pokemon):
            return self._energy_target_score(card, option.area == AreaType.ACTIVE)
        return 0

    def _score_active_choice(self, option, card: Pokemon | Card) -> float:
        if not isinstance(card, Pokemon):
            return 0

        if option.playerIndex != self.my_index:
            return 100 if option.index == plan.target - 1 else 0

        score = len(card.energies) * 2
        if option.index == plan.attacker - 1:
            score += 100
        if card.id == C.MEGA_LUCARIO_EX:
            score += 8 if self._opponent_is_water_deck() and len(self.opponent.prize) <= 3 else 20
        elif card.id == C.HARIYAMA and len(card.energies) >= 2:
            score += 45 if self._opponent_is_crustle_wall() else 15
        elif card.id == C.MAKUHITA and len(card.energies) >= 2:
            score += 35 if self._opponent_is_crustle_wall() else 10
        elif card.id == C.SOLROCK:
            score += 5
        elif card.id == C.RIOLU:
            score += 4
        return score

    def _score_setup_active(self, card: Pokemon | Card) -> int:
        if card.id == C.SOLROCK:
            return 2 if self.state.firstPlayer == self.my_index else 4
        if card.id == C.RIOLU:
            return 3
        if card.id == C.MAKUHITA:
            return 1
        return 0

    def _score_to_hand(self, card: Pokemon | Card) -> float:
        score = 200 - self.hand_counts[card.id] * 100
        if card.id == C.MAKUHITA:
            if self._opponent_is_crustle_wall():
                score += 80 if self.field_counts[card.id] < 2 else -20
            else:
                score += -10 if self.field_counts[card.id] >= 1 else 10
        elif card.id == C.HARIYAMA:
            if self._opponent_is_crustle_wall():
                score += 120 if self.field_counts[C.MAKUHITA] >= 1 else -5
            else:
                score += 20 if self.field_counts[C.MAKUHITA] >= 1 else -20
        elif card.id == C.LUNATONE:
            score += -250 if self.field_counts[card.id] >= 1 else 60
        elif card.id == C.SOLROCK:
            score += -250 if self.field_counts[card.id] >= 1 else 50
        elif card.id == C.RIOLU:
            lucario_line = self.field_counts[C.RIOLU] + self.field_counts[C.MEGA_LUCARIO_EX]
            score += -150 if lucario_line >= 2 else -3 if lucario_line >= 1 else 40
        elif card.id == C.MEGA_LUCARIO_EX:
            score += 40 if self.field_counts[C.RIOLU] >= 1 else -15
        elif card.id == C.BASIC_FIGHTING_ENERGY:
            score += 30 if not ability_used or not self.state.energyAttached else -1
        return score

    def _score_play(self, option) -> float:
        card = get_card(self.obs, AreaType.HAND, option.index, self.my_index)
        data = card_table[card.id]
        if data.cardType == CardType.POKEMON:
            return self._score_play_pokemon(card)
        return self._score_play_trainer(card)

    def _score_play_pokemon(self, card: Card) -> float:
        score = 20000
        if card.id in {C.LUNATONE, C.SOLROCK} and self.field_counts[card.id] >= 1:
            return -1
        if card.id == C.RIOLU and self.field_counts[C.RIOLU] + self.field_counts[C.MEGA_LUCARIO_EX] >= 2:
            return -1
        return score

    def _score_play_trainer(self, card: Card) -> float:
        if card.id == C.SWITCH:
            return 6000 if plan.attacker > 0 else -1
        if card.id == C.PREMIUM_POWER_PRO:
            if self.state.supporterPlayed and plan.remain_hp <= 0:
                return -1
            if not self.can_attack:
                can_bridge_draw = (
                    not self.state.supporterPlayed
                    and self.hand_counts[C.CARMINE] > 0
                    and self.hand_counts[C.LILLIE_DETERMINATION] == 0
                    and not self._low_deck()
                )
                return 3050 if can_bridge_draw else -1
            return 5000
        if card.id == C.BOSS_ORDERS:
            return 3200 if plan.target >= 1 else -1
        if card.id == C.CARMINE:
            return -1 if self._low_deck() else 3000
        if card.id == C.LILLIE_DETERMINATION:
            return -1 if self._low_deck() else 3100
        if card.id == C.GRAVITY_MOUNTAIN:
            return self._score_gravity_mountain()
        return 10000

    def _score_gravity_mountain(self) -> float:
        opponent_has_stage2 = any(
            pokemon is not None and card_table[pokemon.id].stage2 for pokemon in self._opponent_board()
        )
        if opponent_has_stage2:
            return 3500
        return 1200 if self.stadium_id else -1

    def _low_deck(self) -> bool:
        return self.me.deckCount <= LOW_DECK_COUNT

    def _score_attach(self, option) -> float:
        card = get_card(self.obs, AreaType.HAND, option.index, self.my_index)
        pokemon = get_card(self.obs, option.inPlayArea, option.inPlayIndex, self.my_index)
        if not isinstance(pokemon, Pokemon):
            return 0

        if card.id == C.HERO_CAPE:
            score = 7000
            if self._opponent_is_water_deck():
                if pokemon.id == C.RIOLU:
                    return 12200
                if pokemon.id == C.MEGA_LUCARIO_EX:
                    return 12800
            if pokemon.id == C.RIOLU:
                score += 100
            elif pokemon.id == C.MEGA_LUCARIO_EX:
                score += 200
            return score

        score = self._energy_target_score(pokemon, option.inPlayArea == AreaType.ACTIVE)
        board_index = option.inPlayIndex if option.inPlayArea == AreaType.ACTIVE else option.inPlayIndex + 1
        if board_index == plan.attacker and plan.needs_energy:
            score += 200
        return score

    def _score_evolve(self, option) -> float:
        pokemon = get_card(self.obs, option.inPlayArea, option.inPlayIndex, self.my_index)
        if not isinstance(pokemon, Pokemon):
            return 0
        if pokemon.id == C.MAKUHITA and plan.target == 0 and not self._opponent_is_crustle_wall():
            return -1
        return 9000 + len(pokemon.energies)

    def _score_ability(self, option) -> float:
        card = get_card(self.obs, option.area, option.index, self.my_index)
        if card.id == C.LUMIOSE_CITY:
            return 1
        if card.id == C.LUNATONE and self._low_deck():
            return -1
        return 30000

    def _remember_lunatone_ability(self, ranked: list[int]) -> None:
        global ability_used
        if self.context != SelectContext.MAIN or not ranked:
            return
        option = self.select.option[ranked[0]]
        if option.type != OptionType.ABILITY:
            return
        card = get_card(self.obs, option.area, option.index, self.my_index)
        if card is not None and card.id == C.LUNATONE:
            ability_used = True


# ---------------------------------------------------------------------------
# Determinized 1-ply search layer (MAIN only).
# Candidate MAIN options are ranked by the heuristic, the top few are stepped
# in a determinized simulation (search_begin/search_step), greedily rolled out
# to the end of our turn with the same heuristic, and scored by evaluate().
# Any failure falls back to the pure heuristic choice.

SEARCH_CANDIDATES = 8
SEARCH_SAMPLES = 3
SEARCH_MOVE_BUDGET = 1.5     # seconds per MAIN decision
SEARCH_GAME_BUDGET = 450.0   # total seconds of search per game (600s limit)
ROLLOUT_STEP_CAP = 80

PLACEHOLDER_MON = 1072  # basic Pokémon used for unknown opponent cards
if not (card_table.get(PLACEHOLDER_MON) and card_table[PLACEHOLDER_MON].basic):
    PLACEHOLDER_MON = next(c.cardId for c in all_card if c.basic)

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


my_deck_counts = Counter(my_deck)


def _determinize(obs: Observation):
    state = obs.current
    me = state.players[state.yourIndex]
    op = state.players[1 - state.yourIndex]

    remaining = my_deck_counts - _my_visible_counts(obs)
    unknown = list(remaining.elements())
    random.shuffle(unknown)
    need = me.deckCount + len(me.prize)
    while len(unknown) < need:  # counting slack: pad with basic energy
        unknown.append(C.BASIC_FIGHTING_ENERGY)
    your_deck = unknown[: me.deckCount]
    your_prize = unknown[me.deckCount : me.deckCount + len(me.prize)]

    opponent_deck = [PLACEHOLDER_MON] * op.deckCount
    opponent_prize = [PLACEHOLDER_MON] * len(op.prize)
    opponent_hand = [C.BASIC_FIGHTING_ENERGY] * op.handCount
    opponent_active = []
    if op.active and op.active[0] is None:
        opponent_active = [PLACEHOLDER_MON]
    return your_deck, your_prize, opponent_deck, opponent_prize, opponent_hand, opponent_active


def _evaluate(obs: Observation, my_index: int, root_my_prize: int, root_op_prize: int) -> float:
    state = obs.current
    if state.result != -1:
        if state.result == my_index:
            return 1e9
        if state.result == 1 - my_index:
            return -1e9
        return 0.0
    me = state.players[my_index]
    op = state.players[1 - my_index]
    score = 10000.0 * ((root_my_prize - len(me.prize)) - (root_op_prize - len(op.prize)))
    for pokemon in me.active + me.bench:
        if pokemon is not None:
            score += pokemon.hp * 1.0 + len(pokemon.energies) * 100.0
    for pokemon in op.active + op.bench:
        if pokemon is not None:
            score -= pokemon.hp * 1.0 + len(pokemon.energies) * 100.0
    score += 20.0 * (me.handCount - op.handCount)
    score -= 5.0 * max(0, LOW_DECK_COUNT - me.deckCount)  # avoid decking out
    return score


def _rollout(state, root_turn: int, deadline: float, plies: int = 1):
    """Greedy rollout with the heuristic for `plies` player-turns (or cap/deadline).

    plies=1 stops when our turn ends; plies=2 also plays out the opponent's
    reply turn (their hidden hand is a placeholder, but their board — and thus
    their counter-attack — is real), so the evaluation sees the punishment.
    """
    for _ in range(ROLLOUT_STEP_CAP * plies):
        obs = state.observation
        if obs.current.result != -1 or obs.current.turn >= root_turn + plies:
            break
        if time.perf_counter() > deadline:
            break
        try:
            sel = LucarioPolicy(obs).choose()
            if not sel and obs.select.minCount > 0:
                sel = _fallback_selection(obs.select)
        except Exception:
            sel = _fallback_selection(obs.select)
        state = search_step(state.searchId, sel)
    return state.observation


def _lethal_search(obs: Observation, policy: LucarioPolicy) -> list[int] | None:
    """Override the MAIN choice only when a line certainly wins THIS turn.

    Runs only when our remaining prizes are within one turn's reach (<=3).
    A candidate is accepted only if every determinization sample ends the
    rollout with the game already won — no board evaluation is involved,
    so there is no way for a bad eval to overrule the tuned heuristic.
    """
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

    heuristic_scores = [policy._score_option(o) for o in select.option]
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


EVAL_CANDIDATES = 4
EVAL_SAMPLES = 2
OVERRIDE_MARGIN = 3000.0  # ~1/3 prize: override only on a material difference


def _eval_search_2ply(obs: Observation, policy: LucarioPolicy) -> list[int] | None:
    """Re-rank the top MAIN candidates by a 2-ply rollout evaluation.

    Unlike EXP-006 (1-ply, free override), the rollout continues through the
    opponent's reply turn so their real board's counter-attack is priced in,
    and the heuristic's own choice is only overridden when a candidate beats
    it by OVERRIDE_MARGIN on average.
    """
    global _search_time_used

    select = obs.select
    if select.minCount != 1 or select.maxCount != 1 or len(select.option) < 2:
        return None
    if _search_time_used > SEARCH_GAME_BUDGET:
        return None

    t0 = time.perf_counter()
    deadline = t0 + SEARCH_MOVE_BUDGET
    my_index = obs.current.yourIndex
    me = obs.current.players[my_index]
    op = obs.current.players[1 - my_index]
    root_my_prize = len(me.prize)
    root_op_prize = len(op.prize)
    root_turn = obs.current.turn

    heuristic_scores = [policy._score_option(o) for o in select.option]
    ranked = sorted(range(len(heuristic_scores)), key=lambda i: heuristic_scores[i], reverse=True)
    candidates = ranked[:EVAL_CANDIDATES]
    top1 = candidates[0]

    totals = {i: 0.0 for i in candidates}
    counts = {i: 0 for i in candidates}
    try:
        for _ in range(EVAL_SAMPLES):
            if time.perf_counter() > deadline:
                break
            root = search_begin(obs, *_determinize(obs))
            for idx in candidates:
                if time.perf_counter() > deadline:
                    break
                try:
                    child = search_step(root.searchId, [idx])
                    final_obs = _rollout(child, root_turn, deadline, plies=2)
                    totals[idx] += _evaluate(final_obs, my_index, root_my_prize, root_op_prize)
                    counts[idx] += 1
                except Exception:
                    totals[idx] += -1e8
                    counts[idx] += 1
    finally:
        try:
            search_end()
        except Exception:
            pass
        _search_time_used += time.perf_counter() - t0

    if counts[top1] != EVAL_SAMPLES:
        return None
    evaluated = [i for i in candidates if counts[i] == EVAL_SAMPLES]
    best = max(evaluated, key=lambda i: totals[i] / counts[i])
    if best == top1:
        return None
    if totals[best] / counts[best] > totals[top1] / counts[top1] + OVERRIDE_MARGIN:
        return [best]
    return None


def agent(obs_dict: dict) -> list[int]:
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        return my_deck

    global pre_turn
    global ability_used
    global plan

    if pre_turn != obs.current.turn:
        pre_turn = obs.current.turn
        ability_used = False
        plan = AttackPlan()

    policy = LucarioPolicy(obs)

    if obs.select.context == SelectContext.MAIN and obs.current.turn >= 2:
        # Set the heuristic plan for the real trajectory first (sub-selects rely on it),
        # then let the search override only the MAIN choice.
        policy._plan_attack()
        saved = (plan, pre_turn, ability_used)
        try:
            choice = _lethal_search(obs, policy)
            if choice is None:
                choice = _eval_search_2ply(obs, policy)
        except Exception:
            choice = None
        plan, pre_turn, ability_used = saved
        if choice is not None:
            return choice
        return policy.choose()

    return policy.choose()
