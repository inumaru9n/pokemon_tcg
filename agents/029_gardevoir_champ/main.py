"""029_gardevoir_champ — Mega Gardevoir ex axis (2025 Worlds champion adaptation).

islet skeleton (004) rewritten for the Gardevoir engine:
- Ralts -> Kirlia -> Mega Gardevoir ex (Rare Candy shortcut)
- Overflowing Wishes (1P): attach a Basic P energy from deck to every benched Pokemon
- Mega Symphonia (1P): 50 dmg x every P energy attached to all of my Pokemon
- Munkidori: Adrena-Brain moves 3 damage counters (needs D energy)
- Fezandipiti ex: Flip the Script draw
- Lillie's Clefairy ex: Fairy Zone (opponent Dragon weakness -> P x2)
"""
from __future__ import annotations

import os
from collections import defaultdict

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
)


class C:
    RALTS = 745
    KIRLIA = 746
    MEGA_GARDEVOIR_EX = 747
    MUNKIDORI = 112
    FEZANDIPITI_EX = 140
    CLEFAIRY_EX = 272

    P_ENERGY = 5
    D_ENERGY = 7

    RARE_CANDY = 1079
    DUSK_BALL = 1102
    POKE_PAD = 1152
    TR_GREAT_BALL = 1132
    NIGHT_STRETCHER = 1097
    BOSS_ORDERS = 1182
    CARMINE = 1192
    LILLIE_DETERMINATION = 1227
    SWITCH = 1123
    HERO_CAPE = 1159

    LILLIES_PEARL = 1172
    LEGACY_ENERGY = 12


# attack ids
ATK_COLLECT = 1074       # Ralts: draw a card
ATK_HEADBUTT = 1075      # Ralts: 10
ATK_CALL_SIGN = 1076     # Kirlia: search 3 pokemon
ATK_PSYSHOT = 1077       # Kirlia: 30
ATK_WISHES = 1078        # M Gardevoir: mass accel
ATK_SYMPHONIA = 1079     # M Gardevoir: 50 x P energy on my board
ATK_MIND_BEND = 141      # Munkidori: 60 + confusion (P+C)
ATK_CRUEL_ARROW = 183    # Fezandipiti: 100 snipe (CCC) -- unused
ATK_RONDO = 371          # Clefairy: 20 + 20 x benched both sides (P+C)

LOW_DECK_COUNT = 10
GARDE_LINE = {C.RALTS, C.KIRLIA, C.MEGA_GARDEVOIR_EX}


try:  # Kaggle eval loads main.py via exec(): no __file__
    DECK_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deck.csv")
except NameError:
    DECK_PATH = "deck.csv"
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
        attack_id: int = -1,
        remain_hp: int = -1,
        needs_energy: bool = False,
        energy_card: int = -1,
    ):
        self.attacker = attacker
        self.target = target
        self.attack_id = attack_id
        self.remain_hp = remain_hp
        self.needs_energy = needs_energy
        self.energy_card = energy_card


plan = AttackPlan()
pre_turn = -1


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
    score += pokemon.hp
    return score


def p_energy_count(pokemon: Pokemon) -> int:
    return sum(1 for e in pokemon.energies if e in (EnergyType.PSYCHIC, EnergyType.RAINBOW, EnergyType.TEAM_ROCKET))


class GardevoirPolicy:
    def __init__(self, obs: Observation):
        self.obs = obs
        self.state = obs.current
        self.select = obs.select
        self.context = self.select.context
        self.my_index = self.state.yourIndex
        self.op_index = 1 - self.my_index
        self.me = self.state.players[self.my_index]
        self.opponent = self.state.players[self.op_index]

        self.field_counts = defaultdict(int)
        self.hand_counts = defaultdict(int)
        self.discard_counts = defaultdict(int)
        self.can_switch = False
        self.can_gust = False
        self.can_attack = False

        self._count_cards()
        self._scan_main_options()

        self.my_board_p = sum(p_energy_count(p) for p in self._my_board() if p is not None)
        self.bench_count = sum(1 for p in self.me.bench if p is not None)

    def choose(self) -> list[int]:
        if not self.select.option or self.select.maxCount == 0:
            return []
        if self.context == SelectContext.MAIN:
            self._plan_attack()
        scores = [self._score_option(option) for option in self.select.option]
        ranked = [i for i, _ in sorted(enumerate(scores), key=lambda item: item[1], reverse=True)]
        return ranked[: self.select.maxCount]

    # ------------------------------------------------------------- helpers
    def _count_cards(self) -> None:
        for pokemon in self.me.active + self.me.bench:
            if pokemon is None:
                continue
            self.field_counts[pokemon.id] += 1
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
            elif option.type == OptionType.RETREAT:
                self.can_switch = True
            elif option.type == OptionType.ATTACK:
                self.can_attack = True

    def _my_board(self) -> list[Pokemon | None]:
        return self.me.active + self.me.bench

    def _opponent_board(self) -> list[Pokemon | None]:
        return self.opponent.active + self.opponent.bench

    def _opponent_has_dragon(self) -> bool:
        return any(
            p is not None and card_table[p.id].energyType == EnergyType.DRAGON
            for p in self._opponent_board()
        )

    def _garde_line_on_field(self) -> int:
        return (
            self.field_counts[C.RALTS]
            + self.field_counts[C.KIRLIA]
            + self.field_counts[C.MEGA_GARDEVOIR_EX]
        )

    def _my_damaged(self) -> bool:
        return any(p is not None and p.hp < p.maxHp for p in self._my_board())

    def _low_deck(self) -> bool:
        return self.me.deckCount <= LOW_DECK_COUNT

    # ------------------------------------------------------------- attack plan
    def _attack_candidates(self, pokemon: Pokemon):
        """yield (attack_id, p_req, total_req, damage)"""
        if pokemon.id == C.MEGA_GARDEVOIR_EX:
            yield (ATK_WISHES, 1, 1, 0)
            yield (ATK_SYMPHONIA, 1, 1, -1)  # damage computed by caller
        elif pokemon.id == C.KIRLIA:
            yield (ATK_PSYSHOT, 1, 1, 30)
        elif pokemon.id == C.RALTS:
            yield (ATK_HEADBUTT, 1, 1, 10)
        elif pokemon.id == C.MUNKIDORI:
            yield (ATK_MIND_BEND, 1, 2, 60)
        elif pokemon.id == C.CLEFAIRY_EX:
            op_bench = sum(1 for p in self.opponent.bench if p is not None)
            yield (ATK_RONDO, 1, 2, 20 + 20 * (self.bench_count + op_bench))

    def _energy_feasible(self, pokemon: Pokemon, p_req: int, total_req: int):
        """returns (ok, needs_energy, energy_card_id, extra_p)"""
        p_have = p_energy_count(pokemon)
        total_have = len(pokemon.energies)
        if p_have >= p_req and total_have >= total_req:
            return True, False, -1, 0
        if self.state.energyAttached:
            return False, False, -1, 0
        if self.hand_counts[C.P_ENERGY] >= 1 and p_have + 1 >= p_req and total_have + 1 >= total_req:
            return True, True, C.P_ENERGY, 1
        if self.hand_counts[C.D_ENERGY] >= 1 and p_have >= p_req and total_have + 1 >= total_req:
            return True, True, C.D_ENERGY, 0
        return False, False, -1, 0

    def _weakness_multiplier(self, op_data) -> float:
        if op_data.weakness == EnergyType.PSYCHIC:
            return 2.0
        if (
            self.field_counts[C.CLEFAIRY_EX] >= 1
            and op_data.energyType == EnergyType.DRAGON
        ):
            return 2.0  # Fairy Zone
        return 1.0

    def _plan_attack(self) -> None:
        global plan
        best_score = -1.0
        plan = AttackPlan()

        if self.state.turn < 2:
            return

        for attacker_index, my_pokemon in enumerate(self._my_board()):
            if my_pokemon is None:
                continue
            if attacker_index != 0 and not self.can_switch:
                break

            for attack_id, p_req, total_req, dmg in self._attack_candidates(my_pokemon):
                ok, needs_energy, energy_card, extra_p = self._energy_feasible(
                    my_pokemon, p_req, total_req
                )
                if not ok:
                    continue

                if attack_id == ATK_WISHES:
                    # setup attack: attach P from deck to every benched pokemon
                    if (
                        self.bench_count == 0
                        or self.my_board_p + extra_p >= 6
                        or self.me.deckCount == 0
                    ):
                        continue
                    score = 2600.0 + 150.0 * self.bench_count
                    score += 220 if attacker_index == 0 else 0
                    if score > best_score:
                        best_score = score
                        plan = AttackPlan(
                            attacker=attacker_index,
                            target=0,
                            attack_id=attack_id,
                            remain_hp=999,
                            needs_energy=needs_energy,
                            energy_card=energy_card,
                        )
                    continue

                if attack_id == ATK_SYMPHONIA:
                    dmg = 50 * (self.my_board_p + extra_p)
                if dmg <= 0:
                    continue

                for target_index, op_pokemon in enumerate(self._opponent_board()):
                    if op_pokemon is None:
                        continue
                    if target_index != 0 and not self.can_gust:
                        break

                    op_data = card_table[op_pokemon.id]
                    damage = int(dmg * self._weakness_multiplier(op_data))
                    if op_data.resistance == EnergyType.PSYCHIC:
                        damage = max(0, damage - 30)

                    score = float(target_score(op_pokemon))
                    prize = prize_count(op_pokemon) if op_pokemon.hp <= damage else 0
                    if prize == 0:
                        score *= damage / op_pokemon.hp
                    if len(self.opponent.prize) <= prize:
                        score = 50000
                    score += 220 if attacker_index == 0 else 0
                    score += 300 if target_index == 0 else 0
                    score += len(my_pokemon.energies)

                    if score > best_score:
                        best_score = score
                        plan = AttackPlan(
                            attacker=attacker_index,
                            target=target_index,
                            attack_id=attack_id,
                            remain_hp=op_pokemon.hp - damage,
                            needs_energy=needs_energy,
                            energy_card=energy_card,
                        )

    # ------------------------------------------------------------- option scoring
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
            return 1100 if option.attackId == plan.attack_id else 1000
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
            return self._energy_target_score(card, option.area == AreaType.ACTIVE, C.P_ENERGY)
        if self.context == SelectContext.REMOVE_DAMAGE_COUNTER and isinstance(card, Pokemon):
            # Adrena-Brain source: my most valuable damaged pokemon
            score = card.maxHp - card.hp
            if card.id == C.MEGA_GARDEVOIR_EX:
                score += 500
            return score
        if self.context in {SelectContext.DAMAGE_COUNTER, SelectContext.DAMAGE_COUNTER_ANY} and isinstance(card, Pokemon):
            # Adrena-Brain destination: prefer finishing off opponent pokemon
            if option.playerIndex != self.my_index:
                score = float(target_score(card))
                if card.hp <= 30:
                    score += 5000  # can be knocked out by the move
                return score
            return -1000  # never place counters on my own side if avoidable
        if self.context == SelectContext.DISCARD:
            return self._score_discard(card)
        if self.context == SelectContext.EVOLVES_FROM and isinstance(card, Pokemon):
            # Rare Candy: pick a Ralts that did not just come into play
            score = 10
            if card.id == C.RALTS and not card.appearThisTurn:
                score += 100
            score += len(card.energies)
            return score
        if self.context == SelectContext.EVOLVES_TO:
            return 100 if card.id == C.MEGA_GARDEVOIR_EX else 0
        return 0

    def _score_discard(self, card: Pokemon | Card) -> float:
        # lower score = kept, higher = discarded first? maxCount picks top-ranked to discard
        keep = {C.MEGA_GARDEVOIR_EX: 90, C.RARE_CANDY: 80, C.KIRLIA: 60, C.RALTS: 50,
                C.P_ENERGY: 40, C.BOSS_ORDERS: 35, C.HERO_CAPE: 30}
        return 100 - keep.get(card.id, 0)

    def _score_active_choice(self, option, card: Pokemon | Card) -> float:
        if not isinstance(card, Pokemon):
            return 0
        if option.playerIndex != self.my_index:
            return 100 if option.index == plan.target - 1 else 0

        score = len(card.energies) * 2
        if option.index == plan.attacker - 1:
            score += 100
        if card.id == C.MEGA_GARDEVOIR_EX:
            score += 30
        elif card.id == C.MUNKIDORI:
            score += 8
        elif card.id == C.KIRLIA:
            score += 6
        elif card.id == C.RALTS:
            score += 5
        elif card.id == C.FEZANDIPITI_EX:
            score += 3
        elif card.id == C.CLEFAIRY_EX:
            score += 2
        return score

    def _score_setup_active(self, card: Pokemon | Card) -> int:
        if card.id == C.RALTS:
            return 5
        if card.id == C.MUNKIDORI:
            return 3
        if card.id == C.CLEFAIRY_EX:
            return 2
        if card.id == C.FEZANDIPITI_EX:
            return 1
        return 0

    def _score_to_hand(self, card: Pokemon | Card) -> float:
        score = 200 - self.hand_counts[card.id] * 100
        line = self._garde_line_on_field()
        if card.id == C.RALTS:
            score += 60 if line < 3 else -20
        elif card.id == C.KIRLIA:
            score += 40 if self.field_counts[C.RALTS] >= 1 and self.hand_counts[C.RARE_CANDY] == 0 else 5
        elif card.id == C.MEGA_GARDEVOIR_EX:
            if self.field_counts[C.KIRLIA] >= 1 or (
                self.field_counts[C.RALTS] >= 1 and self.hand_counts[C.RARE_CANDY] >= 1
            ):
                score += 80
            else:
                score += 20
        elif card.id == C.RARE_CANDY:
            score += 50 if self.field_counts[C.RALTS] >= 1 else 15
        elif card.id == C.MUNKIDORI:
            score += 15 if self.field_counts[C.MUNKIDORI] == 0 else -40
        elif card.id == C.FEZANDIPITI_EX:
            score += 5 if self.field_counts[C.FEZANDIPITI_EX] == 0 else -60
        elif card.id == C.CLEFAIRY_EX:
            score += 35 if self._opponent_has_dragon() and self.field_counts[C.CLEFAIRY_EX] == 0 else -40
        elif card.id == C.P_ENERGY:
            score += 30 if not self.state.energyAttached else 5
        elif card.id == C.D_ENERGY:
            has_d = any(
                p is not None and p.id == C.MUNKIDORI and EnergyType.DARKNESS in p.energies
                for p in self._my_board()
            )
            score += 20 if self.field_counts[C.MUNKIDORI] >= 1 and not has_d else -30
        elif card.id == C.BOSS_ORDERS:
            score += 10
        return score

    def _score_play(self, option) -> float:
        card = get_card(self.obs, AreaType.HAND, option.index, self.my_index)
        data = card_table[card.id]
        if data.cardType == CardType.POKEMON:
            return self._score_play_pokemon(card)
        return self._score_play_trainer(card)

    def _score_play_pokemon(self, card: Card) -> float:
        if card.id == C.RALTS:
            line = self._garde_line_on_field()
            return 20000 if line < 3 else -1
        if card.id == C.MUNKIDORI:
            return 19000 if self.field_counts[C.MUNKIDORI] == 0 else 500
        if card.id == C.FEZANDIPITI_EX:
            return 18000
        if card.id == C.CLEFAIRY_EX:
            if self._opponent_has_dragon():
                return 19500
            return 15000 if self.bench_count < 3 else -1
        return 17000

    def _score_play_trainer(self, card: Card) -> float:
        if card.id == C.SWITCH:
            return 6000 if plan.attacker > 0 else -1
        if card.id == C.BOSS_ORDERS:
            return 3200 if plan.target >= 1 else -1
        if card.id == C.CARMINE:
            return -1 if self._low_deck() else 3000
        if card.id == C.LILLIE_DETERMINATION:
            return -1 if self._low_deck() else 3100
        if card.id == C.RARE_CANDY:
            ready_ralts = any(
                p is not None and p.id == C.RALTS and not p.appearThisTurn
                for p in self._my_board()
            )
            if ready_ralts and self.hand_counts[C.MEGA_GARDEVOIR_EX] >= 1 and self.state.turn >= 3:
                return 11000
            return -1
        if card.id == C.NIGHT_STRETCHER:
            # only if there is something worth recovering
            useful = (
                self.discard_counts[C.MEGA_GARDEVOIR_EX]
                + self.discard_counts[C.RALTS]
                + self.discard_counts[C.KIRLIA]
                + self.discard_counts[C.P_ENERGY]
            )
            return 9000 if useful else -1
        if card.id in {C.DUSK_BALL, C.POKE_PAD, C.TR_GREAT_BALL}:
            return 10000
        return 8000

    def _energy_target_score(self, pokemon: Pokemon, active: bool, energy_id: int) -> float:
        board_p = p_energy_count(pokemon)
        score = 8000 + (10 if active else 0)

        if energy_id == C.D_ENERGY:
            if pokemon.id == C.MUNKIDORI:
                has_d = EnergyType.DARKNESS in pokemon.energies
                return score + (150 if not has_d else -50)
            return 200  # darkness is only for Munkidori

        # psychic energy: every attachment feeds Mega Symphonia
        if pokemon.id == C.MEGA_GARDEVOIR_EX:
            score += 200 if board_p < 1 else 60
        elif pokemon.id in {C.RALTS, C.KIRLIA}:
            score += 100 if board_p < 1 else 20
        elif pokemon.id == C.MUNKIDORI:
            score += 40 if len(pokemon.energies) < 2 else 10
        elif pokemon.id == C.CLEFAIRY_EX:
            score += 30 if len(pokemon.energies) < 2 else 10
        elif pokemon.id == C.FEZANDIPITI_EX:
            score += 15
        return score

    def _score_attach(self, option) -> float:
        card = get_card(self.obs, AreaType.HAND, option.index, self.my_index)
        pokemon = get_card(self.obs, option.inPlayArea, option.inPlayIndex, self.my_index)
        if not isinstance(pokemon, Pokemon):
            return 0

        if card.id == C.HERO_CAPE:
            if pokemon.id == C.MEGA_GARDEVOIR_EX:
                return 12000
            if pokemon.id == C.KIRLIA:
                return 7100
            return 7000

        score = self._energy_target_score(pokemon, option.inPlayArea == AreaType.ACTIVE, card.id)
        board_index = option.inPlayIndex if option.inPlayArea == AreaType.ACTIVE else option.inPlayIndex + 1
        if board_index == plan.attacker and plan.needs_energy and card.id == plan.energy_card:
            score += 400
        return score

    def _score_evolve(self, option) -> float:
        pokemon = get_card(self.obs, option.inPlayArea, option.inPlayIndex, self.my_index)
        if not isinstance(pokemon, Pokemon):
            return 0
        return 9000 + len(pokemon.energies)

    def _score_ability(self, option) -> float:
        card = get_card(self.obs, option.area, option.index, self.my_index)
        if card is not None and card.id == C.MUNKIDORI:
            # Adrena-Brain: only when we actually have damage to move
            return 30000 if self._my_damaged() else -1
        return 30000


def agent(obs_dict: dict) -> list[int]:
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        return my_deck

    global pre_turn
    global plan

    if pre_turn != obs.current.turn:
        pre_turn = obs.current.turn
        plan = AttackPlan()

    return GardevoirPolicy(obs).choose()
