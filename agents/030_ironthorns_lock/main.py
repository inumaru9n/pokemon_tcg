"""030_ironthorns_lock — Iron Thorns ex ability-lock deck (EXP-030, idea I-060).

Deck concept:
- Iron Thorns ex (37): Initialization locks all Rule Box abilities (both sides,
  Future excluded) while it is Active. Volt Cyclone [L,C,C] 140, moves 1 energy
  to bench each attack (self-charging the backup attacker).
- Miraidon (87): Peak Acceleration [C] searches 2 basic energies onto Future
  Pokemon (accel). Iron Crown ex (80): Cobalt Command +20 damage to Future
  attackers (Future -> works under our own lock).
- Disruption package: Crushing Hammer / Judge / Boss's Orders. Mono-Lightning.

Policy: islet-style scoring skeleton (based on agents/004), rewritten for this
deck. Keep Iron Thorns active (lock uptime), avoid voluntary retreats
(retreat cost 4), switch only via Switch card to a charged backup.
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
    IRON_THORNS_EX = 37
    MIRAIDON = 87
    IRON_CROWN_EX = 80

    L_ENERGY = 4

    ULTRA_BALL = 1121
    DUSK_BALL = 1102
    SWITCH = 1123
    CRUSHING_HAMMER = 1120
    REBOOT_POD = 1089
    NIGHT_STRETCHER = 1097
    ENERGY_RETRIEVAL = 1118
    HERO_CAPE = 1159
    BOSS_ORDERS = 1182
    CARMINE = 1192
    LILLIE_DETERMINATION = 1227
    JUDGE = 1213
    LEVINCIA = 1254

    LILLIES_PEARL = 1172
    LEGACY_ENERGY = 12


VOLT_CYCLONE = 29        # Iron Thorns ex: 140, move 1 energy to bench
PEAK_ACCELERATION = 105  # Miraidon: 40, search 2 basic energy onto Future
LOW_DECK_COUNT = 8

try:  # Kaggle評価環境はexecロードのため __file__ が無い
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
    ):
        self.attacker = attacker
        self.target = target
        self.attack_id = attack_id
        self.remain_hp = remain_hp
        self.needs_energy = needs_energy


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
    if pokemon.id == 112 and len(pokemon.energies) >= 1:  # Munkidori
        score += 300
    score += pokemon.hp
    return score


class ThornsPolicy:
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
        self.crown_in_play = 0
        self.can_switch = False
        self.can_gust = False
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
        return ranked[: self.select.maxCount]

    # ---------- board scan ----------

    def _count_cards(self) -> None:
        for pokemon in self._my_board():
            if pokemon is None:
                continue
            self.field_counts[pokemon.id] += 1
            if pokemon.id == C.IRON_CROWN_EX:
                self.crown_in_play += 1
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

    def _my_board(self) -> list[Pokemon | None]:
        return self.me.active + self.me.bench

    def _opponent_board(self) -> list[Pokemon | None]:
        return self.opponent.active + self.opponent.bench

    def _low_deck(self) -> bool:
        return self.me.deckCount <= LOW_DECK_COUNT

    def _deck_energy_left(self) -> int:
        attached = 0
        for pokemon in self._my_board():
            if pokemon is not None:
                attached += sum(1 for card in pokemon.energyCards if card.id == C.L_ENERGY)
        return 12 - attached - self.hand_counts[C.L_ENERGY] - self.discard_counts[C.L_ENERGY]

    def _board_needs_energy(self) -> bool:
        for pokemon in self._my_board():
            if pokemon is None:
                continue
            if pokemon.id == C.IRON_THORNS_EX and len(pokemon.energies) < 3:
                return True
        return False

    # ---------- attack planning ----------

    def _base_attack(self, pokemon: Pokemon) -> tuple[int, int, int, int] | None:
        """(attack_id, energy_required, base_damage, base_score)"""
        if pokemon.id == C.IRON_THORNS_EX:
            return VOLT_CYCLONE, 3, 140 + 20 * self.crown_in_play, 0
        if pokemon.id == C.MIRAIDON:
            bonus = 0
            if self._board_needs_energy() and self._deck_energy_left() >= 1:
                bonus = 450  # Peak Acceleration = 2 energy accel onto Future
            return PEAK_ACCELERATION, 1, 40, bonus
        return None

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

            attack = self._base_attack(my_pokemon)
            if attack is None:
                continue
            attack_id, energy_required, base_damage, base_score = attack

            energy_count = len(my_pokemon.energies)
            needs_energy = False
            if energy_count < energy_required:
                if (
                    attacker_index == 0
                    and self.hand_counts[C.L_ENERGY] >= 1
                    and not self.state.energyAttached
                ):
                    energy_count += 1
                    needs_energy = energy_count >= energy_required
                if not (energy_count >= energy_required):
                    continue

            for target_index, op_pokemon in enumerate(self._opponent_board()):
                if op_pokemon is None:
                    continue
                if target_index != 0 and not self.can_gust:
                    break

                damage = base_damage
                op_data = card_table[op_pokemon.id]
                if op_data.weakness == EnergyType.LIGHTNING:
                    damage *= 2
                elif op_data.resistance == EnergyType.LIGHTNING:
                    damage -= 30

                score = target_score(op_pokemon)
                prize = prize_count(op_pokemon) if op_pokemon.hp <= damage else 0
                if prize == 0:
                    score *= damage / op_pokemon.hp
                if len(self.opponent.prize) <= prize:
                    score = 50000

                score += base_score
                if attacker_index == 0:
                    # keep the active attacking, but a nearly-dead active should
                    # yield to a healthy charged backup when Switch is available
                    score += 220 if my_pokemon.hp >= 100 else 40
                score += 300 if target_index == 0 else 0
                score += energy_count

                if score > best_score:
                    best_score = score
                    plan = AttackPlan(
                        attacker=attacker_index,
                        target=target_index,
                        attack_id=attack_id,
                        remain_hp=op_pokemon.hp - damage,
                        needs_energy=needs_energy,
                    )

    # ---------- option scoring ----------

    def _score_option(self, option) -> float:
        if option.type == OptionType.NUMBER:
            return option.number
        if option.type == OptionType.YES:
            return 100 if self.context == SelectContext.IS_FIRST else 1
        if option.type == OptionType.NO:
            return 0
        if option.type == OptionType.CARD:
            return self._score_card_choice(option)
        if option.type == OptionType.ENERGY_CARD:
            return self._score_energy_card(option)
        if option.type == OptionType.PLAY:
            return self._score_play(option)
        if option.type == OptionType.ATTACH:
            return self._score_attach(option)
        if option.type == OptionType.ABILITY:
            return self._score_ability(option)
        if option.type == OptionType.RETREAT:
            return self._score_retreat()
        if option.type == OptionType.ATTACK:
            return 1100 if option.attackId == plan.attack_id else 1000
        return 0

    def _score_retreat(self) -> float:
        # Retreat discards energy; Thorns costs 4. Only bail out on a non-Thorns
        # active when a planned attacker waits on the bench and no Switch in hand.
        if plan.attacker < 1:
            return -1
        active = self.me.active[0] if self.me.active else None
        if active is not None and active.id == C.IRON_THORNS_EX:
            return -1
        return 1500

    # ---------- card choices ----------

    def _score_card_choice(self, option) -> float:
        card = get_card(self.obs, option.area, option.index, option.playerIndex)
        if card is None:
            return 0

        if self.context in {SelectContext.SWITCH, SelectContext.TO_ACTIVE}:
            return self._score_active_choice(option, card)
        if self.context == SelectContext.SETUP_ACTIVE_POKEMON:
            return self._score_setup(card, active=True)
        if self.context == SelectContext.SETUP_BENCH_POKEMON:
            return self._score_setup(card, active=False)
        if self.context == SelectContext.TO_HAND:
            return self._score_to_hand(card)
        if self.context == SelectContext.DISCARD:
            return self._score_discard(card)
        if self.context == SelectContext.ATTACH_FROM and isinstance(card, Pokemon):
            return self._energy_target_score(card, option.area == AreaType.ACTIVE)
        if option.area == AreaType.DECK and not isinstance(card, Pokemon):
            # e.g. Peak Acceleration: pick basic L energies out of the deck
            return 50 if card.id == C.L_ENERGY else 0
        return 0

    def _score_active_choice(self, option, card: Pokemon | Card) -> float:
        if not isinstance(card, Pokemon):
            return 0
        if option.playerIndex != self.my_index:
            # Boss's Orders: drag out the planned target
            return 100 if option.index == plan.target - 1 else 0

        score = len(card.energies) * 3
        if option.index == plan.attacker - 1:
            score += 100
        if card.id == C.IRON_THORNS_EX:
            score += 25  # lock uptime + only real attacker
        elif card.id == C.MIRAIDON:
            score += 6 if len(card.energies) >= 1 else 4
        elif card.id == C.IRON_CROWN_EX:
            score += 1
        return score

    def _score_setup(self, card: Pokemon | Card, active: bool) -> int:
        if card.id == C.IRON_THORNS_EX:
            return 5
        if card.id == C.MIRAIDON:
            return 4
        if card.id == C.IRON_CROWN_EX:
            return 1 if active else 3
        return 0

    def _score_to_hand(self, card: Pokemon | Card) -> float:
        score = 200 - self.hand_counts[card.id] * 100
        if card.id == C.IRON_THORNS_EX:
            score += 100 if self.field_counts[card.id] < 2 else -20
        elif card.id == C.MIRAIDON:
            score += 50 if self.field_counts[card.id] < 1 else -30
        elif card.id == C.IRON_CROWN_EX:
            score += 10 if self.field_counts[card.id] < 2 else -100
        elif card.id == C.L_ENERGY:
            score += 60 if self.hand_counts[card.id] == 0 and not self.state.energyAttached else -10
        return score

    def _score_discard(self, card: Pokemon | Card) -> float:
        keep = 30
        if card.id == C.HERO_CAPE:
            keep = 100
        elif card.id == C.BOSS_ORDERS:
            keep = 80
        elif card.id == C.IRON_THORNS_EX:
            keep = 70 if self.field_counts[card.id] < 3 else 35
        elif card.id in {C.CARMINE, C.LILLIE_DETERMINATION}:
            keep = 60
        elif card.id == C.SWITCH:
            keep = 55
        elif card.id == C.MIRAIDON:
            keep = 50 if self.field_counts[card.id] == 0 else 15
        elif card.id == C.ULTRA_BALL:
            keep = 45
        elif card.id == C.DUSK_BALL:
            keep = 40
        elif card.id == C.NIGHT_STRETCHER:
            keep = 35
        elif card.id == C.CRUSHING_HAMMER:
            keep = 28
        elif card.id == C.JUDGE:
            keep = 25
        elif card.id == C.LEVINCIA:
            keep = 25 if self.stadium_id != C.LEVINCIA else 5
        elif card.id == C.ENERGY_RETRIEVAL:
            keep = 30
        elif card.id == C.IRON_CROWN_EX:
            keep = 30 if self.field_counts[card.id] < 2 else 8
        elif card.id == C.L_ENERGY:
            # discarded energy fuels Reboot Pod / Energy Retrieval
            keep = 26 if self.hand_counts[card.id] <= 1 else 12
        return 100 - keep

    def _score_energy_card(self, option) -> float:
        # e.g. Crushing Hammer: discard an energy on an opponent's Pokemon.
        if option.playerIndex == self.my_index:
            return 0
        pokemon = get_card(self.obs, option.area, option.index, option.playerIndex)
        if not isinstance(pokemon, Pokemon):
            return 0
        score = 50
        if option.area == AreaType.ACTIVE:
            score += 40  # slow down the current attacker
        if len(pokemon.energies) == 1:
            score += 20  # full denial
        return score

    # ---------- energy placement ----------

    def _energy_target_score(self, pokemon: Pokemon, active: bool) -> float:
        count = len(pokemon.energies)
        score = 8000 + (10 if active else 0)
        if pokemon.id == C.IRON_THORNS_EX:
            if count < 3:
                score += 500 + count * 40 + (250 if active else 0)
            else:
                score -= 150
        elif pokemon.id == C.MIRAIDON:
            score += 80 if count < 1 else -200
        elif pokemon.id == C.IRON_CROWN_EX:
            score -= 300
        return score

    def _score_attach(self, option) -> float:
        card = get_card(self.obs, AreaType.HAND, option.index, self.my_index)
        pokemon = get_card(self.obs, option.inPlayArea, option.inPlayIndex, self.my_index)
        if not isinstance(pokemon, Pokemon):
            return 0

        if card.id == C.HERO_CAPE:
            if pokemon.id == C.IRON_THORNS_EX:
                return 12000 if option.inPlayArea == AreaType.ACTIVE else 7200
            return 6000

        score = self._energy_target_score(pokemon, option.inPlayArea == AreaType.ACTIVE)
        board_index = option.inPlayIndex if option.inPlayArea == AreaType.ACTIVE else option.inPlayIndex + 1
        if board_index == plan.attacker and plan.needs_energy:
            score += 400
        return score

    # ---------- trainers ----------

    def _score_play(self, option) -> float:
        card = get_card(self.obs, AreaType.HAND, option.index, self.my_index)
        data = card_table[card.id]
        if data.cardType == CardType.POKEMON:
            return self._score_play_pokemon(card)
        return self._score_play_trainer(card)

    def _score_play_pokemon(self, card: Card) -> float:
        if card.id == C.IRON_THORNS_EX:
            return 21000
        if card.id == C.MIRAIDON:
            return 19000 if self.field_counts[card.id] < 2 else -1
        if card.id == C.IRON_CROWN_EX:
            return 20000 if self.field_counts[card.id] < 2 else -1
        return 18000

    def _score_play_trainer(self, card: Card) -> float:
        cid = card.id
        if cid == C.SWITCH:
            return 6000 if plan.attacker >= 1 else -1
        if cid == C.BOSS_ORDERS:
            return 3200 if plan.target >= 1 else -1
        if cid == C.CARMINE:
            return -1 if self._low_deck() else 3000
        if cid == C.LILLIE_DETERMINATION:
            return -1 if self._low_deck() else 3100
        if cid == C.JUDGE:
            return -1 if self._low_deck() else 2900
        if cid == C.CRUSHING_HAMMER:
            return 4500
        if cid == C.DUSK_BALL:
            return 4700
        if cid == C.ULTRA_BALL:
            if self.field_counts[C.IRON_THORNS_EX] == 0 and self.hand_counts[C.IRON_THORNS_EX] == 0:
                return 6500
            if self.field_counts[C.IRON_THORNS_EX] < 2:
                return 4600
            return 1500
        if cid == C.NIGHT_STRETCHER:
            if self.discard_counts[C.IRON_THORNS_EX] >= 1 and self.field_counts[C.IRON_THORNS_EX] < 2:
                return 2600
            if self.discard_counts[C.L_ENERGY] >= 1 and self.hand_counts[C.L_ENERGY] == 0:
                return 2500
            return -1
        if cid == C.ENERGY_RETRIEVAL:
            if self.discard_counts[C.L_ENERGY] >= 1 and self.hand_counts[C.L_ENERGY] == 0:
                return 2550
            return -1
        if cid == C.LEVINCIA:
            return 2500 if self.stadium_id != C.LEVINCIA else -1
        return 10000

    def _score_ability(self, option) -> float:
        card = get_card(self.obs, option.area, option.index, self.my_index)
        if card is not None and card.id == C.LEVINCIA:
            return 25000 if self.discard_counts[C.L_ENERGY] >= 1 else -1
        return 30000


def agent(obs_dict: dict) -> list[int]:
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        return my_deck

    global pre_turn, plan
    if pre_turn != obs.current.turn:
        pre_turn = obs.current.turn
        plan = AttackPlan()

    return ThornsPolicy(obs).choose()
