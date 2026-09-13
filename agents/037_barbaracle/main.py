"""037_barbaracle — Barbaracle/Okidogi 非ルールボックス地雷デッキ (sig 35ce7e0e55, 本番226戦60.6%).

islet骨格 (004) をBarbaracleデッキ用に書き換えたヒューリスティック方策。
Majkel1337個体のリプレイ226戦の行動集計に基づく:
- 先攻は常にYES / セットアップアクティブはSolrock優先
- Prism Energy → Okidogi が最重要アタッチ (Adrena-Power: +100HP/+100dmg)
- Good Punch 170 が主砲、Cosmic Beam 70 が副砲、Mad Bite はフィニッシャー
- Lunar Cycle / Stone Arms を毎ターン回す
- Neutralization Zone で相手ex/Vのワザダメージを無効化、Battle Cage は汎用/スタジアム潰し
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
    F_ENERGY = 6
    PRISM = 16

    OKIDOGI = 116
    URSALUNA = 135
    LUNATONE = 675
    SOLROCK = 676
    BINACLE = 1051
    BARBARACLE = 1052

    NIGHT_STRETCHER = 1097
    FIGHTING_GONG = 1142
    POKE_PAD = 1152
    AIR_BALLOON = 1174
    BOSS_ORDERS = 1182
    COLRESS = 1194
    XEROSIC = 1197
    JUDGE = 1213
    LILLIE = 1227
    TARRAGON = 1238
    NZONE = 1247
    BATTLE_CAGE = 1264

    LILLIES_PEARL = 1172
    LEGACY_ENERGY = 12


class A:
    GOOD_PUNCH = 147
    MAD_BITE = 175
    POWER_GEM = 979
    COSMIC_BEAM = 980
    DOUBLE_DRAW = 1519
    SCRATCH = 1520
    HAMMER_IN = 1521


LOW_DECK_COUNT = 10

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
    try:
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
    except (IndexError, TypeError):
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


def has_prism(pokemon: Pokemon) -> bool:
    if any(card.id == C.PRISM for card in pokemon.energyCards):
        return True
    return EnergyType.RAINBOW in pokemon.energies


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


class BarbaraclePolicy:
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
        self.stadium_id = self.state.stadium[0].id if self.state.stadium else 0
        self.stadium_is_mine = bool(self.state.stadium) and self.state.stadium[0].playerIndex == self.my_index

        self._count_cards()
        self._scan_main_options()

        self.hand_f = self.hand_counts[C.F_ENERGY]
        self.hand_prism = self.hand_counts[C.PRISM]
        self.barbaracle_in_play = self.field_counts[C.BARBARACLE] >= 1
        self.lunatone_on_bench = any(p is not None and p.id == C.LUNATONE for p in self.me.bench)

    def choose(self) -> list[int]:
        if not self.select.option or self.select.maxCount == 0:
            return []
        if self.context == SelectContext.MAIN:
            self._plan_attack()
        scores = [self._score_option(option) for option in self.select.option]
        ranked = [i for i, _ in sorted(enumerate(scores), key=lambda item: item[1], reverse=True)]
        return ranked[: self.select.maxCount]

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
                if card is not None and card.id == C.BOSS_ORDERS:
                    self.can_gust = True
            elif option.type == OptionType.RETREAT:
                self.can_switch = True
            elif option.type == OptionType.ATTACK:
                self.can_attack = True

    def _my_board(self) -> list[Pokemon | None]:
        return self.me.active + self.me.bench

    def _opponent_board(self) -> list[Pokemon | None]:
        return self.opponent.active + self.opponent.bench

    def _opponent_has_rulebox(self) -> bool:
        for pokemon in self._opponent_board():
            if pokemon is None:
                continue
            data = card_table[pokemon.id]
            if data.ex or data.megaEx:
                return True
        for card in self.opponent.discard:
            data = card_table.get(card.id)
            if data is not None and (data.ex or data.megaEx):
                return True
        return False

    # ---- 攻撃プラン ----

    def _attack_candidates(self, pokemon: Pokemon) -> list[tuple[int, int, int, bool, int]]:
        """(attack_id, energy_required, base_damage, apply_wr, base_score)"""
        result = []
        if pokemon.id == C.OKIDOGI:
            damage = 70 + (100 if has_prism(pokemon) else 0)
            result.append((A.GOOD_PUNCH, 2, damage, True, 0))
        elif pokemon.id == C.SOLROCK:
            if self.lunatone_on_bench:
                result.append((A.COSMIC_BEAM, 1, 70, False, 0))
        elif pokemon.id == C.URSALUNA:
            result.append((A.MAD_BITE, 3, -1, True, 0))  # -1: target依存
        elif pokemon.id == C.BARBARACLE:
            result.append((A.HAMMER_IN, 3, 80, True, -300))
        elif pokemon.id == C.LUNATONE:
            result.append((A.POWER_GEM, 2, 50, True, -400))
        elif pokemon.id == C.BINACLE:
            result.append((A.SCRATCH, 2, 30, True, -500))
            result.append((A.DOUBLE_DRAW, 1, 0, True, -800))
        return result

    def _attach_budget(self) -> int:
        """このターン中にまだ追加できるエネルギー枚数の概算(手貼り+Stone Arms)"""
        hand_energy = self.hand_f + self.hand_prism
        budget = 0
        if not self.state.energyAttached and hand_energy > 0:
            budget += 1
        if self.barbaracle_in_play and self.hand_f > 0:
            budget += 1
        return min(budget, hand_energy)

    def _plan_attack(self) -> None:
        global plan
        best_score = -1
        plan = AttackPlan()
        if self.state.turn < 2:
            return

        budget = self._attach_budget()
        for attacker_index, my_pokemon in enumerate(self._my_board()):
            if my_pokemon is None:
                continue
            if attacker_index != 0 and not self.can_switch:
                break
            for attack_id, energy_required, base_damage, apply_wr, base_score in self._attack_candidates(my_pokemon):
                energy_count = len(my_pokemon.energies)
                needs_energy = False
                if energy_count < energy_required:
                    if attacker_index == 0 and energy_count + budget >= energy_required:
                        needs_energy = True
                    elif attacker_index != 0 and energy_count + min(budget, 1) >= energy_required:
                        # ベンチアタッカーへは手貼り/Stone Armsどちらか1枚まで見込む
                        needs_energy = True
                    else:
                        continue

                for target_index, op_pokemon in enumerate(self._opponent_board()):
                    if op_pokemon is None:
                        continue
                    if target_index != 0 and not self.can_gust:
                        break

                    if base_damage < 0:  # Mad Bite
                        counters = max(0, (op_pokemon.maxHp - op_pokemon.hp) // 10)
                        damage = 100 + 30 * counters
                    else:
                        damage = base_damage
                    if apply_wr and damage > 0:
                        op_data = card_table[op_pokemon.id]
                        if op_data.weakness == EnergyType.FIGHTING:
                            damage *= 2
                        elif op_data.resistance == EnergyType.FIGHTING:
                            damage = max(0, damage - 30)

                    score = target_score(op_pokemon)
                    prize = prize_count(op_pokemon) if op_pokemon.hp <= damage else 0
                    if prize == 0:
                        score *= damage / max(1, op_pokemon.hp)
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
                            attack_id=attack_id,
                            remain_hp=op_pokemon.hp - damage,
                            needs_energy=needs_energy,
                        )

    # ---- エネルギー・ツールのアタッチ先 ----

    def _energy_target_score(self, pokemon: Pokemon, active: bool, energy_id: int = C.F_ENERGY) -> int:
        energy_count = len(pokemon.energies)
        score = 8000 + (10 if active else 0)

        if energy_id == C.PRISM:
            if pokemon.id == C.OKIDOGI:
                score += 4000 if not has_prism(pokemon) else -1500
            elif pokemon.id == C.URSALUNA:
                score += 500 if energy_count < 3 else -1000
            elif pokemon.id == C.SOLROCK:
                score += 200 if energy_count < 1 else -1200
            else:
                score -= 1500
            return score

        if pokemon.id == C.OKIDOGI:
            score += 800 if energy_count < 2 else -600
            score += 150 if has_prism(pokemon) else 0
        elif pokemon.id == C.SOLROCK:
            score += 600 if energy_count < 1 else -800
        elif pokemon.id == C.URSALUNA:
            score += 400 if energy_count < 3 else -700
        elif pokemon.id == C.LUNATONE:
            score -= 900
        elif pokemon.id in {C.BINACLE, C.BARBARACLE}:
            score -= 850
        return score

    def _board_index(self, area: AreaType, index: int) -> int:
        return index if area == AreaType.ACTIVE else index + 1

    def _score_attach(self, option) -> float:
        card = get_card(self.obs, option.area, option.index, self.my_index)
        pokemon = get_card(self.obs, option.inPlayArea, option.inPlayIndex, self.my_index)
        if card is None or not isinstance(pokemon, Pokemon):
            return 0

        if card.id == C.AIR_BALLOON:
            score = 6000
            if pokemon.id == C.SOLROCK:
                score += 300
            elif pokemon.id == C.LUNATONE:
                score += 200
            elif pokemon.id == C.OKIDOGI:
                score += 100
            if pokemon.tools:
                score = -1
            return score

        score = self._energy_target_score(pokemon, option.inPlayArea == AreaType.ACTIVE, card.id)
        board_index = self._board_index(option.inPlayArea, option.inPlayIndex)
        if board_index == plan.attacker and plan.needs_energy:
            score += 2000
        return score

    # ---- オプション採点 ----

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
            return 9000
        if option.type == OptionType.ABILITY:
            return self._score_ability(option)
        if option.type == OptionType.RETREAT:
            return 2000 if plan.attacker >= 1 else -1
        if option.type == OptionType.ATTACK:
            return 1100 if option.attackId == plan.attack_id else 1000
        return 0

    def _score_ability(self, option) -> float:
        card = get_card(self.obs, option.area, option.index, self.my_index)
        if card is None:
            return 0
        if card.id == C.LUNATONE:
            if self.me.deckCount <= 4:
                return -1
            if self.hand_f <= 1 and not self.state.energyAttached:
                return 7000  # 最後の闘エネはまず手貼り/Stone Armsに回す
            return 30000
        if card.id == C.BARBARACLE:
            if self.hand_f >= 2:
                return 29000
            return 7500
        return 25000  # 相手スタジアムの起動効果等

    def _score_play(self, option) -> float:
        card = get_card(self.obs, AreaType.HAND, option.index, self.my_index)
        if card is None:
            return 0
        data = card_table[card.id]
        if data.cardType == CardType.POKEMON:
            return self._score_play_pokemon(card)
        return self._score_play_trainer(card)

    def _score_play_pokemon(self, card: Card) -> float:
        if card.id == C.LUNATONE:
            return -1 if self.field_counts[C.LUNATONE] >= 1 else 20000
        if card.id == C.SOLROCK:
            return -1 if self.field_counts[C.SOLROCK] >= 2 else 20000
        if card.id == C.BINACLE:
            line = self.field_counts[C.BINACLE] + self.field_counts[C.BARBARACLE]
            return -1 if line >= 1 else 19500
        if card.id == C.OKIDOGI:
            return 20000 if self.field_counts[C.OKIDOGI] < 2 else 13000
        if card.id == C.URSALUNA:
            board = sum(1 for p in self._my_board() if p is not None)
            if self.hand_f >= 2:
                return 15000
            if board <= 2:
                return 12000
            return -1
        return 18000

    def _score_play_trainer(self, card: Card) -> float:
        cid = card.id
        if cid == C.NZONE:
            if self._opponent_has_rulebox():
                return 3400
            if self.stadium_id and not self.stadium_is_mine:
                return 1300  # 相手スタジアム潰し
            return -1
        if cid == C.BATTLE_CAGE:
            if self.stadium_id == C.NZONE and self.stadium_is_mine:
                return -1  # 自分のNZを上書きしない
            return 3150
        if cid == C.BOSS_ORDERS:
            return 3200 if plan.target >= 1 else -1
        if cid == C.COLRESS:
            return 3050 if self.hand_counts[C.NZONE] + self.hand_counts[C.BATTLE_CAGE] == 0 else -1
        if cid == C.LILLIE:
            if self._low_deck():
                return -1
            if len(self.me.hand) >= 7 and plan.attacker >= 0:
                return -1
            return 3100
        if cid == C.JUDGE:
            return 2900 if len(self.me.hand) <= 3 and not self._low_deck() else -1
        if cid == C.XEROSIC:
            return 2750 if self.opponent.handCount >= 6 else -1
        if cid == C.TARRAGON:
            recoverable = self.discard_counts[C.F_ENERGY] + sum(
                self.discard_counts[i] for i in (C.OKIDOGI, C.SOLROCK, C.LUNATONE, C.BINACLE, C.BARBARACLE, C.URSALUNA)
            )
            return 3000 if recoverable >= 2 else -1
        if cid == C.NIGHT_STRETCHER:
            return 10000
        if cid in (C.FIGHTING_GONG, C.POKE_PAD):
            return 10000
        return 8000

    # ---- カード選択（サーチ・入替等） ----

    def _score_card_choice(self, option) -> float:
        card = get_card(self.obs, option.area, option.index, option.playerIndex)
        if card is None:
            return 0
        if self.context in {SelectContext.SWITCH, SelectContext.TO_ACTIVE}:
            return self._score_active_choice(option, card)
        if self.context == SelectContext.SETUP_ACTIVE_POKEMON:
            return self._score_setup_active(card)
        if self.context == SelectContext.SETUP_BENCH_POKEMON:
            return self._score_setup_bench(card)
        if self.context == SelectContext.TO_HAND:
            return self._score_to_hand(card)
        if self.context == SelectContext.ATTACH_FROM and isinstance(card, Pokemon):
            energy_id = self.select.effect.id if self.select.effect is not None else C.F_ENERGY
            score = self._energy_target_score(card, option.area == AreaType.ACTIVE, energy_id)
            board_index = self._board_index(option.area, option.index)
            if board_index == plan.attacker and plan.needs_energy:
                score += 2000
            return score
        if self.context in {SelectContext.DISCARD, SelectContext.TO_DECK, SelectContext.TO_DECK_BOTTOM}:
            return self._score_discard(card)
        return 0

    def _score_active_choice(self, option, card: Pokemon | Card) -> float:
        if not isinstance(card, Pokemon):
            return 0
        if option.playerIndex != self.my_index:
            return 100 if option.index == plan.target - 1 else 0

        score = len(card.energies) * 2
        if option.index == plan.attacker - 1:
            score += 200
        if card.id == C.OKIDOGI:
            score += 50 if has_prism(card) else 25
        elif card.id == C.SOLROCK:
            score += 30
        elif card.id == C.URSALUNA:
            score += 20 if len(card.energies) >= 2 else 8
        elif card.id == C.BARBARACLE:
            score += 10
        elif card.id == C.LUNATONE:
            score -= 50  # Cosmic Beamの条件(ベンチのLunatone)を守る
        elif card.id == C.BINACLE:
            score += 1
        return score

    def _score_setup_active(self, card: Pokemon | Card) -> int:
        return {
            C.SOLROCK: 5,
            C.OKIDOGI: 4,
            C.BINACLE: 3,
            C.URSALUNA: 2,
            C.LUNATONE: 1,
        }.get(card.id, 0)

    def _score_setup_bench(self, card: Pokemon | Card) -> int:
        return {
            C.LUNATONE: 5,
            C.OKIDOGI: 4,
            C.SOLROCK: 3,
            C.BINACLE: 2,
            C.URSALUNA: 1,
        }.get(card.id, 0)

    def _score_to_hand(self, card: Pokemon | Card) -> float:
        cid = card.id
        score = 200 - self.hand_counts[cid] * 60
        if cid == C.F_ENERGY:
            score += 80 - 30 * min(2, self.hand_f)
        elif cid == C.PRISM:
            score += 90 if self.hand_prism == 0 else -80
        elif cid == C.OKIDOGI:
            total = self.field_counts[cid] + self.hand_counts[cid]
            score += 90 if total < 2 else -100
        elif cid == C.LUNATONE:
            score += 70 if self.field_counts[cid] == 0 and self.hand_counts[cid] == 0 else -220
        elif cid == C.SOLROCK:
            score += 60 if self.field_counts[cid] < 2 else -160
        elif cid == C.BINACLE:
            line = self.field_counts[C.BINACLE] + self.field_counts[C.BARBARACLE]
            score += 40 if line == 0 else -120
        elif cid == C.BARBARACLE:
            score += 60 if self.field_counts[C.BINACLE] >= 1 else -60
        elif cid == C.URSALUNA:
            score -= 60
        elif cid == C.NZONE:
            score += 220 if self._opponent_has_rulebox() else 20
        elif cid == C.BATTLE_CAGE:
            score += 10
        return score

    def _score_discard(self, card: Pokemon | Card) -> float:
        cid = card.id
        if cid == C.F_ENERGY:
            return 100  # Tarragon/Night Stretcherで回収可能
        if cid == C.NZONE:
            return -1000  # トラッシュから戻せない
        if cid == C.PRISM:
            return -300
        score = 0
        if self.hand_counts[cid] >= 2:
            score += 60
        data = card_table.get(cid)
        if data is not None and data.cardType == CardType.POKEMON:
            score -= 30
        return score

    def _low_deck(self) -> bool:
        return self.me.deckCount <= LOW_DECK_COUNT


def agent(obs_dict: dict) -> list[int]:
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        return my_deck

    global pre_turn
    global plan

    if pre_turn != obs.current.turn:
        pre_turn = obs.current.turn
        plan = AttackPlan()

    return BarbaraclePolicy(obs).choose()
