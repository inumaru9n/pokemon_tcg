"""044_barbaracle_full — 037 + フル精読ルール(R31-R46) + lethal限定探索 (I-080).

ベース: 037_barbaracle (sig 35ce7e0e55 の軽量逆設計)。
knowledge/study/barbaracle_35ce_study.md の精読18戦+226戦補助集計から:
- R31-R33: Cornerstone Mask Ogerpon ex (特性持ちからのダメージ0) を正しくモデル化。
  Solrockを唯一の対壁アタッカーとして扱い、Bossで1プライズ級を引きずり出す
- R38-R40: NZ(1枚・回収不能)は相手スタジアム3枚消費まで温存、Battle Cageを囮に
- R41-R43: 対Alakazam = 手札破壊が直接のダメージ軽減 (Powerful Hand=手札x20)
- R35-R37/R44-R46: Boss対象(Fezandipiti ex/進化前優遇)、山切れ管理、Barbaracle2本目
探索: 036と同じlethal限定探索 (SAMPLES=5、残りプライズ<=3のMAINのみ、全サンプル勝利で上書き)。
"""
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

    # opponent cards we model explicitly (study R31/R36/R37/R41)
    CORNERSTONE = 117       # Cornerstone Mask Ogerpon ex: no damage from ability-Pokemon
    FEZANDIPITI = 140
    DUNSPARCE = 305
    DUDUNSPARCE = 66
    ABRA = 741
    KADABRA = 742
    ALAKAZAM = 743


# our Pokemon that have an Ability (walled by Cornerstone Stance, study R31)
ABILITY_MONS = {116, 675, 1052, 135}  # Okidogi, Lunatone, Barbaracle, Ursaluna


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

        # --- opponent modelling (study R31/R38/R41) ---
        op_board_ids = {p.id for p in self._opponent_board() if p is not None}
        op_seen_ids = op_board_ids | {c.id for c in self.opponent.discard}
        self.op_cornerstone_on_board = C.CORNERSTONE in op_board_ids
        self.op_cornerstone_active = bool(
            self.opponent.active
            and self.opponent.active[0] is not None
            and self.opponent.active[0].id == C.CORNERSTONE
        )
        self.op_is_alakazam = bool({C.ALAKAZAM, C.KADABRA, C.ABRA} & op_seen_ids)
        # stadiums the opponent has consumed (discard) or has in play (R38)
        op_stadium_ids = [
            c.id for c in self.opponent.discard
            if card_table[c.id].cardType == CardType.STADIUM
        ]
        if self.stadium_id and not self.stadium_is_mine:
            op_stadium_ids.append(self.stadium_id)
        self.op_stadiums_seen = len(op_stadium_ids)
        # Marnie's stadium suite (3x Spikemuth + Risky) actively hunts our NZ;
        # other decks run only 1-2 utility stadiums and cannot win the
        # stadium war against our 4 (BCx3 + NZ) — hold NZ only vs Marnie.
        self.op_is_marnie_stadiums = any(cid in (1259, 1260) for cid in op_stadium_ids)

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
                    # R31: Cornerstone Stance prevents ALL damage from our
                    # ability-Pokemon (study: 122 zero-damage attacks = main
                    # cause of the 27% Crustle matchup). Solrock/Binacle only.
                    if op_pokemon.id == C.CORNERSTONE and my_pokemon.id in ABILITY_MONS:
                        damage = 0

                    score = target_score(op_pokemon)
                    prize = prize_count(op_pokemon) if op_pokemon.hp <= damage else 0
                    if prize == 0:
                        score *= damage / max(1, op_pokemon.hp)
                        # R37: don't chip recyclable Dunsparce line without KO
                        if op_pokemon.id in (C.DUNSPARCE, C.DUDUNSPARCE):
                            score -= 400
                    if len(self.opponent.prize) <= prize:
                        score = 50000
                    # study(対Marnie機序): killing a D-charged Munkidori removes a
                    # permanent 30/turn drain+heal; worth more than a non-KO chip
                    # into a healing Grimmsnarl
                    if prize > 0 and op_pokemon.id == 112 and EnergyType.DARKNESS in op_pokemon.energies:
                        score += 400

                    score += base_score
                    score += 220 if attacker_index == 0 else 0
                    score += 300 if target_index == 0 else 0
                    # R36: gusted target with no energy cannot retaliate
                    # (Fezandipiti ex farming = top Boss target of the pro)
                    if target_index != 0 and not op_pokemon.energies:
                        score += 120
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
            # R32: vs Cornerstone the ability-less Solrock is our only attacker
            if self.op_cornerstone_on_board:
                score += 1500 if energy_count < 1 else -400
            else:
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
            if self.me.deckCount <= 6:
                return -1
            # R44: in a Cornerstone stall, deck-out is a real loss condition —
            # don't burn our own deck when we are not clearly ahead on cards
            if (
                self.op_cornerstone_active
                and self.me.deckCount <= 15
                and self.me.deckCount <= self.opponent.deckCount + 5
            ):
                return -1
            if self.hand_f <= 1 and not self.state.energyAttached:
                return 7000  # 最後の闘エネはまず手貼り/Stone Armsに回す
            return 30000
        if card.id == C.BARBARACLE:
            if self.hand_f >= 2:
                return 29000
            return 7500
        if option.area == AreaType.STADIUM:
            return -1  # 相手のSpikemuth Gym等は我々には完全な空振り (study: 写しない癖)
        return 25000

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
            if line == 0:
                return 19500
            # R45: second Stone Arms line when we have surplus F to feed it
            # (the pro benches a 2nd line in 23% of games)
            if line == 1 and self.hand_f >= 2:
                return 2500
            return -1
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
            # R38 note: holding NZ until Marnie burns >=3 stadiums was tried and
            # measured NEGATIVE vs 038 (21.1% pooled with hold vs 24.2% ablated;
            # early NZ still blanks 1-2 Shadow Bullets before it dies, and the
            # 4-vs-4 stadium war is symmetric). Keep 037's immediate play.
            if self.stadium_id == C.NZONE and self.stadium_is_mine:
                return -1
            if self._opponent_has_rulebox():
                return 3400
            if self.stadium_id and not self.stadium_is_mine:
                return 1300  # 相手スタジアム潰し
            return -1
        if cid == C.BATTLE_CAGE:
            if self.stadium_is_mine and self.stadium_id:
                return -1  # R39/R40: 自分のスタジアム(NZ/BC)を上書きしない
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
            if self._low_deck():
                return -1
            if len(self.me.hand) <= 3:
                return 2900
            # R42: vs Alakazam a fat opponent hand IS their damage output
            if self.op_is_alakazam and self.opponent.handCount >= 7:
                return 2950
            return -1
        if cid == C.XEROSIC:
            # R41: vs Alakazam use aggressively (hand size x20 = Powerful Hand)
            if self.op_is_alakazam and self.opponent.handCount >= 6:
                return 3300
            return 2750 if self.opponent.handCount >= 6 else -1  # 037 baseline
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


# ---------------------------------------------------------------------------
# Lethal-only determinized search layer (MAIN only), ported from 036/007.
# Runs only when our remaining prizes are <=3. A candidate MAIN option
# overrides the heuristic only if EVERY determinization sample ends the greedy
# rollout of our own turn with the game already won. No board evaluation is
# involved; any failure falls back to the heuristic (structurally >= base).

SEARCH_CANDIDATES = 8
SEARCH_SAMPLES = 5           # EXP-036: draw-order-dependent lines need 5 to
                             # keep the false-positive rate near zero
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
    while len(unknown) < need:  # counting slack: pad with basic Fighting energy
        unknown.append(C.F_ENERGY)
    your_deck = unknown[: me.deckCount]
    your_prize = unknown[me.deckCount : me.deckCount + len(me.prize)]

    opponent_deck = [PLACEHOLDER_MON] * op.deckCount
    opponent_prize = [PLACEHOLDER_MON] * len(op.prize)
    opponent_hand = [C.F_ENERGY] * op.handCount
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
            sel = BarbaraclePolicy(obs).choose()
            if not sel and obs.select.minCount > 0:
                sel = _fallback_selection(obs.select)
        except Exception:
            sel = _fallback_selection(obs.select)
        state = search_step(state.searchId, sel)
    return state.observation


def _lethal_search(obs: Observation, policy: BarbaraclePolicy) -> list[int] | None:
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


def agent(obs_dict: dict) -> list[int]:
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        return my_deck

    global pre_turn
    global plan

    if pre_turn != obs.current.turn:
        pre_turn = obs.current.turn
        plan = AttackPlan()

    policy = BarbaraclePolicy(obs)

    if obs.select.context == SelectContext.MAIN and obs.current.turn >= 2:
        # Set the heuristic plan for the real trajectory first (sub-selects
        # rely on it), then let the search override only the MAIN choice.
        policy._plan_attack()
        saved = (plan, pre_turn)
        try:
            choice = _lethal_search(obs, policy)
        except Exception:
            choice = None
        plan, pre_turn = saved
        if choice is not None:
            return choice
        return policy.choose()

    return policy.choose()
