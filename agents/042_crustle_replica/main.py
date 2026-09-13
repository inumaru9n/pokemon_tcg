"""042_crustle_replica — Crustle壁トップ個体 (sig 8b3183042b, LiamK, 本番332戦55.7%) の軽量逆設計.

islet骨格 (004/037) をCrustle壁デッキ用に書き換えたヒューリスティック方策。
LiamK個体の全332戦リプレイの行動頻度集計に基づく:
- 先攻は常にYES (163/170) / セットアップアクティブは Dwebble 最優先 (183)
- 二枚看板の壁: Crustle (ex/megaExからのワザダメージ無効) と
  Cornerstone Mask Ogerpon ex (特性持ちからのワザダメージ無効)。対面の脅威で壁を選ぶ
- 主砲は Articuno Dark Frost 60 (901回) と Crustle Superb Scissors 120 (743回)、
  Ogerpon Demolish 140 (324回)。攻撃可能なら常に攻撃 (END while attack available = 0)
- Munkidori はベンチの Adrena-Brain エンジン (壁のダメカンを相手に載せ替え)
- Crispin(530)/Lillie(514)/Poké Pad(342)/Brock(274)/Urbain(249) の回転。
  Lillie は「手札を山に戻して6ドロー」= 長期戦の山切れ防止リフューエルにも使う
- Jumbo Ice Cream は3エネ以上のアクティブを80回復、Hero's Cape は壁タンクに付ける
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
    G_ENERGY = 1
    W_ENERGY = 3
    F_ENERGY = 6
    D_ENERGY = 7
    PRISM = 16
    GROW_GRASS = 18

    MUNKIDORI = 112
    OGERPON = 117
    DWEBBLE = 344
    CRUSTLE = 345
    ARTICUNO = 414

    POFFIN = 1086
    HAMMER = 1120
    ICE_CREAM = 1147
    POKE_PAD = 1152
    CAPE = 1159
    CRISPIN = 1198
    BROCK = 1210
    LILLIE = 1227
    URBAIN = 1236

    LILLIES_PEARL = 1172
    LEGACY_ENERGY = 12


class A:
    MIND_BEND = 141
    DEMOLISH = 148
    ASCENSION = 478
    SUPERB_SCISSORS = 479
    DARK_FROST = 583


BASIC_ENERGIES = {C.G_ENERGY, C.W_ENERGY, C.F_ENERGY, C.D_ENERGY}
ALL_ENERGIES = BASIC_ENERGIES | {C.PRISM, C.GROW_GRASS}
LOW_DECK_COUNT = 6

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


def has_ability(card_id: int) -> bool:
    data = card_table.get(card_id)
    return data is not None and bool(data.skills)


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


def target_score(pokemon: Pokemon) -> int:
    data = card_table[pokemon.id]
    score = prize_count(pokemon) * 1000
    score += len(pokemon.energies) * 150
    score += len(pokemon.tools) * 100
    if data.stage2:
        score += 250
    elif data.stage1:
        score += 130
    score += pokemon.hp
    return score


class CrustlePolicy:
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

        self.hand_energy = sum(self.hand_counts[e] for e in ALL_ENERGIES)
        self.hand_basic_energy = sum(self.hand_counts[e] for e in BASIC_ENERGIES)
        self.deck_crustle = max(
            0, 4 - self.field_counts[C.CRUSTLE] - self.hand_counts[C.CRUSTLE] - self.discard_counts[C.CRUSTLE]
        )
        self.deck_dwebble = max(
            0, 4 - self.field_counts[C.DWEBBLE] - self.field_counts[C.CRUSTLE]
            - self.hand_counts[C.DWEBBLE] - self.discard_counts[C.DWEBBLE]
        )
        # 山に残っている基本エネの概算（サイド落ちは無視）
        attached_basics = 0
        for p in self._my_board():
            if p is None:
                continue
            attached_basics += sum(1 for card in p.energyCards if card.id in BASIC_ENERGIES)
        self.deck_basic_energy = max(
            0,
            8 - attached_basics - self.hand_basic_energy
            - sum(self.discard_counts[e] for e in BASIC_ENERGIES),
        )

        self._analyze_threat()
        self._scan_main_options()

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
            if option.type == OptionType.RETREAT:
                self.can_switch = True
            elif option.type == OptionType.ATTACK:
                self.can_attack = True

    def _my_board(self) -> list[Pokemon | None]:
        return self.me.active + self.me.bench

    def _opponent_board(self) -> list[Pokemon | None]:
        return self.opponent.active + self.opponent.bench

    # ---- 相手の脅威分析（壁選択の核） ----

    def _analyze_threat(self) -> None:
        """相手の主アタッカーを推定し、どちらの壁が機能するかを判定する。"""
        self.opp_active = self.opponent.active[0] if self.opponent.active else None
        primary = None
        best = -1
        for pokemon in self._opponent_board():
            if pokemon is None:
                continue
            value = len(pokemon.energies) * 100 + target_score(pokemon) // 10
            if pokemon is self.opp_active:
                value += 150
            if value > best:
                best = value
                primary = pokemon
        self.opp_primary = primary
        self.opp_threat_ex = False
        self.opp_threat_ability = False
        if primary is not None:
            data = card_table[primary.id]
            self.opp_threat_ex = data.ex or data.megaEx
            self.opp_threat_ability = has_ability(primary.id)
        # 盤面のどこかにex/特性持ちアタッカーがいるか（昇格判断用）
        self.opp_any_ex = any(
            p is not None and (card_table[p.id].ex or card_table[p.id].megaEx)
            for p in self._opponent_board()
        )

    def _active_is_walling(self) -> bool:
        """今のアクティブが相手アクティブのワザを無効化できているか。"""
        mine = self.me.active[0] if self.me.active else None
        if mine is None or self.opp_active is None:
            return False
        op_data = card_table[self.opp_active.id]
        if mine.id == C.CRUSTLE and (op_data.ex or op_data.megaEx):
            return True
        if mine.id == C.OGERPON and has_ability(self.opp_active.id):
            return True
        return False

    # ---- 攻撃プラン ----

    def _attack_candidates(self, pokemon: Pokemon) -> list[tuple[int, int, int, int, int]]:
        """(attack_id, energy_required, base_damage, attacker_energy_type, base_score)

        base_damage < 0 は特殊処理 (Ascension)。attacker_energy_type < 0 は弱点抵抗無視。
        """
        result = []
        if pokemon.id == C.CRUSTLE:
            result.append((A.SUPERB_SCISSORS, 3, 120, EnergyType.GRASS, 0))
        elif pokemon.id == C.OGERPON:
            result.append((A.DEMOLISH, 3, 140, -1, -30))
        elif pokemon.id == C.ARTICUNO:
            result.append((A.DARK_FROST, 3, 60, EnergyType.WATER, -80))
        elif pokemon.id == C.MUNKIDORI:
            result.append((A.MIND_BEND, 2, 60, EnergyType.PSYCHIC, -300))
        elif pokemon.id == C.DWEBBLE:
            if self.deck_crustle > 0 and self.hand_counts[C.CRUSTLE] == 0:
                result.append((A.ASCENSION, 1, -1, -1, 0))
        return result

    def _attach_budget(self) -> int:
        budget = 0
        if not self.state.energyAttached and self.hand_energy > 0:
            budget += 1
        if self.hand_counts[C.CRISPIN] > 0 and self.deck_basic_energy > 0:
            budget += 1
        return budget

    def _plan_attack(self) -> None:
        global plan
        best_score = -1
        plan = AttackPlan()
        if self.state.turn < 2:
            return

        hold_wall = self._active_is_walling()
        budget = self._attach_budget()
        for attacker_index, my_pokemon in enumerate(self._my_board()):
            if my_pokemon is None:
                continue
            if attacker_index != 0 and (not self.can_switch or hold_wall):
                break
            for attack_id, energy_required, base_damage, atype, base_score in self._attack_candidates(my_pokemon):
                energy_count = len(my_pokemon.energies)
                needs_energy = False
                if energy_count < energy_required:
                    if attacker_index == 0 and energy_count + budget >= energy_required:
                        needs_energy = True
                    elif attacker_index != 0 and energy_count + min(budget, 1) >= energy_required:
                        needs_energy = True
                    else:
                        continue

                if base_damage < 0:  # Ascension: 山からCrustleに進化してターン終了
                    score = 620 + base_score + (220 if attacker_index == 0 else 0)
                    if score > best_score:
                        best_score = score
                        plan = AttackPlan(attacker_index, 0, attack_id, -1, needs_energy)
                    continue

                for target_index, op_pokemon in enumerate(self._opponent_board()):
                    if op_pokemon is None:
                        continue
                    if target_index != 0 and not self.can_gust:
                        break

                    damage = base_damage
                    if atype >= 0 and damage > 0:
                        op_data = card_table[op_pokemon.id]
                        if op_data.weakness == atype:
                            damage *= 2
                        elif op_data.resistance == atype:
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

    def _energy_target_score(self, pokemon: Pokemon, active: bool, energy_id: int) -> int:
        prov = set(pokemon.energies)
        rainbow = EnergyType.RAINBOW in prov
        has_g = EnergyType.GRASS in prov or rainbow
        has_d = EnergyType.DARKNESS in prov or rainbow
        has_w = EnergyType.WATER in prov or rainbow
        has_f = EnergyType.FIGHTING in prov or rainbow
        n = len(pokemon.energies)
        cid = pokemon.id
        score = 8000 + (10 if active else 0)

        if energy_id in (C.G_ENERGY, C.GROW_GRASS):
            if cid in (C.CRUSTLE, C.DWEBBLE):
                bonus = 50 if energy_id == C.GROW_GRASS else 0  # +20HPぶんCrustle系を優先
                score += (780 + bonus) if not has_g else (300 if n < 3 else -600)
            elif cid == C.ARTICUNO:
                score += 280 if n < 3 else -600
            elif cid == C.OGERPON:
                score += 260 if n < 3 else -600
            elif cid == C.MUNKIDORI:
                score += 140 if n < 2 else -500
        elif energy_id == C.D_ENERGY:
            if cid == C.MUNKIDORI:
                score += 800 if not has_d else (100 if n < 2 else -500)
            elif cid in (C.CRUSTLE, C.DWEBBLE):
                score += (250 if has_g else 120) if n < 3 else -600
            else:
                score += 150 if n < 3 else -600
        elif energy_id == C.W_ENERGY:
            if cid == C.ARTICUNO:
                score += 700 if not has_w else (200 if n < 3 else -600)
            else:
                score += 120 if n < 3 else -600
        elif energy_id == C.F_ENERGY:
            if cid == C.OGERPON:
                score += 700 if not has_f else (200 if n < 3 else -600)
            else:
                score += 120 if n < 3 else -600
        elif energy_id == C.PRISM:
            # たねにPrism=全タイプ / 進化後は無色のみ
            if cid == C.ARTICUNO:
                score += 600 if n < 3 else -500
            elif cid == C.MUNKIDORI:
                score += 550 if not has_d else (80 if n < 2 else -500)
            elif cid == C.OGERPON:
                score += 500 if not has_f else (150 if n < 3 else -500)
            elif cid == C.DWEBBLE:
                score += 380 if n < 3 else -500
            elif cid == C.CRUSTLE:
                score += 180 if (has_g and n < 3) else -300
        return score

    def _board_index(self, area: AreaType, index: int) -> int:
        return index if area == AreaType.ACTIVE else index + 1

    def _score_attach(self, option) -> float:
        card = get_card(self.obs, option.area, option.index, self.my_index)
        pokemon = get_card(self.obs, option.inPlayArea, option.inPlayIndex, self.my_index)
        if card is None or not isinstance(pokemon, Pokemon):
            return 0

        if card.id == C.CAPE:  # Hero's Cape +100HP: 壁タンクへ
            if pokemon.tools:
                return -1
            score = 6000
            score += {C.ARTICUNO: 300, C.CRUSTLE: 250, C.OGERPON: 150, C.MUNKIDORI: 50, C.DWEBBLE: 30}.get(
                pokemon.id, 0
            )
            if option.inPlayArea == AreaType.ACTIVE:
                score += 100
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
            return 26000  # Adrena-Brain: 自分のダメカンを相手に載せ替え。常に得
        if option.type == OptionType.RETREAT:
            return self._score_retreat()
        if option.type == OptionType.ATTACK:
            return 1100 if option.attackId == plan.attack_id else 1000
        return 0

    def _score_retreat(self) -> float:
        if plan.attacker < 1:
            return -1
        if self._active_is_walling():
            return -1  # 機能している壁は動かさない
        active = self.me.active[0] if self.me.active else None
        if active is not None and active.id == C.CRUSTLE and len(active.energies) >= 2 and self.opp_any_ex:
            return -1  # 充電済みCrustleはex系相手に温存
        return 2000

    def _score_play(self, option) -> float:
        card = get_card(self.obs, AreaType.HAND, option.index, self.my_index)
        if card is None:
            return 0
        data = card_table[card.id]
        if data.cardType == CardType.POKEMON:
            return self._score_play_pokemon(card)
        return self._score_play_trainer(card)

    def _score_play_pokemon(self, card: Card) -> float:
        cid = card.id
        if cid == C.DWEBBLE:
            line = self.field_counts[C.DWEBBLE] + self.field_counts[C.CRUSTLE]
            return 19500 if line < 3 else 9000
        if cid == C.MUNKIDORI:
            if self.field_counts[C.MUNKIDORI] == 0:
                return 20000
            return 14000 if self.field_counts[C.MUNKIDORI] < 2 else 8500
        if cid == C.ARTICUNO:
            if self.field_counts[C.ARTICUNO] == 0:
                return 19000
            return 13000
        if cid == C.OGERPON:
            if self.field_counts[C.OGERPON] == 0:
                return 18000 + (500 if self.opp_threat_ability else 0)
            return 8000
        return 15000

    def _score_play_trainer(self, card: Card) -> float:
        cid = card.id
        low_deck = self.me.deckCount <= LOW_DECK_COUNT
        if cid == C.POFFIN:
            if self.deck_dwebble == 0 or self.me.deckCount <= 3:
                return -1
            return 10000
        if cid == C.POKE_PAD:
            if self.me.deckCount <= 3:
                return -1
            return 10000
        if cid == C.ICE_CREAM:
            active = self.me.active[0] if self.me.active else None
            if active is not None and len(active.energies) >= 3 and active.maxHp - active.hp >= 40:
                return 5000
            return -1
        if cid == C.HAMMER:
            return 2600
        if cid == C.CRISPIN:
            if self.deck_basic_energy == 0 or self.me.deckCount <= 3:
                return -1
            return 3300 if (plan.needs_energy or self.hand_energy <= 1) else 2950
        if cid == C.BROCK:
            if self.me.deckCount <= 3:
                return -1
            crustle_line_need = (
                self.field_counts[C.CRUSTLE] + self.hand_counts[C.CRUSTLE] == 0 and self.deck_crustle > 0
            )
            board = sum(1 for p in self._my_board() if p is not None)
            if crustle_line_need or board <= 2:
                return 3050
            return 2450
        if cid == C.LILLIE:
            hand = len(self.me.hand)
            if self.me.deckCount <= 10 and hand >= 8:
                return 3600  # 山切れ防止のリフューエル
            if low_deck:
                return -1
            if hand <= 5:
                return 3000
            return 1800
        if cid == C.URBAIN:
            if low_deck or self.me.deckCount <= 3:
                return -1
            return 2980 if len(self.me.hand) <= 7 else -1
        return 2000

    # ---- カード選択（サーチ・入替・ダメカン等） ----

    def _score_card_choice(self, option) -> float:
        card = get_card(self.obs, option.area, option.index, option.playerIndex)
        if card is None:
            return 0
        if self.context in {SelectContext.SWITCH, SelectContext.TO_ACTIVE}:
            return self._score_active_choice(option, card)
        if self.context == SelectContext.SETUP_ACTIVE_POKEMON:
            return {C.DWEBBLE: 5, C.MUNKIDORI: 3, C.OGERPON: 2, C.ARTICUNO: 1}.get(card.id, 0)
        if self.context == SelectContext.SETUP_BENCH_POKEMON:
            return {C.MUNKIDORI: 4, C.ARTICUNO: 3, C.OGERPON: 2, C.DWEBBLE: 1}.get(card.id, 0)
        if self.context == SelectContext.TO_HAND:
            return self._score_to_hand(card)
        if self.context == SelectContext.TO_BENCH:
            return 100  # Poffin: Dwebble一択
        if self.context == SelectContext.ATTACH_FROM and isinstance(card, Pokemon):
            energy_id = self.select.effect.id if self.select.effect is not None else C.G_ENERGY
            score = self._energy_target_score(card, option.area == AreaType.ACTIVE, energy_id)
            board_index = self._board_index(option.area, option.index)
            if board_index == plan.attacker and plan.needs_energy:
                score += 2000
            return score
        if self.context == SelectContext.REMOVE_DAMAGE_COUNTER:
            # Adrena-Brain移動元: 一番傷んだ壁から
            if isinstance(card, Pokemon):
                dmg = card.maxHp - card.hp
                bonus = {C.CRUSTLE: 150, C.ARTICUNO: 60, C.OGERPON: 60}.get(card.id, 0)
                return dmg + bonus
            return 0
        if self.context in {SelectContext.DAMAGE_COUNTER, SelectContext.DAMAGE}:
            # Adrena-Brain移動先: 取り切れる相手 > 相手の主砲
            if isinstance(card, Pokemon) and option.playerIndex != self.my_index:
                if card.hp <= 30:
                    return 3000
                return target_score(card)
            return 0
        if self.context == SelectContext.HEAL:
            if isinstance(card, Pokemon):
                return card.maxHp - card.hp
            return 0
        if self.context == SelectContext.EVOLVES_TO:
            return 100 if card.id == C.CRUSTLE else 0
        if self.context in {SelectContext.DISCARD, SelectContext.TO_DECK, SelectContext.TO_DECK_BOTTOM}:
            return self._score_discard(card)
        return 0

    def _score_active_choice(self, option, card: Pokemon | Card) -> float:
        if not isinstance(card, Pokemon):
            return 0
        if option.playerIndex != self.my_index:
            return 100 if option.index == plan.target - 1 else 0

        cid = card.id
        score = len(card.energies) * 3
        if option.index == plan.attacker - 1:
            score += 220
        op_data = card_table[self.opp_active.id] if self.opp_active is not None else None
        if cid == C.CRUSTLE:
            score += 90
            if self.opp_threat_ex or (op_data is not None and (op_data.ex or op_data.megaEx)):
                score += 260
            if len(card.energies) >= 3:
                score += 40
        elif cid == C.OGERPON:
            score += 30
            walls = self.opp_threat_ability or (op_data is not None and has_ability(self.opp_active.id))
            if walls:
                score += 240
            else:
                score -= 60  # 壁にならないOgerponは2枚サイドのリスクだけ
        elif cid == C.ARTICUNO:
            score += 110
            if len(card.energies) >= 2:
                score += 30
        elif cid == C.MUNKIDORI:
            score -= 40  # Adrena-Brainエンジンはベンチに
        elif cid == C.DWEBBLE:
            score -= 20
        return score

    def _score_to_hand(self, card: Pokemon | Card) -> float:
        cid = card.id
        score = 200 - self.hand_counts[cid] * 60
        field = self.field_counts
        if cid == C.CRUSTLE:
            if field[C.DWEBBLE] > 0 and self.hand_counts[C.CRUSTLE] == 0:
                score += 150
            else:
                score -= 50
        elif cid == C.MUNKIDORI:
            score += 80 if field[C.MUNKIDORI] < 2 else -60
        elif cid == C.ARTICUNO:
            score += 70 if field[C.ARTICUNO] + self.hand_counts[C.ARTICUNO] < 2 else -60
        elif cid == C.OGERPON:
            total = field[C.OGERPON] + self.hand_counts[C.OGERPON]
            score += (60 + (40 if self.opp_threat_ability else 0)) if total == 0 else -80
        elif cid == C.DWEBBLE:
            line = field[C.DWEBBLE] + field[C.CRUSTLE] + self.hand_counts[C.DWEBBLE]
            score += 50 if line < 3 else -70
        elif cid == C.G_ENERGY:
            need_g = any(
                p is not None and p.id in (C.CRUSTLE, C.DWEBBLE)
                and EnergyType.GRASS not in p.energies and EnergyType.RAINBOW not in p.energies
                for p in self._my_board()
            )
            score += 100 if (need_g and self.hand_counts[C.G_ENERGY] == 0) else 40
        elif cid == C.PRISM:
            score += 90 if self.hand_counts[C.PRISM] == 0 else -50
        elif cid == C.D_ENERGY:
            munki_needs_d = any(
                p is not None and p.id == C.MUNKIDORI
                and EnergyType.DARKNESS not in p.energies and EnergyType.RAINBOW not in p.energies
                for p in self._my_board()
            )
            score += 70 if munki_needs_d else -40
        elif cid == C.GROW_GRASS:
            score += 60
        elif cid in (C.W_ENERGY, C.F_ENERGY):
            score += 20
        return score

    def _score_discard(self, card: Pokemon | Card) -> float:
        cid = card.id
        # トラッシュからの回収手段が無いデッキなので、重要札を最後まで残す
        if cid == C.CAPE:
            return -400
        if cid == C.PRISM:
            return -250
        if cid == C.LILLIE:
            return -150  # 山切れ防止エンジン
        if cid in BASIC_ENERGIES or cid == C.GROW_GRASS:
            return -100
        score = 0
        if self.hand_counts[cid] >= 2:
            score += 60
        if cid == C.HAMMER:
            score += 80
        data = card_table.get(cid)
        if data is not None and data.cardType == CardType.POKEMON:
            score -= 30
        return score


def agent(obs_dict: dict) -> list[int]:
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        return my_deck

    global pre_turn
    global plan

    if pre_turn != obs.current.turn:
        pre_turn = obs.current.turn
        plan = AttackPlan()

    return CrustlePolicy(obs).choose()
