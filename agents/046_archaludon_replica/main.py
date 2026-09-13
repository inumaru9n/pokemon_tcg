"""046_archaludon_replica — Archaludon exトップ個体 (sig 90dd411e61, ShumpeiNomura, 本番288戦52.8%) の軽量逆設計.

islet骨格 (042系) をArchaludonタンクデッキ用に書き換えたヒューリスティック方策。
全288戦リプレイの行動頻度集計 (scratchpad aggregate_046/b/c.py) に基づく主ルール:
- 先攻は常にYES (145/145) / セットアップアクティブ Duraludon > Relicanth > Articuno、ベンチは置かない
- エンジン: Ultra Ball/Carmineで{M}エネをトラッシュに落とし、Assemble Alloy(進化時トラッシュから2枚加速)で
  Archaludon exを即起動 (トラッシュM>=2の時だけ進化する進化ゲート)
- 攻撃: Metal Defender 220 / Raging Hammer 80+10xダメカン (Relicanth Memory DiveでArchaludonも使える) /
  Hammer In 30 の最大ダメージ選択。壁 (Rock Inn/Cornerstone/NZ) に無効化される攻撃は撃たずEND (本人176回)
- 対Alakazam戦 (65.6%, n=131): TR Articunoをアクティブに置いて壁 (Repelling VeilがPowerful Handの
  ダメカン配置を無効化)、ENDし続けて相手を山切れさせる。キルできる時だけ交代して取る。
  Xerosic/Judgeで相手の手札 (=PH打点) を刈る。勝ち試合の中央値はターン41の長期戦
- Boss'sOrdersはキル専用 (Munkidori/Fez/Roserade)。Lillieは手札戻し6ドローで山切れ防止リフューエル兼用
"""
from __future__ import annotations

import os
from collections import defaultdict

from cg.api import (
    AreaType,
    Card,
    CardType,
    Observation,
    OptionType,
    Pokemon,
    SelectContext,
    all_card_data,
    to_observation_class,
)


class C:
    M_ENERGY = 8

    ARCHALUDON = 190
    DURALUDON = 169
    RELICANTH = 57
    ARTICUNO = 414

    CAPE = 1159
    STRETCHER = 1097
    ULTRA_BALL = 1121
    POKEGEAR = 1122
    POKE_PAD = 1152
    BOSS = 1182
    CARMINE = 1192
    LILLIE = 1227
    JUDGE = 1213
    XEROSIC = 1197
    FML = 1244

    # 相手側の壁・検知用
    CRUSTLE_INN = 345
    OGERPON_CORNERSTONE = 117
    BARBARACLE = 1052
    NZONE = 1247

    LILLIES_PEARL = 1172
    LEGACY_ENERGY = 12


class A:
    HAMMER_IN = 223       # {M} 30
    RAGING_HAMMER = 224   # {M}{M}{C} 80 + 10x自分のダメカン
    METAL_DEFENDER = 253  # {M}{M}{M} 220


METAL_TYPE = 8  # EnergyType.METAL
ALAKAZAM_IDS = {109, 741, 742, 743, 245}  # Abra/Kadabra/Alakazam 各版
OUR_POKEMON = {C.ARCHALUDON, C.DURALUDON, C.RELICANTH, C.ARTICUNO}

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


class ArchaludonPolicy:
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
        self.can_attack = False
        self.boss_playable = False

        self._count_cards()
        self._scan_main_options()

        self.discard_m = self.discard_counts[C.M_ENERGY]
        self.hand_energy = self.hand_counts[C.M_ENERGY]
        self.opp_hand_n = self.opponent.handCount  # 相手の手札は非公開 (hand=None)
        self.bench_n = sum(1 for p in self.me.bench if p is not None)
        self.board_n = self.bench_n + sum(1 for p in self.me.active if p is not None)

        self._analyze()

    def choose(self) -> list[int]:
        if not self.select.option or self.select.maxCount == 0:
            return []
        if self.context == SelectContext.SETUP_BENCH_POKEMON:
            return []  # 本人はセットアップでベンチを置かない (288戦で0回)
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
            elif option.type == OptionType.PLAY:
                card = get_card(self.obs, AreaType.HAND, option.index, self.my_index)
                if card is not None and card.id == C.BOSS:
                    self.boss_playable = True

    def _my_board(self) -> list[Pokemon | None]:
        return self.me.active + self.me.bench

    def _opponent_board(self) -> list[Pokemon | None]:
        return self.opponent.active + self.opponent.bench

    # ---- 対面分析 ----

    def _analyze(self) -> None:
        self.opp_active = self.opponent.active[0] if self.opponent.active else None
        opp_visible = [p.id for p in self._opponent_board() if p is not None]
        opp_visible += [c.id for c in self.opponent.discard]
        opp_set = set(opp_visible)
        # 対Alakazamモード: Articuno壁で山切れ勝ちを狙う (本人: 対Alakazam MAINの96%がArticunoアクティブ)
        self.alakazam_mode = bool(opp_set & ALAKAZAM_IDS)
        # 壁デッキ検知: ex攻撃が無効化される対面では進化を止めDuraludonで戦う
        self.wall_deck = bool(opp_set & {C.CRUSTLE_INN, C.OGERPON_CORNERSTONE, C.BARBARACLE})
        self.stadium_id = self.state.stadium[0].id if self.state.stadium else None
        if self.stadium_id == C.NZONE:
            self.wall_deck = True
        self.my_active = self.me.active[0] if self.me.active else None

    def _attack_blocked(self, attacker: Pokemon, target: Pokemon) -> bool:
        """壁特性でこの攻撃のダメージが0になるか (EXP-042のエンジン仕様知見)。"""
        a_data = card_table[attacker.id]
        is_ex = a_data.ex or a_data.megaEx
        if target.id == C.CRUSTLE_INN and is_ex:
            return True
        if target.id == C.OGERPON_CORNERSTONE and has_ability(attacker.id):
            return True
        if self.stadium_id == C.NZONE and is_ex:
            t_data = card_table[target.id]
            if not (t_data.ex or t_data.megaEx):
                return True
        return False

    # ---- 攻撃プラン ----

    def _attack_candidates(self, pokemon: Pokemon) -> list[tuple[int, int, int]]:
        """(attack_id, energy_required, damage)。Memory Dive (Relicanth) 対応。"""
        result = []
        counters = max(0, (pokemon.maxHp - pokemon.hp) // 10)
        if pokemon.id == C.DURALUDON:
            result.append((A.HAMMER_IN, 1, 30))
            result.append((A.RAGING_HAMMER, 3, 80 + 10 * counters))
        elif pokemon.id == C.ARCHALUDON:
            result.append((A.METAL_DEFENDER, 3, 220))
            if self.field_counts[C.RELICANTH] > 0:  # Memory Dive
                result.append((A.RAGING_HAMMER, 3, 80 + 10 * counters))
                result.append((A.HAMMER_IN, 1, 30))
        return result

    def _attach_budget(self) -> int:
        if not self.state.energyAttached and self.hand_energy > 0:
            return 1
        return 0

    def _plan_attack(self) -> None:
        global plan
        best_score = -1
        plan = AttackPlan()
        if self.state.turn < 2:
            return

        budget = self._attach_budget()
        wall_hold = self.alakazam_mode and self.my_active is not None and self.my_active.id == C.ARTICUNO
        for attacker_index, my_pokemon in enumerate(self._my_board()):
            if my_pokemon is None:
                continue
            if attacker_index != 0 and not self.can_switch:
                break
            for attack_id, energy_required, base_damage in self._attack_candidates(my_pokemon):
                energy_count = len(my_pokemon.energies)
                needs_energy = False
                if energy_count < energy_required:
                    if attacker_index == 0 and energy_count + budget >= energy_required:
                        needs_energy = True
                    elif attacker_index != 0 and energy_count + min(budget, 1) >= energy_required:
                        needs_energy = True
                    else:
                        continue

                for target_index, op_pokemon in enumerate(self._opponent_board()):
                    if op_pokemon is None:
                        continue
                    if target_index != 0 and not self.boss_playable:
                        break

                    if self._attack_blocked(my_pokemon, op_pokemon):
                        continue  # 壁に無効化される攻撃は撃たない (本人はENDで待つ, 176回)

                    damage = base_damage
                    op_data = card_table[op_pokemon.id]
                    if op_data.weakness == METAL_TYPE:
                        damage *= 2
                    elif op_data.resistance == METAL_TYPE:
                        damage = max(0, damage - 30)

                    kills = op_pokemon.hp <= damage
                    if attacker_index != 0 and not kills:
                        # ベンチアタッカーへの交代はキルできる時だけ (壁ホールド/ピボット温存)
                        if not (self.my_active is not None and self.my_active.id in (C.ARTICUNO, C.RELICANTH)
                                and not wall_hold):
                            continue
                    if wall_hold and attacker_index != 0 and not kills:
                        continue
                    if target_index != 0 and not kills:
                        continue  # Bossはキル専用 (本人: killable 88%)

                    score = target_score(op_pokemon)
                    prize = prize_count(op_pokemon) if kills else 0
                    if prize == 0:
                        score *= damage / max(1, op_pokemon.hp)
                    if len(self.opponent.prize) <= prize:
                        score = 50000

                    score += 220 if attacker_index == 0 else 0
                    score += 300 if target_index == 0 else 0
                    score += damage / 100  # 同点はダメージ大 (MD 220 > RH低カウンタ)

                    if score > best_score:
                        best_score = score
                        plan = AttackPlan(
                            attacker=attacker_index,
                            target=target_index,
                            attack_id=attack_id,
                            remain_hp=op_pokemon.hp - damage,
                            needs_energy=needs_energy,
                        )

    # ---- アタッチ ----

    def _energy_target_score(self, pokemon: Pokemon, active: bool) -> int:
        n = len(pokemon.energies)
        cid = pokemon.id
        score = 8000 + (10 if active else 0)
        if cid == C.DURALUDON:
            score += {0: 700, 1: 500, 2: 350}.get(n, -400)
        elif cid == C.ARCHALUDON:
            score += {0: 200, 1: 240, 2: 750}.get(n, -400)  # MMM完成が最優先 (出撃キルの前提)
        elif cid == C.ARTICUNO:
            if self.alakazam_mode:
                # 壁の退却エネ=出撃キルの生命線 (本人はArticuno@0に131回アタッチ)
                score += {0: 950, 1: 250}.get(n, -300)
            else:
                score += {0: 420, 1: 300}.get(n, -300)
        elif cid == C.RELICANTH:
            score += 250 if n == 0 else -350
        return score

    def _board_index(self, area: AreaType, index: int) -> int:
        return index if area == AreaType.ACTIVE else index + 1

    def _score_attach(self, option) -> float:
        card = get_card(self.obs, option.area, option.index, self.my_index)
        pokemon = get_card(self.obs, option.inPlayArea, option.inPlayIndex, self.my_index)
        if card is None or not isinstance(pokemon, Pokemon):
            return 0

        if card.id == C.CAPE:  # Hero's Cape: Duraludonタンク優先 (124:60)
            if pokemon.tools:
                return -1
            score = 6000 + {C.DURALUDON: 300, C.ARCHALUDON: 200}.get(pokemon.id, 30)
            if option.inPlayArea == AreaType.ACTIVE:
                score += 100
            return score

        score = self._energy_target_score(pokemon, option.inPlayArea == AreaType.ACTIVE)
        board_index = self._board_index(option.inPlayArea, option.inPlayIndex)
        if board_index == plan.attacker and plan.needs_energy:
            score += 2000
        return score

    # ---- オプション採点 ----

    def _score_option(self, option) -> float:
        if option.type == OptionType.NUMBER:
            return option.number
        if option.type == OptionType.YES:
            return 100 if self.context == SelectContext.IS_FIRST else 1  # 先攻は常にYES / Assemble AlloyもYES
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
            return 26000
        if option.type == OptionType.RETREAT:
            return self._score_retreat()
        if option.type == OptionType.ATTACK:
            if plan.attack_id == -1:
                return -1  # プラン無し (壁で全滅等) は攻撃しないでEND
            if plan.attacker == 0:
                return 1100 if option.attackId == plan.attack_id else 900
            return 500  # ベンチアタッカー待ち (retreat 2000が先に選ばれる)
        return 0

    def _score_evolve(self, option) -> float:
        pokemon = get_card(self.obs, option.inPlayArea, option.inPlayIndex, self.my_index)
        if self.wall_deck:
            return -1  # 壁対面はDuraludon(非ex)のまま戦う (本人の進化率13% vs 49%)
        if self.alakazam_mode and self.field_counts[C.ARCHALUDON] >= 1:
            return -1  # 対Alakazamのex 2体目はBoss+PHに2プライズ献上するだけ
        if self.field_counts[C.ARCHALUDON] >= 2:
            return 600
        if self.discard_m >= 2:  # Assemble Alloyで即MMM起動
            score = 9000
            if option.inPlayArea == AreaType.ACTIVE:
                score += 200
            if isinstance(pokemon, Pokemon):
                score += len(pokemon.energies) * 10
            return score
        if (
            option.inPlayArea == AreaType.ACTIVE
            and isinstance(pokemon, Pokemon)
            and (len(pokemon.energies) >= 2 or pokemon.maxHp - pokemon.hp >= 40)
        ):
            return 1400  # 加速無しでも手貼り済み/被弾中のアクティブは300HP化する価値あり
        return -1

    def _score_retreat(self) -> float:
        if plan.attacker < 1:
            return -1
        active = self.my_active
        if active is None:
            return -1
        if active.id == C.ARCHALUDON:
            return -1  # 充電済みArchaludonは下げない (本人はENDで待つ)
        if self.alakazam_mode and active.id == C.ARTICUNO and plan.remain_hp > 0:
            return -1  # 壁ホールド: キルできる時だけ出る
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
        if cid == C.ARTICUNO and self.alakazam_mode and self.field_counts[C.ARTICUNO] == 0:
            return 21000  # 対Alakazamの壁は最優先
        line = self.field_counts[C.DURALUDON] + self.field_counts[C.ARCHALUDON]
        if self.alakazam_mode:
            # 対Alakazamはベンチを絞る: 並べたポケモンは全てBoss+Powerful Handの的 (負けトレース精読)
            if self.bench_n >= 2:
                return -1
            if cid == C.DURALUDON:
                return 19000 if line < 2 else -1
            if cid == C.RELICANTH:
                return 12000 if self.field_counts[C.RELICANTH] == 0 else -1
            return -1
        if self.bench_n >= 3:
            return -1  # ベンチは小さく保つ (本人のベンチは0-2が85%)
        if cid == C.DURALUDON:
            if line < 2:
                return 19000
            return 13000 if line < 3 else -1
        if cid == C.RELICANTH:
            return 16000 if self.field_counts[C.RELICANTH] == 0 else -1  # Memory Dive要員
        if cid == C.ARTICUNO:
            return 9000 if self.field_counts[C.ARTICUNO] == 0 else -1
        return 5000

    def _score_play_trainer(self, card: Card) -> float:
        cid = card.id
        deck_n = self.me.deckCount
        hand_n = len(self.me.hand)
        stall_hoard = self.alakazam_mode and deck_n <= 15  # 山切れ戦争: 手札を溜めてLillieで山に戻す

        if cid == C.POKE_PAD:
            if deck_n <= 3 or stall_hoard:
                return -1
            return 10000
        if cid == C.ULTRA_BALL:
            if deck_n <= 3 or stall_hoard:
                return -1
            return 9800
        if cid == C.STRETCHER:
            return 9500  # トラッシュ回収は山を消費しない
        if cid == C.POKEGEAR:
            if stall_hoard:
                return -1
            return 9000
        if cid == C.FML:
            if self.stadium_id == C.FML:
                return -1
            return 8800  # スタジアムは早置き (T1-2で134回)。相手のNZ潰しも兼ねる

        # サポーター (優先度は本人のペアワイズ頻度準拠)
        if cid == C.XEROSIC:
            # 対Alakazamは温存して大きい手札 (=PH打点) を刈る (本人の使用中央値T20、手札17-34域が主)
            threshold = 10 if self.alakazam_mode else 5
            if self.opp_hand_n >= threshold:
                return 3400 + (200 if self.alakazam_mode else 0)
            return -1
        if cid == C.BOSS:
            return 3450 if plan.target > 0 else -1  # キル専用 (本人: 対象の88%が即処理)
        if cid == C.LILLIE:
            if deck_n <= 8 and hand_n >= 8:
                return 3600  # リフューエル: 手札を山に戻して山切れ回避
            if deck_n + hand_n <= 7:
                return -1
            if hand_n <= 5:
                return 3000
            return 2200 if hand_n <= 7 else -1
        if cid == C.JUDGE:
            if deck_n + hand_n <= 5:
                return -1
            if self.alakazam_mode and self.opp_hand_n >= 7:
                return 2900  # PH打点を4x20=80にリセット
            if hand_n <= 4 and self.opp_hand_n >= 5:
                return 2900
            return -1
        if cid == C.CARMINE:
            if deck_n <= 6 or stall_hoard:
                return -1
            if hand_n <= 5 and self.state.turn <= 8:
                return 2800  # 手札のMエネをトラッシュに落とす=Assemble燃料
            return -1
        return 1000

    # ---- カード選択 ----

    def _score_card_choice(self, option) -> float:
        card = get_card(self.obs, option.area, option.index, option.playerIndex)
        if card is None:
            return 0
        if self.context in {SelectContext.SWITCH, SelectContext.TO_ACTIVE}:
            return self._score_active_choice(option, card)
        if self.context == SelectContext.SETUP_ACTIVE_POKEMON:
            return {C.DURALUDON: 5, C.RELICANTH: 3, C.ARTICUNO: 1}.get(card.id, 0)
        if self.context == SelectContext.TO_HAND:
            return self._score_to_hand(card)
        if self.context == SelectContext.ATTACH_FROM and isinstance(card, Pokemon):
            # Assemble Alloy加速先: 進化したArchaludonを3枚まで (本人819:62)
            if card.id == C.ARCHALUDON:
                n = len(card.energies)
                return 900 - n * 10 if n < 3 else 200
            if card.id == C.DURALUDON:
                return 300
            return 50
        if self.context == SelectContext.ATTACH_TO:
            return 100  # トラッシュのMエネ (同種のみ)
        if self.context == SelectContext.EVOLVES_TO:
            return 100 if card.id == C.ARCHALUDON else 0
        if self.context in {SelectContext.DISCARD, SelectContext.TO_DECK, SelectContext.TO_DECK_BOTTOM}:
            return self._score_discard(card)
        return 0

    def _score_active_choice(self, option, card: Pokemon | Card) -> float:
        if not isinstance(card, Pokemon):
            return 0
        if option.playerIndex != self.my_index:
            # Boss対象: プランのキル対象 > 低HPの置物
            if option.index == plan.target - 1:
                return 100
            data = card_table[card.id]
            return 30 + (20 if data.skills else 0) - card.hp / 20

        cid = card.id
        score = len(card.energies) * 15
        if option.index == plan.attacker - 1:
            score += 220
        if self.alakazam_mode and cid == C.ARTICUNO:
            score += 500  # 対Alakazamの昇格はArticuno最優先 (本人: available時ほぼ100%)
        elif cid == C.ARCHALUDON:
            score += 300
        elif cid == C.DURALUDON:
            score += 200
        elif cid == C.ARTICUNO:
            score += 150
        elif cid == C.RELICANTH:
            score += 60
        return score

    def _score_to_hand(self, card: Pokemon | Card) -> float:
        cid = card.id
        field = self.field_counts
        hand = self.hand_counts
        if cid == C.DURALUDON:
            line = field[C.DURALUDON] + field[C.ARCHALUDON] + hand[C.DURALUDON]
            return 300 if line < 3 else 60
        if cid == C.ARCHALUDON:
            if field[C.ARCHALUDON] >= 2 or self.wall_deck:
                return 20
            if hand[C.ARCHALUDON] == 0 and field[C.DURALUDON] >= 1:
                return 280 if self.discard_m >= 2 else 180
            return 40
        if cid == C.ARTICUNO:
            have = field[C.ARTICUNO] + hand[C.ARTICUNO]
            if have == 0:
                return 260 if self.alakazam_mode else 150
            return -50
        if cid == C.RELICANTH:
            return 240 if field[C.RELICANTH] + hand[C.RELICANTH] == 0 else -40
        if cid == C.M_ENERGY:
            return 250 if self.hand_energy == 0 else 80
        if cid == C.BOSS:
            return 190
        if cid == C.XEROSIC:
            return 210 if (self.alakazam_mode and self.opp_hand_n >= 5) else 90
        if cid == C.LILLIE:
            return 170 if len(self.me.hand) <= 5 else 120
        if cid == C.CARMINE:
            return 140 if len(self.me.hand) <= 5 else 70
        if cid == C.JUDGE:
            return 100
        return 50

    def _score_discard(self, card: Pokemon | Card) -> float:
        cid = card.id
        # Ultra Ballのコスト等。Mエネを最優先で落とす (本人383回) = Assemble Alloy/Stretcherの燃料
        if cid == C.M_ENERGY:
            return 300
        if cid == C.DURALUDON:
            return -100
        if cid == C.CARMINE:
            return 200 if self.state.turn > 8 else 120
        if cid in (C.STRETCHER, C.POKEGEAR, C.POKE_PAD, C.ULTRA_BALL):
            return 150 + (40 if self.hand_counts[cid] >= 2 else 0)
        if cid == C.FML:
            return 140 if self.stadium_id == C.FML else 90
        if cid == C.JUDGE:
            return 110
        if cid == C.ARCHALUDON:
            return 130 if (self.hand_counts[cid] >= 2 or self.field_counts[cid] >= 2) else 10
        if cid == C.RELICANTH:
            return 100 if self.field_counts[cid] >= 1 else 20
        if cid == C.BOSS:
            return 70
        if cid == C.LILLIE:
            return 60
        if cid == C.ARTICUNO:
            return 60 if self.field_counts[cid] >= 1 else 15
        if cid == C.XEROSIC:
            return -100 if self.alakazam_mode else 50
        if cid == C.CAPE:
            return -200
        return 80


def agent(obs_dict: dict) -> list[int]:
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        return my_deck

    global pre_turn
    global plan

    if pre_turn != obs.current.turn:
        pre_turn = obs.current.turn
        plan = AttackPlan()

    return ArchaludonPolicy(obs).choose()
