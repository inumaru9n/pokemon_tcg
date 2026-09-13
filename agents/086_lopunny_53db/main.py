"""086_lopunny_53db — Mega Lopunny ex / Mega Froslass ex (sig 53db4c87b8, tw_shin個体, 本番257戦56.0%).

islet骨格 (004/037) をLopunnyデッキ用に書き換えたヒューリスティック方策。
tw_shin個体のリプレイ257戦の行動頻度集計 (analyze_086.py) に基づく:
- 先攻は常にYES / セットアップアクティブは Fan Rotom > Dunsparce > Buneary > Snorunt /
  セットアップでベンチには一切出さない (257戦で0回)
- 勝ち筋 = Lopunnyダンス: ベンチのMega Lopunnyへ入替→Gale Thrust 60+170=230 をエネ1枚で連打。
  Air Balloonで入替コストを消し、Wally's Compassionでメガを全回復してループ
- 動いていないターンは Spiky Hopper 160 (低HP相手はこちらでKO継続)
- Dudunsparce の Run Away Draw (3ドロー→山へ戻る) を毎ターン回すドローエンジン
- Mega Froslass ex の Resentful Refrain (相手手札×50) が対ドローエンジンの副砲
- Hand Trimmer は相手手札6枚以上のときだけ (Refrain計画時は撃たない)
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
    W_ENERGY = 3
    MIST = 11
    ENRICH = 13

    DUDUN = 66
    ROTOM = 174
    DUN = 305
    BUNEARY = 848
    LOPUNNY = 849
    SNORUNT = 860
    FROSLASS = 861

    POFFIN = 1086
    TRIMMER = 1087
    ULTRA_BALL = 1121
    POKEGEAR = 1122
    POKE_PAD = 1152
    AIR_BALLOON = 1174
    BOSS_ORDERS = 1182
    HILDA = 1225
    LILLIE = 1227
    WALLY = 1229
    BATTLE_CAGE = 1264

    LILLIES_PEARL = 1172
    LEGACY_ENERGY = 12


class A:
    LAND_CRUSH = 76
    ASSAULT_LANDING = 230
    RAM = 424
    KICK = 1224
    GALE_THRUST = 1225
    SPIKY_HOPPER = 1226
    CHILLY = 1239
    REFRAIN = 1240
    SNOW = 1241


ENERGY_IDS = {C.W_ENERGY, C.MIST, C.ENRICH}
MEGA_IDS = {C.LOPUNNY, C.FROSLASS}
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
        damage: int = 0,
    ):
        self.attacker = attacker
        self.target = target
        self.attack_id = attack_id
        self.remain_hp = remain_hp
        self.needs_energy = needs_energy
        self.damage = damage
        self.flee = False  # アクティブのメガを安全なピボットと入れ替える


plan = AttackPlan()
pre_turn = -1
turn_start_serial = -1  # そのターン開始時のアクティブのserial (Gale Thrustボーナス判定用)


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


class LopunnyPolicy:
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
        self.can_switch = False
        self.can_gust = False
        self.stadium_id = self.state.stadium[0].id if self.state.stadium else 0
        self.stadium_is_mine = bool(self.state.stadium) and self.state.stadium[0].playerIndex == self.my_index

        self._count_cards()
        self._scan_main_options()

        self.hand_energy = sum(self.hand_counts[i] for i in ENERGY_IDS)
        self.hand_water = self.hand_counts[C.W_ENERGY]
        active = self.me.active[0] if self.me.active else None
        self.active_moved = self.state.retreated or (
            active is not None and turn_start_serial >= 0 and active.serial != turn_start_serial
        )
        if active is not None:
            cost = card_table[active.id].retreatCost
            if any(t.id == C.AIR_BALLOON for t in active.tools):
                cost = max(0, cost - 2)
            self.retreat_cost = cost
            self.retreat_free = cost == 0
        else:
            self.retreat_cost = 0
            self.retreat_free = False
        self.bench_ready = any(
            p is not None
            and (
                (p.id == C.LOPUNNY and len(p.energies) >= 1)
                or (p.id == C.FROSLASS and EnergyType.WATER in p.energies)
            )
            for p in self.me.bench
        )
        self.wally_urgent = (
            active is not None
            and active.id in MEGA_IDS
            and active.hp <= 240
            and active.maxHp - active.hp >= 90
            and self.hand_counts[C.WALLY] > 0
        )

    def choose(self) -> list[int]:
        if not self.select.option or self.select.maxCount == 0:
            return []
        if self.context == SelectContext.MAIN:
            self._plan_attack()
        scores = [self._score_option(option) for option in self.select.option]
        ranked = [i for i, _ in sorted(enumerate(scores), key=lambda item: item[1], reverse=True)]
        picked = [i for i in ranked if scores[i] >= 0][: self.select.maxCount]
        if len(picked) < self.select.minCount:
            rest = [i for i in ranked if scores[i] < 0]
            picked += rest[: self.select.minCount - len(picked)]
        return picked

    def _count_cards(self) -> None:
        for pokemon in self.me.active + self.me.bench:
            if pokemon is None:
                continue
            self.field_counts[pokemon.id] += 1
            for pre in pokemon.preEvolution:
                self.field_counts[pre.id] += 1
        for card in self.me.hand:
            self.hand_counts[card.id] += 1

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
            elif option.type == OptionType.ABILITY:
                card = get_card(self.obs, option.area, option.index, self.my_index)
                if (
                    option.area == AreaType.ACTIVE
                    and card is not None
                    and getattr(card, "id", 0) == C.DUDUN
                ):
                    self.can_switch = True  # Run Away Drawで能動的にアクティブを空けられる

    def _my_board(self) -> list[Pokemon | None]:
        return self.me.active + self.me.bench

    def _opponent_board(self) -> list[Pokemon | None]:
        return self.opponent.active + self.opponent.bench

    def _line_total(self, basic_id: int, evo_id: int) -> int:
        return (
            self.field_counts[evo_id]
            + self.hand_counts[evo_id]
        )

    def _missing_evolution(self) -> bool:
        pairs = [(C.DUN, C.DUDUN), (C.BUNEARY, C.LOPUNNY), (C.SNORUNT, C.FROSLASS)]
        for basic, evo in pairs:
            on_board = any(
                p is not None and p.id == basic for p in self._my_board()
            )
            if on_board and self.hand_counts[evo] == 0:
                return True
        return False

    def _damaged_mega(self) -> int:
        worst = 0
        for p in self._my_board():
            if p is not None and p.id in MEGA_IDS:
                worst = max(worst, p.maxHp - p.hp)
        return worst

    # ---- 攻撃プラン ----

    def _attack_candidates(self, pokemon: Pokemon, board_index: int) -> list[tuple[int, int, bool, int, int]]:
        """(attack_id, c_cost, needs_water, base_damage, base_score)"""
        result = []
        if pokemon.id == C.LOPUNNY:
            moved = board_index > 0 or self.active_moved
            result.append((A.GALE_THRUST, 1, False, 230 if moved else 60, 80))
            result.append((A.SPIKY_HOPPER, 2, False, 160, 40))
        elif pokemon.id == C.FROSLASS:
            result.append((A.REFRAIN, 1, True, 50 * self.opponent.handCount, 0))
            result.append((A.SNOW, 3, True, 150, -60))
        elif pokemon.id == C.ROTOM:
            if self.stadium_id:
                result.append((A.ASSAULT_LANDING, 1, False, 70, -150))
        elif pokemon.id == C.DUDUN:
            result.append((A.LAND_CRUSH, 3, False, 90, -350))
        elif pokemon.id == C.DUN:
            result.append((A.RAM, 2, False, 20, -450))
        elif pokemon.id == C.BUNEARY:
            result.append((A.KICK, 2, False, 20, -450))
        elif pokemon.id == C.SNORUNT:
            result.append((A.CHILLY, 1, True, 10, -500))
        return result

    def _plan_attack(self) -> None:
        global plan
        best_score = -1
        plan = AttackPlan()
        if self.state.turn < 2:
            return

        budget = 1 if not self.state.energyAttached and self.hand_energy > 0 else 0
        water_budget = 1 if not self.state.energyAttached and self.hand_water > 0 else 0
        for attacker_index, my_pokemon in enumerate(self._my_board()):
            if my_pokemon is None:
                continue
            if attacker_index != 0 and not self.can_switch:
                break
            attack_type = card_table[my_pokemon.id].energyType
            for attack_id, c_cost, needs_water, base_damage, base_score in self._attack_candidates(
                my_pokemon, attacker_index
            ):
                energy_count = len(my_pokemon.energies)
                has_water = EnergyType.WATER in my_pokemon.energies
                needs_energy = False
                if needs_water and not has_water:
                    if water_budget == 0:
                        continue
                    needs_energy = True
                if energy_count < c_cost:
                    if energy_count + budget >= c_cost:
                        needs_energy = True
                    else:
                        continue

                for target_index, op_pokemon in enumerate(self._opponent_board()):
                    if op_pokemon is None:
                        continue
                    if target_index != 0 and not self.can_gust:
                        break

                    damage = base_damage
                    if damage > 0:
                        op_data = card_table[op_pokemon.id]
                        if op_data.weakness == attack_type:
                            damage *= 2
                        elif op_data.resistance == attack_type:
                            damage = max(0, damage - 30)

                    score = target_score(op_pokemon)
                    prize = prize_count(op_pokemon) if op_pokemon.hp <= damage else 0
                    if prize == 0:
                        score *= damage / max(1, op_pokemon.hp)
                    else:
                        score += 500  # KO優先 (サイドレース)
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
                            damage=damage,
                        )

        self._decide_flee()

    def _decide_flee(self) -> None:
        """アクティブのメガが弱い攻撃しかできない/瀕死なら、ピボットに退避して3枚サイドを守る。"""
        active = self.me.active[0] if self.me.active else None
        if active is None or active.id not in MEGA_IDS or not self.can_switch:
            return
        if plan.attacker >= 1:
            return  # ダンス(ベンチアタッカーへの入替)が優先
        has_pivot = any(
            p is not None and p.id not in MEGA_IDS for p in self.me.bench
        )
        if not has_pivot:
            return
        if plan.attacker == 0 and plan.remain_hp <= 0:
            return  # KOが取れるなら攻撃を通す
        wally_in_hand = self.hand_counts[C.WALLY] > 0
        if active.hp <= 200 and not wally_in_hand:
            # 被弾済みのメガはベンチに退避してWallyで回復する (本番個体はLopunnyを1試合2回退避)
            plan.flee = True
        elif active.hp <= 240 and not wally_in_hand and self.bench_ready:
            # Wally無しでキル圏内(Shadow Bullet+Adrena~240)に居座らない。攻撃はダンサーが継続
            plan.flee = True

    # ---- エネルギー・ツールのアタッチ先 ----

    def _energy_target_score(self, pokemon: Pokemon, active: bool, energy_id: int) -> int:
        energy_count = len(pokemon.energies)
        score = 8000 + (10 if active else 0)
        if energy_id == C.W_ENERGY:
            if pokemon.id == C.FROSLASS:
                score += 900 if EnergyType.WATER not in pokemon.energies else -400
            elif pokemon.id == C.SNORUNT:
                score += 400 if energy_count < 1 else -500
            elif pokemon.id == C.LOPUNNY:
                score += 250 if energy_count < 2 else -400
            elif pokemon.id == C.BUNEARY:
                score += 120 if energy_count < 1 else -400
            elif pokemon.id == C.ROTOM:
                score += 80 if energy_count < 1 else -400
            else:
                score -= 400
            return score

        # Mist / Enriching (無色供給)
        if pokemon.id == C.LOPUNNY:
            if not active and energy_count == 0:
                score += 900  # 次のダンサーへの充填が最優先
            elif energy_count < 2:
                score += 700 if active else 250
            else:
                score -= 300
        elif pokemon.id == C.BUNEARY:
            score += 350 if energy_count < 1 else -350
        elif pokemon.id == C.FROSLASS:
            score += 150 if energy_count < 3 else -300
        elif pokemon.id == C.ROTOM:
            score += 120 if energy_count < 1 else -350
        elif pokemon.id == C.DUDUN:
            score -= 250
        else:
            score -= 300
        if energy_id == C.ENRICH:
            score += 120  # 手貼りで4ドロー
        return score

    def _board_index(self, area: AreaType, index: int) -> int:
        return index if area == AreaType.ACTIVE else index + 1

    def _score_attach(self, option) -> float:
        card = get_card(self.obs, option.area, option.index, self.my_index)
        pokemon = get_card(self.obs, option.inPlayArea, option.inPlayIndex, self.my_index)
        if card is None or not isinstance(pokemon, Pokemon):
            return 0

        if card.id == C.AIR_BALLOON:
            if pokemon.tools:
                return -1
            score = 6000
            if option.inPlayArea == AreaType.ACTIVE:
                score += 300
            if pokemon.id == C.LOPUNNY:
                score += 200
            elif pokemon.id == C.FROSLASS:
                score += 100
            elif pokemon.id in (C.ROTOM, C.DUDUN):
                score += 50
            return score

        score = self._energy_target_score(pokemon, option.inPlayArea == AreaType.ACTIVE, card.id)
        board_index = self._board_index(option.inPlayArea, option.inPlayIndex)
        if board_index == plan.attacker and plan.needs_energy:
            score += 2000
        if plan.flee and option.inPlayArea == AreaType.ACTIVE:
            score = 400  # 退避予定のアクティブには注がない
        if self.wally_urgent and option.inPlayArea == AreaType.ACTIVE:
            score = 400  # Wallyで全回復予定 (エネが手札に戻る) のアクティブには注がない
        if (
            option.inPlayArea == AreaType.ACTIVE
            and not self.can_switch
            and len(pokemon.energies) == 0
            and (
                (self.bench_ready and pokemon.id not in MEGA_IDS)
                or (pokemon.id in MEGA_IDS and pokemon.hp <= 200 and not self.wally_urgent)
            )
        ):
            # 詰まったアクティブにエネを貼って「にげる」を解放
            # (ピボット→ベンチのアタッカーへ / 瀕死メガ→3枚サイド献上の回避)
            score = 8600
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
            return self._score_evolve(option)
        if option.type == OptionType.ABILITY:
            return self._score_ability(option)
        if option.type == OptionType.RETREAT:
            if plan.attacker >= 1:
                return 2500  # ダンス: エネを払ってでも230が出るなら得
            if plan.flee:
                active = self.me.active[0] if self.me.active else None
                if self.retreat_free or (
                    active is not None
                    and (active.hp <= 130 or (active.id in MEGA_IDS and active.hp <= 200))
                ):
                    return 2400  # 無償 or 3枚サイド喪失回避ならエネを払っても退避
                return -1
            return -1
        if option.type == OptionType.ATTACK:
            return 1100 if option.attackId == plan.attack_id else 1000
        if option.type == OptionType.ENERGY_CARD or option.type == OptionType.ENERGY:
            return self._score_energy_pick(option)
        return 0

    def _score_evolve(self, option) -> float:
        pokemon = get_card(self.obs, option.inPlayArea, option.inPlayIndex, self.my_index)
        score = 21000
        if isinstance(pokemon, Pokemon):
            score += len(pokemon.energies) * 20
            if option.inPlayArea == AreaType.ACTIVE:
                score += 10
        return score

    def _score_energy_pick(self, option) -> float:
        # リトリートコスト等で捨てるエネルギー: W > Enriching > Mist の順で手放す
        pokemon = get_card(self.obs, option.area, option.index, self.my_index)
        if not isinstance(pokemon, Pokemon):
            return 0
        idx = option.energyIndex
        if idx is None or idx >= len(pokemon.energyCards):
            return 0
        eid = pokemon.energyCards[idx].id
        return {C.W_ENERGY: 100, C.ENRICH: 60, C.MIST: 30}.get(eid, 50)

    def _score_ability(self, option) -> float:
        card = get_card(self.obs, option.area, option.index, self.my_index)
        if card is None:
            return 25000  # 相手スタジアム等の起動効果
        if card.id == C.DUDUN:
            board = sum(1 for p in self._my_board() if p is not None)
            if self.me.deckCount <= 2:
                return -1
            if option.area == AreaType.ACTIVE and board <= 1:
                return -1
            return 26000
        if card.id == C.ROTOM:
            return 24000  # Fan Call (エンジンが初ターンのみ提示)
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
        cid = card.id
        if cid == C.DUN:
            engine = self.field_counts[C.DUN] + self.field_counts[C.DUDUN]
            if engine < 2:
                return 19000
            # 1サイドの壁+エンジン供給として追加展開は許容、ただし枠は攻め筋優先
            board = sum(1 for p in self._my_board() if p is not None)
            return 9000 if board <= 4 else -1
        if cid == C.BUNEARY:
            line = self.field_counts[C.BUNEARY] + self.field_counts[C.LOPUNNY]
            return 20000 if line < 2 else 8000
        if cid == C.SNORUNT:
            line = self.field_counts[C.SNORUNT] + self.field_counts[C.FROSLASS]
            return 18000 if line < 2 else 6000
        if cid == C.ROTOM:
            return 18500 if self.state.turn <= 2 else 8000
        return 15000

    def _score_play_trainer(self, card: Card) -> float:
        cid = card.id
        if cid == C.POFFIN:
            return 10000
        if cid == C.POKE_PAD:
            return 9900
        if cid == C.ULTRA_BALL:
            return 9800 if len(self.me.hand) >= 3 else -1
        if cid == C.POKEGEAR:
            return 9600
        if cid == C.TRIMMER:
            if (
                self.opponent.handCount >= 6
                and len(self.me.hand) <= 6
                and plan.attack_id != A.REFRAIN
            ):
                return 3000
            return -1
        if cid == C.BATTLE_CAGE:
            if self.stadium_id and self.stadium_is_mine:
                return -1
            return 3150
        if cid == C.BOSS_ORDERS:
            return 3250 if plan.target >= 1 else -1
        if cid == C.WALLY:
            active = self.me.active[0] if self.me.active else None
            active_urgent = (
                active is not None
                and active.id in MEGA_IDS
                and active.hp <= 240
                and active.maxHp - active.hp >= 90
            )
            if active_urgent and (not self.state.energyAttached or active.hp <= 130):
                # 手貼りが残っていれば回復後に貼り直して攻撃継続できる。瀕死なら攻撃を捨てても回復
                return 3400
            bench_dmg = max(
                (p.maxHp - p.hp for p in self.me.bench if p is not None and p.id in MEGA_IDS),
                default=0,
            )
            if bench_dmg >= 120:
                return 3300  # ベンチで休んでいるメガの回復はテンポ損なし
            return -1
        if cid == C.HILDA:
            if self.hand_energy == 0 or self._missing_evolution():
                return 3200
            return 2700
        if cid == C.LILLIE:
            if self.me.deckCount <= LOW_DECK_COUNT:
                return -1
            return 3100 if len(self.me.hand) <= 5 else -1
        return 2500

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
            return -1  # 本番個体はセットアップでベンチに出さない (257戦で0回)
        if self.context in {SelectContext.TO_HAND, SelectContext.TO_BENCH, SelectContext.TO_FIELD}:
            return self._score_to_hand(card)
        if self.context in {SelectContext.HEAL, SelectContext.REMOVE_DAMAGE_COUNTER}:
            if isinstance(card, Pokemon):
                dmg = card.maxHp - card.hp
                score = dmg + (50 if card.id == C.LOPUNNY else 0)
                if option.area == AreaType.ACTIVE and card.id in MEGA_IDS:
                    if card.hp <= 240 and (not self.state.energyAttached or card.hp <= 130):
                        score += 150  # キル圏内のアクティブメガ優先 (回復後に手貼りで攻撃継続可)
                    else:
                        score -= 120  # アタッカーのエネを剥がさない (ベンチのメガを優先)
                return score
            return 0
        if self.context == SelectContext.ATTACH_FROM and isinstance(card, Pokemon):
            energy_id = self.select.effect.id if self.select.effect is not None else C.MIST
            score = self._energy_target_score(card, option.area == AreaType.ACTIVE, energy_id)
            board_index = self._board_index(option.area, option.index)
            if board_index == plan.attacker and plan.needs_energy:
                score += 2000
            return score
        if self.context in {SelectContext.DISCARD, SelectContext.TO_DECK, SelectContext.TO_DECK_BOTTOM}:
            return self._score_discard(card)
        return 0

    def _is_my_turn(self) -> bool:
        if self.state.firstPlayer < 0:
            return True
        return (self.state.turn % 2 == 1) == (self.state.firstPlayer == self.my_index)

    def _score_active_choice(self, option, card: Pokemon | Card) -> float:
        if not isinstance(card, Pokemon):
            return 0
        if option.playerIndex != self.my_index:
            return 100 if option.index == plan.target - 1 else 0

        if plan.flee:
            # 自ターンの退避: 攻撃できる消耗品 > 消耗品ピボット > メガ温存
            score = {
                C.DUDUN: 180,  # Run Away Drawで無償退場できる
                C.DUN: 140,
                C.ROTOM: 130,
                C.SNORUNT: 70,
                C.BUNEARY: 60,
            }.get(card.id, 20)
            if card.id == C.ROTOM and len(card.energies) >= 1 and self.stadium_id:
                score = 210  # Assault Landing 70で攻撃継続できる退避先
            if card.id == C.LOPUNNY:
                score = 100 if card.hp > 200 else 30
            elif card.id == C.FROSLASS:
                score = 80 if card.hp > 200 and EnergyType.WATER in card.energies else 25
            if any(t.id == C.AIR_BALLOON for t in card.tools):
                score += 50  # 風船持ちピボットなら次のターン無償でダンス再開できる
            return score

        if not self._is_my_turn():
            # 相手ターン中のKO後昇格: 頑丈なLopunnyを前に(本番個体はLopunny昇格が最多)
            score = {
                C.DUDUN: 150,
                C.DUN: 120,
                C.ROTOM: 110,
                C.SNORUNT: 60,
                C.BUNEARY: 55,
            }.get(card.id, 20)
            if card.id == C.LOPUNNY:
                score = 180 if card.hp > 200 else 40
            elif card.id == C.FROSLASS:
                score = 140 if card.hp > 200 else 35
            if any(t.id == C.AIR_BALLOON for t in card.tools):
                score += 30
            return score

        score = len(card.energies) * 5
        if option.index == plan.attacker - 1:
            score += 250
        if card.id == C.LOPUNNY:
            score += 100 + 20 * min(2, len(card.energies))
        elif card.id == C.FROSLASS:
            score += 60
            if EnergyType.WATER in card.energies:
                score += min(8, self.opponent.handCount) * 8
        elif card.id == C.DUDUN:
            score += 50  # Run Away Drawで無償で退場できるピボット
        elif card.id == C.ROTOM:
            score += 45
        elif card.id == C.DUN:
            score += 40
        else:
            score += 20
        if any(t.id == C.AIR_BALLOON for t in card.tools):
            score += 15
        return score

    def _score_setup_active(self, card: Pokemon | Card) -> int:
        return {
            C.ROTOM: 5,
            C.DUN: 4,
            C.BUNEARY: 3,
            C.SNORUNT: 2,
        }.get(card.id, 1)

    def _score_to_hand(self, card: Pokemon | Card) -> float:
        cid = card.id
        score = 200 - self.hand_counts[cid] * 60
        lop_field = self.field_counts[C.BUNEARY] + self.field_counts[C.LOPUNNY]
        if cid == C.DUDUN:
            engine = self.field_counts[C.DUN] + self.field_counts[C.DUDUN]
            score += 250 if self.hand_counts[C.DUDUN] == 0 and engine > 0 and lop_field > 0 else 60
        elif cid == C.DUN:
            score += 180 if self.field_counts[C.DUN] + self.hand_counts[C.DUN] < 2 else 20
        elif cid == C.LOPUNNY:
            have = self.field_counts[C.LOPUNNY] + self.hand_counts[C.LOPUNNY]
            near = self.field_counts[C.BUNEARY] + self.hand_counts[C.BUNEARY]
            score += 320 if have < 2 and near > 0 else -60
        elif cid == C.BUNEARY:
            total = lop_field + self.hand_counts[C.BUNEARY]
            score += 340 if lop_field == 0 and self.hand_counts[C.BUNEARY] == 0 else (
                220 if total < 2 else -60
            )
        elif cid == C.FROSLASS:
            score += 130 if self.field_counts[C.SNORUNT] > 0 and self.hand_counts[C.FROSLASS] == 0 else -60
        elif cid == C.SNORUNT:
            score += 90 if self.field_counts[C.SNORUNT] + self.field_counts[C.FROSLASS] < 2 else -60
        elif cid == C.ROTOM:
            score += 60 if self.state.turn <= 2 else -40
        elif cid == C.MIST:
            score += 170 - 60 * min(2, self.hand_energy)
        elif cid == C.ENRICH:
            score += 180 - 60 * min(2, self.hand_energy)
        elif cid == C.W_ENERGY:
            has_fro = self.field_counts[C.FROSLASS] + self.field_counts[C.SNORUNT] > 0
            score += (140 if has_fro else 60) - 40 * min(2, self.hand_water)
        elif cid == C.WALLY:
            score += 150 if self._damaged_mega() >= 100 else 10
        elif cid == C.HILDA:
            score += 100
        elif cid == C.LILLIE:
            score += 110 if len(self.me.hand) <= 4 else 20
        elif cid == C.TRIMMER:
            score += 40 if self.opponent.handCount >= 6 else -20
        elif cid == C.BATTLE_CAGE:
            score += 40 if not self.stadium_id else -20
        elif cid == C.AIR_BALLOON:
            score += 70
        elif cid == C.BOSS_ORDERS:
            score += 10
        return score

    def _score_discard(self, card: Pokemon | Card) -> float:
        cid = card.id
        score = 0
        if self.hand_counts[cid] >= 2:
            score += 70
        if cid == C.BOSS_ORDERS:
            score += 90
        elif cid == C.POKEGEAR:
            score += 40
        elif cid == C.TRIMMER:
            score += 35 if self.opponent.handCount < 6 else 0
        elif cid == C.BATTLE_CAGE:
            score += 40 if self.stadium_is_mine else 10
        elif cid == C.W_ENERGY:
            score += 50 if self.hand_water >= 2 else -40
        elif cid == C.MIST:
            score += 25 if self.hand_energy >= 3 else -60
        elif cid == C.ENRICH:
            score -= 100
        elif cid == C.AIR_BALLOON:
            score -= 40
        elif cid == C.WALLY:
            score += 30 if self.field_counts[C.LOPUNNY] + self.field_counts[C.FROSLASS] == 0 else -20
        elif cid == C.LILLIE:
            score += 20 if len(self.me.hand) >= 7 else -20
        data = card_table.get(cid)
        if data is not None and data.cardType == CardType.POKEMON:
            score -= 100
            if cid in MEGA_IDS:
                score -= 200
            elif cid == C.DUDUN and self.hand_counts[cid] < 2:
                score -= 80
        return score


def agent(obs_dict: dict) -> list[int]:
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        return my_deck

    global pre_turn
    global plan
    global turn_start_serial

    if pre_turn != obs.current.turn:
        pre_turn = obs.current.turn
        plan = AttackPlan()
        me = obs.current.players[obs.current.yourIndex]
        active = me.active[0] if me.active else None
        turn_start_serial = active.serial if active is not None else -1

    return LopunnyPolicy(obs).choose()
