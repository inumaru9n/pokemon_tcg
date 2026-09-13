"""083_spidops_d583 — Team Rocket's Spidops レプリカ (sig d5834c603a, 本番634戦53.5%).

islet骨格 (004/037) をTeam Rocket's Spidops (Mewtwo ex) デッキ用に書き換えた
ヒューリスティック方策。d5834c603a個体のリプレイ302戦の行動頻度集計と
対Alakazam勝ちトレース精読 (episode 87366420) に基づく:

- 主砲は Team Rocket's Mewtwo ex (280HP) の Erasure Ball 160 (+60×ベンチエネ破棄, 最大280)
- Articuno の Repelling Veil が「たねロケット団ポケモンへのワザ効果」を全て無効化
  → Alakazam の Powerful Hand (ダメカンばら撒き=効果) が Mewtwo/たねに0になる (対Alakazam 86%の核)
- Spidops の Charging Up でトラッシュの基本エネをSpidopsに毎ターン回収
  → Erasure Ball の破棄フォダーとして無限ループ
- Rocket Rush = 30×自分のロケット団ポケモン数 (全員ロケット団なので盤面数)
- Power Saver: Mewtwo は場に4匹以上いないと攻撃不可 → 盤面は常にワイドに展開
- Archer (相手手札3枚に) は Powerful Hand の打点も削る二重の対Alakazamテック
- Giovanni で Mewtwo を前に出しつつ相手ベンチの瀕死ex/無力ポケモンを釣り出す
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
    all_attack,
    all_card_data,
    to_observation_class,
)


class C:
    G_ENERGY = 1
    P_ENERGY = 5
    TR_ENERGY = 15

    TAROUNTULA = 400
    SPIDOPS = 401
    ARTICUNO = 414
    MEWTWO = 431
    WOBBUFFET = 432
    MURKROW = 463

    BUG_CATCHING = 1094
    NIGHT_STRETCHER = 1097
    ENERGY_SEARCH = 1119
    TRANSCEIVER = 1134
    POKE_PAD = 1152
    HEROS_CAPE = 1159
    BRAVE_BANGLE = 1175
    ARIANA = 1216
    ARCHER = 1217
    GIOVANNI = 1218
    PETREL = 1219
    PROTON = 1220
    LILLIE = 1227
    FACTORY = 1257

    LILLIES_PEARL = 1172
    LEGACY_ENERGY = 12


class A:
    TAKE_DOWN = 559
    ROCKET_RUSH = 560
    DARK_FROST = 583
    ERASURE_BALL = 608
    ROCKET_MIRROR = 609
    HEADBUTT_BOUNCE = 610
    DECEIT = 652
    TORMENT = 653


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
attack_table = {attack.attackId: attack for attack in all_attack()}

FODDER_IDS = {C.SPIDOPS, C.ARTICUNO, C.TAROUNTULA, C.MURKROW, C.WOBBUFFET}


class AttackPlan:
    def __init__(
        self,
        attacker: int = -1,
        target: int = -1,
        attack_id: int = -1,
        remain_hp: int = -1,
        needs_energy: bool = False,
        discard_need: int = 0,
        fodder_attach_wanted: bool = False,
    ):
        self.attacker = attacker
        self.target = target
        self.attack_id = attack_id
        self.remain_hp = remain_hp
        self.needs_energy = needs_energy
        self.discard_need = discard_need  # Erasure Ballで破棄したいベンチエネ数
        self.fodder_attach_wanted = fodder_attach_wanted


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


def energy_units(pokemon: Pokemon) -> int:
    return sum(2 if e == EnergyType.TEAM_ROCKET else 1 for e in pokemon.energies)


def p_units(pokemon: Pokemon) -> int:
    total = 0
    for e in pokemon.energies:
        if e == EnergyType.TEAM_ROCKET:
            total += 2
        elif e in (EnergyType.PSYCHIC, EnergyType.RAINBOW):
            total += 1
    return total


def g_units(pokemon: Pokemon) -> int:
    return sum(1 for e in pokemon.energies if e in (EnergyType.GRASS, EnergyType.RAINBOW))


def d_units(pokemon: Pokemon) -> int:
    total = 0
    for e in pokemon.energies:
        if e == EnergyType.TEAM_ROCKET:
            total += 2
        elif e in (EnergyType.DARKNESS, EnergyType.RAINBOW):
            total += 1
    return total


def target_score(pokemon: Pokemon) -> int:
    score = prize_count(pokemon) * 1000
    score += len(pokemon.energies) * 150
    score += len(pokemon.tools) * 100
    data = card_table[pokemon.id]
    if data.stage2:
        score += 250
    elif data.stage1:
        score += 130
    score += pokemon.hp
    return score


class SpidopsPolicy:
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
        self.stadium_id = self.state.stadium[0].id if self.state.stadium else 0
        self.stadium_is_mine = bool(self.state.stadium) and self.state.stadium[0].playerIndex == self.my_index

        self._count_cards()
        self._scan_main_options()

        self.board_count = sum(1 for p in self._my_board() if p is not None)
        self.hand_g = self.hand_counts[C.G_ENERGY]
        self.hand_p = self.hand_counts[C.P_ENERGY]
        self.hand_tr = self.hand_counts[C.TR_ENERGY]
        self.hand_basic_energy = self.hand_g + self.hand_p
        self.articuno_in_play = self.field_counts[C.ARTICUNO] >= 1
        self.spidops_in_play = self.field_counts[C.SPIDOPS] >= 1
        # ベンチのエネルギー数（Erasure Ballの破棄フォダー）
        self.bench_fodder_free = 0  # Spidopsの上のエネ（Charging Upで回収可能=タダ）
        self.bench_fodder_any = 0
        for p in self.me.bench:
            if p is None:
                continue
            n = len(p.energies)
            self.bench_fodder_any += n
            if p.id == C.SPIDOPS:
                self.bench_fodder_free += n

    def choose(self) -> list[int]:
        if not self.select.option or self.select.maxCount == 0:
            return []
        if self.context == SelectContext.MAIN:
            self._plan_attack()
        if self.context == SelectContext.DISCARD_ENERGY_CARD:
            return self._choose_energy_card_discard()
        scores = [self._score_option(option) for option in self.select.option]
        ranked = [i for i, _ in sorted(enumerate(scores), key=lambda item: item[1], reverse=True)]
        picked: list[int] = []
        for i in ranked:
            if len(picked) >= self.select.maxCount:
                break
            if scores[i] < 0 and len(picked) >= self.select.minCount:
                break
            picked.append(i)
        return picked

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

    def _opponent_has_rulebox(self) -> bool:
        for pokemon in self._opponent_board():
            if pokemon is None:
                continue
            data = card_table[pokemon.id]
            if data.ex or data.megaEx:
                return True
        return False

    def _est_next_damage(self, target: Pokemon | None = None, only_ready: bool = True) -> int:
        """自分の盤面から出せる最大打点の概算（弱点込み。Giovanni/釣り出し/昇格判断用）"""
        best = 0
        weakness = card_table[target.id].weakness if target is not None else None
        for pokemon in self._my_board():
            if pokemon is None:
                continue
            damage = 0
            if pokemon.id == C.MEWTWO:
                if only_ready and not (p_units(pokemon) >= 2 and energy_units(pokemon) >= 3):
                    continue
                damage = 160 + 60 * min(2, self.bench_fodder_any)
                if weakness == EnergyType.PSYCHIC:
                    damage *= 2
            elif pokemon.id == C.SPIDOPS:
                if only_ready and not (g_units(pokemon) >= 1 and energy_units(pokemon) >= 2):
                    continue
                damage = 30 * self.board_count
                if weakness == EnergyType.GRASS:
                    damage *= 2
            best = max(best, damage)
        return best

    def _opp_max_hit_on(self, candidate: Pokemon) -> int:
        """相手盤面のワザ表記ダメージ最大値（弱点込み）: 前に出す個体のワンパン被弾リスク推定"""
        weakness = card_table[candidate.id].weakness
        best = 0
        for op_pokemon in self._opponent_board():
            if op_pokemon is None:
                continue
            op_data = card_table[op_pokemon.id]
            for attack_id in op_data.attacks:
                attack = attack_table.get(attack_id)
                if attack is None or attack.damage <= 0:
                    continue
                damage = attack.damage
                if weakness is not None and weakness == op_data.energyType:
                    damage *= 2
                best = max(best, damage)
        return best

    def _pokemon_damage_vs(self, pokemon: Pokemon, target: Pokemon | None) -> tuple[int, bool]:
        """(この個体が出せる打点の概算, 今すぐ撃てるか)"""
        weakness = card_table[target.id].weakness if target is not None else None
        if pokemon.id == C.MEWTWO:
            damage = 160 + 60 * min(2, self.bench_fodder_any)
            if weakness == EnergyType.PSYCHIC:
                damage *= 2
            return damage, p_units(pokemon) >= 2 and energy_units(pokemon) >= 3
        if pokemon.id == C.SPIDOPS:
            damage = 30 * self.board_count
            if weakness == EnergyType.GRASS:
                damage *= 2
            return damage, g_units(pokemon) >= 1 and energy_units(pokemon) >= 2
        if pokemon.id == C.WOBBUFFET:
            damage = 70 * (2 if weakness == EnergyType.PSYCHIC else 1)
            return damage, p_units(pokemon) >= 1 and energy_units(pokemon) >= 3
        return 0, False

    # ---- 攻撃プラン ----

    def _attack_candidates(self, pokemon: Pokemon) -> list[tuple[int, bool, int, int]]:
        """(attack_id, affordable_now, base_damage, base_score)  base_damage<0は特殊"""
        units = energy_units(pokemon)
        result = []
        if pokemon.id == C.MEWTWO:
            ok = p_units(pokemon) >= 2 and units >= 3
            result.append((A.ERASURE_BALL, ok, 160, 0))
        elif pokemon.id == C.SPIDOPS:
            ok = g_units(pokemon) >= 1 and units >= 2
            result.append((A.ROCKET_RUSH, ok, 30 * self.board_count, -50))
        elif pokemon.id == C.TAROUNTULA:
            ok = g_units(pokemon) >= 1
            result.append((A.TAKE_DOWN, ok, 30, -300))
        elif pokemon.id == C.MURKROW:
            ok = d_units(pokemon) >= 1 and units >= 2
            result.append((A.TORMENT, ok, 30, -350))
            result.append((A.DECEIT, units >= 1, 0, -600))
        elif pokemon.id == C.WOBBUFFET:
            ok = p_units(pokemon) >= 1 and units >= 3
            result.append((A.HEADBUTT_BOUNCE, ok, 70, -200))
        return result

    def _attack_type(self, attack_id: int) -> EnergyType:
        if attack_id in (A.ERASURE_BALL, A.HEADBUTT_BOUNCE):
            return EnergyType.PSYCHIC
        if attack_id in (A.ROCKET_RUSH, A.TAKE_DOWN):
            return EnergyType.GRASS
        return EnergyType.DARKNESS

    def _plan_attack(self) -> None:
        global plan
        best_score = -1
        plan = AttackPlan()
        if self.state.turn < 2:
            return

        can_hand_attach = not self.state.energyAttached and (self.hand_basic_energy + self.hand_tr) > 0

        can_reach_bench = self.can_switch or self.hand_counts[C.GIOVANNI] > 0
        for attacker_index, my_pokemon in enumerate(self._my_board()):
            if my_pokemon is None:
                continue
            if attacker_index != 0 and not can_reach_bench:
                break
            for attack_id, affordable, base_damage, base_score in self._attack_candidates(my_pokemon):
                needs_energy = False
                if not affordable:
                    # 手貼り1枚で足りるかの概算（ベンチアタッカーにも許可: 対Garchompの
                    # Rocket Rushラインで必須。実際の手貼りはneeds_energyブーストが誘導する）
                    if not can_hand_attach:
                        continue
                    units = energy_units(my_pokemon)
                    pu = p_units(my_pokemon)
                    gu = g_units(my_pokemon)
                    if attack_id == A.ERASURE_BALL:
                        pu_add = 2 if self.hand_tr else (1 if self.hand_p else 0)
                        if not (pu + pu_add >= 2 and units + (2 if self.hand_tr else 1) >= 3):
                            continue
                    elif attack_id == A.ROCKET_RUSH:
                        if not ((gu + (1 if self.hand_g else 0) >= 1) and units + 1 >= 2):
                            continue
                    elif attack_id == A.TAKE_DOWN:
                        if not self.hand_g:
                            continue
                    else:
                        continue
                    needs_energy = True

                atk_type = self._attack_type(attack_id)
                for target_index, op_pokemon in enumerate(self._opponent_board()):
                    if op_pokemon is None:
                        continue
                    if target_index != 0:
                        break  # このデッキに自発ガスト手段はGiovanniのみ（供給トリガーは別枠）

                    damage = base_damage
                    bangle = any(t.id == C.BRAVE_BANGLE for t in my_pokemon.tools)
                    op_data = card_table[op_pokemon.id]
                    if bangle and (op_data.ex or op_data.megaEx):
                        damage += 30
                    discard_need = 0
                    fodder_attach_wanted = False
                    if attack_id == A.ERASURE_BALL and damage > 0:
                        eff_hp = op_pokemon.hp
                        if op_data.weakness == atk_type:
                            eff_hp = (eff_hp + 1) // 2
                        if eff_hp > 160:
                            need = min(2, max(0, -(-(eff_hp - 160) // 60)))
                            avail = self.bench_fodder_any
                            extra = 0
                            if avail < need and can_hand_attach and not needs_energy and self.hand_basic_energy > 0:
                                extra = 1
                                fodder_attach_wanted = True
                            discard_need = min(need, avail + extra)
                            damage += 60 * discard_need
                        elif self.bench_fodder_free > 0:
                            # タダのフォダー（Charging Upで回収できるSpidopsのエネ）は常に上乗せ
                            discard_need = min(2, self.bench_fodder_free)
                            damage += 60 * discard_need
                    if op_data.weakness == atk_type:
                        damage *= 2
                    elif op_data.resistance == atk_type:
                        damage = max(0, damage - 30)

                    score = target_score(op_pokemon)
                    prize = prize_count(op_pokemon) if op_pokemon.hp <= damage else 0
                    if prize == 0:
                        score *= damage / max(1, op_pokemon.hp)
                    if len(self.opponent.prize) <= prize:
                        score = 50000

                    score += base_score
                    score += 220 if attacker_index == 0 else 0
                    score += energy_units(my_pokemon)

                    if score > best_score:
                        best_score = score
                        plan = AttackPlan(
                            attacker=attacker_index,
                            target=target_index,
                            attack_id=attack_id,
                            remain_hp=op_pokemon.hp - damage,
                            needs_energy=needs_energy,
                            discard_need=discard_need,
                            fodder_attach_wanted=fodder_attach_wanted,
                        )

    # ---- Erasure Ball のベンチエネ破棄 ----

    def _choose_energy_card_discard(self) -> list[int]:
        options = self.select.option
        scores = []
        for option in options:
            holder = get_card(self.obs, option.area, option.index, self.my_index)
            score = 0
            if isinstance(holder, Pokemon):
                if holder.id == C.SPIDOPS:
                    score = 600  # Charging Upで回収できるのでタダ
                elif holder.id == C.MEWTWO:
                    score = -400  # 予備アタッカーのエネは温存
                else:
                    score = 100
            scores.append(score)
        ranked = [i for i, _ in sorted(enumerate(scores), key=lambda item: item[1], reverse=True)]
        picked: list[int] = []
        want = max(plan.discard_need, self.select.minCount)
        for i in ranked:
            if len(picked) >= self.select.maxCount:
                break
            if len(picked) < want:
                picked.append(i)
            elif scores[i] >= 500:  # タダのフォダーは追加で破棄
                picked.append(i)
            else:
                break
        return picked

    # ---- エネルギー・ツールのアタッチ先 ----

    def _energy_target_score(self, pokemon: Pokemon, active: bool, energy_id: int) -> int:
        units = energy_units(pokemon)
        pu = p_units(pokemon)
        score = 8000 + (10 if active else 0)

        if energy_id == C.TR_ENERGY:
            if pokemon.id == C.MEWTWO:
                score += 4000 if (pu < 2 or units < 3) else 700
            elif pokemon.id == C.MURKROW:
                score += 200 if units < 2 else -900
            elif pokemon.id == C.WOBBUFFET:
                score += 100 if units < 3 else -900
            else:
                score -= 1200
            return score

        if energy_id == C.P_ENERGY:
            if pokemon.id == C.MEWTWO:
                score += 3000 if pu < 2 else (900 if units < 5 else -500)
            elif pokemon.id == C.WOBBUFFET:
                score += 100 if units < 3 else -800
            elif pokemon.id == C.SPIDOPS:
                score += 200 if units < 2 else -600
            else:
                score -= 900
            return score

        # 草エネルギー
        if pokemon.id == C.MEWTWO:
            score += 2000 if units < 3 else (600 if units < 5 else -500)
        elif pokemon.id == C.SPIDOPS:
            score += 1800 if g_units(pokemon) == 0 else (300 if units < 2 else -400)
        elif pokemon.id == C.TAROUNTULA:
            score += 900 if units == 0 else -500
        elif pokemon.id == C.MURKROW:
            score += 250 if units < 2 else -700
        elif pokemon.id == C.ARTICUNO:
            score += 150 if units == 0 else -700
        elif pokemon.id == C.WOBBUFFET:
            score += 50 if units < 3 else -800
        return score

    def _board_index(self, area: AreaType, index: int) -> int:
        return index if area == AreaType.ACTIVE else index + 1

    def _score_attach(self, option) -> float:
        card = get_card(self.obs, option.area, option.index, self.my_index)
        pokemon = get_card(self.obs, option.inPlayArea, option.inPlayIndex, self.my_index)
        if card is None or not isinstance(pokemon, Pokemon):
            return 0

        if card.id == C.HEROS_CAPE:
            if pokemon.tools:
                return -1
            if pokemon.id == C.MEWTWO:
                return 6000
            if pokemon.id == C.SPIDOPS:
                return 3000
            return 1200
        if card.id == C.BRAVE_BANGLE:
            if pokemon.tools:
                return -1
            if not self._opponent_has_rulebox():
                return 500
            if pokemon.id == C.SPIDOPS:
                return 2600
            if pokemon.id == C.TAROUNTULA:
                return 1500
            return 400

        board_index = self._board_index(option.inPlayArea, option.inPlayIndex)
        if plan.fodder_attach_wanted and card.id in (C.G_ENERGY, C.P_ENERGY):
            if option.inPlayArea == AreaType.BENCH and pokemon.id in FODDER_IDS:
                return 8800  # Erasure Ballの+60用フォダーをベンチに置く
        score = self._energy_target_score(pokemon, option.inPlayArea == AreaType.ACTIVE, card.id)
        if board_index == plan.attacker and plan.needs_energy:
            score += 5000
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
            return 21000
        if option.type == OptionType.ABILITY:
            return self._score_ability(option)
        if option.type == OptionType.RETREAT:
            return 2000 if plan.attacker >= 1 else -1
        if option.type == OptionType.ATTACK:
            return 1100 if option.attackId == plan.attack_id else 1000
        if option.type in (OptionType.ENERGY_CARD, OptionType.ENERGY):
            return self._score_energy_pick(option)
        return 0

    def _score_energy_pick(self, option) -> float:
        """DISCARD_ENERGY等（にげるコスト支払い・相手効果）: 草を先に切りTR/超を守る"""
        holder = get_card(self.obs, option.area, option.index, self.my_index)
        score = 50
        if isinstance(holder, Pokemon):
            try:
                etype = holder.energies[option.energyIndex]
            except (IndexError, TypeError):
                etype = EnergyType.COLORLESS
            if etype == EnergyType.GRASS:
                score += 100
            elif etype == EnergyType.PSYCHIC:
                score += 30
            elif etype == EnergyType.TEAM_ROCKET:
                score -= 40
        return score

    def _score_ability(self, option) -> float:
        card = get_card(self.obs, option.area, option.index, self.my_index)
        if card is None:
            return 25000  # 相手スタジアム等の起動効果
        if card.id == C.SPIDOPS:
            return 26000  # Charging Up: 毎ターン使う
        if card.id == C.FACTORY:
            return 25000 if self.me.deckCount >= 3 else -1
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
        if cid == C.ARTICUNO:
            return 20000 if not self.articuno_in_play else 14500
        if cid == C.TAROUNTULA:
            return 19000
        if cid == C.MEWTWO:
            return 18500 if self.field_counts[C.MEWTWO] < 2 else 13000
        if cid == C.MURKROW:
            return 15000
        if cid == C.WOBBUFFET:
            return 14000
        return 15000

    def _score_play_trainer(self, card: Card) -> float:
        cid = card.id
        hand_size = len(self.me.hand)
        # --- グッズ ---
        if cid == C.NIGHT_STRETCHER:
            return 10000
        if cid in (C.TRANSCEIVER, C.POKE_PAD, C.BUG_CATCHING):
            return 9500
        if cid == C.ENERGY_SEARCH:
            return 9400
        # --- スタジアム ---
        if cid == C.FACTORY:
            if self.stadium_id == C.FACTORY and self.stadium_is_mine:
                return -1
            return 3150
        # --- サポート ---
        if cid == C.ARIANA:
            if self._low_deck():
                return -1
            if hand_size <= 2:
                return 3600
            if hand_size <= 4:
                return 3250
            if hand_size <= 6:
                return 2200
            return -1
        if cid == C.PROTON:
            want_basics = self.board_count < 6
            in_deck_est = 15 - sum(
                self.field_counts[i] + self.hand_counts[i] + self.discard_counts[i]
                for i in (C.TAROUNTULA, C.SPIDOPS, C.ARTICUNO, C.MEWTWO, C.WOBBUFFET, C.MURKROW)
            )
            if want_basics and in_deck_est > 0:
                return 3400 if self.state.turn <= 5 else 2600
            return -1
        if cid == C.ARCHER:
            # 使える時点でロケット団が前ターン倒されている（エンジンが条件を管理）
            if self._low_deck():
                return -1
            if hand_size <= 5:
                return 3450
            if self.opponent.handCount >= 8:
                return 3300
            return 1500
        if cid == C.GIOVANNI:
            return self._score_giovanni()
        if cid == C.LILLIE:
            if self._low_deck():
                return -1
            if hand_size <= 4:
                return 2800
            return -1
        if cid == C.PETREL:
            return 1800
        return 2000

    def _score_giovanni(self) -> float:
        active = self.me.active[0] if self.me.active else None
        op_active = self.opponent.active[0] if self.opponent.active else None

        def ohko_ready(p: Pokemon | None) -> bool:
            if p is None or op_active is None:
                return False
            damage, ready = self._pokemon_damage_vs(p, op_active)
            return ready and damage >= op_active.hp

        def any_ready(p: Pokemon | None) -> bool:
            if p is None:
                return False
            _, ready = self._pokemon_damage_vs(p, op_active)
            return ready

        bench_better = any(
            (ohko_ready(p) and not ohko_ready(active)) or (any_ready(p) and not any_ready(active))
            for p in self.me.bench
        )
        if bench_better:
            return 3350
        gust_kill = any(
            p is not None and p.hp <= self._est_next_damage(p)
            for p in self.opponent.bench
        )
        if gust_kill and any_ready(active):
            if op_active is not None and plan.remain_hp > 0:
                return 3320  # 前は倒せないがベンチに倒せる獲物がいる
        if active is not None and active.id == C.MEWTWO and active.hp <= 100 and self.field_counts[C.MEWTWO] >= 2:
            return 3050  # 傷んだMewtwoを下げて2体目と交代
        return -1

    # ---- カード選択（サーチ・入替等） ----

    def _score_card_choice(self, option) -> float:
        card = get_card(self.obs, option.area, option.index, option.playerIndex)
        if card is None:
            return 0
        if self.context in {SelectContext.SWITCH, SelectContext.TO_ACTIVE}:
            return self._score_active_choice(option, card)
        if self.context == SelectContext.SETUP_ACTIVE_POKEMON:
            return {C.MEWTWO: 5, C.MURKROW: 4, C.TAROUNTULA: 3, C.WOBBUFFET: 2, C.ARTICUNO: 1}.get(card.id, 0)
        if self.context == SelectContext.SETUP_BENCH_POKEMON:
            return {C.TAROUNTULA: 5, C.MEWTWO: 4, C.ARTICUNO: 3, C.MURKROW: 2, C.WOBBUFFET: 1}.get(card.id, 0)
        if self.context == SelectContext.TO_HAND:
            return self._score_to_hand(card)
        if self.context == SelectContext.ATTACH_TO:
            # Charging Up: トラッシュから拾うエネの選択（Spidopsに付く）
            return {C.G_ENERGY: 100, C.P_ENERGY: 50}.get(card.id, 10)
        if self.context == SelectContext.ATTACH_FROM and isinstance(card, Pokemon):
            energy_id = self.select.effect.id if self.select.effect is not None else C.G_ENERGY
            score = self._energy_target_score(card, option.area == AreaType.ACTIVE, energy_id)
            board_index = self._board_index(option.area, option.index)
            if board_index == plan.attacker and plan.needs_energy:
                score += 5000
            return score
        if self.context == SelectContext.REMOVE_DAMAGE_COUNTER and isinstance(card, Pokemon):
            return card.maxHp - card.hp
        if self.context in {SelectContext.DISCARD, SelectContext.TO_DECK, SelectContext.TO_DECK_BOTTOM}:
            return self._score_discard(card)
        return 0

    def _score_active_choice(self, option, card: Pokemon | Card) -> float:
        if not isinstance(card, Pokemon):
            return 0
        if option.playerIndex != self.my_index:
            return self._score_gust_target(card)

        units = energy_units(card)
        score = units * 3
        if card.id == C.MEWTWO:
            score += 200
            if p_units(card) >= 2 and units >= 3:
                score += 120
        elif card.id == C.SPIDOPS:
            score += 150
            if g_units(card) >= 1 and units >= 2:
                score += 60
        elif card.id == C.TAROUNTULA:
            score += 40
        elif card.id == C.MURKROW:
            score += 35
        elif card.id == C.WOBBUFFET:
            score += 30
        elif card.id == C.ARTICUNO:
            score += 10  # Repelling Veilはベンチで維持したい
        # 相手アクティブをワンパンできる個体を優先して前に出す（対Garchomp/Marnie=Rocket Rush弱点360等）
        op_active = self.opponent.active[0] if self.opponent.active else None
        if op_active is not None:
            damage, ready = self._pokemon_damage_vs(card, op_active)
            if damage >= op_active.hp:
                near_ready = ready or (
                    card.id == C.SPIDOPS and self.discard_counts[C.G_ENERGY] > 0
                )  # Charging Up+手貼りで即再武装できる
                score += 200 + (150 if ready else (100 if near_ready else 0))
        # 逆に相手にワンパンされる個体は前に出さない（特に弱点2倍もち2サイドのMewtwo。対Marnie教訓。
        # 1サイドのSpidops/フォダーは軽いペナルティに留める=交換要員として差し出すのは本来のライン）
        if self._opp_max_hit_on(card) >= card.hp:
            score -= 50 + 250 * (prize_count(card) - 1)
        if option.index == plan.attacker - 1:
            score += 200
        return score

    def _score_gust_target(self, card: Pokemon) -> float:
        est = self._est_next_damage(card)
        score = 100.0
        score += (card.maxHp - card.hp) / 2
        if est > 0 and card.hp <= est:
            score += 400 + 300 * prize_count(card)
        score -= 30 * energy_units(card)
        score -= card.hp / 10
        return score

    def _score_to_hand(self, card: Pokemon | Card) -> float:
        cid = card.id
        effect_id = self.select.effect.id if self.select.effect is not None else 0

        if effect_id == C.TRANSCEIVER or effect_id == C.MURKROW:
            hand_size = len(self.me.hand)
            if cid == C.ARIANA:
                return 300 if hand_size <= 3 else 120
            if cid == C.PROTON:
                return 280 if self.board_count < 5 and self.state.turn <= 6 else 100
            if cid == C.GIOVANNI:
                return 150
            if cid == C.ARCHER:
                return 130
            if cid == C.LILLIE:
                return 110 if hand_size <= 3 else 60
            if cid == C.PETREL:
                return 50
            return 40
        if effect_id == C.POKE_PAD:
            line_need = self.field_counts[C.TAROUNTULA] > self.field_counts[C.SPIDOPS] + self.hand_counts[C.SPIDOPS]
            return {
                C.TAROUNTULA: 260,
                C.SPIDOPS: 300 if line_need else 90,
                C.ARTICUNO: 280 if not self.articuno_in_play else 120,
                C.MURKROW: 100,
                C.WOBBUFFET: 80,
            }.get(cid, 50)
        if effect_id == C.PROTON:
            return {
                C.TAROUNTULA: 260,
                C.MEWTWO: 280 if self.field_counts[C.MEWTWO] + self.hand_counts[C.MEWTWO] < 2 else 100,
                C.ARTICUNO: 270 if not self.articuno_in_play else 110,
                C.MURKROW: 90,
                C.WOBBUFFET: 70,
            }.get(cid, 40)
        if effect_id == C.BUG_CATCHING:
            spidops_need = self.field_counts[C.TAROUNTULA] > 0 and self.hand_counts[C.SPIDOPS] == 0
            return {
                C.SPIDOPS: 300 if spidops_need else 120,
                C.G_ENERGY: 250,
                C.TAROUNTULA: 110,
            }.get(cid, 30)
        if effect_id == C.ENERGY_SEARCH:
            mewtwo_needs_p = any(
                p is not None and p.id == C.MEWTWO and p_units(p) < 2 for p in self._my_board()
            )
            if cid == C.P_ENERGY:
                return 300 if mewtwo_needs_p else 150
            if cid == C.G_ENERGY:
                return 200
            return 20
        if effect_id == C.NIGHT_STRETCHER:
            return {
                C.MEWTWO: 320 if self.field_counts[C.MEWTWO] + self.hand_counts[C.MEWTWO] == 0 else 60,
                C.ARTICUNO: 310 if not self.articuno_in_play else 50,
                C.TAROUNTULA: 260,
                C.G_ENERGY: 240 if self.hand_basic_energy == 0 else 150,
                C.P_ENERGY: 140,
                C.SPIDOPS: 120,
            }.get(cid, 40)
        if effect_id == C.PETREL:
            return {
                C.HEROS_CAPE: 300,
                C.POKE_PAD: 250,
                C.ENERGY_SEARCH: 220,
                C.NIGHT_STRETCHER: 210,
                C.TRANSCEIVER: 180,
                C.GIOVANNI: 150,
                C.FACTORY: 120,
            }.get(cid, 100)

        # 汎用（サイド取得等）
        score = 200 - self.hand_counts[cid] * 40
        if cid in (C.MEWTWO, C.TR_ENERGY):
            score += 80
        return score

    def _score_discard(self, card: Pokemon | Card) -> float:
        cid = card.id
        score = 0.0
        score += self.hand_counts[cid] * 30  # 重複は先に切る
        base = {
            C.G_ENERGY: 90,  # Charging Upで拾えるのでむしろ得
            C.ARIANA: 85,
            C.BRAVE_BANGLE: 75,
            C.GIOVANNI: 70,
            C.TAROUNTULA: 65,
            C.POKE_PAD: 60,
            C.TRANSCEIVER: 55,
            C.BUG_CATCHING: 50,
            C.FACTORY: 45,
            C.PROTON: 35,
            C.PETREL: 30,
            C.ARCHER: 30,
            C.ENERGY_SEARCH: 25,
            C.WOBBUFFET: 20,
            C.MURKROW: 10,
            C.NIGHT_STRETCHER: -20,
            C.P_ENERGY: -30,
            C.LILLIE: -40,
            C.TR_ENERGY: -50,
            C.HEROS_CAPE: -70,
        }.get(cid, 0)
        if cid == C.SPIDOPS:
            covered = self.field_counts[C.SPIDOPS] + self.hand_counts[C.SPIDOPS] - 1
            base = 40 if covered >= 1 else -20
        elif cid == C.MEWTWO:
            base = 60 if self.field_counts[C.MEWTWO] >= 2 else -60
        elif cid == C.ARTICUNO:
            base = 55 if self.articuno_in_play else -50
        return score + base

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

    return SpidopsPolicy(obs).choose()
