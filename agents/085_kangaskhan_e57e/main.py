"""085_kangaskhan_e57e — Mega Kangaskhan ex / Crustle wall (sig e57eb95734, 07-22 ladder n=440 WR46.5%).

047_kangaskhan_replica (sig b7cef02500) をベースに、07-22現行トップ個体 e57eb95734 の
全440戦リプレイの行動頻度集計 (EXP-037の軽量逆設計) で差分を焼き込んだレプリカ (EXP-085)。

b7cefとのデッキ差分は60枚中1枚: Shaymin 1 → Hand Trimmer 2枚目 (Flower Curtain喪失)。

方策差分 (440戦の頻度集計、aggregate_085.py):
- セットアップアクティブ Kangaskhan 270 > Dwebble 170 (047と逆転)、ベンチ Dwebble 135 > Kang 62
- Hand Trimmer は相手手札6枚から発火 (315回。自分手札7-8でも撃つ=自己コスト許容)
- Kangaskhan手出し 334回: 盤面0体162 / 1体136 / 2体でも32回 (最大3体)。
  対rulebox の2体目以降はレイト (turn8+ が 187/236)、非rb は序盤から
- 攻撃比 rb Crustle 947 : Kang 466 / nonrb 190:108 (047と同じ ≈64:36)
- Boss 256回: キル用に加え、攻撃不能ターンの壁ストール (Morgrem等を釣り出して手番を潰す)
- 手札破壊への切り順: Hilda/Kangaskhan余剰/Poffin/Crustle余剰 > Switch/Cage
  (047の「IceCream即切り・Kang/Crustle温存」と異なり、余剰ポケモンから切る)
- Pokégear取り先: Lillie 281 > Xerosic 220 > Hilda 194 > Boss 96 (XerosicとHildaが047と逆)
- Cape先 Kangaskhan 106 > Crustle 96 > Dwebble 62 / リトリート6回のみ / Run Errand毎ターン

対面別モード (440戦の対面別ATK/gm集計から):
- kang_mode (Alakazam/Spidops/Lopunny系検出): Kangaskhan主砲 (実測 vs Alakazam Kang 2.01 : Crustle 0.42)
- wall_pref (それ以外のrulebox相手 = Marnie/Garchomp/ミラー等): Crustle壁主砲
  (vs Marnie 3.69:0.66 / vs Garchomp 3.59:0.10)。壁Crustleにはエネ最大6枚スタック
  (Spiky反射+Grow HP、ladder ep 87365232) + Ice Cream回復 + 消耗壁のSwitchローテ
- opp_oneshot_deck (Garchomp/Dragapult検出): Kangaskhan手出し封印
  (kangPlay/gm: vs Garchomp 0.34 / vs Dragapult 0.12。Boss+高打点の3プライズ餌にしない)
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
    MIST = 11
    SPIKY = 14
    GROW = 18

    SHAYMIN = 343
    DWEBBLE = 344
    CRUSTLE = 345
    KANGASKHAN = 756

    POFFIN = 1086
    HAND_TRIMMER = 1087
    POKEGEAR = 1122
    SWITCH = 1123
    ICECREAM = 1147
    CAPE = 1159
    BOSS = 1182
    XEROSIC = 1197
    HILDA = 1225
    LILLIE = 1227
    CAGE = 1264

    LILLIES_PEARL = 1172
    LEGACY_ENERGY = 12


class A:
    ASCENSION = 478
    SUPERB_SCISSORS = 479
    RAPID_FIRE_COMBO = 1092


ENERGY_IDS = {C.G_ENERGY, C.MIST, C.SPIKY, C.GROW}
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


def has_grass(pokemon: Pokemon) -> bool:
    return EnergyType.GRASS in pokemon.energies or EnergyType.RAINBOW in pokemon.energies


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


class KangaskhanPolicy:
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

        self.hand_energy = sum(self.hand_counts[i] for i in ENERGY_IDS)
        self.opp_rulebox = self._opponent_has_rulebox()
        self.opp_oneshot_deck = self._opponent_oneshot_deck()
        self.kang_mode = self._kangaskhan_mode()
        # 壁優先はルールボックス相手のうち高打点系のみ (kang_mode相手はKangaskhan主砲)
        self.wall_pref = self.opp_rulebox and not self.kang_mode
        self.wall_index = self._find_wall_index()

    def choose(self) -> list[int]:
        if not self.select.option or self.select.maxCount == 0:
            return []
        if self.context == SelectContext.MAIN:
            self._plan_attack()
        scores = [self._score_option(option) for option in self.select.option]
        ranked = [i for i, _ in sorted(enumerate(scores), key=lambda item: item[1], reverse=True)]
        if self.context == SelectContext.SETUP_BENCH_POKEMON:
            # ベンチは最小限 (本番個体もセットアップベンチは1体が81/105): スコア正のみ、Kangaskhanは最大1体
            picked = []
            kang = 0
            for i in ranked:
                if len(picked) >= self.select.maxCount or scores[i] < 0:
                    break
                card = get_card(self.obs, self.select.option[i].area, self.select.option[i].index, self.my_index)
                if card is not None and card.id == C.KANGASKHAN:
                    if kang >= 1:
                        continue
                    kang += 1
                picked.append(i)
            while len(picked) < self.select.minCount:
                for i in ranked:
                    if i not in picked:
                        picked.append(i)
                        break
            return picked
        return ranked[: self.select.maxCount]

    def _find_wall_index(self) -> int:
        """相手がexデッキのとき、壁 (Crustle) に差し替えるべきベンチindex (board index)"""
        if not self.wall_pref:
            return -1
        active = self.me.active[0] if self.me.active else None
        if active is not None and active.id == C.CRUSTLE:
            return -1
        best, best_score = -1, -1
        for i, pokemon in enumerate(self.me.bench):
            if pokemon is None or pokemon.id != C.CRUSTLE:
                continue
            s = len(pokemon.energies) * 10 + pokemon.hp // 10
            if s > best_score:
                best, best_score = i + 1, s
        return best

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
                if card is not None and card.id == C.BOSS:
                    self.can_gust = True
                if card is not None and card.id == C.SWITCH:
                    self.can_switch = True
            elif option.type == OptionType.RETREAT:
                self.can_switch = True
            elif option.type == OptionType.ATTACK:
                self.can_attack = True

    def _my_board(self) -> list[Pokemon | None]:
        return self.me.active + self.me.bench

    def _opponent_board(self) -> list[Pokemon | None]:
        return self.opponent.active + self.opponent.bench

    def _opponent_oneshot_deck(self) -> bool:
        """Kangaskhan(300)を実質ワンショット/Boss狩りできるデッキ (Garchomp=Draconic Buster 260+、
        Dragapult) の検出。e57eはこれらの対面でKangaskhan手出しをほぼ封印する
        (kangPlay/gm: Garchomp 0.34 / Dragapult 0.12 vs Marnie 0.78 / Alakazam 0.72)"""
        for pokemon in self._opponent_board():
            if pokemon is None:
                continue
            name = card_table[pokemon.id].name
            if name.startswith("Cynthia") or "Dragapult" in name or "Drakloak" in name or "Dreepy" in name:
                return True
        for card in self.opponent.discard:
            data = card_table.get(card.id)
            if data is not None and (
                data.name.startswith("Cynthia") or "Dragapult" in data.name
                or "Drakloak" in data.name or "Dreepy" in data.name
            ):
                return True
        return False

    def _kangaskhan_mode(self) -> bool:
        """e57eの攻撃比は相手アーキタイプ依存 (ATK/gm, 440戦集計):
        vs Alakazam: Kang 2.01 / Crustle 0.42、vs Spidops: Kang 2.94 — Kangaskhan主砲
        vs Marnie: Crustle 3.69 / Kang 0.66、vs Garchomp: Crustle 3.59 / Kang 0.10 — 壁主砲。
        Alakazam/Spidops/Lopunny系 (低瞬間打点・特性ダメージ系) を検出したらKangaskhan主砲モード"""
        kang_markers = ("Alakazam", "Abra", "Kadabra", "Spidops", "Lopunny", "Buneary")
        for pokemon in self._opponent_board():
            if pokemon is None:
                continue
            name = card_table[pokemon.id].name
            if any(m in name for m in kang_markers):
                return True
        for card in self.opponent.discard:
            data = card_table.get(card.id)
            if data is not None and any(m in data.name for m in kang_markers):
                return True
        return False

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

    def _crustle_line_missing(self) -> int:
        """まだデッキから引き出せるCrustleの枚数の概算 (ライン4-4だが盤面目標は3)"""
        used = self.field_counts[C.CRUSTLE] + self.hand_counts[C.CRUSTLE] + self.discard_counts[C.CRUSTLE]
        return max(0, 3 - used)

    # ---- 攻撃プラン ----

    def _attack_candidates(self, pokemon: Pokemon) -> list[tuple[int, int, int, bool, int]]:
        """(attack_id, energy_required, base_damage, needs_grass, base_score)"""
        result = []
        if pokemon.id == C.CRUSTLE:
            # e57e もCrustle主砲だがKangaskhanの攻撃参加が047より多い (rb 947:466 / nonrb 190:108)
            bonus = 700 if self.wall_pref else 250
            result.append((A.SUPERB_SCISSORS, 3, 120, True, bonus))
        elif pokemon.id == C.KANGASKHAN:
            bonus = 700 if self.kang_mode else (450 if self.opp_rulebox else 300)
            result.append((A.RAPID_FIRE_COMBO, 3, 200, False, bonus))
        elif pokemon.id == C.DWEBBLE:
            # Ascension: デッキからCrustleへ即進化 (攻撃扱い)。他に攻撃手がないとき用
            if self._crustle_line_missing() > 0 and self.hand_counts[C.CRUSTLE] == 0:
                result.append((A.ASCENSION, 1, 0, False, -600))
        return result

    def _attach_budget(self) -> int:
        if not self.state.energyAttached and self.hand_energy > 0:
            return 1
        return 0

    def _grass_available(self, pokemon: Pokemon) -> bool:
        """Superb ScissorsのGコストを満たせる見込みがあるか"""
        if has_grass(pokemon):
            return True
        return self.hand_counts[C.GROW] + self.hand_counts[C.G_ENERGY] > 0

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
            for attack_id, energy_required, base_damage, needs_grass, base_score in self._attack_candidates(my_pokemon):
                energy_count = len(my_pokemon.energies)
                needs_energy = False
                if needs_grass and not self._grass_available(my_pokemon):
                    continue
                if energy_count < energy_required:
                    if energy_count + budget >= energy_required:
                        needs_energy = True
                    else:
                        continue
                elif needs_grass and not has_grass(my_pokemon):
                    # エネ枚数は足りるがGが無い→手貼りでGを足せる場合のみ
                    if budget > 0 and self.hand_counts[C.GROW] + self.hand_counts[C.G_ENERGY] > 0:
                        needs_energy = True
                    else:
                        continue

                my_data = card_table[my_pokemon.id]
                for target_index, op_pokemon in enumerate(self._opponent_board()):
                    if op_pokemon is None:
                        continue
                    if target_index != 0 and not self.can_gust:
                        break

                    damage = base_damage
                    if damage > 0:
                        op_data = card_table[op_pokemon.id]
                        if op_data.weakness == my_data.energyType:
                            damage *= 2
                        elif op_data.resistance == my_data.energyType:
                            damage = max(0, damage - 30)

                    score = target_score(op_pokemon)
                    prize = prize_count(op_pokemon) if op_pokemon.hp <= damage else 0
                    if prize == 0 and damage > 0:
                        score *= damage / max(1, op_pokemon.hp)
                    elif damage == 0:
                        score = 80  # Ascension: 盤面価値でなく進化目的
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
        energy_count = len(pokemon.energies)
        score = 8000 + (10 if active else 0)

        # 相手exデッキ: Crustle系をやや優先するが、本番個体はKangaskhanにも普通に貼る
        # (rb時の貼り先 Crustle系466 : Kangaskhan 311)
        crustle_first = self.wall_pref and (self.field_counts[C.CRUSTLE] + self.field_counts[C.DWEBBLE]) > 0

        if energy_id in (C.GROW, C.G_ENERGY):
            if pokemon.id == C.CRUSTLE:
                # Gコスト充足 + Grow GrassはHP+20/枚のスタックも価値
                score += 900 if not has_grass(pokemon) else (350 if energy_id == C.GROW else -300)
                score += 300 if crustle_first and energy_count < 3 else 0
                cap = 6 if self.wall_pref else 5
                if energy_count >= cap:
                    score -= 700
            elif pokemon.id == C.DWEBBLE:
                score += 500 if energy_count < 3 else -500
                score += 250 if crustle_first and self.field_counts[C.CRUSTLE] == 0 else 0
            elif pokemon.id == C.KANGASKHAN:
                score += (250 if energy_count < 3 else -600) if not crustle_first else (80 if energy_count < 3 else -600)
            else:
                score -= 400  # Shaymin等には貼らない
            return score

        # Mist / Spiky ({C}扱い)
        if pokemon.id == C.KANGASKHAN:
            base = 500 if energy_id == C.MIST else 450
            if crustle_first:
                base = 300
            score += base if energy_count < 3 else -400
        elif pokemon.id == C.CRUSTLE:
            base = 400 if energy_id == C.SPIKY else 380
            # Gがまだ無いCrustleに{C}を積みすぎない (Gスロットを最後に残さない)
            if crustle_first:
                base = 700 if (has_grass(pokemon) or energy_count < 2) else 100
            if self.wall_pref and has_grass(pokemon):
                # 実物は壁Crustleにエネを最大6枚積む (Spiky反射20×枚数+Grow HPが削りエンジン、
                # ladder ep 87365232: E:COCOGRCOCOCO)。Spiky優先で積み増し
                score += (base + (150 if energy_id == C.SPIKY else 0)) if energy_count < 6 else -450
            else:
                score += base if energy_count < 3 else -450
        elif pokemon.id == C.DWEBBLE:
            score += (400 if crustle_first and energy_count < 2 else 150) if energy_count < 3 else -600
        else:
            score -= 400
        return score

    def _board_index(self, area: AreaType, index: int) -> int:
        return index if area == AreaType.ACTIVE else index + 1

    def _score_attach(self, option) -> float:
        card = get_card(self.obs, option.area, option.index, self.my_index)
        pokemon = get_card(self.obs, option.inPlayArea, option.inPlayIndex, self.my_index)
        if card is None or not isinstance(pokemon, Pokemon):
            return 0

        if card.id == C.CAPE:
            # Cape貼り先: Kangaskhan 106 > Crustle 96 > Dwebble 62 (Dwebbleにも普通に貼る)
            score = 6000
            if pokemon.id == C.KANGASKHAN:
                score += 300
            elif pokemon.id == C.CRUSTLE:
                score += 250
            elif pokemon.id == C.DWEBBLE:
                score += 150
            if pokemon.tools:
                score = -1
            return score

        score = self._energy_target_score(pokemon, option.inPlayArea == AreaType.ACTIVE, card.id)
        board_index = self._board_index(option.inPlayArea, option.inPlayIndex)
        if board_index == plan.attacker and plan.needs_energy:
            score += 2000
            # Crustleの不足がGのとき、G供給エネを最優先
            attacker = self._my_board()[plan.attacker] if plan.attacker < len(self._my_board()) else None
            if (
                attacker is not None
                and attacker.id == C.CRUSTLE
                and not has_grass(attacker)
                and card.id in (C.GROW, C.G_ENERGY)
            ):
                score += 800
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
            # 逃げるはエネをコスト分トラッシュするので攻撃プランがある時のみ
            # (壁への退避はSwitchアイテムで行う。本番個体もリトリートは240戦で3回)
            return 2000 if plan.attacker >= 1 else -1
        if option.type == OptionType.ATTACK:
            return 1100 if option.attackId == plan.attack_id else 1000
        return 0

    def _score_ability(self, option) -> float:
        card = get_card(self.obs, option.area, option.index, self.my_index)
        if card is None:
            return 0
        if card.id == C.KANGASKHAN:  # Run Errand: ドロー2 (707回=毎ターン発火)
            if self.me.deckCount <= 2:
                return -1
            return 30000
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
        board = sum(1 for p in self._my_board() if p is not None)
        if card.id == C.KANGASKHAN:
            # e57e: 334回/440戦=0.76/gm。盤面0体162 / 1体136 / 2体32回。
            # 対ruleboxの2体目以降はレイト (turn8+が187/236)、非rbは序盤から。
            # 失った分の補充を含む総投入数はセットアップ込みで約1.5体/戦 → discard込みでカウントし
            # 「死んだら即補充」の給餌ループを作らない
            commits = self.field_counts[C.KANGASKHAN] + self.discard_counts[C.KANGASKHAN]
            if self.opp_oneshot_deck:
                # Garchomp/Dragapult対面はKangaskhanがBoss+高打点の3プライズ餌になるため封印
                # (盤面にアタッカーが全くいない時のみ許可)
                attackers = (
                    self.field_counts[C.KANGASKHAN]
                    + self.field_counts[C.CRUSTLE]
                    + self.field_counts[C.DWEBBLE]
                )
                return 12000 if attackers == 0 else -1
            if commits < 1:
                return 20000
            if commits < 2 and (self.state.turn >= 7 or not self.opp_rulebox):
                return 12000
            if commits < 3 and self.state.turn >= 10 and board < 5:
                return 11500
            return -1
        if card.id == C.DWEBBLE:
            # 壁の替えを切らさない (実物はライン4-4をほぼ使い切る。no_active負け防止)
            line = self.field_counts[C.DWEBBLE] + self.field_counts[C.CRUSTLE]
            if line < 4 and board < 5:
                return 19500
            return -1
        return 13000

    def _score_play_trainer(self, card: Card) -> float:
        cid = card.id
        # ---- アイテム (サポートより先に切る) ----
        if cid == C.POFFIN:
            # 実物のPoffinは1.44-1.71回/戦 (対Garchompでも1.49)。デッキに残るDwebbleを出し切って
            # 壁の替えを確保する (山に残る枚数 = 4 - 見えている枚数)
            dwebble_left = 4 - self.field_counts[C.DWEBBLE] - self.field_counts[C.CRUSTLE] - self.hand_counts[C.DWEBBLE] - self.discard_counts[C.DWEBBLE] - self.discard_counts[C.CRUSTLE]
            board = sum(1 for p in self._my_board() if p is not None)
            return 10000 if dwebble_left > 0 and board < 5 else -1
        if cid == C.POKEGEAR:
            return 9800
        if cid == C.SWITCH:
            if plan.attacker >= 1:
                return 9500
            active = self.me.active[0] if self.me.active else None
            if (
                plan.attacker == -1
                and self.wall_index >= 1
                and active is not None
                and active.id == C.KANGASKHAN
            ):
                return 9300  # 攻撃できないKangaskhanを壁Crustleへ退避
            if (
                self.wall_pref
                and active is not None
                and active.id == C.CRUSTLE
                and active.hp <= 60
            ):
                # 消耗した壁を新しいCrustleへローテ (実物Switch 0.88-1.13回/戦)
                for pokemon in self.me.bench:
                    if pokemon is not None and pokemon.id == C.CRUSTLE and pokemon.hp >= active.hp + 60:
                        return 9250
            return -1
        if cid == C.ICECREAM:
            # 実物は0.8-2.1回/戦 (壁対面ほど多い)。エネ3枚条件を緩和し壁は常時回復
            active = self.me.active[0] if self.me.active else None
            if active is not None:
                dmg = active.maxHp - active.hp
                if dmg >= 80:
                    return 9600
                if dmg >= 40 and (len(active.energies) >= 2 or active.id == C.CRUSTLE):
                    return 9200
            return -1
        if cid == C.HAND_TRIMMER:
            # e57e: 相手手札6枚から発火 (opphand 6:101 / 7:56 / 8:29)。自分手札7-8でも撃つ
            return 9400 if self.opponent.handCount >= 6 else -1
        # ---- スタジアム ----
        if cid == C.CAGE:
            # Battle Cage: ベンチへのダメカン配置を両者防止 (Powerful Hand対策の核)
            if self.stadium_id == 0 or not self.stadium_is_mine:
                return 2500
            return -1
        # ---- サポート ----
        if cid == C.BOSS:
            if plan.target >= 1:
                return 3200
            # e57e: 攻撃不能ターンにCrustle壁の裏からBossで低脅威を釣り出して手番を潰す
            # (ep 87362960 t10: エネ2枚で攻撃不能→Boss→Morgrem釣り出し→END)
            active = self.me.active[0] if self.me.active else None
            if (
                self.opp_rulebox
                and plan.attacker == -1
                and active is not None
                and active.id == C.CRUSTLE
                and len(self.opponent.bench) > 0
            ):
                return 2400
            return -1
        if cid == C.LILLIE:
            hand_size = len(self.me.hand)
            if self.me.deckCount <= max(2, hand_size - 2):
                return -1
            if hand_size <= 4:
                return 3120
            if hand_size <= 6:
                return 2900
            return 2600  # 本番個体は手札7-8でも123回プレイ (他サポートより低優先)
        if cid == C.XEROSIC:
            # 相手手札を3枚に。本番個体は手札4枚から発火 (107回)、5枚以上は最頻サポート
            if self.opponent.handCount >= 5:
                return 3080
            if self.opponent.handCount >= 4:
                return 2950
            return -1
        if cid == C.HILDA:
            need_crustle = self._crustle_line_missing() > 0 and self.hand_counts[C.CRUSTLE] == 0
            if need_crustle:
                return 3100
            if self.hand_energy == 0:
                return 3050
            return 2500
        return 2000

    # ---- カード選択（サーチ・入替等） ----

    def _score_card_choice(self, option) -> float:
        card = get_card(self.obs, option.area, option.index, option.playerIndex)
        if card is None:
            return 0
        if self.context in {SelectContext.SWITCH, SelectContext.TO_ACTIVE}:
            return self._score_active_choice(option, card)
        if self.context == SelectContext.SETUP_ACTIVE_POKEMON:
            # e57e: Kangaskhan 270 > Dwebble 170 (047と逆転)
            return {C.KANGASKHAN: 5, C.DWEBBLE: 4}.get(card.id, 0)
        if self.context == SelectContext.SETUP_BENCH_POKEMON:
            # e57e: Dwebble 135 > Kangaskhan 62 (アクティブがKangaskhanの補完)
            return {C.DWEBBLE: 5, C.KANGASKHAN: 4}.get(card.id, 0)
        if self.context in {SelectContext.TO_BENCH, SelectContext.TO_FIELD}:
            return {C.DWEBBLE: 10, C.KANGASKHAN: 8}.get(card.id, 1)
        if self.context == SelectContext.TO_HAND:
            return self._score_to_hand(card)
        if self.context == SelectContext.ATTACH_FROM:
            return self._score_attach_from(option, card)
        if self.context in {SelectContext.DISCARD, SelectContext.TO_DECK, SelectContext.TO_DECK_BOTTOM}:
            if option.playerIndex != self.my_index:
                return self._score_discard_opp(card)
            return self._score_discard(card)
        return 0

    def _score_active_choice(self, option, card: Pokemon | Card) -> float:
        if not isinstance(card, Pokemon):
            return 0
        if option.playerIndex != self.my_index:
            # Boss: プランのキル対象を最優先、無ければ低HP高価値
            if option.index == plan.target - 1:
                return 100
            return (300 - card.hp) / 100 + prize_count(card)

        score = len(card.energies) * 3
        if option.index == plan.attacker - 1:
            score += 200
        if self.wall_pref:
            # 高打点ex相手: Crustle壁優先だがKangaskhanのアクティブ入りも047より多い
            # (e57e to_active: Crustle 609 > Kangaskhan 393 > Dwebble 171)
            if card.id == C.CRUSTLE:
                score += 40
            elif card.id == C.KANGASKHAN:
                score += 12
            elif card.id == C.DWEBBLE:
                score += 10
        else:
            # 非rb または kang_mode (Alakazam/Spidops系): Kangaskhanをアクティブ主砲に
            if card.id == C.KANGASKHAN:
                score += 40
            elif card.id == C.CRUSTLE:
                score += 15
            elif card.id == C.DWEBBLE:
                score += 10
        return score

    def _score_attach_from(self, option, card: Pokemon | Card) -> float:
        if not isinstance(card, Pokemon):
            return 0
        if option.playerIndex != self.my_index:
            return 5000 - target_score(card)
        energy_id = self.select.effect.id if self.select.effect is not None else C.MIST
        score = self._energy_target_score(card, option.area == AreaType.ACTIVE, energy_id)
        board_index = self._board_index(option.area, option.index)
        if board_index == plan.attacker and plan.needs_energy:
            score += 2000
        return score

    def _score_to_hand(self, card: Pokemon | Card) -> float:
        cid = card.id
        effect_id = self.select.effect.id if self.select.effect is not None else None
        score = 200 - self.hand_counts[cid] * 60

        if effect_id == C.HILDA:
            # 進化ポケモン1+エネ1: Crustle 244 / Grow 104 > Mist 95 > G 69 > Spiky 9
            if cid == C.CRUSTLE:
                return score + (500 if self.field_counts[C.DWEBBLE] > 0 else 300)
            if cid == C.GROW:
                return score + 240
            if cid == C.MIST:
                return score + 200
            if cid == C.G_ENERGY:
                return score + 180
            if cid == C.SPIKY:
                return score + 120
            return score
        if effect_id == C.POKEGEAR:
            # e57e の取り先: Lillie 281 > Xerosic 220 > Hilda 194 > Boss 96 (XerosicとHildaが047と逆)
            if cid == C.LILLIE:
                return score + (300 if len(self.me.hand) <= 5 else 190)
            if cid == C.XEROSIC:
                return score + (230 if self.opponent.handCount >= 5 else 140)
            if cid == C.HILDA:
                return score + (280 if self._crustle_line_missing() > 0 and self.hand_counts[C.CRUSTLE] == 0 else 170)
            if cid == C.BOSS:
                return score + (250 if self.can_attack else 100)
            return score

        # 汎用フォールバック
        if cid == C.KANGASKHAN:
            score += 90 if self.field_counts[C.KANGASKHAN] < 2 else -60
        elif cid == C.CRUSTLE:
            score += 90 if self.field_counts[C.DWEBBLE] > 0 else 20
        elif cid == C.DWEBBLE:
            score += 60 if self.field_counts[C.DWEBBLE] + self.field_counts[C.CRUSTLE] < 2 else -40
        elif cid in ENERGY_IDS:
            score += 70 - 30 * min(2, self.hand_energy)
        elif cid == C.BOSS:
            score += 50
        return score

    def _score_discard(self, card: Pokemon | Card) -> float:
        """自分の手札から捨てる: 高スコア=先に捨てる。
        e57eが相手の手札破壊 (Xerosic/Hand Trimmer) で切った順 (aggregate_085):
        Hilda 124 > Kangaskhan 122 > Poffin 114 > Crustle 106 > Switch 82 > Cage 59
        (047の「IceCream即切り・Kang/Crustle温存」と異なり余剰ポケモン/サーチ札から切る。
        IceCream 21 / Lillie 35 は温存気味)"""
        cid = card.id
        score = 0
        if self.hand_counts[cid] >= 2:
            score += 80
        if cid == C.HILDA:
            score += 55
        elif cid == C.KANGASKHAN:
            score += 50 if self.field_counts[C.KANGASKHAN] >= 1 else -20
        elif cid == C.POFFIN:
            score += 50
        elif cid == C.CRUSTLE:
            score += 45 if self.field_counts[C.CRUSTLE] >= 2 else -40
        elif cid == C.SWITCH:
            score += 35
        elif cid == C.CAGE:
            score += 30
        elif cid == C.POKEGEAR:
            score += 25
        elif cid == C.HAND_TRIMMER:
            score += 25
        elif cid == C.XEROSIC:
            score += 20
        elif cid in ENERGY_IDS:
            score += 30 if self.hand_energy >= 3 else -30
            if cid in (C.GROW, C.G_ENERGY):
                score -= 15  # Gコスト用に温存
        elif cid == C.DWEBBLE:
            score += 20 if self.field_counts[C.DWEBBLE] + self.field_counts[C.CRUSTLE] >= 3 else -20
        elif cid == C.BOSS:
            score += 10
        elif cid == C.ICECREAM:
            score -= 10
        elif cid == C.LILLIE:
            score -= 20
        elif cid == C.CAPE:
            score -= 30
        return score

    def _score_discard_opp(self, card: Pokemon | Card) -> float:
        """相手の手札から捨てさせる効果用 (このデッキには無いが汎用フォールバック)"""
        name = card_table[card.id].name if card.id in card_table else ""
        priority = {
            "Rare Candy": 100,
            "Buddy-Buddy Poffin": 95,
            "Night Stretcher": 90,
            "Ultra Ball": 85,
            "Earthen Vessel": 80,
            "Poké Pad": 75,
            "Jumbo Ice Cream": 70,
            "Switch": 60,
        }
        return priority.get(name, 50)

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

    return KangaskhanPolicy(obs).choose()
