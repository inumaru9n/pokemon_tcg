"""093_starmie_full — 091_starmie_ml + 一致率駆動較正ルール (I-138, EXP-093).

091（=036の30ルール+lethal探索+MAIN行動クラスGBT層）のルール背骨に、
052式の一致率駆動較正（教師=Yushin Ito 491b8bfb26 リプレイとの不一致マイニング
→述語ルール焼き込み）を適用。ML層(model_093)とlethal層は091から無変更。
追加ルールは環境変数 F093_OFF=G,H,... で個別OFF（デフォルト全ON）。
モデル無し / F093_ML=off でルール素の挙動。
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
    WATER = 3               # Basic Water Energy
    IGNITION = 17           # Ignition Energy (CCC on evolution, self-discards at end of turn)
    CINDERACE = 666         # 160HP wall, Explosiveness starter, Turbo Flare ramp
    STARYU = 1030
    MEGA_STARMIE = 1031     # 330HP, Jetting Blow 120+50 / Nebula Beam 210
    POFFIN = 1086
    NIGHT_STRETCHER = 1097
    HAMMER = 1120
    ULTRA_BALL = 1121
    POKEGEAR = 1122
    MEGA_SIGNAL = 1145
    CAPE = 1159
    BOSS = 1182
    SALVATORE = 1189
    HARLEQUIN = 1223
    HILDA = 1225
    LILLIE = 1227
    WALLY = 1229


JETTING_BLOW = 1487   # 120 + 50 bench, cost [W]
NEBULA_BEAM = 1488    # 210 ignores weakness/resistance/effects, cost [C,C,C]
TURBO_FLARE = 965     # 50 + attach up to 3 basic energy from deck to bench, free
WATER_GUN = 1486      # 20, cost [W]

# Opposing card groups (R16/R17)
MUNKIDORI = {112, 139}
DUNSPARCE_LINE = {65, 66, 305, 306, 996, 997}   # shuffles itself back — never waste snipes
EVOLUTION_SEEDS = {
    109, 741, 742,        # Abra / Kadabra
    119, 120,             # Dreepy / Drakloak
    379, 380,             # Cynthia's Gible / Gabite
    333, 677, 974,        # Riolu
    344, 532,             # Dwebble
    103, 860,             # Snorunt
    646, 647,             # Marnie's Impidimp / Morgrem
}
MARNIE_IDS = {646, 647, 648}
LOW_DECK = 8

# ---- EXP-093: 一致率駆動較正ルール（052式）。F093_OFF=T,S,H,U,P で個別OFF ----
# RULE_Gのみデフォルト無効（一致率+6.7ptだがミラー2シードで-3pt級=アリーナ優先で不採用。
# 再現用に F093_ON=G で有効化できる）
_F093_OFF = {x.strip().upper() for x in os.environ.get("F093_OFF", "").split(",") if x.strip()}
_F093_ON = {x.strip().upper() for x in os.environ.get("F093_ON", "").split(",") if x.strip()}
_F093_DEFAULT_OFF = {"G"}


def _r093(flag: str) -> bool:
    if flag in _F093_DEFAULT_OFF:
        return flag in _F093_ON and flag not in _F093_OFF
    return flag not in _F093_OFF

try:  # Kaggle eval loads main.py via exec(): __file__ undefined
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
    def __init__(self):
        self.attacker = -1
        self.target = -1
        self.attack_id = -1
        self.needs_energy = False
        self.energy_card = -1   # card id of the energy to attach this turn (WATER/IGNITION)
        self.remain_hp = -1


plan = AttackPlan()
pre_turn = -1


def get_card(obs: Observation, area: AreaType, index: int, player_index: int):
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
    return 3 if data.megaEx else 2 if data.ex else 1


def target_score(pokemon: Pokemon) -> float:
    """Generic value of an opposing Pokemon (islet-style)."""
    data = card_table[pokemon.id]
    score = prize_count(pokemon) * 1000
    score += len(pokemon.energies) * 150
    score += len(pokemon.tools) * 100
    if data.stage2:
        score += 250
    elif data.stage1:
        score += 130
    if pokemon.id in MUNKIDORI:
        score += 400          # R17: Munkidori is the priority removal target
    score += pokemon.hp
    return score


class StarmiePolicy:
    def __init__(self, obs: Observation):
        self.obs = obs
        self.state = obs.current
        self.select = obs.select
        self.context = self.select.context
        self.my_index = self.state.yourIndex
        self.op_index = 1 - self.my_index
        self.me = self.state.players[self.my_index]
        self.op = self.state.players[self.op_index]

        self.field_counts = defaultdict(int)
        self.hand_counts = defaultdict(int)
        self.discard_counts = defaultdict(int)
        for p in self._my_board():
            if p is not None:
                self.field_counts[p.id] += 1
        if self.me.hand:
            for c in self.me.hand:
                self.hand_counts[c.id] += 1
        for c in self.me.discard:
            self.discard_counts[c.id] += 1

        self.deck_megas = max(0, 3 - self.field_counts[C.MEGA_STARMIE]
                              - self.hand_counts[C.MEGA_STARMIE]
                              - self.discard_counts[C.MEGA_STARMIE])
        self.line_on_field = self.field_counts[C.STARYU] + self.field_counts[C.MEGA_STARMIE]
        self.is_marnie = any(
            p is not None and p.id in MARNIE_IDS for p in self._op_board()
        ) or any(c.id in MARNIE_IDS for c in self.op.discard)
        self.op_big_tank = any(
            p is not None and card_table[p.id].hp >= 280 for p in self._op_board()
        )

        self.can_switch = False
        self.can_gust = False
        if self.context == SelectContext.MAIN:
            for o in self.select.option:
                if o.type == OptionType.RETREAT:
                    self.can_switch = True
                elif o.type == OptionType.PLAY:
                    card = get_card(obs, AreaType.HAND, o.index, self.my_index)
                    if card is not None and card.id == C.BOSS:
                        self.can_gust = True

    # ---------- board helpers ----------

    def _my_board(self):
        return list(self.me.active) + list(self.me.bench)

    def _op_board(self):
        return list(self.op.active) + list(self.op.bench)

    def _my_megas(self):
        return [p for p in self._my_board() if p is not None and p.id == C.MEGA_STARMIE]

    def _mega_best_damage(self) -> int:
        best = 0
        for p in self._my_megas():
            best = max(best, p.maxHp - p.hp)
        return best

    def _wally_threshold(self) -> int:
        # R19: heal at ~150 accumulated damage; vs Marnie Munkidori chips add up faster
        return 120 if self.is_marnie else 150

    # ---------- attack planning (R7/R8/R10-R18) ----------

    def _snipe_value(self, exclude_index: int) -> float:
        """Best Jetting Blow bench-snipe value on opponent's bench (R16/R17)."""
        best = 0.0
        for k, p in enumerate(self._op_board()):
            if k == 0 or p is None or k == exclude_index:
                continue
            if p.id in DUNSPARCE_LINE and p.hp > 50:
                continue  # R17: never waste on recyclable Dunsparce
            if p.hp <= 50:
                v = 600 + prize_count(p) * 400   # guaranteed extra prize
            elif p.id in MUNKIDORI:
                v = 350
            elif p.id in EVOLUTION_SEEDS:
                v = 250
            elif p.energies:
                v = 150
            else:
                v = 40
            best = max(best, v)
        return best

    def _attack_candidates(self, mine: Pokemon):
        """Yield (attack_id, feasible, needs_energy, energy_card, damage_base, applies_weakness)."""
        have_any = len(mine.energies)
        have_w = sum(1 for e in mine.energies if e == EnergyType.WATER)
        w_in_hand = self.hand_counts[C.WATER] >= 1
        ign_in_hand = self.hand_counts[C.IGNITION] >= 1
        can_attach = not self.state.energyAttached
        cands = []
        if mine.id == C.MEGA_STARMIE:
            # Jetting Blow: needs a real Water energy (cost [W])
            if have_w >= 1:
                cands.append((JETTING_BLOW, False, -1, 120, True))
            elif can_attach and w_in_hand:
                cands.append((JETTING_BLOW, True, C.WATER, 120, True))
            # Nebula Beam: 3 of anything; Ignition alone provides CCC on an evolution (R13/R15)
            if have_any >= 3:
                cands.append((NEBULA_BEAM, False, -1, 210, False))
            elif can_attach and w_in_hand and have_any >= 2:
                cands.append((NEBULA_BEAM, True, C.WATER, 210, False))
            elif can_attach and ign_in_hand:
                cands.append((NEBULA_BEAM, True, C.IGNITION, 210, False))
        elif mine.id == C.CINDERACE:
            # R7: Turbo Flare costs 1 colorless — hand-attach W (or Ignition) to fire it
            if have_any >= 1:
                cands.append((TURBO_FLARE, False, -1, 50, True))
            elif can_attach and w_in_hand:
                cands.append((TURBO_FLARE, True, C.WATER, 50, True))
            elif can_attach and ign_in_hand:
                cands.append((TURBO_FLARE, True, C.IGNITION, 50, True))
        elif mine.id == C.STARYU:
            if have_w >= 1:
                cands.append((WATER_GUN, False, -1, 20, True))
            elif can_attach and w_in_hand:
                cands.append((WATER_GUN, True, C.WATER, 20, True))
        return cands

    def _plan_attack(self) -> None:
        global plan
        plan = AttackPlan()
        if self.state.turn < 2:
            return  # R6: turn 1 cannot attack
        best_score = -1.0
        my_board = self._my_board()
        op_board = self._op_board()
        my_type = EnergyType.WATER

        for i, mine in enumerate(my_board):
            if mine is None:
                continue
            if i != 0 and not self.can_switch:
                break
            atk_type = card_table[mine.id].energyType
            for attack_id, needs_energy, energy_card, base_damage, applies_weak in \
                    self._attack_candidates(mine):
                for j, opp in enumerate(op_board):
                    if opp is None:
                        continue
                    if j != 0:
                        if not self.can_gust:
                            break
                        # R18: Boss (1 copy) only for a kill or Munkidori removal
                        dmg_chk = base_damage
                        data_chk = card_table[opp.id]
                        if applies_weak:
                            if data_chk.weakness == atk_type:
                                dmg_chk *= 2
                            elif data_chk.resistance == atk_type:
                                dmg_chk = max(0, dmg_chk - 30)
                        if opp.hp > dmg_chk and opp.id not in MUNKIDORI:
                            continue

                    damage = base_damage
                    data = card_table[opp.id]
                    if applies_weak:
                        if data.weakness == atk_type:
                            damage *= 2
                        elif data.resistance == atk_type:
                            damage = max(0, damage - 30)

                    score = target_score(opp)
                    prize = prize_count(opp) if opp.hp <= damage else 0
                    if prize == 0:
                        score *= damage / max(1, opp.hp)
                        if opp.id in DUNSPARCE_LINE:
                            score *= 0.3   # R17: chip on recyclable line is wasted
                    if len(self.op.prize) <= prize:
                        score = 50000      # lethal

                    if attack_id == JETTING_BLOW:
                        score += self._snipe_value(j)   # R16 bench snipe rider
                        score += 80                     # R15: Jetting is the default
                    elif attack_id == TURBO_FLARE:
                        # R7: value of ramping the bench line
                        needs_ramp = any(
                            p is not None and p.id in (C.STARYU, C.MEGA_STARMIE)
                            and len(p.energies) < 2
                            for p in self.me.bench
                        )
                        score += 300 if needs_ramp else -50
                    if needs_energy and energy_card == C.IGNITION:
                        score -= 120       # R13: burning Ignition costs a card

                    score += 220 if i == 0 else 0
                    score += 300 if j == 0 else 0
                    score += len(mine.energies)

                    if score > best_score:
                        best_score = score
                        plan = AttackPlan()
                        plan.attacker = i
                        plan.target = j
                        plan.attack_id = attack_id
                        plan.needs_energy = needs_energy
                        plan.energy_card = energy_card
                        plan.remain_hp = opp.hp - damage

    # ---------- option scoring ----------

    def choose(self) -> list[int]:
        if not self.select.option or self.select.maxCount == 0:
            return []
        if self.context == SelectContext.MAIN:
            self._plan_attack()
        scores = [self._score_option(o) for o in self.select.option]
        ranked = [i for i, _ in sorted(enumerate(scores), key=lambda t: t[1], reverse=True)]
        picked = ranked[: self.select.maxCount]
        # decline optional picks that we scored negative
        while len(picked) > self.select.minCount and scores[picked[-1]] < 0:
            picked.pop()
        # RULE_G (EXP-093): Poffin等のベンチ展開は1体だけ（本人98%=ベンチ規律R28の実データ）
        if _r093("G") and self.context == SelectContext.TO_BENCH and picked:
            picked = picked[: max(self.select.minCount, 1)]
        # RULE_T (EXP-093): Turbo Flareのデッキからのエネ加速はW1枚だけ（本人95%、R11低燃費）
        if _r093("T") and self.context == SelectContext.ATTACH_TO and picked:
            picked = picked[: max(self.select.minCount, 1)]
        return picked

    def _score_option(self, option) -> float:
        t = option.type
        if t == OptionType.NUMBER:
            return option.number  # R5: draw max on opponent mulligan
        if t == OptionType.YES:
            return self._score_yes()
        if t == OptionType.NO:
            return self._score_no()
        if t == OptionType.CARD:
            return self._score_card(option)
        if t == OptionType.PLAY:
            return self._score_play(option)
        if t == OptionType.ATTACH:
            return self._score_attach(option)
        if t == OptionType.EVOLVE:
            return 13000  # R8: evolve to Mega at the first opportunity, always
        if t == OptionType.ABILITY:
            return -1     # R30: only opposing stadium abilities exist for us — skip
        if t == OptionType.RETREAT:
            return 6500 if plan.attacker >= 1 else -1   # R10
        if t == OptionType.ATTACK:
            if option.attackId == plan.attack_id:
                return 2100
            return 1200   # attacking still beats passing
        if t == OptionType.ENERGY_CARD or t == OptionType.ENERGY:
            return self._score_energy_pick(option)
        return 0

    def _score_yes(self) -> float:
        if self.context == SelectContext.IS_FIRST:
            return 100    # R1: always go first
        if self.context == SelectContext.MULLIGAN:
            # R3: keep the Cinderace-only hand iff Poffin is in it (YES = redraw)
            return 0 if self.hand_counts[C.POFFIN] >= 1 else 10
        return 1

    def _score_no(self) -> float:
        if self.context == SelectContext.MULLIGAN:
            return 10 if self.hand_counts[C.POFFIN] >= 1 else 0
        return 0

    # ----- CARD options, dispatched by context -----

    def _score_card(self, option) -> float:
        card = get_card(self.obs, option.area, option.index, option.playerIndex)
        if card is None:
            return 0
        ctx = self.context
        if ctx in (SelectContext.SWITCH, SelectContext.TO_ACTIVE):
            return self._score_active_choice(option, card)
        if ctx == SelectContext.SETUP_ACTIVE_POKEMON:
            # R2: Cinderace wall first, then Staryu
            return 4 if card.id == C.CINDERACE else 2 if card.id == C.STARYU else 1
        if ctx in (SelectContext.SETUP_BENCH_POKEMON, SelectContext.TO_BENCH,
                   SelectContext.TO_FIELD):
            # R4: bench every Staryu at setup / from Poffin
            return 4 if card.id == C.STARYU else -1
        if ctx == SelectContext.TO_HAND:
            return self._score_to_hand(card)
        if ctx in (SelectContext.DAMAGE, SelectContext.DAMAGE_COUNTER,
                   SelectContext.DAMAGE_COUNTER_ANY):
            return self._score_snipe(option, card)
        if ctx in (SelectContext.HEAL, SelectContext.REMOVE_DAMAGE_COUNTER):
            if option.playerIndex == self.my_index and isinstance(card, Pokemon):
                return (card.maxHp - card.hp) * 10   # R19/R20: heal the most damaged Mega
            return -100
        if ctx == SelectContext.EVOLVES_TO:
            return 100 if card.id == C.MEGA_STARMIE else 10
        if ctx == SelectContext.ATTACH_FROM and isinstance(card, Pokemon):
            return self._energy_target_score(card, option.area == AreaType.ACTIVE)
        if ctx in (SelectContext.DISCARD, SelectContext.TO_DECK,
                   SelectContext.TO_DECK_BOTTOM):
            if option.playerIndex == self.my_index and option.area == AreaType.HAND:
                return self._score_discard_from_hand(card)
            return 0
        return 0

    def _score_active_choice(self, option, card) -> float:
        if option.playerIndex != self.my_index:
            # Boss pull target (R17/R18)
            score = 100 if option.index == plan.target - 1 else 0
            if isinstance(card, Pokemon):
                if card.id in MUNKIDORI:
                    score += 60
                if card.hp <= 120:
                    score += 40
            return score
        if not isinstance(card, Pokemon):
            return 0
        score = len(card.energies) * 5
        if option.index == plan.attacker - 1:
            score += 100
        if card.id == C.MEGA_STARMIE:
            score += 30 if card.energies else 20
        elif card.id == C.CINDERACE:
            score += 12   # wall (R2)
        elif card.id == C.STARYU:
            score += 4
        return score

    def _score_snipe(self, option, card) -> float:
        # R16/R17: Jetting Blow bench 50 target
        if option.playerIndex == self.my_index:
            return -100
        if not isinstance(card, Pokemon):
            return 0
        if card.id in DUNSPARCE_LINE and card.hp > 50:
            return -500    # R17: never feed the Dunsparce recycler
        score = 0.0
        if card.hp <= 50:
            score += 5000 + prize_count(card) * 500   # 1. guaranteed KO
        if _r093("S"):
            # RULE_S (EXP-093): 本人のsnipe実頻度は「進化ライン（Grimmsnarlライン含む）>
            # Munkidori」+ 同格なら残HPが低い個体（50×2でKO圏）を優先
            # （対Marnie damage一致24%の主因。studyのR17仮説を実データで訂正）
            if card.id in MARNIE_IDS or card.id in EVOLUTION_SEEDS:
                score += 1000
            elif card.id in MUNKIDORI:
                score += 800
            score += max(0, 100 - card.hp) * 4
        else:
            if card.id in MUNKIDORI:
                score += 1200                          # vs Marnie: Munkidori first
            elif card.id in EVOLUTION_SEEDS:
                score += 800                           # 2. evolution seeds
        score += len(card.energies) * 150              # 3. charged attackers
        score += target_score(card) * 0.1
        return score

    def _score_to_hand(self, card) -> float:
        effect = self.select.effect
        eid = effect.id if effect is not None else -1
        if eid == C.POKEGEAR:
            return self._score_pokegear_pick(card)
        if eid == C.MEGA_SIGNAL:
            return 1000 if card.id == C.MEGA_STARMIE else 0
        if eid == C.HILDA:
            return self._score_hilda_pick(card)
        if eid == C.NIGHT_STRETCHER:
            return self._score_stretcher_pick(card)
        if _r093("U"):
            # RULE_U (EXP-093): 汎用サーチ（Ultra Ball等）の優先順。本人はStaryu補充を
            # Mega本体より優先（ライン未完時）。Cinderaceは手札で死に札
            if card.id == C.STARYU:
                return 230 if self.line_on_field < 2 else 80
            if card.id == C.MEGA_STARMIE:
                return 210 if self.hand_counts[C.MEGA_STARMIE] == 0 else 90
            return {C.WATER: 140, C.WALLY: 110, C.LILLIE: 100, C.HILDA: 95,
                    C.IGNITION: 85, C.CINDERACE: 5}.get(card.id, 50)
        # generic (prize take etc.) — any is fine (R29)
        return 50

    def _score_pokegear_pick(self, card) -> float:
        # R23: Wally when hurt / vs Marnie > Boss for the kill > draw supporters
        cid = card.id
        if cid == C.WALLY:
            hurt = self._mega_best_damage() >= 60
            if (hurt or self.is_marnie) and self.hand_counts[C.WALLY] == 0:
                return 400
            return 120
        if cid == C.BOSS:
            good_target = any(
                p is not None and (p.hp <= 50 or p.id in MUNKIDORI)
                for p in self.op.bench
            )
            if good_target and self.hand_counts[C.BOSS] == 0:
                return 380
            return 100
        if cid == C.SALVATORE:
            if not _r093("P"):
                if self.field_counts[C.STARYU] >= 1 and self.deck_megas > 0:
                    return 300
                return 90
            # RULE_P (EXP-093): 本人はPokegearでSalvatoreよりHarlequin/Lillie/Wallyを
            # 取る頻度が高い（x41/x36/x29不一致）。当ターン進化が確定する時だけ最優先
            if (self.field_counts[C.STARYU] >= 1 and self.deck_megas > 0
                    and self.hand_counts[C.MEGA_SIGNAL] + self.hand_counts[C.HILDA] >= 1):
                return 300
            if self.field_counts[C.STARYU] >= 1 and self.deck_megas > 0:
                return 220
            return 70
        if cid == C.LILLIE:
            return 250 if self.me.handCount <= 5 else 110
        if cid == C.HILDA:
            return 240 if self.hand_counts[C.MEGA_STARMIE] == 0 else 105
        if cid == C.HARLEQUIN:
            return 80
        return 60

    def _score_hilda_pick(self, card) -> float:
        # R24: Mega Starmie first; energy matching the turn's attack
        cid = card.id
        data = card_table.get(cid)
        if cid == C.MEGA_STARMIE:
            return 500
        if data is not None and data.cardType == CardType.POKEMON:
            return 60
        if _r093("H"):
            # RULE_H (EXP-093): 本人はHildaのエネ枠でIgnitionを広く優先
            # （対タンク=常時、それ以外もW手持ち時。x237不一致の是正）
            if cid == C.WATER:
                return 300 if self.hand_counts[C.WATER] == 0 else 120
            if cid == C.IGNITION:
                return 320 if self.op_big_tank else 200
        else:
            if cid == C.WATER:
                return 300 if self.hand_counts[C.WATER] == 0 else 150
            if cid == C.IGNITION:
                return 180 if self.op_big_tank else 100
        return 50

    def _score_stretcher_pick(self, card) -> float:
        cid = card.id
        if cid == C.MEGA_STARMIE:
            return 500 if self.deck_megas == 0 and self.hand_counts[C.MEGA_STARMIE] == 0 else 260
        if cid == C.STARYU:
            return 450 if self.line_on_field < 2 else 120
        if cid == C.WATER:
            return 300 if self.hand_counts[C.WATER] == 0 else 100
        if cid == C.CINDERACE:
            return -1   # dead card in hand (evolution, setup-only)
        return 40

    def _score_discard_from_hand(self, card) -> float:
        # R27: dump finished-role cards; keep Water, draw supporters and Wally
        cid = card.id
        line_done = self.field_counts[C.MEGA_STARMIE] >= 1
        table = {
            C.IGNITION: 500 if not self.op_big_tank or self.hand_counts[C.IGNITION] >= 2 else 320,
            C.ULTRA_BALL: 480,
            C.CINDERACE: 460,       # unplayable from hand
            C.HARLEQUIN: 420,
            C.HAMMER: 380,
            C.POKEGEAR: 360,
            C.NIGHT_STRETCHER: 340,
            C.MEGA_SIGNAL: 450 if line_done or self.hand_counts[C.MEGA_STARMIE] >= 1 else 260,
            C.SALVATORE: 440 if line_done else 280,
            C.POFFIN: 430 if self.line_on_field >= 2 else 300,
            C.HILDA: 320,
            C.STARYU: 350 if self.line_on_field >= 2 else 200,
            C.LILLIE: 240,
            C.BOSS: 180,
            C.MEGA_STARMIE: 300 if self.field_counts[C.MEGA_STARMIE] >= 2 else 120,
            C.CAPE: 200,
            C.WATER: 160 if self.hand_counts[C.WATER] >= 2 else 90,
            C.WALLY: 30,            # keep Wally (R27 fix of the pro's own leak)
        }
        return table.get(cid, 250)

    # ----- PLAY -----

    def _score_play(self, option) -> float:
        card = get_card(self.obs, AreaType.HAND, option.index, self.my_index)
        if card is None:
            return 0
        data = card_table[card.id]
        if data.cardType == CardType.POKEMON:
            return self._score_play_pokemon(card)
        return self._score_play_trainer(card)

    def _score_play_pokemon(self, card) -> float:
        if card.id == C.STARYU:
            if self.line_on_field >= 2:
                return -1   # R28: bench stays lean (2 Starmie lines max)
            if (
                self.is_marnie
                and self.state.turn >= 4
                and self.line_on_field >= 1
                and self.hand_counts[C.SALVATORE] == 0
                and self.hand_counts[C.CAPE] == 0
            ):
                return -1   # R28: never pass the turn with a naked Staryu vs Marnie
            return 20000
        return -1           # Cinderace can't be played post-setup anyway

    def _score_play_trainer(self, card) -> float:
        cid = card.id
        if cid == C.WALLY:
            # R19/R20: full heal when ~150+ damage accumulated (bench Mega counts too)
            return 12500 if self._mega_best_damage() >= self._wally_threshold() else -1
        if cid == C.POFFIN:
            # R9: dig for Staryu early and often
            deck_staryu = max(0, 3 - self.line_on_field - self.hand_counts[C.STARYU]
                              - self.discard_counts[C.STARYU])
            if self.line_on_field >= 2 or deck_staryu == 0:
                return -1
            if (
                self.is_marnie
                and self.state.turn >= 4
                and self.line_on_field >= 1
                and self.hand_counts[C.SALVATORE] == 0
                and self.hand_counts[C.CAPE] == 0
            ):
                return -1   # R28
            return 12000
        if cid == C.POKEGEAR:
            return 11800    # R23: fire it nearly every turn, before the supporter
        if cid == C.SALVATORE:
            staryus = [p for p in self._my_board()
                       if p is not None and p.id == C.STARYU]
            if staryus and self.deck_megas > 0:
                evolvable_normally = (
                    self.hand_counts[C.MEGA_STARMIE] >= 1
                    and any(not p.appearThisTurn for p in staryus)
                )
                if not evolvable_normally:
                    return 11500   # R8: same-turn evolve via Salvatore
                return 2600
            return -1
        if cid == C.BOSS:
            return 11200 if plan.target >= 1 else -1   # R18
        if cid == C.MEGA_SIGNAL:
            # R9: grab the Mega body early
            return 11000 if self.hand_counts[C.MEGA_STARMIE] == 0 and self.deck_megas > 0 else -1
        if cid == C.HAMMER:
            # R21: strip the charged attacker / Munkidori's Dark
            if any(p is not None and p.energies for p in self._op_board()):
                return 10500
            return -1
        if cid == C.NIGHT_STRETCHER:
            if self.discard_counts[C.MEGA_STARMIE] or self.discard_counts[C.STARYU]:
                return 10000
            if self.discard_counts[C.WATER] and self.hand_counts[C.WATER] == 0:
                return 9900
            return -1
        if cid == C.HILDA:
            if self.hand_counts[C.MEGA_STARMIE] == 0 and self.deck_megas > 0:
                return 9700
            if self.hand_counts[C.WATER] == 0 and self.hand_counts[C.IGNITION] == 0:
                return 9600
            return 2500
        if cid == C.HARLEQUIN:
            # R25: refresh a starved hand / scramble a stacked opponent hand
            if self.me.handCount <= 3:
                return 9500
            if self.is_marnie and self.op.handCount >= 7 and self.me.handCount <= 5:
                return 9300
            return -1
        if cid == C.LILLIE:
            # R22: only after playables are spent; R6: don't torch a working hand
            if self.me.deckCount <= LOW_DECK:
                return -1
            if self.me.handCount <= 4:
                return 7200
            if self.me.handCount <= 6 and len(self.me.prize) == 6:
                return 7100
            return -1
        if cid == C.ULTRA_BALL:
            return -1       # R26: dead slot
        if cid == C.CAPE:
            return 0        # attached via ATTACH options
        return 5000

    # ----- ATTACH (energy + Cape) -----

    def _energy_target_score(self, pokemon: Pokemon, active: bool) -> float:
        # R11/R12 (+ Turbo Flare ramp targets)
        if pokemon.id == C.CINDERACE:
            return -1       # attacks for free; never invest
        ec = len(pokemon.energies)
        score = 8000.0 + (10 if active else 0)
        if pokemon.id == C.MEGA_STARMIE:
            if ec < 1:
                score += 200
            elif ec < 3:
                score += 60
            else:
                score -= 200   # over-invested (R11 low burn)
        elif pokemon.id == C.STARYU:
            score += 100 if ec < 1 else -150   # pre-load one W (R12)
        else:
            score -= 300
        return score

    def _score_attach(self, option) -> float:
        card = get_card(self.obs, AreaType.HAND, option.index, self.my_index)
        pokemon = get_card(self.obs, option.inPlayArea, option.inPlayIndex, self.my_index)
        if card is None or not isinstance(pokemon, Pokemon):
            return 0
        if card.id == C.CAPE:
            # R14: Cape on the main Mega or the Staryu about to become one
            if pokemon.id == C.MEGA_STARMIE:
                return 12300
            if pokemon.id == C.STARYU:
                return 12100
            return -1
        board_index = option.inPlayIndex if option.inPlayArea == AreaType.ACTIVE \
            else option.inPlayIndex + 1
        if card.id == C.IGNITION:
            # R13: only for a Nebula Beam this turn
            if (
                plan.needs_energy
                and plan.energy_card == C.IGNITION
                and board_index == plan.attacker
            ):
                return 8300
            return -1
        if card.id == C.WATER:
            if plan.needs_energy and plan.energy_card == C.WATER and board_index == plan.attacker:
                return 8300
            score = self._energy_target_score(pokemon, option.inPlayArea == AreaType.ACTIVE)
            # R12: if the active Mega already holds Water, feed the backup instead
            if pokemon.id == C.MEGA_STARMIE and option.inPlayArea == AreaType.BENCH:
                active0 = self.me.active[0] if self.me.active else None
                if (
                    active0 is not None
                    and active0.id == C.MEGA_STARMIE
                    and any(e == EnergyType.WATER for e in active0.energies)
                    and not any(e == EnergyType.WATER for e in pokemon.energies)
                ):
                    score += 180
            return score
        return 0

    # ----- Hammer target -----

    def _score_energy_pick(self, option) -> float:
        # Crushing Hammer: choose whose energy to discard (R21)
        pokemon = get_card(self.obs, option.area, option.index, option.playerIndex)
        if option.playerIndex == self.my_index:
            return -100
        if not isinstance(pokemon, Pokemon):
            return 0
        score = 100.0
        if option.area == AreaType.ACTIVE:
            score += 500    # stop the attacker for a turn
        if pokemon.id in MUNKIDORI:
            score += 450    # shut off Adrena-Brain
        score += len(pokemon.energies) * 20
        score += target_score(pokemon) * 0.05
        return score


# ---------------------------------------------------------------------------
# Lethal-only determinized search layer (MAIN only), ported from 007.
# Runs only when our remaining prizes are <=3. Candidate MAIN options are
# ranked by the heuristic; a candidate overrides the heuristic choice only if
# EVERY determinization sample ends the greedy rollout of our own turn with
# the game already won. No board evaluation is involved, so a bad eval can
# never overrule the tuned heuristic. Any failure falls back to the heuristic.

SEARCH_CANDIDATES = 8
SEARCH_SAMPLES = 5           # 007 used 3; Starmie lethal lines are draw-order
                             # dependent (Pokegear/Lillie), so more samples are
                             # needed to keep the false-positive rate near zero
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
    while len(unknown) < need:  # counting slack: pad with basic Water energy
        unknown.append(C.WATER)
    your_deck = unknown[: me.deckCount]
    your_prize = unknown[me.deckCount : me.deckCount + len(me.prize)]

    opponent_deck = [PLACEHOLDER_MON] * op.deckCount
    opponent_prize = [PLACEHOLDER_MON] * len(op.prize)
    opponent_hand = [C.WATER] * op.handCount
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
            sel = StarmiePolicy(obs).choose()
            if not sel and obs.select.minCount > 0:
                sel = _fallback_selection(obs.select)
        except Exception:
            sel = _fallback_selection(obs.select)
        state = search_step(state.searchId, sel)
    return state.observation


def _lethal_search(obs: Observation, policy: StarmiePolicy) -> list[int] | None:
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


# ---- EXP-091: 選択クラス予測モデル（GBT、EXP-057定式化のStarmie適用） ----
# ロード順: lightgbm+model_093.txt（ローカル高速）→ model_093.npz（純Python、
# Kaggle用。argmax一致検証済み）→ どちらも無ければ036ルールへ自然フォールバック
try:  # Kaggle評価環境はexecロードのため __file__ が無い
    _AGENT_DIR_093 = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _AGENT_DIR_093 = ("/kaggle_simulations/agent"
                      if os.path.exists("/kaggle_simulations/agent/main.py")
                      else os.getcwd())

_M093 = None
if os.environ.get("F093_ML", "on") != "off":
    try:
        try:
            import feat_093 as _ft093
        except ImportError:
            import importlib.util as _ilu093

            _fp = os.path.join(_AGENT_DIR_093, "feat_093.py")
            _sp = _ilu093.spec_from_file_location("feat_093", _fp)
            _ft093 = _ilu093.module_from_spec(_sp)
            _sp.loader.exec_module(_ft093)
        try:
            import lightgbm as _lgb093

            _mp = os.path.join(_AGENT_DIR_093, "model_093.txt")
            if not os.path.exists(_mp):
                raise FileNotFoundError(_mp)
            _M093 = {"mode": "lgb", "bst": _lgb093.Booster(model_file=_mp),
                     "ft": _ft093}
        except Exception:  # noqa: BLE001  lightgbm無し → npz純Python推論
            import numpy as _np093

            _z = _np093.load(os.path.join(_AGENT_DIR_093, "model_093.npz"))
            _M093 = {"mode": "npz", "ft": _ft093,
                     "feat": _z["feat"], "thr": _z["thr"],
                     "left": _z["left"], "right": _z["right"],
                     "val": _z["val"], "roots": _z["roots"],
                     "tcls": _z["tree_cls"],
                     "ncls": int(_z["num_class"])}
    except Exception:  # noqa: BLE001
        _M093 = None


def _m093_scores(x: list) -> list:
    """行動クラスごとの生スコア（softmax前。argmax用途なので正規化不要）。"""
    if _M093["mode"] == "lgb":
        return list(_M093["bst"].predict([x])[0])
    feat, thr = _M093["feat"], _M093["thr"]
    left, right, val = _M093["left"], _M093["right"], _M093["val"]
    scores = [0.0] * _M093["ncls"]
    for r, c in zip(_M093["roots"], _M093["tcls"]):
        i = int(r)
        while feat[i] >= 0:
            i = int(left[i]) if x[feat[i]] <= thr[i] else int(right[i])
        scores[int(c)] += float(val[i])
    return scores


def _ml_layer(obs: Observation, policy: StarmiePolicy) -> list[int] | None:
    """MAIN決定: GBTで行動クラスをargmax→クラス内は036policyスコアで具体option化。

    呼び出し時点で policy._plan_attack() 実行済み（planグローバルが有効）である
    こと。_score_option のタイブレークが036と同じ文脈で働く。
    """
    select = obs.select
    if select.minCount != 1 or select.maxCount != 1 or len(select.option) < 2:
        return None
    state = obs.current
    my = state.yourIndex
    ft = _M093["ft"]
    feats, _keys = ft.feat_vector(state, my)
    proba = _m093_scores(feats)
    cls_of = [ft.option_class(state, select, i, my)
              for i in range(len(select.option))]
    best_cls = max(set(cls_of), key=lambda c: proba[ft.CLASS_ID[c]])
    cands = [i for i, c in enumerate(cls_of) if c == best_cls]
    if len(cands) > 1:
        cands = [max(cands, key=lambda i: policy._score_option(select.option[i]))]
    return [cands[0]]


def agent(obs_dict: dict) -> list[int]:
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        return my_deck

    global pre_turn, plan
    if pre_turn != obs.current.turn:
        pre_turn = obs.current.turn
        plan = AttackPlan()

    policy = StarmiePolicy(obs)

    if obs.select.context == SelectContext.MAIN and obs.current.turn >= 2:
        # Set the heuristic plan for the real trajectory first (sub-selects rely on it),
        # then let the search override only the MAIN choice.
        policy._plan_attack()
        saved = (plan, pre_turn)
        try:
            choice = _lethal_search(obs, policy)
        except Exception:
            choice = None
        plan, pre_turn = saved
        if choice is not None:
            return choice
        if _M093 is not None:
            try:
                choice = _ml_layer(obs, policy)
            except Exception:  # noqa: BLE001
                choice = None
            if choice is not None:
                return choice
        return policy.choose()

    return policy.choose()
