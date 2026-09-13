import os
from collections import defaultdict

from cg.api import AreaType, CardType, Observation, SelectContext, OptionType, Card, Pokemon, all_card_data, to_observation_class

"""
040_alakazam_replica — Alakazamトップ個体 (sig a73706e527, THIRD PTCG Club,
本番272戦53.7%・対Marnie 70.8%) の軽量逆設計レプリカ。

010_alakazam の骨格をベースに、全272戦リプレイの行動頻度集計で判明した癖を反映:
- 先攻は常にYES / セットアップアクティブは Dunsparce > Abra > Fezandipiti ex
- Powerful Hand は使えるターンは必ず撃つ（保留ゼロ。非致死でも1割は撃つ）
- Fez ex / Dudunsparce のドロー能力は「キルに必要な時だけ」でなく毎回使う
  （ACTIVATE系は deckCount<=5 でのみ NO = 本人の実測 60% 拒否と一致）
- Battle Cage は空スタジアムでも即置き（対Marnie: Adrena-Brainのベンチ狙撃を遮断）
- サポート優先: Boss(キル限定) > Xerosic(相手手札4+) > Lana's Aid(回収2+) > Dawn/Hilda
- Boss はキル専用（gust対象の92%がそのターンPowerful Handで処理可能な相手）
- エネルギーは1体1枚まで。Telepath/P→Alakazamライン、Enriching→Dunsparceライン/Fez
- サーチ系（Poffin/Poké Pad/Dawn/Hilda/Telepath）に山札残数ゲートは掛けない
  （ドローしないので山札切れに寄与しない。010はここが過剰に保守的だった）
"""

# Load deck.csv in the dataset
try:  # Kaggle評価環境はexecロードのため __file__ が無い
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deck.csv")
except NameError:
    file_path = "deck.csv"
if not os.path.exists(file_path):
    file_path = "/kaggle_simulations/agent/deck.csv"
with open(file_path, "r") as file:
    csv = file.read().split("\n")
my_deck = []
for i in range(60):
    my_deck.append(int(csv[i]))

all_card = all_card_data()
card_table = {c.cardId: c for c in all_card}

# Decklist (sig a73706e527)
Abra = 741               # x4
Kadabra = 742            # x4
Alakazam = 743           # x4
Dunsparce = 305          # x3
Dudunsparce = 66         # x3
Fezandipiti_ex = 140     # x1
Buddy_Buddy_Poffin = 1086  # x4
Poke_Pad = 1152          # x4
Enhanced_Hammer = 1081   # x4
Rare_Candy = 1079        # x3
Night_Stretcher = 1097   # x2
Sacred_Ash = 1129        # x1
Tool_Scrapper = 1137     # x1
Dawn = 1231              # x4
Hilda = 1225             # x4
Boss_Orders = 1182       # x3
Xerosic = 1197           # x1
Lanas_Aid = 1184         # x1
Battle_Cage = 1264       # x2
Basic_Psychic_Energy = 5     # x2
Telepath_Psychic_Energy = 19  # x4
Enriching_Energy = 13    # x1 (ACE SPEC)

# Special defense energies to watch on opponents
Mist_Energy = 11
Rock_Fighting_Energy = 20

# Attack IDs
ATTACK_TELEPORTATION = 1070   # Abra: 10 dmg, switch self
ATTACK_SUPER_PSY_BOLT = 1071  # Kadabra: 30 dmg
ATTACK_POWERFUL_HAND = 1072   # Alakazam: 20 per card in hand
ATTACK_TRADING_PLACES = 423   # Dunsparce: 0 dmg, switch self

ABRA_LINE = {Abra, Kadabra, Alakazam}
DUNSPARCE_LINE = {Dunsparce, Dudunsparce}
PSYCHIC_ENERGY_IDS = {Basic_Psychic_Energy, Telepath_Psychic_Energy}
SUPPORTER_IDS = {Dawn, Hilda, Boss_Orders, Xerosic, Lanas_Aid}

pre_turn = 0
ability_used_dudunsparce = False
ability_used_fezandipiti = False


def get_card(obs: Observation, area: AreaType, index: int, player_index: int) -> Pokemon | Card | None:
    ps = obs.current.players[player_index]
    try:
        match area:
            case AreaType.DECK:
                return obs.select.deck[index]
            case AreaType.HAND:
                return ps.hand[index]
            case AreaType.DISCARD:
                return ps.discard[index]
            case AreaType.ACTIVE:
                return ps.active[index]
            case AreaType.BENCH:
                return ps.bench[index]
            case AreaType.PRIZE:
                return ps.prize[index]
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
        if card.id == 12:  # Legacy Energy
            count -= 1
    for card in pokemon.tools:
        if card.id == 1172 and "Lillie" in data.name:
            count -= 1
    return max(0, count)


def count_special_defense_energies(pokemon: Pokemon) -> int:
    cnt = 0
    for ec in pokemon.energyCards:
        if ec.id == Mist_Energy or ec.id == Rock_Fighting_Energy:
            cnt += 1
    return cnt


def agent(obs_dict: dict) -> list[int]:
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        return my_deck

    state = obs.current
    select = obs.select
    context = select.context
    my_index = state.yourIndex
    my_state = state.players[my_index]
    op_state = state.players[1 - my_index]
    my_prize_count = len(my_state.prize)

    global pre_turn, ability_used_dudunsparce, ability_used_fezandipiti
    if pre_turn != state.turn:
        pre_turn = state.turn
        ability_used_dudunsparce = False
        ability_used_fezandipiti = False

    # ---- Count cards on field / hand / discard ----
    field_counts = defaultdict(int)
    hand_counts = defaultdict(int)
    discard_counts = defaultdict(int)

    my_field = []  # (field_index, pokemon) where 0=active, 1..=bench
    for card in my_state.active:
        if card is not None:
            field_counts[card.id] += 1
            my_field.append((0, card))
    for idx, card in enumerate(my_state.bench):
        if card is not None:
            field_counts[card.id] += 1
            my_field.append((idx + 1, card))

    for card in (my_state.hand or []):
        hand_counts[card.id] += 1

    for card in my_state.discard:
        discard_counts[card.id] += 1

    abra_line_on_field = field_counts[Abra] + field_counts[Kadabra] + field_counts[Alakazam]
    dunsparce_line_on_field = field_counts[Dunsparce] + field_counts[Dudunsparce]

    # ---- Opponent field analysis ----
    op_all_pokemon = []
    for card in op_state.active:
        if card is not None:
            op_all_pokemon.append(card)
    for card in op_state.bench:
        if card is not None:
            op_all_pokemon.append(card)

    op_tool_count = sum(len(p.tools) for p in op_all_pokemon)

    stadium_id = 0
    stadium_is_mine = False
    for card in state.stadium:
        stadium_id = card.id
        stadium_is_mine = card.playerIndex == my_index

    bench_count = sum(1 for c in my_state.bench if c is not None)
    bench_max = my_state.benchMax
    bench_free = bench_max - bench_count

    def has_psychic_energy(pokemon: Pokemon) -> bool:
        # Powerful Hand / Super Psy Bolt のコスト{P}を払えるか（EnrichingのCは不可）
        return any(ec.id in PSYCHIC_ENERGY_IDS for ec in pokemon.energyCards)

    # ---- Active pokemon info ----
    active_pokemon = my_state.active[0] if my_state.active else None
    active_id = active_pokemon.id if active_pokemon else -1
    active_has_energy = bool(active_pokemon.energies) if active_pokemon else False
    active_has_psychic = has_psychic_energy(active_pokemon) if active_pokemon else False

    # ---- Opponent active info ----
    op_active = op_state.active[0] if op_state.active else None
    op_active_hp = op_active.hp if op_active else 9999

    hand_size = len(my_state.hand) if my_state.hand else my_state.handCount
    deck_count = my_state.deckCount

    # ---- Estimate hand size increase this turn (for kill math) ----
    def estimate_hand_increase():
        max_inc = 0
        for _, p in my_field:
            if p.id == Abra and hand_counts[Kadabra] > 0:
                max_inc += 1  # evolve: -1 +2
            elif p.id == Abra and hand_counts[Rare_Candy] > 0 and hand_counts[Alakazam] > 0:
                max_inc += 1  # -2 +3
            elif p.id == Kadabra and hand_counts[Alakazam] > 0:
                max_inc += 2  # -1 +3
            elif p.id == Dunsparce and hand_counts[Dudunsparce] > 0:
                max_inc += 1
            elif p.id == Dudunsparce and not ability_used_dudunsparce:
                max_inc += 3
            elif p.id == Fezandipiti_ex and not ability_used_fezandipiti:
                max_inc += 3
        if hand_counts[Fezandipiti_ex] > 0 and bench_free > 0 and field_counts[Fezandipiti_ex] == 0:
            max_inc += 2
        supporter_options = []
        if not state.supporterPlayed:
            if hand_counts[Hilda] > 0:
                supporter_options.append(1)
            if hand_counts[Dawn] > 0:
                supporter_options.append(2)
            if hand_counts[Lanas_Aid] > 0:
                supporter_options.append(2)
        if supporter_options:
            max_inc += max(supporter_options)
        if hand_counts[Enriching_Energy] > 0 and not state.energyAttached:
            max_inc += 3
        return max_inc

    max_hand_size = hand_size + estimate_hand_increase()
    max_damage = max_hand_size * 20

    # ---- Target selection for Powerful Hand ----
    target_idx = -1
    target_pokemon = None
    target_use_boss = False
    target_can_kill = False
    target_prize_gain = 0
    target_hammer_needed = 0
    use_kadabra_finish = False

    if state.turn >= 2 and op_active is not None:
        if op_active_hp <= 30 and (field_counts[Kadabra] >= 1 or active_id == Kadabra) and field_counts[Alakazam] == 0:
            target_idx = 0
            target_pokemon = op_active
            target_can_kill = True
            target_prize_gain = prize_count(op_active)
            use_kadabra_finish = True
        else:
            all_op = [(0, op_active)]
            for bi, bp in enumerate(op_state.bench):
                if bp is not None:
                    all_op.append((bi + 1, bp))

            candidates = []
            for oi, pkmn in all_op:
                pz = prize_count(pkmn)
                sp_e = count_special_defense_energies(pkmn)
                eff_max_dmg = max_damage
                hm_need = 0
                if sp_e > 0:
                    if hand_counts[Enhanced_Hammer] >= sp_e:
                        hm_need = sp_e
                        eff_max_dmg = (max_hand_size - hm_need) * 20
                    else:
                        eff_max_dmg = 0
                ck = pkmn.hp <= eff_max_dmg and eff_max_dmg > 0
                candidates.append((oi, pkmn, pz, ck, hm_need))

            win_cands = [c for c in candidates if c[3] and my_prize_count <= c[2]]
            if win_cands:
                best = min(win_cands, key=lambda x: (0 if x[0] == 0 else 1, -x[1].hp))
                target_idx, target_pokemon, target_prize_gain, target_can_kill, target_hammer_needed = best
                target_use_boss = target_idx != 0
            else:
                killable = [c for c in candidates if c[3]]
                if killable:
                    best = max(killable, key=lambda x: (x[2], x[1].hp))
                    target_idx, target_pokemon, target_prize_gain, target_can_kill, target_hammer_needed = best
                    target_use_boss = target_idx != 0
                else:
                    target_idx = 0
                    target_pokemon = op_active
                    target_can_kill = False

    can_win_this_turn = target_can_kill and my_prize_count <= target_prize_gain

    # Boss がベンチのキル対象を吊るために必要か
    need_boss = target_use_boss and target_can_kill

    # キルに追加ドローが必要か（Fez/Dudunsparce判断の補助。基本は毎回使う）
    need_draw_for_kill = False
    if target_pokemon is not None and target_can_kill:
        current_dmg = (hand_size - target_hammer_needed) * 20
        if current_dmg < target_pokemon.hp:
            need_draw_for_kill = True

    # 山札切れガード: ドロー効果は残数が薄いときだけ止める（本人実測: deck<=5でNO 60%）
    draw_ok = deck_count >= 6 or can_win_this_turn

    # リトリートのためのエネルギーが必要か
    need_retreat_energy = False
    ready_bench_alakazam = any(
        p.id == Alakazam and has_psychic_energy(p) for fi, p in my_field if fi != 0
    )
    bench_has_kadabra = any(p.id == Kadabra for fi, p in my_field if fi != 0)
    active_is_attacker = (active_id == Alakazam and active_has_psychic) or (
        use_kadabra_finish and active_id == Kadabra and active_has_psychic)
    bench_has_attacker = ready_bench_alakazam or (
        use_kadabra_finish and bench_has_kadabra)
    if active_pokemon is not None and state.turn >= 2:
        if not active_is_attacker and bench_has_attacker:
            retreat_cost = card_table[active_pokemon.id].retreatCost
            if len(active_pokemon.energies) < retreat_cost:
                need_retreat_energy = True

    # ---- Score each option ----
    scores = []
    for o in select.option:
        score = 0

        if o.type == OptionType.NUMBER:
            score = o.number

        elif o.type == OptionType.YES:
            # 先攻・マリガン等は常にYES。ドロー能力のACTIVATEだけ山札残数で拒否
            if context == SelectContext.ACTIVATE and not draw_ok:
                score = -1
            else:
                score = 1

        elif o.type == OptionType.NO:
            score = 0

        elif o.type == OptionType.CARD:
            card = get_card(obs, o.area, o.index, o.playerIndex)
            if card is None:
                scores.append(score)
                continue
            energy_count = len(card.energies) if isinstance(card, Pokemon) else 0

            if context == SelectContext.SWITCH or context == SelectContext.TO_ACTIVE:
                if o.playerIndex == my_index:
                    # 昇格/入替: Alakazam(エネ付き優先) > Kadabra > Abra > Dunsparce。
                    # Dudunsparce/Fez はベンチ温存（本人: 選択 Zam523 / 拒否 Dudun700, Fez559）
                    if card.id == Alakazam:
                        score += 100 + energy_count * 10
                    elif card.id == Kadabra:
                        score += 90 if (op_active_hp <= 30) else 30
                    elif card.id == Abra:
                        score += 10
                    elif card.id == Dunsparce:
                        score += 5
                    elif card.id == Dudunsparce:
                        score += 3
                    else:
                        score += 1
                else:
                    if target_use_boss and target_pokemon is not None:
                        if o.index == target_idx - 1:
                            score += 100

            elif context == SelectContext.SETUP_ACTIVE_POKEMON:
                # 本人: Dunsparce 128 > Abra 124 (pairwise 48:4) > Fez 20
                if card.id == Dunsparce:
                    score = 10
                elif card.id == Abra:
                    score = 5
                elif card.id == Fezandipiti_ex:
                    score = 2

            elif context == SelectContext.SETUP_BENCH_POKEMON:
                # 本人はセットアップで持っている基本ポケモンを全部置く
                if card.id == Abra:
                    score = 200
                elif card.id == Dunsparce:
                    score = 150
                elif card.id == Fezandipiti_ex:
                    score = 120
                else:
                    score = 50

            elif context == SelectContext.TO_HAND:
                score = 200 - hand_counts.get(card.id, 0) * 50
                if card.id == Dudunsparce:
                    score += 80 if (field_counts[Dunsparce] >= 1 and field_counts[Dudunsparce] == 0) else -50
                elif card.id == Kadabra:
                    score += 70 if field_counts[Abra] >= 1 else -20
                elif card.id == Alakazam:
                    score += 60 if (field_counts[Kadabra] >= 1 or field_counts[Abra] >= 1) else -20
                elif card.id == Abra:
                    score += 50 if abra_line_on_field < 3 else -50
                elif card.id == Dunsparce:
                    score += 40 if dunsparce_line_on_field < 2 else -50
                elif card.id == Fezandipiti_ex:
                    score += 45 if (field_counts[Fezandipiti_ex] == 0 and hand_counts[Fezandipiti_ex] == 0) else -60
                elif card.id == Enriching_Energy:
                    score += 65  # Hildaで強く優先 (chosen123 : rejected31)
                elif card.id in PSYCHIC_ENERGY_IDS:
                    score += 30 if not state.energyAttached else 10
                elif card.id == Rare_Candy:
                    score += 40 if field_counts[Abra] >= 1 else -10

            elif context == SelectContext.ATTACH_FROM:
                if isinstance(card, Pokemon):
                    if need_retreat_energy and o.area == AreaType.ACTIVE:
                        score = 150
                    elif len(card.energyCards) >= 1:
                        score = -1  # 1体1枚（本人: 重ね貼りは1%未満）
                    elif card.id in ABRA_LINE:
                        score = 100
                        if card.id == Alakazam:
                            score += 20
                        elif card.id == Kadabra:
                            score += 10
                        if o.area == AreaType.ACTIVE:
                            score += 5
                    elif card.id in DUNSPARCE_LINE:
                        score = 50
                    else:
                        score = 10

            elif context == SelectContext.TO_BENCH:
                # Poffin / Telepath のベンチ搬入（本人: Abra 534 > Dunsparce 302）
                if card.id == Abra:
                    score = 100
                elif card.id == Dunsparce:
                    score = 80 if dunsparce_line_on_field < 2 else 40
                else:
                    score = 10

            elif context == SelectContext.TO_DECK or context == SelectContext.TO_DECK_BOTTOM:
                # Sacred Ash: Powerful Handライン優先で戻す
                if card.id in ABRA_LINE:
                    score = 100
                elif card.id in DUNSPARCE_LINE:
                    score = 50
                else:
                    score = 10

            elif context == SelectContext.DISCARD:
                # 相手のXerosic/Judge等で自分の手札を捨てる順（本人実測の捨て率準拠）
                score = {
                    Boss_Orders: 100,
                    Enhanced_Hammer: 95,
                    Hilda: 88,
                    Dawn: 86,
                    Buddy_Buddy_Poffin: 80,
                    Poke_Pad: 78,
                    Battle_Cage: 70,
                    Tool_Scrapper: 65,
                    Sacred_Ash: 60,
                    Night_Stretcher: 58,
                    Rare_Candy: 50,
                    Kadabra: 45,
                    Dunsparce: 40,
                    Basic_Psychic_Energy: 30,
                    Telepath_Psychic_Energy: 25,
                    Dudunsparce: 20,
                    Abra: 15,
                    Alakazam: 10,
                    Xerosic: 8,
                    Lanas_Aid: 8,
                    Enriching_Energy: 5,
                    Fezandipiti_ex: 2,
                }.get(card.id, 55)
                # 複数持ちは捨てやすい
                score += hand_counts.get(card.id, 0) * 5

        elif o.type == OptionType.PLAY:
            card = get_card(obs, AreaType.HAND, o.index, my_index)
            if card is None:
                scores.append(score)
                continue
            data = card_table[card.id]

            if data.cardType == CardType.POKEMON:
                score = 20000
                is_early = state.turn <= 2

                if card.id == Abra:
                    if is_early:
                        score += 500
                    elif abra_line_on_field < 4:
                        score += 200
                    elif bench_free <= 1:
                        score = -1
                    else:
                        score += 50

                elif card.id == Dunsparce:
                    if dunsparce_line_on_field < 1:
                        score += 400 if is_early else 100
                    elif dunsparce_line_on_field < 2:
                        score += 50
                    else:
                        score = -1

                elif card.id == Fezandipiti_ex:
                    # 本人は自由に置く（193回）。ドロー能力を毎回使う前提
                    if field_counts[Fezandipiti_ex] == 0 and bench_free >= 1:
                        score += 100
                    else:
                        score = -1

                if bench_free <= 1 and score > 0:
                    score -= 5000

            else:
                score = 10000

                if card.id == Buddy_Buddy_Poffin:
                    # サーチのみ（ドローなし）: 山札ゲート不要
                    if state.turn <= 2:
                        score = 18000 if (abra_line_on_field < 3 or dunsparce_line_on_field < 1) else 8000
                    elif abra_line_on_field < 4 or dunsparce_line_on_field < 2:
                        score = 15000
                    elif bench_free >= 2:
                        score = 8000
                    else:
                        score = -1

                elif card.id == Poke_Pad:
                    score = 17000 if state.turn <= 2 else 14000

                elif card.id == Rare_Candy:
                    if field_counts[Abra] >= 1 and hand_counts[Alakazam] >= 1:
                        score = 16000
                    else:
                        score = -1

                elif card.id == Night_Stretcher:
                    dis_abra = discard_counts[Abra] + discard_counts[Kadabra] + discard_counts[Alakazam]
                    dis_energy = discard_counts[Basic_Psychic_Energy] + discard_counts[Telepath_Psychic_Energy]
                    if dis_abra >= 1:
                        score = 13000
                    elif dis_energy >= 1 or discard_counts[Dunsparce] + discard_counts[Dudunsparce] >= 1:
                        score = 11000
                    else:
                        score = -1

                elif card.id == Sacred_Ash:
                    dis_pokemon = sum(discard_counts[c] for c in
                                      (Abra, Kadabra, Alakazam, Dunsparce, Dudunsparce))
                    if dis_pokemon >= 3:
                        score = 13500
                    elif dis_pokemon >= 1:
                        score = 11000
                    else:
                        score = -1

                elif card.id == Enhanced_Hammer:
                    if target_hammer_needed > 0:
                        score = 6500
                    else:
                        any_special = any(
                            any(card_table.get(ec.id) is not None and card_table[ec.id].cardType == CardType.SPECIAL_ENERGY
                                for ec in p.energyCards)
                            for p in op_all_pokemon
                        )
                        score = 5000 if any_special else -1

                elif card.id == Tool_Scrapper:
                    score = 6000 if op_tool_count >= 1 else -1

                elif card.id == Boss_Orders:
                    score = 3400 if need_boss else -1

                elif card.id == Xerosic:
                    # 相手の手札が4枚以上なら最優先で撃つ（本人: 相手手札4+で159回）
                    score = 3300 if op_state.handCount >= 4 else -1

                elif card.id == Lanas_Aid:
                    recoverable = (discard_counts[Basic_Psychic_Energy]
                                   + sum(discard_counts[c] for c in
                                         (Abra, Kadabra, Alakazam, Dunsparce, Dudunsparce)))
                    score = 3150 if recoverable >= 2 else -1

                elif card.id == Dawn:
                    # 進化ライン部品のサーチ。部品が手札に無い時ほど価値大
                    if hand_counts[Alakazam] == 0 or hand_counts[Kadabra] == 0:
                        score = 3100
                    else:
                        score = 2950

                elif card.id == Hilda:
                    # 進化ポケモン+エネルギー
                    energy_in_hand = (hand_counts[Basic_Psychic_Energy]
                                      + hand_counts[Telepath_Psychic_Energy]
                                      + hand_counts[Enriching_Energy])
                    score = 3050 if energy_in_hand == 0 else 3000

                elif card.id == Battle_Cage:
                    # 本人は空スタジアムでも即置き（対Marnie: ベンチのAbraをAdrena-Brainから守る）
                    if stadium_id == Battle_Cage and stadium_is_mine:
                        score = -1
                    else:
                        score = 7000

        elif o.type == OptionType.ATTACH:
            card = get_card(obs, AreaType.HAND, o.index, my_index)
            pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
            if card is None or pokemon is None:
                scores.append(score)
                continue

            if card.id in PSYCHIC_ENERGY_IDS:
                if need_retreat_energy and o.inPlayArea == AreaType.ACTIVE:
                    score = 9500
                elif pokemon.id in ABRA_LINE and not has_psychic_energy(pokemon):
                    # P未装備のアタッカーには（Enriching汚染されていても）Pを貼る
                    score = 8000 if len(pokemon.energyCards) == 0 else 7900
                    if pokemon.id == Alakazam:
                        score += 30
                    elif pokemon.id == Kadabra:
                        score += 20
                    elif pokemon.id == Abra:
                        score += 10
                    if o.inPlayArea == AreaType.ACTIVE:
                        score += 5
                elif len(pokemon.energyCards) >= 1:
                    score = -1
                elif pokemon.id in DUNSPARCE_LINE:
                    score = 7000  # リトリート用
                else:
                    score = -1

            elif card.id == Enriching_Energy:
                # ドロー4: 山札ガード付き。付け先は Dunsparceライン > Fez（本人: 143/41/37）。
                # {C}しか出ないため Abraライン(Powerful Handの{P}コスト)には絶対貼らない
                if deck_count < 7:
                    score = -1
                elif len(pokemon.energyCards) >= 1:
                    score = -1
                elif pokemon.id in DUNSPARCE_LINE:
                    score = 8500
                    if pokemon.id == Dudunsparce:
                        score += 10
                elif pokemon.id == Fezandipiti_ex:
                    score = 8300
                else:
                    score = -1

        elif o.type == OptionType.EVOLVE:
            card = get_card(obs, AreaType.HAND, o.index, my_index)
            pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
            score = 9000
            if card is None or pokemon is None:
                scores.append(score)
                continue

            if card.id == Alakazam:
                if o.inPlayArea == AreaType.ACTIVE:
                    score += 200
                else:
                    score += 50
                score += len(pokemon.energies) * 10

            elif card.id == Kadabra:
                score += 100
                if len(pokemon.energies) == 0:
                    score += 50
                else:
                    score -= 20
                    if hand_counts[Rare_Candy] > 0 and hand_counts[Alakazam] > 0:
                        score -= 100  # エネ付きAbraはRare Candy直行を温存

            elif card.id == Dudunsparce:
                score += 80

        elif o.type == OptionType.ABILITY:
            card = get_card(obs, o.area, o.index, my_index)
            if card is None:
                scores.append(score)
                continue

            if card.id == Dudunsparce:
                # 本人は毎ターン使う(327回)。山札が薄い時だけ止める
                if draw_ok or (need_draw_for_kill and deck_count >= 4):
                    score = 30000
                else:
                    score = -1
            elif card.id == Fezandipiti_ex:
                # 本人は使える時は必ず使う(395回)
                if draw_ok or (need_draw_for_kill and deck_count >= 4):
                    score = 29000
                else:
                    score = -1
            elif card.id == Battle_Cage:
                score = 1
            else:
                score = 28000  # 相手スタジアムの起動効果等（本人: Spikemuth Gym 35回）

        elif o.type == OptionType.RETREAT:
            if active_id == Alakazam and active_has_psychic:
                score = -1
            elif use_kadabra_finish and active_id != Kadabra and bench_has_kadabra:
                score = 2500
            elif active_id in (Abra, Dunsparce, Dudunsparce, Fezandipiti_ex, Kadabra, Alakazam):
                # ベンチに撃てるAlakazamがいる時だけ下がる（アクティブ自身は数えない）
                score = 2000 if ready_bench_alakazam else -1
            else:
                score = -1

        elif o.type == OptionType.ATTACK:
            # Powerful Hand は使えるターンは必ず撃つ（本人: fire 1344 / hold 0）
            score = 1000
            if o.attackId == ATTACK_POWERFUL_HAND:
                score += 500
            elif o.attackId == ATTACK_SUPER_PSY_BOLT:
                score += 600 if op_active_hp <= 30 else 100
            elif o.attackId == ATTACK_TELEPORTATION:
                # ベンチにアタッカーが立っているなら脱出、いなければ10点でも殴る
                score += 200 if ready_bench_alakazam else 50
            elif o.attackId == ATTACK_TRADING_PLACES:
                score += 150 if ready_bench_alakazam else 20

        scores.append(score)

    desc_indices = [i for i, _ in sorted(enumerate(scores), key=lambda x: x[1], reverse=True)]

    if context == SelectContext.MAIN:
        o = select.option[desc_indices[0]]
        if o.type == OptionType.ABILITY:
            card = get_card(obs, o.area, o.index, my_index)
            if card is not None:
                if card.id == Dudunsparce:
                    ability_used_dudunsparce = True
                elif card.id == Fezandipiti_ex:
                    ability_used_fezandipiti = True

    return desc_indices[:select.maxCount]
