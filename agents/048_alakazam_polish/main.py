"""048_alakazam_polish — 045_alakazam_full の弱点3対面磨き込み (EXP-048)。
045 = LBトップ Yushin Ito の Alakazam個体 (sig 81f1758c92) フル精読レプリカ+lethal探索
(EXP-045)。deck.csv は045と同一60枚。差分は knowledge/study/045_loss_study.md の F1〜F8:

- F1: 対Kangaskhan/Crustle消耗戦の山経済（deck<=20でキル見込みターン以外の任意ドロー停止）
- F2: 効果無効アクティブへの空撃ちターン最小化（Kadabraフィニッシュの無効貫通 /
      Land Crush線の一般化 / 無効+キル不能時のドロー停止）
- F3: 無効エネモデルに Rock Fighting Energy(id20) を追加（045固有バグ修正）
- F4: 対タンクの Enhanced Hammer は「剥がす→同ターンキル」成立時のみ
- F5: 対タンクは3プライズバースト志向（1przドリップのための掘り/Boss吊りを抑制）
- F6: 対Marnie/Cynthia系では Fez ex を手札から出さない（2プライズ献上防止）
- F7: Spiritomb被弾モデル（Raging Curse threat>=140でDunsparce壁昇格、
      Cynthia'sアクティブへの非キルチップ攻撃禁止）
- F8: 対タンク・残プライズ<=2では捨て順のBossを後方へ温存

方策仕様: knowledge/study/alakazam_81f1_study.md（R1〜R26、頻度証拠付き）。
040_alakazam_replica の骨格をベースに、精読で判明した差分を焼き込み:
- セットアップアクティブ Abra > Dunsparce > Shaymin > Fez（a737と逆）
- セットアップベンチは Abra/Dunsparce のみ置く（Fez/Shayminはskip）
- Xerosic×3: 相手手札>=7で最優先級、<=3では撃たない
- Nighttime Mine は相手スタジアムのバウンス用（ミラーでは手札温存=PH打点）
- Powerful Hand はダメカン「効果」: Mist Energy / TR Articuno の Repelling Veil
  （たねロケット団全員）に完全無効 → キル候補から除外し Boss で回避
- Enhanced Hammer 優先: PH対象のMist > ミラーの相手AbraラインTelepath > TR Energy
- ドロー能力の自制: 手札が既に致死 & 山が薄いときは進化ドロー/能力を止める（山切れ管理）
- 対TR(Articuno) は Dudunsparce にエネ3枚 → Land Crush 90 の迂回打点
- lethal探索: 残りプライズ<=3のMAINのみ、全サンプル勝利の手だけ上書き
  （ロールアウトとグローバル方策状態はスナップショットで分離、EXP-043の教訓）
"""

import os
import random
import time
from collections import Counter, defaultdict

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
    search_begin,
    search_step,
    search_end,
)

# Load deck.csv (3段構え: __file__ / cwd / kaggle固定パス)
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

# Decklist (sig 81f1758c92)
Abra = 741               # x4
Kadabra = 742            # x4
Alakazam = 743           # x4
Dunsparce = 305          # x3
Dudunsparce = 66         # x2
Fezandipiti_ex = 140     # x1
Shaymin = 343            # x1 (Flower Curtain)
Buddy_Buddy_Poffin = 1086  # x4
Poke_Pad = 1152          # x4
Enhanced_Hammer = 1081   # x3
Rare_Candy = 1079        # x3
Night_Stretcher = 1097   # x1
Sacred_Ash = 1129        # x1
Dawn = 1231              # x4
Hilda = 1225             # x4
Boss_Orders = 1182       # x3
Xerosic = 1197           # x3
Lanas_Aid = 1184         # x1
Nighttime_Mine = 1266    # x3 (stadium)
Basic_Psychic_Energy = 5     # x2
Telepath_Psychic_Energy = 19  # x4
Enriching_Energy = 13    # x1 (ACE SPEC)

# Opponent-side cards we must model
Mist_Energy = 11
Rock_Fighting_Energy = 20  # "Prevent all effects" = Mist同等（041系Garchompが4枚採用）
TR_Articuno = 414        # Repelling Veil: counters/effects blocked on basic TR mons

# ---- 048 polish flags（F1〜F8、knowledge/study/045_loss_study.md。切り分け用） ----
# EXP-048アブレーションの結論: 採用は F3 のみ。
#   F1/F2/F4/F5（山経済・空撃ち抑制・Hammer/バースト規律）は対047で-3〜-10pt劣化
#   （ドローエンジン抑制は「PH打点」だけでなく「アタッカー部品供給」も止めて逆効果）、
#   F6（Fez抑制）は対038で-5.5pt（Fezドローの喪失が2prz献上リスクを上回る）、
#   F7（Spiritombチップ禁止）は対041で-7.5pt、F8は効果なし。
# 環境変数 F048_ON / F048_OFF（カンマ区切り、ONが優先）で切替可能。
_F_ON = set(filter(None, os.environ.get("F048_ON", "").split(",")))
_F_OFF = set(filter(None, os.environ.get("F048_OFF", "").split(",")))
_F_DEFAULT_ON = {"F3"}


def _flag(name: str) -> bool:
    if name in _F_ON:
        return True
    if name in _F_OFF:
        return False
    return name in _F_DEFAULT_ON


F1_DECK_ECONOMY = _flag("F1")
F2_BLOCKED_TURN = _flag("F2")
F3_ROCK_FIGHTING = _flag("F3")
F4_HAMMER_ECONOMY = _flag("F4")
F5_BURST_TARGETING = _flag("F5")
F6_FEZ_SUPPRESS = _flag("F6")
F7_SPIRITOMB_MODEL = _flag("F7")
F8_BOSS_PRESERVE = _flag("F8")

BLOCKING_ENERGY_IDS = (
    {Mist_Energy, Rock_Fighting_Energy} if F3_ROCK_FIGHTING else {Mist_Energy}
)
TANK_IDS = {756, 344, 345, 532, 533}    # Mega Kangaskhan ex / Dwebble / Crustle（両刷り）
MARNIE_IDS = {646, 647, 648, 112, 139}  # Impidimp / Morgrem / Grimmsnarl ex / Munkidori(ex)
CYNTHIA_IDS = {c.cardId for c in all_card if c.name.startswith("Cynthia's")}
CYNTHIA_SPIRITOMB = 387

# Attack IDs
ATTACK_TELEPORTATION = 1070   # Abra: 10 dmg, switch self
ATTACK_SUPER_PSY_BOLT = 1071  # Kadabra: 30 dmg
ATTACK_POWERFUL_HAND = 1072   # Alakazam: 2 counters per card in hand
ATTACK_TRADING_PLACES = 423   # Dunsparce: 0 dmg, switch self
ATTACK_RAM = 424              # Dunsparce: 20
ATTACK_LAND_CRUSH = 76        # Dudunsparce: 90 (CCC) — Mist/Veilを素通しするダメージ
ATTACK_SMASH_KICK = 477       # Shaymin: 30
ATTACK_CRUEL_ARROW = 183      # Fez ex: 100 to any

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


def is_tr_basic(cid: int) -> bool:
    data = card_table.get(cid)
    return data is not None and data.basic and "Team Rocket" in data.name


def blocking_count(pokemon: Pokemon) -> int:
    # 効果無効エネの枚数（F3: Mist + Rock Fighting。PH=ダメカン効果を完全無効化）
    return sum(1 for ec in pokemon.energyCards if ec.id in BLOCKING_ENERGY_IDS)


def policy(obs: Observation) -> list[int]:
    """Score every option and return chosen indices (score<0 = skip if allowed)."""
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

    op_articuno = any(p.id == TR_Articuno for p in op_all_pokemon)
    op_has_tera = any(card_table.get(p.id) is not None and card_table[p.id].tera for p in op_all_pokemon)
    op_is_alakazam_mirror = any(p.id in ABRA_LINE for p in op_all_pokemon)

    # ---- 048: 対面検知（盤面+進化元+相手トラッシュ） ----
    op_seen_ids = set()
    for p in op_all_pokemon:
        op_seen_ids.add(p.id)
        for c in p.preEvolution:
            op_seen_ids.add(c.id)
    for c in op_state.discard:
        op_seen_ids.add(c.id)
    op_tank = bool(op_seen_ids & TANK_IDS)        # Kangaskhan/Crustle系消耗戦（F1/F4/F5/F8）
    op_marnie = bool(op_seen_ids & MARNIE_IDS)
    op_cynthia = bool(op_seen_ids & CYNTHIA_IDS)
    fez_suppress = F6_FEZ_SUPPRESS and (op_marnie or op_cynthia)  # F6
    op_block_energy_seen = any(
        ec.id in BLOCKING_ENERGY_IDS for p in op_all_pokemon for ec in p.energyCards
    ) or any(c.id in BLOCKING_ENERGY_IDS for c in op_state.discard)
    # F2: Land Crush（ダメージ=無効エネ貫通）線を無効エネ運用デッキ全般に一般化
    land_crush_mode = op_articuno or (F2_BLOCKED_TURN and op_block_energy_seen)
    # F7: Raging Curse被弾モデル（相手場のCynthia'sのダメカン合計×10が被弾予測）
    spiritomb_on_field = any(p.id == CYNTHIA_SPIRITOMB for p in op_all_pokemon)
    raging_threat = 0
    if spiritomb_on_field:
        for p in op_all_pokemon:
            if p.id in CYNTHIA_IDS:
                raging_threat += max(0, p.maxHp - p.hp)
    spiritomb_danger = F7_SPIRITOMB_MODEL and spiritomb_on_field and raging_threat >= 140

    def veil_immune(pokemon: Pokemon) -> bool:
        # PH = damage counters (attack effect): blocked entirely for basic TR mons
        # while Repelling Veil (TR Articuno) is in play. Unfixable.
        return op_articuno and is_tr_basic(pokemon.id)

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
    active_has_psychic = has_psychic_energy(active_pokemon) if active_pokemon else False

    # ---- Opponent active info ----
    op_active = op_state.active[0] if op_state.active else None
    op_active_hp = op_active.hp if op_active else 9999
    op_active_blocked = op_active is not None and (veil_immune(op_active) or blocking_count(op_active) > 0)

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
        if (hand_counts[Fezandipiti_ex] > 0 and bench_free > 0
                and field_counts[Fezandipiti_ex] == 0 and not fez_suppress):
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
    candidates = []

    if state.turn >= 2 and op_active is not None:
        # F2: Super Psy Bolt はダメージ（Mist/Veil貫通）→ 無効アクティブ相手なら
        # Alakazamが居てもKadabraフィニッシュを許可（045は not blocked 条件で封じていた）
        kadabra_finish_ok = (
            (not op_active_blocked or F2_BLOCKED_TURN)
            and (field_counts[Alakazam] == 0 or (F2_BLOCKED_TURN and op_active_blocked))
        )
        if (op_active_hp <= 30 and kadabra_finish_ok
                and (field_counts[Kadabra] >= 1 or active_id == Kadabra)):
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

            for oi, pkmn in all_op:
                pz = prize_count(pkmn)
                eff_max_dmg = max_damage
                hm_need = 0
                if veil_immune(pkmn):
                    eff_max_dmg = 0        # 直せない（R21）
                else:
                    blockers = blocking_count(pkmn)  # F3: Mist + Rock Fighting
                    if blockers > 0:
                        if hand_counts[Enhanced_Hammer] >= blockers:
                            hm_need = blockers
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
                # F5: 対タンクは3przバースト志向 — 残prz>2の間は「Bossで吊る1przドリップ」
                # を選ばない（多プライズ対象 or 吊り不要のアクティブキルを優先）
                if F5_BURST_TARGETING and op_tank and my_prize_count > 2:
                    pref = [c for c in killable if c[2] >= 2 or c[0] == 0]
                    if pref:
                        killable = pref
                if killable:
                    # R13: 同点はプライズ→HPで選ぶ（相手Fez ex 210/2枚が自然に最上位になる）
                    best = max(killable, key=lambda x: (x[2], x[1].hp))
                    target_idx, target_pokemon, target_prize_gain, target_can_kill, target_hammer_needed = best
                    target_use_boss = target_idx != 0
                else:
                    target_idx = 0
                    target_pokemon = op_active
                    target_can_kill = False

    can_win_this_turn = target_can_kill and my_prize_count <= target_prize_gain

    # Boss が必要か: ベンチのキル対象を吊る（R13。アクティブが無効/倒せない時の回避を含む）
    need_boss = target_use_boss and target_can_kill

    # キルに追加ドローが必要か
    need_draw_for_kill = False
    if target_pokemon is not None and target_can_kill:
        current_dmg = (hand_size - target_hammer_needed) * 20
        if current_dmg < target_pokemon.hp:
            need_draw_for_kill = True

    # ---- R6: ドロー自制（手札が既に致死で山が薄いなら止める）+ 山切れガード ----
    kills_now = (target_pokemon is not None and target_can_kill and not need_draw_for_kill)
    if deck_count <= 4 and not need_draw_for_kill:
        draw_ok = False
    elif kills_now and hand_size >= 9 and deck_count <= 12:
        draw_ok = False
    else:
        draw_ok = deck_count >= 4 or can_win_this_turn

    # ---- F1/F2: 消耗戦の山経済（キル成立見込みターン以外は掘らない） ----
    conserve_deck = False
    if (F1_DECK_ECONOMY and op_tank and deck_count <= 20 and not can_win_this_turn):
        # F5: 掘ってよいのは「多プライズ対象を取り切れる見込み」か「掘らずに取れる」ターンのみ
        worthy = target_can_kill and (
            not F5_BURST_TARGETING or target_prize_gain >= 2 or kills_now)
        if not worthy:
            draw_ok = False
            conserve_deck = True
    if (F2_BLOCKED_TURN and op_active_blocked and not target_can_kill
            and deck_count <= 25):
        # 無効アクティブ+何も取れないターン: ドローを止めて山温存（空撃ちターンの丸損防止）
        draw_ok = False
        conserve_deck = True
    if conserve_deck and need_draw_for_kill:
        # 掘らない=このターンのキルは不成立 → Boss/Hammerも浪費しない
        need_boss = False
        target_hammer_needed = 0
        kills_now = False

    # ---- ベンチのアタッカー準備状況 ----
    ready_bench_alakazam = any(
        p.id == Alakazam and has_psychic_energy(p) for fi, p in my_field if fi != 0
    )
    bench_has_kadabra = any(p.id == Kadabra for fi, p in my_field if fi != 0)
    bench_kadabra_ready = bench_has_kadabra and hand_counts[Alakazam] >= 1 and (
        hand_counts[Basic_Psychic_Energy] + hand_counts[Telepath_Psychic_Energy] >= 1
        or any(p.id == Kadabra and has_psychic_energy(p) for fi, p in my_field if fi != 0)
    )
    # 対TR/無効エネ系: エネ3枚Dudunsparce = Land Crushアタッカー（R11、F2で一般化）
    ready_bench_dudun = land_crush_mode and any(
        p.id == Dudunsparce and len(p.energyCards) >= 3 for fi, p in my_field if fi != 0
    )
    active_is_attacker = (active_id == Alakazam and active_has_psychic) or (
        use_kadabra_finish and active_id == Kadabra and active_has_psychic) or (
        land_crush_mode and active_id == Dudunsparce and len(active_pokemon.energyCards) >= 3
        if active_pokemon is not None else False)
    bench_has_attacker = ready_bench_alakazam or bench_kadabra_ready or ready_bench_dudun or (
        use_kadabra_finish and bench_has_kadabra)

    # リトリートのためのエネルギーが必要か
    need_retreat_energy = False
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
            # 先攻・マリガン等は常にYES。ドロー能力のACTIVATEだけR6で拒否
            if context == SelectContext.ACTIVATE and not draw_ok:
                score = -1
            else:
                score = 1

        elif o.type == OptionType.NO:
            score = 0

        elif o.type in (OptionType.ENERGY, OptionType.ENERGY_CARD):
            # Enhanced Hammer のターゲット / 自分のリトリート費用（R17）
            pl = o.playerIndex if o.playerIndex is not None else my_index
            pk = get_card(obs, o.area, o.index, pl)
            eng = None
            if isinstance(pk, Pokemon):
                try:
                    eng = pk.energyCards[o.energyIndex]
                except (IndexError, TypeError):
                    eng = None
            if pk is None or eng is None:
                score = 1
            elif pl != my_index:
                # 相手のエネを剥がす: 無効エネ(Mist/Rock Fighting)最優先 > ミラーTelepath > 特殊
                if eng.id in BLOCKING_ENERGY_IDS:
                    score = 200
                elif pk.id == Alakazam:
                    score = 150
                elif pk.id in (Kadabra, Abra):
                    score = 140
                elif card_table.get(eng.id) is not None and card_table[eng.id].cardType == CardType.SPECIAL_ENERGY:
                    score = 80
                else:
                    score = 30
            else:
                # 自分のエネを捨てる（リトリート等）: P(Night Stretcherで回収可)を残す
                if eng.id == Telepath_Psychic_Energy:
                    score = 30
                elif eng.id == Enriching_Energy:
                    score = 25
                else:
                    score = 20
                if pk.id == Alakazam and has_psychic_energy(pk):
                    score -= 15  # 攻撃可能個体からは剥がさない

        elif o.type == OptionType.CARD:
            card = get_card(obs, o.area, o.index, o.playerIndex)
            if card is None:
                scores.append(score)
                continue
            energy_count = len(card.energies) if isinstance(card, Pokemon) else 0

            if context == SelectContext.SWITCH or context == SelectContext.TO_ACTIVE:
                if o.playerIndex == my_index:
                    # R24: このターン攻撃再開できる個体を昇格
                    if card.id == Alakazam:
                        score += (110 + energy_count * 10) if has_psychic_energy(card) else 60
                        if spiritomb_danger and has_psychic_energy(card):
                            score = 45  # F7: 即死予測ターンにエネ付きZamを差し出さない
                    elif card.id == Kadabra:
                        # F2: SPBはダメージ=無効貫通なので blocked チェック不要
                        if op_active_hp <= 30 and (not op_active_blocked or F2_BLOCKED_TURN):
                            score += 95
                        elif hand_counts[Alakazam] >= 1:
                            score += 90
                        else:
                            score += 30
                    elif card.id == Abra:
                        score += 50 if (hand_counts[Rare_Candy] >= 1 and hand_counts[Alakazam] >= 1) else 10
                    elif card.id == Dunsparce:
                        score += 125 if spiritomb_danger else 5  # F7: Dunsparce壁を差し出す
                    elif card.id == Dudunsparce:
                        score += 90 if (land_crush_mode and energy_count >= 3) else 3
                    elif card.id == Shaymin:
                        score += 2
                    else:
                        score += 1
                else:
                    if target_use_boss and target_pokemon is not None:
                        if o.index == target_idx - 1:
                            score += 100

            elif context == SelectContext.SETUP_ACTIVE_POKEMON:
                # R2: Abra > Dunsparce > Shaymin > Fez（本人140/75/21/14, pairwise Abra>Dun 43:0）
                if card.id == Abra:
                    score = 10
                elif card.id == Dunsparce:
                    score = 5
                elif card.id == Shaymin:
                    score = 3
                elif card.id == Fezandipiti_ex:
                    score = 2

            elif context == SelectContext.SETUP_BENCH_POKEMON:
                # R3: AbraとDunsparceだけ置く。Fez/Shayminはskip（本人skip29/15）。
                # ただし他にたねが無い時は置く（盤面0リスクの no-active 負け防止）
                if card.id == Abra:
                    score = 200
                elif card.id == Dunsparce:
                    score = 150
                elif hand_counts[Abra] + hand_counts[Dunsparce] == 0:
                    score = 2 if card.id == Shaymin else 1
                else:
                    score = -1

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
                    score += 45 if (field_counts[Fezandipiti_ex] == 0 and hand_counts[Fezandipiti_ex] == 0
                                    and not fez_suppress) else -60
                elif card.id == Enriching_Energy:
                    score += 65
                elif card.id in PSYCHIC_ENERGY_IDS:
                    score += 30 if not state.energyAttached else 10
                elif card.id == Rare_Candy:
                    score += 40 if field_counts[Abra] >= 1 else -10
                elif card.id == Shaymin:
                    score += 5 if field_counts[Shaymin] == 0 else -60
                elif card.id == Nighttime_Mine:
                    score -= 30

            elif context == SelectContext.ATTACH_FROM:
                if isinstance(card, Pokemon):
                    if need_retreat_energy and o.area == AreaType.ACTIVE:
                        score = 150
                    elif land_crush_mode and card.id == Dudunsparce and len(card.energyCards) < 3:
                        score = 120  # R11: Land Crush起動（F2で無効エネ系に一般化）
                    elif len(card.energyCards) >= 1:
                        score = -1  # 1体1枚（R7）
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
                # Poffin / Telepath のベンチ搬入: Abra > Dunsparce（R16）
                if card.id == Abra:
                    score = 100
                elif card.id == Dunsparce:
                    score = 80 if dunsparce_line_on_field < 2 else 40
                else:
                    score = 10

            elif context == SelectContext.TO_DECK or context == SelectContext.TO_DECK_BOTTOM:
                # Sacred Ash: Powerful Handライン優先で戻す（R18: Abra161/Zam160/Kadabra92）
                if card.id in ABRA_LINE:
                    score = 100
                elif card.id in DUNSPARCE_LINE:
                    score = 50
                else:
                    score = 10

            elif context == SelectContext.DISCARD:
                # 相手の手札破壊で自分の手札を捨てる順（R26、本人頻度実測）
                score = {
                    Hilda: 100,
                    Dawn: 96,
                    Nighttime_Mine: 94,
                    Boss_Orders: 90,
                    Buddy_Buddy_Poffin: 85,
                    Enhanced_Hammer: 80,
                    Rare_Candy: 75,
                    Poke_Pad: 72,
                    Kadabra: 60,
                    Telepath_Psychic_Energy: 55,
                    Xerosic: 50,
                    Shaymin: 48,
                    Sacred_Ash: 45,
                    Dunsparce: 40,
                    Dudunsparce: 38,
                    Basic_Psychic_Energy: 30,
                    Alakazam: 25,
                    Fezandipiti_ex: 15,
                    Enriching_Energy: 12,
                    Lanas_Aid: 10,
                    Abra: 8,
                    Night_Stretcher: 5,
                }.get(card.id, 55)
                if F8_BOSS_PRESERVE and op_tank and my_prize_count <= 2 and card.id == Boss_Orders:
                    # F8: 終盤クローズは常に「Boss+小PH」が勝ち筋 → Hammer/Xerosicより温存
                    score = 18
                # 複数持ちは捨てやすい（余剰Alakazam等）
                score += hand_counts.get(card.id, 0) * 8

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
                    if fez_suppress:
                        score = -1  # F6: 対Marnie/Cynthiaでは2プライズ献上になる
                    elif field_counts[Fezandipiti_ex] == 0 and bench_free >= 1:
                        score += 100
                    else:
                        score = -1

                elif card.id == Shaymin:
                    # Flower Curtain: ベンチ保護。余裕があれば置く（本人48回、t1でも7回）
                    if field_counts[Shaymin] == 0 and (bench_free >= 2 or len(my_field) <= 1):
                        score += 40
                    else:
                        score = -1

                if bench_free <= 1 and score > 0:
                    score -= 5000
                # 盤面が枯れかけなら、たねを何でも置く（no-active 負け防止）
                if len(my_field) <= 1 and data.basic and score < 0:
                    score = 20010

            else:
                score = 10000

                if card.id == Buddy_Buddy_Poffin:
                    if conserve_deck and len(my_field) >= 3:
                        score = -1  # F1: サーチも山を2枚焼く。盤面維持に必要な時以外は温存
                    elif state.turn <= 2:
                        score = 18000 if (abra_line_on_field < 3 or dunsparce_line_on_field < 1) else 8000
                    elif abra_line_on_field < 4 or dunsparce_line_on_field < 2:
                        score = 15000
                    elif bench_free >= 2:
                        score = 8000
                    else:
                        score = -1

                elif card.id == Poke_Pad:
                    if conserve_deck:
                        score = -1  # F1: 山温存
                    else:
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
                    # R17: PH対象の無効エネ / ミラーの相手Telepath / その他特殊エネ
                    if target_hammer_needed > 0:
                        score = 6500
                    elif F4_HAMMER_ECONOMY and op_tank:
                        # F4: 対タンクは「剥がす→同ターンキル」成立時のみ。Grow/Spikyへは温存
                        score = -1
                    else:
                        opp_mist = any(blocking_count(p) > 0 for p in op_all_pokemon)
                        opp_abra_special = any(
                            p.id in ABRA_LINE and any(
                                card_table.get(ec.id) is not None
                                and card_table[ec.id].cardType == CardType.SPECIAL_ENERGY
                                for ec in p.energyCards)
                            for p in op_all_pokemon)
                        any_special = any(
                            any(card_table.get(ec.id) is not None and card_table[ec.id].cardType == CardType.SPECIAL_ENERGY
                                for ec in p.energyCards)
                            for p in op_all_pokemon
                        )
                        if opp_mist:
                            score = 6400
                        elif opp_abra_special:
                            score = 6200
                        elif any_special:
                            score = 5000
                        else:
                            score = -1

                elif card.id == Boss_Orders:
                    score = 3400 if need_boss else -1

                elif card.id == Xerosic:
                    # R12: 相手手札>=7で最優先級。<=3では絶対撃たない
                    if op_state.handCount >= 7:
                        score = 3350
                    elif op_state.handCount >= 4:
                        score = 2900
                    else:
                        score = -1

                elif card.id == Lanas_Aid:
                    recoverable = (discard_counts[Basic_Psychic_Energy]
                                   + sum(discard_counts[c] for c in
                                         (Abra, Kadabra, Alakazam, Dunsparce, Dudunsparce)))
                    score = 3150 if recoverable >= 2 else -1

                elif card.id == Dawn:
                    if hand_counts[Alakazam] == 0 or hand_counts[Kadabra] == 0:
                        score = 3100
                    else:
                        score = 2950

                elif card.id == Hilda:
                    energy_in_hand = (hand_counts[Basic_Psychic_Energy]
                                      + hand_counts[Telepath_Psychic_Energy]
                                      + hand_counts[Enriching_Energy])
                    score = 3050 if energy_in_hand == 0 else 3000

                elif card.id == Nighttime_Mine:
                    # R19: 相手スタジアムのバウンスが主用途。ミラーでは手札温存
                    if stadium_id == Nighttime_Mine and stadium_is_mine:
                        score = -1
                    elif stadium_id != 0 and not stadium_is_mine:
                        score = 7000
                    elif stadium_id == 0 and op_has_tera:
                        score = 6800
                    else:
                        score = -1

        elif o.type == OptionType.ATTACH:
            card = get_card(obs, AreaType.HAND, o.index, my_index)
            pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
            if card is None or pokemon is None:
                scores.append(score)
                continue

            if card.id in PSYCHIC_ENERGY_IDS:
                if need_retreat_energy and o.inPlayArea == AreaType.ACTIVE:
                    score = 9500
                elif land_crush_mode and pokemon.id == Dudunsparce and len(pokemon.energyCards) < 3:
                    score = 7800  # R11: Land Crush起動（F2で無効エネ系に一般化）
                elif pokemon.id in ABRA_LINE and not has_psychic_energy(pokemon):
                    # P未装備のアタッカーには（Enriching汚染でも）Pを貼る
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
                    score = 10

            elif card.id == Enriching_Energy:
                # R9: ドロー4は貼り先不問で早撃ち（山札ガードのみ）。
                # ただしAbraラインは{P}を別途要するため優先はDunsparceライン/Fez
                if deck_count < 7:
                    score = -1
                elif land_crush_mode and pokemon.id == Dudunsparce and len(pokemon.energyCards) < 3:
                    score = 8600
                elif len(pokemon.energyCards) >= 1:
                    score = -1
                elif pokemon.id in DUNSPARCE_LINE:
                    score = 8500
                    if pokemon.id == Dudunsparce:
                        score += 10
                elif pokemon.id == Fezandipiti_ex:
                    score = 8300
                elif pokemon.id in ABRA_LINE:
                    score = 8200
                elif pokemon.id == Shaymin:
                    score = 8000
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
                if (draw_ok or (need_draw_for_kill and deck_count >= 4)) and not conserve_deck:
                    score = 30000
                else:
                    score = -1
            elif card.id == Fezandipiti_ex:
                if (draw_ok or (need_draw_for_kill and deck_count >= 4)) and not conserve_deck:
                    score = 29000
                else:
                    score = -1
            else:
                score = 28000  # 相手スタジアムの起動効果等

        elif o.type == OptionType.RETREAT:
            if active_is_attacker:
                score = -1
            elif use_kadabra_finish and active_id != Kadabra and bench_has_kadabra:
                score = 2500
            elif active_id in (Abra, Dunsparce, Dudunsparce, Fezandipiti_ex, Kadabra, Alakazam, Shaymin):
                # R25: ベンチに攻撃再開手段がある時だけ下がる
                score = 2000 if bench_has_attacker else -1
            else:
                score = -1

        elif o.type == OptionType.ATTACK:
            # R20: PHは撃てるなら毎ターン撃つ。R21: 無効対象には迂回打点を優先
            # F7: Spiritombが相手盤面にいる間、Cynthia'sアクティブへの非キルチップは
            # Raging Curseの燃料になるため撃たない（R23の補完）
            chip_guard = (F7_SPIRITOMB_MODEL and spiritomb_on_field
                          and op_active is not None and op_active.id in CYNTHIA_IDS)
            score = 1000
            if o.attackId == ATTACK_POWERFUL_HAND:
                if chip_guard and hand_size * 20 < op_active_hp:
                    score = -1
                else:
                    score += 500 if not op_active_blocked else 60
            elif o.attackId == ATTACK_LAND_CRUSH:
                if chip_guard and op_active_hp > 90:
                    score = -1
                else:
                    score += 400  # ダメージ攻撃: Mist/Veil素通し
            elif o.attackId == ATTACK_SUPER_PSY_BOLT:
                if chip_guard and op_active_hp > 30:
                    score = -1
                else:
                    score += 600 if op_active_hp <= 30 else 100
            elif o.attackId == ATTACK_TELEPORTATION:
                if chip_guard and op_active_hp > 10:
                    # 自己スイッチ用途のみ残す（燃料10は許容）。用途が無ければ撃たない
                    score = 1110 if bench_has_attacker else 5
                else:
                    score += 200 if bench_has_attacker else 50
            elif o.attackId == ATTACK_TRADING_PLACES:
                score += 150 if bench_has_attacker else 20
            elif o.attackId == ATTACK_SMASH_KICK:
                if chip_guard and op_active_hp > 30:
                    score = -1
                else:
                    score += 90 if op_active_hp <= 30 else 30
            elif o.attackId == ATTACK_RAM:
                if chip_guard and op_active_hp > 20:
                    score = -1
                else:
                    score += 25
            elif o.attackId == ATTACK_CRUEL_ARROW:
                score += 110

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

    # score<0 の選択肢はスキップ（minCountまではやむなく補充）
    chosen = [i for i in desc_indices if scores[i] >= 0][:select.maxCount]
    if len(chosen) < select.minCount:
        for i in desc_indices:
            if i not in chosen:
                chosen.append(i)
            if len(chosen) >= select.minCount:
                break
    return chosen


def _policy_scores(obs: Observation) -> list[float]:
    """Root candidate ranking for the search (reuses policy ordering)."""
    order = policy(obs)
    n = len(obs.select.option)
    sc = [0.0] * n
    for rank, idx in enumerate(order):
        sc[idx] = float(n - rank)
    return sc


# ---------------------------------------------------------------------------
# Lethal-only determinized search layer (MAIN only), 036/043 pattern.
# Runs only when our remaining prizes are <=3. A candidate overrides the
# heuristic choice only if EVERY determinization sample ends the greedy rollout
# of our own turn with the game already won. No board evaluation involved.

SEARCH_CANDIDATES = 8
SEARCH_SAMPLES = 8           # Alakazamのlethal線はドロー順依存が強い（Poffin/Pad/進化
                             # ドロー/Fez/Dudunsparce）: Marnie(043)と同じ8で偽陽性を抑える
SEARCH_MOVE_BUDGET = 1.5     # seconds per MAIN decision
SEARCH_GAME_BUDGET = 450.0   # total seconds of search per game (600s limit)
ROLLOUT_STEP_CAP = 80

PLACEHOLDER_MON = 1072
if not (card_table.get(PLACEHOLDER_MON) and card_table[PLACEHOLDER_MON].basic):
    PLACEHOLDER_MON = next(c.cardId for c in all_card if c.basic)

my_deck_counts = Counter(my_deck)

_search_time_used = 0.0


def _fallback_selection(select) -> list[int]:
    count = max(select.minCount, min(1, select.maxCount))
    return list(range(count))


def _my_visible_counts(obs: Observation) -> Counter:
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
    while len(unknown) < need:  # counting slack: pad with basic Psychic energy
        unknown.append(Basic_Psychic_Energy)
    your_deck = unknown[: me.deckCount]
    your_prize = unknown[me.deckCount : me.deckCount + len(me.prize)]

    opponent_deck = [PLACEHOLDER_MON] * op.deckCount
    opponent_prize = [PLACEHOLDER_MON] * len(op.prize)
    opponent_hand = [Basic_Psychic_Energy] * op.handCount
    opponent_active = []
    if op.active and op.active[0] is None:
        opponent_active = [PLACEHOLDER_MON]
    return your_deck, your_prize, opponent_deck, opponent_prize, opponent_hand, opponent_active


def _rollout(state, root_turn: int, deadline: float):
    for _ in range(ROLLOUT_STEP_CAP):
        obs = state.observation
        if obs.current.result != -1 or obs.current.turn != root_turn:
            break
        if time.perf_counter() > deadline:
            break
        try:
            sel = policy(obs)
            if not sel and obs.select.minCount > 0:
                sel = _fallback_selection(obs.select)
        except Exception:
            sel = _fallback_selection(obs.select)
        state = search_step(state.searchId, sel)
    return state.observation


def _snapshot_globals():
    return (pre_turn, ability_used_dudunsparce, ability_used_fezandipiti)


def _restore_globals(snap):
    global pre_turn, ability_used_dudunsparce, ability_used_fezandipiti
    pre_turn, ability_used_dudunsparce, ability_used_fezandipiti = snap


def _lethal_search(obs: Observation) -> list[int] | None:
    global _search_time_used

    select = obs.select
    if select.minCount != 1 or select.maxCount != 1 or len(select.option) < 2:
        return None

    my_index = obs.current.yourIndex
    me = obs.current.players[my_index]
    if len(me.prize) > 3:
        return None

    if obs.current.turn <= 2:
        _search_time_used = 0.0
    if _search_time_used > SEARCH_GAME_BUDGET:
        return None

    t0 = time.perf_counter()
    deadline = t0 + SEARCH_MOVE_BUDGET
    root_turn = obs.current.turn

    snap = _snapshot_globals()
    heuristic_scores = _policy_scores(obs)
    _restore_globals(snap)
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
                    _restore_globals(snap)  # ロールアウト間の状態リーク防止（EXP-043）
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
        _restore_globals(snap)
        _search_time_used += time.perf_counter() - t0

    winners = [i for i in candidates if always_wins[i] and tried[i] == SEARCH_SAMPLES]
    if not winners:
        return None
    return [max(winners, key=lambda i: heuristic_scores[i])]


def agent(obs_dict: dict) -> list[int]:
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        return my_deck

    if obs.select.context == SelectContext.MAIN and obs.current.turn >= 2:
        try:
            choice = _lethal_search(obs)
        except Exception:
            choice = None
        if choice is not None:
            return choice

    return policy(obs)
