"""052_alakazam_mimic — 051 + 不一致マイニング由来の4ルール (I-101 / EXP-052)。

ベース: 051_alakazam_deck（=045方策 + deck_v3[Xerosic4/Mine2] + time_usedリセット）。
knowledge/study/agreement_045_vs_81f1.md の不一致クラスタC1/C2/C4/discardを4ルール化:

- RULE_A (kills_nowゲート、C1本命): 現手札から攻撃セットアップ費用（進化/エネ/Boss/
  Hammer）を引いた残り×20が選定ターゲットの残HP以上なら「攻撃前にやらない」行動を
  抑制して即攻撃シーケンスへ。無効対象（Mist/Veil/Rock Fighting）やBossが撃てない
  ベンチ標的では発動しない。検証セットの両方向マイニングで許可/抑制を較正:
  許可=Kadabra/Dudunsparce進化+進化ドロー・Dunsparce設置・Hammer・Xerosic（マージン≥1）、
  抑制=Dudunsparce/Fez能力・Poffin/Pad・Alakazamベンチ進化・Abra設置・Dawn/Hilda・
  余分なエネ貼り・リトリート。
- RULE_B (Fez/Shaymin温存、C2): Fezは 15<=deck<=38 & draw_ok & hand<=6 のときのみ
  ベンチへ（本人: 展開394 vs 045願望1,794、置く時deck中央値24/turn6）。Shayminは原則
  手札温存（盤面枯渇の生存ガードは残す）。※vs035 Starmie対面の主な退行源（EXP-052）
- RULE_C (to_hand優先、C4): deck>=33（序盤）は攻撃ライン Kadabra(95)/Alakazam(88) >
  Dudunsparce(45)、中盤以降は045原表（本人のKadabra採取はdeck中央37、Dudunは28）。
  エネはTelepath>Enriching。
- RULE_D (Xerosic被弾discard表): 実測傾向スコア（捨て頻度/デッキ枚数）で再構築。
  ※studyの「Hilda/Dawn温存」仮説は両方向実測で反証（本人もHilda/Dawnから捨てる。
  045原表との主差分はAlakazam=捨てる寄り、Abra=最も残す）
- RULE_E (ターン内順序、C3・開発中に追加): Hammer→装填(アクティブ)→進化→Rare Candy→
  アイテムの順（045はアイテム全消化→進化）。Rare Candy直行時はKadabra進化を保留。
- RULE_F (ピボット昇格、C5・開発中に追加): SWITCH時のみ Dunsparce（Trading Placesで
  無償帰還）とエネ無しAlakazamの昇格値を引き上げ。TO_ACTIVE込みは逆効果（検証済み）。

フラグ: 環境変数 F052_OFF=A,B,C,D,E,F / F052_ON=... で個別ON/OFF（デフォルト全ON）。
開発KPI: tools/agreement.py の非強制一致率（検証07-10）: ベース051=56.4% → 65.0%。
アリーナ: vs051直接1200戦57.2%、vs047 52.0%（同日051基準36.75%、+15.3pt）、
プールv3 weighted 58.3% [55.2-61.4] validated（EXP-052）。
deck.csv は051採用のdeck_v3（Xerosic4/Mine2）のまま。Tool Scrapper/Battle Cageの
プレイ条件コードはデッキに無いため不活性のまま残置。

--- 以下 045 のドキュメント ---
045_alakazam_full — LBトップ Yushin Ito の Alakazam個体 (sig 81f1758c92,
本人250戦 WR76.4% = ladder史上最高, 対Alakazamミラー82.2%) のフル精読逆設計
+ 036式 lethal限定探索 (I-081 / EXP-045)。

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

# Deck-sweep tech cards (inert unless present in deck.csv variant)
Tool_Scrapper = 1137     # V1: 相手ツール(Hero's Cape等)剥がし
Battle_Cage = 1264       # V4: スタジアム（ベンチへのダメカン配置を両者防止）

# Opponent-side cards we must model
Mist_Energy = 11
Rock_Fighting_Energy = 20  # 効果無効（Mist同型、Garchomp系。EXP-048 F3で確認）
TR_Articuno = 414        # Repelling Veil: counters/effects blocked on basic TR mons

# ---- EXP-052 rule flags (default all ON; F052_OFF=A,B,C,D で個別無効化) ----
_F052_ON = set(x for x in (os.environ.get("F052_ON") or "").split(",") if x)
_F052_OFF = set(x for x in (os.environ.get("F052_OFF") or "").split(",") if x)


def _flag052(name: str, default: bool = True) -> bool:
    if name in _F052_OFF:
        return False
    if name in _F052_ON:
        return True
    return default


RULE_A = _flag052("A")   # kills_nowゲート（C1: lethal-now即攻撃）
RULE_B = _flag052("B")   # Fez/Shaymin温存（C2）
RULE_C = _flag052("C")   # to_hand攻撃ライン優先（C4）
RULE_D = _flag052("D")   # Xerosic被弾discard表の実測再構築
RULE_E = _flag052("E")   # ターン内順序: Xerosic/Hammer/装填/進化→アイテム（C3、開発中に追加）
RULE_F = _flag052("F")   # ピボット昇格: Dunsparce/エネ無しAlakazamの昇格（C5、開発中に追加）

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


def mist_count(pokemon: Pokemon) -> int:
    return sum(1 for ec in pokemon.energyCards if ec.id == Mist_Energy)


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

    op_tool_count = sum(len(p.tools) for p in op_all_pokemon)
    op_articuno = any(p.id == TR_Articuno for p in op_all_pokemon)
    op_has_tera = any(card_table.get(p.id) is not None and card_table[p.id].tera for p in op_all_pokemon)
    op_is_alakazam_mirror = any(p.id in ABRA_LINE for p in op_all_pokemon)

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
    op_active_blocked = op_active is not None and (veil_immune(op_active) or mist_count(op_active) > 0)

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
        if (op_active_hp <= 30 and not op_active_blocked
                and (field_counts[Kadabra] >= 1 or active_id == Kadabra) and field_counts[Alakazam] == 0):
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
                eff_max_dmg = max_damage
                hm_need = 0
                if veil_immune(pkmn):
                    eff_max_dmg = 0        # 直せない（R21）
                else:
                    mists = mist_count(pkmn)
                    if mists > 0:
                        if hand_counts[Enhanced_Hammer] >= mists:
                            hm_need = mists
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

    # ---- RULE_A (EXP-052/C1): kills_nowゲート ----
    # 現手札から攻撃セットアップ費用（進化/エネ貼り/Boss/必要Hammer）を引いた残り×20が
    # 選定ターゲットの残HP以上なら、ドロー・展開を全て止めて即攻撃する。
    # 手札は打点そのもの（プレイ=減点20/枚）、ドローは山切れの時計、ベンチ増はBoss標的。
    lethal_gate = False
    gate_need_attach = False       # アクティブへのエネ貼りが攻撃に必要
    gate_evolve_active = 0         # アクティブに進化が必要（card_id、0=不要）
    gate_use_candy = False         # アクティブAbra→(Rare Candy)→Alakazamが必要
    gate_margin = 0                # 致死を保ったまま追加で切れる手札枚数
    if (RULE_A and state.turn >= 2 and target_pokemon is not None and target_can_kill
            and active_pokemon is not None):
        hand_psy = hand_counts[Basic_Psychic_Energy] + hand_counts[Telepath_Psychic_Energy]
        can_attach = not state.energyAttached
        setup = None  # 攻撃者をアクティブで今ターン完成させる最小手札コスト
        if use_kadabra_finish:
            # Super Psy Bolt 30 (op_active.hp<=30) — 打点は手札数に依存しない
            if active_id == Kadabra:
                if active_has_psychic:
                    setup = 0
                elif can_attach and hand_psy >= 1:
                    setup, gate_need_attach = 1, True
            elif active_id == Abra and hand_counts[Kadabra] >= 1:
                if active_has_psychic:
                    setup, gate_evolve_active = 1, Kadabra
                elif can_attach and hand_psy >= 1:
                    setup, gate_evolve_active, gate_need_attach = 2, Kadabra, True
        else:
            # Powerful Hand
            if active_id == Alakazam:
                if active_has_psychic:
                    setup = 0
                elif can_attach and hand_psy >= 1:
                    setup, gate_need_attach = 1, True
            elif active_id == Kadabra and hand_counts[Alakazam] >= 1:
                if active_has_psychic:
                    setup, gate_evolve_active = 1, Alakazam
                elif can_attach and hand_psy >= 1:
                    setup, gate_evolve_active, gate_need_attach = 2, Alakazam, True
            elif (active_id == Abra and hand_counts[Rare_Candy] >= 1
                  and hand_counts[Alakazam] >= 1):
                if active_has_psychic:
                    setup, gate_use_candy = 2, True
                elif can_attach and hand_psy >= 1:
                    setup, gate_use_candy, gate_need_attach = 3, True, True
        if setup is not None and target_use_boss and (
                hand_counts[Boss_Orders] < 1 or state.supporterPlayed):
            setup = None  # Bossが撃てないならベンチ標的の即殺は成立しない
        if setup is not None:
            cost = setup + target_hammer_needed + (1 if target_use_boss else 0)
            if use_kadabra_finish:
                # 成立条件（hp<=30, not blocked）はターゲット選定側で検証済み
                lethal_gate = True
                gate_margin = 99  # 固定打点30なので手札プレイは打点を落とさない
            else:
                mists_left = mist_count(target_pokemon) - target_hammer_needed
                rock = any(ec.id == Rock_Fighting_Energy for ec in target_pokemon.energyCards)
                blocked_ok = (not veil_immune(target_pokemon)) and mists_left <= 0 and not rock
                if blocked_ok and (hand_size - cost) * 20 >= target_pokemon.hp:
                    lethal_gate = True
                    gate_margin = ((hand_size - cost) * 20 - target_pokemon.hp) // 20

    # ---- R6: ドロー自制（手札が既に致死で山が薄いなら止める）+ 山切れガード ----
    # 注: lethal_gateはdraw_okに触れない（本人は攻撃ターンでも進化ドローはYES、検証x763）。
    # ゲートのドロー抑制はMAINのDudunsparce/Fez能力のみ（下の抑制フィルタ）
    kills_now = (target_pokemon is not None and target_can_kill and not need_draw_for_kill)
    if deck_count <= 4 and not need_draw_for_kill:
        draw_ok = False
    elif kills_now and hand_size >= 9 and deck_count <= 12:
        draw_ok = False
    else:
        draw_ok = deck_count >= 4 or can_win_this_turn

    # ---- ベンチのアタッカー準備状況 ----
    ready_bench_alakazam = any(
        p.id == Alakazam and has_psychic_energy(p) for fi, p in my_field if fi != 0
    )
    bench_has_kadabra = any(p.id == Kadabra for fi, p in my_field if fi != 0)
    bench_kadabra_ready = bench_has_kadabra and hand_counts[Alakazam] >= 1 and (
        hand_counts[Basic_Psychic_Energy] + hand_counts[Telepath_Psychic_Energy] >= 1
        or any(p.id == Kadabra and has_psychic_energy(p) for fi, p in my_field if fi != 0)
    )
    # 対TR: エネ3枚Dudunsparce = Land Crushアタッカー（R11）
    ready_bench_dudun = op_articuno and any(
        p.id == Dudunsparce and len(p.energyCards) >= 3 for fi, p in my_field if fi != 0
    )
    active_is_attacker = (active_id == Alakazam and active_has_psychic) or (
        use_kadabra_finish and active_id == Kadabra and active_has_psychic) or (
        op_articuno and active_id == Dudunsparce and len(active_pokemon.energyCards) >= 3
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
                # 相手のエネを剥がす: Mist最優先 > AbraラインのTelepath(ミラー) > その他特殊
                if eng.id == Mist_Energy:
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
                    _is_switch = context == SelectContext.SWITCH
                    if card.id == Alakazam:
                        # RULE_F: SWITCH時はエネ無しでも昇格して次ターン再開（本人x52）
                        _zam_lo = 92 if (RULE_F and _is_switch) else 60
                        score += (110 + energy_count * 10) if has_psychic_energy(card) else _zam_lo
                    elif card.id == Kadabra:
                        if op_active_hp <= 30 and not op_active_blocked:
                            score += 95
                        elif hand_counts[Alakazam] >= 1:
                            score += 90
                        else:
                            score += 30
                    elif card.id == Abra:
                        score += 50 if (hand_counts[Rare_Candy] >= 1 and hand_counts[Alakazam] >= 1) else 10
                    elif card.id == Dunsparce:
                        # RULE_F (C5): Trading Placesで無償で戻れるピボット（本人昇格x160。
                        # SWITCH限定: TO_ACTIVE込みだとto_active-87の副作用、検証済み）
                        score += 86 if (RULE_F and _is_switch) else 5
                    elif card.id == Dudunsparce:
                        score += 90 if (op_articuno and energy_count >= 3) else 3
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
                _early_c = RULE_C and deck_count >= 33
                if card.id == Dudunsparce:
                    # RULE_C (C4): 序盤（deck>=33）は攻撃ライン優先・ドローエンジン格下げ。
                    # 中盤以降は本人もDudunsparceを取る（本人: Kadabra採取deck中央37/Dudun 28）
                    score += (45 if _early_c else 80) if (field_counts[Dunsparce] >= 1 and field_counts[Dudunsparce] == 0) else -50
                elif card.id == Kadabra:
                    score += (95 if _early_c else 70) if field_counts[Abra] >= 1 else -20
                elif card.id == Alakazam:
                    score += (88 if _early_c else 60) if (field_counts[Kadabra] >= 1 or field_counts[Abra] >= 1) else -20
                elif card.id == Abra:
                    score += 50 if abra_line_on_field < 3 else -50
                elif card.id == Dunsparce:
                    score += 40 if dunsparce_line_on_field < 2 else -50
                elif card.id == Fezandipiti_ex:
                    score += 45 if (field_counts[Fezandipiti_ex] == 0 and hand_counts[Fezandipiti_ex] == 0) else -60
                elif card.id == Enriching_Energy:
                    score += 42 if RULE_C else 65  # RULE_C: Telepath > Enriching（×28）
                elif card.id in PSYCHIC_ENERGY_IDS:
                    if RULE_C:
                        score += 48 if not state.energyAttached else 25
                    else:
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
                    elif op_articuno and card.id == Dudunsparce and len(card.energyCards) < 3:
                        score = 120  # R11: Land Crush起動
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
                if RULE_D:
                    # 本人の実選択の傾向スコア（=検証セット捨て頻度/デッキ枚数）順で再構築。
                    # studyの(d)仮説「Hilda/Dawnを残しAlakazamやAshを捨てる」は両方向実測で反証:
                    # 実際の傾向は Mine330>Shaymin184>Hilda168>Boss154>Dawn137>Candy117>
                    # Ash112>Poffin110>Hammer99>Alakazam97>...>Abra11（Abraは最も残す）
                    score = {
                        Nighttime_Mine: 100,
                        Shaymin: 92,
                        Hilda: 88,
                        Boss_Orders: 84,
                        Dawn: 80,
                        Rare_Candy: 76,
                        Sacred_Ash: 72,
                        Buddy_Buddy_Poffin: 68,
                        Enhanced_Hammer: 64,
                        Alakazam: 60,
                        Basic_Psychic_Energy: 56,
                        Xerosic: 50,
                        Fezandipiti_ex: 46,
                        Poke_Pad: 44,
                        Kadabra: 42,
                        Telepath_Psychic_Energy: 40,
                        Dudunsparce: 38,
                        Dunsparce: 34,
                        Enriching_Energy: 30,
                        Night_Stretcher: 26,
                        Lanas_Aid: 22,
                        Abra: 8,
                    }.get(card.id, 55)
                else:
                    # 旧R26（045の頻度実測ベース）
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
                # 複数持ちは捨てやすい（余剰Alakazam等）
                score += hand_counts.get(card.id, 0) * 8

        elif o.type == OptionType.TOOL_CARD:
            # Tool Scrapper: 相手のツールだけ剥がす（自デッキにツール無し、038と同実装）
            if o.playerIndex != my_index:
                score = 300
            else:
                score = -200

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
                    if field_counts[Fezandipiti_ex] != 0 or bench_free < 1:
                        score = -1
                    elif RULE_B:
                        # C2: 原則手札温存（=PH打点+2プライズBoss標的の回避）。
                        # 設置帯はdeck<=38（序盤の即置き禁止）+ドローが必要なとき（hand<=6）
                        if draw_ok and 15 <= deck_count <= 38 and hand_size <= 6:
                            score += 100
                        else:
                            score = -1
                    else:
                        score += 100

                elif card.id == Shaymin:
                    # Flower Curtain: ベンチ保護。余裕があれば置く（本人48回、t1でも7回）
                    if RULE_B:
                        score = -1  # C2: 原則手札温存（本人118 vs 045願望960）。盤面枯渇時は下の生存ガードが復活させる
                    elif field_counts[Shaymin] == 0 and (bench_free >= 2 or len(my_field) <= 1):
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
                    # R17: PH対象のMist / ミラーの相手Telepath / その他特殊エネ
                    # RULE_E: 本人はHammerを進化より先に撃つ（153:6）
                    _hm_hi = 16400 if RULE_E else 6500
                    if target_hammer_needed > 0:
                        score = _hm_hi
                    else:
                        opp_mist = any(mist_count(p) > 0 for p in op_all_pokemon)
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
                            score = _hm_hi - 20
                        elif opp_abra_special:
                            score = _hm_hi - 40
                        elif any_special:
                            score = 5000
                        else:
                            score = -1

                elif card.id == Boss_Orders:
                    score = 3400 if need_boss else -1

                elif card.id == Xerosic:
                    # R12: 相手手札>=7で最優先級。<=3では絶対撃たない
                    # 注: 進化より先への引き上げ(16450)は検証で逆効果（evolve-397>supporter+224）
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

                elif card.id == Tool_Scrapper:
                    # V1: 相手の場にツールがあるときだけ剥がす（Cape/Power Weight/Gravity Gemstone）
                    score = 12000 if op_tool_count >= 1 else -1

                elif card.id == Battle_Cage:
                    # V4: 自分のスタジアムが無ければ設置（b7cef: 176回/240戦、相手スタジアム上書き含む）
                    score = -1 if stadium_is_mine else 9000

        elif o.type == OptionType.ATTACH:
            card = get_card(obs, AreaType.HAND, o.index, my_index)
            pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
            if card is None or pokemon is None:
                scores.append(score)
                continue

            if card.id in PSYCHIC_ENERGY_IDS:
                if need_retreat_energy and o.inPlayArea == AreaType.ACTIVE:
                    score = 9500
                elif op_articuno and pokemon.id == Dudunsparce and len(pokemon.energyCards) < 3:
                    score = 7800  # R11: Land Crush起動（対TRのみ）
                elif pokemon.id in ABRA_LINE and not has_psychic_energy(pokemon):
                    # P未装備のアタッカーには（Enriching汚染でも）Pを貼る
                    score = 8000 if len(pokemon.energyCards) == 0 else 7900
                    if RULE_E and o.inPlayArea == AreaType.ACTIVE:
                        # 本人はアクティブのアタッカー装填をアイテム・進化より先に行う（302:78）
                        score = 16300
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
                elif op_articuno and pokemon.id == Dudunsparce and len(pokemon.energyCards) < 3:
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
            # RULE_E (C3): 本人は「進化→アイテム→攻撃」。045の「アイテム全消化→進化」を反転
            score = 16200 if RULE_E else 9000
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
                if RULE_E and hand_counts[Rare_Candy] > 0 and hand_counts[Alakazam] > 0:
                    # 本人はRare Candy直行（Abra→Alakazam）を優先しKadabra進化を保留（x120）
                    score = 15500
                else:
                    score += 100
                    if len(pokemon.energies) == 0:
                        score += 50
                    else:
                        score -= 20
                        if hand_counts[Rare_Candy] > 0 and hand_counts[Alakazam] > 0:
                            score -= 100  # エネ付きAbraはRare Candy直行を温存

            elif card.id == Dudunsparce:
                # 注: Dudunsparce最優先へのフリップは検証で逆効果（新誤りx338>旧x210）→維持
                score += 80

        elif o.type == OptionType.ABILITY:
            card = get_card(obs, o.area, o.index, my_index)
            if card is None:
                scores.append(score)
                continue

            if card.id == Dudunsparce:
                if draw_ok or (need_draw_for_kill and deck_count >= 4):
                    score = 30000
                else:
                    score = -1
            elif card.id == Fezandipiti_ex:
                if draw_ok or (need_draw_for_kill and deck_count >= 4):
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
            score = 1000
            if o.attackId == ATTACK_POWERFUL_HAND:
                score += 500 if not op_active_blocked else 60
            elif o.attackId == ATTACK_LAND_CRUSH:
                score += 400  # ダメージ攻撃: Mist/Veil素通し
            elif o.attackId == ATTACK_SUPER_PSY_BOLT:
                score += 600 if op_active_hp <= 30 else 100
            elif o.attackId == ATTACK_TELEPORTATION:
                score += 200 if bench_has_attacker else 50
            elif o.attackId == ATTACK_TRADING_PLACES:
                score += 150 if bench_has_attacker else 20
            elif o.attackId == ATTACK_SMASH_KICK:
                score += 90 if op_active_hp <= 30 else 30
            elif o.attackId == ATTACK_RAM:
                score += 25
            elif o.attackId == ATTACK_CRUEL_ARROW:
                score += 110

        # ---- RULE_A: ゲート発火中は「攻撃前にやらない」行動を抑制（即攻撃シーケンスへ） ----
        # 検証セットの両方向実測（98%のゲートターンで本人も同ターン内にPH攻撃）に基づく許可/抑制:
        #   許可: Kadabra/Dudunsparce進化+進化ドロー（本人does204/384 vs skip56/90）、
        #         Dunsparce設置（does326 vs skip152、マージン≥1）、Hammer（does118 vs skip88）、
        #         Xerosic（does100 vs skip100、マージン≥1）、Abraラインへのエネ貼り（does~190）
        #   抑制: Dudunsparce/Fez能力（skip540 vs use193）、Poffin（skip263 vs 153）、
        #         Pad（skip434 vs 152）、Alakazamベンチ進化（skip204 vs does118）、
        #         Abra設置（skip117 vs does78）、Dawn/Hilda/Stretcher/Ash/Mine、リトリート
        if lethal_gate and score >= 0:
            if o.type == OptionType.ABILITY:
                card = get_card(obs, o.area, o.index, my_index)
                if card is not None and card.id in (Dudunsparce, Fezandipiti_ex):
                    score = -1  # ドロー能力は山切れの時計
            elif o.type == OptionType.RETREAT:
                score = -1  # アクティブが攻撃者（ゲート成立条件）
            elif o.type == OptionType.PLAY:
                card = get_card(obs, AreaType.HAND, o.index, my_index)
                if card is not None:
                    cid = card.id
                    data = card_table[cid]
                    if data.cardType == CardType.POKEMON:
                        # 非exピボットのDunsparceのみマージン内で置く。盤面枯渇時は生存優先
                        if (cid != Dunsparce or gate_margin < 1) and len(my_field) > 1:
                            score = -1
                    elif cid == Boss_Orders and need_boss:
                        pass  # 攻撃に必要（コストはgateで計上済み）
                    elif cid == Enhanced_Hammer and (target_hammer_needed > 0 or gate_margin >= 1):
                        pass  # 必要分 or マージン内の追加剥がし（本人x118）
                    elif cid == Rare_Candy and gate_use_candy:
                        pass  # 攻撃に必要
                    elif cid == Xerosic and gate_margin >= 1 and not need_boss:
                        pass  # マージン内の妨害は本人も半々で撃つ（Boss必要時は枠を譲る）
                    else:
                        score = -1  # Poffin/Pad/Dawn/Hilda/Stretcher/Ash/Mine等の展開停止
            elif o.type == OptionType.ATTACH:
                card = get_card(obs, AreaType.HAND, o.index, my_index)
                pk_g = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
                if (gate_need_attach and o.inPlayArea == AreaType.ACTIVE
                        and card is not None and card.id in PSYCHIC_ENERGY_IDS):
                    score = 16500  # 攻撃に必要な1枚は進化より先に貼る（本人の順序、x123）
                elif (gate_margin >= 1 and card is not None and card.id in PSYCHIC_ENERGY_IDS
                      and isinstance(pk_g, Pokemon) and pk_g.id in ABRA_LINE
                      and not has_psychic_energy(pk_g) and score >= 0):
                    pass  # マージン内の将来アタッカー装填は本人もやる（x190）
                else:
                    score = -1  # それ以外のエネ貼りは手札-1
            elif o.type == OptionType.EVOLVE:
                card = get_card(obs, AreaType.HAND, o.index, my_index)
                if card is not None and card.id == Alakazam:
                    need = (gate_evolve_active == Alakazam) or gate_use_candy
                    if not (need and o.inPlayArea == AreaType.ACTIVE):
                        score = -1  # Alakazam進化は手中のPH打点を盤面に変える=攻撃ターンは温存
                # Kadabra/Dudunsparce進化は許可（本人は攻撃前に済ませる）

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
    global _search_time_used
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        return my_deck

    if obs.current is not None and obs.current.turn <= 2:
        _search_time_used = 0.0   # per-game budget reset (045の潜在バグ修正、050と同じ)

    if obs.select.context == SelectContext.MAIN and obs.current.turn >= 2:
        try:
            choice = _lethal_search(obs)
        except Exception:
            choice = None
        if choice is not None:
            return choice

    return policy(obs)
