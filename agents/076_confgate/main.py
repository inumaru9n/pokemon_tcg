"""076_confgate — 058 + MLレイヤの信頼度ゲート委譲 (EXP-076, アーキテクチャ比較A)。

ベース: 058_alakazam_policyml2 の完全コピー（モデル model_058.* も同一、再学習なし）。
変更点は _ml_layer() のみ:
- GBTのクラス確率を選択可能クラスにマスクして正規化し、その最大値を確信度とする
- 確信度 < TAU_076 なら None を返して 052 条件文policy に委譲（F054_LINESはoffのため直行）
- TAU_076 は環境変数 F076_TAU で上書き可（τスイープ用）。F076_NORM=off でマスク正規化なし
  の生softmax最大値に切替。F076_STATS=<path> で委譲率カウンタをファイルに定期ダンプ
- npzモード（Kaggle）は生logitのためsoftmaxを追加。lgbモード（ローカル）はpredictが
  softmax済み確率を返すのでそのまま使う（argmaxは従来どおり生スコアで両モード不変）

--- 以下 058(=054ファイル系譜) のドキュメント ---
054_alakazam_lines — 052 + 同ターン内の決定的ライン列挙層 (I-104 / EXP-054)。

ベース: 052_alakazam_mimic（下記ドキュメント）。F054_LINES=off で052と同一挙動。

LINES層（MAIN・turn>=2・minCount==maxCount==1、lethal探索不発時のみ）:
- 「このターン中に到達可能なターン終了時状態」を抽象状態のBFSで決定的に列挙
  （ロールアウトなし=ノイズゼロ、相手モデル不要=決定化バイアスなし。EXP-053の帰結）
- ライン=行動の集合+攻撃選択。正規化状態でdedupe（順序等価の圧縮）
- 終端状態を経済スコア（線形特徴: プライズ/チップダメ/手札=PH弾薬/山消費/盤面価値/
  Xerosic/Hammer/次ターン攻撃態勢）で採点し、最良ラインを選択
- 実行順は052の較正済み順序を再利用: 最良ラインの行動集合にマッチする現在の選択肢の
  うちpolicyスコア最高のものを返す（RULE_Eの一致率資産を保持）
- 確率的ドローはフロンティア: ドロー枚数は確定値（PH打点は手札枚数のみ依存）、
  内容は「実行→次のMAIN決定で再列挙」で取り込む（receding horizon）
- Bossはターン終端の攻撃修飾（吊り先はラインが決定、SWITCHにヒントを渡す）
- フラグ: F054_LINES=off で層無効、F054_W_<NAME>=x で経済スコアの重み上書き、
  F054_CAP/F054_BUDGET で列挙上限

--- 以下 052 のドキュメント ---
052_alakazam_mimic — 051 + 不一致マイニング由来の4ルール (I-101 / EXP-052)。

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

# エージェントディレクトリの解決（3段構え: __file__ / kaggle固定パス / cwd）
# Kaggle評価環境はexecロードのため __file__ が無い（EXP-007の教訓）
try:
    _AGENT_DIR_057 = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _AGENT_DIR_057 = ("/kaggle_simulations/agent"
                      if os.path.exists("/kaggle_simulations/agent/main.py")
                      else os.getcwd())
import sys as _sys057

if _AGENT_DIR_057 not in _sys057.path:
    _sys057.path.insert(0, _AGENT_DIR_057)

# Load deck.csv (3段構え: __file__ / cwd / kaggle固定パス)
file_path = os.path.join(_AGENT_DIR_057, "deck.csv")
if not os.path.exists(file_path):
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
RULE_M2 = _flag052("M2", default=False)  # EXP-056の未検証ルール。057ベースは純052相当にするためoff

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

# ---- EXP-056: 攻撃テーブル（EN_Card_Data.csv由来、自動生成モジュール） ----
try:
    from attack_table_056 import ATTACKS_056
except ImportError:
    import importlib.util as _ilu056

    _p056 = os.path.join(_AGENT_DIR_057, "attack_table_056.py")
    _spec056 = _ilu056.spec_from_file_location("attack_table_056", _p056)
    _m056 = _ilu056.module_from_spec(_spec056)
    _spec056.loader.exec_module(_m056)
    ATTACKS_056 = _m056.ATTACKS_056

# ---- EXP-054 LINES layer globals ----
_lines_boss_bench = None   # 選択したラインのBoss吊り先（相手ベンチindex）。SWITCHで参照
_lines_gate_off = False    # LINES層のランキング用policy呼び出し中はRULE_Aゲートを無効化
_policy_raw_scores = None  # policy()が直近に計算した選択肢別の生スコア


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

    global pre_turn, ability_used_dudunsparce, ability_used_fezandipiti, _lines_boss_bench
    if pre_turn != state.turn:
        pre_turn = state.turn
        ability_used_dudunsparce = False
        ability_used_fezandipiti = False
        _lines_boss_bench = None

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

    # RULE_M2 (EXP-056マイニング): 場のAlakazamが攻撃準備済み({P}装備) かつ 手札4-6枚の
    # ターンは、本人は追加のエネ手貼りをしない（val精度72-89%、n=274/1877ターン）。
    # 経済: エネは山に6枚のみ、毎ターン落ちるアタッカーへの「オンデマンド供給」が正
    m2_hold = (RULE_M2 and 4 <= hand_size <= 6
               and any(p is not None and p.id == Alakazam and has_psychic_energy(p)
                       for _a, p in my_field))

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
    if (RULE_A and not _lines_gate_off and state.turn >= 2
            and target_pokemon is not None and target_can_kill
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
                    if _lines_boss_bench is not None:
                        # LINES層が選んだBoss吊り先を最優先（EXP-054）
                        if o.index == _lines_boss_bench:
                            score += 120
                    elif target_use_boss and target_pokemon is not None:
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
                elif m2_hold and card.id == Basic_Psychic_Energy:
                    score = -1  # RULE_M2: エネ温存（Telepathはサーチ側効果があるため対象外）
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

    global _policy_raw_scores
    _policy_raw_scores = scores  # EXP-054: LINES層のマッチ選択肢ランキング用に生スコアを公開

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
    return (pre_turn, ability_used_dudunsparce, ability_used_fezandipiti, _lines_boss_bench)


def _restore_globals(snap):
    global pre_turn, ability_used_dudunsparce, ability_used_fezandipiti, _lines_boss_bench
    pre_turn, ability_used_dudunsparce, ability_used_fezandipiti, _lines_boss_bench = snap


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


# =========================================================================
# EXP-054: LINES layer — 同ターン内の決定的ライン列挙 (I-104)
#
# MAIN決定ごとに「このターン中に到達可能なターン終了時状態」を抽象状態のBFSで
# 決定的に列挙し、経済スコアで採点して最良ラインを選ぶ。実行は最良ラインの行動
# 集合にマッチする現在の選択肢のうちpolicy(052)スコア最高のもの（順序資産を保持）。
# ドロー内容は「実行→次のMAIN決定で再列挙」で取り込む（receding horizon）。
# =========================================================================

# EXP-056: LINES層はデフォルト無効（056=052相当policy+マイニングルールの位置づけ）
F054_LINES = os.environ.get("F054_LINES", "off") != "off"
LINES_STATE_CAP = int(os.environ.get("F054_CAP", "6000"))
LINES_MOVE_BUDGET = float(os.environ.get("F054_BUDGET", "0.9"))
LINES_GAME_BUDGET = float(os.environ.get("F054_GAME_BUDGET", "250"))

_W054 = {
    # 終局経済
    "win": 1e9,          # このターンで勝ち切り
    "prize": 900.0,      # 取得プライズ1枚
    "chip": 3.0,         # 倒せない相手に残すダメージ（10ダメ単位）
    "hand": 25.0,        # 終端手札1枚（=次PH+20打点+資源）
    "burn": -5.0,        # 山消費1枚
    "lowdeck": -80.0,    # 終端deck<=2
    "middeck": -25.0,    # 終端deck<=6
    "deckout": -300000.0,  # 終端deck==0（次の自ドローで負け。勝ち切り時は不問）
    # 盤面価値（ポケモン基礎値+攻撃準備ボーナス）
    "v_abra": 30.0, "v_kadabra": 42.0, "v_alakazam": 50.0,
    "v_dunsparce": 14.0, "v_dudunsparce": 26.0, "v_fez": 20.0, "v_shaymin": 10.0,
    "b_alakazam": 45.0,  # Alakazam+{P}（攻撃可能）
    "b_kadabra": 12.0, "b_abra": 4.0,
    "b_dudun3": 25.0,    # 対Articuno時のエネ3 Dudunsparce（Land Crush線）
    "fez_bench": -55.0,  # Fezを場に出す=2プライズBoss標的（RULE_B相当）
    "shaymin_bench": -20.0,
    # 妨害・その他
    "xero": 22.0,        # 相手手札を1枚削る
    "hammer": 15.0,      # 相手特殊エネ1枚剥がす
    "mine": 8.0,         # スタジアム提出の価値
    "attack": 40.0,      # 攻撃して番を終える（vs 素END）
    "next_ready": 25.0,  # 終端アクティブが次ターン攻撃可能
    # 行動バイアス（ライン内で行動を1回使うごとの加点。経済で写せない本人の性向ダイヤル）
    "a_dud": 0.0, "a_fez": 0.0, "a_evo_kad": 0.0, "a_evo_zam": 0.0, "a_candy": 0.0,
    "a_evo_dud": 0.0, "a_attach": 0.0, "a_enrich": 0.0, "a_poffin": 0.0, "a_pad": 0.0,
    "a_stretcher": 0.0, "a_ash": 0.0, "a_hammer": 0.0, "a_mine": 0.0, "a_abra": 0.0,
    "a_dunsparce": 0.0, "a_fezmon": 0.0, "a_shaymon": 0.0, "a_dawn": 0.0, "a_hilda": 0.0,
    "a_lana": 0.0, "a_xero": 0.0, "a_retreat": 0.0, "a_boss": 0.0,
    # 盤面絶対量（EXP-055: V=MLP用の状態特徴。線形スコアでは重み0=不使用）
    "s_turn": 0.0, "s_prz_me": 0.0, "s_prz_op": 0.0, "s_deck": 0.0, "s_opph": 0.0,
    "s_my_mons": 0.0, "s_my_psy": 0.0, "s_my_energy": 0.0,
    "s_ready3": 0.0, "s_ready2": 0.0, "s_my_act_hp": 0.0, "s_asleep": 0.0,
    "s_opp_mons": 0.0, "s_opp_hp": 0.0, "s_opp_act_hp": 0.0,
    "s_opp_ex": 0.0, "s_opp_energy": 0.0,
    # 探索相関の計算述語（EXP-056: 1-ply脅威/圧力+ターン内文脈+相手モデル。線形では不使用）
    "t_op_maxdmg": 0.0, "t_op_kill": 0.0, "t_op_var": 0.0, "t_op_przrisk": 0.0,
    "t_my_ph_next": 0.0, "t_my_kill_next": 0.0,
    "c_sup": 0.0, "c_eatt": 0.0, "o_mega": 0.0,
}
# 較正v3確定値（dev 07-09座標降下 65.52%。val 07-10は63.90%<052ベース64.95%で非転移 — EXP-054）
# "k@early"/"k@late"（deck>=30 / <=14）/"k@race"（自プライズ<=2）は実効重みへの加算デルタ
_W054.update({
    "hand": 8.0, "hand@early": 5.0, "hand@late": 5.0, "hand@race": -5.0,
    "burn": -4.0, "burn@early": 12.0, "burn@late": -12.0, "burn@race": 3.0,
    "chip": 12.0, "chip@early": 5.0, "chip@late": -10.0,
    "attack": 15.0, "attack@early": -25.0,
    "prize": 900.0, "prize@early": -500.0, "prize@late": 600.0,
    "xero": 25.0, "xero@early": -25.0, "xero@late": -10.0,
    "hammer": 0.0, "mine": 8.0,
    "next_ready": 25.0, "next_ready@early": -20.0,
    "lowdeck": -30.0, "middeck": -10.0,
    "v_abra": 30.0, "v_abra@early": 10.0, "v_abra@late": -10.0,
    "v_kadabra": 85.0, "v_kadabra@early": -40.0, "v_kadabra@late": -40.0,
    "v_alakazam": 10.0,
    "v_dunsparce": 14.0,
    "v_dudunsparce": 60.0, "v_dudunsparce@early": -40.0, "v_dudunsparce@late": -20.0,
    "v_fez": 55.0, "v_shaymin": 10.0,
    "b_alakazam": 45.0, "b_alakazam@early": 40.0,
    "b_kadabra": 12.0, "b_abra": 10.0, "b_dudun3": 50.0,
    "fez_bench": -55.0, "shaymin_bench": -40.0,
})
# EXP-056: 手札内容特徴（自デッキ60枚固定=外挿なし）。h_<cardId>=終端手札のそのカードの
# 枚数、h_unk=このライン中の未知ドロー枚数。線形では重み0=不使用、マイニング/V用
for _cid in sorted(my_deck_counts):
    _W054[f"h_{_cid}"] = 0.0
_W054["h_unk"] = 0.0
for _k in list(_W054):
    _env_v = os.environ.get("F054_W_" + _k.upper().replace("@", "_AT_"))
    if _env_v is not None:
        _W054[_k] = float(_env_v)

# ---- EXP-055: 価値ネット V(終端盤面)=勝率。value_055.npz があれば線形スコアを置換 ----
_V055 = None
if os.environ.get("F055_VALUE", "on") != "off":
    try:
        import numpy as _np055

        _v_path = os.path.join(_AGENT_DIR_057, "value_055.npz")
        if os.path.exists(_v_path):
            _z = _np055.load(_v_path, allow_pickle=False)
            _V055 = {
                "np": _np055,
                "keys": [k.decode() if isinstance(k, bytes) else str(k)
                         for k in _z["feat_keys"]],
                "mu": _z["mu"].astype("float32"),
                "sd": _z["sd"].astype("float32"),
                "Ws": [_z[f"W{i}"].astype("float32")
                       for i in range(int(_z["n_layers"]))],
                "bs": [_z[f"b{i}"].astype("float32")
                       for i in range(int(_z["n_layers"]))],
            }
            _V055["ki"] = {k: i for i, k in enumerate(_V055["keys"])}
    except Exception:  # noqa: BLE001  numpy無し/破損npz → 線形フォールバック
        _V055 = None


def _v055_scores(feats: list[dict]):
    """終端feat dictのバッチをV=MLPで採点してスコア配列を返す。"""
    np_ = _V055["np"]
    ki = _V055["ki"]
    X = np_.zeros((len(feats), len(ki)), dtype="float32")
    for r, f in enumerate(feats):
        for k, v in f.items():
            j = ki.get(k)
            if j is not None:
                X[r, j] = v
    X = (X - _V055["mu"]) / _V055["sd"]
    for li, (W, b) in enumerate(zip(_V055["Ws"], _V055["bs"])):
        X = X @ W + b
        if li < len(_V055["Ws"]) - 1:
            np_.maximum(X, 0.0, out=X)
    return X[:, 0]


# ライン内実行順序のクラス優先度（大きいほど先に実行。0なら052のpolicyスコア順のみ）
_O054 = {k: 0.0 for k in
         ("a_dud", "a_fez", "a_evo_kad", "a_evo_zam", "a_candy", "a_evo_dud",
          "a_attach", "a_enrich", "a_poffin", "a_pad", "a_stretcher", "a_ash",
          "a_hammer", "a_mine", "a_abra", "a_dunsparce", "a_fezmon", "a_shaymon",
          "a_dawn", "a_hilda", "a_lana", "a_xero", "a_retreat", "a_boss")}
# 較正v3確定値（fit_v3の順序優先度。大きいほどライン内で先に実行）
_O054.update({
    "a_abra": 1.0, "a_ash": -3.0, "a_boss": -3.0, "a_candy": 1.0, "a_dawn": -2.0,
    "a_evo_zam": 1.0, "a_fez": 1.0, "a_hammer": 2.0, "a_lana": -3.0,
    "a_retreat": -1.0, "a_stretcher": -3.0,
})
for _k in list(_O054):
    _env_v = os.environ.get("F054_O_" + _k.upper())
    if _env_v is not None:
        _O054[_k] = float(_env_v)


def _tag_bias(tag) -> str | None:
    """抽象行動タグ→行動バイアス/順序クラスのキー。"""
    t = tag[0]
    if t == "ABILITY":
        return "a_dud" if tag[1] == Dudunsparce else "a_fez"
    if t == "EVOLVE":
        return {Kadabra: "a_evo_kad", Alakazam: "a_evo_zam",
                Dudunsparce: "a_evo_dud"}.get(tag[1])
    if t == "ATTACH":
        return "a_enrich" if tag[1] == Enriching_Energy else "a_attach"
    if t == "RETREAT":
        return "a_retreat"
    if t == "PLAY":
        return {Rare_Candy: "a_candy", Buddy_Buddy_Poffin: "a_poffin",
                Poke_Pad: "a_pad", Night_Stretcher: "a_stretcher",
                Sacred_Ash: "a_ash", Enhanced_Hammer: "a_hammer",
                Nighttime_Mine: "a_mine", Abra: "a_abra", Dunsparce: "a_dunsparce",
                Fezandipiti_ex: "a_fezmon", Shaymin: "a_shaymon", Dawn: "a_dawn",
                Hilda: "a_hilda", Lanas_Aid: "a_lana", Xerosic: "a_xero",
                Boss_Orders: "a_boss"}.get(tag[1])
    return None

_lines_stats = {"calls": 0, "fired": 0, "fallback": 0, "states": 0,
                "time": 0.0, "capped": 0, "boss": 0}
_lines_debug = None  # 較正モードの直近呼び出しの選択肢メタ（calib_054.py用）

_EVO_PRE = {Kadabra: Abra, Alakazam: Kadabra, Dudunsparce: Dunsparce}
_EVO_DRAW = {Kadabra: 2, Alakazam: 3, Dudunsparce: 0}
_POKE_IDS = {Abra, Kadabra, Alakazam, Dunsparce, Dudunsparce, Fezandipiti_ex, Shaymin}
_NONRULE_POKE = {Abra, Kadabra, Alakazam, Dunsparce, Dudunsparce, Shaymin}
# flagsビット
_F_EATT, _F_SUP, _F_RET, _F_MINE, _F_FEZOK = 1, 2, 4, 8, 16


def _score_feat(feat: dict) -> float:
    W = _W054
    return sum(W[k] * v for k, v in feat.items())


def _lines_layer(obs: Observation, _collect: list | None = None) -> list[int] | None:
    """最良ラインの次アクションのoption indexを返す。不発/マッピング失敗はNone。

    _collect が list のとき（較正モード）: 全終端の (feat, pred_option_idx) を
    _collect に追記して None を返す（副作用なし。pred=-1はマッピング失敗=policy委譲）。
    """
    global _search_time_used, _lines_boss_bench
    global ability_used_dudunsparce, ability_used_fezandipiti

    select = obs.select
    if select.minCount != 1 or select.maxCount != 1 or (
            len(select.option) < 2 and _collect is None):
        return None  # 較正/収集モードは強制END等（選択肢1）でも終端特徴を出す
    state = obs.current
    my_index = state.yourIndex
    me = state.players[my_index]
    op = state.players[1 - my_index]
    if not me.active or me.active[0] is None or me.hand is None:
        return None
    op_active = op.active[0] if op.active else None
    if op_active is None:
        return None
    if _search_time_used > 400.0:
        return None  # 時間予算枯渇 → 052挙動へ
    t0 = time.perf_counter()
    deadline = t0 + (LINES_MOVE_BUDGET if _search_time_used <= LINES_GAME_BUDGET else 0.25)
    _lines_boss_bench = None
    _lines_stats["calls"] += 1

    # ---- 静的コンテキスト ----
    op_field = [p for p in (op.active or []) if p is not None] + \
               [p for p in op.bench if p is not None]
    op_articuno = any(p.id == TR_Articuno for p in op_field)
    opps = []      # (id, hp, prize, veil)
    ospec0 = []    # (mist, rock, other_special) 可変（Hammer）
    oeng_norm = []  # 通常エネ枚数（静的）
    op_bench_map = []  # opps index-1 -> 相手ベンチ実index（Boss/SWITCH用）

    def _add_opp(p):
        veil = op_articuno and is_tr_basic(p.id)
        mist = sum(1 for ec in p.energyCards if ec.id == Mist_Energy)
        rock = sum(1 for ec in p.energyCards if ec.id == Rock_Fighting_Energy)
        spec = sum(1 for ec in p.energyCards
                   if card_table.get(ec.id) is not None
                   and card_table[ec.id].cardType == CardType.SPECIAL_ENERGY) - mist - rock
        opps.append((p.id, p.hp, prize_count(p), veil))
        ospec0.append((mist, rock, max(0, spec)))
        oeng_norm.append(len(p.energyCards) - mist - rock - max(0, spec))

    _add_opp(op_active)
    for bi, bp in enumerate(op.bench):
        if bp is not None:
            op_bench_map.append(bi)
            _add_opp(bp)
    spec_sum0 = sum(a + b + c for (a, b, c) in ospec0)
    # EXP-055 静的コンテキスト（s_*特徴用）
    turn_no = state.turn
    op_prizes = len(op.prize)
    my_act_hp0 = me.active[0].hp
    opp_hp_sum0 = sum(hp for (_i, hp, _p, _v) in opps)
    opp_ex_cnt0 = sum(1 for (_i, _h, prz, _v) in opps if prz >= 2)
    # EXP-056 静的コンテキスト: 相手の次ターン脅威（1-ply、公開情報のみ）
    my_act_id0 = me.active[0].id
    op_dmg_next = []  # j -> (固定打点max, 可変攻撃フラグ: 2=相手PH, 1=その他可変)
    op_etype = []
    for j, (oid, _hp, _prz, _v) in enumerate(opps):
        e_next = oeng_norm[j] + sum(ospec0[j]) + 1  # 次ターン=手貼り1枚込み
        mx, var = 0, 0
        for (cn, dmg, mv) in ATTACKS_056.get(oid, ()):
            if cn > e_next:
                continue
            if dmg is None:
                var = 2 if mv == "Powerful Hand" else max(var, 1)
            elif dmg > mx:
                mx = dmg
        op_dmg_next.append((mx, var))
        cd_o = card_table.get(oid)
        op_etype.append(cd_o.energyType if cd_o is not None else -1)
    o_mega0 = sum(1 for (oid, _h, _p, _v) in opps
                  if card_table.get(oid) is not None and card_table[oid].megaEx)

    def _prz_of_id(aid):
        cd = card_table.get(aid)
        if cd is None:
            return 1
        return 3 if cd.megaEx else (2 if cd.ex else 1)

    my_prizes = len(me.prize)
    opph0 = op.handCount
    root_deck = me.deckCount
    bench_max = me.benchMax
    asleep_par = me.asleep or me.paralyzed

    # フェーズ/レース文脈で実効重みを構成（"k@early"/"k@late"/"k@race" は加算デルタ）
    phase = "early" if root_deck >= 30 else ("late" if root_deck <= 14 else "mid")
    race = my_prizes <= 2
    Weff = {k: v for k, v in _W054.items() if "@" not in k}
    for k, v in _W054.items():
        if "@" not in k:
            continue
        bk, cond = k.split("@", 1)
        if (cond == phase) or (cond == "race" and race):
            Weff[bk] = Weff.get(bk, 0.0) + v

    root_dud_opt = root_fez_opt = root_retreat = False
    for o in select.option:
        if o.type == OptionType.ABILITY:
            c = get_card(obs, o.area, o.index, my_index)
            if c is not None:
                if c.id == Dudunsparce:
                    root_dud_opt = True
                elif c.id == Fezandipiti_ex:
                    root_fez_opt = True
        elif o.type == OptionType.RETREAT:
            root_retreat = True

    stadium_id = 0
    stadium_is_mine = False
    for c in state.stadium:
        stadium_id = c.id
        stadium_is_mine = c.playerIndex == my_index
    op_has_tera = any(card_table.get(p.id) is not None and card_table[p.id].tera
                      for p in op_field)
    mine_ok = (stadium_id != 0 and not stadium_is_mine) or (stadium_id == 0 and op_has_tera)

    def _mk_mon(p):
        psy = sum(1 for ec in p.energyCards if ec.id in PSYCHIC_ENERGY_IDS)
        oth = len(p.energyCards) - psy
        return (p.id, psy, oth, 1 if p.appearThisTurn else 0)

    act0 = _mk_mon(me.active[0])
    bench0 = tuple(sorted(_mk_mon(p) for p in me.bench if p is not None))
    hand0 = tuple(sorted(c.id for c in me.hand))
    pool_c = my_deck_counts - _my_visible_counts(obs)   # 山+サイドの未視認プール
    pool0 = tuple(sorted(pool_c.items()))
    dis_c = Counter()
    for c in me.discard:
        if c.id in _POKE_IDS or c.id == Basic_Psychic_Energy:
            dis_c[c.id] += 1
    dis0 = tuple(sorted(dis_c.elements()))

    flags0 = ((_F_EATT if state.energyAttached else 0)
              | (_F_SUP if state.supporterPlayed else 0)
              | (0 if root_retreat else _F_RET)
              | (_F_FEZOK if root_fez_opt else 0))
    root = (hand0, 0, root_deck, act0, bench0, flags0, opph0,
            tuple(ospec0), dis0, pool0)

    # ---- ヘルパー ----
    _MON_FEAT = {Abra: ("v_abra", "b_abra"), Kadabra: ("v_kadabra", "b_kadabra"),
                 Alakazam: ("v_alakazam", "b_alakazam"), Dunsparce: ("v_dunsparce", None),
                 Dudunsparce: ("v_dudunsparce", None), Fezandipiti_ex: ("v_fez", None),
                 Shaymin: ("v_shaymin", None)}

    def add_mon_feat(f, m):
        mid, psy, oth, _fresh = m
        vk, bk = _MON_FEAT.get(mid, (None, None))
        if vk is None:
            return
        f[vk] = f.get(vk, 0) + 1
        if bk is not None and psy >= 1:
            f[bk] = f.get(bk, 0) + 1
        if mid == Dudunsparce and op_articuno and psy + oth >= 3:
            f["b_dudun3"] = f.get("b_dudun3", 0) + 1
        if mid == Fezandipiti_ex:
            f["fez_bench"] = f.get("fez_bench", 0) + 1
        if mid == Shaymin:
            f["shaymin_bench"] = f.get("shaymin_bench", 0) + 1

    def ready_rank(m):
        mid, psy, oth, _fresh = m
        if mid == Alakazam and psy >= 1:
            return 3
        if op_articuno and mid == Dudunsparce and psy + oth >= 3:
            return 3
        if mid == Kadabra and psy >= 1:
            return 2
        return 0

    def pref_fetch(source_counts, field_ids, allowed):
        """052のTO_HAND優先の近似で1枚選ぶ（source_counts: dict-like）"""
        order = []
        if field_ids.get(Abra, 0) >= 1:
            order.append(Kadabra)
        if field_ids.get(Kadabra, 0) >= 1 or field_ids.get(Abra, 0) >= 1:
            order.append(Alakazam)
        if field_ids.get(Dunsparce, 0) >= 1 and field_ids.get(Dudunsparce, 0) == 0:
            order.append(Dudunsparce)
        order += [Kadabra, Alakazam, Abra, Dunsparce, Dudunsparce, Shaymin]
        for cid in order:
            if cid in allowed and source_counts.get(cid, 0) >= 1:
                return cid
        return None

    def attacks_of(m):
        mid, psy, oth, _fresh = m
        tot = psy + oth
        res = []  # (attack_id, kind, dmg)
        if asleep_par:
            return res
        if mid == Alakazam and psy >= 1:
            res.append((ATTACK_POWERFUL_HAND, "ph", 0))
        if mid == Kadabra and psy >= 1:
            res.append((ATTACK_SUPER_PSY_BOLT, "dmg", 30))
        if mid == Abra and psy >= 1:
            res.append((ATTACK_TELEPORTATION, "pivot", 10))
        if mid == Dunsparce:
            if tot >= 1:
                res.append((ATTACK_TRADING_PLACES, "pivot", 0))
            if tot >= 2:
                res.append((ATTACK_RAM, "dmg", 20))
        if mid == Dudunsparce and tot >= 3:
            res.append((ATTACK_LAND_CRUSH, "dmg", 90))
        if mid == Shaymin and tot >= 2:
            res.append((ATTACK_SMASH_KICK, "dmg", 30))
        if mid == Fezandipiti_ex and tot >= 3:
            res.append((ATTACK_CRUEL_ARROW, "any", 100))
        return res

    # ---- 終端評価: (feat_dict, attack_id|None, boss_opps_j|None) ----
    def terminals(st):
        (hand, unk, deck, act, bench, flags, opph, ospec, dis, pool) = st
        hand_n = len(hand) + unk
        base = {}
        add_mon_feat(base, act)
        for m in bench:
            add_mon_feat(base, m)
        if opph0 - opph:
            base["xero"] = opph0 - opph
        hm = spec_sum0 - sum(a + b + c for (a, b, c) in ospec)
        if hm:
            base["hammer"] = hm
        if flags & _F_MINE:
            base["mine"] = 1
        if root_deck - deck:
            base["burn"] = root_deck - deck
        if deck <= 2:
            base["lowdeck"] = 1
        if deck <= 6:
            base["middeck"] = 1
        # EXP-055 盤面絶対量（st依存分）
        field_all = (act,) + tuple(bench)
        psy_tot = sum(m[1] for m in field_all)
        base["s_turn"] = turn_no
        base["s_prz_op"] = op_prizes
        base["s_deck"] = deck
        base["s_opph"] = opph
        base["s_my_mons"] = len(field_all)
        base["s_my_psy"] = psy_tot
        base["s_my_energy"] = psy_tot + sum(m[2] for m in field_all)
        base["s_ready3"] = sum(1 for m in field_all if ready_rank(m) == 3)
        base["s_ready2"] = sum(1 for m in field_all if ready_rank(m) == 2)
        base["s_my_act_hp"] = my_act_hp0
        if asleep_par:
            base["s_asleep"] = 1
        # EXP-056 ターン内文脈・相手モデル
        if flags & _F_SUP:
            base["c_sup"] = 1
        if flags & _F_EATT:
            base["c_eatt"] = 1
        if o_mega0:
            base["o_mega"] = o_mega0
        zam_ready = any(m[0] == Alakazam and m[1] >= 1 for m in field_all)
        # EXP-056: 手札内容（カード名別カウント。ラインの消費が正確に反映される）
        for _hc, _hn in Counter(hand).items():
            base[f"h_{_hc}"] = _hn
        if unk:
            base["h_unk"] = unk
        opp_energy_now = sum(oeng_norm) + sum(a + b + c for (a, b, c) in ospec)

        def threat_feats(f, hand_after, killed_j=None):
            """EXP-056: 1-ply脅威/圧力述語（opp_abs の後に呼ぶこと）"""
            aid = act[0]
            cd_a = card_table.get(aid)
            my_hp = my_act_hp0 if aid == my_act_id0 else \
                (cd_a.hp if cd_a is not None else 60)
            weak = cd_a.weakness if cd_a is not None else None
            mx, var = 0, 0
            for j2 in range(len(opps)):
                if killed_j is not None and j2 == killed_j:
                    continue
                m_j, v_j = op_dmg_next[j2]
                if v_j == 2:  # 相手のPowerful Hand（ミラー）: 手札+ドロー1枚換算
                    m_j = max(m_j, 20 * (opph + 1))
                elif v_j == 1:
                    var = 1
                if weak is not None and op_etype[j2] == weak:
                    m_j *= 2
                if m_j > mx:
                    mx = m_j
            f["t_op_maxdmg"] = min(mx, 990) // 10
            if mx >= my_hp > 0:
                f["t_op_kill"] = 1
            if var:
                f["t_op_var"] = 1
            f["t_op_przrisk"] = _prz_of_id(aid)
            ph_next = 20 * (hand_after + 1) if zam_ready else 0
            f["t_my_ph_next"] = min(ph_next, 990) // 10
            oah = f.get("s_opp_act_hp", 0)
            if oah > 0 and ph_next >= oah:
                f["t_my_kill_next"] = 1

        def opp_abs(f, j=None, killed=False, dmg=0):
            """相手側の絶対量（攻撃効果を反映）をfへ書き込む"""
            f["s_opp_mons"] = len(opps) - (1 if killed else 0)
            if killed:
                _oid, hp_j, prz_j, _v = opps[j]
                f["s_opp_hp"] = opp_hp_sum0 - hp_j
                f["s_opp_ex"] = opp_ex_cnt0 - (1 if prz_j >= 2 else 0)
                f["s_opp_energy"] = opp_energy_now - oeng_norm[j] - sum(ospec[j])
                f["s_opp_act_hp"] = 0 if j == 0 else opps[0][1]
            else:
                hit = min(dmg, opps[j][1]) if j is not None else 0
                f["s_opp_hp"] = opp_hp_sum0 - hit
                f["s_opp_ex"] = opp_ex_cnt0
                f["s_opp_energy"] = opp_energy_now
                f["s_opp_act_hp"] = (opps[0][1] - (hit if j == 0 else 0))

        out = []
        # END（攻撃なし）
        f = dict(base)
        f["hand"] = hand_n
        f["s_prz_me"] = my_prizes
        opp_abs(f)
        threat_feats(f, hand_n)
        if deck == 0:
            f["deckout"] = 1
        if ready_rank(act) > 0:
            f["next_ready"] = 1
        out.append((f, None, None))
        boss_ok = (not (flags & _F_SUP)) and (Boss_Orders in hand)
        for atk_id, kind, dmg0 in attacks_of(act):
            nr = 1 if ready_rank(act) > 0 else 0
            if kind == "pivot":
                nr = 1 if max((ready_rank(m) for m in bench), default=0) > 0 else 0
            targets = [(0, False)]
            if kind == "any":
                targets = [(j, False) for j in range(len(opps))]
            elif boss_ok:
                for j in range(1, len(opps)):
                    targets.append((j, True))
            for j, use_boss in targets:
                oid, hp, prz, veil = opps[j]
                mist, rock, _sp = ospec[j]
                hv = hand_n - (1 if use_boss else 0)
                if kind == "ph":
                    dmg = 0 if (veil or mist > 0 or rock > 0) else 20 * hv
                else:
                    dmg = dmg0  # ダメージ攻撃はMist/Veil素通し
                killed = dmg >= hp and dmg > 0
                if use_boss and not killed:
                    continue  # 吊ってまで倒せないラインは列挙しない
                prize_gain = prz if killed else 0
                win = killed and prize_gain >= my_prizes
                f = dict(base)
                f["hand"] = hv
                f["attack"] = 1
                f["s_prz_me"] = my_prizes - prize_gain
                opp_abs(f, j=j, killed=killed, dmg=dmg)
                threat_feats(f, hv, killed_j=(j if killed else None))
                if use_boss:  # Boss消費を手札内容に反映
                    _hb = f.get(f"h_{Boss_Orders}", 0) - 1
                    f[f"h_{Boss_Orders}"] = _hb if _hb > 0 else 0
                if nr:
                    f["next_ready"] = 1
                if prize_gain:
                    f["prize"] = prize_gain
                if win:
                    f["win"] = 1
                else:
                    if deck == 0:
                        f["deckout"] = 1
                    if not killed and dmg >= 10:
                        f["chip"] = dmg // 10
                out.append((f, atk_id, j if use_boss else None))
        return out

    # ---- 遷移生成: (tag, new_state) ----
    def succ(st):
        (hand, unk, deck, act, bench, flags, opph, ospec, dis, pool) = st
        out = []
        hset = set(hand)
        field = [act] + list(bench)
        fc = Counter(m[0] for m in field)
        bench_free = bench_max - len(bench)
        pool_d = dict(pool)

        def hrm(*cids):
            l = list(hand)
            for cid in cids:
                l.remove(cid)
            return tuple(sorted(l))

        def badd(b, m):
            return tuple(sorted(b + (m,)))

        def brm(b, m):
            l = list(b)
            l.remove(m)
            return tuple(l)

        def pool_rm(*cids):
            d = dict(pool_d)
            for cid in cids:
                d[cid] -= 1
                if d[cid] <= 0:
                    del d[cid]
            return tuple(sorted(d.items()))

        def dr(n):
            return n if deck - n >= 1 else 0

        # --- ドロー能力: Dudunsparce（本体+付属を山に戻す） ---
        for i, m in enumerate(field):
            if m[0] != Dudunsparce or not (root_dud_opt or m[3] == 1):
                continue
            d = dr(3)
            if d <= 0:
                continue
            nd = deck - d + 1 + m[1] + m[2]
            if i == 0:
                # アクティブ使用→ベンチから昇格（ready優先）
                if not bench:
                    continue
                promo = max(bench, key=lambda x: (ready_rank(x), x[0] == Alakazam,
                                                  x[0] == Kadabra, x[1] + x[2]))
                out.append((("ABILITY", Dudunsparce),
                            (hand, unk + d, nd, promo, brm(bench, promo), flags,
                             opph, ospec, dis, pool)))
            else:
                out.append((("ABILITY", Dudunsparce),
                            (hand, unk + d, nd, act, brm(bench, m), flags,
                             opph, ospec, dis, pool)))
        # --- ドロー能力: Fez（前相手番きぜつ条件はroot選択肢の存在で判定） ---
        if (flags & _F_FEZOK) and any(m[0] == Fezandipiti_ex for m in field):
            d = dr(3)
            if d > 0:
                out.append((("ABILITY", Fezandipiti_ex),
                            (hand, unk + d, deck - d, act, bench, flags & ~_F_FEZOK,
                             opph, ospec, dis, pool)))

        # --- 進化（進化ドロー込み） ---
        for evo in (Kadabra, Alakazam, Dudunsparce):
            if evo not in hset:
                continue
            pre = _EVO_PRE[evo]
            d = dr(_EVO_DRAW[evo])
            seen_t = set()
            for i, m in enumerate(field):
                if m[0] != pre or m[3] == 1 or (i > 0 and m in seen_t):
                    continue
                if i > 0:
                    seen_t.add(m)
                nm = (evo, m[1], m[2], 1)
                if i == 0:
                    ns = (hrm(evo), unk + d, deck - d, nm, bench, flags,
                          opph, ospec, dis, pool)
                    out.append((("EVOLVE", evo, "A"), ns))
                else:
                    ns = (hrm(evo), unk + d, deck - d, act, badd(brm(bench, m), nm),
                          flags, opph, ospec, dis, pool)
                    out.append((("EVOLVE", evo, "B"), ns))
        # --- Rare Candy: Abra→Alakazam ---
        if Rare_Candy in hset and Alakazam in hset:
            d = dr(3)
            seen_t = set()
            for i, m in enumerate(field):
                if m[0] != Abra or m[3] == 1 or (i > 0 and m in seen_t):
                    continue
                if i > 0:
                    seen_t.add(m)
                nm = (Alakazam, m[1], m[2], 1)
                if i == 0:
                    ns = (hrm(Rare_Candy, Alakazam), unk + d, deck - d, nm, bench,
                          flags, opph, ospec, dis, pool)
                else:
                    ns = (hrm(Rare_Candy, Alakazam), unk + d, deck - d, act,
                          badd(brm(bench, m), nm), flags, opph, ospec, dis, pool)
                out.append((("PLAY", Rare_Candy), ns))

        # --- エネ手貼り ---
        if not (flags & _F_EATT):
            for cid in (Telepath_Psychic_Energy, Basic_Psychic_Energy, Enriching_Energy):
                if cid not in hset:
                    continue
                if cid == Enriching_Energy and deck - 4 < 1:
                    continue
                seen_t = set()
                for i, m in enumerate(field):
                    if i > 0 and m in seen_t:
                        continue
                    if i > 0:
                        seen_t.add(m)
                    if cid == Enriching_Energy:
                        nm = (m[0], m[1], m[2] + 1, m[3])
                        d, nde = 4, deck - 4
                    else:
                        nm = (m[0], m[1] + 1, m[2], m[3])
                        d, nde = 0, deck
                    nb, na = bench, act
                    npool, nu = pool, unk + d
                    if i == 0:
                        na = nm
                    else:
                        nb = badd(brm(bench, m), nm)
                    # Telepath: {P}ポケモンに貼るとAbraを2枚までベンチ搬入
                    if cid == Telepath_Psychic_Energy and m[0] in ABRA_LINE:
                        free_now = bench_max - len(nb)
                        k = min(2, pool_d.get(Abra, 0), free_now)
                        if k > 0:
                            for _ in range(k):
                                nb = badd(nb, (Abra, 0, 0, 1))
                            npool = pool_rm(*([Abra] * k))
                            nde -= k
                    ns = (hrm(cid), nu, nde, na, nb, flags | _F_EATT,
                          opph, ospec, dis, npool)
                    out.append((("ATTACH", cid, m[0], "A" if i == 0 else "B"), ns))

        # --- たね設置 ---
        if bench_free > 0:
            for cid in (Abra, Dunsparce, Fezandipiti_ex, Shaymin):
                if cid in hset:
                    ns = (hrm(cid), unk, deck, act, badd(bench, (cid, 0, 0, 1)),
                          flags, opph, ospec, dis, pool)
                    out.append((("PLAY", cid), ns))

        # --- アイテム ---
        if Buddy_Buddy_Poffin in hset and bench_free > 0:
            picks = []
            avail = dict(pool_d)
            for _ in range(min(2, bench_free)):
                pick = None
                for cid in (Abra, Dunsparce):  # HP<=70のたね
                    if avail.get(cid, 0) >= 1:
                        pick = cid
                        break
                if pick is None:
                    break
                avail[pick] -= 1
                picks.append(pick)
            if picks:
                nb = bench
                for cid in picks:
                    nb = badd(nb, (cid, 0, 0, 1))
                ns = (hrm(Buddy_Buddy_Poffin), unk, deck - len(picks), act, nb,
                      flags, opph, ospec, dis, pool_rm(*picks))
                out.append((("PLAY", Buddy_Buddy_Poffin), ns))
        if Poke_Pad in hset:
            pick = pref_fetch(pool_d, fc, _NONRULE_POKE)
            if pick is not None:
                ns = (tuple(sorted(hrm(Poke_Pad) + (pick,))), unk, deck - 1, act,
                      bench, flags, opph, ospec, dis, pool_rm(pick))
                out.append((("PLAY", Poke_Pad), ns))
        if Night_Stretcher in hset and dis:
            dis_cnt = Counter(dis)
            pick = pref_fetch(dis_cnt, fc, _POKE_IDS)
            if pick is None and dis_cnt.get(Basic_Psychic_Energy, 0) >= 1:
                pick = Basic_Psychic_Energy
            if pick is not None:
                nd = list(dis)
                nd.remove(pick)
                ns = (tuple(sorted(hrm(Night_Stretcher) + (pick,))), unk, deck, act,
                      bench, flags, opph, ospec, tuple(nd), pool)
                out.append((("PLAY", Night_Stretcher), ns))
        if Sacred_Ash in hset:
            dis_poke = [cid for cid in dis if cid in _POKE_IDS]
            if dis_poke:
                back = sorted(dis_poke,
                              key=lambda c: (0 if c in ABRA_LINE else
                                             1 if c in DUNSPARCE_LINE else 2))[:5]
                nd = list(dis)
                for cid in back:
                    nd.remove(cid)
                ns = (hrm(Sacred_Ash), unk, deck + len(back), act, bench,
                      flags, opph, ospec, tuple(nd), pool)
                out.append((("PLAY", Sacred_Ash), ns))
        if Enhanced_Hammer in hset:
            seen_j = set()
            for j, (mist, rock, sp) in enumerate(ospec):
                if mist + rock + sp <= 0:
                    continue
                key = (opps[j][0], mist, rock, sp)
                if key in seen_j:
                    continue
                seen_j.add(key)
                if mist > 0:
                    nsj = (mist - 1, rock, sp)
                elif rock > 0:
                    nsj = (mist, rock - 1, sp)
                else:
                    nsj = (mist, rock, sp - 1)
                no = list(ospec)
                no[j] = nsj
                ns = (hrm(Enhanced_Hammer), unk, deck, act, bench, flags,
                      opph, tuple(no), dis, pool)
                out.append((("PLAY", Enhanced_Hammer), ns))
        if Nighttime_Mine in hset and mine_ok and not (flags & _F_MINE):
            ns = (hrm(Nighttime_Mine), unk, deck, act, bench, flags | _F_MINE,
                  opph, ospec, dis, pool)
            out.append((("PLAY", Nighttime_Mine), ns))

        # --- サポーター（1ターン1枚。Bossは終端修飾で扱う） ---
        if not (flags & _F_SUP):
            nflags = flags | _F_SUP
            if Xerosic in hset and opph >= 4:
                ns = (hrm(Xerosic), unk, deck, act, bench, nflags,
                      3, ospec, dis, pool)
                out.append((("PLAY", Xerosic), ns))
            if Dawn in hset:
                gets = []
                avail = dict(pool_d)
                for group in ((Abra, Dunsparce, Shaymin),
                              (Kadabra, Dudunsparce),
                              (Alakazam,)):
                    for cid in group:
                        if avail.get(cid, 0) >= 1:
                            avail[cid] -= 1
                            gets.append(cid)
                            break
                if gets:
                    ns = (tuple(sorted(hrm(Dawn) + tuple(gets))), unk,
                          deck - len(gets), act, bench, nflags,
                          opph, ospec, dis, pool_rm(*gets))
                    out.append((("PLAY", Dawn), ns))
            if Hilda in hset:
                gets = []
                evo = pref_fetch(pool_d, fc, {Kadabra, Alakazam, Dudunsparce})
                if evo is not None:
                    gets.append(evo)
                for cid in (Telepath_Psychic_Energy, Basic_Psychic_Energy,
                            Enriching_Energy):
                    if pool_d.get(cid, 0) >= 1:
                        gets.append(cid)
                        break
                if gets:
                    ns = (tuple(sorted(hrm(Hilda) + tuple(gets))), unk,
                          deck - len(gets), act, bench, nflags,
                          opph, ospec, dis, pool_rm(*gets))
                    out.append((("PLAY", Hilda), ns))
            if Lanas_Aid in hset and dis:
                nd = list(dis)
                gets = []
                for _ in range(3):
                    dis_cnt = Counter(nd)
                    pick = pref_fetch(dis_cnt, fc, _NONRULE_POKE)
                    if pick is None and dis_cnt.get(Basic_Psychic_Energy, 0) >= 1:
                        pick = Basic_Psychic_Energy
                    if pick is None:
                        break
                    nd.remove(pick)
                    gets.append(pick)
                if len(gets) >= 2:  # R15: 回収2枚以上で
                    ns = (tuple(sorted(hrm(Lanas_Aid) + tuple(gets))), unk, deck,
                          act, bench, nflags, opph, ospec, tuple(nd), pool)
                    out.append((("PLAY", Lanas_Aid), ns))

        # --- リトリート（1ターン1回・エネ支払い） ---
        if not (flags & _F_RET) and bench and not asleep_par:
            cost = card_table[act[0]].retreatCost
            if act[1] + act[2] >= cost:
                # 支払い: 非Pエネ優先で捨てる
                pay_oth = min(act[2], cost)
                pay_psy = cost - pay_oth
                old = (act[0], act[1] - pay_psy, act[2] - pay_oth, act[3])
                seen_t = set()
                for m in bench:
                    if m in seen_t:
                        continue
                    seen_t.add(m)
                    ns = (hand, unk, deck, m, badd(brm(bench, m), old),
                          flags | _F_RET, opph, ospec, dis, pool)
                    out.append((("RETREAT",), ns))
        return out

    # ---- 現在の選択肢のタグとpolicyスコア（052の順序資産）を先に用意 ----
    global _lines_gate_off
    snap = _snapshot_globals()
    _lines_gate_off = True
    try:
        policy(obs)
        pscores = list(_policy_raw_scores or [])
    finally:
        _lines_gate_off = False
        _restore_globals(snap)
    if len(pscores) != len(select.option):
        return None

    def opt_tag(o):
        if o.type == OptionType.ATTACK:
            return ("ATTACK", o.attackId)
        if o.type == OptionType.END:
            return ("END",)
        if o.type == OptionType.RETREAT:
            return ("RETREAT",)
        if o.type == OptionType.ABILITY:
            c = get_card(obs, o.area, o.index, my_index)
            return ("ABILITY", c.id) if c is not None else None
        if o.type == OptionType.PLAY:
            c = get_card(obs, AreaType.HAND, o.index, my_index)
            return ("PLAY", c.id) if c is not None else None
        if o.type == OptionType.EVOLVE:
            c = get_card(obs, AreaType.HAND, o.index, my_index)
            if c is None:
                return None
            return ("EVOLVE", c.id, "A" if o.inPlayArea == AreaType.ACTIVE else "B")
        if o.type == OptionType.ATTACH:
            c = get_card(obs, AreaType.HAND, o.index, my_index)
            pk = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
            if c is None or not isinstance(pk, Pokemon):
                return None
            return ("ATTACH", c.id, pk.id, "A" if o.inPlayArea == AreaType.ACTIVE else "B")
        return None

    opt_tags = [opt_tag(o) for o in select.option]
    opt_cls = [(_tag_bias(tg) if tg is not None and tg[0] not in ("ATTACK", "END")
                else None) for tg in opt_tags]
    ppick = max(range(len(pscores)), key=lambda i: pscores[i])

    # ---- BFS列挙 ----
    seen = {root: (None, None)}
    queue = [root]
    qi = 0
    best = None  # (score, state, attack_id, boss_j)
    capped = False
    bias_memo = {root: {}}
    # EXP-055: V使用時は終端を貯めてバッチ採点（numpy 1回のforward）
    v_cands = [] if (_V055 is not None and _collect is None) else None

    def bias_of(sk):
        r = bias_memo.get(sk)
        if r is not None:
            return r
        parent, tag = seen[sk]
        r = dict(bias_of(parent))
        bk = _tag_bias(tag)
        if bk is not None:
            r[bk] = r.get(bk, 0) + 1
        bias_memo[sk] = r
        return r

    while qi < len(queue):
        st = queue[qi]
        qi += 1
        if qi % 128 == 0 and time.perf_counter() > deadline:
            capped = True
            break
        pb_feat = bias_of(st)
        for feat, atk, bj in terminals(st):
            if pb_feat:
                feat.update(pb_feat)
            if bj is not None:
                feat["a_boss"] = feat.get("a_boss", 0) + 1
            if _collect is not None:
                _collect.append((st, feat, atk, bj))
            elif v_cands is not None:
                v_cands.append((feat, st, atk, bj))
            else:
                sc = 0.0
                for k, v in feat.items():
                    sc += Weff[k] * v
                if best is None or sc > best[0]:
                    best = (sc, st, atk, bj)
        if len(seen) < LINES_STATE_CAP:
            for tag, ns in succ(st):
                if ns not in seen:
                    seen[ns] = (st, tag)
                    queue.append(ns)
        else:
            capped = True
    _lines_stats["states"] += qi
    if capped:
        _lines_stats["capped"] += 1

    if v_cands:
        _sc_arr = _v055_scores([c[0] for c in v_cands])
        # win=1（確定勝ち切り）終端はVの近似に任せず厳密優先
        win_idx = [i for i, c in enumerate(v_cands) if c[0].get("win")]
        pool_idx = win_idx if win_idx else range(len(v_cands))
        bi_ = max(pool_idx, key=lambda i: _sc_arr[i])
        best = (float(_sc_arr[bi_]), v_cands[bi_][1],
                v_cands[bi_][2], v_cands[bi_][3])

    # ---- ラインの行動集合→現選択肢へのマッピング（順序は052スコアで決定） ----
    allowed_memo = {}

    def allowed_of(sk):
        r = allowed_memo.get(sk)
        if r is not None:
            return r
        parent, tag = seen[sk]
        r = frozenset() if tag is None else (allowed_of(parent) | {tag})
        allowed_memo[sk] = r
        return r

    def map_term(sk, atk, bj, with_mask=False):
        allowed = set(allowed_of(sk))
        if atk is not None:
            allowed.add(("ATTACK", atk))
            if bj is not None:
                allowed.add(("PLAY", Boss_Orders))
        elif not allowed:
            allowed.add(("END",))
        nonterm, term = [], []
        for i, tg in enumerate(opt_tags):
            if tg is None or tg not in allowed:
                continue
            (term if tg[0] in ("ATTACK", "END") else nonterm).append(i)
        if nonterm:
            # 順序 = (クラス優先度, 052policyスコア)。優先度全0なら052の順序そのまま
            pick = max(nonterm, key=lambda i: (_O054.get(opt_cls[i] or "", 0.0),
                                               pscores[i]))
        elif term:
            pick = max(term, key=lambda i: pscores[i])
        else:
            pick = -1
        if not with_mask:
            return pick
        mask = 0
        for i in nonterm:
            mask |= 1 << i
        for i in term:
            mask |= 1 << i
        return pick, mask

    if _collect is not None:
        # 較正モード: (feat, pred_option_idx, cand_mask) へ変換。副作用なし
        for k in range(len(_collect)):
            st, feat, atk, bj = _collect[k]
            pred, mask = map_term(st, atk, bj, with_mask=True)
            _collect[k] = (feat, pred, mask)
        global _lines_debug
        _lines_debug = {"pscores": pscores, "opt_cls": opt_cls}
        _search_time_used += time.perf_counter() - t0
        return None

    if best is None:
        _search_time_used += time.perf_counter() - t0
        return None
    _sc, bst, atk_id, boss_j = best
    pick = map_term(bst, atk_id, boss_j)

    elapsed = time.perf_counter() - t0
    _search_time_used += elapsed
    _lines_stats["time"] += elapsed
    if pick < 0:
        _lines_stats["fallback"] += 1
        return None

    o = select.option[pick]
    # Boss吊り先ヒント（SWITCHで参照）
    if boss_j is not None and o.type == OptionType.PLAY:
        c = get_card(obs, AreaType.HAND, o.index, my_index)
        if c is not None and c.id == Boss_Orders:
            _lines_boss_bench = op_bench_map[boss_j - 1]
            _lines_stats["boss"] += 1
    # 手番内グローバル（能力使用フラグ）を自分の選択に合わせて更新
    if o.type == OptionType.ABILITY:
        c = get_card(obs, o.area, o.index, my_index)
        if c is not None:
            if c.id == Dudunsparce:
                ability_used_dudunsparce = True
            elif c.id == Fezandipiti_ex:
                ability_used_fezandipiti = True
    # fire計測: policyの素の選択と異なるか
    if ppick != pick:
        _lines_stats["fired"] += 1
    return [pick]


# ---- EXP-057: 選択クラス予測モデル（GBT） ----
# ロード順: lightgbm+model_058.txt（ローカル高速）→ model_058.npz（純Python、Kaggle用。
# argmax一致100%検証済み）→ どちらも無ければ052フォールバック
_M057 = None
if os.environ.get("F057_ML", "on") != "off":
    try:
        try:
            import feat_058 as _ft057
        except ImportError:
            import importlib.util as _ilu057

            _fp = os.path.join(_AGENT_DIR_057, "feat_058.py")
            _sp = _ilu057.spec_from_file_location("feat_058", _fp)
            _ft057 = _ilu057.module_from_spec(_sp)
            _sp.loader.exec_module(_ft057)
        _dir057 = _AGENT_DIR_057
        try:
            import lightgbm as _lgb057

            _mp = os.path.join(_dir057, "model_058.txt")
            if not os.path.exists(_mp):
                raise FileNotFoundError(_mp)
            _M057 = {"mode": "lgb", "bst": _lgb057.Booster(model_file=_mp),
                     "ft": _ft057}
        except Exception:  # noqa: BLE001  lightgbm無し → npz純Python推論
            import numpy as _np057

            _z = _np057.load(os.path.join(_dir057, "model_058.npz"))
            _M057 = {"mode": "npz", "ft": _ft057,
                     "feat": _z["feat"], "thr": _z["thr"],
                     "left": _z["left"], "right": _z["right"],
                     "val": _z["val"], "roots": _z["roots"],
                     "tcls": _z["tree_cls"],
                     "ncls": int(_z["num_class"])}
    except Exception:  # noqa: BLE001
        _M057 = None


def _m057_scores(x: list) -> list:
    """行動クラスごとの生スコア（softmax前。argmax用途なので正規化不要）。"""
    if _M057["mode"] == "lgb":
        return list(_M057["bst"].predict([x])[0])
    feat, thr = _M057["feat"], _M057["thr"]
    left, right, val = _M057["left"], _M057["right"], _M057["val"]
    scores = [0.0] * _M057["ncls"]
    for r, c in zip(_M057["roots"], _M057["tcls"]):
        i = int(r)
        while feat[i] >= 0:
            i = int(left[i]) if x[feat[i]] <= thr[i] else int(right[i])
        scores[int(c)] += float(val[i])
    return scores


# ---- EXP-076: 信頼度ゲート ----
TAU_076 = float(os.environ.get("F076_TAU", "0.5"))     # 委譲閾値 τ
_CONF_NORM_076 = os.environ.get("F076_NORM", "on") != "off"  # マスク後正規化の有無
_STATS_PATH_076 = os.environ.get("F076_STATS")         # 委譲率カウンタのダンプ先
_conf_stats_076 = {"n": 0, "deleg": 0}


def _m057_proba(scores: list) -> list:
    """クラス確率（softmax後）。lgbモードのpredictは既にsoftmax済み。"""
    if _M057["mode"] == "lgb":
        return scores
    import math
    m = max(scores)
    e = [math.exp(s - m) for s in scores]
    z = sum(e)
    return [v / z for v in e]


def _ml_layer(obs: Observation) -> list[int] | None:
    """MAIN決定: GBTで行動クラスをargmax→クラス内は052policyスコアで具体option化。

    EXP-076: 選択可能クラスでマスクした確率の最大値（正規化後）が TAU_076 未満なら
    None を返して052 policyに委譲する。
    """
    global ability_used_dudunsparce, ability_used_fezandipiti
    select = obs.select
    if select.minCount != 1 or select.maxCount != 1 or len(select.option) < 2:
        return None
    state = obs.current
    my = state.yourIndex
    ft = _M057["ft"]
    feats, _keys = ft.feat_vector(state, my)
    proba = _m057_scores(feats)
    cls_of = [ft.option_class(state, select, i, my)
              for i in range(len(select.option))]
    best_cls = max(set(cls_of), key=lambda c: proba[ft.CLASS_ID[c]])

    # ---- EXP-076: 確信度ゲート（低確信なら052へ委譲） ----
    probs = _m057_proba(proba)
    sel_ids = [ft.CLASS_ID[c] for c in set(cls_of)]
    best_p = max(probs[i] for i in sel_ids)
    if _CONF_NORM_076:
        mass = sum(probs[i] for i in sel_ids)
        conf = best_p / mass if mass > 0 else 0.0
    else:
        conf = best_p
    _conf_stats_076["n"] += 1
    if conf < TAU_076:
        _conf_stats_076["deleg"] += 1
    if _STATS_PATH_076 and _conf_stats_076["n"] % 200 == 0:
        try:
            with open(_STATS_PATH_076, "a") as _f:
                _f.write(f"{os.getpid()},{_conf_stats_076['n']},{_conf_stats_076['deleg']}\n")
        except OSError:
            pass
    if conf < TAU_076:
        return None
    cands = [i for i, c in enumerate(cls_of) if c == best_cls]
    if len(cands) > 1:
        ps = _policy_scores(obs)
        cands = [max(cands, key=lambda i: ps[i])]
    pick = cands[0]
    # 手番内グローバル（能力使用フラグ）を自分の選択に合わせて更新
    o = select.option[pick]
    if o.type == OptionType.ABILITY:
        c = get_card(obs, o.area, o.index, my)
        if c is not None:
            if c.id == Dudunsparce:
                ability_used_dudunsparce = True
            elif c.id == Fezandipiti_ex:
                ability_used_fezandipiti = True
    return [pick]


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
        if _M057 is not None:
            try:
                choice = _ml_layer(obs)
            except Exception:
                choice = None
            if choice is not None:
                return choice
        if F054_LINES:
            try:
                choice = _lines_layer(obs)
            except Exception:
                choice = None
            if choice is not None:
                return choice

    return policy(obs)
