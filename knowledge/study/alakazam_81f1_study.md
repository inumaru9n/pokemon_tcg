# Alakazam個体 81f1758c92（Yushin Ito, LBトップ）方策スタディ — I-081

- データ源: `data/episodes/2026-07-09/`（本人250戦, 総合勝率76.4% = ladder史上最高）。
  リプレイ精読20戦（勝10/負9+参照1）+ 全250戦の行動頻度集計3本
  （scratchpad/aggregate_045.py / 045b.py / 045c.py）で裏取り。
- **注意**: 同一60枚リストは多数のチームにコピーされ全体n=1205では54.8%。
  本人 vs コピー勢の同型ミラーで本人側84%勝ち＝**強さの本体はデッキでなく方策**。
  集計・精読は team=='Yushin Ito' の250行のみを対象にした。

## 対面別成績（250戦）

| 相手 | n | 勝率 | | 相手 | n | 勝率 |
|---|---|---|---|---|---|---|
| **Alakazam(ミラー)** | **163** | **82.2%** | | Archaludon ex | 6 | 83.3% |
| Mega Kangaskhan ex | 27 | 55.6% | | Brambleghast | 6 | 100% |
| Marnie's Grimmsnarl | 13 | 76.9% | | Crustle | 3 | 100% |
| Cynthia's Garchomp ex | 11 | 45.5% | | Mega Starmie ex | 3 | 66.7% |
| Team Rocket's Spidops | 10 | 40.0% | | Dragapult ex | 2 | 50% |

## 60枚リスト（a73706e527=EXP-040 との差分）

Abra/Kadabra/Alakazam 4-4-4、Dunsparce 3/Dudunsparce 2、Fez ex 1、**Shaymin(DRI,343) 1**、
Poffin 4 / Poké Pad 4 / Enhanced Hammer 3 / Rare Candy 3 / Night Stretcher 1 / Sacred Ash 1、
Dawn 4 / Hilda 4 / Boss 3 / **Xerosic 3** / Lana 1、**Nighttime Mine 3**、
Telepath 4 / P 2 / Enriching 1。
= a737比: Battle Cage×2, Tool Scrapper, Dudunsparce-1, Stretcher-1, Hammer-1 を抜き、
Xerosic+2 / Nighttime Mine+3 / Shaymin+1。

- **Nighttime Mine**（スタジアム）: 場のテラポケモン（両者）のワザコスト+1。自デッキにテラ0
  ＝一方的な税（対Dragapult/Ogerpon/Cornerstone）。実際は**相手スタジアムのバウンス**が主用途
- **Xerosic's Machinations**: 相手手札を3枚まで破壊。ミラーでは相手のPowerful Hand弾薬
  （手札枚数=打点）とキーパーツを直接削る主兵装
- **Shaymin (Flower Curtain)**: 自ベンチの非ルールボックスへのワザダメージを無効
  （Shadow Bulletベンチ30 / Cruel Arrow / Starmieベンチ50等を遮断）

## エンジン仕様（リプレイで確認した事実）

- **Powerful Hand = 「ダメカンを置く」効果でありダメージではない**。したがって
  (a) **Mist Energy**が対象に付いていると完全無効（counters 0を多数確認）、
  (b) **Team Rocket's Articuno「Repelling Veil」**が場にいると相手の**たねロケット団ポケモン
  全員**（Mewtwo ex/Tarountula/Mimikyu/Articuno自身）に完全無効、
  (c) 弱点・抵抗・Full Metal Lab等のダメージ修正を全て無視して通る
- 通常ダメージ攻撃（Super Psy Bolt 30 / Teleportation 10 / Land Crush 90 / Cruel Arrow 100）は
  Mist/Veilを素通しで通る（ダメージは効果ではない）
- Alakazamの進化ドロー（Psychic Draw）はACTIVATE(YES/NO)プロンプト。
  Fez / Dudunsparce のドローはMAINのABILITY選択
- Cynthia's Spiritomb「Raging Curse」= 相手ベンチのCynthia'sポケモンのダメカン×10。
  **こちらのPHチップが撃ち残しになると自軍への440砲に変換される**（対Garchomp敗着）

## ゲームプラン

1. Abraを並べ、Kadabra/Alakazam化のたび進化ドローで手札を膨らませる（Poffin/Poké Padで部品供給）
2. 毎ターン「昇格→進化→エネ1枚→全ドロー→（Xerosic）→Powerful Hand」のサイクル。
   PH平均手札13.4枚=268ダメージ、可能ターンの発射率99.9%（1355機会中不発14）
3. 自分のAlakazamは140HPで毎ターン落ちる前提。**Alakazam 4枚+Sacred Ash+Night Stretcher+
   Poké Padの循環で「次のアタッカー+エネ」を絶やさない**（非exなので1プライズ交換）
4. ミラーはXerosic×3で相手の手札（=打点と部品）を枯らし、Enhanced Hammerで相手AbraラインのTelepathを剥がして
   攻撃サイクルを止める。相手Fez ex（210, 2プライズ）をBossで吊ってPHで抜く
5. 山切れ注意（ミラーは1プライズ交換で長期化）。手札が既に致死なら追加ドローを自制

## 判断ルール集（islet式・頻度証拠付き）

### セットアップ
- **R1**: 先攻選択は常にYES（115/115）
- **R2**: アクティブ **Abra > Dunsparce > Shaymin > Fez**（140/75/21/14、pairwise Abra>Dunsparce 43:0。
  a737と逆なので040の値から変更必須）
- **R3**: ベンチは**AbraとDunsparceだけ**置く。Fezは置かない（skip29:置0）、Shayminも原則skip（15:4）
- **R4**: マリガン追加ドローは最大まで

### ドロー能力（進化ドロー/Fez/Dudunsparce）
- **R5**: 基本は必ず使う（YES 1394 / NO 91、MAIN ABIL Dudunsparce 671・Fez 261）
- **R6**: **手札が既に相手アクティブを倒せる(手札×20≥HP)とき、山が薄ければ(≤12目安)自制**
  （NO 91件中86件がkills_already=True、手札11+が71/91、deck≤8が53/91）。
  山1〜3ではキル必要時以外NO（山切れガード）

### エネルギー管理
- **R7**: 手貼りは「このターン/次に攻撃する個体」へ1枚。重ね貼りしない（e1+への貼りは全体の3%）
- **R8**: Telepath/P → Abraライン優先（Alakazam>Kadabra>Abra。Abraへの先行貼り160回は普通にやる）
- **R9**: **Enriching（ドロー4）は貼り先不問で早撃ち**（Dunsparce39/Abra31/Kadabra28/Fez18/Shaymin10。
  a737の「Dunsparceライン限定」より緩い）。ただしAbraラインに付くと{P}が別途必要（040の教訓）
- **R10**: アクティブで立ち往生するFez/Shaymin/Dunsparceには**リトリート費用のためだけにエネを貼る**
  （Fezへの貼り62回≒Fezリトリート54回）
- **R11**: 対Spidops/Archaludon/Crustle長期戦では**Dudunsparceにエネ3枚→Land Crush 90**を起動
  （46回、全てこの3対面。PHがVeil/Mistで無効な相手への迂回打点）

### サポーター
- **R12**: **Xerosicは相手手札≥7で最優先級**（プレイ率: 手札7=54%, 8+=68-82%）。
  4〜6は状況次第（16-22%）、**≤3では絶対に撃たない**（0/151）。
  ミラー勝ち134戦の平均使用2.3回/ゲーム
- **R13**: Bossは**キル専用+回避用**（202回中84%がそのターン処理可能な対象）。優先順:
  (a)勝利確定キル、(b)**相手Fez ex**（30回。210HP=手札11枚で抜けて2プライズ）、
  (c)**進化前の脅威**（Abra24/Kadabra16/Tarountula/Dwebble/Gible/Roselia）、
  (d)**カウンター無効アクティブの回避**（Articuno/Mimikyu/Mist付きが正面のとき裏の殺せる相手を吊る。
  active-before-Boss上位にArticuno16/Mimikyu13/未処理Kangaskhan12）
- **R14**: Dawn/Hildaは部品補充（Zam/Kadabra/エネ）。優先度はXerosic(≥7)/キルBossの下
- **R15**: Lana's Aidは回収2枚以上で（回収対象: P/たね）

### アイテム
- **R16**: Poffin/Poké Padは常時プレイ可（サーチ=手札中立、山も掘れる）。
  Poffin搬入は Abra > Dunsparce
- **R17**: **Enhanced Hammer優先ターゲット**: (1)PH対象に付いたMist Energy
  （Kangaskhan22/Crustle13）、(2)**ミラー: 相手Alakazam/Abraラインの Telepath**
  （108+47+16回=相手の攻撃サイクルを1ターン止める）、(3)Mewtwo exのTR Energy。
  対象がなければ温存（手札=PH打点）
- **R18**: Night Stretcher/Sacred Ash/Lana でAbraライン・エネを循環
  （Sacred Ash戻し: Abra161/Alakazam160/Kadabra92）

### スタジアム
- **R19**: **Nighttime Mineは「相手スタジアムのバウンス」が主用途**（プレイ70 vs END温存205。
  置換先: TR Factory15/Spikemuth12/Full Metal Lab8/Battle Cage7、空盤面20）。
  ミラーでは置かない（163戦で2回のみ。手札温存=打点）。
  相手が場にテラを出しているときは空でも置く価値あり（コスト+1の税）

### 攻撃・ターゲット選択
- **R20**: PHは撃てるなら毎ターン撃つ（不発1/828ミラーターン）。手札を全て展開/温存し切ってから
  ターン最後に発射
- **R21**: **カウンター無効(Mist付き/Veil下のたねTR)の対象はPHのキル候補から除外**。
  正面が無効ならHammer→それでも無効なら(a)Bossで裏の殺せる対象、(b)ダメージ系攻撃
  （Super Psy Bolt/Land Crush）、(c)それも無ければPH空撃ちでよい（本人は185回空撃ち=改善余地）
- **R22**: Kadabraフィニッシュ: 相手アクティブ残HP≤30ならSuper Psy Bolt 30で取る（27回）
- **R23**: 対Garchomp: **チップの撃ち残し禁止**。Power Weight付きGarchomp ex(400)は
  手札20枚未満で殴らない（38ダメカン残しがSpiritomb Raging Curse 380-440に変換され敗着）。
  BossでGabite/Roselia（進化前）を摘む方が価値が高い

### 昇格・リトリート
- **R24**: きぜつ後の昇格は「このターン攻撃再開できる個体」:
  エネ付きベンチAlakazam ≈ Kadabra（手札にAlakazamあり）292:280 > Abra(Candy+Zamあり)108 >
  Dunsparce（壁として差し出す）48。Fez/Dudunsparce/Shayminはベンチ温存（昇格7-4回のみ）
- **R25**: リトリートは「アクティブが攻撃不能でベンチに攻撃再開手段があるとき」。
  Fez ex（54回）/Shaymin（19回）が正面に居座ったら1エネ払って逃がす。
  Bossで吊られたAbra等には**その場でエネを貼って→リトリート**する小技（84996772 t12）

### 受け（相手の手札破壊への捨て順）
- **R26**: 捨てやすい順（頻度実測）: 余剰Hilda/Dawn > Nighttime Mine > Boss(余剰) > Poffin >
  余剰Alakazam > Hammer > Rare Candy > Kadabra > Telepath > Poké Pad > Xerosic > Shaymin >
  Dunsparce/Dudunsparce > P > Sacred Ash > Fez > Enriching > Lana > Abra > Night Stretcher

## ミラー戦82.2%の機序（勝ち134/負け29の対比）

1. **毎ターン攻撃を絶やさない**のが全て。攻撃サイクル=「次のKadabra/Alakazam+エネ1枚+手札7+」。
   負けゲームは立ち上がり遅れ（相手が先にt3-t5にPH開始）か事故ハンド
2. **Xerosic**で相手の手札を3枚に破壊（打点≤60化+部品損失。Boss×2やKadabra×2を落とした例）。
   勝ちゲーム平均2.3発
3. **Enhanced Hammerで相手Abraライン/AlakazamのTelepathを剥がす**=相手の攻撃サイクルを1ターン止める
   （PHのKOで自エネも道連れになるため、エネ供給は両者ギリギリ。1回の停止が決定打になる）
4. **相手Fez exをBossで吊ってPHで抜く**（+2プライズ。自分のFezも同じ liability＝相手に吊られて負けた
   ゲームあり）
5. 山切れ管理: 相手コピー勢は毎回全ドローして山2まで焼く。本人は手札が致死になったらドロー自制
   （R6）で山6-10を維持

## 対面弱点（自然に losses から）

- **Garchomp(45.5%)**: 400HP+回復は2連PHが必要でチップ残しがSpiritomb砲に変換される構造。
  Boss進化前摘みと OHKO まで待つ規律が必要（R23）
- **Spidops(40.0%)**: Articuno Veilで主砲が半減（Spidops本体としか交換できない）。
  Land Crush Dudunsparce線への切替が必要（R11）。40-65ターンの消耗戦
- **Kangaskhan(55.6%)**: Mist Energy+回復。Hammer→PHの2枚コンボ依存で、Hammer切れ時に膠着

## 実装への含意（045_alakazam_full、040との差分）

- 040骨格を流用しつつ: R2(setup逆転)/R6(ドロー自制)/R9(Enriching緩和)/R12(Xerosic×3運用)/
  R13(Boss拡張)/R17(Hammer優先順)/R19(Mine)/R21(カウンター無効の除外)/R23(Garchomp規律)/
  R24-25(昇格・リトリート)/R26(捨て順) を仕様化
- lethal探索層は036/043パターン（per-MAINスナップショット分離、SAMPLES計装で5→8判断）
