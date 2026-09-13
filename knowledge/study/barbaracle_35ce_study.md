# Barbaracle個体 35ce7e0e55（Majkel1337, 本番226戦60.6%）フル精読スタディ — I-080第1段階

- データ源: `data/episodes/2026-07-08/`（226戦）。精読18戦（勝10 / 負8。負けは対Marnie 3・対Crustle 3・対Alakazam 2）
  + 全226戦の補助集計（scratchpad/agg_044.py。EXP-037の頻度集計の追加版）。
- レンダラ: scratchpad/render_replay.py（EXP-035と同じ。--player指定で任意個体視点）。
- 精読ゲーム: 勝=84752919/84805388/84791843(Marnie), 84825625/84841500/84746571(Crustle), 84864024(Alakazam),
  84807374(Kangaskhan), 84806383(Starmie), 84808862(Garchomp)。
  負=84745534/84760784/84799373(Marnie), 84745036/84774401/84809815(Crustle), 84749772/84794318(Alakazam)。
- 037（軽量逆設計）との差分に注目した2周目精読。EXP-037の集計済みルール（Prism→Okidogi/Good Punch主砲/
  Stone Armsエンジン/先攻100%/Solrock setup優先）は再掲しない。

## 対面別成績（このデータセット226戦の実数）

| 相手 | 勝-負 | 勝率 | | 相手 | 勝-負 | 勝率 |
|---|---|---|---|---|---|---|
| Alakazam | 31-37 | 45.6% | | Kangaskhan | 26-0 | 100% |
| **Marnie's Grimmsnarl** | **14-22** | **38.9%** | | Mega Starmie | 15-2 | 88% |
| **Crustle** | **6-11** | **35.3%** | | Garchomp | 15-10 | 60% |

## エンジン仕様（リプレイ+カードデータで確認した対面の物理法則）

- **Cornerstone Mask Ogerpon ex（Crustleデッキの壁）**: 特性Cornerstone Stance =「特性を持つポケモンからのワザダメージを全て防ぐ」。
  我々の特性持ち = **Okidogi/Lunatone/Barbaracle/Ursaluna → 全員ダメージ0**。特性なし = **Solrock（Cosmic Beam 70）とBinacleのみ通る**。
  Demolish 140は「相手アクティブへの効果を無視」= **NZ貫通**（EXP-042の無効果系と同じ）。Hero's Capeで310HP+Jumbo Ice Cream回復80。
- **NZはGrimmsnarl exのShadow Bulletを完全無効化する**（我々は全員非ルールボックス）。84752919のt7/t9/t11で
  「Shadow Bullet→HP 0 / HP 0」を確認。**対MarnieでNZが立っている間、相手の主砲はゼロになる**（Munkidoriの特性チップ30×体数だけ残る）。
- **Alakazam Powerful Handはダメージカウンター配置**（手札×20相当。実測 -120〜-340）→ NZで防げない。**Battle Cageはベンチのみ防護**（アクティブは食らう）。
  Alakazamは**闘抵抗-30**: Prism付きGood Punch 170-30=**140=ちょうどOHKO**、Prism無しだと70-30=**40**。Enhanced Hammer×4がPrismを剥がしに来る。
- Munkidori/Articuno/Fezandipitiも闘抵抗-30（Good Punch 140でMunkidori 110はOHKO維持）。Dunsparce/Dudunsparce/Kangaskhanは闘弱点×2。
- Risky Ruins（Marnie変種）: 非悪たねをベンチに出すたびダメカン2個。Budew Itchy Pollen=アイテムロック（対策不能、被害は小）。
- Spikemuth Gymは「Marnie'sポケモンをサーチ」。我々には完全な無意味（個体は毎ターン空起動する癖あり=写す必要なし）。

## 負けの機序

### 対Crustle 35%（本番27%）: 実は「相性」ではなく方策バグ
1. **Cornerstoneに特性持ちで殴り続ける**: 全226戦で**ダメージ0の攻撃を122回（12ゲーム）**。負け6戦中5戦がこのパターンで、
   84745036ではBoss's Orders×2を手に持ったままJudgeで流し、8ターン連続0ダメージ→Demolishで各個撃破された。
2. 正解はゲーム内に存在する: (a) **Boss's Ordersで1プライズ級を引きずり出して6枚取り切る**
   （Good PunchはDwebble70/Munkidori140/Articuno140/Crustle150を全てOHKO。Crustleデッキの非Ogerponは計14体）、
   (b) **Cornerstoneには唯一手が通るSolrockで70ずつ**（Jumbo回復80はエネ3枚以上が条件で序盤は不成立）、
   (c) 手がない時は0ダメ攻撃でお茶を濁さずベンチ整備。
3. 勝ち6戦の内訳が証拠: 相手ベンチ事故（84746571: 単騎CrustleをMad Bite 310でOHKO）、Cornerstone不在の変種（8126ef8305=Teal Mask型）、
   **相手のデッキ切れ（84841500: 57ターン、こちらプライズ2枚しか取れず相手が先に山切れ）**。= 個体はCornerstoneに勝ち筋を持っていない。
4. 副次: 長期戦になる対面なので**山管理が勝敗条件**（Lunar Cycleで自分の山を掘りすぎると自分が先に切れる）。

### 対Marnie 39%（本番30-35%）: NZの切りどころと打点レース
1. Marnie側の構造: Shadow Bullet 180が我々の全ポケモン（Okidogi 230以外）をOHKO+ベンチ30。Munkidori×3-4が毎ターン
   30×体数を移動（Grimmsnarl回復+こちらチップ）。**Grimmsnarl(320/Cape 420)を落とすにはGood Punch 170×3発**（回復込み）。
   素の交換レースは「あちらは毎ターン1枚、こちらは3ターンに2枚」で構造的に負け。
2. **唯一の逆転装置はNZ**（Shadow Bullet完全無効）だが、Marnieは**スタジアム4枚（Spikemuth×3+Risky）**でNZを即割りに来る。
   負け3戦は全てNZをt3-t4に早出し→1-2ターンで割られて以後裸。勝ち84752919はt6にNZが**相手のSpikemuth 1枚目を上書き**する形で
   入り、相手のスタジアム引きが止まって6ターン無敵→レース逆転。
3. 個体のNZ運用は雑（150回中、相手ルールボックス不在で70回・**自分のスタジアム上書き72回**）で、ここは学ばず改善余地:
   **相手のスタジアム消費数を数え、Battle Cage×3を先に囮として切り、NZ（1枚しかない）は相手スタジアムが薄くなってから**。
4. Battle Cageの防御価値: **Munkidoriのダメカン移動はベンチに置けなくなる**（84760784 t9「damage counters 0」で実証）。
   置き場がアクティブに限定されると回復効率も落ちる。
5. 勝ち筋の共通点: (a) Grimmsnarl ex(2プライズ)を3回落とす=6枚、Okidogi 230がShadow Bullet+チップを2ターン耐える、
   (b) **Mad Bite**が蓄積ダメージで310-400点になりCape付きGrimmsnarlも一撃圏（84805388 t11: -310）、
   (c) Xerosicで整った手札を流す、(d) Barbaracleライン2本目でStone Arms×2/ターン（84752919、全226戦の23%で2本置き）。

### 対Alakazam 46%（プール最大ウェイト31.6）: 手札枚数=被ダメージ
1. Powerful Hand（手札×20カウンター）が中盤以降ほぼ全てをOHKO（-220〜-340）。防御札は効かない。
2. **相手の手札を減らすことが直接のダメージ軽減**: Xerosic（→3枚=60点化）とJudge（→4枚=80点化）。
   勝ち84864024はXerosicを相手の初攻撃前に当てて以後-120〜-180に抑えた。負け2戦はどちらもXerosicがt10-11と遅い。
3. Enhanced Hammer×4にPrismを剥がされたOkidogiは40点しか出ない（抵抗-30）→ Prism再装填の優先度が高い。
   Prism付きGood Punch 140はAlakazam(140)のちょうどOHKO。
4. Boss's Ordersの主要ターゲットは**Fezandipiti ex**（個体の全Boss対象1位=18回。エネ0で反撃なし・2プライズ）と
   進化前のAbra/Kadabra。

## 判断ルール集（islet式に実装可能な形。R番号はEXP-037集計ルールの続番）

### 攻撃・ターゲット選択
- **R31（最重要・修正）**: 相手アクティブがCornerstone Mask Ogerpon exのとき、特性持ち（Okidogi/Lunatone/Barbaracle/Ursaluna）の
  攻撃ダメージを0として計画する。個体は真似ない（0ダメ攻撃122回の敗因）。
- **R32**: 対CornerstoneはSolrockが唯一のアタッカー。Cornerstoneが見えたらSolrockへのエネ供給と
  ベンチのLunatone維持（Cosmic Beam条件）を最優先。
- **R33**: Cornerstoneが壁のとき、Boss's Ordersで1プライズ級（Dwebble/Munkidori/Crustle/Articuno）を引きずり出して
  Good PunchでOHKO。6プライズは壁を無視して取り切れる。
- **R34**: Cosmic BeamはLunatoneが**ベンチ**に居ないと0ダメージ（個体は空撃ち25回。真似ない）。
- **R35**: Mad Biteの打点=100+30×相手ダメカン。**Cape付きGrimmsnarl/大型taнкには蓄積後のMad Biteがフィニッシャー**
  （実測310〜400）。Ursalunaは対Marnie/Kangaskhan/大物対面で価値が上がる。
- **R36**: Boss's Ordersの優先対象: (1)KO確定の1プライズ級 (2)**Fezandipiti ex**（エネ0・2プライズ・反撃なし）
  (3)進化途中の種（Abra/Impidimp/Gible/Dwebble/Snorunt） (4)対MarnieのD付きMunkidori。
- **R37**: Dunsparce/Dudunsparce（闘弱点340）はKO確定のときのみ（Dudunsparceは山に戻る再利用系）。

### スタジアム運用
- **R38（個体より改善）**: NZは1枚で回収不能。**相手がスタジアムを使うデッキ（Spikemuth/Risky確認済み=Marnie系）では、
  相手のスタジアム消費数（場+相手トラッシュ）が3枚以上になるまでNZを温存**し、それまでBattle Cage×3を
  「相手スタジアム潰し兼囮」に使う。スタジアム非搭載の相手（ex持ち）にはこれまで通り即出しでよい。
- **R39**: NZが立ったら上書きしない（Battle Cageを持っていても出さない。個体の自傷5回/72回を真似ない）。
- **R40**: Battle Cageは対Marnie/対Munkidori系で常設優先（ベンチへのダメカン移動を封じる）。相手スタジアムの上書きは常に価値あり。

### サポーター・リソース
- **R41**: Xerosicは相手の手札が肥えたとき（≥8枚目安、個体の最頻値は12枚）。**対Alakazamは特別で、
  相手のアタッカーがAlakazamなら手札6枚以上で最優先撃ち**（Powerful Handの直接減衰）。
- **R42**: Judgeも対Alakazamでは攻撃的に使う（相手手札≥7で自分の手札が良くても打つ価値がある）。
- **R43**: 対Alakazam: OkidogiのPrismがEnhanced Hammerで剥がされたら再装填最優先（Prism無しGood Punch=40点）。
- **R44**: 長期戦（Cornerstone壁・お互い決定打なし）では山切れ負けに注意。Lunar Cycle/Lillie/Judgeの
  低山ガードを強める（相手が攻撃できない膠着では山温存が勝ち筋: 84841500）。

### 盤面・エネルギー
- **R45**: Barbaracleライン2本目は許容（個体は23%のゲームで2本置き、Stone Arms 2回/ターンでOkidogi後続とSolrockを同時充填）。
  ただし優先度は低く、余剰があるときのみ。
- **R46**: Ursalunaは早出しでよい（初見ターンの最頻値はt1-t2。Battle-Hardenedの2枚加速が序盤の展開を作る）。
  手札F2枚が条件（037実装済み）。
- **R47**: セットアップベンチは手札の種を1-3体置く（1体:71 / 2体:42 / 3体:10）。Risky Ruins下では非悪たねに2カウンター乗ることに注意（軽微）。

### 写しない癖（個体の観測されたミス）
- 0ダメ攻撃×122（R31）、Cosmic Beam空撃ち×25（R34）、NZ自陣上書き×72（R38/39）、
  攻撃可能なのにEND×2、Spikemuth Gym空起動（相手のスタジアムをタダで起動する癖・無害だが無意味）。

## 実装への含意（044での優先順）

1. R31-R33（対Crustle: 本番27%の主因は確定的な計算誤り。エンジン仕様に基づく修正でEXP-042のNebula焼き込みと同型）
2. R38-R40（対Marnie: NZ温存ゲート。唯一の逆転装置を囮運用しない）
3. R41-R43(対Alakazam: プール最大ウェイト。手札破壊=ダメージ軽減の対面特化)
4. R35/R36/R44-R46（汎用の質改善）
5. lethal限定探索の移植（EXP-036の手順: SAMPLES=5、転換率計装）
