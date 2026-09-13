# Marnie個体 1ec0f47981（Luca本家、07-16初出〜07-22覇権）フル精読スタディ — EXP-092フェーズ1

- データ源: `data/episodes/{2026-07-16,17,18,22}/` の **Luca本人（team=Luca, sig=1ec0f47981）2,344戦**
  （W1393 L951 = 59.4%）。EXP-089が本家と特定済み（コピー勢は勝率一段下）。
- 定量: 全2,344戦の条件付き行動頻度集計（`agents/092_marnie_full/aggregate_092.py`、
  「対壁=盤面にCrustle(345)あり」で分割）+ Xerosic/Trimmer被弾時の捨て率
  （`discard_propensity_092.py`）。
- 精読トレース: ep 87365211（対e57e Kangaskhan勝ち・壁戦の教科書）/ 87406381（対e57e負け・
  ダブルCrustle耐久に沈む）/ 87371739・87363009（対fc15 Garchomp負け/勝ち）。
  レンダラ: tools/render_replay.py。
- 対面成績（2,344戦）: Alakazam 64.8% (n=1237) / ミラー 49.5% (305) / **Garchomp 36.8% (261)** /
  **Kangaskhan 71.9% (178)** / Spidops 72.3% (159) / Dragapult 59.4% (101) / MFroslass 38.8% (67)。
  旧7308個体（marnie_7308_study.md）の対Kangaskhan 21%を71.9%に反転させたのがこの個体の存在意義。

## デッキの骨格（082と同一60枚）

Grimmsnarl line 4-3-3 + Munkidori 4 + Snorunt-Froslass 2-2 + D energy 10 /
Petrel 4・Lillie 4・Poké Pad 4・Poffin 4・Spikemuth Gym 4・Night Stretcher 3・Candy 3・
Boss 2・Stamp(ACE)・Pokégear・Scrapper・Dawn。

エンジン仕様の要点:
- **Punk Up**（Grimmsnarl進化時）: 山から基本悪エネ**最大5枚**を自陣Marnie'sポケモンに任意配分
  （CardImpl.h 7764）。エネ加速の本体。ATTACH_TO(山から選ぶ)→ATTACH_FROM(貼り先)×枚数。
- **Spikemuth Gym**: 両者が毎ターン1回、山からMarnie'sポケモンをサーチ（=線の供給源。
  ability使用 8,481+749回/2344戦 ≈ 3.9回/戦）。
- Freezing Shroud（Froslass）: チェックアップ毎に**特性持ち全員**へダメカン1（敵味方無差別、
  自分のMunkidori/Grimmsnarlにも乗る=Adrenaの弾）。
- Shadow Bullet 180 [DD] + ベンチ30スナイプ。Corkscrew Punch 60 [DD]（Morgrem、非ex）。
  Filch 0 [無] ドロー1（Impidimp）。

## ルール仕様（R1..R27。頻度の根拠は n=2,344戦集計、例は ep/turn）

### セットアップ・展開

- **R1 先攻は常にYES**（1186/1186）。
- **R2 セットアップアクティブ: Impidimp 1235 > Munkidori 635 > Snorunt 474**。
  ベンチ: Munkidori 896 > Snorunt 321 > Impidimp 214（082と同一）。
- **R3 Poffinのベンチ供給は Impidimp 2810 > Snorunt 1821 の2枚のみ**（Munkidoriは手出し）。
- **R4 Munkidoriは最大3体・Snowラインは2体まで**（082と同一。maxgrimm集計と盤面分布から）。
- **R5 Froslassは2体目まで積極的に立てる**: Poké Padのサーチ先は Munkidori 2064 >
  **Froslass 1713** > Impidimp 649 > Snorunt 589 > Morgrem 529。082はFroslass 1体で満足する
  （field_counts[Froslass]==0条件）が、本人は壁戦・ミラーで2体並べてチップを倍にする
  （ep 87365211 t11-12: SKILL_ORDER Froslass×2）。

### エネルギー管理（Punk Up / 手貼り）

- **R6 手貼りの第一優先は0エネMunkidori**（4914/8399=59%。Adrena起動が最優先）。
  以下 Impidimp(en0) 1132 > Morgrem(en0) 464 > Froslass(en0) 424 > Snorunt(en0) 387 >
  Grimmsnarl(en1) 211。**Snow線への貼りは「前で詰まった時の逃げ賃」だけでなく
  リトリート準備として恒常的に発生**（下記R8）。
- **R7 Punk Upは常に全量取る**（辞退0/13,565選択）。**配分順**: ①進化したGrimmsnarl本体を2枚に
  （Shadow Bullet起動）②次の攻撃ライン（Impidimp/Morgrem）を2枚に（次代の事前装填）
  ③Grimmsnarlの3〜5枚目（リトリート賃+ローテ耐性。en2:578, en3:413, en4:285, en5:63）
  ④Impidimp/Morgremの3枚目（en2: 1095/342）。**Munkidori/Snow線へはPunk Upで貼らない**
  （dst分布に出現しない）。ゲーム中のGrimmsnarl最大エネ: 2が44%、**3+が56%**
  （3:629 4:276 5:253 6:58ゲーム）。082の「2枚でキャップ、以後-5で辞退」は誤り。
- **R8 リトリート賃の意図的貼り**: 攻撃プランが「ベンチの充填済みGrimmsnarlに交代して攻撃」の
  ターンは、手貼りをアクティブに入れて逃げ賃にする（ep 87365211 t5: Impidimpに3枚目を貼って
  リトリート→Grimmsnarl交代→Shadow Bullet。t7: Froslassに貼って同型）。
  Grimmsnarlのエネ3+はこの自己資金化でもある（リトリート消費2枚→まだ[DD]が残る）。

### 攻撃・ターゲット選択

- **R9 攻撃はほぼ常に実行**: 攻撃可能でENDしたのは30/5,540ターン（0.5%）。
  過剰打点の温存やスキップは存在しない。
- **R10 アタッカー序列**: Shadow Bullet 6,925 >> Corkscrew(Morgrem) 862 > Filch 817。
  Munkidori/Froslassは**一度も攻撃しない**（0/8,611）。
- **R11 Filchは山切れガード付きの早期つなぎ**: 使用の89%はdeck≥23（山を掘るのは序盤だけ）。
  deck≤2では使わない。
- **R12 ベンチスナイプ30の対象**: 低HP進化前（Abra hp50: 2,278回が最頻）>キル完成>その他。
  壁戦ではベンチCrustleへのスナイプは0点（Rock Innはex攻撃の全ダメージを消す。
  非CrustleのDwebble/Kangには通る）。
- **R13 Bossの3用途**（gust 1,203回の分布）:
  (a) **Fezandipiti ex吊り**（open戦の46%、497回。ほぼ全部en0）: キルでなく「重い置物を前に
  縛って2ターンかけて180×2で2プライズ」+ 相手のテンポを1ターン奪う。hp210フルでも吊る。
  (b) **キル完成**: Munkidori(en1) / 進化前（Dunsparce 85, Gabite 47, Roserade 36等）。
  (c) **壁戦の2モード**: Dwebble吊り52回（=次の壁の供給を断つ。Shadow Bullet 180が素通り）と
  Mega Kangaskhan吊り46回（3プライズバースト。ep 87362960、EXP-085で相手側から観測済み）。
- **R14 壁(Crustle)戦のアタッカー運用**: Grimmsnarlは壁アクティブ相手でもShadow Bulletを撃つ
  （343回。本体0ダメでもベンチスナイプ30が仕事）。壁の削り本体は
  **Morgrem Corkscrew 60（139回）+ Adrena30/ターン + Froslass Shroudチップ×体数**。
  Superb Scissors 120は草弱点Grimmsnarlに240=2発で落ちるため、壁戦の前線は
  Morgrem/Impidimp/Froslass（1プライズ献上帯）を優先し、Grimmsnarlはバースト時だけ前に出す
  （promote_wall: Morgrem 240 > Grimm 124 ≈ Imp 124。ep 87365211 t10/t12: KO後はMorgrem昇格）。
- **R15 Scrapperの壁戦価値**: Hero's Cape剥がし（max HP -100）でCrustleを射程に入れる
  （ep 87365211 t13: Petrel→Scrapperサーチ→Cape剥がし→そのターンからキルレンジ）。
  play_wall比率はopen比2.9倍（63/2392 vs 275/38431）。

### サポート・グッズ

- **R16 サポーター使い分け**: Lillie=手札が小さい時（使用時中央値4枚、rate: hand2-4が57%）、
  Petrel=手札5+でも撃てる万能サーチ（中央値5）。Boss/Dawnは用途があるときのみ。
- **R17 Petrelサーチ先**: **Unfair Stamp 1093 > Night Stretcher 751 > Poké Pad 568 >
  Lillie 331 > Gym 316 > Candy 251 > Scrapper 237**（Scrapperは壁/tool持ち相手で急上昇）。
- **R18 Lillieの山切れガード**: deck+手札が小さいときは撃たない（deck≤9での使用は全3,645回中
  約30回=1%未満）。Dawnはdeck>6でのみ（604/704はdeck20+）。
- **R19 Unfair Stampの発射条件**: 相手手札が肥えているとき最優先（発射時op手札の中央値8、
  12枚が777/1724=45%）。自分手札は小さいほど得（中央値3-4）。**temporal: 自サイド6のまま
  =序盤から撃つ**（myprz6が46%）。KO条件を満たした最初の機会に近い。
- **R20 Spikemuth Gymサーチ**: Grimmsnarl 3231 > Morgrem 2633 > Impidimp 2268。
  「線の欠けている段」を上から埋める。
- **R21 Night Stretcher**: エネ回収 2528/4100=62% >> Impidimp 487 > Snorunt 281 > Munkidori 265。
  手札にエネが無いときのエネ回収が基本。
- **R22 Pokégear**: Lillie 424 ≈ Petrel 422 > Boss 185 > Dawn 57。

### Adrena-Brain（毎ターンの中核エンジン、10,471+1,398回）

- **R23 ソース（取り元）**: 自分のMunkidori 7,330 > Grimmsnarl 3,497（=Shroud自傷とSuperbの
  被弾を回収して**回復しながら**弾にする。取り元は「傷んでいる方」で、アクティブの
  傷んだGrimmsnarl最優先=タンク回復。maxHp-hpに比例+Munkidori/Grimmボーナスの082モデルは正）。
- **R24 送り先**: (a) 低HP進化前（Abra 50/20/30が圧倒的。3個で1ターンキル圏）
  (b) 壁戦は**残HP無関係にCrustle**へ毎ターン積む（hp10〜280まで一様に分布=倒し切るまで積む）
  (c) キル完成が最優先。NUMBERは常に最大数。

### 被妨害応答（Xerosic 3,695回・Trimmer 151回の捨て率）

- **R25 捨て率テーブル**（offered→discarded、n=13,556オファー）:
  **捨てる**: Poffin 95% > Impidimp 91% > Poké Pad 82% > Morgrem 81% > Snorunt 78% >
  Pokégear 78% > Munkidori 76% > Dawn 72% > Candy 70% > Gym 66% > Scrapper 61%
  **残す**: **Unfair Stamp 3.5%** < Grimmsnarl 19% < Lillie 33% ≈ Petrel 33% ≈ Energy 34.5%
  ≈ Froslass 34% < Boss 39% < Stretcher 42%
  原理: 山からサーチで再入手できる線ポケモン/アイテムを捨て、再入手不能な妨害弾（Stamp）・
  ドローエンジン（Lillie/Petrel）・実弾（Grimmsnarl・エネ）を残す。082は**DISCARD文脈を
  一切スコアしておらず先頭選択**（=最大の未実装穴）。

### その他

- **R26 リトリート**: 傷んだGrimmsnarl（hp140/170帯）はベンチの充填済みGrimmsnarlがいれば下げる
  （タンクローテ、082実装済み）。壁戦はFroslass/Morgremを前に出し直す。
- **R27 ENDへの逃げは無い**: 選択肢があるときのENDは0.5%。展開・サーチ・チップを全部使ってから
  攻撃で締める（勝ち筋はサイドレースとチップの複利）。

## 対壁バースト（負け筋も含めた機序、EXP-085の宿題）

勝ちの型（ep 87365211）:
1. Kang/非Crustleがアクティブのうちに**Punk Upバースト**: ベンチでGrimmsnarl進化→5エネ配分→
   手貼りでアクティブに逃げ賃→**同ターン内にリトリート→交代→Shadow Bullet 180**。
   300HPのMega Kangaskhanを2ターン（180×2>300+IceCream）で撃ち抜き3プライズ先行。
2. 壁だけが残ったら**非ex前線に切替**: Morgrem 60 + Adrena 30×/T + Shroud 10×体数×2/Tペア +
   Scrapper Cape剥がし + （Boss Dwebble吊りで壁の再生産を断つ）。回復80/Tを上回る。
3. Grimmsnarlはこの局面では2プライズの的（Superb 240）。前に置かない。

負けの型（ep 87406381）: 相手がCrustleを2体立て、Grow+Capeで310HP+回復まで積むと、
バーストの的（Kang）が出て来ず削り速度が回復を上回れない。Grimmsnarlを2体トレードして
サイド1まで詰められて詰み。**Luca本人でも28%は落とす対面**であり、
レプリカの忠実度目標はこの挙動分布まで（勝率の上限はアーキ相性）。

## 082骨格との主要差分（092で直すべき点の優先順）

1. **DISCARD無実装**（R25。Xerosic 1.6回/戦×取り返し不可）
2. **Punk Up配分**（R7。082はGrimm2枚キャップで辞退→本人は3-5枚+次代装填。5.6配分/戦）
3. **Poké Pad/GymサーチのFroslass過小**（R5。Pad第2位がFroslass）
4. **Boss運用にFez吊り/Dwebble断ちが無い**（R13。gustの46%がFez）
5. **リトリート賃の意図的貼り**（R8。バーストの成立条件）
6. **Petrelサーチ順の較正**（R17。082はStamp>Stretcher>Padの順自体は近いが重みが粗い）
7. 細部: Filch dеck床（R11）、Dawn床（R18）、スナイプ先の進化前優先（R12）
