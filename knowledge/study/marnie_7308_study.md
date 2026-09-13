# Marnie個体 7308938c4d（Yushin Ito, 07-08 ladderトップ）フル精読スタディ — I-079第1フェーズ

- データ源: `data/episodes/2026-07-08/`（371戦, 総合勝率60.6%）。リプレイ精読18戦
  （勝10 / 負8、うち対Mega Kangaskhanアーキ 負6+勝3）。
- レンダラ: scratchpad/render_replay.py（EXP-035と同じ、--player指定で流用）。
- 前提: この個体の行動頻度集計にもとづく7ルールは **038_marnie_replica に実装済み**（EXP-038）。
  本スタディは頻度集計で写らない「条件文・順序・対面特化」の抽出が目的。
  特に対Kangaskhan 21.3%（W10 L36 D1）の敗因機序。

## 対面別成績（371戦、summary.csv再集計）

| 相手 | n | 勝率 | | 相手 | n | 勝率 |
|---|---|---|---|---|---|---|
| Alakazam | 129 | 64% | | Mega Starmie ex | 11 | 73% |
| Marnie ミラー | 48 | 73% | | Archaludon ex | 10 | 60% |
| **Mega Kangaskhan ex** | **47** | **21%** | | Dragapult ex | 10 | 90% |
| Cynthia's Garchomp | 38 | 61% | | Chandelure | 8 | 100% |
| **Crustle** | **24** | **38%** | | Mega Lucario ex | 7 | 86% |
| Barbaracle | 17 | 71% | | Comfey | 11 | 82% |

弱点は Kangaskhan(21%) と Crustle(38%) の2つで、**敗因機序は同一**（下記）。

## 対Kangaskhan（21.3%）敗因の機序 — 精読6敗+3勝の結論

### 相手デッキの正体
「Mega Kangaskhan ex」アーキの最多個体 sig 89d834e4d4（=039_kangaskhan_replicaの元）は
実体が **Crustle(345)壁 + Mega Kangaskhan(756) のハイブリッド**。対exデッキでは
Crustleのみで戦い、Kangaskhanをほぼ出さない試合も多い（84748156はCrustle 1枚に6プライズ献上）。
- **Crustle 345 の特性 Mysterious Rock Inn: 相手のex/megaExのワザダメージを全て0にする**。
  Shadow Bullet本体180も**ベンチスナイプ30も**両方0（84790848 t12で確認）。
- Superb Scissors 120はGrimmsnarl（悪=草弱点）に**240**。Cape 420でも2発+αで落ちる。
- 回復: Jumbo Ice Cream(+60〜80) / Grow Grass Energy(1枚+HP20) / Hero's Cape。
- 貫通手段は **(a) Adrena-Brainのダメカン移動30×体数/ターン（特性なので通る）、
  (b) 非exのワザ**（Budew Itchy Pollen 10+アイテムロック、Morgrem 60、Yveltal 110）。

### 負けゲーム共通の機序（84748156 / 84790848 / 84812826 / 84829646 / 84787780）
1. **ゼロダメージ攻撃の空回し**: Grimmsnarlで毎ターンShadow Bullet→常時0ダメージ。
   その体制を維持するためのリソース（進化・Punk Up・手貼り）が全部死ぬ。
2. **タンクローテーションでエネ自己破壊**: 傷んだGrimmsnarlをリトリート（エネ2枚捨て）→
   D枚数は山に10枚しかなく、中盤にはMunkidoriに貼る1枚すら無くなる（84790848 t18以降
   Munkidori2体が素のまま）。Adrena率が落ち唯一の貫通打点が消える。
3. **Munkidoriが狩られる**: 相手はBoss'sでMunkidoriを引き出しSuperbで処理（84812826で2体、
   84829646で3体）。1プライズ+チップエンジン喪失の二重損。
4. **プライズの安売り**: きぜつ後の押し出しにMunkidori/Dunsparceを出し、相手は壁のまま
   6プライズを回収して勝つ（84748156: Crustle 1体が6プライズ取り切り）。
5. **Run Away Draw/Lillieで自分の山を焼き尽くしデッキ切れ負け**（84790848: 残1枚で敗北、
   84829646: 残2枚で敗北。相手山は16〜20枚）。壁戦は長期戦なのに毎ターン3〜6ドローを続けた。

### 勝ちゲームで違ったこと（84915275 / 84907064 / 84877485）
1. **勝ち筋はAdrena-Brainのみ**: 84915275は攻撃ダメージ0のままダメカン移動だけで勝利。
   エネ付きMunkidori 2体=60/ターン ＋ Superb被弾がダメカン供給源（Adrenaは自陣の
   ダメカンを移す＝被弾がそのまま弾になる）。回復供給(60-80/t)を上回るには2体以上必須。
2. **ダメカンはアクティブのCrustle 1体に集中**（分散させない）。
3. **Shadow Bulletの使い所はKangaskhan**（756は素通しで180×2=キル、3プライズ）。
   BossでベンチのKangaskhanを引き出して2ターンで倒す（84907064 t10、84790848 t8-10）。
4. **安い壁で受ける**: 前で受けるのはCape付きGrimmsnarl（420+毎ターンAdrena回復30-60）か
   Dunsparce（1プライズ、Run Away Drawで山に戻る駒）。Munkidoriを前に出さない。
5. Xerosicで回復物資（Jumbo Ice Cream / Boss）を捨てさせる（84790848 t10、84915275 t11）。
6. Tool ScrapperでCrustleのCapeを剥がす（84748156 t11、84790848 t32）。
7. Budew Itchy Pollen=アイテムロックがJumbo Ice Creamを1ターン止める（84787780 t3/t7）。

## 実装可能な条件文ルール（038との差分中心）

前提: 「壁対面」= 相手の場に Crustle(345) がいる。W = wall_active（相手アクティブが345）。

### 壁対面（Kangaskhan 89d8系・Crustle 8b3183系の両方に適用）
- **K1**: 攻撃プラン計算で、対象が345かつ自分の攻撃者がex → ダメージ0として扱う
  （プラン対象から除外）。Shadow Bullet自体は「他に価値ある行動が無ければ」撃ってよい（無害）。
- **K2**: DAMAGE（ワザのベンチスナイプ）とDAMAGE_COUNTER（Adrena）を区別する。
  スナイプ先の345は0点（無駄撃ち）。Adrena先の345は最優先（+固定ボーナス、
  傷んでいる個体を優先=集中砲火）。
- **K3**: 壁対面ではベンチ攻撃者へのローテーション計画を止める（リトリートのエネ2枚捨てが
  Adrenaエンジンを殺す）。リトリートは原則しない。
- **K4**: Munkidoriは2〜3体並べ、全てに1エネ確保（手貼り最優先は038で実装済み。
  展開制限を壁対面では3体まで緩和）。
- **K5**: 相手ベンチにMega Kangaskhan(756)がいて自分のアクティブが攻撃可能なGrimmsnarlなら
  Bossで引き出す（キルでなくても価値が高い、180×2で3プライズ）。
- **K6**: きぜつ後の押し出し: Cape付きGrimmsnarl > 素のGrimmsnarl > Dunsparce ≫ Munkidori
  （Munkidoriは出さない）。
- **K7**: Budewのアイテムロック（Itchy Pollen）は壁対面で加点（Ice Cream封じ）。
- **K8**: Tool Scrapper/Xerosicは既存実装のままで機能（Cape剥がし・物資破壊）。

### 山管理（全対面、特に壁・スタミナ戦）
- **K9**: 山が少ないとき任意ドローを止める: Run Away Draw / Flip the Script は deck≤4 で
  使わない。Lillie / Dawn は deck≤6 で撃たない。
  （2敗がデッキ切れ。Run Away Drawは net -1枚/回、Lillieは手札<6のとき山が減る）

### ミラー・汎用（精読で確認できた個体の癖）
- **M1**: Adrena/スナイプの送り先は**エネ付きの相手Munkidori最優先**（ミラー勝ち2戦とも
  フルHPのMunkidoriへ30×2を毎ターン集中→相手の回復・チップエンジンを先に折る）。
  038は「≤60で+400」のみでフルHP Munkidoriには送らない → 差分。
- **M2**: Xerosicは相手の手札が肥えた時に最優先（Alakazam Powerful Handは手札×20ダメージ、
  対壁は物資破壊）。038の handCount>=4 発火で概ね一致。
- **M3**: CapeはGrimmsnarlに貼る（進化前Impidimpに早貼りして420進化も可: 84750337 t2、
  84751391 t1）。038はGrimmsnarl限定+300 → 概ね一致、差分小。
- **M4**: 個体はセットアップでMunkidori/Impidimpを並べ、その他は手札温存（038実装済み）。
- **M5**: 先攻は常に取る（038実装済み=YES既定）。

### 実装後の反証結果（EXP-043での検証。本スタディの読み違い）

- **K3（壁対面でローテーション禁止）は誤り**: 「リトリートのエネ2枚捨てが敗因」は負けゲームの
  症状であって原因ではない。勝ちゲーム（84915275 t9/t11/t13）でも本人はローテーションしており、
  最適化された039レプリカ相手では「その場でGrimmsnarlが死ぬ」害の方が大きい（K3ありで21%、
  外して28%、n=100 seed99）。防御的ローテーションは維持するのが正解。
- **K4（Munkidori 3体目）も不発**: エネ供給（手貼り1/ターン）が3体を養えず、素のMunkidoriが
  ベンチ枠を潰すだけ。2体で十分（勝ちゲームの実態も2体）。
- **K2の「壁にダメカン集中」は相手依存**: 039/042のような回復スタック壁（250-290HP）には
  ダメカンを注いでも落ちない。正解は「壁でないex（Kangaskhan/Cornerstone）＝プライズ化できる
  対象」への集中で、壁への集中は壁しかいない盤面のみ（vs 042: 集中5%→非壁ex優先16%）。

### 実装しない（費用対効果・リスクで見送り）
- Ogerpon対面（3cfb系）のMyriad Leaf Shower（両アクティブのエネ×30追加）対策
  =エネ過積載回避。ニッチかつ副作用リスク大。
- Morgrem Corkscrew 60を壁対面の主砲にする案（個体は使っていない。進化を止める害が大きい）。

## エンジン仕様メモ（リプレイから確認できた事実）
- Mysterious Rock Innは**ベンチへのワザダメージ（Shadow Bulletの30）も0にする**。
  ダメカン移動（Adrena-Brain）とダメカン配置は通る。非exのワザも通る。
- Superb Scissors「相手アクティブの効果を無視」だが弱点計算は適用される（Grimmsnarl 240）。
- Grow Grass Energyは付いたポケモンのHPを+20/枚（Crustle 150→190等）。
- Xerosic's Machinationsは相手の手札を大量破壊（観測では5〜9枚捨て）。
- Spiky Energy: これが付いた相手を攻撃すると自分に20ダメカン（84907064 t6等）。
- Risky Ruins（自分のスタジアム）: たねを出すたび自陣にもダメカン20が乗る点に注意
  （84756000 t5、Adrenaで回収して相手に送れるので実害は小さい）。
