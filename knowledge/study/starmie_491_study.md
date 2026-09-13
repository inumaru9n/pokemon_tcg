# Starmie個体 491b8bfb26（Yushin Ito, LB1300+）方策スタディ — I-071第1フェーズ

- データ源: `data/episodes/2026-07-08/`（290戦, 総合勝率61%）。リプレイ精読16戦（勝10/負6, うち対Marnie 5戦）+ 全290戦の集計スクリプトで裏取り。
- リプレイ描画スクリプト: scratchpadの `render_replay.py`（obs[i]+action[i+1]のペアリング、logs差分dedupe。再利用価値あり→必要ならtools/へ移植）。
- **注意: このデッキはカタログ(721010)の「Starmie/Froslass」とは別物**。Froslass/Munkidori/Dudunsparceは不採用。実体は **Mega Starmie ex + Cinderace(壁/加速) + Wally回復ループ** の単騎アタッカー型。episodesのarchetype「Mega Starmie ex」表記でも中身はこの型。

## 対面別成績（290戦）

| 相手 | n | 勝率 | | 相手 | n | 勝率 |
|---|---|---|---|---|---|---|
| Alakazam | 101 | 64% | | Mega Lucario ex | 13 | 54% |
| Mega Kangaskhan ex | 41 | 76% | | Mega Starmie ex(ミラー) | 7 | 43% |
| **Marnie's Grimmsnarl ex** | **35** | **43%** | | Cinderace(他デッキの看板誤分類含む) | 7 | 29% |
| Cynthia's Garchomp ex | 27 | 52% | | Dragapult ex | 6 | 67% |
| Crustle | 19 | 95% | | Mega Froslass ex | 3 | 0% |
| Barbaracle | 15 | 47% | | Chandelure | 4 | 100% |

## 60枚リスト（deck_ids）

```
3 3 3 3 3 3 3 3 3 17 17 17 17 666 666 666 666 1030 1030 1030 1031 1031 1031
1086 1086 1086 1086 1097 1097 1120 1120 1120 1120 1121 1122 1122 1122 1122
1145 1145 1145 1145 1159 1182 1189 1189 1189 1189 1223 1223 1225 1225
1227 1227 1227 1227 1229 1229 1229 1229
```

| ID | 枚数 | カード | 役割 |
|---|---|---|---|
| 3 | 9 | 基本水エネ | Jetting Blowは水1枚で撃てる |
| 17 | 4 | Ignition Energy | 進化ポケに付くとCCC=Nebula Beam即撃ち。ターン末に自壊 |
| 666 | 4 | Cinderace (160HP, 特性Explosiveness) | セットアップ時に手札からアクティブ配置可。壁+Turbo Flare(50+デッキからW3枚ベンチ加速) |
| 1030 | 3 | Staryu (70HP) | Mega Starmieの種 |
| 1031 | 3 | Mega Starmie ex (330HP) | Jetting Blow 120+ベンチ50 / Nebula Beam 210(弱点抵抗・効果無視) |
| 1086 | 4 | Buddy-Buddy Poffin | Staryuベンチ展開 |
| 1097 | 2 | Night Stretcher | ポケモン/基本エネ回収 |
| 1120 | 4 | Crushing Hammer | コイン表でエネ破壊 |
| 1121 | 1 | Ultra Ball | ほぼ使わない(全290戦で27回) |
| 1122 | 4 | Pokégear 3.0 | 上7枚からサポーターサーチ |
| 1145 | 4 | Mega Signal | Mega進化ex手札サーチ |
| 1159 | 1 | Hero's Cape | +100HP(Mega=430) |
| 1182 | 1 | Boss's Orders | 詰め/Munkidori排除 |
| 1189 | 4 | Salvatore | 当ターン設置ポケも進化させられる(特性なし進化限定) |
| 1223 | 2 | Harlequin | 両者手札シャッフル+コインで5/3 or 3/5ドロー |
| 1225 | 2 | Hilda | 進化ポケ+エネをサーチ |
| 1227 | 4 | Lillie's Determination | 手札シャッフル→6枚(プライズ6なら8枚) |
| 1229 | 4 | Wally's Compassion | Mega ex全回復+エネ手札戻し=回復ループの核 |

## ゲームプラン概要

1. Cinderaceを壁にしてStaryuをベンチで安全にMega化(Cape付き430HPが理想)
2. T3前後からMega StarmieのJetting Blow(120+ベンチ50)を毎ターン継続。手貼りW1枚/ターンで足りる超低燃費
3. 被弾したらWally全回復→貼り直して同ターン攻撃続行(Cape込みで実質不沈)
4. ベンチ50で相手の進化ラインを摘みつつ、瀕死ベンチを撃ち抜いて追加プライズ
5. 大物(HP121-210超)にはIgnition→Nebula Beam 210

## 判断ルール集（islet式に実装可能な条件文形式）

### セットアップ
- **R1**: 先攻/後攻選択は**常に先攻**（290戦中154回の選択機会で154回YES）
- **R2**: アクティブ配置はCinderace優先、なければStaryu（Cinderace 167 / Staryu 123）。理由: 160HP壁+後攻ならTurbo Flare加速+Megaの盾
- **R3**: たねがCinderaceのみの手札でマリガン選択が出たら: **Buddy-Buddy Poffinが手札にあればkeep、なければマリガン**（観測12回で6/6+6/6の完全一致）
- **R4**: セットアップベンチ: 手札のStaryuは全部置く。2枚目のCinderaceは置かない
- **R5**: 相手マリガン時の追加ドローは最大まで引く

### 序盤（T1〜T3）
- **R6**: T1(先攻)は「Poffin→Staryu展開」「Mega Signal→Mega Starmie回収」「Staryuに手貼りW」だけ行いEND。攻撃できないターンでも山は掘るが、**不要な手札流し(Lillie)は撃たない**（手札が既に機能しているとき温存した例: 84744071 t1）
- **R7**: 後攻T2でCinderaceがアクティブなら: WをCinderaceに手貼りして**Turbo Flare**（50点+デッキからWをベンチStaryu/Cape付きStaryuに加速）
- **R8**: StaryuはMega化できるようになった**最初の機会に必ず**進化（全16戦例外なし）。Salvatoreで当ターン設置Staryuも即進化
- **R9**: Mega Signal/Poffinは温存せず序盤に連打してよい（1ターン4枚例あり）。目的はMega本体の確保とStaryu2体目
- **R10**: ベンチのMegaにエネが付いたターンに**retreatしてMegaを前へ**、そのまま攻撃開始。初攻撃はT2/T3が83%（T2:87戦, T3:155戦）。CinderaceのretreatコストはIgnition一時貼りで払える小技あり(84893893 t9)

### エネ管理
- **R11**: 手貼りは基本「次に攻撃するMega」へW1枚。Jetting Blowは水1で撃てるため過剰投資しない
- **R12**: アクティブMegaが既にW持ちなら、2枚目のWは**ベンチの控えMega**へ（後継機準備。主力が落ちても即Jetting再開できたことが複数戦の生命線: 84753937 t9-10）
- **R13**: Ignition Energyは**そのターンNebula Beamを撃つときだけ**貼る（ターン末自壊のため貯め無意味）。Wが引けていないターンの攻撃手段代替にもなる（84900107はT3の初攻撃がIgnition→Nebula 210）
- **R14**: Hero's Cape(1枚)は主力Mega、またはこれからMega化するStaryuに早期に貼る。Cape付きStaryu=170HPは先2の攻撃を耐えてからMega化できる（84809838 t2, 84818805 t2）。**Cape+Wally=430HP全回復ループがこのデッキ最強のエンジン**

### 攻撃選択
- **R15**: デフォルトは**Jetting Blow**（総使用1045回 vs Nebula 354回）。**Nebula Beam(210)を選ぶ条件**: (a)相手アクティブHPが121〜210でJettingでは落ちずNebulaで落ちる (b)高HP大物(Grimmsnarl 320/Crustle 290/Okidogi/Archaludon)を最速で削りたい (c)手貼り可能なエネがIgnitionしかない。※両方撃てる盤面は稀（意思決定は実質「どのエネを貼るか」の時点で完了）
- **R16**: ベンチ50の対象優先順位:
  1. **HP≤50で確定きぜつ**するポケモン（追加プライズ。Abraへのlethal snipeが全体1位198回）
  2. 進化ラインの種/中間（Abra/Kadabra/Dreepy/Drakloak/Gible/Gabite/Riolu/Dwebble/Snorunt/Impidimp/Morgrem）を育つ前に潰す
  3. エネ付きのアタッカー候補
- **R17**: 対Marnieでは**Munkidori(110HP)最優先**でsnipe/Bossする（勝ちゲームの共通点。50×2+αで落ちる）。相手がDudunsparceで山に戻すDunsparceへのsnipeは無駄撃ち（負けゲーム84846187で4連続浪費）
- **R18**: Boss's Orders(1枚)の使い所: (a)詰め=低HPベンチを引き出しJetting+snipeのダブルKOでプライズ2〜3枚(84744071 t9 Spiritomb) (b)対Marnie=Munkidori排除(84753937 t12)

### サステイン（Wallyループ）
- **R19**: アクティブMegaの累計被ダメが**~150以上**なら、ターン冒頭でWally's Compassion全回復（戻ったWを同ターン貼り直して攻撃継続）。43%のゲームで1回以上、最大4回/ゲーム
- **R20**: アクティブが無傷でベンチの控えMegaが重傷なら、Wallyはベンチに使ってよい（84756519 t15）
- **R21**: Crushing Hammerは**相手アクティブのメインアタッカーのエネを最優先**で剥がす（攻撃を1ターン止める=Wallyループと後続準備の時間を買う。84756519 t13のヒットで相手が1ターン攻撃不能になり勝ち確定）。対MarnieではMunkidoriのD剥がし(特性Adrena-Brain停止)も高価値(84753937 t2)

### ドロー/サーチ
- **R22**: Lillie's Determinationは**手札のプレイ可能カードを使い切ってから**撃つ（シャッフルで失うものを減らす）。プライズ6枚時は8枚ドローなので序盤ほど強い
- **R23**: Pokégearはほぼ毎ターン即使用（総使用357回=1.2回/戦で最多アイテム）。取得優先: 被弾中/対Marnie→**Wally** > 詰め局面→**Boss** > それ以外→Lillie/Hilda。対象なしなら取らない選択もする
- **R24**: Hildaで取るもの: 進化ポケ=Mega Starmie優先（尽きたらCinderace）、エネ=そのターンの攻撃に必要な種類（Nebula撃つならIgnition、通常はW）
- **R25**: Harlequinは手札が痩せたとき（≤3枚目安）の回復ドロー。副次効果として**相手の整った手札を流す**ため対Marnie/コントロールでは能動的に価値あり（84756519 t3で相手8枚を流した）
- **R26**: Ultra Ballはほぼ死にスロット（27回/290戦）。使うときのコストはNight Stretcher/Harlequin等の余剰から

### 受け（相手の手札破壊・強制討伐への対応）
- **R27**: Xerosic等での自分の捨て札選択: 盤面完成後はSalvatore/Mega Signal/Ignition等の役割終了カードから捨て、**W・ドローサポーター・Wallyを残す**。ただし観測ではWallyを捨てた例もあり一貫していない（84900107 t6でWally捨て→負け筋。**ここは改善余地**）
- **R28**: **ベンチ規律**: ベンチは最小限(1〜2体)。裸のStaryuを置いたままターンを渡さない（Munkidoriのダメカン移動+Shadow Bulletベンチ30で狩られ、Mega本体と合わせ1ターン4プライズ献上した敗着例: 84846187 t8-9）。置くならそのターン中にMega化するか、Capeを貼る

### 実装不要の癖
- **R29**: プライズ取得の選択は常に先頭（裏向きで情報なし。任意でよい）
- **R30**: 相手のSpikemuth Gym（両プレイヤー使用可: 山からMarnie'sポケモンサーチ）を毎ターン起動しているが、こちらにMarnie'sポケモンはいないため**完全な空振り**。「使える能力は使う」ヒューリスティックの癖であり真似る必要なし

## 対Marnie戦（43%）: 敗因の機序と勝ち筋

### Marnie側の構造
Grimmsnarl ex 320HP・Shadow Bullet 180+ベンチ30、**Munkidori複数体**（特性Adrena-Brain: D付きなら毎ターン自分のダメカン3個を相手に移動=Grimmsnarl 30回復+こちら30チップ×体数）、Punk Up（進化時に山からD5枚加速）、Unfair Stamp（こちらのKO直後に手札2枚化）、Xerosic（手札破壊）、Dudunsparce（山戻り再利用でsnipeの的を消す）。

### 負けの機序（84846187 / 84818805 / 84900107 に共通）
1. **Munkidori放置**: 3〜5体並ぶとGrimmsnarlが実効毎ターン90〜150回復になりJetting 120/turnでは落ちなくなる。同時にこちらのMegaへ毎ターン90〜150のチップが飛び、**Wallyの回復供給(手札のWally枚数)が先に尽きる**
2. **Mega Starmie=3プライズ液性**: 1体落ちると6→3で一気にレース劣勢。Munkidoriのダメカン移動が「Shadow Bullet圏外」を「圏内」に変えるため、330素体は数ターンで落ちる
3. **Unfair Stamp/Xerosicでループ切断**: KOを取った直後に手札2枚化され、Wally/エネが消えた次の相手ターンに主力が落ちる（84818805 t12-13が典型: Nebula KO→Stamp→Munkidori 120点移動+Shadow Bulletで430 Cape Megaが即死、3プライズ）
4. **ベンチの裸Staryu/Cinderace**がMunkidori移動+ベンチ30で狩られ追加プライズを献上
5. snipeを再利用Dunsparceに浪費し、Munkidoriが無傷で増え続ける

### 勝ちゲームで違ったこと（84756519 / 84753937）
1. **Munkidoriを能動的に排除**: Boss's Ordersで引き出しKO+ベンチsnipeで計2体処理(84753937)。ダメカン移動元が減ると回復レースが逆転する
2. **Cape+Wallyループの早期成立**: 430HPはShadow Bullet 180+Munkidori移動でも1ターンで落ちず、毎ターン全回復が間に合う(84756519は被弾4回全てWallyで帳消し)
3. **Crushing HammerをGrimmsnarl/Munkidoriに集中**: 表を引いた2回がそれぞれ「相手の攻撃不能ターン」を生み、その間にプライズ差を広げた
4. **Harlequinで相手の完成手札を流す**（Punk Up後のリソースを崩す）
5. 常に**W1枚のみアクティブに置く**運用でCrushing Hammer/エネ攻撃のリスクも最小化されている

### islet実装への含意（対Marnie改善分）
- snipe対象に「Munkidori(特性でこちらを削る系)最優先・山に戻る系(Dudunsparceライン)最劣後」の重み付けを入れる
- 「相手にUnfair Stamp級の手札リセットがある対面では、KOを取るターンにWally/エネを抱えず使い切る or 取らされKOを避ける」考慮
- 裸Staryuを相手ターンに晒さない(置く=そのターン進化が原則)

## エンジン仕様メモ（リプレイから確認できた事実）
- Mega進化exは通常の進化ルール（設置の翌ターン以降）。**Salvatoreだけが当ターン進化を許す**
- Cinderace Explosivenessは「セットアップ時に手札からアクティブへ裏向き配置可」。たね無しCinderaceのみ手札ではマリガン選択が発生する（＝キープも可能）
- Ignition Energyは進化ポケに付くとCCC（Nebula Beamのコストを単独で満たす）、ターン終了時に自動破棄
- Wally's Compassionの「エネを手札に戻す」は同ターンの手貼りとは独立（戻したWをそのターン貼り直せる）
- Kaggle episode JSONの構造: steps[i][p].observation（status==ACTIVEの行）への応答が steps[i+1][p].action。logsは同一観測の反復行で重複する（差分dedupe要）
