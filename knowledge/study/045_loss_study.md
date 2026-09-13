# 045_alakazam_full 弱点3対面の敗因精読スタディ（診断専用、実装なし）

- 日付: 2026-07-11
- 対象: 045_alakazam_full（81f1758c92フル精読レプリカ、EXP-045）の実弱点3対面
  1. 対b7cef02500型Kangaskhan（047_kangaskhan_replica）: プールv3実測 37% / 本番81f1側38.2%(n=34)
  2. 対Garchomp（041_garchomp_replica、元個体fc15d0c926）: 45.5%（本人も45.5%）
  3. 対Marnie（038_marnie_replica、元個体7308938c4d）: 42.5-45%
- データ源:
  - Part 1: `data/episodes/2026-07-09/` の Yushin Ito 本人（81f1758c92）の負け全リプレイ精読
  - Part 2: ローカル 045 vs 047/041/038 各200戦（arena/results/045_loss_study_vs{047,041,038}.json）の負けトレース精読
- 紐づくBACKLOG: I-087（対Kangaskhan）/ I-091（対Garchomp）/ I-084（対Marnie）/ I-090（デッキ変更系）

## Part 1: Yushin本人（81f1758c92）の負け内訳（07-09データ、250戦）

対象対面の全成績（本人視点）:

| 相手 | 全体 | 内訳（相手sig別） |
|---|---|---|
| Mega Kangaskhan ex | 15W-12L (55.6%) | **b7cef02500: 6W-8L (42.9%)**、89d834e4d4: 4W-2L、b44ef036a1: 1W-1L、その他 |
| Cynthia's Garchomp ex | 5W-6L (45.5%) | fc15d0c926: 3W-5L、98a4d6a80a: 2W-1L |
| Marnie's Grimmsnarl ex | 10W-3L (76.9%) | 負けは bc391c4243 / bd53b96ded / 108ac5bde2 各1（7308938c4dへの負けなし） |

精読対象: Kangaskhan負け12戦全部（b7cef×8含む）+ Garchomp負け6戦全部 + Marnie負け3戦全部 = 21戦。
レンダリング: `tools/render_replay.py`、要約: scratchpad/digest.py（ターン別盤面+攻撃+強制ディスカード+KOの抽出）。

### エンジン仕様の追認（精読で確認）

- **Powerful Hand（PH）はアクティブ限定**（カードテキスト: "Place 2 damage counters on your opponent's **Active** Pokémon for each card in your hand"）。ベンチは狙えない → Mist Energy付きアクティブの前ではPHは完全に沈黙する（Battle CageはPHに関係なし）
- Rapid-Fire Combo = 200+50×表（表が続く限りコイン）。Alakazam(140)は常にOHKO、期待値250
- Raging Curse = 「**自分の**ベンチのCynthia'sポケモンのダメカン×10、弱点無視」。こちらが付けた撃ち残しダメカンがそのまま自軍への砲弾になる
- Lillie's Determination = 手札を山に戻して10枚ドロー → Kangaskhan側は手札を山に還流でき**山切れ耐性が構造的に高い**

## 対面1: 対Kangaskhan（Yushin本人 15W-12L、対b7cefは6W-8L=42.9%）

### 敗因機序の頻度（負け12戦）

| 機序 | 頻度 | 説明 |
|---|---|---|
| **M1: 山切れ（プライズ有利のまま）** | **10/12** | 終局時 自deck=0〜2 vs 相手deck=5〜20。うち5戦は**プライズで勝っているか同数のまま**山切れ負け（残prz 1v3 / 2v4 / 2v2 / 3v4 / 1v3）。Dudunsparce/Fez/進化ドローのフル回転が毎ターン5〜10枚山を焼くのに対し、Kangaskhan側は消費2-3枚+Lillieで手札を山へ還流 |
| **M2: Mist EnergyへのPH空撃ち（0ダメージ）** | 9/12戦・計15回 | 相手はMist×4+付け直しでアクティブを恒常的に無効化。空撃ち自体は無料だが、そのターンのために焼いた山5-10枚が丸損（M1の主因）。85094122ではT8/T10/T12と3連続空撃ちで山22→5 |
| **M3: Xerosic×4の手札破壊** | 12/12戦・毎戦2-4回 | 溜めた手札（=PH打点）が3枚に戻される。最大17枚/20枚捨て。**ターンを跨ぐ手札の溜め込みが構造的に無価値**（自ターン内で引いて同ターンに撃つ分だけがXerosic耐性を持つ） |
| M4: タンク性能でPHの中打点が消える | 8/12 | Cape(+100)で実効400、Jumbo Ice Cream 80回復×複数、Lillieで引き直し。手札≤7のPH(手札×20=≤140)では2発でも落ちない |
| M5: ベンチ最小化でBossが死に札 | 4/12 | b7cef/b44efは盤面1-2体運用。Bossの吊り先が無く、3プライズバーストの的も消える |

注: 弱点(闘)はKangaskhan側に関係なし。負け筋は「壁+回復+手札破壊 vs 山切れ」の消耗戦構造そのもの。

### 決定的な分岐点の具体例

- **85074418 T13**（prz=1で勝ち1枚の場面）: BossでベンチのMega Kangaskhan(60/300)を吊ってPH → **Mist付きで0ダメージ**、勝ちターンを丸損 → そのまま山切れ負け（最終 自deck0/prz1 vs 相手prz3）。**Mist付きをBossで吊るのは自殺手**（045はR21実装済みだが本人はやる）
- **85094122 T8-T12**: Mist+Cape Crustle(270)アクティブに3ターン連続PH空撃ち、その間山22→5。T14にHammerを引いてやっと除去→KO、T16に3プライズ取ってprz2まで来たが**T17山切れ**。代替手: 空撃ちターンはドロー能力を止めて山温存（Hammer/Bossを引くまで最小行動）か、Dudunsparce Land Crush線
- **85076020 T11-T15**: 相手盤面Kangaskhan 400(Cape)1体のみ。PH空撃ち2回+Xerosic3連打（相手手札が小さく効果薄）で山3→0。手札にHilda×4が腐る。prz5v2の**大差リード状態から山切れ負け**
- 85057524/85081116（89d8型）: ベンチ0の超タンクCrustle(290, Cape, エネ6)に対しPH 0×2、26ターンかけて山切れ。**壁1体+ベンチ0は「勝ち筋が存在しない」盤面**になりうる（Boss不能・PH無効・Land Crush 90 vs IceCream回復80+Lillie）

### 本人の勝ちパターン（対比、勝ち15戦のサンプル確認）

勝ち戦は短い（中央値T9-11）: 序盤にMistが付く前のKangaskhan/Dwebbleへ**1ターン内でドローを全部回して手札12-17枚を作り、PH 240-340の単発バースト**で3プライズ機を2体抜く。ターンを跨ぐ溜めではなく**ターン内バースト**がXerosic耐性の本体。

## 対面2: 対Garchomp（fc15d0c926等、本人5W-6L=45.5%）

### 敗因機序の頻度（負け6戦）

| 機序 | 頻度 | 説明 |
|---|---|---|
| **G1: チップ撃ち残し→Raging Curse砲変換** | 4/4（長期戦全部） | PH 380を実効400(Power Weight)のGarchompに撃ち20/400残し（38ダメカン）、または200-220の中途打ち。ベンチに下がった傷Garchompのダメカン×10がSpiritomb Raging Curse **340-470**になり、こちらのAlakazam(140)どころか何でも即死。84995367ではT7とT9で2体を20/400にして、T12にRaging Curse470を受けた |
| **G2: 終盤クローズ失敗+山切れ** | 3/4 | prz=1-2に到達後、Xerosic+Corkscrew Dive圧で手札が小さく、Spiritombローテーションを抜けない。84995367は prz1で自deck0（山切れ）、85050963も自deck1で終局 |
| **G3: セットアップ事故（たね不足）** | 2/6 | T5即負け×2: 手札にAbraライン0（Fez単騎スタート/Dunsparce×2スタート）→ Corkscrew Dive 100圏で全滅。方策で救えない配り事故成分 |
| G4: Fez ex/Shayminがベンチで2プライズ献上 | 2/6 | 85030029: 相手Bossで裏のFez ex(210)とShayminを連続で吊られ、レース致命傷 |

- 本人はR23（チップ禁止）を**持っていない**: 400HPに380を平気で撃つ。045はR23実装済み（手札20未満は殴らない）だが45.5%で本人と同水準 → ローカル負けトレースでR23の実効性を検証（Part 2）
- Corkscrew Dive 100はAlakazamに-130〜-160（+30修正等）で2発、Raging Curseは1発。**傷Garchompを盤面に残す限り毎ターン全滅リスク**

## 対面3: 対Marnie（本人10W-3L=76.9%。負けはbc39/bd53/108acで、7308=038元個体への負けは0）

### 敗因機序の頻度（負け3戦）

| 機序 | 頻度 | 説明 |
|---|---|---|
| **N1: Shadow Bullet=弱点×2でAlakazam即死+ベンチ30スナイプ** | 3/3 | Alakazamは悪弱点、180×2=360で全員OHKO。ベンチ30がAbra(50)を2発圏、Shaymin(Flower Curtain)が出る前の序盤に盤面を削られる |
| **N2: 早期Munkidori量産→盤面全滅レース** | 2/3 | 84992728: T3にShadow Bullet起動、T6 Xerosicで16枚捨てさせられ、T9までに盤面がShaymin+Fezだけに。EXP-045の既知機序と一致 |
| N3: セットアップ事故（Fez単騎スタート） | 1/3 | 85035483: Fez ex単騎で開始しT3から180を受け続ける |
| N4: 420HP(Cape)Grimmsnarlに2発必要+2体目が続く | 2/3 | PH 320-340×2で1体は抜けるが、その間に1プライズ交換でこちらだけ枯れる |

本人が76.9%で勝っている対面であり、負け3戦は事故成分が大きい。**045ローカルの42.5%は本人と乖離**（038はトップ個体レプリカで平均Marnieより強い、EXP-045既出）→ ローカル負けトレースが本命（Part 2）。

## Part 2: 045自身のローカル負けトレース精読

再測定（200戦, seed42, errors=0）: **vs047 43.0% [36.3-49.9] / vs041 50.5% [43.6-57.4] / vs038 41.5% [34.9-48.4]**
（既知の実測 37% / 45.5% / 42.5-45% と整合）

トレーサ: scratchpad/trace_045.py（in-processでrender_replay.pyの整形関数を流用、負けゲームをreplay同形式で保存→digest.py）。

### vs 047_kangaskhan_replica（負け12戦精読、seed42 15ゲーム中12敗のサンプル）

**終局reason分布: deck out 7 / prizes 4 / no active 1**（さらにprizes負けのうちg4/g7は自deck 0-1で実質山切れ）。
**本人と同じ構造的敗因（M1/M2/M3）がローカルでも主因**。045固有のバグではなく、追加の劣化要素が3つ:

| 機序 | 頻度 | 本人との異同 |
|---|---|---|
| M1: 山切れ | 9/12（reason=deck out 7 + 実質2） | 同じ。deck out 7戦中6戦は**残プライズで有利か同数のまま**山切れ（残prz 4v5/3v3/3v4/2v5/3v4/2v4）＝レース自体は取れているのに山が先に尽きる |
| M2: MistへのPH空撃ち | 7/12戦・計10回 | 同じ（R21はキル候補からの除外だけで、空撃ちターンの山消費は止めていない） |
| M3: Xerosicで溜め手札消滅 | 12/12 | 同じ。**最悪例 g7: 山2枚まで掘って手札16枚→Xerosicで3枚に→そのまま詰み**。「山を掘り切って手札に溜める」行動がXerosic×4環境で自殺行為になっている |
| **L1(045固有): 1プライズの的ばかり倒す** | 6/12 | 本人の勝ち筋は「手札12-17のターン内バーストでKangaskhan(3prz)を2体抜く」。045はShaymin/Dwebble/Crustle(各1prz)をドリップで拾い、6キル要求のレースに自分から入っている |
| **L2(045固有): Hammer経済の失敗** | 5/12 | 序盤にGrow/Spiky相手へHammerを消費（g0 T3/T7）、終盤の本命Mist（打点を通す1ターン）に残っていない。Mist×4+付け直し vs Hammer×3 の消耗戦は撃ち所の選別が必須 |
| L3(045固有): 立ち上がり事故 | 3/12 | Abraライン不在ハンド（g4/g11/g13）→ RFC 200+に轢き殺される（no active 1件含む）。配り事故成分で方策では部分緩和のみ |

### 決定的な分岐点の具体例（ローカル045）

- **g2 T11**: Mist付きCrustle **10/170** に PH空撃ち0。ベンチにKadabra+手札に{P}エネがあり、エネ付与→リトリート→Super Psy Bolt 30（通常ダメージ=Mist貫通）で確殺できた。**「Mist相手はダメージ攻撃に切替」の具体的欠落**
- **g7 T12-T13**: 山12→2まで1ターンで掘って手札16枚に（PHは空撃ち0）→ 相手Xerosicで3枚→山1で敗着。掘る前に「この手札でこのターン殺せるか」の判定がない
- **g0 T21-T23**: Hammer切れ状態でMist Crustleに40→0とドリップ、山2→0。残り山でHammer/Bossを引く展望もなくドロー能力をフル回転
- **g11**: 相手Crustle×4+Kangaskhanの盤面で045は終始Kadabra止まり（Xerosic連打が刺さる）、prz 6v0 完封負け

## 対面2ローカル: vs 041_garchomp_replica（負け12戦精読、seed42 18ゲーム中12敗のサンプル※）

※トレース用シードでは負けが偏った（200戦実測は50.5%）。分類には十分。
**終局reason分布: prizes 6 / no active 5 / deck out 1**。

| 機序 | 頻度 | 本人との異同 |
|---|---|---|
| **G-L1(045固有バグ): Rock Fighting EnergyへのPH空撃ち** | 4/12戦・5回 | **045の無効エネモデル（R21）がMist(id11)とArticuno Veilしか知らない**。041はRock Fighting Energy(id20, "Prevent all effects"=Mist同等)×4を採用しGarchompに常時貼る。g8 T13/T15: 手札13-17でBossで吊ったGarchompにPH→0ダメージ×2。本人も85000547 T15で同じ0を踏んでいる |
| G-L2: キル皆無の完封負け | 5/12 | prz=6のまま敗北が4戦（g8/g9/g13/g16、本人には無いパターン）。立ち上がり失敗+PH無効+Corkscrew Dive(100+Roserade補正30×n)でAlakazamが毎ターン落ち、一度も大打点を通せない |
| G-L3: 早期盤面全滅（no active） | 5/12 | うち3戦はT5-T7の即死（Abraライン不在のdudハンド、本人のG3と同じ）。2戦は中盤に盤面が痩せて全滅 |
| G-L4: Raging Curse被弾 | 6/12戦 | 本人と同じ機序（チップ→砲変換）。ただし045はR23実装済みで380残しは減っており、被弾の多くは30-100のSPBチップや相打ち時の残骸から |
| G-L5: Fez exが2プライズ献上 | 4/12 | 相手Boss→Corkscrew DiveでベンチのFezを狩られる（本人のG4と同じ） |

## 対面3ローカル: vs 038_marnie_replica（負け12戦精読、seed42 15ゲーム中12敗のサンプル）

**終局reason分布: prizes 9 / no active 2 / deck out 1**。PH-0は0回（Marnieに無効カード無し）。

| 機序 | 頻度 | 本人との異同 |
|---|---|---|
| N1: Shadow Bullet=悪弱点×2でAlakazam毎ターンOHKO | 12/12 | 本人と同じ構造。180×2=360、Alakazam(140)/Kadabra(80)即死。ベンチ30スナイプはShaymin(Flower Curtain)が着地後は止まっている（g5でAbra HP 0を確認）＝R実装は機能 |
| **N-L1(045固有): Fez exの2プライズ献上** | **7/12** | 038はBossキル特化レプリカ（EXP-038）。ベンチのFez ex(210)をBoss→Shadow Bullet 360で狩る。038の必要プライズ6のうち2を1手で献上し、実質「4キルで負け」のレースにしている。Yushinの負け3戦では顕在化していないが、本人スタディでも「自分のFezは同じliability」と明記（R13の裏返し） |
| N2: 早期全滅（T2-T4 Shadow Bullet起動） | 4/12 | 038はT2-T3にShadow Bulletが起動する turbo型。dudハンドと重なると no active（g2: 22 steps） |
| N-L2: クローズ失敗の僅差負け | 4/12 | prz 1-2 vs 0 で敗北（g11/g14/g3/g4/g9）。045はGrimmsnarl 2体は抜けるが、320HPに280チップ（手札14）を置いてCape付与+回しに逃げられるターンロスが響く（g14 T7）。OHKOライン=手札16(320)/21(Cape420) |

## 修正候補ルール案（F番号、islet式条件文で実装可能な粒度）

優先度順。各案の紐づけ: **I-087=対Kangaskhan / I-091=対Garchomp / I-084=対Marnie / I-090=デッキ変更**。

### F1 [I-087] 消耗戦モードの山経済（最重要・期待効果最大）

- **条件文**: 相手デッキがKangaskhan/Crustle系（Mega Kangaskhan ex or Crustle/Dwebbleを盤面・トラッシュに検知）のとき、`deck_count <= 20` で任意ドロー（Psychic Draw ACTIVATE / Dudunsparce / Fez / 手札中立でないPoffin・Pad連打）を「このターンの投射打点(手札×20)でアクティブの実効HPを取り切れる見込みがある場合のみ」に制限。現行R6は `kills_now & hand>=9 & deck<=12`（+deck<=4ガード）で、キル不能な消耗戦では deck=4 まで掘り続ける
- **期待効果**: vs047の敗因1位（山切れ9/12、本人も10/12）の直接除去。相手はLillieで手札を山へ還流するため山切れ戦争は構造的に不利＝掘らない選択が唯一の防御。37-43%→50%圏を狙う本命
- **根拠試合**: g7（山12→2まで掘って手札16→Xerosicで3枚→詰み）、g0（山2→0、prz4v5）、本人85094122（山22→5を空撃ち3ターンで焼き、prz2v4のまま山切れ）
- **注意**: PH打点=手札のため一律抑制は打点も削る。「バースト成立ターンだけ全開で掘る」(F5)とセットで設計

### F2 [I-087/I-091] 効果無効アクティブへの空撃ちターン最小化（R21の拡張）

- **条件文**: PH予測ダメージ0（対象にMist/Rock Fighting付き or Veil下たねTR）のとき、優先: (a)HammerでMist系を剥がしてから撃つ (b)Bossで殺せる裏を吊る (c)通常ダメージ攻撃へ切替（Kadabra Super Psy Bolt 30 / Dudunsparce Land Crush 90。**ダメージは効果でないためMist系を貫通**、エンジン実測済み） (d)いずれも不能なら**ドロー能力も止めてEND**（山温存、F1と連動）
- **期待効果**: PH-0はvs047で18回/9戦、vs041で5回/4戦。1回の空撃ちターン=山5-10枚の丸損
- **根拠試合**: g2 T11（**Mist付き10/170のCrustleにPH-0。Kadabra+手貼りでSPB 30の確殺線があった**）、本人85074418 T13（prz=1でMist付きをBossで吊ってPH-0→そのまま山切れ負け）

### F3 [I-091] 無効エネモデルにRock Fighting Energy(id20)を追加（最安価・バグ級欠落の修正）

- **条件文**: `Mist_Energy = 11` 参照箇所（mist_count/killable判定/Hammer優先）を「効果無効エネ集合 {11, 20}」に一般化
- **期待効果**: vs041のPH-0 5回を除去。Hammer対象にもRock Fightingを追加（041はHammerで剥がせる特殊エネ×8: Rock Fighting4+α）
- **根拠試合**: g8 T13/T15（手札13+でBoss+PHの決め手が2ターン連続0ダメージ）、本人85000547 T15

### F4 [I-087] Hammer経済の対面ゲート（R17の修正）

- **条件文**: 対Kangaskhan（Mist×4採用）では、Enhanced HammerはPH対象のMistのみに使う（Grow Grass/Spikyへは温存）。さらに「Hammerで剥がす→同ターンPH」のセットが成立するターンまで手札に保持
- **期待効果**: 終盤の「Mistを剥がせず空撃ち連発」の緩和（g0: T3/T7にGrow/Spikyへ浪費→T21-23のMist Crustleに1枚も無し）
- **注意**: R26捨て順でHammerは中位。Xerosic被弾で捨てさせられる分は防げない

### F5 [I-087] 3プライズバースト志向のターゲット評価

- **条件文**: 対Kangaskhanのキル対象評価で「ターン内バーストでMega Kangaskhan ex(3prz)を取る」プラン（必要手札=15/実効300、20/Cape400）に大幅加点。1prz対象（Shaymin/Dwebble/Crustle）のドリップより優先。ドローループ全開はバースト成立見込みターンのみ（F1の裏面）
- **期待効果**: 本人の勝ちパターン（Kangaskhanバースト×2、勝ち中央値T9-11）の再現。045は1prz×6キルのレースに入って山が先に尽きる（L1: 6/12）
- **根拠試合**: 本人85100743 T14（Boss+Hammer+PH340でKangaskhan KO）vs 045のg0（Shaymin/Dwebble/Crustleを260/100/200で拾いprz4止まり）

### F6 [I-084/I-091] Fez exベンチ出し抑制の対面ゲート

- **条件文**: 相手が悪タイプ高打点（Marnie系: Grimmsnarl/Munkidori/Impidimp検知）またはBoss+高打点（Garchomp系）のとき、Fez exを手札から出さない（既にR3でセットアップ時は出さない。中盤のドロー目的の着地を禁止）。すでに盤面に居る場合はリトリート優先度を上げる
- **期待効果**: vs038で7/12戦、vs041で4/12戦がFezで2プライズ献上。相手の要求キル数を6→4に自ら短縮している。Fezドロー(ABILITY)の便益とのトレードオフなのでアブレーション必須
- **根拠試合**: g5 T4（038がBossでFezを吊り2prz）、g14（Fez 210→50まで削られ続けAlakazamと共倒れ）

### F7 [I-091] Spiritomb被弾モデル（R23の補完）

- **条件文**: 相手ベンチにCynthia's Spiritombがいるとき、被弾予測=「相手ベンチCynthia'sの合計ダメカン×10」。これが140以上なら次の自アクティブは即死前提でプランする（昇格はDunsparce壁を差し出す、エネ付きAlakazamをアクティブに置かない）。加えてSPB/Teleportationの30-10チップをCynthia's進化前に置かない（Raging Curseの燃料になる）
- **期待効果**: Raging Curse被弾6/12戦の損害軽減。R23（撃ち残し禁止）は実装済みだが、小チップと相打ち残骸の分が残っている
- **根拠試合**: g8（SPB 30×3をGible/Gabite/Garchompに分散→Roserade Cheer On+Corkscrewで毎ターン即死レース）、本人85050963（200/400×2体残し→Raging Curse 470×2被弾）

### F8 [I-087] 終盤クローズ用のBoss温存

- **条件文**: 対タンク系で自分の残プライズ≤2のとき、R26捨て順のBoss(余剰)を最後尾へ（Hammer/Xerosicより温存）。prz=1なら「Boss+小PH」で殺せる1prz対象（Shaymin80/Dwebble70）が常に勝ち筋
- **根拠試合**: 本人85104353（prz=1でT16-T20の5ターン、ベンチにShaymin(80)が居るのにBossが手札に無く300HP本体を殴り続けて山切れ）

### デッキ変更が必要な対策（I-090、方策案とは別枠）

- **Tool Scrapper 1-2枚の再投入**: 81f1リストはa737から Tool Scrapper/Battle Cage を抜いた形。Cape剥がしで (a)Kangaskhan実効400→300（バースト要求手札20→15） (b)038のCape付きGrimmsnarl 420→320（OHKO要求21→16） (c)Crustleタンク290→190。3弱点対面すべてに効く。Marnieトップ(7308)の対Kangaskhan勝ち筋も「Tool Scrapper+Boss」だった（EXP-047）。候補カット: Nighttime Mine 3→2 か Lana's Aid（R19/R15の使用頻度が低い）
- Enhanced Hammer 3→4: Mist×4+Rock Fighting×4に対する剥がし枚数の増強（F4と代替関係、枠経済はEXP-013の教訓に注意）
- 注意: 1枚テックでも有意劣化の前例（EXP-013）。必ず400戦アブレーション

## 結論（対面ごとの敗因機序トップ3）

- **対Kangaskhan(b7cef, 37-43%)**: ①山切れ（本人10/12・045 9/12、多くはプライズ有利のまま）②Mist空撃ちで山を無駄焼き（PH-0計18回/9戦）③Xerosic×4でターン跨ぎの手札溜めが無価値化。→ F1+F2+F5（山経済+バースト志向）が本命、F4/F8が補助
- **対Garchomp(fc15, 45-50%)**: ①Rock Fighting Energy無効を知らないPH空撃ち（045バグ、F3で即修正可）②チップ→Raging Curse砲変換（R23済みだが小チップ分が残る、F7）③立ち上がり事故+Fez献上（F6）
- **対Marnie(7308系, 41-45%)**: ①悪弱点×2の構造負け（方策で不可避）②**Fez ex献上で相手の要求キル6→4**（7/12戦、F6が本命）③クローズ時のチップ置き（280/320）でOHKOライン管理が甘い
- **最も期待効果の大きい単一修正**: F1（対Kangaskhan山経済）。最重要ウィークネス対面の敗因1位で、本人も同じ穴＝直せば本人超えの余地が大きい
- 045側の実装が本人比で劣化になっている点は「Rock Fighting無効の未知（F3）」と「1przドリップ志向（F5）」の2つ。残りは本人と同じ構造的敗因

## 再現手順

```
# Part 1: 本人リプレイ
uv run tools/render_replay.py <episode_id> --zip data/episodes/2026-07-09/pokemon-tcg-ai-battle-episodes-2026-07-09.zip --player <idx>
# 対象episode: 本文の表参照（Kangaskhan 12 / Garchomp 6 / Marnie 3）
# Part 2: ローカル再測定
uv run arena/run_match.py agents/045_alakazam_full agents/047_kangaskhan_replica -n 200 -w 2 --json arena/results/045_loss_study_vs047.json
uv run arena/run_match.py agents/045_alakazam_full agents/041_garchomp_replica -n 200 -w 2 --json arena/results/045_loss_study_vs041.json
uv run arena/run_match.py agents/045_alakazam_full agents/038_marnie_replica -n 200 -w 2 --json arena/results/045_loss_study_vs038.json
# 負けトレース: scratchpad/trace_045.py（in-process、render_replay.py流用）+ digest.py
```
