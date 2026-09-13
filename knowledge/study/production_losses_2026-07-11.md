# 本番リプレイ精読: 045_alakazam_full の負け全戦診断（2026-07-11）

- 対象: **045_alakazam_full**（提出54525332、`data/my_episodes/54525332/`、77戦）
- 成績: **46W-31L = 59.7%**（LB表記と一致。うち1戦は自チーム対戦=85193422で、player0側=本提出の勝ち扱い。負け分析は31戦）
- スコープ注: 当初は043（54523110）も対象だったが、親指示により**045のみ**に変更。043は未着手（打ち切りコストなし）
- ツール: scratchpad/agg.py（集計）、render_my.py（個別JSON対応レンダラ、render_replay.py関数流用）、movetimes.py（per-move時間）、decks.py（相手60枚ダンプ）

## 0. リプレイ形式と身元判定の注意（再現時の必読）

- my_episodesの個別JSONは**logsが差分形式でRESULTログを含まない**（日次zipの累積形式と別物）。敗因reasonは最終盤面+精読から推定
- **身元判定はdeck照合では壊れる**: 045のdeck.csvはYushinの81f1758c92と同一60枚のため、81f1系の相手と当たると両者がマッチする。**`info.TeamNames`の'haruto'インデックスを正とする**こと（このバグで初回集計は「Yushin本人に0-2」という誤結論を出しかけた。実際は81f1コピー使い213tuboに1勝+自チーム戦1）
- per-move時間はstepごとの`remainingOverageTime`差分で取れる（§6のlethal検証に使用）

## 1. 全体成績と相手分布（本番77戦）

| 相手（実体ベース） | 勝敗 | 勝率 | シェア | 備考 |
|---|---|---|---|---|
| Alakazam（自チーム戦除く） | 13-6 | 68% | 24.7% | 402d88b918(81f1の亜種)×9が最多。81f1コピー(213tubo)に1-0 |
| **Archaludon ex**（classifyは"Cinderace"/"Mega Mawile ex"誤分類、9de969/c09fb含む） | 10-7 | 59% | **22.1%** | 本番最多実体。負け7のうち4はT2-3初手全滅（除くと10-3=77%） |
| Mega Lucario ex | 8-2 | 80% | 13.0% | 負け2は相手のT2-3 Mega Brave級ぶん回り |
| Dragapult ex | 3-3 | 50% | 7.8% | **2b31cd4174に0-3**、他sigに3-0 |
| Cynthia's Garchomp ex | 3-2 | 60% | 6.5% | fc15d0c926（041の元個体）×3 |
| Marnie's Grimmsnarl ex | 3-2 | 60% | 6.5% | 7308938c4d（038の元個体）に0-1 |
| Iono's Bellibolt ex | 2-2 | 50% | 5.2% | da04d37318×4 |
| **Cubchoo系ハンマーストール**（classify: Ogerpon/Cubchoo/Dudunsparce(9926)） | 0-3 | 0% | 3.9% | §3b。TR Honchkrow(妨害系)も含めると**0-4** |
| Mega Kangaskhan ex | 1-1 | 50% | 2.6% | **両方89d834e4d4（旧039型）。b7cef02500（047型）は77戦中0出現** |
| Crustle | 1-1 | 50% | 2.6% | f16dcb22ae(壁型)に勝ち、363629eb70(Great Tusk攻撃型)に負け |
| TR Honchkrow / Mega Starmie ex | 0-2 | 0% | 2.6% | |
| Dudunsparce(Hop's型15dbd) | 1-0 | — | 1.3% | |
| 自チーム戦（045同型ミラー） | 1-0 | — | 1.3% | 85193422。もう1つのharuto枠も81f1デッキ=045系 |

- **このレート帯の本番メタはローカルプールと大きく違う**: Archaludon実体22.1%（プールweight 3.0）、Lucario 13%（3.7）、Alakazam 24.7%（40.4）、Kangaskhan 2.6%（13.0）。警戒していたb7cef Kangaskhanとは**一度も当たっていない**
- それでも総合59.7%はプールv3 weighted 56.7%[54.3-59.2]のCI上限付近で整合

## 2. 負け31戦の敗因分類（トップライン）

| 主因 | 戦数 | 率 | 内訳 |
|---|---|---|---|
| **D: 山切れ（自deck=0）** | 12/31 | 39% | **うち9戦はプライズ有利or同数のまま**。相手: ストール/妨害系4、Alakazamミラー4、Archaludon 2、Kangaskhan 1、Garchomp 1 |
| **P: プライズレース負け** | 13/31 | 42% | 構造負け4（Marnie悪弱点2、Dragapultスナイプ2）+ 相手ぶん回り3（Lucario2、Starmie1）+ 接戦6 |
| **W: 初手全滅（T2-3、ベンチ0）** | 6/31 | 19% | **全戦「初手にたね1枚のみ」**（Dunsparce/Abra単騎）を確認。純粋な配り事故。相手: Archaludon×4、Dragapult×1、Cubchoo系×1 |

ローカル診断（045_loss_study.md）のM1（山切れ）/M2（無効対象へのPH空撃ち）/M3（Xerosic）が本番でも主因であることを確認。**山切れは本番の敗因1位**。

### 方策で救えた可能性が高い負け（精読ベース、§5）

| episode | 相手 | 取りこぼした点 | 救済確度 |
|---|---|---|---|
| 85205262 | Honchkrow(TR妨害) | **Veil下Porygon(60HP)にPH-0×6。Super Psy Bolt 30×2/Land Crush 90で普通に倒せた**（Veilは効果のみ無効、ダメージは通る） | 高（バグ級） |
| 85193990 | Garchomp fc15 | Rock Fighting PH-0（既知F3バグ）+380/400チップ残し→Raging Curse+prz3-1リードで山切れ | 中 |
| 85214552 | Archaludon c09f | **手札にBoss×2を持ちながらPH380-460をDuraludon(130HP,1prz)に過剰投射×5。Boss→Archaludon ex(300)バーストなら2prz×2で勝ち筋** | 中 |
| 85199362/85205764/85208679 | Alakazamミラー | プライズ同数〜1差の消耗戦で**自分だけ4-7枚早く山切れ**（ミラーは同一60枚なので山経済の差=引き癖の差） | 中（3戦で2戦程度） |
| 85280170/85200291 | Cubchooストール | Snotted Upロック中もEND TURNだけで山を燃やし続けた（85200291は相手残9枚まで肉薄=山温存なら勝ち筋あり） | 低-中 |

## 3. ローカルプール不在の相手の正体（テールスポットチェック素材）

### 3a. 「Cinderace」= Archaludon ex（本番シェア実質22.1%、最多相手）

classify_deckの既知バグ（INSIGHTS: Cinderaceエンジンがstage2のためstage1のArchaludon exから看板を奪う）。実60枚（debbe71c08代表、7戦）:

```
Duraludon×4 / Archaludon ex×4 (300HP) / Cinderace×4 / Relicanth×1
Basic Metal×11、Full Metal Lab×4（M全体-30壁）、Jumbo Ice Cream×4（回復）、
Poké Pad/Ultra Ball/Pokégear/Explorer's Guidance/Lillie's Determination×4、
Boss's Orders×4、Night Stretcher×3、Hero's Cape×1
```

- 4リスト確認（debbe/3b84/c3e7/7ca0）+ Mawile1枚テックのb0213。**Mist/Rock Fighting不採用→PH空撃ちリスクなし**、Xerosicはc3e7/7ca0のみ×2-3
- 045は実体合計10-7(59%)、初手全滅4を除く実力対面は10-3。負け筋は「300(Cape400)タンク+IceCream回復+Lillie還流を削る間の山消耗」
- ローカル代表046（90dd411e61）は**Articuno壁+手札刈り型で、本番主流のdebbe型とは別物**（§7参照）

### 3b. Cubchoo系ハンマーストール（0-3、+同系Honchkrowで0-4）

3リスト（36b4960b0a"Ogerpon" / 27b6fa193a"Cubchoo" / 9926811bf8"Dudunsparce"）は同族:

```
共通コア: Cubchoo×3-4 + Crushing/Enhanced Hammer×3-8 + Battle Cage +
Gravity Gemstone + Neutralization Zone×1 + Lillie×4 + Boss + Xerosic +
TR Petrel/Watchtower。アタッカーはほぼ無し（Ogerpon ex×1 / Munkidori×3 等）
```

- **ロック機序（85280170精読で確定）**: Cubchoo「Snotted Up 10」=**次の相手ターン、防御ポケモンは攻撃不可**。毎ターン撃たれると自アクティブは恒久攻撃ロック。Gravity Gemstone（両アクティブ逃げ+1）+Hammer連打でリトリートエネも枯らされ、045は**t11-t19をEND TURNのみで過ごし**義務ドロー+序盤の特性ドローの分だけ先に山切れ
- 唯一の脱出線は「ベンチにエネを溜めて毎ターンリトリート回転（新しいアクティブはロック未付与なので攻撃可）」だが045はベンチ付けはするのにリトリートしない
- 勝ち筋は山切れ一本のデッキ。85280170はキル4つ(prz4/6)リードのまま自deck0、85200291（30ターン）はprz5/6・相手残り9枚まで来て自分が先に切れた
- TR Honchkrow（8bcfabb2c1）は妨害サポ連打型。85205262は18ターン両者キル0のまま045だけdeck0（§5c）
- **本番シェア約5%で0勝、ローカルに相手役ゼロ** → テールスポットチェック最優先の一族

### 3c. Iono's Bellibolt ex（da04d37318、2-2）

```
Iono's Voltorb/Tadbulb/Bellibolt ex(280)×各3 + Wattrel/Kilowattrel×各3
Basic Lightning×22、Canari×4（雷ポケモン4枚サーチ）、Levincia×3、Lillie×4
```

- 雷単ビートダウン。エネ22枚で事故が少ない。負け2はどちらも4-5/5-3級の接戦レース負けで構造問題は見えない

### 3d. その他

- **Crustle 363629eb70**: 壁型でなく**Great Tusk×4/Terrakion+Rock Fighting×4/Mist×4**の攻撃型闘デッキ。85197378は16ターンかけて我々のPHが3回全部0（Mist/Rock Fighting）・キル0-0のまま自deck0。**042（壁型）と別物で、045には現状勝ち筋の薄い型**
- **Kangaskhan 89d834e4d4**（旧039元個体、Crustleハイブリッド）: 負け85213558（34T）は**単騎Mist Crustle相手にPH-0×8**でprz2/6リードのまま山切れ（ローカル既知の「壁1体+ベンチ0=勝ち筋なし」盤面の本番実例）
- **Mega Starmie 69fc361f09**: Dusknoirライン入り。負けはT2 Jetting Blowで2prz級の速攻+こちら盤面薄（事故成分大）

## 4. 負け全31戦の戦別分類表

分類: 構造（デッキ相性/ゲームプラン不成立）/ 方策（分岐点特定可）/ 事故（配り・ぶん回られ）。

| episode | 相手（実体） | T | prz(me/op) | 終局 | 分類 | 敗因 |
|---|---|---|---|---|---|---|
| 85193990 | Garchomp fc15 | 14 | 3/5 | 山切れ | **方策+バグ** | prz3-1リードのままdeck0。Rock Fighting PH-0×1、Garchomp 20/400チップ残し→Raging Curse被弾、Hilda×n腐り |
| 85196298 | Archaludon debbe | 12 | 2/1 | プライズ | 接戦 | 4-5の接戦負け |
| 85197378 | Crustle 3636(Tusk型) | 16 | 6/6 | 山切れ | 構造 | Mist/Rock FightingでPH3発全部0、両者キル0のまま自deck0 |
| 85197528 | Lucario 7236 | 12 | 6/1 | プライズ | 事故 | 相手T2 Mega Lucario Aura Jab→Mega Brave 270連打。こちら攻撃1回のみ |
| 85199362 | Alakazam 2fc6 | 20 | 4/4 | 山切れ | **方策** | ミラー同数のまま自deck0（相手残4） |
| 85200291 | Cubchooストール 36b4 | 30 | 5/6 | 山切れ | 構造+方策 | prz1-0リード、Snotted Upロックで攻撃不能のまま30T。相手残9まで肉薄 |
| 85201311 | Marnie 108ac | 10 | 4/1 | プライズ | 構造 | 悪弱点360+ベンチAbra狩り。PH280/320はCape圏で撃ち残し |
| 85202278 | Alakazam 402d | 14 | 5/1 | プライズ | 事故寄り | 相手T2からPH攻撃サイクル完成（Rare Candy高速）、こちら初攻撃T12 |
| 85205262 | TR Honchkrow 8bcf | 18 | 6/6 | 山切れ | **方策（バグ級）** | Veil下Porygon(50/60)にPH-0×6。SPB30×2/LandCrushで倒せた。18T両者キル0で自deck0 |
| 85205764 | Alakazam 03bb | 12 | 3/3 | 山切れ | **方策** | ミラー同数、自deck0（相手残6） |
| 85206718 | Archaludon c3e7 | 13 | 2/1 | 山切れ | 接戦 | 4-5接戦。相手はXerosic×3型+消費299.5s（探索系?） |
| 85208679 | Alakazam b196 | 16 | 2/1 | 山切れ | **方策** | 4-5接戦、自deck0（相手残7） |
| 85210634 | Archaludon 7ca0 | 3 | 6/6 | 全滅 | 事故 | 初手Dunsparce単騎 |
| 85213065 | Cubchooストール 9926 | 2 | 6/6 | 全滅 | 事故 | 初手Dunsparce単騎 |
| 85213558 | Kangaskhan 89d8 | 34 | 2/6 | 山切れ | 構造 | prz4-0リード→単騎Mist CrustleにPH-0×8、T18以降勝ち筋なし |
| 85214552 | Archaludon c09f | 18 | 2/6 | 山切れ | **方策** | Boss×2温存のままPH380-460をDuraludon(130,1prz)に過剰投射。Articuno Veil PH-0×1 |
| 85215027 | Bellibolt da04 | 12 | 2/1 | プライズ | 接戦 | 4-5接戦 |
| 85217006 | Marnie 7308(038元) | 10 | 6/1 | プライズ | 構造 | T3 Shadow Bullet起動のturbo型、完封負け（ローカル42.5-45%と整合） |
| 85217508 | Archaludon debbe | 2 | 6/6 | 全滅 | 事故 | 初手Abra単騎 |
| 85218545 | Archaludon debbe | 2 | 6/6 | 全滅 | 事故 | 初手Dunsparce単騎 |
| 85220848 | Lucario 7143 | 12 | 6/1 | プライズ | 事故 | 相手T3 Mega Lucario+Solrock。こちらTeleportation 10のみ |
| 85222146 | Starmie 69fc | 6 | 6/2 | 全滅 | 事故 | T2 Jetting Blow速攻、盤面が育つ前に全滅 |
| 85222333 | Dragapult 2b31 | 8 | 4/2 | 全滅 | 構造 | Phantom Dive 200+ベンチ60散布で裸Abra群が毎ターン消滅 |
| 85241786 | Dragapult 2b31 | 2 | 6/6 | 全滅 | 事故 | 初手Dunsparce単騎 |
| 85246130 | Alakazam 428a | 14 | 3/1 | プライズ | 構造+方策 | 相手はPH400-480×4の溜め撃ち型。手札比べで完敗 |
| 85254334 | Dragapult 2b31 | 8 | 6/2 | 全滅 | 構造 | Phantom Dive散布でベンチAbra2-3体同時KO、キル0全滅 |
| 85259370 | Alakazam f17b | 20 | 6/1 | 山切れ | 構造+方策 | 攻撃サイクル崩壊（PH -200/-40/-60の3発のみ）+deck0。相手Xerosicで手札枯渇 |
| 85260504 | Garchomp fc15 | 20 | 5/2 | プライズ | 構造 | 1-4ビハインドから粘るも届かず（残deck1=山経済も限界） |
| 85278255 | Archaludon debbe | 3 | 6/6 | 全滅 | 事故 | 初手Dunsparce単騎 |
| 85280170 | Cubchooストール 27b6 | 19 | 4/6 | 山切れ | 構造+方策 | キル2先行→Snotted Upロック+Hammerでt9以降攻撃0、END TURNのみで山切れ |
| 85299607 | Bellibolt da04 | 14 | 3/1 | プライズ | 接戦 | 3-5接戦 |

## 5. 主要精読の詳細

### 5a. Alakazamミラーの負け7戦（本番13-7の負け側）

- **消耗戦型3戦（85199362/85205764/85208679）**: プライズ同数〜1差の互角レースで、終局時**自deck0 vs 相手残4-7枚**。同一（ほぼ同一）60枚のミラーなので、この差は純粋に「特性ドローの回し方の差」。**負け幅4-7枚は方策で埋まる距離**（逆に85209190では相手を山切れさせて勝っており、ミラー山切れレースは1勝3敗）
- **テンポ負け2戦**: 85202278は相手がT2からPHサイクル完成（こちら初攻撃T12）。85246130は相手が手札20超を溜めてPH400-480×4（うちのXerosicが間に合わず）。402d/428a系は81f1と微差のリストだが、**「ターン内バースト」でなく「数ターン溜めてから大PH」型の亜種**が存在する
- 85259370は相手Xerosic×2+Hammerでこちらの攻撃サイクルが崩壊（20TでPH3発計300点）した完敗

### 5b. 対Garchomp 85193990（prz3-1リードから山切れ）

ローカルスタディの3機序が本番でも同時発生することを確認:
1. T13 PH→**Rock Fighting Energy付きGarchompに0ダメージ**（F3バグ、無効エネモデルがMistとVeilしか知らない）
2. Garchomp ex 20/400のチップ残し（PH -360/-180/-200の分割投射）→ T13 **Spiritomb Raging Curse被弾**（R23実装済みでも380/400残しが起きる=手札係数の読みがCape/Power Weight込みで甘い）
3. Hilda×2、Boss×2を余らせたまま自deck0（prz3-1リード時点で山を掘り続けた）

### 5c. 対TR Honchkrow 85205262（バグ級の見逃し、勝てた試合）

- 相手はT5から**Porygon(50/60)をArticuno Repelling Veil下でアクティブに置きっぱなし**。045はT5-T15の6回PHを撃ち**全部0ダメージ**、その間Fez/Dudunsparce/Psychic Drawで山を焼き切りT17に自deck0（相手残32）。両者プライズ0-0
- **Repelling Veilの正確な仕様（カードテキスト確認）**: 「Prevent all **effects** of attacks... **Damage is not an effect.**」= 効果のみ無効。**Super Psy Bolt 30/Land Crush 90/Teleportation 10等の通常ダメージは素通し**。Porygon 60HPはSPB×2で落ち、Murkrow 80も3発。**盤面に常時いたKadabra/Dudunsparceで普通に殴れば勝てた**
- 045のR21はVeil対象を「キル候補から除外」するだけで、(a)ダメージ攻撃への切替、(b)攻撃が無意味なターンのドロー停止、のどちらも持っていない（ローカルスタディF2の(c)(d)がそのまま本番で失点した形）
- 同型の見逃しは85213558（Mist Crustle単騎にPH-0×8。こちらはDudunsparce Land Crush 90×2ならIceCream回復80を上回れた）にも

### 5d. 対Archaludon 85214552（ターゲット選択の過剰投射）

- PHを-220/-400/-380/-380/-460と5回撃ち、うち4回は**Duraludon(130HP、1プライズ)への200-330点過剰投射**。その間、手札にBoss's Orders×2が眠っており、**Boss→Archaludon ex(300)+PH380-460なら2プライズ×2**が取れた。4プライズ取ったところで自deck0（相手残31）
- ローカルスタディF5（マルチプライズバースト志向）のArchaludon版。「forced activeを殴り続ける」ドリップ癖はKangaskhan対面のL1と同根

### 5e. 対Dragapult 2b31cd4174（0-3の構造穴）

- Phantom Dive 200+**ベンチ60点散布**が、045のセットアップ様式（裸Abra 50HPを3-4体並べる）に直撃。85254334はT6のPhantom Dive 1発でベンチAbra×2同時KO+3prz、85222333も毎ターンAbra/Kadabraが蒸発
- 2b31はプールnote既載の強個体（07-09時点n=195、56.4%）。**公式方策021（045が67%勝つ）とは別物**。「Abraを裸で並べすぎない」（EXP-045派生アイデア）の本番裏付け

## 6. lethal探索の本番動作検証（発火の直接的傍証）

per-move時間 = stepごとの`remainingOverageTime`差分（77戦、自分の全5,022手）:

| 自残プライズ | 手数 | 中央値 | p90 | 最大 |
|---|---|---|---|---|
| 6 | 2,822 | 30.7ms | 35.1ms | 0.06s |
| 5 | 792 | 30.7ms | 35.2ms | 0.05s |
| 4 | 497 | 32.2ms | 36.1ms | **0.04s** |
| **3** | 398 | 54.1ms | 102.4ms | **0.35s** |
| **2** | 271 | 53.6ms | 226.5ms | **0.77s** |
| **1** | 216 | 105.5ms | 356.1ms | **0.77s** |

- **prz≥4は最大0.06s（greedy即答+環境オーバーヘッド約30ms）、prz≤3に入った瞬間p90が3〜10倍・最大0.77s** = lethal探索のゲート（残プライズ≤3のMAINのみ）と正確に一致。**本番評価環境で探索が設計どおり発火している直接的傍証**（007での「動く」確認に続き、045でも確認）
- ゲーム合計消費: 中央3.1s / p90 4.9s / 最大8.5s（600s予算の1.4%）。1手>1sは0回、statuses全DONE、タイムアウト・エラー0
- **時間予算は98%以上未使用**。ゲート緩和（prz≤4化・SAMPLES増・候補数増）の時間的余地は大きい

## 7. 本番⇔ローカル乖離表

本番nは小さく個別対面CIは±20-35pt。太字は解釈に値する乖離のみ。

| 対面 | 本番 | ローカル参照（プールv3/EXP-045） | 差 | 解釈 |
|---|---|---|---|---|
| Alakazam（自チーム戦除く） | 13/19 = 68% | vs040 67.5%（v2）。v3は自枠につき参照なし | ≈0 | 一致。本番ミラー群の主流402dは81f1亜種で、勝ち越しは実装品質差（EXP-045の主張どおり） |
| **Archaludon実体** | 10/17 = 59%（全滅4含む） | vs046 51.5% | **+8** | **046はArticuno壁+手札刈り型、本番主流debbeはPH無効カード0のタンク型**。全滅4を除く実力対面は10/13=77%。プールのArchaludon枠は045に対し実勢より辛い |
| **Mega Lucario ex** | 8/10 = 80% | vs007 63.5% | **+17** | 007は較正済み自作。本番Lucario勢は瞬発力はあるが安定せず、負け2は相手のT2-3ぶん回りのみ |
| **Dragapult ex** | 3/6 = 50%（**2b31に0-3**） | vs021 67% | **-17** | 021（公式方策）はフォールバック枠で弱すぎる。個体2b31が真の穴。プールDragapult枠の較正不足が現実の失点になった初の実例 |
| Garchomp | 3/5 = 60% | vs041 45.5% | +15 | n=5。負け2戦の機序（Rock Fighting/RagingCurse/山切れ）はローカル精読と同一なので、乖離は主にn |
| Marnie | 3/5 = 60% | vs038 45.0% | +15 | n=5。038はトップ個体レプリカ=フィールド平均より辛い（EXP-045既知） |
| Kangaskhan | 1/2 | vs047 37.0% | — | **本番で当たったのは旧039型(89d8)のみで047型(b7cef)は0出現**。プールweight13.0はこのレート帯では過大 |
| Crustle | 1/2 | vs042 67.5% | — | 負けた363629はGreat Tusk攻撃型で042と別亜種 |
| Bellibolt / ストール系 / Honchkrow / Starmie(69fc) | 2/4, 0/3, 0/1, 0/1 | 相手役なし | — | テール計12%。**ストール一族+攻撃型Crustle+69fcで0-6**がテールの実損 |

- 加重整合性: 本番59.7% vs プールv3 weighted 56.7%[54.3-59.2]。**対面ごとの乖離（Archaludon/Lucario甘め、Dragapult/テール辛め）が相殺して総合値は合っている**が、個別対面の判断にv3の値をそのまま使うのは危険（特にArchaludon/Dragapult/Kangaskhan枠）

## 8. アイデア候補（効果量見積もり付き。BACKLOG追記は親判断）

見積もりは「本番77戦での該当損失×救済確度」。ノイズ床（個別対面±5-7pt）を意識し、全体勝率換算で+1pt未満は列挙しない。

1. **[方策/バグ級] 無効モデルの一般化+ダメージ攻撃への切替**（ローカルスタディF2/F3の本番確証版）: 「PH予測0（Mist/Rock Fighting/Veil）のとき、(a)通常ダメージ攻撃（SPB30/Land Crush90/Teleportation10）が対象を確殺・削れるなら切替、(b)Rock Fighting Energy(id20)を無効エネ集合に追加」。→ 該当負け3戦（85205262は確実に勝てた、85193990/85214552は寄与）で**+2〜3勝/77 ≈ +3pt**。根拠: §5b/5c。EXP-048の「行動を止める」系ではなく**確定計算に基づく攻撃選択の追加**なので副作用リスク小
2. **[方策] ミラー山経済**: Alakazamミラー（同一60枚検知）で「バースト不成立ターンの任意ドロー（Fez/Dudunsparce/Psychic Draw）を、自deck≤相手deck+マージンのとき抑制」。→ ミラー消耗戦1勝3敗の3敗中2敗が4-7枚差の山切れ、**+2勝/77 ≈ +2.5pt**。Alakazamシェア26%で発生頻度は十分。**注意: EXP-048でドロー抑制は対047で-14ptの前例**。ミラー限定ゲート+「PH打点=手札」の両立設計が必須で、要アブレーション
3. **[方策] マルチプライズバースト志向のBoss運用**（F5のArchaludon拡張）: 手札バースト（PH≥300）が成立するターンに1prz forced active（Duraludon等130HP級）へ過剰投射する代わりに、Bossで2-3prz対象（Archaludon ex 300/Kangaskhan 300）を吊って撃つ。→ 85214552の直接救済+Archaludonシェア22.1%での取りこぼし削減、**+1〜2勝 ≈ +1.5-2.5pt**
4. **[方策] 攻撃ロック（Snotted Up）対応**: 自アクティブが攻撃不可の状態を検知したら (a)リトリート回転（エネはアクティブ優先で貼る）、(b)不可能なら任意ドロー全停止で山温存。→ ストール系0-4のうち85200291（相手残9枚）と85280170の部分救済、**+1〜2勝 ≈ +1.5-2.5pt**。実装はやや重い（ロック検知+リトリート経済）
5. **[プール] Dragapult枠を021→2b31cd4174レプリカに交代**: 0-3の実穴かつ現枠が甘い方向に較正ずれ（67%対50%以下）。n=195→200到達確認後、軽量逆設計（EXP-037手法）。**プール較正案件**（045の勝率改善ではない）
6. **[プール] Archaludon枠にdebbe71c08型（タンク型）を追加または046と差替検討**: 本番最多実体（22.1%）とプール代表（Articuno型3.0%）の乖離は、主力候補選定をArchaludonの多いレート帯で誤らせる。8月の最終選定前に要判断
7. **[デッキ/保留] 初手全滅対策**: 負けの19%が「たね1枚初手」。たね10枚構成の構造コスト（本人も同じ）で、方策の余地はほぼ無し。Poffin×4は採用済み。たね追加はEXP-013（1枚テックで有意劣化）の教訓によりコスト高→**提案しない**（記録のみ）

### 見送り（ノイズ床未満/EXP-048教訓に抵触）

- Xerosic温存/Hilda捨て順の微調整（85193990で腐ったが頻度低）
- 「相手ぶん回り」対策（Lucario/Starmie速攻負け3戦は方策で不可避）
- IS_FIRST選択の変更（先手全滅と後手全滅が両方あり、期待差が測れない）

## 9. 結論

- **敗因トップ3**: ①山切れ12/31（39%、うち9戦はプライズ有利/同数のまま）②プライズレース13/31（構造4+ぶん回られ3+接戦6）③初手たね1枚事故6/31（19%）
- **最大の本番⇔ローカル乖離**: Dragapult -17pt（021が弱すぎ、2b31に0-3）とArchaludon +10pt（046がArticuno型で本番主流タンク型と別物）。Kangaskhan枠（w13.0）はこのレート帯で0出現=weightが実勢とずれている
- **「Cinderace」の正体はArchaludon ex**（Duraludon/Archaludon+Cinderaceエンジン、本番シェア22.1%で最多）。classify_deck誤分類の実害を本番データで再確認
- **lethal探索は本番で設計どおり発火**（prz≤3でのみ手時間が跳ねる、時間予算の98%は未使用）
- 方策アイデアの本命は「無効モデル一般化+ダメージ攻撃切替」（バグ級、+3pt見込み）と「ミラー山経済」（+2.5pt見込み、要副作用検証）

## 再現手順

```
# 集計・レンダラ（scratchpadに作成。tools/には未反映）
uv run python <scratchpad>/agg.py 54525332 045_alakazam_full
uv run python <scratchpad>/render_my.py <episode_id>   # 負け31戦を個別レンダリング
uv run python <scratchpad>/movetimes.py                # per-move時間（lethal検証）
uv run python <scratchpad>/decks.py <sig> ...          # 相手60枚ダンプ
# 身元判定はTeamNames必須（§0）。tools/get_my_episodes.pyの流用時も同様の注意
```
