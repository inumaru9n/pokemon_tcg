# Majkel1337 156952a871 精読メモ（EXP-068準備）

真の精読(実リプレイを1手ずつ読み解釈)。2026-07-17。全2923戦の全数精読は不可能なため
負け試合優先+代表勝ち試合を精読し、抽出条件は全数データで裏取り検証する方針。

## Majkel実成績と負けの構造(全数集計)
- 勝率 59.0%（勝1723/負1195）
- 負けの内訳: **Alakazamミラー45.2% / Mega Kangaskhan 26.4%** / Marnie10.5% / TRSpidops8.4%
- 負けのターン: 短<15が458 / 中15-30が603 / 長>30が134
- **結論: ladderトップMajkelはミラーもKangaskhanも強くない。多様フィールドの量産型で勝率を稼ぐ。
  模倣はこの弱点(ミラー弱)を継承する = 062のミラー弱化の正体**

## 精読した試合と発見
### 勝ち: ep85828069 vs Comfey
- t3から攻撃開始、t5でAlakazam完成、以降Powerful Handでグラインド
- 手札が11→27に膨張、山24→10。サイド2-6で長期化 → **山切れリスク**(Majkelは管理しない)

### 負け: ep85309304 vs Kangaskhan(10ターン)
- **Shaymin先頭**の遅い立ち上がり。Alakazam(140)が200打点で毎回KO、t9でサイド0取得
- 発見: consistency-grind型は速攻に構造的に弱い。Shaymin先頭は対アグロで裏目

### 負け: ep85215782 vs Alakazamミラー(15ターン)
- 完全互角のグラインド。終盤の山切れ+アタッカー交代で決まる
- t14でZamがKOされAbra(50)先頭になり負け → **2体目ready Alakazam確保の重要性**

## 抽出条件文spec(A/B/Cに構造化して実装)
- S1 開始=Dunsparce/Abra先頭（Shaymin先頭回避）
- S2 Abra→Kadabra→Alakazamを速く（Candy活用）
- S3 2体目ready Alakazamを常時確保（KO後即交代）
- S4 Dunsparce→Dudunsparce→Dud能力を毎ターン（主エンジン、終盤54%）
- S5 Alakazam完成後turn5前後から攻撃
- S6 相手手札≥8でXerosic
- S7 turn5-9でHammer積極（1.5回/G）
- S8 サイド射程でBoss締め
- S9 (改良)長期戦の山切れ管理（Majkel未管理の弱点=模倣超えの余地）
