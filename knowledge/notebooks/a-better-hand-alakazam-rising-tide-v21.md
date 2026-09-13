# A Better Hand — Alakazam Rising Tide v21（jazivxt・votes 38・2026-08）

## 3行サマリ

- **公開されている最良水準のAlakazam構築が60枚まるごと読める**（base64・pickle・DLモデル無しの明示的な方針）
- 我々の104（教師 Yushin の 156952a871）と**11枚違う**。方向は
  **Dudunsparce 2→4 / Neutralization Zone を ACE SPEC に / Nighttime Mine と Shaymin を全抜き**
- ladder実測で対Spidops 50.0% を出す構築 `8c7d3f998f` と**独立に同じ変更に到達している**

## ★★★ 60枚の差分（104 → 公開版）

| card | 104 | 公開版 | 差 | |
|---|---|---|---|---|
| **Dudunsparce** | 2 | **4** | +2 | Land Crush 90＝**実ダメージ**（Veilを貫通） |
| Dunsparce (id 65) | 0 | 2 | +2 | Gnaw 10 / Dig 30（コイン表で被ダメ・効果を防ぐ） |
| Dunsparce (id 305) | 3 | 1 | −2 | Trading Places / Ram 20 |
| Night Stretcher | 1 | 3 | +2 | |
| **Neutralization Zone** | 0 | **1** | +1 | **ACE SPEC**。ルールボックス無しを ex/V の攻撃から守る |
| **Enriching Energy** | 1 | **0** | −1 | ACE SPEC の枠を明け渡す |
| Lillie's Determination | 0 | 1 | +1 | |
| Enhanced Hammer | 4 | 3 | −1 | |
| Fezandipiti ex | 1 | 0 | −1 | |
| **Nighttime Mine** | 2 | **0** | −2 | |
| **Shaymin** | 1 | **0** | −1 | |

同枚数: Abra4 / Alakazam4 / Kadabra4 / Rare Candy3 / Buddy-Buddy Poffin4 / Dawn4 / Hilda4 /
Poké Pad4 / Telepath Psychic Energy4 / Boss's Orders3 / Xerosic's Machinations3 / Basic{P}2 /
Lana's Aid1 / Sacred Ash1

## 取れる情報

★★★ **「Dudunsparce系を増やす」に3つの独立な根拠が揃った**:
1. ladder実測: 構築 `8c7d3f998f`（Dudunsparce 3）は**対Spidops 50.0%**、我々の `156952a871`（同2）は30.3%
2. この公開notebook（Dudunsparce 4）
3. 機序: Alakazam の唯一の攻撃 Powerful Hand は**ダメカン配置＝効果**なので
   Articuno の Repelling Veil に消えるが、Land Crush 90 は**ダメージ**なので通る（EXP-106）

★★ **ACE SPEC の入れ替え（Enriching Energy → Neutralization Zone）も2ソースで一致**。
Neutralization Zone は「ルールボックスを持たないポケモン（自分・相手とも）が、
相手の ex/V の攻撃から受けるダメージを全て防ぐ」スタジアム。
Alakazam・Kadabra・Abra・Dunsparce系は**全て非ルールボックス**なので、
**ex中心の相手（Marnie's Grimmsnarl ex / Mega Kangaskhan ex 等）に対して自分だけが一方的に守られる**。

★ 著者の自己申告（ローカル診断・LB主張はしていない）:
公開v21との直接対戦 24–16、five-control 70–50、重み付きスコア 52.8%。

★ 実装方針が明示的（base64・minify・pickle・DLモデル・隠しペイロードを使わない）で、
**Strategy Category の「explanations の質」の観点で参考になる書き方**。

## 注意点

- **Dunsparce の id が違う**（65 と 305 は別のカード）。id 65 は我々のデッキに無く、
  **NNの語彙外**になる。Lillie's Determination・Neutralization Zone も同様（計3種）
- 我々の 112/113 は Dudunsparce 3 までしか上げていない。**公開版は4**
- この構築は**教師データが存在しない**（Yushin は 156952a871 を使う）。
  104の方策をそのまま載せると分布ずれが乗る。効果はデッキ単独の寄与として測る必要がある
