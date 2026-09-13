# What actually wins on the ladder（Busya PRIME・votes 30）

## 3行サマリ

- 生のエピソードログから**アーキタイプ別シェア・勝率・対面表・推奨デッキを毎回再計算**するnotebook。
  貼り付けの数字を一切使わない方針
- 最後のセクションが**「フィールド構成で加重した期待勝率」**＝我々が `run_pool` の weighted で
  やっていることと同じ推定量に、公開データ側から到達している
- 計算結果は**維持されたデータセット** `busyaprime/pokemon-tcg-ai-battle-live-meta` として公開されている

## 取れる情報

★★ **推定量の定義が我々と一致**（原文）:
> "What you want is the **expected win rate against the field you will actually face**, which is each
> deck's matchup results **weighted by how often each opponent shows up**."

我々の `arena/pool.json` のweightと `run_pool` のweighted score rate はこれのローカル版。
**独立に同じ設計に至っている**ことは、この推定量が正しいことの傍証。

★★ 「人気は強さを追跡しているか」の節: **過小評価されている強いデッキ（左上）が
レートを盗める場所**という整理。729926 の「Garchomp が最高EVでシェア7%」と同じ結論の道具立て。

★ ゲーム長の分析: **ハーネスは時計で動くので、長引くデッキで長考するエージェントがタイムアウトする**。
step数を長さの代理指標にしている。

★ 出力を tidy CSV で書き出す設計＝**フォークせずデータセットを取るだけで対面表が手に入る**。

## 注意点

- アーキタイプのラベルは**「実際に出した中で最大HPのexポケモン」**というヒューリスティック。
  **非exアタッカー中心のデッキは catch-all に落ちて順位から除外される**。
  我々の `classify_deck`（EXP-017 で Mega Starmie ex の誤分類が既知）とは別の癖を持つので、
  数字を突合するときは分類定義の違いに注意
- 1日分・最大4,000エピソードのランダム抽出（固定シード）＝**スナップショットであって法則ではない**と著者自身が明記
- 我々は `tools/get_episodes.py` で全数を持っているので、**このnotebookを使う必要はない**。
  価値は「独立な検算」と「公開データセットで対面表が取れる」点
