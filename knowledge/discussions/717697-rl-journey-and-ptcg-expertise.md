# 717697 Sharing my Reinforcement Learning journey（22票、更新）

3行サマリ: 純自己対戦RLプレイヤーの体験談 + トップ1%の実PTCGプレイヤー(Ryan Rumble)による
戦略解説。RLは~4時間訓練で公開トップbotに勝ち上位30%、追加訓練でsilver到達。
**deck-out報酬シェイピングでデッキアウト負けが激減=我々のK1精読知見を独立に再確認**。

## 取れる情報
- ★★★ **報酬シェイピング「通常負け-1.0 / deckout負け-1.5」で自分のデッキアウトが激減**。
  我々の精読K1（Majkelの負けの36.4%が山切れ）を、RLプレイヤーが独立に「主要敗因」と発見。
  **I-109（自己対戦RL）実装時の報酬設計にそのまま使える具体知見**
- ★★ **sequencing（実PTCGプロの一般則）: 「汎用ドロー(Fez/Dud)を先に、特定サーチ(Poffin/Pad)は
  後に」**。汎用ドローで何が来るか見てから、確定サーチで補完すると情報量最大。既知の需要が確定
  したら「thin to win」=山を薄めて目的札の確率を上げる。→ **背骨/方策に足せる高価値ミクロ規則**
  （現状の我々の方策はこの順序を明示していない可能性）
- ★★ **明確なブランダー3つ（=避けるべきルール）: ①攻撃コスト超過のエネ過剰添付 ②山を0にする
  カード使用 ③KOできるのに攻撃しない**。①は新規（我々のlethal層は③をカバー、②はK1関連）
- ★ prize mapping/prize trading: 6サイドの取り方を2-2-2等で設計。「相手に7-8枚取らせる」防御
  （1-2-2に強制→3プライズMegaをベンチ）。Legacy EnergyやBossでprize mapを維持/妨害
- ★ **デッキ構築の示唆**: KaggleのZamリストは最近のリスト踏襲だが「もっと良く組める」。
  Indianapolis優勝Zamは対Dragapult特化で4 Nighttime Mine+Elgyem+Dedenne（control寄り）。
  KaggleはTera/Watchtower少なくBo1なので、control型Zamの余地
- ★ Alakazamの弱点: Unfair Stamp+Team Rocket's Watchtower（Kaggleメタには少ない）、
  「2プライズ負け×相手が単プライズ盤面」は勝ちにくい
- 参考リンク: limitlesstcg.com（高順位デッキ）、spinningup.openai.com（RL入門）

## 注意点・疑問
- RLの「4時間で上位30%」は本人環境の話（我々のIntel Mac 4コアでの再現性は不明、高速エンジン依存）
- sequencing規則（汎用ドロー先）は我々の方策で未検証 → 実装して一致率/アリーナで検証する価値（要検証）
- control型Zamデッキ構築は主力デッキ変更を伴うため慎重（1提出1デッキ固定・Strategy評価）
