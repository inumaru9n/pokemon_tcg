# improved-probabilistic-agent（Ivan Ternovskiy、67票）

3行サマリ: 「04 Probabilistic Expectimax Agent」= 純ヒューリスティック(ML/RL無し)の探索エージェント。
search_begin/search_stepで決定化探索+expectimax+MCTS(15反復)+ビームサーチ(幅3)、時間予算1.5秒。
デッキは**Mega Lucario ex闘系**(Makuhita/Hariyama/Lunatone/Solrock/Riolu/Mega Lucario ex、Alakazamでない)。

## 盗めるアイデア
- ★ **探索の構造化フォールバック実装の実例**: USE_SEARCH フラグ + SEARCH_TIME_BUDGET=1.5s +
  BEAM_WIDTH=3 + MCTS_ITERATIONS=15 + SEARCH_MAX_CANDIDATES=8。search API不可時に自然に
  フォールバックする try/except構造（_SEARCH_OK）。我々の007/054と同系の「探索無効時フォールバック」の別実装
- ★ **expectimax（確率的期待値最大化）でチャンスノード（ドロー等）を扱う**アプローチ。
  我々のEXP-053/054は決定化ロールアウトで交絡・順序等価に苦しんだ。expectimaxは確率ノードを
  明示的に期待値化する点が異なる → **不確実性の扱いが違う探索の比較対象**（ただし後述の要検証）
- Mega Lucario ex闘デッキの60枚レシピ（プール相手役 or 主力候補の素材。knowledge/episodesのLucarioシェアと照合）

## 注意点・疑問
- ★ **search_begin/search_stepが本番評価環境で動くかは未確認**（コミュニティ未解決、
  knowledge/discussions/713608）。このエージェントの探索は本番で無効化される可能性あり
- **要検証**: 「baselineを大きく上回る」の主張は自己申告で検証方法不明。67票だが実LB成績は不明
- **要検証**: expectimax+MCTS 15反復が本ゲームの高不確実性(discussion 724362で「探索は難しい」)で
  実際に効くか。我々のEXP-053（毎手MC探索は構造的不成立）と整合するか要確認
- ML/RL無しの純探索なので提出は自己完結（numpy不要）。移植は容易
