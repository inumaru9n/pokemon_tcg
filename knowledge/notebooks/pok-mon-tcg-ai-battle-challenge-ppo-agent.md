# pok-mon-tcg-ai-battle-challenge-ppo-agent（hmnshudhmn24、低票）

## 3行サマリ

- タイトルは「PPO Agent」だが、**PPOの実装は存在しない**。cgエンジンも使わず、自作の簡易ゲーム状態クラス（PlayerState/Action enum）を定義した段階で終わっている未完成ノートブック
- デッキはMega Charizard X ex軸（炎エネ18枚）を定義しているが、エージェント関数も学習ループもない
- 唯一の情報価値: カードデータを `/kaggle/input/competitions/pokemon-tcg-ai-battle-challenge-strategy` から読んでおり、**StrategyカテゴリコンペにもEN/JP_Card_Data.csvが配布されている**ことが分かる

## 盗めるアイデア

- なし（技術的内容が未完成）

## 注意点・疑問

- タイトル詐欺気味の典型例。票数の少なさ（Web検索経由で発見、vote閾値未満）と内容が整合
- RL系の実装参考には公式のreinforcement-learning-and-mcts-sample-codeを使うべき
