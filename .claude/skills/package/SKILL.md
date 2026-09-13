---
name: package
description: エージェントをKaggle提出用tar.gzにパッケージし検証、確認の上でCLI提出まで行う。「提出物を作って」「サブミットして」と言われたときに使う。
---

# /package — 提出物の作成と提出

引数: エージェントディレクトリ（省略時は `knowledge/INSIGHTS.md` の現行ベスト）。

## 手順

1. `uv run tools/package_submission.py agents/NNN_slug` を実行
   - デッキ検証（60枚、同名4枚まで、ACE SPEC1枚まで）→ 自己対戦検証 → tar.gz作成まで自動で行われる
2. 出力された `submissions/*.tar.gz` のパスと検証結果を報告
3. **提出はユーザの明示的な承認を得てから**実行する:
   ```
   uv run kaggle competitions submit -c pokemon-tcg-ai-battle -f submissions/<tar.gz> -m "<エージェント名: 方策の一言+ローカル評価値>"
   ```
   - 1日5提出まで・**最新2提出のみ評価対象**
   - **最終盤の提出は同一tar.gzを2枠に出す**のが実践知（LBレートは同一エージェントでも150〜400pt割れる。knowledge/discussions/712621）。開発中は「現行ベスト+挑戦者」の2枠使い分けも可
4. `uv run kaggle competitions submissions -c pokemon-tcg-ai-battle` で受理（PENDING）とsub IDを確認
5. `knowledge/SUBMISSIONS.md` に日付・エージェント・tar.gz名・sub ID・狙いを記録する。後日レーティングが判明したら結果列に追記する

## 注意

- 提出後は日次エピソードデータセットに自エージェントの対戦が載る（チーム名で抽出）→ /episodes で本番挙動の分析が可能
- 単発のLBスコアでエージェントの優劣を判断しない（高分散）。ローカルのプール評価が一次判断
