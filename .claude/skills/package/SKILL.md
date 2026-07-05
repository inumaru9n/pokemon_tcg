---
name: package
description: エージェントをKaggle提出用tar.gzにパッケージし検証する。「提出物を作って」「サブミットの準備」と言われたときに使う。
---

# /package — 提出物の作成

引数: エージェントディレクトリ（省略時は `knowledge/INSIGHTS.md` の現行ベスト）。

## 手順

1. `uv run tools/package_submission.py agents/NNN_slug` を実行
   - デッキ検証（60枚、同名4枚まで、ACE SPEC1枚まで）→ 自己対戦検証 → tar.gz作成まで自動で行われる
2. 出力された `submissions/*.tar.gz` のパスと検証結果を報告
3. 実際のアップロードはユーザが行う（Kaggle: My Submissions タブ）。1日5提出まで・最新2提出のみ評価対象であることを添える
4. 提出したら `knowledge/SUBMISSIONS.md` に日付・エージェント・tar.gz名・狙いを記録する（ファイルがなければ作る）。後日レーティングが判明したら追記する
