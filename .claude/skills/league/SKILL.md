---
name: league
description: agents/配下の全エージェントで総当たりリーグを実施しローカルEloランキングを更新する。「どのエージェントが最強か」「リーグ戦して」と言われたときに使う。
---

# /league — ローカルリーグ戦

## 手順

1. `uv run arena/league.py -n 100` を実行（引数でエージェントを絞れる。エージェント数が多く時間がかかる場合は run_in_background で実行して進捗を報告）
2. 結果は `arena/results/league.md` / `league.json` に書かれる
3. ランキングを報告し、以下を確認する:
   - `knowledge/INSIGHTS.md` 冒頭の「現行ベスト」がEloトップと一致しているか。違うなら更新
   - 意外な結果（三すくみ、期待外れの改良版など）があれば INSIGHTS/BACKLOG に反映
