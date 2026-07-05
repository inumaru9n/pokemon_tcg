---
name: episodes
description: 公式日次エピソード（リプレイ）データセットを取得・要約し、メタレポートをknowledge/episodes/に蓄積する（BACKLOG登録はしない。それは/ideateの仕事）。「メタを確認して」「最新のladderデータを取って」と言われたときに使う。
---

# /episodes — 公式エピソードデータの収穫

## 手順

1. `uv run tools/get_episodes.py download --latest` で最新日のzipを取得（日付指定は `--date YYYY-MM-DD`。`data/episodes/<date>/` に保存、約750MB）
   - Kaggle API認証エラーの場合は `~/.kaggle/kaggle.json` の設置をユーザに依頼して停止
2. `uv run tools/get_episodes.py summarize --date <date> -w 6` で summary.csv を生成（5千戦で数分。バックグラウンド実行推奨）
3. `uv run tools/get_episodes.py report --date <date>` で `knowledge/episodes/<date>.md` を生成
4. summarize 完了後、zipは削除してよい（summary.csv と manifest.csv が残っていれば report は再実行可能。deck_ids も summary.csv に保存済み）
5. **`knowledge/BACKLOG.md` には追記しない**。台帳登録は /ideate に一本化されている
6. 収穫サマリを報告する: 前回レポート（knowledge/episodes/ の直近日付）と比べたシェア・勝率の変動、勝率上位デッキの入れ替わり、TIMEOUT/INVALID件数の異常。台帳登録には /ideate の実行を提案する

## 注意

- daily datasetは20GiB上限で切り詰められている可能性が高く、全対戦の完全収録ではない
- アーキタイプ分類は看板ポケモン推定の簡易ヒューリスティック（tools/get_episodes.py の classify_deck）。誤分類率は未検証
- 対面勝率はladder母集団（強さピンキリ）に基づくため、シェアの大きいアーキタイプ以外は試合数を確認してから解釈する
