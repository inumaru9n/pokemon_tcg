---
name: notebooks
description: Kaggle公開ノートブックを取得・分析し、要約をknowledge/notebooks/に蓄積する（BACKLOG登録はしない。それは/ideateの仕事）。「他の参加者のコードを見たい」「ノートブックを取得して」と言われたときに使う。
---

# /notebooks — 公開ノートブックの収穫

## 手順

1. `uv run tools/get_notebooks.py` で新規・更新ノートブックを取得（`data/notebooks/` に保存。更新判定はlastRunTimeの変化。閾値を変えたい時は `--min-votes N`）
   - Kaggle API認証エラーの場合は `~/.kaggle/kaggle.json` の設置をユーザに依頼して停止
2. ツール出力の NEW を分析し、`knowledge/notebooks/<notebook名>.md` に要約を書く。UPDATED は既存要約の更新要否を中身を見て判断する
   - 分析の観点と要約フォーマットは notebook-analyzer（`.claude/agents/notebook-analyzer.md`）の定義が原本。数が多い場合は同サブエージェントに分担させ、少数なら自分で同じ観点・フォーマットで分析する
3. **`knowledge/BACKLOG.md` には追記しない**。台帳登録は /ideate に一本化されている（ideateが knowledge/notebooks/ を材料に既存知見と照合してから登録する）
4. 収穫サマリ（新規ノートブック数、有望なアイデア候補、重要な警告事項）を報告し、台帳登録には /ideate の実行を提案する
