---
name: discussions
description: コンペ公式Discussionを取得・分析し、要約をknowledge/discussions/に蓄積する（BACKLOG登録はしない。それは/ideateの仕事）。「ディスカッションを見たい」「運営のアナウンスを確認して」と言われたときに使う。
---

# /discussions — 公式Discussionの収穫

## 手順

1. `uv run tools/get_discussions.py` で新規・更新トピックを取得（`data/discussions/<id>.md` にトランスクリプト保存。更新判定はコメント数の変化。絞りたい時は `--min-votes N`）
   - kaggle CLI認証エラーの場合は `~/.kaggle/kaggle.json` の設置をユーザに依頼して停止
2. 取得された新規・更新トピックを読み、価値のあるものだけ `knowledge/discussions/<id>-<slug>.md` に要約を書く:
   - **運営アナウンスは最優先**（ルール変更・環境更新・データセット追加は即CLAUDE.md/INSIGHTSへの反映を検討）
   - フォーマット: 3行サマリ / 取れる情報（★=新規性・実用性高） / 注意点・疑問（未検証の主張は要検証フラグ）
   - 質問系・雑談系など情報価値のないトピックは要約せずスキップしてよい（スキップした旨だけ報告）
3. **`knowledge/BACKLOG.md` には追記しない**。台帳登録は /ideate に一本化されている
4. 収穫サマリ（新規/更新トピック数、重要アナウンス、有望な情報、要検証事項）を報告し、台帳登録には /ideate の実行を提案する

## 注意

- 本文はtable出力・コメントはJSON由来（tools/get_discussions.py参照）。コメント数変化のない編集は検知できない
- 運営の環境更新アナウンス（エンジン変更等）を見つけたら、ローカルcgとの乖離リスクをユーザに警告する
