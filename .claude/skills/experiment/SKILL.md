---
name: experiment
description: BACKLOGのアイデアを1つ実装し、対戦評価して知見を記録する（アイデアID指定可、例 /experiment I-002）。エージェントの新規実装・改良・検証を頼まれたときに使う。
---

# /experiment — アイデアの実装と検証

引数: BACKLOGのアイデアID（例 `I-002`）。省略時はBACKLOGから優先度Hの `proposed` を1つ選ぶ。

## 手順（省略禁止）

1. **文脈を読む**: `CLAUDE.md`、`knowledge/INSIGHTS.md`、`knowledge/BACKLOG.md`、関連する `knowledge/experiments/EXP-*.md`
2. **BACKLOG更新**: 対象アイデアの状態を `in-progress` に変更
3. **実装**: `agents/NNN_slug/` を新規作成（NNNは既存最大+1）
   - ベースにするエージェントがあれば main.py をコピーして改変（既存ディレクトリは変更しない）
   - deck.csv も必ず置く（変更しないなら現行ベストのものをコピー）
4. **スモークテスト**: `uv run arena/run_match.py agents/NNN_slug agents/000_random -n 20 -w 4`
   - `errors > 0` なら修正してから先へ進む（error_detailsに例外内容が出る）
5. **本評価**: 標準相手プール（`arena/pool.json`）と **各相手200戦以上**
   `uv run arena/run_pool.py agents/NNN_slug -n 200 -w 4 --json arena/results/EXP-NNN.json`
   - 判定は**weighted CI**（メタシェア加重総合勝率）: 下限>50%で validated、上限<50%で rejected、跨ぐ場合は `-n 400` に増やし、それでも跨ぐなら「判定保留」として rejected 扱い（BACKLOGに保留と明記）
   - 個別対面の勝率もEXPに記録する（単一対面はフィールド勝率を代表しない。EXP-002参照）
6. **記録**（これを書くまで実験は完了ではない）:
   - `knowledge/experiments/EXP-NNN.md` を TEMPLATE.md に沿って作成
   - BACKLOG の状態・EXPリンクを更新
   - 一般化できる知見は INSIGHTS.md に追記（3行以内+EXPリンク）
   - 現行ベスト（プールweighted勝率が最高のエージェント）が入れ替わったら INSIGHTS.md 冒頭を更新
   - validated のエージェントは `arena/pool.json` への追加を検討（weightは knowledge/episodes/ 最新のアーキタイプシェア）
7. **派生アイデア**: 実験中に気づいたアイデアをBACKLOGに追記（重複確認すること）

## 注意

- EXP番号はエージェント番号と揃える（agents/007_xxx → EXP-007）
- 敗因分析には結果JSONの `records` と、デバッグ時は RESULT ログの reason を使う
- 実装がKaggle環境で動く形か常に意識する（自己完結main.py、`from cg.api import ...`、時間制限）
