---
name: ideate
description: 新しいエージェント改善アイデアを出してBACKLOGに追記する。ユーザがアイデア出し・ブレスト・「次に何を試すか」を求めたときに使う。
---

# /ideate — アイデア出し

引数（任意）: アイデアのタネ（例 `/ideate エネルギー配分の最適化`）や個数指定。

## 手順

1. **文脈を読む**: `knowledge/INSIGHTS.md`（確定知見）、`knowledge/BACKLOG.md`（既出アイデア）、直近の `knowledge/experiments/EXP-*.md` 数件
2. **材料を集める**。以下が材料の全リスト。毎回全部を使う必要はないが、どれを使いどれを使わなかったかを意識すること:
   - **自分の実験結果**: `knowledge/experiments/EXP-*.md`（特にrejectedの失敗原因）、`arena/results/*.json`（敗因reason別の内訳、エラー詳細）、`arena/results/league.md`（リーグ相性表。現行ベストが苦手な対面=改善アイデアの種。古い場合は /league を提案）
   - **公開notebooksの要約**: `knowledge/notebooks/*.md` の「盗めるアイデア」節（★=新規性高）。未取得・古い場合は /notebooks を提案
   - **公開Discussionの要約**: `knowledge/discussions/*.md`（運営アナウンス・他参加者の知見）。未取得・古い場合は /discussions を提案
   - **Web**: WebSearchでPokémon TCGの強デッキ・戦略、カードゲームAI（ISMCTS/決定化/強化学習）の手法
   - **メタレポート**: `knowledge/episodes/<date>.md`（本番ladderのアーキタイプ分布・対面勝率・勝率上位デッキ）。トップメタ対策や勝率上位レシピの移植（60枚は `data/episodes/<date>/summary.csv` の deck_ids から復元）の一次資料。古い場合は /episodes を提案
   - **カードプール分析**: `pokemon-tcg-ai-battle/data/EN_Card_Data.csv` または `cg.api.all_card_data()`。1267枚から未活用のシナジー・アーキタイプ・対メタカードを探す（デッキ構築系アイデアの一次資料）
   - **ユーザから与えられたタネ**
   - （提出後）Kaggleの実対戦成績・リプレイ: ローカル評価と本番μの乖離はそれ自体がアイデアの種
3. **アイデアを書く**: 各アイデアは「何を変える→なぜ勝率が上がるはず」が1行で言える粒度に分解する
   - **BACKLOGに載せるのはエージェント（デッキ+方策）のアイデアのみ**。`agents/NNN/` を作って対戦検証できる形になるものが対象
   - 基盤・ツール・調査の思いつきはBACKLOGに入れない。重要ならCLAUDE.mdの「知識ベースの運用方針」の改善候補リストに1行追記するに留める
4. **BACKLOGに追記**: 既存行との重複を確認し、I-NNN（連番）、出典、優先度H/M/Lを付けて追加
5. 追加したアイデアの一覧と、次に検証すべき1件の推薦を報告する

## 優先度の目安

- H: 勝率に直結（プレイング方策、探索、デッキ構築）
- M: 特定局面の改善、相手モデリング
- L: 基盤整備、調査、長期投資（強化学習など）
