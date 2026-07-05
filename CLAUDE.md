# Pokémon TCG AI Battle — 開発規約

Kaggleコンペ [pokemon-tcg-ai-battle](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle) で
最高レーティングのエージェントを作るプロジェクト。提出締切 2026-08-16。

姉妹コンペ [Strategy Category](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle-challenge-strategy)（戦略レポート審査、
上位8チーム各$30,000でFinal Stage進出。評価は安定性・デッキ設計思想・Simulation成績）にも参加予定。
`knowledge/experiments/` のEXPレポート蓄積がそのままレポートの素材になるため、実験記録の質を保つこと。
詳細要件（締切・フォーマット）は未調査（Kaggleページで要確認）。

## 絶対ルール

- Pythonの実行は必ず `uv run`（pip直接実行やsystem python禁止）。リポジトリルートから実行する。
- 実験結果・知見は必ず `knowledge/` に記録してから次に進む（下記ワークフロー参照）。
- エージェントは `agents/NNN_slug/` に1アイデア=1ディレクトリ。既存エージェントのディレクトリは変更しない（比較可能性を守る。改良は新番号を振る）。
- `main.py` は自己完結（同ディレクトリ内のファイルと `cg` 以外に依存しない）。ヘルパーモジュールを作る場合はファイル名にエージェント番号を含めて衝突を避ける。

## ディレクトリ構成

- `cg/` — 対戦エンジンSDK（`libcg.dylib` はこのIntel Mac向けにソースからビルドしたもの。提出時は使わない）
- `agents/NNN_slug/` — main.py + deck.csv（提出形式と同一）
- `arena/run_match.py` — A vs B 対戦評価CLI。`arena/league.py` — 総当たりリーグ
- `knowledge/` — INSIGHTS.md（確定知見）/ BACKLOG.md（アイデア台帳）/ experiments/EXP-NNN.md（実験レポート）/ notebooks/ episodes/ discussions/（各収穫スキルの要約）
- `tools/` — get_notebooks.py / get_episodes.py / get_discussions.py（外部データ取得。同名の収穫スキル /notebooks /episodes /discussions が使用。共通の作り: kaggle CLI経由・`--min-votes`/`--force`・index.jsonによる差分取得。共通部は kaggle_util.py）、package_submission.py（提出物作成）
- `data/` — 取得した生データのキャッシュ（notebooks/ episodes/ discussions/。episodesのzipはsummarize後に削除可）
- **命名規約**: 外部データの取り込みは「データ源の名詞」で skill / tool / data/ / knowledge/ の名前を揃える（例: discussions → /discussions, get_discussions.py, data/discussions/, knowledge/discussions/）
- `pokemon-tcg-ai-battle/` — コンペ配布物（読み取り専用。変更禁止）

## 実験ワークフロー（/experiment スキルが手順を定義）

1. `knowledge/BACKLOG.md` のアイデアを1つ選ぶ（状態を in-progress に更新）
2. `knowledge/INSIGHTS.md` と関連EXPを読む
3. `agents/NNN_slug/` を実装
4. スモークテスト: `uv run arena/run_match.py agents/NNN_slug agents/000_random -n 20` で errors=0 を確認
5. 本評価: 標準相手プール（`arena/pool.json`）に対し **各相手最低200戦**
   `uv run arena/run_pool.py agents/NNN_slug -n 200 --json arena/results/EXP-NNN.json`
6. `knowledge/experiments/EXP-NNN.md` を書き、BACKLOG状態更新、確定知見はINSIGHTSへ昇格

### 統計基準

- 200戦で95%CIは約±7%。合否は**メタシェア加重の総合勝率**（run_poolのweighted CI）で判定: CI下限>50%で validated、上限<50%で rejected、跨いだら戦数を増やす（-n 400）か判定保留
- **winner's curse**: 係数スイープ等で選ばれた勝者の成績は上振れしている。採用判断は選定に使った対戦とは別の400戦で再測定する（EXP-008→009、EXP-010→011で実証）
- **改良系（ベース+差分）の例外**: プールにベース自身が入るためweighted verdictは甘くなる（EXP-005）。合否は (a)ベースとの直接対戦で劣化なし かつ (b)狙い対面の改善 の両方で判定
- 単一対面の勝率はフィールド勝率を代表しない（EXP-002）。個別対面の結果は解釈材料としてEXPに残す
- 「現行ベスト」は `knowledge/INSIGHTS.md` の冒頭に明記されているエージェント（改良のベース選定用。合否はプール判定）
- `arena/pool.json` は検証済みエージェントが増えたら/メタが動いたら更新する（weightは knowledge/episodes/ 最新のアーキタイプシェア）

## 評価CLIの使い方

```bash
uv run arena/run_match.py <agentA_dir> <agentB_dir> -n 200 -w 4 --json out.json   # 1対1
uv run arena/run_pool.py <agent_dir> -n 200 --json out.json   # 標準プール評価（合否判定用）
uv run arena/league.py -n 100          # agents/ 全体の総当たり + Elo
uv run tools/package_submission.py agents/NNN_slug   # 提出tar.gz作成+検証
```

- 対戦は自動で先後入替。エージェント例外・不正選択は即負け（errorsに計上）。
- エンジンのRNGはシード不可（`--seed` はエージェント側Pythonのrandomのみ）。

## エンジンの注意点

- この開発機はIntel Mac。配布物のdylibはarm64専用のため、`cg/libcg.dylib` はC++ソースからビルド済み。再ビルド:
  ```
  cd "pokemon-tcg-ai-battle/data/ptcg_engine/ptcgProgram 22" && SDK=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk && clang++ -std=c++20 -O2 -shared -fPIC -isysroot $SDK -isystem $SDK/usr/include/c++/v1 -o ../../../../cg/libcg.dylib Export.cpp
  ```
- Pythonは `.python-version` で x86_64 版に固定済み（arm64版はこのMacで動かない）
- 1ゲーム数ms〜数十ms。200戦×4並列で数秒。実験時に戦数をケチる理由はない
- `search_begin`/`search_step`（cg/api.py）で決定化シミュレーション（先読み探索）が可能
- 公式ルールとの差異は `pokemon-tcg-ai-battle/overview.md` 後半を参照（同時きぜつでの両者サイド取り切りは引き分け等）
- 提出時のcgは配布オリジナル（`pokemon-tcg-ai-battle/data/sample_submission/sample_submission/cg`）を同梱する（ローカルビルドのdylibは含めない）
- **Kaggle環境の時間制限は「1ゲームあたり合計600秒/エージェント」**（運営公式回答で確定。1手ごとの個別制限はない）。実行環境は**CPUのみ1.6vCPU/RAM 8GB/外部ネットワーク禁止**。探索系は1手平均1〜2秒の時間予算+累計時間の自己計測+超過時フォールバック（greedy即答）を実装すること
- **6/30に本番環境が更新**: step上限による引き分けは廃止され、ループする側がタイムアウト負けになる（ローカルarenaのstep_cap引き分けとは挙動が異なる点に注意）。マッチングは48戦/日/提出+10%ランダム対戦
- **search_begin/search_stepが本番評価環境で動くかは未確認**（コミュニティでも未解決。knowledge/discussions/713608参照）。探索を使うエージェントは必ず探索無効時に自然にフォールバックする構造にする（007はこの形）
- 大規模自己対戦（数万戦/プロセス）はlibcgのメモリリークでOOMする報告あり。ワーカープロセスをN戦ごとに使い捨てる設計にする（knowledge/discussions/709152）

## 知識ベースの運用方針

- `knowledge/BACKLOG.md` は**エージェント（デッキ+方策）のアイデア専用**。基盤・ツール・調査のタスクはBACKLOGに入れず、必要になった時点で直接実施し、成果はINSIGHTSや該当ツールに反映する
- 評価基盤の既知の改善候補（未実施）: 敗因reason別集計、方策間の行動一致率測定（メタシェア加重プール評価は arena/run_pool.py として実装済み）、公開されたエンジンソースの高速化ビルド（RL自己対戦・大規模スイープの基盤。knowledge/discussions/717141）

## 提出形式

- `.tar.gz` のトップレベルに `main.py` と `deck.csv`（60枚、同名カード4枚まで、基本エネルギーは無制限、ACE SPECは1枚まで）
- **1提出=1デッキ固定**（運営ルール。ゲームごとのデッキ切替は禁止、Strategyカテゴリで負評価）
- **LBレートは高分散**（同一エージェント2枠で150〜400pt差の報告例）。単発のLBスコアで優劣を判断せず、最終提出は同一エージェントを2枠に出すのが実践知（knowledge/discussions/712621）
- Kaggle実行環境では deck.csv は `/kaggle_simulations/agent/deck.csv` に置かれる（sample main.py の read_deck_csv がその形）
