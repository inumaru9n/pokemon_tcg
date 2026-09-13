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
- **メタは10日で半分入れ替わりうる**（2026-07-28→08-07で Marnie 64%→32% / Lopunny 0.6%→15.8%。EXP-119）。
  **締切直前に必ず重みを更新する**。重み更新は既存の対面別結果を再集計するだけでよく、プールを回し直す必要はない
- **プールの相手役の忠実度は自分の本番リプレイで直接検証する**: `uv run tools/get_my_episodes.py <submission_id> --limit 400`。
  実測で 084 Alakazam は真値より+31pt、130 Lopunny は+41pt 甘く、**プールの絶対値は約16pt過大**（EXP-119）。
  **絶対値を本番期待値として読まない。相対比較にだけ使う**

## 評価CLIの使い方

```bash
uv run arena/run_match.py <agentA_dir> <agentB_dir> -n 200 -w 4 --json out.json   # 1対1
uv run arena/run_pool.py <agent_dir> -n 200 --engine-seed 20260804 --json out.json   # 標準プール評価（合否判定用）
uv run arena/league.py -n 100          # agents/ 全体の総当たり + Elo
uv run tools/package_submission.py agents/NNN_slug   # 提出tar.gz作成+検証
```

- 対戦は自動で先後入替。エージェント例外・不正選択は即負け（errorsに計上）。
- **候補どうしを比較するときは `--engine-seed` に同じ値を必ず指定する**（CRN。現行ビルドはEXP-079のシード注入パッチ入りなのでエンジンのRNGを固定できる。`--seed` の方はエージェント側Pythonのrandomのみ）。指定を忘れると各実行が独立標本になり、**その対面で一手も違わない2エージェントですら差7.2pt・z>2の「有意差」が出る**（EXP-106で実測。CI幅は ±2.7pt → ±0.35pt に縮む）。
- **`--pool` も必ず明示する**（EXP-140）。既定の `arena/pool.json` は上位2枠が新レプリカに差し替わっており、過去の数値はすべて `arena/pool_v7.json`。指定を忘れると**errors=0 のまま別の相手と戦い、比較不能な数字が出る**（実測: 同一エージェントが 72.10%(v7) と 64.58%(既定) になった）。**想定外に大きく動いたら、まず対面表で相手が同じか確認する**。

## エンジンの注意点

- この開発機はIntel Mac。配布物のdylibはarm64専用のため、`cg/libcg.dylib` はC++ソースからビルド済み。**現行ビルドはCRN(シード注入)パッチ入り**（cg/Export_seeded.cpp が配布Export.cppを取り込む方式、EXP-079）。再ビルド:
  ```
  SDK=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk && clang++ -std=c++20 -O2 -shared -fPIC -isysroot $SDK -isystem $SDK/usr/include/c++/v1 -I "pokemon-tcg-ai-battle/data/ptcg_engine/ptcgProgram 22" -o cg/libcg.dylib cg/Export_seeded.cpp
  ```
- **エンジンは07-15版に同期済み（2026-07-24）**: 運営が07-17アナウンスでTRエネ規則・tool解決のバグ修正+3000アクション引き分け撤廃（knowledge/discussions/727094参照）。旧版バックアップ=dist_backup_pre0715/。**配布データセットの更新アナウンスを見たら必ずdiff→同期→再ビルド→CRN決定性再検証を行うこと**
- Pythonは `.python-version` で x86_64 版に固定済み（arm64版はこのMacで動かない）
- 1ゲーム数ms〜数十ms。200戦×4並列で数秒。実験時に戦数をケチる理由はない
- `search_begin`/`search_step`（cg/api.py）で決定化シミュレーション（先読み探索）が可能
- 公式ルールとの差異は `pokemon-tcg-ai-battle/overview.md` 後半を参照（同時きぜつでの両者サイド取り切りは引き分け等）
- 提出時のcgは配布オリジナル（`pokemon-tcg-ai-battle/data/sample_submission/sample_submission/cg`）を同梱する（ローカルビルドのdylibは含めない）
- **Kaggle環境の時間制限は「1ゲームあたり合計600秒/エージェント」**（運営公式回答で確定。1手ごとの個別制限はない）。実行環境は**CPUのみ1.6vCPU/RAM 8GB/外部ネットワーク禁止**。探索系は1手平均1〜2秒の時間予算+累計時間の自己計測+超過時フォールバック（greedy即答）を実装すること
- **6/30に本番環境が更新**: step上限による引き分けは廃止され、ループする側がタイムアウト負けになる（ローカルarenaのstep_cap引き分けとは挙動が異なる点に注意）。マッチングは48戦/日/提出+10%ランダム対戦
- **search_begin/search_stepが本番評価環境で動くかは未確認**（コミュニティでも未解決。knowledge/discussions/713608参照）。探索を使うエージェントは必ず探索無効時に自然にフォールバックする構造にする（007はこの形）
- 大規模自己対戦（数万戦/プロセス）はlibcgのメモリリークでOOMする報告あり。ワーカープロセスをN戦ごとに使い捨てる設計にする（knowledge/discussions/709152）

## プールと主力候補の選定基準（2026-07-10改訂、EXP-035〜042で確立）

**用語**: ladder=本番のレーティング対戦環境（Top Episodesデータで観測）/ アーキタイプ=デッキ系統（classify_deckの分類）/ 個体=特定の60枚リスト（deck_signature。独自リストなら1エージェント、公開レシピは複数方策の混合値になりうる点に注意）


**プール（=対戦相手の分布モデル。目的は本番期待勝率の推定）**
- 構成は2段階: **第1段=アーキタイプをシェアで絞る**（シェア上位からカバー率90%目安。現行v2はシェア2.6%以上の9系統=合計89.6%）、**第2段=各枠の代表を個体で選ぶ**（そのアーキタイプのトップ個体レプリカ）。重みは最新メタシェア。/episodesでシェアが動いたら一括更新（過去weighted値との比較を切断するのでEXPに構成バージョン明記）
- **テール（カバー外の約10%）はweighted値に映らない**。主力候補の最終確定前にテール数系統とのスポットチェック（各100戦程度）を必須とする（Comfey系等。7/9メタスナップショットの「公開Alakazam参照はComfeyに17.5%」の教訓）
- レプリカの製法は**軽量逆設計で十分**（リプレイ行動頻度集計→islet骨格に焼き込み。フル精読製の流用も可＝合格条件は製法でなくゲート）。合格ゲート=本番対面表と±5pt以内。超過したら「負けトレース1本の精読」で補正（EXP-039: 50.7→73.0%）。フル精読の新規作成は不要（枠の忠実度誤差×重みは評価CIのノイズ床未満）
- **血縁バイアス**: 候補の元個体（または兄弟レプリカ）がプールに入っている場合、その枠はほぼミラー（≈50%保証）となりweighted値が甘く出る（EXP-005の一般化）。**該当候補の合否はweightedだけでなく、他候補との直接対戦+本番A/Bを併用する**（現行: 036↔035、038↔038本人が該当）
- 例外: その枠を相手に主力を磨く場合は忠実度が転移を決めるためフル格上げを検討（実質、重み最大のAlakazam枠のみ正当化される）
- **フォールバック**: n≥200の個体が存在しないアーキタイプ（2026-07-08時点: Lucario最大n=90、Dragapult最大n=116）は較正済み自作（007=本番136戦実測）or 公式方策（021）で代替。個体選択はWR最優先だが差がノイズ圏（±1pt未満）なら戦数の多い方を可とする
- 注意: トップ個体構成のためプールは本番全体よりやや辛口（狙うレート帯のモデル）。壁系など対面特化知識が効く対面はレプリカ対戦値が±14-35ptズレるので本番対面表を正とする（EXP-042）

**主力候補（=自分が使うデッキ+方策）**
- 選定=ladderの**個体勝率上位**（個体勝率はそれ自体シェア加重の期待勝率）。条件: 個体戦数≥200（未満はノイズ/winner's curse。閾値を下げても上位が変わらないことは2026-07-08データで確認済み: 60%台は3個体のみ）+ ローカル再測定で確定
- 候補は**フル精読**で作る（勝敗両方の試合を精読→条件文ルール仕様書→実装の2段階、EXP-035形式。軽量比+24.5ptの実績）。**軽量逆設計の背骨で主力にするのは禁止**（プール相手役専用。実証: 軽量背骨090→フル精読背骨092で対壁+9.75pt、EXP-092）。手順の詳細は [knowledge/study/backbone_recipe.md](knowledge/study/backbone_recipe.md)（全数集計を先→目的的精読4〜8戦→較正対象はML層の有無で変える）
- シェアは選定基準でなく**将来リスク管理**に使う: 流行→被対策で勝率が落ちるため、異アーキタイプの主力候補を複数保持し、最終提出直前に/episodesの最新メタで確定する（決定の先送りが正しい戦略）

## 実験コードの衛生（2026-08-10 追加。EXP-120で事故）

- **実験で棄却した変更は、その場で共通モジュールから外すか、特徴グループを切って既定offにする**。
  `tools/featgen_core.py` に棄却済みの軸A（相手の隠れ情報推論）を残した結果、
  **以降に生成した特徴器9本すべてに自動混入**した（除去で+0.31pt）。
  特徴を足すときは `_GRULES` にグループを登録し、`gen_featurizer.py` の `GROUPS`（既定集合）に
  入れるかを明示的に決めること
- **配布ドキュメントとエンジン実装が食い違うことがある**。`cg/api.py` の docstring は
  「ex は MegaEx を含む」と書くが実装（Api.h）は排他。**判定に使うフラグは必ず実物で確認する**
- 提出物にはエージェントディレクトリの全ファイルが入る。**使わないモデルを置きっぱなしにしない**
  （104に model_104_v.npz が残り、提出tar.gzが6.9MB→12.0MBになっていた）

## 知識ベースの運用方針

- `knowledge/BACKLOG.md` は**エージェント（デッキ+方策）のアイデア専用**。基盤・ツール・調査のタスクはBACKLOGに入れず、必要になった時点で直接実施し、成果はINSIGHTSや該当ツールに反映する
- 評価基盤の既知の改善候補（未実施）: 敗因reason別集計、方策間の行動一致率測定（メタシェア加重プール評価は arena/run_pool.py として実装済み）
- AIVAT型の制御変量による評価分散削減（arXiv 1612.06915）: **検証済み・効果小で保留**（arena/varred_eval.py）。CUPED実装は無バイアス（007v007で点推定0.496不変）だが、無バイアスに使える共変量は処置前=ターン≤2の盤面に限られ、それは長いゲームの勝敗をほとんど予測できない（ρ²=0.07、CI幅3.6%縮小のみ）。強い中盤シグナルは処置後でバイアスを生むため、無バイアスに使うにはAIVAT本体（チャンスノード補正）が必要=研究グレード重量実装。**教訓: CRN不可の本環境では分散削減にフリーランチ無し。微調整6連敗は「測れていない」でなく「改善が本当に小さい」の証左（大改善なら弱い共変量でも生CI±5%で検出できたはず）**、公開されたエンジンソースの高速化ビルド（RL自己対戦・大規模スイープの基盤。knowledge/discussions/717141）、**classify_deckの誤分類修正**（tools/get_episodes.py: Mega Starmie exがstage1判定で同居stage2に看板を奪われる→megaEx最優先に。現状メタレポートがStarmie系を過小評価。EXP-017）、**スコア帯別メタ集計**（LB帯でチーム層化→公開エピソードからデッキ復元。自レート帯の相手分布を体系取得=081型デッキ採否や対面優先度の判断材料。設計図=knowledge/notebooks/ptcg-ai-battle-leaderboard-deck-meta-by-score-band.md）

## 提出形式

- `.tar.gz` のトップレベルに `main.py` と `deck.csv`（60枚、同名カード4枚まで、基本エネルギーは無制限、ACE SPECは1枚まで）
- **1提出=1デッキ固定**（運営ルール。ゲームごとのデッキ切替は禁止、Strategyカテゴリで負評価）
- **LBレートは高分散**（同一エージェント2枠で150〜400pt差の報告例）。単発のLBスコアで優劣を判断せず、最終提出は同一エージェントを2枠に出すのが実践知（knowledge/discussions/712621）
- Kaggle実行環境では deck.csv は `/kaggle_simulations/agent/deck.csv` に置かれる（sample main.py の read_deck_csv がその形）
