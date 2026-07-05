# strong-start-baseline-agent-v10-lb-950（Roman Rozen、LB950+、118票）

## 3行サマリ
- 運営配布のサンプル「Mega Lucario ex」エージェント（Makuhita/Hariyama＋Riolu/Mega Lucario ex＋Lunatone/Solrock）をベースに、(1) クラッシュ防止ラッパー、(2) Crustle壁対策（ex/megaEx攻撃をCrustleの「Mysterious Rock Inn」で0ダメージ化される問題を検知し、非exのHariyamaで迂回する）、(3) オプションのForward Search（`USE_SEARCH=False`がデフォルト）の3点を追加した堅牢性重視の改良版。
- デッキもデータドリブンに再調整：基本ポケモン10→12枚（マリガン率25.9%→19.1%と自己対戦で実測）、Hariyama/Switch増量、Gravity Mountain採用（相手のBattle Cageを上書きしてBossでベンチ狙撃を可能にする）。ノートブック内に実際のエンジンでの自己対戦検証コード（mirror・vs random・Crustle再現デッキとのA/B、40〜30戦規模）が組み込まれている点が特徴。
- Crustle対策ポリシーON/OFFのA/Bで「10%→70%」への改善を自己対戦で実測したと主張（ただし対戦相手はノートブック内で著者が再構築した簡易Crustleエージェントであり、実際のリーダーボード上のCrustle実装そのものではない）。

## 盗めるアイデア
- ★ Crustleの「ex/megaEx攻撃無効化」アビリティを検知し、`crustle_immune`のときそのオプションのスコアを-10000にして選択させない、という「相手の耐性/無効化を検知して自分の攻撃選択を動的に変える」パターン。他の耐性カード（Milotic ex、Drednaw等のダメージ無効系）にも一般化できる考え方。
- ★ ノートブック内に「本物のcg engineでの自己対戦検証」を組み込む文化（`battle_start`/`battle_select`/`battle_finish`を直接呼び、mirror・vs random・vs 特定アーキタイプ相手に数十戦回して勝率を出す）。デッキのA/B（basics 10→12等）を「勝率の実測」で判断している点は本プロジェクトの実験ワークフローと同じ思想であり、手法として参考になる。
- ★ `evaluate_state`によるボード評価関数（サイド差×10000を支配項に、盤面のエネルギー数・HP・手札枚数等を加点）を用意し、`search_begin`/`search_step`で「1手先の各候補を打ってから残りターンをgreedyに終端まで進め、evaluate_stateでスコアリングする」という1-ply lookahead + greedy rolloutの実装（`search_plan`）。ただし`USE_SEARCH=False`がデフォルトで、SDKでの動作確認とターンあたりの時間予算検証をしてから有効化するよう明記されている（本プロジェクトのCLAUDE.mdの注意と一致する慎重な姿勢）。
- 「クラッシュ=負け」という前提を踏まえた`agent()`ラッパー：`to_observation_class`の例外も含めあらゆる例外をキャッチし、`_legal_fallback`（`minCount`個の先頭インデックスを返す）にフォールバックする設計。エラー0件を実測で確認する自己対戦ハーネスとセットで運用している。
- `EXTRA_CONTEXTS`フラグ（ベンチ配置・捨て札・ダメージカウンター配置のスコアリングを追加するかどうか）をデフォルトOFFにし、「ミラー戦で唯一の方針差だったのに~70-30で負けていたため、疑わしい改悪として保留し、A/Bで確認してからONにする」という慎重な運用。未検証の追加ロジックを機能フラグで隔離し、検証してから有効化するという開発姿勢自体が参考になる。
- Prize換算（`prize_count`）でLegacy EnergyやLillie's Pearlによるサイド減少効果を考慮している点（ex系の正確なサイド計算）。

## 注意点・疑問
- Crustle戦での「10%→70%」やdeck再調整の「52.5%」等の数値は、著者が自作した簡易`crustle_agent`（優先度固定のダミー実装）を相手にした検証であり、実際のリーダーボード上のCrustleエージェント（例えば本グループの`beating-the-day-1-1-crustle-bot`のような実装）とは異なる可能性がある。実際の強豪Crustle実装に対しても同様の改善幅があるかは未検証。
- `USE_SEARCH=False`がデフォルトであり、このノートブックの「LB950+」の実績はサーチ無しの純粋ヒューリスティック版によるものと思われるが、その点はノートブック内で明示されていない（要確認）。
- Forward Searchの`search_begin`の入力形式（`obs.search_begin_input`）はSDKビルド依存と明記されており、本プロジェクトの`cg/api.py`で同じインターフェースが提供されているか要検証。
