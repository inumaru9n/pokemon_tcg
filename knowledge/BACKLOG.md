# BACKLOG — エージェントアイデア台帳

**この台帳はエージェント（デッキ+方策）のアイデア専用。** 1行 = `/experiment` で `agents/NNN/` を作って対戦検証できる粒度のアイデア。
基盤・ツール・調査タスクはここに入れず、必要になったタイミングで直接実施する。

状態: `proposed` → `in-progress` → `validated`（採用） / `rejected`（棄却） / `blocked`

- 新アイデアは必ず既存行との重複を確認してから追記する
- 検証したら状態とEXPリンクを更新する
- 優先度: H（勝率への直結度が高い） / M / L

| ID | アイデア | 出典 | 優先度 | 状態 | EXP |
|----|---------|------|-------|------|-----|
| I-001 | 公式サンプル（Mega Lucario ex ルールベース）を提出形式に移植しベースライン化 → 以降の全比較基準になる | notebooks/a-sample-rule-based-agent-mega-lucario-ex-deck.md | H | validated | [EXP-001](experiments/EXP-001.md) |
| I-002 | 本番ladderトップメタのMarnie's Grimmsnarl exレシピ移植（sig 108ac5bde2、63.6%/1137戦）+ サンプル型スコアリング方策 → メタ最強デッキをそのまま使う | episodes/2026-07-03.md | H | rejected（判定保留: 400戦で52.4%、CI 47.5-57.2%が50%を跨ぐ。対Lucarioはladder対面表でも五分であり、デッキ自体の否定ではない） | [EXP-002](experiments/EXP-002.md) |
| I-003 | Crustle耐久デッキ（sig 89d834e4d4、65.2%/89戦）+ 固定優先順位の単純方策 → 「単純な方策でも勝てるデッキ」思想の検証 | episodes/2026-07-03.md, notebooks/beating-the-day-1-1-crustle-bot.md | H | rejected（プール加重38.3%、CI上限44%<50%。対策可能な相手には壁が通らない） | [EXP-003](experiments/EXP-003.md) |
| I-004 | Dragapult ex移植: 公式サンプルの部分和DFSダメカン配分 + ladderレシピ（sig 2b31cd4174、59.5%/121戦） → 複数同時KOで平均サイド獲得効率を上げる | notebooks/a-sample-rule-based-agent-dragapult-ex-deck.md, episodes/2026-07-03.md | H | proposed | - |
| I-005 | islet式「MAIN時に1手先攻撃プラン確定→全サブ選択をプラン整合で採点」を現行ベストに導入 → 探索なしで1ターン内の行動一貫性を確保 | notebooks/simple-baseline-matchup-tests.md | H | validated（プール加重65.4%、新現行ベスト） | [EXP-004](experiments/EXP-004.md) |
| I-006 | MAIN選択限定ビームサーチ（1手1.5秒予算+greedyフォールバック、600秒/ゲーム制限は確認済み） → ヒューリスティックの盲点を探索で補いLB940実例に並ぶ | notebooks/multiply-agent-best-940-lb.md | H | rejected（ベース004に44.5%、対Marnie9pt悪化。自ターン末静的評価では調整済みヒューリスティックに勝てない） | [EXP-006](experiments/EXP-006.md) |
| I-007 | 消去法カードトラッキング層（相手サイド推定PrizeTracker + 自デッキ残り内訳逆算）を現行ベストに追加 → サーチ/ドロー/Boss判断の精度向上 | notebooks/prize-card-tracking-1300-starmie.md, notebooks/a-sample-rule-based-agent-dragapult-ex-deck.md | M | proposed | - |
| I-008 | gated対面テック: 相手盤面からMarnie系（シェア21.7%・勝率59%）を検出したら専用スコア上書き → 副作用ゼロでトップメタ対面だけ強化 | notebooks/ptcg-mega-lucario-ex-v62.md, episodes/2026-07-03.md | M | rejected（対Marnie 62.0%で改善なし、ミラー47.5%。係数無調整のgated techは効かない） | [EXP-005](experiments/EXP-005.md) |
| I-009 | 相手アーキタイプ推定→サブポリシー切替の二層方策（複数ノートブックで収束したパターンの一般化） → 苦手対面の局所改善を安全に積み上げる | notebooks/a-sample-archaludon-75-wr-vs-my-1300-starmie.md, notebooks/pok-mon-tcg-ai-battle-meta-snapshot-04-july.md | M | proposed | - |
| I-010 | 決定化1-ply lookahead + greedyロールアウト（search_begin/search_step、盤面評価関数付き） → ビームサーチより軽い探索の費用対効果を測る | notebooks/strong-start-baseline-agent-v10-lb-950.md, notebooks/reinforcement-learning-and-mcts-sample-code.md | M | proposed | - |
| I-011 | 相手最大打点見積もりによる防御判断（交代・回復・ケープ添付の閾値化）を方策に追加 → 無警戒な被KOを削る（山札切れガードはEXP-004で導入済み） | notebooks/a-sample-archaludon-75-wr-vs-my-1300-starmie.md | M | proposed | - |
| I-012 | スコア関数係数の進化的最適化（変異→arena対戦→選抜、学習済み重みをmain.pyに焼き込み） → 手調整の係数を実測で置き換える | notebooks/ptcg-tiny-rl-to-submission-baseline-guide.md | L | proposed | - |
| I-013 | ミル/デッキアウト軸（Great Tusk + Crustle壁） → ダメージレース前提のメタ全体に対する非対称戦略 | notebooks/i-have-one-rear-card.md | L | proposed | - |
| I-014 | 上位エピソード行動ログからの模倣学習方策（episodes JSONのsearch_begin_input+action traceを教師に） → 強エージェントの行動傾向を直接写す。**EXP-012の結論（状態評価でなく行動の直接学習が残り筋）により昇格** | notebooks/en-replay-archetype-analysis.md, episodes/2026-07-03.md, experiments/EXP-012.md | H | proposed | - |
| I-015 | 002のGrimmsnarl方策改良（Punk Upの5枚配分最適化、Adrena-BrainのKO直結判断、Boss温存、Froslass採用判断の見直し） → 五分の対Lucarioを崩す | experiments/EXP-002.md | M | proposed | - |
| I-016 | 探索の上書きを「今ターンの確定勝ち（lethal）検出」に限定（決定化探索で勝ち筋が見つかった時だけヒューリスティックを上書き） → 評価関数不要の高精度な部分だけ探索を使い、EXP-006の失敗を回避 | experiments/EXP-006.md | H | validated（ミラー52.4%で劣化なし+発火ゲーム敗北0、新現行ベスト） | [EXP-007](experiments/EXP-007.md) |
| I-017 | 自己対戦ログで価値関数を学習（盤面特徴→勝敗のロジスティック回帰/小型MLP、重みはmain.pyに焼き込み）し、EXP-006探索フレームの手書き評価関数を置換 → 「探索は評価関数が律速」の穴を学習で埋める | experiments/EXP-006.md, notebooks/reinforcement-learning-and-mcts-sample-code.md, notebooks/ptcg-tiny-rl-to-submission-baseline-guide.md | M | rejected（AUC0.85のVでも全対面劣化。状態予測精度と行動ランキング能力は別物） | [EXP-012](experiments/EXP-012.md) |
| I-018 | 本格RL: 公式エピソードでBC初期化→自己対戦で方策改善（AlphaZero風方策価値ネット or PPO）→ 手書き方策の天井を突破する長期投資（I-014/I-017の発展形） | notebooks/reinforcement-learning-and-mcts-sample-code.md, notebooks/en-replay-archetype-analysis.md | L | proposed | - |
| I-019 | 対Marnie草弱点狙撃デッキ: episodesから草アーキタイプの高勝率レシピを探索、なければLeafeon ex（1進化/HP270/230打点）軸をislet骨格で構築 → シェア21.7%のGrimmsnarlラインに弱点2倍が刺さる | episodes/2026-07-03.md, カードプール分析 | H | proposed | - |
| I-020 | Alakazam移植（sig 779e01b394、59.4%/106戦）: 最多シェア25.7%のアーキタイプをプール・自陣に追加。手札連動打点はsue124の手札増減min/max積算パターンで → メタ最多対面の理解とプール多様化を兼ねる | episodes/2026-07-03.md, notebooks/rule-based-not-psychic-alakazam-best-5th.md | H | rejected（提出候補としては対Marnie37%で不可。ただし対007に53%と強く、プールへ採用=当初目的の半分は達成） | [EXP-010](experiments/EXP-010.md) |
| I-021 | 2-ply防御評価: EXP-006探索フレームの評価タイミングを「自ターン末」から「相手の最善返し1手後」に延長 → 近視眼（相手の返しを見ない）というEXP-006の敗因を直接修正 | experiments/EXP-006.md | H | rejected（ミラー43.0%で有意劣化。ただし対Marnie66.5%と対面依存の効果あり→I-027） | [EXP-008](experiments/EXP-008.md) |
| I-022 | 004の対面ガード係数スイープ: 主要係数（対Crustleエネ優先260、対水ケープ12800等）±数段階をrun_poolで格子探索 → EXP-005の教訓「gated techは係数調整とセット」の実践（I-012の限定版） | experiments/EXP-005.md, notebooks/ptcg-mega-lucario-ex-v62.md | M | proposed | - |
| I-023 | Boss's Orders温存ルール: 「ガスト対象のKOでサイド2枚以上取れる or 勝ち確」以外では温存する明示ガードを004に追加 → EXP-005で観測したガスト浪費を直接抑制 | experiments/EXP-005.md | M | proposed | - |
| I-024 | 初手セットアップ最適化: 敗因ログから事故パターン（エネ引けず・ベンチ展開不足）を特定し、SETUP_BENCH展開幅・エネ温存判断を改善 → vs randomですら5%落とす事故率を削る | experiments/EXP-001.md, arena/results/ | M | proposed | - |
| I-025 | メタ事前分布ガード: 相手アーキタイプが盤面から確定する前（1〜2ターン目）はメタシェア事前確率（Alakazam25.7%/Marnie21.7%）で対面ガードを先行発火 → ガード発動の1〜2ターン遅れを解消 | episodes/2026-07-03.md, notebooks/simple-baseline-matchup-tests.md | M | proposed | - |
| I-026 | Munkidori+悪エネパッケージを004に混載（ダメカン移動3個/ターン） → Lucarioの「あと20〜30足りない」圏を埋める。闘悪2色のエネ管理コストが見合うか検証 | episodes/2026-07-03.md, experiments/EXP-002.md | M | proposed | - |
| I-028 | 007の対Alakazam強化: ladder対面表ではLucario有利64%だが007は53%止まり（sue124版が強い）。sue124のPrinciples（Battle Cage下では特性ポケモン温存等）を読み対策ガードを追加 → 最多シェア25.7%対面で+10ptの伸び代 | experiments/EXP-010.md, notebooks/rule-based-not-psychic-alakazam-best-5th.md | H | rejected（係数スイープ4値すべて効果なし。ベースライン自体が400戦で53.8%と判明し前提も一部崩れ） | [EXP-011](experiments/EXP-011.md) |
| I-031 | 007デッキにJudge(1213)を2枚差し+「相手手札7枚以上で優先プレイ」の方策 → Powerful Hand打点を80に制限しMarnieのセットアップも崩す構造的カウンター（両者手札4枚にリセット） | カードプール分析, experiments/EXP-011.md | H | proposed | - |
| I-032 | 007デッキにShaymin(343)を1枚差し（ルールボックス無しベンチへの攻撃ダメージ全無効） → Fezandipiti/Shadow Bulletのベンチ狙撃からRiolu/進化前を守る。sue124も同目的で採用済みの実績テク | カードプール分析, notebooks/rule-based-not-psychic-alakazam-best-5th.md | H | rejected（対Marnie -10.5pt有意悪化。ベンチ枠の機会費用>防御価値） | [EXP-013](experiments/EXP-013.md) |
| I-033 | 先手後手で方策を分岐（後手時のセットアップ・エネ配分・攻勢閾値を専用調整） → recordsでミラー先手56%/後手48.5%の非対称を確認。後手時の劣化を埋めれば全対面に効く | arena/results/EXP-007.json records分析 | M | proposed | - |
| I-034 | Archaludon（鋼テンポ）移植: episodesからレシピ復元+専用方策 → 06-29スナップショットでシェア急上昇・スコア率60%超の新興メタ。プール多様化も兼ねる | notebooks/pok-mon-tcg-ai-battle-meta-snapshot-06-29.md, episodes/2026-07-03.md | M | proposed | - |
| I-035 | 特殊エネ破壊テック（Enhanced Hammer 1081等）を007に1〜2枚 → Telepath(Alakazam)/Mist・Spiky(Crustle)/Grow等、特殊エネ依存デッキのテンポを崩す | カードプール分析 | M | proposed | - |
| I-036 | 手札ロックコントロールデッキ: Vivillon(1019、毎ターン相手手札をデッキ底へ)またはGothitelle(596)軸 → 手札リソース依存のメタ全体（Alakazam/Marnie）への非対称戦略。新アーキタイプ | カードプール分析 | M | proposed | - |
| I-037 | テンポ認識方策: 相手盤面にチャージ済みアタッカーがいない（次ターン攻撃不能）を検出したら交換より育成・展開を優先 → 見えている情報だけで安全なターンを識別しテンポ差を作る | arena/results records分析, experiments/EXP-004.md | M | proposed | - |
| I-038 | Meddling Memo(1103、アイテム版手札リセット)を007に投入 → Judgeと異なりサポート権を消費せず妨害できる。I-031と比較検証 | カードプール分析 | M | proposed | - |
| I-039 | Hero's Capeの付け先を動的最適化（相手の見えている最大打点との差分でHP+100が生死を分ける対象を計算） → 現在は対面固定ルール。防御系は微調整3連敗の対象外（targeting系でない） | カードプール分析, experiments/EXP-011.md | M | proposed | - |
| I-040 | I-014の具体化: BC教師を「Lucario系アーキタイプの上位エージェント」のepisodesに限定 → 同一デッキの行動だけ学ぶことで転移ギャップを消し、方策ヘッドを007の選択肢スコアリングに直結 | notebooks/en-replay-archetype-analysis.md, experiments/EXP-012.md | L | proposed | - |
| I-041 | エンジン列挙順の事前分布化: 全コンテキストのスコアに -選択肢index×ε のタイブレークを明示し、方策が扱わないコンテキストではエンジン順に完全委譲 → Discussion 713608の「先頭選択（B1）だけでランダムに88-90%勝つ=列挙順は強い良→悪順」という知見を吸収。現状は安定ソート頼みで、雑なスコアが良順を壊す文脈での改悪を防ぐ | discussions/713608-what-we-tried-ceilings.md | M | rejected（λ=2でもミラー41.2%と有意劣化。列挙順の価値は安定ソートの同点処理で既に取り込み済み） | [EXP-014](experiments/EXP-014.md) |
| I-042 | 対受動デッキ限定の評価探索: Crustle系（壁・受け）検出時のみ2-ply評価探索（008の資産）を有効化 → 713608で「探索は受け相手に+44.5pt、攻め相手に有害」の報告があり、EXP-008/009で観測した対面依存性と整合。合否はプール+vs 003の400戦スポット検証で | discussions/713608-what-we-tried-ceilings.md, experiments/EXP-009.md | M | proposed | - |
| I-029 | 相手アーキタイプ推定→本番レシピによる決定化: 相手の見えたカードから knowledge/episodes のアーキタイプを同定し、episodesの実レシピ60枚から既出カードを差し引いて相手の手札・山札を決定化 → プレースホルダー（全エネ）による偽の全知を減らし、lethal探索の精度（発火率・正確性）を上げる | Web（Emergent bluffing and inference with MCTS, orangehelicopter.com/academic/papers/cig15.pdf）, notebooks/prize-card-tracking-1300-starmie.md, episodes/2026-07-03.md | M | proposed | - |
| I-030 | 情報集合を保った探索（ISMCTS/EPIMC）: 決定化ごとに別の木を作らず単一の情報集合木で探索、または完全情報化を葉まで遅延 → EXP-006/008/012の敗因（strategy fusion・fake omniscience）の文献上の正攻法。実装重量級 | Web（Cowling et al. ISMCTS, eprints.whiterose.ac.uk; arXiv:2408.02380 EPIMC）, experiments/EXP-012.md | L | proposed | - |
| I-027 | 2-ply評価探索の対面ゲーティング: 相手アーキタイプ検出でミラー系（Lucario検出時）は無効化、異アーキタイプのみ有効化。または上書きマージンの対面別スイープ → EXP-008で観測した「対Marnie+4pt/ミラー-9pt」の対面依存性を利用 | experiments/EXP-008.md | M | rejected（ゲートは機能したが対Marnie62.5%=007と同一。EXP-008の改善はノイズで再現せず） | [EXP-009](experiments/EXP-009.md) |
