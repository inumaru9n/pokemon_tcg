# pok-mon-tcg-ai-battle-meta-snapshot-09-july（Pilkwang Kim系列、07-04版と同一著者・同一テンプレート）

## 3行サマリ

- 07-01〜07-08のladderメタ推移（TREND_CSV）と全ペア対面表（MATCHUP_CSV）を集計し、「シェアは最大だが勝率50%割れのAlakazam」と「Starmie/Cynthia's Garchompに狩られ始めて弱体化したはずが再びシェア回復したCrustle」の綱引きを語るnotebook。埋め込みデータはJSON1行のべた書きで、コード出力（グラフ・display）は保存されておらず本文の主張は埋め込みCSV文字列から直接検算した。
- 直近日付は07-08（タイトルの07-09時点では未反映、07-04版notebookも07-03データが最新だったのと同じ1日ラグのパターン）。全候補（自作Dragapult系4種＋公開Alakazam参照1種）とそのペア組み合わせは、Field-held/Top-meta/Tail-stressの3視点複合スコアによる厳格ゲートで**全滅HOLD_DO_NOT_SUBMIT**——著者自身が「どの案も安全な昇格ではない」と結論している。
- 後半はAGENT_SELECTION（"A"/"B"）で選んだ候補のmain.py+deck.csv+公式cgを束ねてtar.gzを作る提出ビルダー（智の内容はbase64埋め込みで中身は精読していない）。

## 観点別の数値（要求された1〜4）

### 1. 最新アーキタイプ別シェア・勝率（07-08時点、TREND_CSV最新行）と07-04版（07-03時点）との差分

| アーキタイプ | 07-03（04-julyノートブック内の埋め込みデータ） | 07-08（本ノートブック最新行） | 傾向 |
|---|---|---|---|
| Alakazam/Dunsparce | share 25.7% / win 44.3% | share **31.6%** / win **48.2%** | シェア↑勝率↑（最大母集団だが依然50%割れ） |
| Marnie's Impidimp\|Munkidori | share 17.8% / win 59.6% | share **11.3%** / win **44.7%** | シェア↓勝率↓（大幅減速） |
| Marnie's Grimmsnarl ex\|Impidimp | share 3.9% / win 56.1% | share **5.4%** / win **58.1%** | シェア↑勝率安定良好（Munkidori系と明暗が分かれた、というのが本文の主張） |
| Crustle | share 3.8% / win 50.5% | share **13.1%** / win **52.0%** | シェア急回復（3倍以上） |
| Cynthia's Gabite\|Gible（Garchomp系） | データ内に**行自体が存在しない**（07-03時点シェア実質0） | share **6.1%** / win **56.9%** | 07-04以降の新興勢（INSIGHTSの「07-06新出9.1%」と符合、その後6%台へ微減） |
| Starmie（Mega Starmie ex） | share 9.2% / win 54.1% | share **6.9%** / win **57.2%** | シェア↓勝率↑ |
| Mega Kangaskhan ex\|Meowth ex | share 4.6% / win 49.6% | share **3.2%** / win **54.6%** | シェア↓勝率↑（INSIGHTSの07-06急増12.4%からは大きく後退。ただし本notebook内の07-06行は7.5%で、INSIGHTS記載の12.4%＝episodes実測とは食い違う。データソース差に注意、下記「注意点」参照） |
| Archaludon | share 4.9% / win 38.9% | share **3.5%** / win **48.3%** | シェア↓勝率↑（依然50%割れ） |
| Lucario（自分たちの系統） | share 5.0% / win 51.0% | share **4.3%** / win **45.8%** | シェア↓勝率↓ |
| team_rocket_spidops | 07-04版データに**存在しない** | share 1.2% / win 53.3% | 新規出現（下記「新興デッキ」参照） |

一行差分まとめ: **07-04時点はAlakazam(25.7%)/Marnie-Munkidori(17.8%)の2強構造だったが、07-08はAlakazam(31.6%)がさらに肥大化する一方Marnie-Munkidoriは11.3%へ失速し、代わりにCrustleが3.8%→13.1%へ急回復、Cynthia's Garchompが0%→6.1%で新興、Archaludon/Kangaskhanはシェアを落としつつ勝率だけ上げるという「小さく強く」への収斂が起きている。**

### 2. Starmie系（Mega Starmie ex / Froslass）の位置づけ

- Starmie本体: share 6.85%（07-03の9.2%から減少）、win 57.2%（54.1%から上昇）。シェアは縮んでいるが質は上がっている中堅アーキタイプという位置づけ。対面表では **Crustle撃破率83.9%**（`crustle,starmie` 118戦でcrustle側勝率16.1%）、**Alakazam撃破率62.2%**（`alakazam_dunsparce,starmie` 251戦でalakazam側勝率37.8%）と、メタ最大2勢力に強い対面を持つとnotebookは位置づけている。
- Froslass派生（`other:Mega Froslass ex|Snorunt[ID:860]`）はshare 0.46%・win 45.8%（n=48）とごく小さいテール扱いで、Starmie本体とは別カウント。ノートブックの本文でもFroslass単独への言及はなく、Tail Stress Ledgerの一部としても名指しされていない（Comfey/Cubchoo系がテールの主役）。

### 3. レート帯別の分布

- **該当データなし**。本notebookにレート帯別（rating band/percentile別）のシェア・勝率テーブルは存在しない。唯一のレート言及は候補プロファイル`live_b_alakazam_944`という命名中の「943.8」という単一の参照レート値のみで、分布ではなく1エージェントの実測アンカーとして使われている。トップ帯特有の傾向を語る記述も見当たらない。

### 4. 新興・急上昇デッキ

- **Crustle**: 07-03の3.8%→07-08の13.1%へシェア3倍増で最大の「急回復」。ただし本文は同時に「Starmie/Cynthia's Garchompという明確なカウンターに狩られている最中の回復」と評しており、単純な強化ではなく一時的な数合わせ的増加という含み。
- **Cynthia's Garchomp ex（Cynthia's Gabite|Gible）**: 07-03時点データ不在→07-08で6.1%・win56.9%まで定着。INSIGHTSの「07-06新出9.1%」からは微減しており、ピークを過ぎつつある可能性。
- **team_rocket_spidops**: 07-04版データに存在しない完全新顔。share1.2%・win53.3%（n=120）とまだ小さいがプラス収支。
- Tail全般（Comfey/Handheld Fan・Comfey/Yveltal・Cubchoo/Dunsparce・Air Balloon/Buneary等）が本文で「単なるノイズでなく、ローカルパネルが過大評価を生む要因」と明言されている。実際TAIL_CSVでは公開Alakazam参照candidate（live_b_alakazam_944）が Comfey/Handheld Fanに17.5%、Comfey/Yveltalに25%、Cubchoo/Dunsparceに38.75%としか勝てておらず、トップ2アーキタイプ対策だけでは足元をすくわれるテールが実在する。

## 盗めるアイデア

- ★ **3視点複合ゲート（Field-held / Top-meta / Tail-stress）による合否判定**: 単一プール平均でなく、「メタ本体（ladder全体加重）」「公開実績のある強豪」「崩されやすいテール（Comfey/Cubchoo/Air Balloon等の少数派）」の3枠それぞれにスコアと閾値を持たせ、どれか一つでも基準未達なら`HOLD_DO_NOT_SUBMIT`という運用。現状の`arena/pool.json`はメタシェア加重の単一プールのみで、「テール専用の弱点検出パネル」という発想は無い。プールに小シェアだが対策不足だと刺さるアーキタイプ（Comfey系・Air Balloon系相当）を数枚加えて別集計する価値はあるかもしれない。
- 機械可読な合否ラベル（`strict_submit_status`, `coverage_gap_status`等）をレポートに埋め込む運用。EXPレポートのverdict欄をこの粒度（validated/rejected以外に「HOLD、理由=カバレッジ不足」のような中間状態）で構造化するアイデアの参考になる。
- 「ペアをmax-of-twoで評価する」（`PAIR_CSV`: 2つの候補が互いに弱点を補完するか）という発想。本コンペは1エージェント固定なので直接は使えないが、discussions/712621の「同一エージェントを2枠に出す」実践知の背景にある発想（各候補の弱点相関を見る）として参考になる。
- トップ2アーキタイプ（Alakazam/Marnie）への対策だけに閉じず、シェア1〜2%台の「テール」に足元を掬われていないかを定期的にチェックする視点。我々のプール構成（`arena/pool.json`）にシェア下位だが対策依存度が高そうな相手（例えば妨害・ロック系、Comfey系のような特殊戦術）が含まれているか確認する価値はある。

## 注意点・疑問

- 本notebookのセルoutputsは全て空（未実行 or 出力除去済み）で、グラフ・display表は目視確認できない。本メモの数値はすべてコード文字列内に直書きされたCSVリテラル（`TREND_CSV`/`MATCHUP_CSV`/`CANDIDATE_CSV`/`TAIL_CSV`/`PAIR_CSV`）をripgrepで直接抽出したもの。集計コード自体（filteringやgroupby）は実行されておらず、著者の主張（"Starmie beats Crustle hard"等）が実際にこの生データから正しく導出されているかは未検証（要検証）。
- **07-03時点の数値はknowledge/INSIGHTS.mdの実測値（classify_deck修正後）と一致**（Alakazam 25.7%/44.3%、Starmie 9.2%/54.1%が完全一致）。このため少なくとも07-03分は信頼度の高い実データの可能性が高い。ただし**07-06時点のKangaskhanシェアは本notebook内で7.5%だが、INSIGHTSの独自episodes分析では12.4%**と食い違っており、集計方法（対象母集団・classify_deckロジックの版）が我々の分析と異なる可能性がある。本notebookの絶対値は方向感の参考にとどめ、意思決定の根拠には我々自身のepisodes分析を優先すべき（要検証）。
- classify_deckのラベル付け方法（何のカードでアーキタイプを判定しているか）が本文に記載されておらず、INSIGHTSで指摘されている「megaEx優先で無いと看板が奪われる」問題（Cinderace/Dusknoir誤分類等）がこの著者側の集計にも及んでいるかどうかは不明。
- 候補デッキ`phantom_archaludon_focus_v1`はCANDIDATE_CSVの`candidate_archetype`列で**"dragapult"**に分類されている。名前に"archaludon_focus"とあるのはターゲティング・プロファイル名であって、デッキ自体がArchaludonというわけではなさそうに見える（B案`phantom_dragapult`も同じdragapult分類）。BACKLOG I-004（Dragapult ex、EXP-021でプール相手役採用済み・候補としては却下）と同系統である可能性が高いが、レシピの詳細（base64埋め込み）までは精読していない（要検証）。
- 本notebookはladder全体（ラダー戦全体の観測パネル）ベースと見られ、レート帯別分布は無い。INSIGHTSで確認済みの「本番895帯はArchaludon31%が最多でladder全体シェア（本notebookでは3.5%）とは全く別物」という既知の乖離が改めて裏付けられる形になっており、矛盾ではなく想定通り。
