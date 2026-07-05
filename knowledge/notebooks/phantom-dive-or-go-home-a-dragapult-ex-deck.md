# phantom-dive-or-go-home-a-dragapult-ex-deck（Arin、34票）

## 3行サマリ
- Dreepy/Drakloak/Dragapult ex（Phantom Dive：アクティブに攻撃＋ベンチに6個の任意配置ダメージカウンター）を主軸に、Budew（Itchy Pollen：アイテム封じ）・Crushing Hammer（コイントスでエネルギー破壊）・Fezandipiti ex等を組み合わせた「メタを読んで刺さる型を選ぶ」デッキ選定が主眼のノートブック。著者はメタ分析（Colress/Mist壁~25%、Lucario~15%、Alakazam~12%、Petrel~10%など）からDragapultが対メタ最強と結論づけている。
- エージェント実装は7本中最も技術的に凝っており、①Phantom Diveの6個のダメージカウンターを相手の場（アクティブ＋ベンチ）にどう配分すればサイド獲得数を最大化できるかを**部分和（subset-sum風）のDFS探索で全パターン列挙**し最良の配分を選ぶ、②自分のデッキ内訳を（サイド未確定も含め）カード単位で追跡する`card_counts`（deck - 手札 - 場 - 捨て札 - 効果解決中カード、で残数を逆算）という、prize-card-tracking notebookと類似だがより実装が凝ったカード計数機構、③「Mist Energyのような効果無効カードはダメージ自体は防がない」という細かいルール理解に基づく実装（`no_damage_dex`/`no_damage_counter`）を持つ。
- 「Dragapult exが対Colress/Mist壁81%、対Petrel 79%、対Lucario 50%、対Alakazam有利」という具体的な勝率主張があるが、これらの数値を検証する自己対戦コードはノートブック内に一切なく、根拠は不明（他の6本中、strong-start-baselineとmultiply-agentのみ自己対戦検証コードを持つ）。

## 盗めるアイデア
- ★ Phantom Dive型の「範囲ダメージ＋自由配置ダメージカウンター」攻撃に対する最適配分探索：相手の場の各ポケモン（アクティブ＋ベンチ）についてHPを列挙し、残りダメージ量の範囲内で「どの部分集合にカウンターを配れば合計サイド獲得数が最大化されるか」をスタックベースのDFSで全探索する`main_option_proc`のロジック（`counter_indices`列挙＋`plan_score`比較）。範囲攻撃・分割配分系の攻撃を持つデッキ全般に転用できる汎用アルゴリズム。
- ★ 自分のデッキの残数を「デッキ全体 - 可視領域（手札・捨て札・場・スタジアム・効果解決中カード）」の消去法で逆算する`set_card_counts`（`add_card_count`が`serial`を使った二重カウント防止付き）。prize-card-trackingノートブックのPrizeTrackerと同種の発想だが、対象を「自分のデッキ内訳」（何を引く確率が高いか）に応用しており、Ultra Ball等の非公開探索効果の評価に有効。
- ★ 「効果Xはダメージそのものではなく“効果”のみを防ぐ」という細かいルール差（Mist Energy）を明示的にコード化し、耐性カードの誤判定を避けている点（`no_damage_dex`＝ダメージ自体無効、`no_damage_counter`＝ダメージカウンター配置のみ無効、を区別）。相手の耐性・防御ギミックを実装する際は「何を防ぐ効果か」を正確に区別する必要がある、という一般的な注意点として参考になる。
- 相手のログから「直前ターンにItchy Pollen（アイテム封じ）が使われたか」「自分のポケモンが直前ターンにKOされたか（`pre_ko`）」を検出し、それぞれアイテムプレイの抑制やFezandipiti ex/Unfair Stampの発動タイミング判断に利用している点（ログベースの状態推定パターン）。
- サポーター1枚制限下で複数の候補（Crispin/Brock's Scouting/Boss's Orders等）を`hand_score`で横断比較し、最もスコアの高い1枚を`use_support`として選ぶ「サポーター選択の統一スコアリング」設計。

## 注意点・疑問
- ノートブックが主張する具体的なマッチアップ勝率（対Colress/Mist壁81%等）を裏付ける検証コード・データが本ノートブックには存在しない。「実際のladder結果を使った」との記述はあるが再現性のある数値ではなく、信頼性は要検証としてフラグを立てる。
- 部分和探索`main_option_proc`のダメージカウンター配分ロジック（`remain_damage`の増減とスタック`ci`の操作）はやや複雑で、コードを読んだ限り正しく全パターンを尽くしているか（同じ組み合わせの重複列挙や見落としがないか）は要検証。
- `no_damage_dex`/`no_damage_counter`の対象カードID列挙（Drednaw, Milotic ex, Sylveon, Crustle, Poltchageist, Empoleon ex等）はカードプールの正確な理解に依存しており、ID・効果の正しさ自体は未検証。
- Fezandipiti ex/Unfair Stamp等のACE SPEC・レアカードの取り扱い判断（`pre_ko`後にのみ高スコアにする等）はゲーム固有ルールの理解に強く依存しており、細部の正当性は要検証。
