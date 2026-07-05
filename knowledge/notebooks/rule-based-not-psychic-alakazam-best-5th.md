# rule-based-not-psychic-alakazam-best-5th（sue124、大会2日目最高5位、67票）

## 3行サマリ
- Abra/Kadabra/Alakazam＋Dunsparce/Dudunsparceのドローエンジンで「Powerful Hand（手札枚数×20ダメージ）」を伸ばし切るコンボデッキ。Fezandipiti ex（Flip the Script）やHilda/Dawn等の追加ドロー効果も動員して打点を積み増す。
- 最大の実装ポイントは「このターン中に手札枚数をどこまで増やせるか」を厳密に列挙する`estimate_hand_increase`（進化ドロー、アビリティドロー、サポーター1枚制限、Enriching Energyのカード引き効果等を+/-で積算し min/max を算出）で、Powerful Handの理論上限ダメージを毎ターン計算してから、KOできる相手を選ぶ。
- サイド差・相手の防御手段（Mist Energy/Rock Fighting Energyの枚数とEnhanced Hammerでの除去可否）・自分の山札切れ回避（`safe_draws`のガード）まで含めた比較的網羅的なルールベース設計。定量的な勝率検証はノートブック内には無い（プレイ原則の記述のみ）。

## 盗めるアイデア
- ★ 「このターン中に増減しうる手札枚数」を要因ごとに列挙して min/max を計算する`estimate_hand_increase`パターン。Powerful Hand系（手札依存ダメージ）だけでなく、一般に「この後の行動でどれだけ手札/盤面が変化しうるか」を事前計算してから行動選択する設計は他のコンボデッキにも転用できる。
- ★ 山札切れ自滅の明示的ガード：`safe_draws = deck_count - my_prize_count - 1`（勝てるターンでない限りこれを下回るドローをしない）。デッキアウト負けを避けるための一般的なガード条件として汎用性が高い。
- 相手の場に特定カード（Duskull, Slowpoke/Froakie/Ogerpon/N's Darumaka, Dreepy/Drakloak/Dragapult系）が見えたときだけ対抗ポケモン（Psyduck, Shaymin）やスタジアム（Battle Cage）を投入する条件付きテック採用ロジック。「ハードカウンターは相手が見えてから初めて投入し、見えなければ腐らせない」という汎用パターン。
- Enhanced Hammerの使用判断：相手の防御的特殊エネルギー（Mist/Rock Fighting）の枚数と手持ちのEnhanced Hammer枚数を比較し、除去しきれる場合のみ使う／しきれないならターゲットを変える、という「中途半端な妨害を避ける」判断。
- ターン単位でリセットするワンショットアビリティのフラグ管理（`ability_used_dudunsparce`, `ability_used_fezandipiti`）はシンプルで実装しやすい。

## 注意点・疑問
- 「最高5位」の裏付けとなる定量データ（対戦数・勝率）はノートブックに含まれておらず、著者の申告のみ。競合が少なかった大会2日目時点の順位である点にも留意（メタが薄い時期のスナップショット）。
- `estimate_hand_increase`のカード別の増減値（例：Hilda +1, Dawn +2等）はカード効果の正確な理解に依存しており、値の正当性そのものは要検証。
- スコアの選択方式が「score降順ソートしてargmax（先頭のみ採用しmaxCountで打ち切り）」であり、Archaludonノートブックのような負スコアスキップ機構は無い（`sorted`結果の上位`maxCount`個をそのまま返す）。minCountとの整合性は保たれているように見えるが、負スコアの選択回避ロジックが無い点は他ノートより単純。
