# a-sample-archaludon-75-wr-vs-my-1300-starmie（tomatomato、75票）

## 3行サマリ
- Archaludon ex / Cinderace / Duraludon / Relicanth のMetalデッキ。Cinderaceの「Explosiveness」でセットアップ時に先出しActive固定→ターン1 Turbo Flareでベンチのデュラルドンにエネルギー加速→Archaludon ex進化でAssemble Alloyから3エネルギー到達という、ワンショット的な高速立ち上げが核。
- 相手デッキをActive/ベンチのカードIDから`detect_matchup`で判定し（crustle/hop/starmie/lucario/alakazam/generic）、マッチアップごとに攻撃選択・回復閾値・スタジアム採否をハードコードで上書きする`apply_overrides`層を持つ。
- 著者の1300+ Starmie/Froslass自作エージェント相手に1000戦で74.4%勝率と主張しているが、これは自分の特定ビルド（Froslassがメタ弱点）に対する自己対戦のみで、汎用性は未検証と著者自身が明記している。

## 盗めるアイデア
- ★ マッチアップ検出（相手の場のカードIDセットで敵デッキアーキタイプを推定）→専用スコア上書きレイヤーを重ねる設計パターン。汎用スコア関数の上にif文で例外処理する二層構成は読みやすく拡張しやすい。
- ★ ログから「相手が直前ターンに使った攻撃ID」を追跡する`_update_opp_attack_tracking`（TURN_ENDログを境に集計）。これにより「相手がMega Braveで手が止まっている（エネルギー要求増で動けない）」等の状態推定→Boss温存判断に使っている。
- ★ 相手の"最大与ダメージ推定"を明示的に見積もる（`opp_max_damage`）。特にAlakazam戦は相手の手札枚数×20ダメージの床/天井を、盤面のDudunsparce/Kadabra等のドロー要因から`_estimate_alakazam`で推定し、自分の回復判断（Jumbo Ice Cream使用可否）に直結させている。これは「相手の最大打点を先読みして自分の耐久ラインを決める」防御的読み合いの好例。
- Jumbo Ice Cream（回復）のHP閾値をマッチアップ別にグリッドサーチで最適化したと明記（Lucario戦: N=500候補から200-280を10刻みで探索、250-270が70%で最適）。実際にローカルでチューニングした数値という点で参考になる（値そのものは自環境non-portableだが、閾値グリッドサーチという手法は流用可）。
- スコア関数のディスパッチ構造（`_MAIN_DISPATCH`で OptionType→専用スコア関数）、および「スコアが負なら`minCount`を満たすまではスキップする」という選択ロジック（`choose_options`）は汎用的で読みやすい実装。
- prize_value（ex=2枚、megaEx=3枚)の考慮や、`Boss's Orders`使用判断で「相手アクティブをKOしても取り切れる残りサイド枚数か」を見て、取り切れないなら温存するロジック。

## 注意点・疑問
- 主張の74.4%勝率は自分自身の別エージェント相手の1000戦のみであり、多様な対戦相手プールでの検証ではない。汎用的な強さの指標としては要注意（著者本人も認めている）。
- Crustle対策で「Metal Defenderは0ダメージだから使わない」等の判定は、Crustleのアビリティ「Mysterious Rock Inn」の効果（ex/megaEx攻撃を無効化）に依存しており、実装がこの効果の正確な発動条件（ex限定か等）を正しく捉えているかは要検証。
- `_estimate_alakazam`のダメージ見積もり式（`_ALA_BOARD_GAIN`の係数など）はカード効果の正確な把握に依存しており、数値の正当性は未検証（要検証としてフラグ）。
