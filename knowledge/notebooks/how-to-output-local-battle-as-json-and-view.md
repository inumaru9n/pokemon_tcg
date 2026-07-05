# how-to-output-local-battle-as-json-and-view（Kiyota、公式、166票）

## 3行サマリ

- 運営公式のnotebook。ローカルで対戦を実行してリプレイJSON（`vis.json`）を出力し、同梱の`visualizer.html`をブラウザで開いて閲覧する、という最小構成の手順を示す2種類の方法を提示している。
- 方法1: `kaggle_environments.make("cabt")` を使い `env.run(["random","random"])` した後 `env.steps[0][0]["visualize"]` をそのままJSON保存。
- 方法2: `cg.game` の低レベルAPI（`battle_start`, `battle_finish`, `battle_select`, `visualize_data`）を直接呼び、各ステップで`obs_log`/`action_log`を貯めておいて`visualize_data()`の出力とマージし`vis.json`として保存する。

## 盗めるアイデア

- ★ `cg.game.visualize_data()` の出力に、各ステップの`obs`と`action`（`[action_log[i], action_log[i]]`という両陣営分の形）を後付けで結合するのが、公式visualizer.htmlが期待するJSONフォーマットであることが分かる。これは我々の`cg/game.py`にも同名APIが存在するため、`arena/run_match.py`の対戦ログをこの形式に変換する薄いラッパーを書けば、公式`visualizer.html`（同notebookのコードセルに埋め込み）でそのまま自分たちの対戦を可視化できる可能性が高い。評価基盤の「敗因reason別集計」「方策間の行動一致率測定」を目視デバッグで補う際の即戦力ツールになりうる。
- `kaggle_environments.make("cabt")` 経由でも同じ`visualize`データが取れるため、cg低レベルAPIを直接使わずに済ませる簡易ルートとしても使える。

## 注意点・疑問

- 埋め込まれた`visualizer.html`はHTML/JSベタ書きで、対応しているアクション種別・カード効果表示の網羅性は未検証。自分たちのエージェントが使う全アクション種別を正しく描画できるかは実際に試すまで不明（要検証）。
- `obs_log`/`action_log`を手動で貯める実装が必要であり、`arena/run_match.py`側にログ収集フックを追加する改修が前提になる（基盤タスクとして親セッションで検討）。
