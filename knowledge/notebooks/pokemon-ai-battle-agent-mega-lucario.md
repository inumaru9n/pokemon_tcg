# pokemon-ai-battle-agent-mega-lucario.ipynb

Nur Srijan氏、40票。Kiyota版Mega Lucario exサンプル（`japanese-language-ex.ipynb`と同じカードIDブラックリスト
`{144,322,323,337}`を採用しているため、恐らくそちら経由の派生）を`LucarioPolicy`クラスにOOPリファクタし、
さらに`search_begin`/`search_step`による前方探索（フォワードサーチ）を「オプション機能」として追加したと主張する意欲作。

## 3行サマリ

- 元のMega Lucario exヒューリスティックをクラス化（`LucarioPolicy`）し、Crustle（メガex系無効の壁ポケモン）への0ダメージ対応など小さな対面テックを追加。
- 目玉機能として「40ステップ先までの決定化フォワードサーチ（`search_begin`/`search_step`、1.5秒のタイムバジェット、候補6手ロールアウト）」を謳っているが、**下記の理由で実際には一度も実行されず常にフォールバックしている疑いが強い**。
- 例外は`try/except`で握りつぶして`_legal_fallback`（先頭からminCount個を機械的に選ぶだけ）に落とす設計のため、内部で例外が起きても対戦は継続する＝**見た目上は動くが中身が壊れている**タイプの障害が起きやすい構造。

## 盗めるアイデア

- 例外セーフティネット全体を`try/except`＋`_legal_fallback`で包む設計思想自体は良い（対戦中に例外でエージェントが即負けする事故を防ぐ）。ただし本ノートブックのように「常にフォールバックに落ちているのに気づけない」リスクとセットである点は要注意（後述）。
- `LucarioPolicy`へのクラス化（状態を`__init__`で一括収集し、`choose()`で採点）はコードの見通しを良くする一般的にプラスなリファクタ。`ptcg-mega-lucario-ex-v62.ipynb`（PyJa版）も同型のクラス構造を採用しており、収束した「良い書き方」と言えそう。

## 注意点・疑問

- **要検証・重大な疑義（コード上の根拠あり）**: `search_plan()`内の`search_begin(sbi)`という呼び出しは、本リポジトリの`cg/api.py`（`search_begin(agent_observation, your_deck, your_prize, opponent_deck, opponent_prize, opponent_hand, opponent_active, manual_coin=False)`）のシグネチャと一致しない。必須引数5個（`your_deck`等）が渡されておらず、`sbi`（`search_begin_input`という文字列）を`agent_observation`位置に渡している。これは呼び出す度に`TypeError`になり、`except Exception: return None`で握りつぶされて**探索は常に不発、必ず`LucarioPolicy(obs).choose()`のみが動く**と考えられる。
- **要検証・追加のバグ疑い**: `LucarioPolicy._plan_attack()`内で、Crustle対面以外の相手ポケモンに対しては`damage`変数が定義されないまま`op_pokemon.hp <= damage`で参照されており（Crustle分岐でのみ`damage = 0`を代入、通常の弱点/耐性計算による`damage`代入コードが無い）、**Crustle以外への攻撃プランニングは`NameError`で毎回失敗する**と読める。この例外は`agent()`トップレベルの`try/except`で捕捉され`_legal_fallback(select)`に落ちるため、MAINコンテキストの意思決定は（探索が不発なのと合わせ）ターン2以降ほぼ全て「先頭からminCount個を機械的に選ぶ」という、ヒューリスティックの体を成していない挙動になっている可能性が高い。
- 上記2点はどちらも「エラー無く完走する（=対戦は成立する）が、狙った知的挙動をしていない」タイプの不具合であり、勝率検証（200戦対戦）だけでは気づきにくい。実装を真似る際は`agent()`の返り値を単体テストでロギングし、`_legal_fallback`に落ちていないか・`search_plan`が`None`以外を返しているかを直接確認することを推奨。
- このノートブック自体に勝率の主張・検証結果の記載はない。
