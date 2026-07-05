# ptcg-mega-lucario-ex-v62.ipynb

PyJa氏、42票。Kiyota版Mega Lucario exサンプルを`LucarioPolicy`クラスへOOPリファクタした上で、
「v62」というバージョン番号が示す通り**反復チューニングを重ねた形跡がある**、対面テック追加版。
`pokemon-ai-battle-agent-mega-lucario.ipynb`（Nur Srijan版）と同系のクラス構造だが、探索機能は無く、
コード自体は健全（後述のNameErrorバグは無い）に見える。

## 3行サマリ

- ベースロジックはKiyota版Mega Lucario exと基本同一だが、`LucarioPolicy`クラスに整理され、ダメージ計算（弱点・耐性・Crustle無効化）も一貫して正しく実装されている（`damage = base_damage`→弱点/耐性補正→Crustle特例、の順で欠落なく計算）。
- 3つの対面別ピンポイントテックを"gated"（該当カードが盤面にある時だけ発火）で追加: ①Crustle軸（Dwebble/Crustle）に対してHariyama温存、②Abomasnow/水軸に対しSnover早期KO優先度+950・Hero's Capeを大幅優先、③Alakazam軸（メガルカリオexの弱点である超タイプ）に対しAbra/Kadabraの早期KO優先度+400（「高すぎる値は良いKOを見逃すので中庸に調整した」とスイープ検証済みと明記）。
- 元のKiyota版に無かった「山札残数によるドロー/サーチ系トレーナーズの自重（`LOW_DECK_COUNT=10`、`_low_deck()`）」を追加しており、Dragapult/Iono版にあった自己デッキアウト防止をLucario系にも移植した形。

## 盗めるアイデア

- ★ **"gated" 対面テックパターン**: 相手ボードに特定カード（Snover/Kyogre/Mega Abomasnow ex、Abra/Kadabra/Alakazam、Dwebble/Crustle）が実際に存在する場合のみボーナススコアを発火させる設計。他の全マッチアップでは元のロジックと完全に同一に振る舞うため「副作用ゼロで特定の苦手対面だけ強化する」ことができる、安全な改善の入れ方として汎用性が高い。
- ★ ノートブック内に明記された「スイープでボーナス値をチューニングした（`_ABRA_BONUS = 400`は高すぎると良いKOを逃すことが分かった）」という記述は、**係数を勘で決め打ちせず複数値を実際に対戦させて比較検証した**という開発姿勢の証拠。本プロジェクトの`/experiment`ワークフロー（200戦評価）と方向性が一致しており、係数調整のたびに評価をやり直す価値を裏付ける傍証になる。
- Kiyota版に不足していた山札残量セーフガード（`_low_deck`）をMega Lucario exデッキに移植するのは、そのままBACKLOG候補になりうる（他のKiyota版4種のうちAbomasnow・Lucarioにはこの安全弁が無い）。
- Hero's Cape（ACE SPEC、耐久強化系と推測）を「相手が水デッキ（Abomasnow軸）と判定した時だけ」大幅優先度アップ（7000→12200/12800）するのは、限定的なリソース（ACE SPECは1枚のみ）を対面によって使い分ける良い実装例。

## 注意点・疑問

- 要検証: `_should_preserve_hariyama()`はCrustle軸相手かつ手札にHariyamaがあり場にMakuhitaがいる時、Carmine（ドロー効果と引き換えに手札を全て切るサポート想定）の使用を止める設計だが、これがHariyamaを守る効果と本当に対応しているかは`Carmine`の実カード効果（サポーターの詳細テキスト）と突き合わせないと確証が持てない。
- 勝率や対戦検証の主張は無し（自己検証は「60枚デッキを正しく返すか」の単体テストのみ）。
- Nur Srijan版で確認された`search_begin`のシグネチャ誤り・`damage`未定義バグは本ノートブックには見当たらない（探索機能自体を実装していないため）。
