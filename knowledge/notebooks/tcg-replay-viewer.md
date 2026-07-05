# tcg-replay-viewer（Hum1060、53票）

## 3行サマリ

- `ipywidgets.FileUpload`でリプレイJSONをアップロードし、ターン単位で盤面を再生するシンプルなビューア（`battle_viewer.py`の`render_battle()`関数として実装）。
- カード画像は公式配布の`Card_ID List_JP.pdf`から`pymupdf`でページを切り出してローカルキャッシュ化し、表示に利用している。
- モバイル対応レイアウト・カード拡大表示など、UI面での作り込みはあるが、分析色（メタ・勝率統計）はほぼ無く、単体の可視化ツールという位置付け。

## 盗めるアイデア

- 公式カードPDF（`Card_ID List_JP.pdf`等）から`pymupdf`でページ単位にカード画像を切り出すテクニックは、他のカード画像系notebook（en-jp-deck-image-renderer, ptcg-card-list-viewer-deck-build）でも共通して使われている。Kaggle環境に`poppler`が無いため`pdfimages`ではなく`pymupdf (fitz)`を使う、という制約情報は基盤整備時に有用。
- `ipywidgets.FileUpload`でJSONをその場アップロードして描画するUIパターンは、対話的な小ツールをNotebook上で素早く作る際の定型として参考になる。

## 注意点・疑問

- 本ツール固有の技術的主張（勝率など）は無く、分析価値は低い。ツールとしての完成度・正確性も未検証。
