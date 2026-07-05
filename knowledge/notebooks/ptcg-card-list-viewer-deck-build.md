# ptcg-card-list-viewer-deck-build（Taimo、137票）

## 3行サマリ

- 競技用CSV/PDF（`EN_Card_Data.csv`等とカード画像PDF）から、ブラウザで閲覧・デッキ構築ができる単体HTML（`card_list.html` + `images/{card_id}.jpg`）を生成するツール。
- 生成スクリプトを`/kaggle/working/card_list_viewer/card_list.py`に書き出し、その中に`deck.csv`妥当性チェック用のバリデーションロジック（`VALIDATE_SOURCE`）を含んでいる。
- KaggleにはPopplerの`pdfimages`が無いため、PyMuPDF（`fitz`）でカード画像を抽出する、という環境制約への対処が明示されている。

## 盗めるアイデア

- ★ 埋め込まれている`deck.csv`妥当性チェックロジック（60枚、同名カード4枚まで、ACE SPEC 1枚まで等のルール検証）は、我々の`tools/package_submission.py`が行っている検証と重複が無いか確認し、抜けているチェック観点があれば取り込む価値がある（提出直前のデッキ検証精度を上げる）。
- カード一覧をブラウザ上でクリックしてデッキを組み立てるUIは、デッキ構築の試行錯誤を高速化する体験として参考になる（ただし我々は基本的にCSVを直接編集・スクリプト生成する運用なので優先度は低い）。

## 注意点・疑問

- 生成されるバリデーションロジックの正確性（公式ルールとの完全一致）は未検証。`tools/package_submission.py`の検証内容と比較のうえで採否を判断すべき（要検証）。
- 純粋なツールnotebookであり、方策・勝率・メタに関する主張は無い。
