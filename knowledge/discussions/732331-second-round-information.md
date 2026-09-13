# 732331 — Second Round Information（運営アナウンス・2026-08-02）

**★★★ プロジェクトの前提を変えるアナウンス。必読。**

## 3行サマリ

- **Second Round（東京・現地開催）へ進む8チームは「Strategy Category の結果」で決まる**。
  Simulation の順位は「考慮される」だけで、**選抜の主体は戦略レポート**
- Simulation の最終LBは 08-16 締切＋約2週間の追加対戦で確定し、**Strategy の結果はこれを変えない**（運営 shige が明言）
- Second Round は BO3・H100 GPU・30分/ゲーム・**カードプール拡張**。First Round と違うデッキ/コードを使ってよい

## 取れる情報

★★★ **選抜基準（運営原文）**:
> "While your ranking in the Simulation Category will be taken into consideration, submissions will be
> evaluated **holistically** based on additional factors, including **deck construction, the originality of
> the proposed approach, and the quality of the explanations provided in the report**."

★★★ **Strategy Category への参加が必須**。「Simulation から Second Round へ進むには、
**同じチームで Strategy Category にも参加**していなければならない」。評価基準ページ:
https://www.kaggle.com/competitions/pokemon-tcg-ai-battle-challenge-strategy/overview/evaluation

★★ **Second Round の環境**（First Round と別物）:
| 項目 | 内容 |
|---|---|
| 形式 | **BO3**。同じデッキ・同じコードを3試合通して使う。逐次実行 |
| 情報 | **Game 2 では Game 1 のログが読める**（Game 3 では1・2両方）＝試合内適応が可能 |
| 計算資源 | AWS p5.4xlarge 相当。**NVIDIA H100 (80GB VRAM)** / 256GiB RAM / 16 vCPU |
| 時間 | **30分/ゲーム**（ゲーム間で繰り越し不可。超過は敗北）。First Roundの600秒×3倍 |
| カード | First Round の全カード＋**追加カード**（内容は Simulation 最終順位確定後に発表） |

★ 縛りの確認（運営 shige の回答）: Second Round では First Round の提出と
**違うデッキ・違うコードを使ってよい**。追加の制約も設けない予定。

★ 参加者 greySnow の懸念「originality / quality のような主観的要素で決まるのは聞いていない」に対し、
運営は「Simulation の最終LBは Strategy の結果で変わらない」と回答するに留まり、
**選抜が主観評価を含むこと自体は否定していない**。

## 注意点・我々への含意

- **本プロジェクトは Simulation LB の最大化に100%を割いてきた**。CLAUDE.md には Strategy Category への
  参加意向が書かれているが「詳細要件（締切・フォーマット）は未調査」のまま。
  **このアナウンスにより、Strategy 側の締切と要件の確認が最優先事項になった**（要WebFetch/確認）
- 評価軸「deck construction / originality of approach / quality of explanations」は、
  **`knowledge/experiments/` の EXP レポート群がそのまま素材になる**（CLAUDE.mdの想定どおり）。
  特に EXP-102/106（測定装置の修正）や EXP-035形式のフル精読は originality の材料
- **Second Round に GPU が出る**＝そこで戦うなら重い探索/大きなNNが解禁される。ただし
  進出できなければ関係ないので、**今やるべきは Strategy レポートの質**であって Second Round 用の実装ではない
- 追加カードの内容は未発表。**Expanded（旧カード）や本大会専用のオリジナルカードは入れない**予定（運営 shige）
