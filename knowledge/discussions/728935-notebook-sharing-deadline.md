# 728935 — 公開Notebook共有の締切: 8月2日（運営アナウンス・2026-07-24）

## 3行サマリ

- **Kaggle上でのNotebook公開は 2026-08-02 23:59 UTC で締切**。通常より1週間早い。**既に経過している**
- 作業やNotebook作成は自由。**「Kaggleで公開すること」だけが止まる**
- Strategy Category の提出に添付したNotebookは、**そちらの締切時に公開される**

## 取れる情報

★★ **`/notebooks` の収穫はこれ以上増えない**。8/2以降に新規公開されるものは無く、
`data/notebooks/` は実質的に確定した資産。ただし**既存Notebookの更新が可能かは運営未回答**
（c-number の質問「Will contestants still be able to update publicized notebooks during that period?」に返答なし）

★ 運営 Addison Howard: **今回の大会で Bo1 → Bo3 への変更はしない**。
ただし**締切後は対戦頻度を上げる予定**（＝最終2週間の追加対戦でLBは今より速く収束する）

★ 参加者 razilyrazi: メタ分析用にNotebook提出ができなくなったため、
運営に**エピソードデータセットの追加提供**を要望中（709160 形式）。
参照Notebook: `myso1987/ptcg-ai-battle-leaderboard-deck-meta-by-score-band`
（我々が `knowledge/notebooks/` に設計図として持っているもの）

## 注意点・我々への含意

- **我々は自前で `tools/get_episodes.py` による日次取得を運用しているので影響を受けない**
- **Strategy Category に添付するNotebookは締切後に公開される**＝レポートの一部として
  再現可能なNotebookを用意する価値がある（732331 の評価軸「quality of the explanations」に直結）
- LB分散について Mikael Kerimov「対戦頻度を上げても±25ptの揺れの頻度が増えるだけ」との反論あり。
  **CLAUDE.md の「単発のLBスコアで優劣を判断しない」方針は維持**
