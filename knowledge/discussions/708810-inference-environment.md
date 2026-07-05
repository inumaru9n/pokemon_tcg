# Question on inference environment（35v、運営回答あり）

## 3行サマリ
- 運営公式回答: **1ゲーム合計600秒/チーム、ターン毎の加算なし**（我々のエピソードデータ解析と一致、公式確認が取れた）
- ハードウェア: **CPUのみ、1.6 vCPU、RAM 8GB**
- Pythonパッケージはnotebook環境と同一（kaggle-environmentsリポジトリのDockerfile参照）

## 取れる情報
- ★ 1.6vCPUなので並列探索は不可、シングルスレッド前提の時間予算設計が正しい
- ★ 8GB RAMが学習済みモデルのサイズ上限を規定（小型transformerなら余裕）

## 注意点・疑問
- なし（一次情報）
