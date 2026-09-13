"""Kaggle Notebook で「自己対戦生成 → Critic事前学習 → キルゲート判定」を完結させる。

I-109 Phase RL の Stage 0/1 を1セッションで回す。GPU Notebook（T4）を前提とし、
生成はCPU（4 vCPU）、学習はGPUを使う。

構成:
  A. コンペ紐付けのソースから **CRNパッチ入り libcg.so をビルド**（ローカルと同一エンジン）
  B. 自己対戦を生成してシャードを /kaggle/working/data に書き出す
  C. LSTM価値関数を **スクラッチ / 094からの暖機** の2本学習し、キルゲートを判定
  D. 重みと要約を /kaggle/working に残す（次セッションから再開できる）

**なぜKaggleで完結させるか**: RLループは生成と更新が交互に来るので、ローカル生成と
Kaggle学習に分けると往復のたびに数十分を失う。1セッション内なら重みはメモリ上で回る。

**落とし穴（実測で踏んだもの）**:
  - `cg/` のPythonラッパは必ず自前データセット側を使う。コンペ配布側の `cg/sim.py` には
    `HAS_SEEDED_START`（CRNの能力検出）が無く ImportError になる
  - 配布 `libcg.so` は07-04版でハッシュが違い CRN も無い → 必ずソースからビルドする
"""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys
import time

WORK = "/kaggle/working"
REPO = f"{WORK}/repo"
DATA = f"{WORK}/data"

# ---- 実行量（セッション12時間に収まるよう調整する） ----
GAMES = int(os.environ.get("RL_GAMES", "20000"))
GEN_WORKERS = max(1, (os.cpu_count() or 4) - 1)
ROWS_PER_SHARD = 400_000


def sh(cmd: str, check: bool = True) -> bool:
    print(f"$ {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"失敗: {cmd}")
    return r.returncode == 0


def setup() -> None:
    """エンジンをビルドし、実行用のディレクトリ構成を作る。"""
    src = glob.glob("/kaggle/input/**/ptcgProgram*", recursive=True)
    assert src, "コンペのエンジンソースが見つからない（competition_sourcesの紐付けを確認）"
    # cg のPythonラッパは **自前データセット側**（HAS_SEEDED_STARTを持つ）
    cg_api = [p for p in glob.glob("/kaggle/input/**/cg/api.py", recursive=True)
              if "/datasets/" in p]
    assert cg_api, "自前データセットの cg/ が見つからない"
    seeded = glob.glob("/kaggle/input/**/Export_seeded.cpp", recursive=True)
    assert seeded, "Export_seeded.cpp が見つからない"
    code = glob.glob("/kaggle/input/**/selfplay", recursive=True)[0]
    agent = glob.glob("/kaggle/input/**/094_yushin_nn", recursive=True)[0]

    os.makedirs(REPO, exist_ok=True)
    os.makedirs(DATA, exist_ok=True)
    if not os.path.exists(f"{REPO}/cg"):
        shutil.copytree(os.path.dirname(cg_api[0]), f"{REPO}/cg")
    for d, dst in ((code, f"{REPO}/selfplay"), (agent, f"{REPO}/agents/094_yushin_nn")):
        if os.path.exists(dst):
            shutil.rmtree(dst)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copytree(d, dst)
    ar = glob.glob("/kaggle/input/**/run_match.py", recursive=True)
    os.makedirs(f"{REPO}/arena", exist_ok=True)
    if ar:
        shutil.copy(ar[0], f"{REPO}/arena/run_match.py")

    so = f"{REPO}/cg/libcg.so"
    if not os.path.exists(so) or os.path.getsize(so) < 100:
        t0 = time.time()
        sh(f'g++ -std=c++20 -O2 -shared -fPIC -I "{src[0]}" -o {so} "{seeded[0]}"')
        print(f"エンジンビルド完了 {time.time()-t0:.0f}秒", flush=True)
    sh(f"nm -D {so} | grep -c BattleStartSeeded", check=False)
    # arm64版が残っていると環境によってはそちらを掴むので消す
    for junk in glob.glob(f"{REPO}/cg/libcg-arm64.so") + glob.glob(f"{REPO}/cg/*.dylib"):
        os.remove(junk)


def generate() -> list[str]:
    """自己対戦を生成。既にシャードがあれば追加ぶんだけ作る（セッション跨ぎの再開）。"""
    have = sorted(glob.glob(f"{DATA}/sp_seq_p*.npz"))
    if have:
        print(f"既存シャード {len(have)}本 を検出。生成をスキップ", flush=True)
        return have
    t0 = time.time()
    sh(f"cd {REPO} && python selfplay/gen_selfplay.py --games {GAMES} --batch 32 "
       f"-w {GEN_WORKERS} --rows-per-shard {ROWS_PER_SHARD} "
       f"--seed-base 20260727 --out {DATA}/sp_seq.npz")
    shards = sorted(glob.glob(f"{DATA}/sp_seq_p*.npz"))
    print(f"生成完了 {time.time()-t0:.0f}秒 / シャード {len(shards)}本", flush=True)
    return shards


def train(shards: list[str]) -> None:
    """スクラッチと暖機の2本を学習し、キルゲートを判定する。"""
    npz = " ".join(f'"{s}"' for s in shards)
    model = f"{REPO}/agents/094_yushin_nn/model_094.npz"
    for tag, extra in (("scratch", ""), ("warm", "--warm")):
        print("\n" + "=" * 60, flush=True)
        print(f"### critic: {tag}", flush=True)
        print("=" * 60, flush=True)
        sh(f"cd {REPO} && python selfplay/train_value.py --npz {npz} "
           f"--policy {model} {extra} --out {WORK}/value_{tag}.npz", check=False)


def main() -> None:
    print(f"CPU {os.cpu_count()} / 目標 {GAMES:,}試合 / 生成ワーカー {GEN_WORKERS}",
          flush=True)
    setup()
    shards = generate()
    if not shards:
        print("シャードが無いので学習をスキップ")
        return
    train(shards)
    print("\n完了。/kaggle/working に value_scratch.npz / value_warm.npz と "
          "data/sp_seq_p*.npz を出力", flush=True)


if __name__ == "__main__":
    sys.exit(main())
