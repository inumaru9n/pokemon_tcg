"""Kaggle T4 で **critic学習のみ** を行う（I-109 Phase RL Stage 1）。

自己対戦の生成はローカルのほうが約9倍速い（実測: ローカル63,000試合/時 vs
Kaggle 7,234試合/時）ため、critic事前学習の段階では
  生成=ローカル / 学習=Kaggle T4
に分割する。PPOに入ったら生成と更新が交互になるのでKaggle内完結に戻す。

**アクセラレータは必ずID名でT4以上を指定する**（`--accelerator NvidiaTeslaT4`）。
P100(sm_60)はKaggleイメージのPyTorch(sm_70+)と非互換で、CUDAが使えず
CPUフォールバックして12時間セッションでも終わらない（EXP-094の教訓を再度踏んだ）。

スクラッチと094暖機の2本を学習し、キルゲートを判定する。
"""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys

WORK = "/kaggle/working"
REPO = f"{WORK}/repo"


def sh(cmd: str, check: bool = False) -> bool:
    print(f"$ {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"失敗: {cmd}")
    return r.returncode == 0


def main() -> None:
    import torch
    print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "", flush=True)

    code = glob.glob("/kaggle/input/**/selfplay", recursive=True)
    agent = glob.glob("/kaggle/input/**/094_yushin_nn", recursive=True)
    shards = sorted(glob.glob("/kaggle/input/**/sp*_p*.npz", recursive=True))
    print(f"selfplay={code} agent={agent}")
    print(f"シャード {len(shards)}本", flush=True)
    assert code and agent and shards, "入力が揃っていない"

    os.makedirs(REPO, exist_ok=True)
    for src, dst in ((code[0], f"{REPO}/selfplay"), (agent[0], f"{REPO}/agents/094_yushin_nn")):
        if os.path.exists(dst):
            shutil.rmtree(dst)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copytree(src, dst)

    npz = " ".join(f'"{s}"' for s in shards)
    model = f"{REPO}/agents/094_yushin_nn/model_094.npz"
    for tag, extra in (("scratch", ""), ("warm", "--warm")):
        print("\n" + "=" * 60, flush=True)
        print(f"### critic: {tag}", flush=True)
        print("=" * 60, flush=True)
        sh(f"cd {REPO} && python selfplay/train_value.py --npz {npz} "
           f"--policy {model} {extra} --out {WORK}/value_{tag}.npz")


if __name__ == "__main__":
    sys.exit(main())
