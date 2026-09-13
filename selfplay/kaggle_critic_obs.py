"""Kaggle で観測のみVをスクラッチ学習する（価値誘導探索の土台）。"""

from __future__ import annotations

import glob
import os
import subprocess
import sys

for _d in glob.glob("/kaggle/input/**/selfplay", recursive=True):
    if _d not in sys.path:
        sys.path.insert(0, _d)

WORK = "/kaggle/working"
REPO = f"{WORK}/repo"
GAMES = int(os.environ.get("CO_GAMES", "20000"))
EPOCHS = int(os.environ.get("CO_EPOCHS", "6"))


def main() -> None:
    import kaggle_league as kl
    kl.setup()
    cmd = (f"cd {REPO} && python selfplay/train_critic_obs.py "
           f"--games {GAMES} --epochs {EPOCHS} --out {WORK}/value_obs.npz")
    print(f"$ {cmd}", flush=True)
    if subprocess.run(cmd, shell=True, text=True).returncode != 0:
        raise RuntimeError("失敗")


if __name__ == "__main__":
    main()
