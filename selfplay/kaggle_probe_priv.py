"""Kaggle **CPU** で仮説④の直接検証（probe_privileged.py）を回す。

GPUクォータを使い切ったので CPU 枠で実行する。前向き計算だけなので CPU で足りる。
"""

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
GAMES = int(os.environ.get("PP_GAMES", "40"))
RESAMPLE = int(os.environ.get("PP_RESAMPLE", "8"))


def main() -> None:
    import kaggle_league as kl
    kl.setup()
    ck = glob.glob("/kaggle/input/**/ckpt.pt", recursive=True)
    assert ck, "ckpt.pt が無い"
    cmd = (f"cd {REPO} && python selfplay/probe_privileged.py --ckpt {ck[0]} "
           f"--games {GAMES} --resample {RESAMPLE}")
    print(f"$ {cmd}", flush=True)
    if subprocess.run(cmd, shell=True, text=True).returncode != 0:
        raise RuntimeError("失敗")


if __name__ == "__main__":
    main()
