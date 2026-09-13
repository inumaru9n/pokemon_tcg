"""Kaggle で Critic の予測性能を測る。

**特権あり / 観測のみ の両方を同じデータで測る**。
提出物では `GetHiddenData` が使えないので、**推論時探索に載せられるのは「観測のみ」の方**。
その AUC が探索の質を直接決める（EXP-053で探索が死んだ主因は評価関数の弱さだった）。
"""

from __future__ import annotations

import glob
import os
import subprocess
import sys

# **Kaggleではこのスクリプトだけが作業ディレクトリにコピーされる**ので、
# 同梱モジュール（kaggle_league.py）はデータセット側から探す必要がある。
for _d in glob.glob("/kaggle/input/**/selfplay", recursive=True):
    if _d not in sys.path:
        sys.path.insert(0, _d)

WORK = "/kaggle/working"
REPO = f"{WORK}/repo"
GAMES = int(os.environ.get("EC_GAMES", "300"))


def main() -> None:
    import kaggle_league as kl
    kl.setup()
    ck = glob.glob("/kaggle/input/**/ckpt.pt", recursive=True)
    assert ck, "ckpt.pt が見つからない"
    for tag, extra in (("特権あり", ""), ("観測のみ（提出物で使える条件）", "--no-privileged")):
        print("\n" + "=" * 60, flush=True)
        print(f"### {tag}", flush=True)
        print("=" * 60, flush=True)
        cmd = (f"cd {REPO} && python selfplay/eval_critic.py --ckpt {ck[0]} "
               f"--games {GAMES} {extra}")
        print(f"$ {cmd}", flush=True)
        if subprocess.run(cmd, shell=True, text=True).returncode != 0:
            raise RuntimeError(f"失敗: {tag}")


if __name__ == "__main__":
    main()
