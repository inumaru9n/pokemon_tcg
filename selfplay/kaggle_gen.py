"""Kaggle CPU Notebook で自己対戦を生成する（GPUクォータは学習用に温存）。

実測（統制した比較、EXP-095）:
  エンジン+JSON  ローカル 0.659 ms/決定 / Kaggle 0.405 ms/決定  ← Kaggleのほうが速い
  NN込み1ワーカー ローカル 12,653試合/時 / Kaggle 7,939試合/時
→ 1ワーカーあたり1.6倍差。**Kaggleも生成に十分使える**。

**`--tasks-per-child 0` が必須**。ワーカー再生成はmacOSではデッドロックし、
Linuxでも3.3倍の劣化を招く（Kaggleの初回測定2,411試合/時/ワーカーはこれが原因だった）。
"""
from __future__ import annotations
import glob, os, shutil, subprocess, sys, time

WORK = "/kaggle/working"; REPO = f"{WORK}/repo"; DATA = f"{WORK}/data"
GAMES = int(os.environ.get("RL_GAMES", "80000"))
SEED_BASE = int(os.environ.get("RL_SEED", "99001"))   # ローカルと重複させない

def sh(c, check=False):
    print(f"$ {c}", flush=True)
    r = subprocess.run(c, shell=True, text=True)
    if check and r.returncode: raise RuntimeError(c)
    return r.returncode == 0

def main():
    src = glob.glob("/kaggle/input/**/ptcgProgram*", recursive=True)[0]
    seeded = glob.glob("/kaggle/input/**/Export_seeded.cpp", recursive=True)[0]
    cg_api = [p for p in glob.glob("/kaggle/input/**/cg/api.py", recursive=True)
              if "/datasets/" in p][0]
    code = glob.glob("/kaggle/input/**/selfplay", recursive=True)[0]
    agent = glob.glob("/kaggle/input/**/094_yushin_nn", recursive=True)[0]
    os.makedirs(REPO, exist_ok=True); os.makedirs(DATA, exist_ok=True)
    if not os.path.exists(f"{REPO}/cg"): shutil.copytree(os.path.dirname(cg_api), f"{REPO}/cg")
    for s, d in ((code, f"{REPO}/selfplay"), (agent, f"{REPO}/agents/094_yushin_nn")):
        if os.path.exists(d): shutil.rmtree(d)
        os.makedirs(os.path.dirname(d), exist_ok=True); shutil.copytree(s, d)
    os.makedirs(f"{REPO}/arena", exist_ok=True)
    ar = glob.glob("/kaggle/input/**/run_match.py", recursive=True)
    if ar: shutil.copy(ar[0], f"{REPO}/arena/")
    for j in glob.glob(f"{REPO}/cg/libcg-arm64.so") + glob.glob(f"{REPO}/cg/*.dylib"): os.remove(j)
    t0 = time.time()
    sh(f'g++ -std=c++20 -O2 -shared -fPIC -I "{src}" -o {REPO}/cg/libcg.so "{seeded}"', True)
    print(f"エンジンビルド {time.time()-t0:.0f}秒 / CPU {os.cpu_count()}", flush=True)

    # 2万試合ずつプロセスを使い捨てる（libcgのリーク対策。ワーカー再生成は使わない）
    per = 20000
    done = 0; b = 0
    while done < GAMES:
        n = min(per, GAMES - done)
        # **check=True**。失敗を握り潰すと「完了」と嘘のログを出したまま終わる（実際に踏んだ）
        sh(f"cd {REPO} && python selfplay/gen_selfplay.py --games {n} --batch 32 "
           f"-w 3 --tasks-per-child 0 --rows-per-shard 400000 "
           f"--seed-base {SEED_BASE + b * 7919} --out {DATA}/kg{b:02d}.npz", check=True)
        done += n; b += 1
        sh(f"du -sh {DATA}")
    print(f"完了 {GAMES:,}試合 / {time.time()-t0:.0f}秒", flush=True)

if __name__ == "__main__":
    sys.exit(main())
