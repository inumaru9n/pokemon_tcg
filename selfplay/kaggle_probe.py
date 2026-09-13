"""Kaggle Notebook上でRLループが成立するかを1回で確かめる調査スクリプト。

確認する3点:
  1. コンペ紐付けで **エンジンのC++ソース**（ptcgProgram 22）が見えるか
  2. そこから **CRNパッチ入り libcg.so** をビルドできるか（ローカルと同一エンジンにする）
  3. 4 vCPU で自己対戦が何試合/時 出るか（ローカルは8コア5ワーカーで63,000試合/時）

**なぜソースからビルドするか**: 配布の libcg.so はリポジトリのもの（07-04版）と
ハッシュが異なり、かつ BattleStartSeeded（CRN）を持たない。ソースからビルドすれば
07-15版エンジン + CRN の両方がKaggle側でも揃い、ローカルと同一条件になる。

このスクリプトはKaggle Notebookとして実行される（ローカルでは実行しない）。
"""

from __future__ import annotations

import glob
import os
import subprocess
import sys
import time

WORK = "/kaggle/working"
COMP = "/kaggle/input/pokemon-tcg-ai-battle"


def sh(cmd, **kw):
    print(f"$ {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, **kw)
    if r.stdout:
        print(r.stdout[-3000:])
    if r.returncode != 0:
        print("STDERR:", r.stderr[-3000:])
    return r.returncode == 0


def main():
    print("=" * 60)
    print("1. コンペ入力の探索")
    print("=" * 60)
    for root in sorted(glob.glob("/kaggle/input/*")):
        print(" ", root)
    src_dirs = glob.glob("/kaggle/input/**/ptcgProgram*", recursive=True)
    print("エンジンソース候補:", src_dirs)
    so_files = glob.glob("/kaggle/input/**/libcg*.so", recursive=True)
    print("配布libcg.so候補:", so_files)
    if not src_dirs:
        print("→ ソースが見えない。データセット同梱にフォールバックが必要")
    else:
        print(f"→ ソース発見: {src_dirs[0]}")
        print("  中身:", sorted(os.listdir(src_dirs[0]))[:8])

    print()
    print("=" * 60)
    print("2. CRNパッチ入り libcg.so のビルド")
    print("=" * 60)
    ds = glob.glob("/kaggle/input/**/Export_seeded.cpp", recursive=True)
    print("Export_seeded.cpp:", ds)
    ok = False
    if src_dirs and ds:
        os.makedirs(f"{WORK}/cg", exist_ok=True)
        sh("g++ --version | head -1")
        t0 = time.time()
        ok = sh(f'g++ -std=c++20 -O2 -shared -fPIC -I "{src_dirs[0]}" '
                f'-o {WORK}/cg/libcg.so "{ds[0]}"')
        print(f"ビルド {'成功' if ok else '失敗'} ({time.time()-t0:.0f}秒)")
        if ok:
            sh(f"ls -la {WORK}/cg/libcg.so")
            sh(f"nm -D {WORK}/cg/libcg.so | grep -c BattleStartSeeded")

    print()
    print("=" * 60)
    print("3. 自己対戦スループット（4 vCPU）")
    print("=" * 60)
    print("CPU数:", os.cpu_count())
    code_dirs = glob.glob("/kaggle/input/**/selfplay", recursive=True)
    agent_dirs = glob.glob("/kaggle/input/**/094_yushin_nn", recursive=True)
    # **必ず我々のデータセット側の cg/ を使う**。コンペ配布側の cg/sim.py には
    # ローカルで追加した HAS_SEEDED_START（CRNの能力検出）が無く、import で落ちる。
    cg_py = [p for p in glob.glob("/kaggle/input/**/cg/api.py", recursive=True)
             if "/datasets/" in p] or glob.glob("/kaggle/input/**/cg/api.py", recursive=True)
    print("selfplay:", code_dirs, "/ agent:", agent_dirs, "/ cg python:", cg_py)
    if not (ok and code_dirs and agent_dirs and cg_py):
        print("→ 必要物が揃っていないのでスループット計測はスキップ")
        return

    # 作業ディレクトリにレイアウトを再現する（cg/ にビルドした .so を置く）
    sh(f"mkdir -p {WORK}/repo && cp -r {os.path.dirname(cg_py[0])} {WORK}/repo/cg")
    sh(f"cp {WORK}/cg/libcg.so {WORK}/repo/cg/libcg.so")
    sh(f"cp -r {code_dirs[0]} {WORK}/repo/selfplay")
    sh(f"mkdir -p {WORK}/repo/agents && cp -r {agent_dirs[0]} {WORK}/repo/agents/")
    sh(f"mkdir -p {WORK}/repo/arena")
    arena = glob.glob("/kaggle/input/**/run_match.py", recursive=True)
    if arena:
        sh(f"cp {arena[0]} {WORK}/repo/arena/")
    os.chdir(f"{WORK}/repo")
    sys.path.insert(0, f"{WORK}/repo")
    t0 = time.time()
    ok2 = sh(f"cd {WORK}/repo && python selfplay/gen_selfplay.py --games 320 "
             f"--batch 32 -w {max(1, (os.cpu_count() or 2) - 1)} "
             f"--rows-per-shard 100000 --out {WORK}/probe.npz")
    print(f"生成 {'成功' if ok2 else '失敗'} ({time.time()-t0:.0f}秒)")
    sh(f"ls -la {WORK}/probe_p*.npz 2>/dev/null | head")


if __name__ == "__main__":
    main()
