"""データ量とcriticのAUCの関係を測る（Stage 1の締め）。

**目的はVの改善ではない**。PPOの各イテレーションに何試合必要かを知るための測定。
PPOではVが現在の方策を追随する必要があり、この事前学習Vはあくまで初期値なので、
ここで精度を追う価値は薄い。一方「何試合あればVが立つか」はPPOループで繰り返し
発生する判断なので、いま一度だけ測っておく。

10k / 20k / 40k の3点で学習し、**1倍化あたりのAUC改善**を出す:
  改善 < 0.01 → 2万試合規模で足りる。PPOのイテレーション予算をそこで決める
  改善が大きい → Vはデータを要求する。PPOの生成量を増やす判断材料にする

暖機は効果ゼロと確定済み（EXP-095）なのでスクラッチのみ。
"""
from __future__ import annotations
import glob, os, re, shutil, subprocess, sys

W = "/kaggle/working"; R = f"{W}/repo"

def sh(c, check=False):
    print(f"$ {c}", flush=True)
    r = subprocess.run(c, shell=True, text=True, capture_output=True)
    if r.stdout: print(r.stdout[-6000:], flush=True)
    if r.returncode and r.stderr: print("STDERR:", r.stderr[-2000:], flush=True)
    if check and r.returncode: raise RuntimeError(c)
    return r.stdout or ""

def main():
    code = glob.glob("/kaggle/input/**/selfplay", recursive=True)[0]
    agent = glob.glob("/kaggle/input/**/094_yushin_nn", recursive=True)[0]
    shards = sorted(glob.glob("/kaggle/input/**/sp*_p*.npz", recursive=True))
    print(f"シャード {len(shards)}本", flush=True)
    assert shards
    os.makedirs(R, exist_ok=True)
    for s, d in ((code, f"{R}/selfplay"), (agent, f"{R}/agents/094_yushin_nn")):
        if os.path.exists(d): shutil.rmtree(d)
        os.makedirs(os.path.dirname(d), exist_ok=True); shutil.copytree(s, d)
    npz = " ".join(f'"{x}"' for x in shards)
    model = f"{R}/agents/094_yushin_nn/model_094.npz"
    res = []
    for n in (10000, 20000, 37000):
        print("\n" + "=" * 60, flush=True)
        print(f"### 学習ゲーム={n:,}（valは全データ量で共通）", flush=True)
        print("=" * 60, flush=True)
        out = sh(f"cd {R} && python selfplay/train_value.py --npz {npz} "
                 f"--policy {model} --limit-games {n} --out {W}/value_{n}.npz")
        m = re.search(r"ターン5以降の平均AUC: ([0-9.]+)", out)
        o = re.search(r"全体AUC: ([0-9.]+)", out)
        res.append((n, float(m.group(1)) if m else float("nan"),
                    float(o.group(1)) if o else float("nan")))
    print("\n=== データ量スケーリング曲線 ===")
    print(f"{'学習games':>10}{'t5+AUC':>10}{'全体AUC':>10}{'1倍化あたり改善':>16}")
    for i, (n, a, o) in enumerate(res):
        d = f"{a - res[i-1][1]:+.4f}" if i else "—"
        print(f"{n:>8,}{a:>10.4f}{o:>10.4f}{d:>16}")
    if len(res) >= 2:
        last = res[-1][1] - res[-2][1]
        print(f"\n最後の1倍化での改善 {last:+.4f} → "
              f"{'飽和（2万試合規模で足りる）' if last < 0.01 else 'まだ伸びる（生成量を増やす価値あり）'}")

if __name__ == "__main__":
    sys.exit(main())
