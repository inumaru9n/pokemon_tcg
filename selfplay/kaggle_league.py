"""Kaggle Notebook でリーグ学習（I-109 Stage 3）を回す。

構成は `kaggle_rl.py` と同じ:
  A. コンペ紐付けのソースから **特権APIつき libcg.so をビルド**（ローカルと同一エンジン）
  B. `selfplay/league.py` を実行

**落とし穴（kaggle_rl.py で実測済み。同じ轍を踏まない）**:
  - `cg/` のPythonラッパは必ず自前データセット側を使う。コンペ配布側には
    `HAS_SEEDED_START` / `HAS_HIDDEN_DATA` が無い
  - 配布 `libcg.so` は旧版でハッシュが違い CRN も特権APIも無い → 必ずソースからビルド
  - arm64版のdylibが残っていると環境によってはそちらを掴むので消す

**このスクリプト固有の確認事項**: ビルドした .so が `GetHiddenData` を持つこと。
無いと Critic の特権ブロックが**無言でゼロになる**（例外は出ない）。
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

# **main は 117**（現行ベスト、デッキ改良版）。相手は忠実度ゲート合格の3枠のみ。
AGENT_DIRS = ("117_alakazam_dspline", "101_marnie_luca",
              "131_ogerpon_majkel", "098_spidops_gen")
# 本走の設定（安全側）: 2,000試合×30イテレーション ≈ 6時間（セッション上限12時間）
#  - 最初の10回は **Criticウォームアップ**（2万試合＝EXP-095で実測したVの飽和点）。
#    Vが無情報だとadvantageが実質±1のコインフリップになり方策勾配の信号が消えるため。
#  - 途中経過を多く見たいので eval は3イテレーションごと。
#  - **ckpt.pt を毎回保存する**ので、飽和していなければ --resume で継ぎ足せる。
# E1: **相手を101のみに固定**した最小構成。判定指標を「vs 101 の argmax 勝率」1本に絞る。
# 混合相手だと改善が相手ごとに分かれて埋もれるため、まずここで
# 「機構がそもそも方策を改善できるのか」を確定させる。
ITERS = int(os.environ.get("LG_ITERS", "30"))
GAMES = int(os.environ.get("LG_GAMES", "2000"))
EVAL_GAMES = int(os.environ.get("LG_EVAL_GAMES", "600"))
# **main=117 / 相手=忠実度ゲート合格の3枠**（メタの58.2%をカバー）。
# 判定は「凍結117との argmax 直接対戦」＝プール（甘さ+11.1pt）に依存しない。
EXTRA = os.environ.get(
    "LG_EXTRA",
    "--main-dir agents/117_alakazam_dspline --main-feat feat_117 "
    "--main-model model_117.npz --eval-vs sl --eval-every 3 --snap-every 3 "
    + os.environ.get("LG_MODE",
                     # **既定は criticなし**（CRN群相対）。ウォームアップ2万試合が不要で、
                     # 配札分散を全ターンで厳密に除去する。criticの序盤AUCは0.599しかない。
                     "--no-critic --group 4"))
os.environ.setdefault("LG_CONDS", "")


def sh(cmd: str, check: bool = True) -> bool:
    print(f"$ {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"失敗: {cmd}")
    return r.returncode == 0


def setup() -> None:
    src = glob.glob("/kaggle/input/**/ptcgProgram*", recursive=True)
    assert src, "コンペのエンジンソースが無い（competition_sourcesの紐付けを確認）"
    # **パスでなく中身で選ぶ**。Kaggleのマウント経路はランによって
    # /kaggle/input/datasets/<user>/<slug>/... と /kaggle/input/<slug>/... の
    # 両方があり、"/datasets/" を含むかで判定すると自前データセットを取り逃す
    # （実際に踏んだ）。コンペ配布側の cg には HAS_SEEDED_START が無いので、
    # 同じディレクトリの sim.py がそれを持つかで自前版を特定する。
    cg_api = []
    for p_ in glob.glob("/kaggle/input/**/cg/api.py", recursive=True):
        sim = os.path.join(os.path.dirname(p_), "sim.py")
        try:
            if os.path.exists(sim) and "HAS_SEEDED_START" in open(sim).read():
                cg_api.append(p_)
        except OSError:
            pass
    assert cg_api, ("自前データセットの cg/ が無い（HAS_SEEDED_START を持つ sim.py が "
                    "見つからない）")
    seeded = glob.glob("/kaggle/input/**/Export_seeded.cpp", recursive=True)
    assert seeded, "Export_seeded.cpp が無い"
    code = glob.glob("/kaggle/input/**/selfplay", recursive=True)[0]

    os.makedirs(REPO, exist_ok=True)
    if not os.path.exists(f"{REPO}/cg"):
        shutil.copytree(os.path.dirname(cg_api[0]), f"{REPO}/cg")
    if os.path.exists(f"{REPO}/selfplay"):
        shutil.rmtree(f"{REPO}/selfplay")
    shutil.copytree(code, f"{REPO}/selfplay")
    for name in AGENT_DIRS:
        hit = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
        assert hit, f"エージェント {name} が無い"
        dst = f"{REPO}/agents/{name}"
        if os.path.exists(dst):
            shutil.rmtree(dst)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copytree(hit[0], dst)
    ar = glob.glob("/kaggle/input/**/run_match.py", recursive=True)
    os.makedirs(f"{REPO}/arena", exist_ok=True)
    if ar:
        shutil.copy(ar[0], f"{REPO}/arena/run_match.py")

    so = f"{REPO}/cg/libcg.so"
    if not os.path.exists(so) or os.path.getsize(so) < 100:
        t0 = time.time()
        sh(f'g++ -std=c++20 -O2 -shared -fPIC -I "{src[0]}" -o {so} "{seeded[0]}"')
        print(f"エンジンビルド完了 {time.time()-t0:.0f}秒", flush=True)
    for junk in glob.glob(f"{REPO}/cg/*.dylib") + glob.glob(f"{REPO}/cg/libcg-arm64.so"):
        os.remove(junk)

    # **特権APIが載っているかを必ず確認する**。無いとCriticの特権ブロックが
    # 無言でゼロになり、「動くが情報が入っていない」状態を見逃す。
    out = subprocess.run(f"nm -D {so} | grep -c -E 'GetHiddenData|BattleStartSeeded'",
                         shell=True, text=True, capture_output=True).stdout.strip()
    print(f"エンジンの追加API数（BattleStartSeeded + GetHiddenData）= {out}", flush=True)
    if out != "2":
        raise RuntimeError("追加APIが揃っていない。Export_seeded.cpp が古い可能性")


def main() -> None:
    print(f"CPU {os.cpu_count()}", flush=True)
    setup()
    sys.path.insert(0, REPO)
    # 条件スイープ: 学習済みCriticを使い回し、方策はSL初期から**完全に同一条件**で走らせる。
    # LG_CONDS は "名前=追加引数" を ';' 区切りで並べる。
    conds = os.environ.get("LG_CONDS", "").strip()
    if conds:
        ck = glob.glob("/kaggle/input/**/ckpt.pt", recursive=True)
        assert ck, "ckpt.pt が無い（Criticの使い回しに必要）"
        for c in conds.split(";"):
            name, _, extra = c.partition("=")
            name, extra = name.strip(), extra.strip()
            print("\n" + "=" * 60, flush=True)
            print(f"### 条件 {name}: {extra}", flush=True)
            print("=" * 60, flush=True)
            sh(f"cd {REPO} && python selfplay/league.py --iters {ITERS} "
               f"--games {GAMES} --eval-games {EVAL_GAMES} "
               f"--critic-warmup 0 --init-critic {ck[0]} "
               f"--out {WORK}/league_{name} {EXTRA} {extra}")
        return
    ck = glob.glob("/kaggle/input/**/ckpt.pt", recursive=True)
    init = f"--init-critic {ck[0]}" if ck else ""
    cmd = (f"cd {REPO} && python selfplay/league.py --iters {ITERS} "
           f"--games {GAMES} --eval-games {EVAL_GAMES} {init} "
           f"--out {WORK}/league {EXTRA}")
    sh(cmd)
    for f in sorted(glob.glob(f"{WORK}/league/*.npz")):
        print(f"  成果物 {f} ({os.path.getsize(f)/1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
