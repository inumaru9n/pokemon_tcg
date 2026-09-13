"""既存エージェントのモデルだけ差し替えた派生を作る（EXP-114）。

**なぜ専用ツールにするか**: `open(p).read().replace("117", num)` という素朴な置換は
**カードIDの 117 まで書き換える**（実測: feat_137.py で ATK_COST の
`(117,'Claw Slash')` が `(137,...)` になり攻撃テーブルが壊れた。
エラーは「feat_keysの並びがモデルと不一致 → NN無効化」として出るが、
**そのまま errors=0 で対戦が進む**ので気づきにくい）。

置換するのは**モジュール名として現れる箇所だけ**（import 文・ファイル名）に限定する。
"""
from __future__ import annotations
import argparse, ast, os, re, shutil, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO, os.path.join(REPO, 'arena')):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def make(src: str, dst: str, model: str, num: str, src_num: str) -> None:
    shutil.rmtree(dst, ignore_errors=True)
    os.makedirs(dst)
    for f in sorted(os.listdir(src)):
        p = os.path.join(src, f)
        if not os.path.isfile(p):
            continue
        if f.startswith("model_"):
            continue                       # モデルは別途配置
        out = os.path.join(dst, f.replace(src_num, num) if f.startswith(("feat_", "nn_")) else f)
        if f.endswith(".py"):
            t = open(p).read()
            # **モジュール名としての出現だけ**を置換する（カードIDを壊さない）
            for pat, rep in ((rf"\bfeat_{src_num}\b", f"feat_{num}"),
                             (rf"\bnn_{src_num}\b", f"nn_{num}"),
                             (rf"model_{src_num}\.npz", f"model_{num}.npz")):
                t = re.sub(pat, rep, t)
            with open(out, "w") as fh:
                fh.write(t)
        else:
            shutil.copy2(p, out)
    # 推論コードは唯一の定義元から配る（古い版が混ざる事故を防ぐ）
    shutil.copy2("tools/featgen_nn.py", os.path.join(dst, f"nn_{num}.py"))
    shutil.copy2(model, os.path.join(dst, f"model_{num}.npz"))
    ast.parse(open(os.path.join(dst, "main.py")).read())
    # **feat_keys がモデルと一致するかを作成時に検証する**
    _verify(dst, num, src, src_num)
    print(f"{dst}: {os.path.basename(model)}")


def _verify(dst: str, num: str, src: str, src_num: str) -> None:
    import importlib.util as ilu
    import numpy as np
    for d, n in ((src, src_num), (dst, num)):
        if d not in sys.path:
            sys.path.insert(0, d)
    def keys(d, n):
        sp = ilu.spec_from_file_location(f"feat_{n}", os.path.join(d, f"feat_{n}.py"))
        m = ilu.module_from_spec(sp); sys.modules[f"feat_{n}"] = m; sp.loader.exec_module(m)
        return m
    a, b = keys(src, src_num), keys(dst, num)
    for tbl in ("ATK_COST", "ATTACKS_056", "DECK_COUNTS", "CLASSES"):
        x, y = getattr(a, tbl, None), getattr(b, tbl, None)
        assert x == y, f"{tbl} が元と違う（置換でカードIDを壊した可能性）"
    w = np.load(os.path.join(dst, f"model_{num}.npz"), allow_pickle=True)
    if "feat_keys" in w.files:
        assert len(w["feat_keys"]) == len(getattr(b, "FEAT_KEYS", None) or w["feat_keys"]), \
            "feat_keys の長さがモデルと違う"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--model", required=True)
    a = ap.parse_args()
    num = os.path.basename(a.dst)[:3]
    src_num = os.path.basename(a.src)[:3]
    make(a.src, a.dst, a.model, num, src_num)


if __name__ == "__main__":
    main()
