#!/bin/bash
# SL（模倣）学習をKaggle GPUで回す汎用パイプライン。
#
# push_kaggle.sh は 094 専用にslugを焼き込んでいたが、プールのSLレプリカを枠ごとに
# 作るのでタグで切り替えられる形にした。データセットとカーネルのslugは
#   ptcg-sl-<tag> / ptcg-sl-<tag>-train
# になる（Kaggleのkernel slugはtitle由来なのでtitleも合わせている）。
#
# 使い方:
#   bash selfplay/push_sl.sh dataset gen092 /path/to/train.npz
#   bash selfplay/push_sl.sh kernel  gen092            # GPUはT4を既定（P100は非互換）
#   bash selfplay/push_sl.sh status  gen092
#   bash selfplay/push_sl.sh output  gen092 data/nn/out_gen092
set -e
cd "$(dirname "$0")/.."
CMD="$1"; TAG="$2"
[ -z "$TAG" ] && { echo "usage: $0 {dataset|kernel|status|output} <tag> [arg]"; exit 1; }
USER=$(uv run python -c "import json,os;print(json.load(open(os.path.expanduser('~/.kaggle/kaggle.json')))['username'])")
DS_SLUG="ptcg-sl-$TAG"
KN_TITLE="ptcg sl $TAG train"
KN_SLUG="ptcg-sl-$TAG-train"
DS_DIR="data/nn/push/$TAG/ds"
KN_DIR="data/nn/push/$TAG/kn"

case "$CMD" in
dataset)
  NPZ="$3"
  [ -f "$NPZ" ] || { echo "npzが無い: $NPZ"; exit 1; }
  mkdir -p "$DS_DIR"
  cp "$NPZ" "$DS_DIR/train.npz"
  # **継続学習の初期値**（--init-from）も同梱する。Kaggle側はファイル名で再帰検索する。
  for extra in "${@:4}"; do cp "$extra" "$DS_DIR/"; done
  cat > "$DS_DIR/dataset-metadata.json" <<EOF
{
  "title": "PTCG SL decisions $TAG",
  "id": "$USER/$DS_SLUG",
  "licenses": [{"name": "CC0-1.0"}]
}
EOF
  if uv run kaggle datasets status "$USER/$DS_SLUG" >/dev/null 2>&1; then
    uv run kaggle datasets version -p "$DS_DIR" -m "update $(date +%Y%m%d_%H%M)" --dir-mode zip
  else
    uv run kaggle datasets create -p "$DS_DIR" --dir-mode zip
  fi
  ;;
kernel)
  # accelerator は **ID名**で指定する。**P100(sm_60)はKaggleイメージのPyTorch(sm_70+)と
  # 非互換でLSTMのcudnnカーネルが無い**ので必ずT4以上にする（EXP-094で実測）。
  ACC="${3:-NvidiaTeslaT4}"
  mkdir -p "$KN_DIR"
  # **投入前に compile で検査する**。ast.parse はスコープ検査をしないので
  # `global` の重複宣言などを見逃し、Kaggle 側で ERROR になる（実際に踏んだ）。
  uv run python -c "compile(open('selfplay/train_nn_kaggle.py').read(),'t','exec')" \
    || { echo "train_nn_kaggle.py が compile できない。投入を中止"; exit 1; }
  cp selfplay/train_nn_kaggle.py "$KN_DIR/$KN_SLUG.py"
  cat > "$KN_DIR/kernel-metadata.json" <<EOF
{
  "id": "$USER/$KN_SLUG",
  "title": "$KN_TITLE",
  "code_file": "$KN_SLUG.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": false,
  "dataset_sources": ["$USER/$DS_SLUG"],
  "competition_sources": ["pokemon-tcg-ai-battle"],
  "kernel_sources": []
}
EOF
  uv run kaggle kernels push -p "$KN_DIR" --accelerator "$ACC"
  ;;
status)
  uv run kaggle kernels status "$USER/$KN_SLUG"
  ;;
output)
  OUT="${3:-data/nn/push/$TAG/out}"
  mkdir -p "$OUT"
  uv run kaggle kernels output "$USER/$KN_SLUG" -p "$OUT"
  ls -la "$OUT"
  ;;
*)
  echo "usage: $0 {dataset|kernel|status|output} <tag> [arg]"; exit 1;;
esac
