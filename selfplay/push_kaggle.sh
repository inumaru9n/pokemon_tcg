#!/bin/bash
# EXP-094 Step3: 学習データをKaggleにアップロードし、GPU Notebookで学習を回す。
#
# 前提: data/nn/train.npz が生成済み（pack_dataset.py）
# 使い方:
#   bash selfplay/push_kaggle.sh dataset   # データセット作成/更新
#   bash selfplay/push_kaggle.sh kernel    # Notebook投入（GPU）
#   bash selfplay/push_kaggle.sh status    # 実行状況
#   bash selfplay/push_kaggle.sh output    # 学習済み重みを取得
set -e
cd "$(dirname "$0")/.."
USER=$(uv run python -c "import json,os;print(json.load(open(os.path.expanduser('~/.kaggle/kaggle.json')))['username'])")
DS_SLUG="ptcg-yushin-nn"
# kernel の id は title から導出される slug と一致させる必要がある（不一致だと409）
KN_TITLE="PTCG 094 LSTM policy all context imitation"
KN_SLUG="ptcg-094-lstm-policy-all-context-imitation"
DS_DIR="data/nn/kaggle_ds"
KN_DIR="data/nn/kaggle_kernel"

case "$1" in
dataset)
  mkdir -p "$DS_DIR"
  cp data/nn/train.npz "$DS_DIR/"
  cat > "$DS_DIR/dataset-metadata.json" <<EOF
{
  "title": "PTCG Yushin 156952a871 all-context decisions",
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
  # accelerator は **ID名**で指定する（docs/kernels.md）。UIの表示名ではない点に注意。
  # 有効値: NvidiaTeslaP100 / NvidiaTeslaT4 / NvidiaTeslaT4Highmem / NvidiaTeslaA100 /
  #        NvidiaL4 / NvidiaL4X1 / NvidiaH100 / NvidiaRtxPro6000 / TpuV38 / Tpu1VmV38 /
  #        TpuV5E8 / TpuV6E8
  # **P100(sm_60)はKaggleイメージのPyTorch(sm_70+)と非互換なのでT4以上を指定すること**。
  ACC="${2:-NvidiaTeslaT4}"
  mkdir -p "$KN_DIR"
  cp selfplay/train_nn_kaggle.py "$KN_DIR/ptcg-094-lstm-policy.py"
  # コンペを紐付けると /kaggle/input/pokemon-tcg-ai-battle/ 配下に
  # EN_Card_Data.csv・エンジンソース・sample_submission/cg（libcg.so=Linux版）が入る。
  # → RLフェーズではKaggle上で自己対戦の生成まで回せる。
  cat > "$KN_DIR/kernel-metadata.json" <<EOF
{
  "id": "$USER/$KN_SLUG",
  "title": "$KN_TITLE",
  "code_file": "ptcg-094-lstm-policy.py",
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
  mkdir -p data/nn/out
  uv run kaggle kernels output "$USER/$KN_SLUG" -p data/nn/out
  ls -la data/nn/out
  ;;
*)
  echo "usage: $0 {dataset|kernel|status|output}"; exit 1;;
esac
