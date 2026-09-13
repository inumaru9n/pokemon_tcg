#!/bin/bash
# リーグ学習（I-109 Stage 3）を Kaggle GPU で回す。
#
# SL学習（push_sl.sh）と違い **npz1本では済まない**。RLはオンポリシー生成が要るので
# エンジン・エージェント一式・selfplayコードを丸ごと送る:
#   cg/（HAS_SEEDED_START を持つ自前ラッパ）/ Export_seeded.cpp（特権API付きソース）
#   agents/117,101,131,098 / selfplay/ / arena/run_match.py
# Notebook 側（kaggle_league.py）がソースからエンジンをビルドして league.py を実行する。
#
#   bash selfplay/push_rl.sh dataset
#   bash selfplay/push_rl.sh kernel
#   bash selfplay/push_rl.sh status | output
set -e
cd "$(dirname "$0")/.."
CMD="$1"
USER=$(uv run python -c "import json,os;print(json.load(open(os.path.expanduser('~/.kaggle/kaggle.json')))['username'])")
# 条件ごとに別カーネルにする（Kaggleは同時2本まで走らせられる）。
#   bash selfplay/push_rl.sh kernel nocritic   → ptcg-rl-nocritic-train
TAG="${2:-league}"
DS_SLUG="ptcg-rl-league"; KN_SLUG="ptcg-rl-$TAG-train"; KN_TITLE="ptcg rl $TAG train"
DS_DIR="data/nn/push/rl/ds"; KN_DIR="data/nn/push/rl/kn_$TAG"

case "$CMD" in
dataset)
  rm -rf "$DS_DIR"; mkdir -p "$DS_DIR"
  cp -r cg "$DS_DIR/cg"
  rm -f "$DS_DIR"/cg/*.dylib "$DS_DIR"/cg/libcg-arm64.so "$DS_DIR"/cg/libcg.so
  rm -rf "$DS_DIR"/cg/__pycache__
  grep -q HAS_SEEDED_START "$DS_DIR/cg/sim.py" || { echo "cg/sim.py に HAS_SEEDED_START が無い"; exit 1; }
  cp cg/Export_seeded.cpp "$DS_DIR/Export_seeded.cpp"
  cp -r selfplay "$DS_DIR/selfplay"; rm -rf "$DS_DIR"/selfplay/__pycache__
  mkdir -p "$DS_DIR/arena"; cp arena/run_match.py "$DS_DIR/arena/"
  for a in 117_alakazam_dspline 101_marnie_luca 131_ogerpon_majkel 098_spidops_gen; do
    cp -r "agents/$a" "$DS_DIR/$a"; rm -rf "$DS_DIR/$a/__pycache__"
  done
  cat > "$DS_DIR/dataset-metadata.json" <<EOF
{"title": "PTCG RL league sources", "id": "$USER/$DS_SLUG",
 "licenses": [{"name": "CC0-1.0"}]}
EOF
  du -sh "$DS_DIR"
  if uv run kaggle datasets status "$USER/$DS_SLUG" >/dev/null 2>&1; then
    uv run kaggle datasets version -p "$DS_DIR" -m "update $(date +%Y%m%d_%H%M)" --dir-mode zip
  else
    uv run kaggle datasets create -p "$DS_DIR" --dir-mode zip
  fi
  ;;
kernel)
  ACC="${3:-NvidiaTeslaT4}"
  rm -rf "$KN_DIR"; mkdir -p "$KN_DIR"
  cp selfplay/kaggle_league.py "$KN_DIR/$KN_SLUG.py"
  cat > "$KN_DIR/kernel-metadata.json" <<EOF
{"id": "$USER/$KN_SLUG", "title": "$KN_TITLE", "code_file": "$KN_SLUG.py",
 "language": "python", "kernel_type": "script", "is_private": true,
 "enable_gpu": true, "enable_internet": false,
 "dataset_sources": ["$USER/$DS_SLUG"],
 "competition_sources": ["pokemon-tcg-ai-battle"], "kernel_sources": []}
EOF
  # **データセットに最新コードが載っているかを確認してから投入する**。
  # 同じ轍を3回踏んだ: dataset の反映前に kernel を押すと、古い league.py で走って
  # "unrecognized arguments" で落ちる（しかも数分後にしか分からない）。
  uv run kaggle datasets status "$USER/$DS_SLUG" | tail -1
  echo "→ 'ready' でなければ dataset の処理待ち。少し待って再実行すること"
  uv run kaggle kernels push -p "$KN_DIR" --accelerator "$ACC"
  ;;
status) uv run kaggle kernels status "$USER/$KN_SLUG" ;;
output)
  OUT="${3:-data/nn/push/rl/$TAG}"; mkdir -p "$OUT"
  uv run kaggle kernels output "$USER/$KN_SLUG" -p "$OUT"; ls -la "$OUT" ;;
*) echo "usage: $0 {dataset|kernel|status|output}"; exit 1;;
esac
