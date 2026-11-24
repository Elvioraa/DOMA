#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${GPU_ID:-1}"
stage1=${1:?Usage: train_ch4_domar_plugin.sh M1_DIR CHECKPOINT [training options]}
ckpt=${2:?Expected Stage 1 checkpoint filename}
shift 2
out=${OUTPUT_DIR:-"$stage1/stage15"}
mkdir -p "$out"
python -u -m opencood.tools.train_domar_plugin --model_dir "$stage1" \
  --checkpoint "$ckpt" --output_dir "$out" "$@" 2>&1 | tee "$out/train.log"
