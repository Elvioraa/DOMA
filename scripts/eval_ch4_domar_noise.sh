#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${GPU_ID:-1}"
model=${1:?Usage: eval_ch4_domar_noise.sh ASSEMBLED_MODEL_DIR STAGE15_DIR [inference options]}
plugin=${2:?Expected Stage 1.5 output directory}/domar_plugin_best.pth
shift 2
logs=${LOG_DIR:-"$model/ch4_noise_logs/domar"}
mkdir -p "$logs"
for std in 0.0 0.2 0.4 0.6; do
  python -u -m opencood.tools.inference_doma_heter_in_order --model_dir "$model" \
    --domar_plugin_ckpt "$plugin" --enable_quality_weight --enable_iou_calib \
    --use_cav '[4]' --eval_distance_buckets --pose_noise_seed 303 \
    --pose_noise_xy "$std" --pose_noise_yaw "$std" --note "_ch4_domar_${std}" "$@" \
    2>&1 | tee "$logs/xy${std}_yaw${std}.log"
done
