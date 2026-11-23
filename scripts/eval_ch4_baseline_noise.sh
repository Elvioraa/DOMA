#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=0

model_dir=${1:?Usage: bash scripts/eval_ch4_baseline_noise.sh MODEL_DIR [heal|doma] [inference options]}
method=${2:-heal}
shift
if (( $# )); then shift; fi
case "$method" in
  heal) module=opencood.tools.inference_heter_in_order ;;
  doma) module=opencood.tools.inference_doma_heter_in_order ;;
  *) echo "Expected heal or doma" >&2; exit 2 ;;
esac

log_dir="$model_dir/ch4_noise_logs/$method"
mkdir -p "$log_dir"
for std in 0.0 0.2 0.4 0.6; do
  python -u -m "$module" --model_dir "$model_dir" --use_cav '[4]' "$@" \
    --pose_noise_xy "$std" --pose_noise_yaw "$std" --pose_noise_seed 303 \
    --eval_distance_buckets --note "_ch4_${method}_${std}" 2>&1 | tee "$log_dir/xy${std}_yaw${std}.log"
done
