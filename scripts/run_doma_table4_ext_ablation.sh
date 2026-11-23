#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,2
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
PY="${DOMA_PYTHON:-/home/fkk/my_data/disk1/miniconda3/envs/DOMA/bin/python}"
OUT="$ROOT/opencood/logs/DOMA_V2_R2_2GPU_TABLE4_EXT_ABLATION_SEED303"
SESSION="doma-table4-ext-$(printf '%s' "$ROOT" | sha256sum | cut -c1-10)"
MODE="${1:-status}"
case "$MODE" in
  check) shift; exec "$PY" -B -m opencood.tools.table4_ext check "$@" ;;
  status) exec "$PY" -B -m opencood.tools.table4_ext status ;;
  start|resume)
    mkdir -p "$OUT/logs"
    exec 9>"$OUT/.launch.lock"
    flock -n 9 || { echo 'Another launch/check is already in progress'; exit 1; }
    if tmux has-session -t "$SESSION" 2>/dev/null; then
      echo "Already running: tmux attach -t $SESSION"
      exit 0
    fi
    # Held worker lock protects launches even if someone renamed the tmux session.
    flock -n "$OUT/.worker.lock" true || { echo 'Worker already running'; exit 1; }
    "$PY" -B -m opencood.tools.table4_ext check
    "$PY" -B -m opencood.tools.table4_ext prepare
    printf -v COMMAND 'cd %q && DOMA_PYTHON=%q bash %q _run' "$ROOT" "$PY" "$ROOT/scripts/run_doma_table4_ext_ablation.sh"
    tmux new-session -d -s "$SESSION" "$COMMAND"
    echo "Started background session: $SESSION"
    echo "SSH may disconnect. Server reboot requires manual resume."
    echo "Logs: $OUT/logs (status reports startup failures)"
    ;;
  _run)
    mkdir -p "$OUT/logs"
    LOG="$OUT/logs/controller-$(date +%Y%m%dT%H%M%S)-$$.log"
    exec "$PY" -B -m opencood.tools.table4_ext worker >>"$LOG" 2>&1
    ;;
  *) echo 'Usage: run_doma_table4_ext_ablation.sh check|start|status|resume'; exit 2 ;;
esac
