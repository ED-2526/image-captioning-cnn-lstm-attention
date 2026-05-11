#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p logs

export PY="${PY:-$HOME/miniconda3/envs/dl_env/bin/python}"
export WANDB_MODE="${WANDB_MODE:-offline}"

RUN_LOG="logs/clean_all_$(date +%Y%m%d_%H%M%S).log"

nohup bash -lc '
set -euo pipefail

for group in backbone embeddings loss dataset learning-rate implementation-first; do
  echo "===== $(date) running ${group} ====="
  bash scripts/run_clean_ablations.sh "$group"
  echo "===== $(date) finished ${group} ====="
done
' > "$RUN_LOG" 2>&1 &

echo "PID: $!"
echo "Log: $RUN_LOG"
echo "Python: $PY"
echo "WANDB_MODE: $WANDB_MODE"
