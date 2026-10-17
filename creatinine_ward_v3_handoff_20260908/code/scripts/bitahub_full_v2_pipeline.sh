#!/usr/bin/env bash
# Launch train+eval detached on BitaHub. Closing your laptop is fine;
# rebooting the GPU instance is not unless resume.pt exists (auto-loaded).
set -Eeuo pipefail
ROOT="${ROOT:-/clin-jepa-storage/medin_hosp_v1}"
REPO="${REPO:-$ROOT/Clin-JEPA-main_new}"
export ROOT REPO
MEDIN_OUTPUT="${MEDIN_OUTPUT:-$ROOT/output_v2_episode_mask_v3}"
export MEDIN_OUTPUT
mkdir -p "$MEDIN_OUTPUT/logs"
LOG="$MEDIN_OUTPUT/logs/full_pipeline_$(date +%Y%m%d_%H%M%S).out"
nohup bash -lc "
  bash '$REPO/scripts/bitahub_train_v2.sh'
  bash '$REPO/scripts/bitahub_eval_v2.sh'
" >>"$LOG" 2>&1 &
echo $! | tee "$MEDIN_OUTPUT/logs/full_pipeline.pid"
echo "log $LOG"
