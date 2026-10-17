#!/usr/bin/env bash
# Detach full-MIMIC extraction so SSH / laptop shutdown does not stop it.
# The job runs on THIS server; closing your notebook is fine. Rebooting
# this machine is not — then rerun the same command (it will not auto-resume
# a half-written DuckDB extract; use --from to skip finished steps).
set -euo pipefail
ROOT="${MEDIN_DATA_ROOT:-/home/zhoujiale/medin_v2_full}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/logs"
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="$ROOT/logs/full_data_${STAMP}.out"
PIDFILE="$ROOT/logs/full_data.pid"
if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "already running pid=$(cat "$PIDFILE")"
  exit 1
fi
nohup bash "$REPO/scripts/run_medin_v2_full_data.sh" "$@" \
  >>"$LOG" 2>&1 &
echo $! >"$PIDFILE"
echo "started pid=$(cat "$PIDFILE")"
echo "log $LOG"
echo "tail -f $LOG"
