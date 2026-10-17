#!/usr/bin/env bash
# Full-MIMIC MedIn v0.3 data: cohort → observations → mixed bins → episode
# trajectories → dense / gap pools. Does NOT subsample 3000 patients.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-$(pwd)}"

FROM=01
N_WORKERS="${N_WORKERS:-8}"
PATHS="hosp_paths_v2_full.yaml"
FEAT="hosp_features_v2_full.yaml"
TRAJ="hosp_trajectories_v2_full.yaml"
COHORT="hosp_cohort_v2_full.yaml"
ICU_SPLIT_CSV="${ICU_SPLIT_CSV:-}"
ROOT="${MEDIN_DATA_ROOT:-/home/zhoujiale/medin_v2_full}"
export MEDIN_DATA_ROOT="$ROOT"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --from) FROM="$2"; shift 2 ;;
    --n_workers) N_WORKERS="$2"; shift 2 ;;
    --icu_split_csv) ICU_SPLIT_CSV="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

should_run() { [[ "$FROM" -le "$1" ]]; }

LOG_DIR="${ROOT}/logs"
mkdir -p "$LOG_DIR"
STAMP=$(date +%Y%m%d_%H%M%S)
run_step() {
  local name="$1"; shift
  echo "[$(date +%H:%M:%S)] START $name"
  "$@" 2>&1 | tee "$LOG_DIR/${name}_${STAMP}.log"
  echo "[$(date +%H:%M:%S)] DONE  $name"
}

if should_run 1; then
  run_step step01 python3 -m clin_jepa.data_hosp.step01_hosp_cohort \
    --paths_config "$PATHS" --cohort_config "$COHORT"
fi
if should_run 5; then
  extra=()
  if [[ -n "$ICU_SPLIT_CSV" ]]; then
    extra+=(--icu_split_csv "$ICU_SPLIT_CSV")
  fi
  run_step step05 python3 -m clin_jepa.data_hosp.step05_hosp_split \
    --paths_config "$PATHS" --seed 42 "${extra[@]}"
fi
if should_run 2; then
  run_step step02 python3 -m clin_jepa.data_hosp.step02_hosp_observations \
    --paths_config "$PATHS" --features_config "$FEAT"
fi
if should_run 3; then
  run_step step03 python3 -m clin_jepa.data_hosp.step03_hosp_actions \
    --paths_config "$PATHS" --features_config "$FEAT"
fi
if should_run 4; then
  run_step step04_mixed python3 -m clin_jepa.data_hosp.step04_hosp_mixed_discretization \
    --paths_config "$PATHS" --features_config "$FEAT"
fi
if should_run 6; then
  run_step step06 python3 -m clin_jepa.data_hosp.step06_hosp_trajectories \
    --paths_config "$PATHS" --traj_config "$TRAJ" --n_workers "$N_WORKERS"
  run_step convert_episodes python3 scripts/convert_mixed_to_episodes_v2.py \
    --src "$ROOT/trajectories" --dst "$ROOT/trajectories_episode"
  run_step dense python3 scripts/build_dense_segments_v2.py \
    --src "$ROOT/trajectories_episode" --dst "$ROOT/trajectories_dense"
  run_step gap python3 scripts/build_gap_trajectories_v2.py \
    --src "$ROOT/trajectories_episode" --dst "$ROOT/trajectories_gap"
  run_step verify_v2 python3 scripts/verify_medin_v2_full.py --root "$ROOT"
fi

echo "MedIn v0.3 full data pipeline finished at $ROOT"
