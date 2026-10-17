#!/usr/bin/env bash
# ICU-free 3000-patient, fixed-24h creatinine representation experiment.
# This script builds data only. It never starts SFT/JEPA training.

set -euo pipefail
cd "$(dirname "$0")/.."

FROM=01
N_WORKERS="${N_WORKERS:-4}"
PATHS="creatinine_ward_paths_v1.yaml"
FEATURES="creatinine_ward_features_v1.yaml"
TRAJECTORIES="creatinine_ward_trajectories_v1.yaml"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --from) FROM="$2"; shift 2 ;;
    --n-workers) N_WORKERS="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

: "${CREATININE_WARD_DATA_ROOT:?set CREATININE_WARD_DATA_ROOT}"
: "${MIMIC_DUCKDB:?set MIMIC_DUCKDB}"

run_step() {
  local number="$1"; shift
  if [[ "$FROM" -le "$number" ]]; then
    echo "[$(date +%FT%T)] step${number}: $*"
    "$@"
  fi
}

run_step 01 python3 -m clin_jepa.data_hosp.creatinine_ward_v1 \
  all --paths-config "$PATHS"
run_step 02 python3 -m clin_jepa.data_hosp.step02_hosp_observations \
  --paths_config "$PATHS" --features_config "$FEATURES"
run_step 03 python3 -m clin_jepa.data_hosp.step03_hosp_actions \
  --paths_config "$PATHS" --features_config "$FEATURES"
run_step 04 python3 -m clin_jepa.data_hosp.step04_hosp_discretization \
  --paths_config "$PATHS" --features_config "$FEATURES"
# Step 01 already writes the frozen 2100/450/450 patient split. Do not call
# the generic step05 splitter here; doing so would silently change the cohort.
if [[ "$FROM" -le 05 ]]; then
  test -s "$CREATININE_WARD_DATA_ROOT/splits/splits.csv"
  echo "[$(date +%FT%T)] step05: keep frozen cohort split"
fi
run_step 06 python3 -m clin_jepa.data_hosp.step06_hosp_trajectories \
  --paths_config "$PATHS" --traj_config "$TRAJECTORIES" \
  --n_workers "$N_WORKERS"
run_step 07 python3 scripts/verify_creatinine_ward_v1.py \
  --data-root "$CREATININE_WARD_DATA_ROOT"

echo "CREATININE_WARD_DATA_PIPELINE_OK"
