#!/usr/bin/env bash
# Build Creatinine Ward v3 data and train downstream AKI classifiers only.
# This script reuses the frozen v1 3000-patient cohort and v2 encoder/predictor.

set -Eeuo pipefail
cd "$(dirname "$0")/.."

: "${CREATININE_WARD_DATA_ROOT:?set CREATININE_WARD_DATA_ROOT to formal v1 data root}"
: "${CREATININE_WARD_V3_DATA_ROOT:?set CREATININE_WARD_V3_DATA_ROOT}"
: "${CREATININE_WARD_V2_OUTPUT:?set CREATININE_WARD_V2_OUTPUT}"
: "${CREATININE_WARD_V3_OUTPUT:?set CREATININE_WARD_V3_OUTPUT}"
: "${QWEN_PATH:?set QWEN_PATH}"
: "${HF_HOME:?set HF_HOME}"
STAGE="${1:-all}"
if [[ "$STAGE" == "data" || "$STAGE" == "all" ]]; then
  : "${MIMIC_DUCKDB:?set MIMIC_DUCKDB to build v3 labels from raw labevents}"
fi

if [[ "$CREATININE_WARD_V3_DATA_ROOT" == "$CREATININE_WARD_DATA_ROOT" ]]; then
  echo "CREATININE_WARD_V3_DATA_ROOT must not equal v1 CREATININE_WARD_DATA_ROOT" >&2
  exit 2
fi
if [[ "$CREATININE_WARD_V3_OUTPUT" == "$CREATININE_WARD_V2_OUTPUT" ]]; then
  echo "CREATININE_WARD_V3_OUTPUT must not equal v2 CREATININE_WARD_V2_OUTPUT" >&2
  exit 2
fi

ROOT="${ROOT:-/clin-jepa-storage/medin_hosp_v1}"
if [[ -f "$ROOT/venvs/medin/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/venvs/medin/bin/activate"
fi
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

NPROC="${NPROC_PER_NODE:-4}"
PATHS_CONFIG="configs/data_hosp/creatinine_ward_paths_v3.yaml"
PRECOMPUTE_CONFIG="configs/eval/precompute_creatinine_ward_v3.yaml"
PROBE_CONFIG="configs/eval/creatinine_aki_v3_probe.yaml"
LOG_DIR="$CREATININE_WARD_V3_OUTPUT/logs"
mkdir -p "$LOG_DIR" "$CREATININE_WARD_V3_OUTPUT" "$CREATININE_WARD_V3_DATA_ROOT"

preflight() {
  test -s "$CREATININE_WARD_DATA_ROOT/manifests/cohort_3000.csv"
  test -s "$CREATININE_WARD_DATA_ROOT/splits/splits.csv"
  test -d "$CREATININE_WARD_DATA_ROOT/trajectories/train"
  test -d "$CREATININE_WARD_DATA_ROOT/trajectories/val"
  test -d "$CREATININE_WARD_DATA_ROOT/trajectories/test"
  test -s "$CREATININE_WARD_V2_OUTPUT/clin_jepa/checkpoints/final/online/adapter_config.json"
  test -s "$CREATININE_WARD_V2_OUTPUT/clin_jepa/checkpoints/final/predictor.pt"
  if [[ "$STAGE" == "data" || "$STAGE" == "all" ]]; then
    test -s "$MIMIC_DUCKDB"
  else
    test -s "$CREATININE_WARD_V3_DATA_ROOT/verify_creatinine_ward_v3.json"
    test -d "$CREATININE_WARD_V3_DATA_ROOT/trajectories/train"
  fi
  test -f "$PATHS_CONFIG"
  test -f "$PRECOMPUTE_CONFIG"
  test -f "$PROBE_CONFIG"
  python3 - "$CREATININE_WARD_DATA_ROOT/splits/splits.csv" <<'PY'
import pandas as pd, sys
s = pd.read_csv(sys.argv[1])
assert s.subject_id.nunique() == 3000, s.subject_id.nunique()
assert s.hadm_id.nunique() == 3000, s.hadm_id.nunique()
print("V3_INPUT_PREFLIGHT_OK", s.split.value_counts().sort_index().to_dict())
PY
  python3 - "$PROBE_CONFIG" <<'PY'
import yaml, sys
cfg = yaml.safe_load(open(sys.argv[1]))
assert cfg["anchor"]["min_anchor_idx"] == 2, cfg["anchor"]
assert any(t["name"] == "future48_aki_binary" for t in cfg["targets"])
assert any(t["name"] == "future_48h_scr_aki_occurrence" for t in cfg["targets"])
assert any(t["name"] == "aki72_course_among_current_aki" for t in cfg["targets"])
print("V3_CONFIG_PREFLIGHT_OK")
PY
}

build_data() {
  python3 -m clin_jepa.data_hosp.creatinine_ward_v3 all \
    --paths-config "$PATHS_CONFIG" \
    --v1-data-root "$CREATININE_WARD_DATA_ROOT" \
    --output-root "$CREATININE_WARD_V3_DATA_ROOT" \
    2>&1 | tee "$LOG_DIR/data_v3.log"
  touch "$CREATININE_WARD_V3_OUTPUT/DATA_OK"
}

precompute_split() {
  local split="$1"
  local pids=()
  for gpu in $(seq 0 $((NPROC - 1))); do
    python3 -m clin_jepa.evaluation.precompute_embeddings \
      --config "$PRECOMPUTE_CONFIG" --plan jepa_v2 --split "$split" \
      --shard_stride "$NPROC" --shard_offset "$gpu" --gpu_id "$gpu" \
      > "$LOG_DIR/embeddings_${split}_gpu${gpu}.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "$pid"
  done
}

precompute_embeddings() {
  precompute_split train
  precompute_split val
  precompute_split test
  touch "$CREATININE_WARD_V3_OUTPUT/EMBEDDINGS_OK"
}

evaluate_probe() {
  python3 -m clin_jepa.evaluation.creatinine_aki_v3_probe \
    --config "$PROBE_CONFIG" --gpu-id 0 \
    2>&1 | tee "$LOG_DIR/probe_v3.log"
  touch "$CREATININE_WARD_V3_OUTPUT/EVAL_OK"
}

preflight
case "$STAGE" in
  preflight) ;;
  data) build_data ;;
  embeddings) precompute_embeddings ;;
  probe) evaluate_probe ;;
  all)
    build_data
    precompute_embeddings
    evaluate_probe
    touch "$CREATININE_WARD_V3_OUTPUT/CREATININE_WARD_V3_OK"
    ;;
  *) echo "usage: $0 [preflight|data|embeddings|probe|all]" >&2; exit 2 ;;
esac
