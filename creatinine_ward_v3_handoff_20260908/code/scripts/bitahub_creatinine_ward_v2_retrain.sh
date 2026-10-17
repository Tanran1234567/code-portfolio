#!/usr/bin/env bash
# Re-train only Clin-JEPA Phase 1-5 with target-interval action alignment.
# Reuses the verified v1 trajectories and v1 encoder_sft/best.

set -Eeuo pipefail
cd "$(dirname "$0")/.."

: "${CREATININE_WARD_DATA_ROOT:?set CREATININE_WARD_DATA_ROOT}"
: "${CREATININE_WARD_OUTPUT:?set CREATININE_WARD_OUTPUT to the v1 output}"
: "${CREATININE_WARD_V2_OUTPUT:?set CREATININE_WARD_V2_OUTPUT}"
: "${QWEN_PATH:?set QWEN_PATH}"
: "${HF_HOME:?set HF_HOME}"

if [[ "$CREATININE_WARD_V2_OUTPUT" == "$CREATININE_WARD_OUTPUT" ]]; then
  echo "CREATININE_WARD_V2_OUTPUT must not equal v1 CREATININE_WARD_OUTPUT" >&2
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
STAGE="${1:-all}"
CONFIG="configs/train/pretrain_creatinine_ward_v2_action_aligned.yaml"
PRECOMPUTE="configs/eval/precompute_creatinine_ward_v2_action_aligned.yaml"
PROBE="configs/eval/creatinine_probe_v2_action_aligned.yaml"
LOG_DIR="$CREATININE_WARD_V2_OUTPUT/logs"
mkdir -p "$LOG_DIR" "$CREATININE_WARD_V2_OUTPUT/clin_jepa"

preflight() {
  test -s "$CREATININE_WARD_DATA_ROOT/verify_creatinine_ward_v1.json"
  test -s "$CREATININE_WARD_OUTPUT/encoder_sft/best/adapter_config.json"
  test -d "$CREATININE_WARD_DATA_ROOT/trajectories/train"
  test -d "$CREATININE_WARD_DATA_ROOT/trajectories/val"
  test -d "$CREATININE_WARD_DATA_ROOT/trajectories/test"
  python3 - "$CREATININE_WARD_DATA_ROOT/verify_creatinine_ward_v1.json" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
assert report["status"] == "VERIFY_OK", report
assert report["totals"]["admissions"] == 3000, report["totals"]
assert report["totals"]["windows"] == 3000, report["totals"]
assert report["invariants"]["icu_admissions"] == 0, report["invariants"]
print("V2_INPUT_PREFLIGHT_OK", report["cohort_sha256"])
PY
  python3 - "$CONFIG" <<'PY'
import math, os, sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
phases = cfg["training"]["phases"]
assert sum(phases.values()) == 1600, phases
assert cfg["predictor"]["prediction_mode"] == "residual"
assert cfg["predictor"]["action_alignment"] == "target_interval"
tau = float(cfg["training"]["ema"]["tau_phase_2b"])
remaining = tau ** int(phases["phase_2b_steps"])
assert 0.01 < remaining < 0.03, remaining
print("V2_CONFIG_PREFLIGHT_OK", phases, "alignment_residual", remaining)
PY
}

build_indices() {
  python3 scripts/build_trajectory_indices_v03.py --config "$CONFIG"
  touch "$CREATININE_WARD_V2_OUTPUT/INDEX_OK"
}

train_jepa() {
  local resume_arg=(--resume_from none)
  if [[ "${RESUME:-0}" == "1" ]]; then
    resume_arg=()
  fi
  torchrun --standalone --nproc_per_node="$NPROC" \
    -m clin_jepa.training.pretrain_clin_jepa \
    --config "$CONFIG" "${resume_arg[@]}" \
    2>&1 | tee "$LOG_DIR/jepa_v2.log"
  touch "$CREATININE_WARD_V2_OUTPUT/JEPA_OK"
}

precompute_embeddings() {
  for split in train val test; do
    python3 -m clin_jepa.evaluation.precompute_embeddings \
      --config "$PRECOMPUTE" --plan jepa_v2 --split "$split" \
      2>&1 | tee "$LOG_DIR/embeddings_${split}.log"
  done
  touch "$CREATININE_WARD_V2_OUTPUT/EMBEDDINGS_OK"
}

evaluate_probe() {
  python3 -m clin_jepa.evaluation.creatinine_probe \
    --config "$PROBE" \
    2>&1 | tee "$LOG_DIR/probe_v2.log"
  touch "$CREATININE_WARD_V2_OUTPUT/EVAL_OK"
}

preflight
case "$STAGE" in
  preflight) ;;
  train) build_indices; train_jepa ;;
  embeddings) precompute_embeddings ;;
  eval) evaluate_probe ;;
  all)
    build_indices
    train_jepa
    precompute_embeddings
    evaluate_probe
    touch "$CREATININE_WARD_V2_OUTPUT/CREATININE_WARD_V2_OK"
    ;;
  *) echo "usage: $0 [preflight|train|embeddings|eval|all]" >&2; exit 2 ;;
esac
