#!/usr/bin/env bash
# Full-validation then one-shot test for MedIn v0.3.
set -Eeuo pipefail

ROOT="${ROOT:-/clin-jepa-storage/medin_hosp_v1}"
REPO="${REPO:-$ROOT/Clin-JEPA-main_new}"
MEDIN_DATA_ROOT="${MEDIN_DATA_ROOT:-$ROOT/data/medin_v2_episode_mask_v3}"
MEDIN_OUTPUT="${MEDIN_OUTPUT:-$ROOT/output_v2_episode_mask_v3}"
QWEN_PATH="${QWEN_PATH:-/model/bitahub-model/pice35408784b54431987c4d13c457b9cd/Qwen3-8B}"
HF_HOME="${HF_HOME:-$ROOT/hf}"
EVAL_STAGE="${EVAL_STAGE:-all}"

source "$ROOT/venvs/medin/bin/activate"
export QWEN_PATH MEDIN_DATA_ROOT MEDIN_OUTPUT HF_HOME
unset CLIN_JEPA_BASELINE
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd "$REPO"
mkdir -p "$MEDIN_OUTPUT/logs"
for required in migration_report.json verify_v2_report.json; do
  [[ -f "$MEDIN_DATA_ROOT/$required" ]] || {
    echo "Missing migrated-data proof $MEDIN_DATA_ROOT/$required" >&2; exit 2;
  }
done
python -c 'import json,os; p=os.path.join(os.environ["MEDIN_DATA_ROOT"],"verify_v2_report.json"); r=json.load(open(p)); assert r.get("schema_version")=="medin_v03_episode_time_no_action_verify_v2", r'

PRECOMPUTE=configs/eval/precompute_embeddings_v2_unified.yaml
ROLLOUT=configs/eval/rollout_v2_unified.yaml

if [[ "$EVAL_STAGE" != "test" ]]; then
  for i in 0 1 2; do
    plan=$(echo best latest final | awk -v n="$i" '{print $(n+1)}')
    python -m clin_jepa.evaluation.precompute_embeddings \
      --config "$PRECOMPUTE" --plan "$plan" --split val --gpu_id "$i"
  done
  for i in 0 1 2; do
    plan=$(echo best latest final | awk -v n="$i" '{print $(n+1)}')
    python -m clin_jepa.evaluation.evaluate_hosp_mixed \
      --config "$ROLLOUT" --plan "$plan" --split val --gpu_id "$i"
  done
  python -m clin_jepa.evaluation.select_hosp_checkpoint --config "$ROLLOUT"
fi

if [[ "$EVAL_STAGE" == "val" ]]; then
  echo "FULL_VALIDATION_OK"
  exit 0
fi

SELECTED_PLAN=$(python -c 'import json,os; print(json.load(open(os.path.expandvars(os.environ["MEDIN_OUTPUT"] + "/evaluation/v2_unified/checkpoint_selection.json")))["selected_plan"])')
echo "Selected validation checkpoint: $SELECTED_PLAN"
python -m clin_jepa.evaluation.precompute_embeddings \
  --config "$PRECOMPUTE" --plan "$SELECTED_PLAN" --split test --gpu_id 0
python -m clin_jepa.evaluation.evaluate_hosp_mixed \
  --config "$ROLLOUT" --plan "$SELECTED_PLAN" --split test --gpu_id 0
echo "V2_TEST_OK"
