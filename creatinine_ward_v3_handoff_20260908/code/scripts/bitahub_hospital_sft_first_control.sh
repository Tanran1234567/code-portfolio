#!/usr/bin/env bash
# Optional E1 control. It is intentionally separate from the primary M0 run.
set -Eeuo pipefail

ROOT="${ROOT:-/clin-jepa-storage/medin_hosp_v1}"
REPO="${REPO:-$ROOT/Clin-JEPA-main}"
MEDIN_DATA_ROOT="${MEDIN_DATA_ROOT:-$ROOT/data/medin_hosp_v1_mixed}"
MEDIN_OUTPUT="${MEDIN_OUTPUT:-$ROOT/output}"
CLIN_JEPA_BASELINE="${CLIN_JEPA_BASELINE:-$ROOT/models/clin_jepa_best}"
QWEN_PATH="${QWEN_PATH:-/model/bitahub-model/pice35408784b54431987c4d13c457b9cd/Qwen3-8B}"
HF_HOME="${HF_HOME:-$ROOT/hf}"
ICU_PRETRAIN_SPLIT="${ICU_PRETRAIN_SPLIT:-}"
source "$ROOT/venvs/medin/bin/activate"
export ROOT REPO MEDIN_DATA_ROOT MEDIN_OUTPUT CLIN_JEPA_BASELINE
export ICU_PRETRAIN_SPLIT QWEN_PATH HF_HOME
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd "$REPO"

if [[ -z "$ICU_PRETRAIN_SPLIT" || ! -e "$ICU_PRETRAIN_SPLIT" ]]; then
  echo "ICU_PRETRAIN_SPLIT is required before any held-out test." >&2
  exit 2
fi

torchrun --standalone --nproc_per_node=4 \
  -m clin_jepa.training.encoder_sft \
  --config configs/train/encoder_sft_hosp_v1_mixed.yaml

python -m clin_jepa.evaluation.evaluate_encoder_sft \
  --config configs/train/encoder_sft_hosp_v1_mixed.yaml --split val --gpu_id 0

python scripts/audit_pretrain_split_overlap.py \
  --icu_pretrain "$ICU_PRETRAIN_SPLIT" \
  --hospital_train "$MEDIN_DATA_ROOT/trajectories/train" \
  --hospital_val "$MEDIN_DATA_ROOT/trajectories/val" \
  --hospital_test "$MEDIN_DATA_ROOT/trajectories/test" \
  --column subject_id \
  --output "$MEDIN_OUTPUT/evaluation/hosp_v1_mixed_sft_first/patient_overlap_audit.json"

python -m clin_jepa.evaluation.evaluate_encoder_sft \
  --config configs/train/encoder_sft_hosp_v1_mixed.yaml --split test --gpu_id 0

export HOSPITAL_SFT_ADAPTER
HOSPITAL_SFT_ADAPTER="$(python - <<'PY'
import json
import os
path = os.path.join(
    os.environ["MEDIN_OUTPUT"], "evaluation/hosp_v1_mixed/sft/selection.json"
)
with open(path) as f:
    print(json.load(f)["selected_adapter"])
PY
)"

torchrun --standalone --nproc_per_node=4 \
  -m clin_jepa.training.pretrain_clin_jepa \
  --config configs/train/pretrain_clin_jepa_hosp_v1_mixed_sft_first.yaml \
  --resume_from none

PRECOMPUTE_CONFIG=configs/eval/precompute_embeddings_hosp_v1_mixed_sft_first.yaml \
ROLLOUT_CONFIG=configs/eval/rollout_hosp_v1_mixed_sft_first.yaml \
PLANS_CSV=sft_first_best,sft_first_latest,sft_first_final \
EVAL_NAMESPACE=hosp_v1_mixed_sft_first \
bash scripts/bitahub_eval_hosp_v1_mixed.sh
