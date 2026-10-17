#!/usr/bin/env bash
# BitaHub: paper-style from-scratch SFT + Phase 1–5. No ICU LoRA / predictor.pt.
set -Eeuo pipefail

ROOT="${ROOT:-/clin-jepa-storage/medin_hosp_v1}"
REPO="${REPO:-$ROOT/Clin-JEPA-main_new}"
QWEN_PATH="${QWEN_PATH:-/model/bitahub-model/pice35408784b54431987c4d13c457b9cd/Qwen3-8B}"
MEDIN_DATA_ROOT="${MEDIN_DATA_ROOT:-$ROOT/data/medin_v2_episode_mask_v3}"
MEDIN_OUTPUT="${MEDIN_OUTPUT:-$ROOT/output_v2_episode_mask_v3}"
HF_HOME="${HF_HOME:-$ROOT/hf}"
NPROC="${NPROC:-4}"
SMOKE_ONLY="${SMOKE_ONLY:-0}"

# A smoke-only SFT must never create the production `final/` sentinel, or the
# subsequent full run would incorrectly skip full-corpus SFT.
if [[ "$SMOKE_ONLY" == "1" ]]; then
  MEDIN_OUTPUT="${MEDIN_OUTPUT}_smoke"
fi

mkdir -p "$MEDIN_OUTPUT/logs"
LOG="$MEDIN_OUTPUT/logs/v2_train_$(date +%Y%m%d_%H%M%S).log"
exec >>"$LOG" 2>&1

source "$ROOT/venvs/medin/bin/activate"
export QWEN_PATH MEDIN_DATA_ROOT MEDIN_OUTPUT HF_HOME
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
unset CLIN_JEPA_BASELINE
cd "$REPO"

for required in \
  trajectories_episode/train \
  trajectories_episode/val \
  trajectories_dense/train \
  trajectories_dense/val \
  trajectories_gap/train \
  trajectories_gap/val \
  trajectories_gap/test \
  migration_report.json \
  verify_v2_report.json; do
  if [[ ! -e "$MEDIN_DATA_ROOT/$required" ]]; then
    echo "Missing $MEDIN_DATA_ROOT/$required" >&2
    echo "Run the v0.3 conversion + verification before training. Refusing fallback." >&2
    exit 2
  fi
done
python -c 'import json,os; p=os.path.join(os.environ["MEDIN_DATA_ROOT"],"verify_v2_report.json"); r=json.load(open(p)); assert r.get("schema_version")=="medin_v03_episode_time_no_action_verify_v2", r'
python scripts/preflight_v03_runtime.py --data-root "$MEDIN_DATA_ROOT"

SFT_DIR="$MEDIN_OUTPUT/encoder_sft_v2_unified/final"
if [[ -d "$SFT_DIR" ]]; then
  python -c 'import json,os; p=os.path.join(os.environ["MEDIN_OUTPUT"],"encoder_sft_v2_unified","training_meta.json"); r=json.load(open(p)); assert r.get("schema_version")=="encoder_sft_v3_episode_no_action_lazy", r'
fi
if [[ ! -d "$SFT_DIR" ]]; then
  echo "[$(date -Is)] Encoder SFT from scratch (fresh LoRA, no ICU adapter)"
  sft_args=()
  if [[ "$SMOKE_ONLY" == "1" ]]; then
    sft_args+=(--max_samples 200)
  fi
  python scripts/build_sft_index_v03.py \
    --config configs/train/encoder_sft_v2_unified.yaml \
    "${sft_args[@]}"
  torchrun --standalone --nproc_per_node="$NPROC" \
    -m clin_jepa.training.encoder_sft \
    --config configs/train/encoder_sft_v2_unified.yaml \
    "${sft_args[@]}"
fi
if [[ ! -d "$SFT_DIR" ]]; then
  echo "Encoder SFT did not write $SFT_DIR; refusing ICU fallback." >&2
  exit 2
fi

PRETRAIN_CFG=configs/train/pretrain_clin_jepa_v2_unified.yaml
echo "[$(date -Is)] JEPA smoke (random predictor, SFT encoder only)"
python -m clin_jepa.training.pretrain_clin_jepa_v2 \
  --config "$PRETRAIN_CFG" \
  --smoke --max_windows 200 \
  --per_gpu_batch_size 1 --grad_accum 1 \
  --log_every 1 --eval_every 20 \
  --resume_from none \
  --save_dir "$MEDIN_OUTPUT/smoke_v2"

if [[ "$SMOKE_ONLY" == "1" ]]; then
  echo "[$(date -Is)] SMOKE_ONLY done"
  exit 0
fi

echo "[$(date -Is)] full Phase 1-5 from scratch (auto-resume if resume.pt exists)"
python scripts/build_trajectory_indices_v03.py --config "$PRETRAIN_CFG"
torchrun --standalone --nproc_per_node="$NPROC" \
  -m clin_jepa.training.pretrain_clin_jepa_v2 \
  --config "$PRETRAIN_CFG"

echo "[$(date -Is)] V2_TRAINING_OK"
touch "$MEDIN_OUTPUT/V2_TRAINING_OK"
