#!/usr/bin/env bash
# Training/evaluation launcher for the ICU-free creatinine ward experiment.
# Run only after scripts/run_creatinine_ward_v1.sh and verifier success.

set -euo pipefail
cd "$(dirname "$0")/.."

: "${CREATININE_WARD_DATA_ROOT:?set CREATININE_WARD_DATA_ROOT}"
: "${CREATININE_WARD_OUTPUT:?set CREATININE_WARD_OUTPUT}"
: "${QWEN_PATH:?set QWEN_PATH}"
: "${HF_HOME:?set HF_HOME}"

NPROC="${NPROC_PER_NODE:-4}"
FROM="${FROM:-sft}"

case "$FROM" in
  sft) start=1 ;;
  jepa) start=2 ;;
  embeddings) start=3 ;;
  probes) start=4 ;;
  *) echo "FROM must be sft|jepa|embeddings|probes" >&2; exit 2 ;;
esac

if [[ "$start" -le 1 ]]; then
  torchrun --nproc_per_node="$NPROC" -m clin_jepa.training.encoder_sft \
    --config configs/train/encoder_sft_creatinine_ward_v1.yaml
fi

if [[ "$start" -le 2 ]]; then
  torchrun --nproc_per_node="$NPROC" -m clin_jepa.training.pretrain_clin_jepa \
    --config configs/train/pretrain_creatinine_ward_v1.yaml
fi

if [[ "$start" -le 3 ]]; then
  for plan in base sft jepa; do
    for split in train val test; do
      python3 -m clin_jepa.evaluation.precompute_embeddings \
        --config configs/eval/precompute_creatinine_ward_v1.yaml \
        --plan "$plan" --split "$split"
    done
  done
fi

if [[ "$start" -le 4 ]]; then
  python3 -m clin_jepa.evaluation.creatinine_probe \
    --config configs/eval/creatinine_probe_v1.yaml \
    --embeddings "$CREATININE_WARD_OUTPUT/evaluation/embeddings/base" \
    --output "$CREATININE_WARD_OUTPUT/evaluation/creatinine_probe_base" \
    --skip-rollout
  python3 -m clin_jepa.evaluation.creatinine_probe \
    --config configs/eval/creatinine_probe_v1.yaml \
    --embeddings "$CREATININE_WARD_OUTPUT/evaluation/embeddings/sft" \
    --output "$CREATININE_WARD_OUTPUT/evaluation/creatinine_probe_sft" \
    --skip-rollout
  python3 -m clin_jepa.evaluation.creatinine_probe \
    --config configs/eval/creatinine_probe_v1.yaml
fi

echo "CREATININE_WARD_TRAINING_AND_EVAL_OK"
