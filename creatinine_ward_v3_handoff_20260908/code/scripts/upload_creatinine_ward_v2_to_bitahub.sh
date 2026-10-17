#!/usr/bin/env bash
# Upload v2 action-aligned code only. Does not touch v1 data or v1 output.
set -euo pipefail

export BITAHUB_HOST="${BITAHUB_HOST:-xj-member.bitahub.com}"
export BITAHUB_PORT="${BITAHUB_PORT:-42081}"
export BITAHUB_USER="${BITAHUB_USER:-root}"
REMOTE_ROOT="${REMOTE_ROOT:-/clin-jepa-storage/medin_hosp_v1}"
LOCAL_CODE="${LOCAL_CODE:-/home/zhoujiale/Clin-JEPA-main_new}"

SSH=(ssh -o StrictHostKeyChecking=accept-new -p "$BITAHUB_PORT")

echo "== create v2 output dir (do not touch v1 output) =="
"${SSH[@]}" "$BITAHUB_USER@$BITAHUB_HOST" \
  "mkdir -p '$REMOTE_ROOT/Clin-JEPA-main_new' \
            '$REMOTE_ROOT/output_creatinine_ward_v2_action_aligned/logs'"

echo "== upload code =="
rsync -avP \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '.git/' \
  --exclude 'reports/' \
  --exclude '.pytest_cache/' \
  -e "ssh -o StrictHostKeyChecking=accept-new -p $BITAHUB_PORT" \
  "$LOCAL_CODE/" \
  "$BITAHUB_USER@$BITAHUB_HOST:$REMOTE_ROOT/Clin-JEPA-main_new/"

echo "== remote sanity =="
"${SSH[@]}" "$BITAHUB_USER@$BITAHUB_HOST" bash -s <<EOF
set -e
echo REMOTE_ROOT=$REMOTE_ROOT
test -f $REMOTE_ROOT/Clin-JEPA-main_new/scripts/bitahub_creatinine_ward_v2_retrain.sh
test -f $REMOTE_ROOT/Clin-JEPA-main_new/configs/train/pretrain_creatinine_ward_v2_action_aligned.yaml
test -s $REMOTE_ROOT/data/creatinine_ward_v1/verify_creatinine_ward_v1.json
test -d $REMOTE_ROOT/data/creatinine_ward_v1/trajectories/train
test -s $REMOTE_ROOT/output_creatinine_ward_v1/encoder_sft/best/adapter_config.json
test -s $REMOTE_ROOT/output_creatinine_ward_v1/EVAL_OK
chmod +x $REMOTE_ROOT/Clin-JEPA-main_new/scripts/bitahub_creatinine_ward_v2_retrain.sh
echo V1_DATA_AND_SFT_STILL_THERE
ls -lh $REMOTE_ROOT/data/creatinine_ward_v1/verify_creatinine_ward_v1.json
ls -d $REMOTE_ROOT/output_creatinine_ward_v1/encoder_sft/best
echo UPLOAD_V2_OK
EOF

echo
echo "Code uploaded. On the BitaHub SSH session run preflight, then the nohup block."
