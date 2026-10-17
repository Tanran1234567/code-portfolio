#!/usr/bin/env bash
# Upload Creatinine Ward v3 downstream code only. Does not touch v1/v2 outputs.
set -euo pipefail

export BITAHUB_HOST="${BITAHUB_HOST:-xj-member.bitahub.com}"
export BITAHUB_PORT="${BITAHUB_PORT:-42024}"
export BITAHUB_USER="${BITAHUB_USER:-root}"
REMOTE_ROOT="${REMOTE_ROOT:-/clin-jepa-storage/medin_hosp_v1}"
LOCAL_CODE="${LOCAL_CODE:-/home/zhoujiale/Clin-JEPA-main_new}"

SSH=(ssh -o StrictHostKeyChecking=accept-new -p "$BITAHUB_PORT")

echo "== create v3 dirs (do not touch v1/v2) =="
"${SSH[@]}" "$BITAHUB_USER@$BITAHUB_HOST" \
  "mkdir -p '$REMOTE_ROOT/Clin-JEPA-main_new' \
            '$REMOTE_ROOT/data/creatinine_ward_v3_kdigo_scr' \
            '$REMOTE_ROOT/output_creatinine_ward_v3_downstream/logs'"

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
set -u
echo REMOTE_ROOT=$REMOTE_ROOT
fail=0
check_file() {
  if test -f "\$1"; then echo "OK file \$1"; else echo "MISSING file \$1"; fail=1; fi
}
check_nonempty() {
  if test -s "\$1"; then echo "OK nonempty \$1"; else echo "MISSING/EMPTY \$1"; fail=1; fi
}
check_dir() {
  if test -d "\$1"; then echo "OK dir \$1"; else echo "MISSING dir \$1"; fail=1; fi
}
check_file $REMOTE_ROOT/Clin-JEPA-main_new/scripts/bitahub_creatinine_ward_v3_downstream.sh
check_file $REMOTE_ROOT/Clin-JEPA-main_new/clin_jepa/data_hosp/creatinine_ward_v3.py
check_file $REMOTE_ROOT/Clin-JEPA-main_new/clin_jepa/evaluation/creatinine_aki_v3_probe.py
check_file $REMOTE_ROOT/Clin-JEPA-main_new/configs/data_hosp/creatinine_ward_paths_v3.yaml
check_file $REMOTE_ROOT/Clin-JEPA-main_new/configs/eval/precompute_creatinine_ward_v3.yaml
check_file $REMOTE_ROOT/Clin-JEPA-main_new/configs/eval/creatinine_aki_v3_probe.yaml
# BitaHub copy lives under the shared disk, not the local /data/projects path.
check_nonempty $REMOTE_ROOT/data/creatinine_ward_v1/manifests/cohort_3000.csv
check_nonempty $REMOTE_ROOT/data/creatinine_ward_v1/splits/splits.csv
check_dir $REMOTE_ROOT/data/creatinine_ward_v1/trajectories/train
check_nonempty $REMOTE_ROOT/output_creatinine_ward_v2_action_aligned/clin_jepa/checkpoints/final/online/adapter_config.json
check_nonempty $REMOTE_ROOT/output_creatinine_ward_v2_action_aligned/clin_jepa/checkpoints/final/predictor.pt
if test "\$fail" -ne 0; then
  echo REMOTE_SANITY_FAILED
  exit 1
fi
chmod +x $REMOTE_ROOT/Clin-JEPA-main_new/scripts/bitahub_creatinine_ward_v3_downstream.sh
echo V1_3000_AND_V2_MODEL_STILL_THERE
echo UPLOAD_V3_OK
EOF

echo
echo "Code uploaded. On BitaHub SSH, run the v3 preflight and all stages."
