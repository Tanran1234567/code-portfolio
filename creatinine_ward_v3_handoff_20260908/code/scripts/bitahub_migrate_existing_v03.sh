#!/usr/bin/env bash
# Non-destructive migration of uploaded v0.3 trajectory shards. No DuckDB access.
set -Eeuo pipefail

ROOT="${ROOT:-/clin-jepa-storage/medin_hosp_v1}"
REPO="${REPO:-$ROOT/Clin-JEPA-main_new}"
OLD_MEDIN_DATA_ROOT="${OLD_MEDIN_DATA_ROOT:-$ROOT/data/medin_v2_full}"
MEDIN_DATA_ROOT="${MEDIN_DATA_ROOT:-$ROOT/data/medin_v2_episode_mask_v3}"
STAGE="${STAGE:-audit}"               # audit | migrate | verify | all
CONFIRM_MIGRATION="${CONFIRM_MIGRATION:-NO}"
AUDIT_SHARDS="${AUDIT_SHARDS:-3}"
INPUT_KIND="${INPUT_KIND:-auto}"

source "$ROOT/venvs/medin/bin/activate"
cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

if [[ -d "$OLD_MEDIN_DATA_ROOT/trajectories_episode/train" ]]; then
  SRC="$OLD_MEDIN_DATA_ROOT/trajectories_episode"
elif [[ -d "$OLD_MEDIN_DATA_ROOT/trajectories/train" ]]; then
  SRC="$OLD_MEDIN_DATA_ROOT/trajectories"
else
  echo "No uploaded trajectory source under $OLD_MEDIN_DATA_ROOT" >&2
  exit 2
fi

if [[ "$OLD_MEDIN_DATA_ROOT" == "$MEDIN_DATA_ROOT" ]]; then
  echo "OLD_MEDIN_DATA_ROOT and MEDIN_DATA_ROOT must differ" >&2
  exit 2
fi

mkdir -p "$MEDIN_DATA_ROOT/logs"
LOG="$MEDIN_DATA_ROOT/logs/migration_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

run_audit() {
  python scripts/migrate_existing_v03_trajectories.py \
    --src-root "$SRC" --dst-root "$MEDIN_DATA_ROOT" \
    --input-kind "$INPUT_KIND" --audit-only --max-shards "$AUDIT_SHARDS"
}

run_migration() {
  if [[ ! -f "$MEDIN_DATA_ROOT/migration_audit.json" ]]; then
    echo "Run STAGE=audit and inspect migration_audit.json first." >&2
    exit 2
  fi
  if [[ "$CONFIRM_MIGRATION" != "YES" ]]; then
    echo "Set CONFIRM_MIGRATION=YES after reviewing the audit." >&2
    exit 2
  fi
  src_bytes=$(du -sb "$SRC" | awk '{print $1}')
  free_bytes=$(df -PB1 "$MEDIN_DATA_ROOT" | awk 'NR==2 {print $4}')
  existing_bytes=$(du -sb "$MEDIN_DATA_ROOT" | awk '{print $1}')
  # Episode+dense+gap serialization can approach 3x the source episode pool.
  full_required=$((src_bytes * 3 + src_bytes / 5))
  required_bytes=$((full_required - existing_bytes))
  temp_margin=$((src_bytes / 5))
  (( required_bytes < temp_margin )) && required_bytes=$temp_margin
  echo "source_bytes=$src_bytes existing_destination_bytes=$existing_bytes free_bytes=$free_bytes conservative_additional_required=$required_bytes"
  if (( free_bytes < required_bytes )); then
    echo "Insufficient free space for non-destructive migration." >&2
    echo "Do not use in-place conversion. Ask the server AI to provision space" >&2
    echo "or confirm actual pool sizes before lowering the safety multiplier." >&2
    exit 3
  fi
  python scripts/migrate_existing_v03_trajectories.py \
    --src-root "$SRC" --dst-root "$MEDIN_DATA_ROOT" \
    --input-kind "$INPUT_KIND" --resume
}

run_verify() {
  python scripts/verify_medin_v2_full.py --root "$MEDIN_DATA_ROOT"
}

case "$STAGE" in
  audit) run_audit ;;
  migrate) run_migration ;;
  verify) run_verify ;;
  all)
    [[ "$CONFIRM_MIGRATION" == "YES" ]] || {
      echo "STAGE=all requires CONFIRM_MIGRATION=YES" >&2; exit 2;
    }
    run_audit; run_migration; run_verify
    ;;
  *) echo "Unknown STAGE=$STAGE" >&2; exit 2 ;;
esac

echo "MIGRATION_PIPELINE_OK new_root=$MEDIN_DATA_ROOT log=$LOG"
