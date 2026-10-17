# Creatinine Ward v3 downstream runbook

Scope: reuse the frozen v1 3000-patient cohort (local `/data/projects/zhoujiale/creatinine_ward_v1`, BitaHub `$ROOT/data/creatinine_ward_v1`), reuse the v2 encoder/predictor, rebuild only v3 AKI labels/state text and downstream probes. v1 and v2 directories are not overwritten.

This GPU instance does not have MIMIC DuckDB mounted. Build v3 data on the local machine (`/data/general_disease_pred/mimic4.db`), upload only the new v3 data directory, then run embeddings and probes on BitaHub. Do not re-upload v1 data, and do not run `all` on GPU.

Key differences from v1/v2:

- Reference A is prior 7-365 day median serum creatinine when available.
- If Reference A is unavailable, v3 records and uses the first creatinine in the index admission as fallback.
- All serum creatinine measurements inside each 24h bucket are retained in state text.
- AKI labels are computed from raw `mimiciv_hosp.labevents`, not from state text.
- Main anchors require fixed 72h history: current bucket plus previous two buckets, so `anchor_idx >= 2`.

## 1. Local: build v3 data (no GPU)

Keep the BitaHub SSH session open. In a **new local** terminal:

```bash
export REPO=/home/zhoujiale/Clin-JEPA-main_new
export MIMIC_DUCKDB=/data/general_disease_pred/mimic4.db
export CREATININE_WARD_DATA_ROOT=/data/projects/zhoujiale/creatinine_ward_v1
export CREATININE_WARD_V3_DATA_ROOT=/data/projects/zhoujiale/creatinine_ward_v3_kdigo_scr
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
cd "$REPO"
mkdir -p "$CREATININE_WARD_V3_DATA_ROOT/_duckdb_tmp" "$CREATININE_WARD_V3_DATA_ROOT/logs"
python3 -m clin_jepa.data_hosp.creatinine_ward_v3 all \
  --paths-config configs/data_hosp/creatinine_ward_paths_v3.yaml \
  --v1-data-root "$CREATININE_WARD_DATA_ROOT" \
  --output-root "$CREATININE_WARD_V3_DATA_ROOT" \
  | tee "$CREATININE_WARD_V3_DATA_ROOT/logs/data_v3.log"
```

Must print `CREATININE_WARD_V3_VERIFY_OK`. Do not overwrite the v1 directory.

## 2. Local: upload v3 data + updated launcher

```bash
export BITAHUB_HOST=xj-member.bitahub.com
export BITAHUB_PORT=42024
export BITAHUB_USER=root
export REMOTE_ROOT=/clin-jepa-storage/medin_hosp_v1
SSH="ssh -o StrictHostKeyChecking=accept-new -p $BITAHUB_PORT"

rsync -avP -e "$SSH" \
  /home/zhoujiale/Clin-JEPA-main_new/scripts/bitahub_creatinine_ward_v3_downstream.sh \
  "$BITAHUB_USER@$BITAHUB_HOST:$REMOTE_ROOT/Clin-JEPA-main_new/scripts/"

rsync -avP --exclude '_duckdb_tmp/' --exclude 'logs/' -e "$SSH" \
  /data/projects/zhoujiale/creatinine_ward_v3_kdigo_scr/ \
  "$BITAHUB_USER@$BITAHUB_HOST:$REMOTE_ROOT/data/creatinine_ward_v3_kdigo_scr/"
```

## 3. BitaHub SSH: embeddings then probe (no data rebuild)

```bash
export ROOT=/clin-jepa-storage/medin_hosp_v1
export REPO=$ROOT/Clin-JEPA-main_new
export CREATININE_WARD_DATA_ROOT=$ROOT/data/creatinine_ward_v1
export CREATININE_WARD_V2_OUTPUT=$ROOT/output_creatinine_ward_v2_action_aligned
export CREATININE_WARD_V3_DATA_ROOT=$ROOT/data/creatinine_ward_v3_kdigo_scr
export CREATININE_WARD_V3_OUTPUT=$ROOT/output_creatinine_ward_v3_downstream
export QWEN_PATH=/model/bitahub-model/pice35408784b54431987c4d13c457b9cd/Qwen3-8B
export HF_HOME=$ROOT/hf
export NPROC_PER_NODE=4

source "$ROOT/venvs/medin/bin/activate"
cd "$REPO"
chmod +x scripts/bitahub_creatinine_ward_v3_downstream.sh
bash scripts/bitahub_creatinine_ward_v3_downstream.sh preflight
```

Must print `V3_INPUT_PREFLIGHT_OK` and `V3_CONFIG_PREFLIGHT_OK`. Then:

```bash
mkdir -p "$CREATININE_WARD_V3_OUTPUT/logs"
nohup env \
  ROOT="$ROOT" \
  REPO="$REPO" \
  CREATININE_WARD_DATA_ROOT="$CREATININE_WARD_DATA_ROOT" \
  CREATININE_WARD_V2_OUTPUT="$CREATININE_WARD_V2_OUTPUT" \
  CREATININE_WARD_V3_DATA_ROOT="$CREATININE_WARD_V3_DATA_ROOT" \
  CREATININE_WARD_V3_OUTPUT="$CREATININE_WARD_V3_OUTPUT" \
  QWEN_PATH="$QWEN_PATH" \
  HF_HOME="$HF_HOME" \
  NPROC_PER_NODE=4 \
  PATH="$PATH" \
  VIRTUAL_ENV="${VIRTUAL_ENV:-}" \
  bash -lc 'bash scripts/bitahub_creatinine_ward_v3_downstream.sh embeddings && \
            bash scripts/bitahub_creatinine_ward_v3_downstream.sh probe && \
            touch "$CREATININE_WARD_V3_OUTPUT/CREATININE_WARD_V3_OK"' \
  >"$CREATININE_WARD_V3_OUTPUT/logs/driver.log" 2>&1 </dev/null &
echo $! | tee "$CREATININE_WARD_V3_OUTPUT/driver.pid"
disown
sleep 8
ps -fp "$(cat "$CREATININE_WARD_V3_OUTPUT/driver.pid")"
tail -n 40 "$CREATININE_WARD_V3_OUTPUT/logs/driver.log"
```

Success marker: `$CREATININE_WARD_V3_OUTPUT/CREATININE_WARD_V3_OK`.

Monitor:

```bash
export OUT=/clin-jepa-storage/medin_hosp_v1/output_creatinine_ward_v3_downstream
ps -fp "$(cat "$OUT/driver.pid" 2>/dev/null)" || true
ls -l "$OUT"/EMBEDDINGS_OK "$OUT"/EVAL_OK "$OUT"/CREATININE_WARD_V3_OK 2>/dev/null
tail -n 80 "$OUT/logs/driver.log"
tail -n 80 "$OUT/logs/probe_v3.log"
```
