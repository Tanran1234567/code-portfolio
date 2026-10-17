# Creatinine Ward v2: BitaHub Predictor Retraining Guide

## Scope

This run reuses the verified v1 trajectories and `encoder_sft/best`. It does
not rerun MIMIC extraction, trajectory generation, encoder SFT, or the old
generic training smoke.

The corrected transition is:

```text
(state z_t, recorded actions u_(t+1) during the target 24h interval)
    -> predicted state z_hat_(t+1)
```

For two steps, real `u_(t+2)` is paired with predicted `z_hat_(t+1)` to produce
`z_hat_(t+2)`. State and action tokens receive separate absolute times. This is
an oracle/recorded target-action conditional forecast, not an action-free online
forecast.

Stability changes are deliberately scoped:

- residual prediction initialized as exact identity;
- 1600 optimizer steps with the original 6:6:2:9 phase ratio;
- EMA `tau=0.971`, so `0.971^139` matches `0.996^1024` residual lag;
- predictor LR `3e-4` and encoder LR `1e-5`;
- explicit effective batch `4 GPU x 11 x accumulation 2 = 88`;
- pre-hard-sync gap logging and separate final validation loss.

The old v1 output remains read-only input. v2 writes to a new directory.

## Expected time

The v1 400-step JEPA run took 213.4 minutes. At the same throughput:

| Stage | Expected |
|---|---:|
| Index prebuild | 5-20 min |
| 1600-step JEPA | 14-16 h |
| Three-split embedding cache | 1.5-3 h |
| Current probe and 24/48h rollout | 1-3 h |
| Total | 17-22 h |

Use a 24-hour instance reservation if possible.

## Environment

```bash
export CREATININE_WARD_DATA_ROOT=/data/projects/zhoujiale/creatinine_ward_v1
export CREATININE_WARD_OUTPUT=/clin-jepa-storage/medin_hosp_v1/output_creatinine_ward_v1
export CREATININE_WARD_V2_OUTPUT=/clin-jepa-storage/medin_hosp_v1/output_creatinine_ward_v2_action_aligned
export QWEN_PATH=/path/to/Qwen3-8B
export HF_HOME=/path/to/huggingface-cache
export NPROC_PER_NODE=4
```

`CREATININE_WARD_OUTPUT` points to v1 because v2 reads
`$CREATININE_WARD_OUTPUT/encoder_sft/best`. Never set the v1 and v2 output
variables to the same directory.

## Files to sync

```text
clin_jepa/model/predictor.py
clin_jepa/training/pretrain_clin_jepa.py
clin_jepa/evaluation/creatinine_probe.py
configs/train/pretrain_creatinine_ward_v2_action_aligned.yaml
configs/eval/precompute_creatinine_ward_v2_action_aligned.yaml
configs/eval/creatinine_probe_v2_action_aligned.yaml
scripts/bitahub_creatinine_ward_v2_retrain.sh
tests/test_medin_v2_episode_and_time.py
tests/test_creatinine_ward_v1.py
```

## Preflight without training smoke

The previous run already proved the data, SFT checkpoint, four-GPU Qwen path,
and batch 11 memory fit. The v2 launcher skips SFT and old training smokes and
uses a fast fail-closed preflight:

```bash
bash scripts/bitahub_creatinine_ward_v2_retrain.sh preflight
```

Required output:

```text
V2_INPUT_PREFLIGHT_OK
V2_CONFIG_PREFLIGHT_OK
```

If pytest is available, run the targeted alignment/identity tests. These are
not training smokes:

```bash
pytest -q tests/test_medin_v2_episode_and_time.py \
  tests/test_creatinine_ward_v1.py
```

## Full run

Foreground:

```bash
bash scripts/bitahub_creatinine_ward_v2_retrain.sh all
```

Recommended background launch:

```bash
mkdir -p "$CREATININE_WARD_V2_OUTPUT/logs"
nohup bash scripts/bitahub_creatinine_ward_v2_retrain.sh all \
  > "$CREATININE_WARD_V2_OUTPUT/logs/driver.log" 2>&1 &
echo $! > "$CREATININE_WARD_V2_OUTPUT/driver.pid"
```

Monitor with:

```bash
tail -f "$CREATININE_WARD_V2_OUTPUT/logs/driver.log"
tail -f "$CREATININE_WARD_V2_OUTPUT/logs/jepa_v2.log"
watch -n 30 nvidia-smi
```

Success markers are `INDEX_OK`, `JEPA_OK`, `EMBEDDINGS_OK`, `EVAL_OK`, and
`CREATININE_WARD_V2_OK`.

## Resume and stage reruns

```bash
RESUME=1 bash scripts/bitahub_creatinine_ward_v2_retrain.sh train
bash scripts/bitahub_creatinine_ward_v2_retrain.sh embeddings
bash scripts/bitahub_creatinine_ward_v2_retrain.sh eval
```

Do not set `RESUME=1` for a clean new v2 output directory.

## Training evidence

After JEPA, inspect:

```bash
python3 - <<'PY'
import json, os
root = os.environ["CREATININE_WARD_V2_OUTPUT"]
d = json.load(open(f"{root}/clin_jepa/training_meta.json"))
for key in [
    "total_steps", "best_val_loss", "final_val_loss", "action_alignment",
    "ema_tau_phase_2a", "ema_tau_phase_2b",
    "pre_sync_max_param_gap", "pre_sync_mean_param_gap",
]:
    print(key, d.get(key))
print("phase_steps", d["phase_steps"])
PY
```

Expected invariants:

```text
total_steps: 1600
action_alignment: target_interval
EMA tau: 0.971 / 0.971
phase steps: 417 / 417 / 139 / 627
final_val_loss: finite
pre_sync gaps: present and finite
```

## Evaluation and acceptance

Primary outputs:

```text
$CREATININE_WARD_V2_OUTPUT/evaluation/creatinine_probe/results.json
$CREATININE_WARD_V2_OUTPUT/evaluation/creatinine_probe/rollout_metrics.csv
```

The JSON must say `action_alignment=target_interval` and
`target-interval recorded-action conditional rollout`.

Compare against v1:

| Metric | v1 |
|---|---:|
| 24h predictor accuracy / macro-F1 | 0.4197 / 0.2819 |
| 24h persistence accuracy / macro-F1 | 0.9251 / 0.6910 |
| 48h predictor accuracy / macro-F1 | 0.4250 / 0.2963 |
| 48h persistence accuracy / macro-F1 | 0.9096 / 0.6494 |

Acceptance levels:

1. Minimal engineering success: v2 clearly exceeds v1 predictor at both
   horizons and balanced accuracy is no longer near random.
2. Useful predictor: v2 exceeds persistence in macro-F1.
3. Strong result: v2 exceeds persistence in both macro-F1 and accuracy with
   patient-bootstrap confidence intervals supporting the gain.

If v2 is worse than persistence, report it. Residual identity builds the strong
baseline into initialization, so a worse final model means the learned delta
damaged the representation.

## Preserve outputs

Before releasing the GPU, save `clin_jepa/training_meta.json`, the final
checkpoint, probe `results.json`, `rollout_metrics.csv`, and `logs/`. Do not
delete or overwrite v1; it remains the fixed SFT source and comparison run.
