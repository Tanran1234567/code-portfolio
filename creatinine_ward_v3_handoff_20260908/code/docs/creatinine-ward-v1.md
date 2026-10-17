# Creatinine Ward v1: ICU-free encoder and rollout validation

This experiment is isolated from the mixed Ward/ICU and Grounding-Ward v4
pipelines. It validates whether the shared Qwen encoder preserves a simple
piece of information explicitly present in a 24-hour state text, then tests
whether the AC Transformer carries that information through one- and two-step
native rollout.

## Frozen cohort semantics

- Exactly 3000 unique patients and 3000 unique admissions.
- Patient split: train 2100, validation 450, test 450.
- One deterministically selected admission per patient.
- The admission has no row in `mimiciv_icu.icustays`.
- The admission has no transfer careunit matching `ICU|INTENSIVE CARE`.
- Age 18-119 years.
- Four to 30 complete 24-hour bins (modeled LOS 96-720 hours).
- The final partial interval is discarded (`floor(LOS / 24h)`).
- Every modeled bin contains at least one whitelisted numeric lab observation.
- Every selected admission has at least one creatinine measurement inside the
  modeled complete-bin interval. Never-measured patients are not imported from
  outside the agreed 50,914-person source pool.
- Within each split, approximately 33% have at least one creatinine >1.5 mg/dL
  and 67% are measured but never exceed 1.5 mg/dL.
- Approximately 22% of patients in every split have at least one 24-hour bin
  without creatinine; the remaining 78% have creatinine in every modeled bin.
- Sampling crosses HIGH status with missing-bin coverage, then ranks by
  creatinine measurement count inside each of the four cells. This preserves
  node-level `NOT_MEASURED` support without adding a different patient cohort.

This is an intentionally enriched representation-learning cohort. Its class
prevalence is not an estimate of prevalence in the full MIMIC-IV inpatient
population.

## Node and label semantics

The first node covers `[admittime, admittime + 24h)`. Model time is the bucket
end, so nodes carry `time_hours = 24, 48, 72, ...`. Machine time is removed
from state and action text before Qwen encoding. The predictor receives time
only through the 4096-dimensional additive absolute-time embedding.

The continuous label `label_creatinine` is the last numeric creatinine value
retained in that bin. The three-class label is:

| id | name | definition |
|---:|---|---|
| 0 | HIGH | measured and last creatinine > 1.5 mg/dL |
| 1 | NOT_HIGH | measured and last creatinine <= 1.5 mg/dL |
| 2 | NOT_MEASURED | no valid creatinine in the bin |

The threshold is an operational representation label, not an AKI diagnosis.

## Trajectory semantics

Each admission produces one full trajectory window starting at node zero.
There is no sliding window and no stride duplication. The original five-phase
Clin-JEPA curriculum is used without artificial-mask or natural-gap phases:

Before JEPA, encoder SFT runs for at most 10 epochs. Validation is evaluated
throughout training; early stopping uses `val_combined_loss`, patience 4 and a
minimum relative improvement of 1%. JEPA and the SFT probe both consume
`encoder_sft/best`, never the unconditionally last `final` checkpoint.

1. predictor warmup: 104 optimizer steps;
2. online encoder/predictor co-training: 104 steps;
3. target alignment: 35 steps;
4. instantaneous hard sync;
5. predictor finalization: 157 steps.

Only phase 2 updates the online encoder LoRA. The predictor uses recorded
actions for the primary 24/48-hour rollout, so the reported result is
recorded-action conditional rather than an unconditional real-time forecast.

Historical note: v1 paired `z_t` with same-bin `u_t` when predicting
`z_(t+1)`. The corrective v2 retraining pairs `z_t` with target-interval
`u_(t+1)`, uses separate state/action times, residual identity initialization,
and a recalibrated EMA schedule. See
[`../BITAHUB_CREATININE_WARD_V2_RETRAIN.md`](../BITAHUB_CREATININE_WARD_V2_RETRAIN.md).

## Environment

```bash
export MIMIC_DUCKDB=/path/to/mimic4.db
export QWEN_PATH=/path/to/Qwen3-8B
export HF_HOME=/path/to/hf-cache
export CREATININE_WARD_DATA_ROOT=/path/to/creatinine_ward_v1_data
export CREATININE_WARD_OUTPUT=/path/to/creatinine_ward_v1_output
```

## Build and verify data

```bash
bash scripts/run_creatinine_ward_v1.sh
```

The data run stops unless all of the following hold: no ICU admissions, at
least one modeled creatinine measurement per admission, the fixed crossed
sampling quotas, one window per admission, complete 24-hour bins, no empty
state bin, model time at bucket end, no machine-time prefix in encoder text,
and exact agreement between the continuous creatinine value and the derived
three-class label.

The verification report also records per-split class node/patient counts, the
3x3 current-to-next class transition matrix, and the number of patients whose
longest consecutive `NOT_MEASURED` run is one, two, or at least three days.
These statistics bound what can legitimately be claimed about 24/48-hour
measurement-process prediction.

## Train and evaluate

```bash
bash scripts/run_creatinine_ward_training.sh
```

Resume from a later stage without rerunning earlier work:

```bash
FROM=jepa bash scripts/run_creatinine_ward_training.sh
FROM=embeddings bash scripts/run_creatinine_ward_training.sh
FROM=probes bash scripts/run_creatinine_ward_training.sh
```

The launcher precomputes raw-Qwen, SFT and final-JEPA embeddings. It trains a
patient-weighted multinomial linear probe for all three encoders. Only the
final JEPA encoder is paired with the retained predictor for native 24/48-hour
rollout.

Primary outputs:

```text
$CREATININE_WARD_DATA_ROOT/manifests/cohort_3000_summary.json
$CREATININE_WARD_DATA_ROOT/manifests/node_labels.parquet
$CREATININE_WARD_DATA_ROOT/verify_creatinine_ward_v1.json
$CREATININE_WARD_OUTPUT/evaluation/creatinine_probe_base/results.json
$CREATININE_WARD_OUTPUT/evaluation/creatinine_probe_sft/results.json
$CREATININE_WARD_OUTPUT/evaluation/creatinine_probe/results.json
```
