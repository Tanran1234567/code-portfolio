"""Hospital (inpatient) scoped data pipeline for Clin-JEPA.

Mirrors the six-step ICU pipeline in :mod:`clin_jepa.data`, but the unit of
analysis is the hospital admission (``hadm_id``) anchored at ``admittime``
rather than the ICU stay anchored at ``icu_intime``.

Steps::

    step01_hosp_cohort          adult admissions with LOS >= 24h
    step02_hosp_observations    admission-wide state events  -> bucketed parquet
    step03_hosp_actions         admission-wide interventions -> bucketed parquet
    step04_hosp_discretization  6-hour binning               -> bucketed pickle
    step05_hosp_split           patient-level train/val/test
    step06_hosp_trajectories    state/action text + windows  -> .pt shards

The final ``.pt`` shards use the same ``clin_jepa_trajectories_v1`` schema as
the ICU pipeline, so :class:`clin_jepa.training.trajectory_dataset.TrajectoryWindowDataset`
consumes them unchanged. The ``stay_ids`` field of a shard therefore holds
``hadm_id`` values.

Two substantive departures from the ICU pipeline, both forced by measurement
rather than preference (see ``reports/hosp_pipeline/``):

1. **6-hour bins instead of 1-hour.** At 1h the hospital cohort leaves ~91%
   of bins with no observation, against 0.06% for the ICU cohort.
2. **Admission-wide sources instead of ICU concept tables.** chartevents
   covers 19.4% of admissions; the mimic-code lab concepts, re-derived on
   ``hadm_id``, cover ~92.8%.
"""
