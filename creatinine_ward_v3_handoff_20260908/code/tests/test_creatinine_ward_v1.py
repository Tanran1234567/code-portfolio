import numpy as np
import pandas as pd
import torch

from clin_jepa.data_hosp.creatinine_ward_v1 import (
    _candidate_sql,
    select_cohort,
    stable_u64,
)
from clin_jepa.data_hosp.step06_hosp_trajectories import (
    _strip_machine_time_prefix,
    slice_windows,
)
from clin_jepa.evaluation.creatinine_probe import patient_equal_weights
from clin_jepa.training.pretrain_clin_jepa import align_action_conditions


def test_full_admission_window_has_no_stride_duplicates():
    config = {"t_max": 30, "min_len": 4, "stride": 30}
    assert slice_windows(4, config) == [(0, 4)]
    assert slice_windows(17, config) == [(0, 17)]
    assert slice_windows(30, config) == [(0, 30)]


def test_candidate_sql_excludes_all_icu_and_uses_complete_bins():
    sql = _candidate_sql(18, 4, 30)
    assert "NOT EXISTS" in sql
    assert "mimiciv_icu.icustays" in sql
    assert "ICU|INTENSIVE CARE" in sql
    assert "floor(los_hours_raw / 24.0)" in sql
    assert "s.n_filled_bins = b.n_complete_bins" in sql
    assert "s.n_creatinine > 0" in sql


def test_machine_time_is_removed_from_encoder_text():
    assert _strip_machine_time_prefix("t=24h | Creatinine: 1.2 mg/dL") == (
        "Creatinine: 1.2 mg/dL"
    )
    assert _strip_machine_time_prefix("No observations recorded.") == (
        "No observations recorded."
    )


def test_patient_equal_weights_sum_to_patient_count():
    subjects = np.asarray([1, 1, 2, 3, 3, 3])
    weights = patient_equal_weights(subjects)
    assert np.isclose(weights.sum(), 3.0)
    assert np.allclose(weights, [0.5, 0.5, 1.0, 1 / 3, 1 / 3, 1 / 3])


def test_target_interval_action_alignment_shifts_one_bucket():
    actions = torch.tensor([[[10.0], [20.0], [30.0], [40.0]]])
    times = torch.tensor([[24.0, 48.0, 72.0, 96.0]])
    durations = torch.full_like(times, 24.0)
    aligned, action_times, action_durations = align_action_conditions(
        actions, times, durations, "target_interval"
    )
    assert aligned[:, :, 0].tolist() == [[20.0, 30.0, 40.0, 40.0]]
    assert action_times.tolist() == [[48.0, 72.0, 96.0, 96.0]]
    assert action_durations.tolist() == [[24.0, 24.0, 24.0, 24.0]]


def test_selection_is_exact_and_icu_free():
    rows = []
    # Enough deterministic-hash candidates for every 70/15/15 split.
    for subject_id in range(1, 20001):
        has_missing = subject_id % 4 in (0, 1)
        rows.append({
            "subject_id": subject_id,
            "hadm_id": 100000 + subject_id,
            "n_creatinine": subject_id % 12 + 1,
            "n_creatinine_bins": 3 if has_missing else 4,
            "n_complete_bins": 4,
            "ever_creatinine": True,
            "has_missing_creatinine_bin": has_missing,
            "creat_max": 2.0 if subject_id % 2 == 0 else 1.0,
            "has_icu": False,
        })
    candidates = pd.DataFrame(rows)
    cohort = select_cohort(
        candidates,
        seed=20260828,
        split_sizes={"train": 2100, "val": 450, "test": 450},
        high_fraction=0.33,
        any_missing_bin_fraction=0.22,
    )
    assert cohort.subject_id.nunique() == 3000
    assert cohort.hadm_id.nunique() == 3000
    assert cohort.split.value_counts().to_dict() == {
        "train": 2100, "val": 450, "test": 450
    }
    assert cohort.ever_creatinine.all()
    missing = cohort.groupby("split").has_missing_creatinine_bin.sum().astype(int).to_dict()
    assert missing == {"train": 462, "val": 99, "test": 99}
    strata = cohort.groupby(["split", "creatinine_sampling_stratum"]).size().to_dict()
    assert strata[("train", "high_with_missing_bin")] == 152
    assert strata[("train", "high_full_coverage")] == 541
    assert strata[("train", "not_high_with_missing_bin")] == 310
    assert strata[("train", "not_high_full_coverage")] == 1097
    assert stable_u64(20260828, 1) == stable_u64(20260828, 1)
