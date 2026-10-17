import torch

from clin_jepa.evaluation.evaluate_hosp_mixed import (
    aggregate_records,
    context_end_index,
    horizon_targets,
    patient_bootstrap,
    rollout_entries,
    transition_stratum,
)
from clin_jepa.model.predictor import ACTransformerPredictor


def _window():
    return {
        "length": 4,
        "subject_id": 10,
        "hadm_id": 20,
        "time_hours": torch.tensor([1.0, 2.0, 3.0, 4.0]),
        "start_hours": torch.tensor([48.0, 49.0, 50.0, 51.0]),
        "end_hours": torch.tensor([49.0, 50.0, 51.0, 52.0]),
        "duration_hours": torch.tensor([1.0, 1.0, 1.0, 1.0]),
        "is_icu": torch.tensor([True, True, True, True]),
        "has_measured": torch.tensor([True, True, False, True]),
        "action_present": torch.tensor([False, True, True, False]),
        "emb_indices_state": torch.arange(4),
        "emb_indices_action": torch.arange(4),
        "static_idx": 0,
    }


def test_context_horizons_and_transition_strata_use_real_hours():
    window = _window()
    context_idx = context_end_index(window, 1.0, 0.75, 0.01)
    assert context_idx == 0
    assert horizon_targets(window, context_idx, [1.0, 2.0, 6.0], 0.01) == {
        1.0: 1,
        2.0: 2,
    }
    assert transition_stratum(window["is_icu"], 0, 2) == "icu"


class _IncrementModel(torch.nn.Module):
    use_continuous_time = True

    def forward(
        self,
        z_states,
        z_static,
        z_actions,
        lengths,
        return_last_only,
        time_hours,
        duration_hours,
    ):
        assert time_hours.shape[1] == z_states.shape[1]
        assert duration_hours.shape == time_hours.shape
        return z_states[:, -1] + 1.0


def test_batched_rollout_passes_time_and_beats_copy_forward():
    window = _window()
    shard = {
        "z_state_embeddings": torch.arange(4).view(4, 1).repeat(1, 4).half(),
        "z_action_embeddings": torch.zeros(4, 4).half(),
        "z_statics": torch.zeros(1, 4).half(),
    }
    entries = [{
        "window_index": 0,
        "window": window,
        "shard": shard,
        "context_idx": 0,
        "targets": {1.0: 1, 2.0: 2},
    }]
    records, trajectories = rollout_entries(
        _IncrementModel(), entries, torch.device("cpu"), use_bf16=False,
        collect_embeddings=True,
    )
    assert len(records) == 2
    assert all(record["model_l1"] == 0.0 for record in records)
    assert all(record["copy_l1"] > 0.0 for record in records)
    assert trajectories[0]["steps"][-1]["elapsed_hours"] == 2.0


def test_patient_bootstrap_and_stratified_aggregation():
    records = []
    for patient_id, error in ((1, 1.0), (2, 3.0), (3, 5.0)):
        records.append({
            "subject_id": patient_id,
            "horizon_hours": 24.0,
            "stratum": "ward",
            "target_is_icu": False,
            "target_has_measured": True,
            "model_l1": error,
            "copy_l1": error * 2,
            "cosine_distance": error / 10,
            "copy_cosine_distance": error / 5,
            "predicted_z_std": 1.0,
            "actual_z_std": 1.1,
            "any_action_in_rollout": patient_id % 2 == 0,
            "last_action_present": patient_id % 2 == 0,
        })
    summary = aggregate_records(records, [24.0])
    assert summary["24"]["all"]["mase"] == 0.5
    assert summary["24"]["ward"]["n_patients"] == 3
    intervals = patient_bootstrap(records, [24.0], 20, seed=42)
    assert intervals["24"]["all"]["mase"]["lower"] == 0.5
    assert intervals["24"]["all"]["mase"]["upper"] == 0.5


def test_continuous_predictor_refuses_missing_time():
    model = ACTransformerPredictor(
        state_dim=4,
        action_dim=4,
        static_dim=4,
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
        ffn_dim=16,
        max_timesteps=4,
        use_continuous_time=True,
    )
    try:
        model(
            torch.zeros(1, 2, 4),
            torch.zeros(1, 4),
            torch.zeros(1, 2, 4),
            torch.tensor([2]),
        )
    except ValueError as exc:
        assert "time_hours is required" in str(exc)
    else:
        raise AssertionError("continuous-time predictor accepted missing time")
