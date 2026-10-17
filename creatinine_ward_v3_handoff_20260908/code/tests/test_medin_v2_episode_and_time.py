import torch

from clin_jepa.data_hosp.episode_windows import (
    CANONICAL_NO_ACTION,
    clinical_text_only,
    dense_observed_segments,
    normalize_action_text,
    normalize_state_text,
    slice_episode_windows,
    split_care_episodes,
)
from clin_jepa.model.predictor import ACTransformerPredictor
from clin_jepa.training.pretrain_v2_losses import (
    sample_artificial_mask,
    sequence_normalized_l1,
)


def test_split_care_episodes_uses_episode_relative_model_time():
    starts = [0, 24, 48, 48, 49, 50, 51, 90]
    ends = [24, 48, 48, 49, 50, 51, 90, 114]
    # The 48-48 zero-width would be skipped by discretization; use clean nodes.
    starts = [0.0, 24.0, 48.0, 49.0, 50.0, 90.0]
    ends = [24.0, 48.0, 49.0, 50.0, 90.0, 114.0]
    is_icu = [False, False, True, True, True, False]
    eps = split_care_episodes(starts, ends, is_icu, [True] * 6)
    types = [e["episode_type"] for e in eps]
    assert types == ["WARD_PRE_ICU", "ICU", "WARD_POST_ICU"]
    assert abs(float(eps[0]["time_hours"][0]) - 24.0) < 1e-5
    assert abs(float(eps[1]["time_hours"][0]) - 1.0) < 1e-5
    assert abs(float(eps[2]["time_hours"][0]) - 24.0) < 1e-5
    assert abs(float(eps[1]["episode_time_hours"][0]) - 1.0) < 1e-5
    assert abs(float(eps[2]["episode_time_hours"][0]) - 24.0) < 1e-5
    assert eps[1]["source_type"] == "ICU"
    assert eps[0]["source_type"] == "WARD"


def test_dense_segments_and_windows():
    observed = [True, True, True, True, False, True, True, True, True, True]
    segs = dense_observed_segments(observed, min_len=4)
    assert segs == [(0, 4), (5, 5)]
    wins = slice_episode_windows(
        80, np_starts(), np_ends(), "ICU", t_max=72, min_len=3, icu_stride_hours=12,
    )
    assert wins[0] == (0, 72)
    assert wins[1][0] > 0


def test_v2_text_strips_machine_resolution_and_time():
    legacy = (
        "t=48h | Interval: [48h, 49h). Interval duration: 1h. "
        "Care setting resolution: ICU 1h. Measured observation available: yes. "
        "Observed groups: vitals. Hours since last measured observation: 0h. "
        "heart_rate: 90 bpm"
    )
    assert clinical_text_only(legacy) == "heart_rate: 90 bpm"
    assert clinical_text_only("t=24h | No actions recorded.") == "No actions recorded."
    assert normalize_state_text(legacy, True) == "Patient state:\nheart_rate: 90 bpm"
    no_action, present = normalize_action_text("t=24h | No actions recorded.")
    assert no_action == CANONICAL_NO_ACTION
    assert present is False
    real_action, present = normalize_action_text("t=24h | Norepinephrine: 0.08")
    assert real_action.startswith("Clinical action:\nNorepinephrine")
    assert present is True


def np_starts():
    import numpy as np
    return np.arange(80, dtype=np.float32)


def np_ends():
    import numpy as np
    return np.arange(80, dtype=np.float32) + 1


def test_absolute_time_additive_zero_init_and_pure_latent():
    model = ACTransformerPredictor(
        state_dim=8, action_dim=8, static_dim=8, hidden_dim=16,
        num_layers=1, num_heads=2, ffn_dim=32, max_timesteps=4,
        absolute_time_additive=True, use_continuous_time=False,
    )
    assert torch.equal(model.abs_time_proj[-1].weight, torch.zeros_like(model.abs_time_proj[-1].weight))
    B, T, D = 2, 4, 8
    z_s = torch.randn(B, T, D)
    z_a = torch.randn(B, T, D)
    z_d = torch.randn(B, D)
    lengths = torch.tensor([4, 3])
    t = torch.tensor([[1.0, 2.0, 3.0, 4.0], [24.0, 48.0, 72.0, 96.0]])
    out = model(z_s, z_d, z_a, lengths, time_hours=t)
    assert out.shape == (B, T - 1, D)


def test_residual_predictor_starts_as_exact_identity():
    model = ACTransformerPredictor(
        state_dim=8, action_dim=8, static_dim=8, hidden_dim=16,
        num_layers=1, num_heads=2, ffn_dim=32, max_timesteps=4,
        prediction_mode="residual", absolute_time_additive=True,
    ).eval()
    assert torch.equal(
        model.output_proj.weight, torch.zeros_like(model.output_proj.weight)
    )
    z_s = torch.randn(2, 4, 8)
    z_a = torch.randn(2, 4, 8)
    z_d = torch.randn(2, 8)
    lengths = torch.tensor([4, 4])
    state_t = torch.tensor([[24.0, 48.0, 72.0, 96.0]]).repeat(2, 1)
    action_t = torch.tensor([[48.0, 72.0, 96.0, 96.0]]).repeat(2, 1)
    out = model(
        z_s, z_d, z_a, lengths,
        time_hours=state_t,
        action_time_hours=action_t,
    )
    assert torch.equal(out, z_s[:, :-1, :].float())


def test_sequence_normalized_l1_equalizes_long_and_short():
    preds = torch.zeros(2, 4, 3)
    targets = torch.zeros(2, 4, 3)
    targets[0] = 1.0
    targets[1, :2] = 1.0
    mask = torch.tensor([
        [True, True, True, True],
        [True, True, False, False],
    ])
    loss = sequence_normalized_l1(preds, targets, mask)
    # per-seq mean L1 is 3 for both sequences (sum over 3 dims)
    assert abs(float(loss) - 3.0) < 1e-5


def test_mask_skips_first_two_and_terminal():
    observed = torch.tensor([[True, True, True, True, True]])
    lengths = torch.tensor([5])
    torch.manual_seed(0)
    art = sample_artificial_mask(observed, lengths, rate=1.0, min_context=2)
    assert not bool(art[0, 0])
    assert not bool(art[0, 1])
    assert not bool(art[0, 4])
    assert bool(art[0, 2:4].any())


def test_source_balanced_sampler_is_distributed_and_deterministic():
    from clin_jepa.training.source_balanced_sampler import SourceBalancedSampler

    sources = ["ICU"] * 3 + ["WARD"] * 9
    rank0 = list(SourceBalancedSampler(
        sources, icu_frac=0.5, seed=7, num_replicas=2, rank=0,
    ))
    rank1 = list(SourceBalancedSampler(
        sources, icu_frac=0.5, seed=7, num_replicas=2, rank=1,
    ))
    merged = rank0 + rank1
    icu_fraction = sum(sources[i] == "ICU" for i in merged) / len(merged)
    assert len(rank0) == len(rank1)
    assert abs(icu_fraction - 0.5) < 0.11
