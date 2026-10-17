#!/usr/bin/env python3
"""Fast runtime gate before loading Qwen or starting distributed SFT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from clin_jepa.data_hosp.episode_windows import CANONICAL_NO_ACTION
from clin_jepa.model.predictor import ACTransformerPredictor
from clin_jepa.training.pretrain_clin_jepa_v2 import (
    v2_gap_loss,
    v2_teacher_forced_loss,
)
from clin_jepa.training.trajectory_dataset import TrajectoryWindowDataset


def _tiny_encoding(observed: torch.Tensor) -> dict:
    batch, timesteps = observed.shape
    dim = 8
    return {
        "z_states": torch.randn(batch, timesteps, dim),
        "z_actions": torch.randn(batch, timesteps, dim),
        "z_static": torch.randn(batch, dim),
        "lengths": torch.full((batch,), timesteps, dtype=torch.long),
        "padding_mask": torch.ones(batch, timesteps, dtype=torch.bool),
        "time_hours": torch.arange(1, timesteps + 1).float().repeat(batch, 1),
        "duration_hours": torch.ones(batch, timesteps),
        "state_observed": observed,
    }


def _tiny_predictor() -> ACTransformerPredictor:
    return ACTransformerPredictor(
        state_dim=8, action_dim=8, static_dim=8, hidden_dim=16,
        num_layers=1, num_heads=2, ffn_dim=32, max_timesteps=8,
        ffn_dropout=0.0, attn_dropout=0.0,
        absolute_time_additive=True, use_continuous_time=False,
    )


def _assert_backward(loss: torch.Tensor, model: torch.nn.Module, name: str) -> None:
    if not bool(torch.isfinite(loss)) or float(loss.detach()) <= 0:
        raise AssertionError(f"{name} loss invalid: {loss}")
    loss.backward()
    if not any(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
        raise AssertionError(f"{name} produced no finite predictor gradients")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    args = parser.parse_args()
    root = Path(args.data_root)
    verify = json.loads((root / "verify_v2_report.json").read_text())
    if verify.get("schema_version") != "medin_v03_episode_time_no_action_verify_v2":
        raise AssertionError(f"wrong verify schema: {verify.get('schema_version')}")
    quadrants = verify["trajectories_episode"]["train"]["quadrants"]
    no_action_nodes = int(quadrants.get("S1A0", 0)) + int(quadrants.get("S0A0", 0))

    episode_dir = root / "trajectories_episode" / "train"
    episode_files = sorted(episode_dir.glob("*.pt"))
    markers = sorted(episode_dir.glob("*.migration_v4_complete"))
    if len(markers) != len(episode_files):
        raise AssertionError(
            f"migration markers {len(markers)} != episode shards {len(episode_files)}"
        )
    sample = torch.load(episode_files[0], map_location="cpu", weights_only=False)
    ph = sample["per_hour"]
    state = torch.as_tensor(ph["state_observed"], dtype=torch.bool)
    action = torch.as_tensor(ph["action_present"], dtype=torch.bool)
    for idx in torch.where(~action)[0][:10].tolist():
        if ph["action_texts"][idx] != CANONICAL_NO_ACTION:
            raise AssertionError(f"noncanonical NO_ACTION at sample node {idx}")
    if not bool(state.any()):
        raise AssertionError("sample shard contains no observed state")

    lazy = TrajectoryWindowDataset(
        root / "trajectories_dense", split="train", max_windows=8,
        min_length=3, shard_cache_size=1,
    )
    _ = lazy[0]
    if len(lazy._shard_cache) > 1:
        raise AssertionError("lazy trajectory cache exceeded configured size")

    torch.manual_seed(7)
    dense = _tiny_encoding(torch.ones(2, 5, dtype=torch.bool))
    model = _tiny_predictor()
    mask_loss = v2_teacher_forced_loss(
        model, dense, None, mask_rate=0.5, lambda_mask=1.0,
        min_context=2, native=False,
    )
    _assert_backward(mask_loss, model, "artificial-mask")

    gap = _tiny_encoding(torch.tensor([[True, True, False, False, True]]))
    model = _tiny_predictor()
    gap_loss = v2_gap_loss(
        model, gap, None, min_context=2,
        max_gap_steps=8, max_gaps_per_sequence=2,
    )
    _assert_backward(gap_loss, model, "natural-gap")

    print(json.dumps({
        "status": "PREFLIGHT_OK",
        "quadrants": quadrants,
        "no_action_nodes": no_action_nodes,
        "sample_episode_shard": episode_files[0].name,
        "mask_loss": float(mask_loss.detach()),
        "gap_loss": float(gap_loss.detach()),
    }, indent=2))


if __name__ == "__main__":
    main()
