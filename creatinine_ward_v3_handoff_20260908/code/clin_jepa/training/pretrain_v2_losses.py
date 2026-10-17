"""v0.3 loss helpers: sequence-normalized L1, artificial mask, natural-gap rollout."""

from __future__ import annotations

import torch


def sequence_normalized_l1(
    preds: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Mean over sequences of (mean L1 over valid targets in that sequence)."""
    l1 = (preds - targets).abs().sum(dim=-1)
    valid = mask.float()
    # NaN * 0 is still NaN. Remove unsupervised/padded positions explicitly.
    safe_l1 = torch.where(mask, l1, torch.zeros_like(l1))
    per_seq = safe_l1.sum(dim=-1) / valid.sum(dim=-1).clamp(min=1.0)
    has = mask.any(dim=-1)
    if not bool(has.any()):
        return preds.new_zeros(())
    return per_seq[has].mean()


def sample_artificial_mask(
    state_observed: torch.Tensor,
    lengths: torch.Tensor,
    rate: float,
    min_context: int = 2,
) -> torch.Tensor:
    """True at interior observed positions that are artificially hidden."""
    mask = torch.zeros_like(state_observed)
    if rate <= 0:
        return mask
    B = state_observed.shape[0]
    for b in range(B):
        length = int(lengths[b].item())
        observed = torch.where(state_observed[b, :length])[0]
        if observed.numel() <= min_context + 1:
            continue
        candidates = observed[min_context:-1]
        n_pick = min(
            int(candidates.numel()),
            max(1, int(round(float(candidates.numel()) * rate))),
        )
        perm = torch.randperm(candidates.numel(), device=candidates.device)
        selected: list[int] = []
        for raw_idx in candidates[perm].tolist():
            idx = int(raw_idx)
            if all(abs(idx - previous) > 1 for previous in selected):
                selected.append(idx)
            if len(selected) >= n_pick:
                break
        if selected:
            mask[b, torch.tensor(selected, device=candidates.device)] = True
    return mask


def next_step_masks(
    state_observed: torch.Tensor,
    artificial_mask: torch.Tensor,
    padding_mask: torch.Tensor,
    min_context: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Masks on the *target* timestep (index 1..T-1).

    Returns:
        visible: observed, not artificially masked, after min_context
        masked: artificially masked observed targets
        gap: natural-empty targets (no direct loss; used only to identify gaps)
    """
    target_obs = state_observed[:, 1:]
    target_pad = padding_mask[:, 1:]
    target_art = artificial_mask[:, 1:]
    ctx = torch.zeros_like(target_obs)
    B, Tm = target_obs.shape
    for b in range(B):
        obs_idx = torch.where(state_observed[b])[0]
        if obs_idx.numel() > min_context:
            keep_from = int(obs_idx[min_context - 1].item())
            ctx[b, keep_from:] = True
    visible = target_obs & target_pad & (~target_art) & ctx
    masked = target_obs & target_pad & target_art & ctx
    gap = (~target_obs) & target_pad
    return visible, masked, gap
