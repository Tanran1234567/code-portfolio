"""Hospital mixed-resolution latent rollout evaluation.

This evaluator complements the paper's discrete ICU rollout with the hospital
time semantics required by MedIn: context and horizons are measured in real
hours, future observed actions are supplied at their actual timestamps, and
all uncertainty intervals are clustered at patient level.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F

from clin_jepa.evaluation.rollout import load_predictor, load_test_data
from clin_jepa.evaluation.select_hosp_checkpoint import _sha256_path
from clin_jepa.utils import load_config, setup_logging

logger = logging.getLogger(__name__)
REQUIRED_WINDOW_KEYS = {
    "subject_id",
    "hadm_id",
    "time_hours",
    "start_hours",
    "end_hours",
    "duration_hours",
    "is_icu",
    "has_measured",
    "action_present",
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, default=_jsonable, allow_nan=False)


def validate_mixed_schema(windows: list[dict]) -> None:
    """Fail early rather than silently evaluating mixed data as hourly ICU data."""
    if not windows:
        raise ValueError("No windows were loaded")
    missing = REQUIRED_WINDOW_KEYS - set(windows[0])
    if missing:
        raise ValueError(
            "Embedding cache does not contain mixed-time schema v2 fields "
            f"{sorted(missing)}. Re-run precompute_embeddings.py with the "
            "hospital config; legacy ICU caches are not valid here."
        )
    for idx, window in enumerate(windows):
        length = int(window["length"])
        for key in REQUIRED_WINDOW_KEYS - {"subject_id", "hadm_id"}:
            if len(window[key]) != length:
                raise ValueError(
                    f"Window {idx} field {key!r} has length {len(window[key])}, "
                    f"expected {length}"
                )


def context_end_index(
    window: dict,
    context_hours: float,
    min_context_fraction: float,
    tolerance_hours: float,
) -> int | None:
    """Return the final observed node inside the physical-time context."""
    start = float(torch.as_tensor(window["start_hours"])[0])
    end = torch.as_tensor(window["end_hours"], dtype=torch.float32)
    observed = torch.as_tensor(window["has_measured"], dtype=torch.bool)
    elapsed = end - start
    inside = torch.nonzero(
        (elapsed <= context_hours + tolerance_hours) & observed
    ).flatten()
    if inside.numel() == 0:
        return None
    idx = int(inside[-1])
    if float(elapsed[idx]) < context_hours * min_context_fraction:
        return None
    if idx + 1 >= int(window["length"]):
        return None
    return idx


def horizon_targets(
    window: dict,
    context_idx: int,
    horizons_hours: Iterable[float],
    tolerance_hours: float,
) -> dict[float, int]:
    """Match each requested physical horizon to the nearest future node."""
    end = torch.as_tensor(window["end_hours"], dtype=torch.float32)
    elapsed = end[context_idx + 1 :] - end[context_idx]
    targets: dict[float, int] = {}
    if elapsed.numel() == 0:
        return targets
    for horizon in horizons_hours:
        distance = (elapsed - float(horizon)).abs()
        rel_idx = int(distance.argmin())
        if float(distance[rel_idx]) <= tolerance_hours:
            targets[float(horizon)] = context_idx + 1 + rel_idx
    return targets


def transition_stratum(is_icu: torch.Tensor, context_idx: int, target_idx: int) -> str:
    """Classify the complete context-to-target segment, not just its endpoint."""
    segment = torch.as_tensor(is_icu, dtype=torch.bool)[: target_idx + 1]
    if bool(segment.all()):
        return "icu"
    if not bool(segment.any()):
        return "ward"
    return "mixed"


def _gather_window(window: dict, shard: dict, device: torch.device) -> dict:
    state_idx = torch.as_tensor(window["emb_indices_state"], dtype=torch.long)
    action_idx = torch.as_tensor(window["emb_indices_action"], dtype=torch.long)
    return {
        "z_states": shard["z_state_embeddings"][state_idx].to(device),
        "actions": shard["z_action_embeddings"][action_idx].to(device),
        "z_static": shard["z_statics"][int(window["static_idx"])].to(device),
        "time_hours": torch.as_tensor(
            window["time_hours"], dtype=torch.float32, device=device
        ),
        "end_hours": torch.as_tensor(
            window["end_hours"], dtype=torch.float32, device=device
        ),
        "duration_hours": torch.as_tensor(
            window["duration_hours"], dtype=torch.float32, device=device
        ),
    }


@torch.no_grad()
def rollout_window(
    model,
    tensors: dict,
    context_idx: int,
    target_indices: set[int],
    use_bf16: bool,
) -> tuple[list[dict], dict[int, dict[str, torch.Tensor]]]:
    """Autoregress through every actual node needed by the requested horizons."""
    z_states = tensors["z_states"]
    actions = tensors["actions"]
    z_static = tensors["z_static"].unsqueeze(0)
    time_hours = tensors["time_hours"]
    end_hours = tensors["end_hours"]
    duration_hours = tensors["duration_hours"]

    z_history = z_states[:context_idx].float()
    action_history = actions[:context_idx]
    z_current = z_states[context_idx].float()
    z_copy = z_current.clone()
    max_target = max(target_indices)
    steps: list[dict] = []
    selected: dict[int, dict[str, torch.Tensor]] = {}

    for current_idx in range(context_idx, max_target):
        z_sequence = torch.cat([z_history, z_current.unsqueeze(0)], dim=0)
        action_sequence = torch.cat(
            [action_history, actions[current_idx].unsqueeze(0)], dim=0
        )
        sequence_end = current_idx + 1
        lengths = torch.tensor([sequence_end], device=z_states.device)
        with torch.amp.autocast(
            "cuda", dtype=torch.bfloat16,
            enabled=use_bf16 and z_states.device.type == "cuda",
        ):
            predicted = model(
                z_sequence.unsqueeze(0),
                z_static,
                action_sequence.unsqueeze(0),
                lengths,
                return_last_only=True,
                time_hours=time_hours[:sequence_end].unsqueeze(0),
                duration_hours=duration_hours[:sequence_end].unsqueeze(0),
            )[0].float()

        target_idx = current_idx + 1
        actual = z_states[target_idx].float()
        model_l1 = float((predicted - actual).abs().sum())
        copy_l1 = float((z_copy - actual).abs().sum())
        cosine = float(1.0 - F.cosine_similarity(
            predicted.unsqueeze(0), actual.unsqueeze(0)
        )[0])
        copy_cosine = float(1.0 - F.cosine_similarity(
            z_copy.unsqueeze(0), actual.unsqueeze(0)
        )[0])
        steps.append({
            "target_index": target_idx,
            "elapsed_hours": float(end_hours[target_idx] - end_hours[context_idx]),
            "model_l1": model_l1,
            "copy_l1": copy_l1,
            "cosine_distance": cosine,
            "copy_cosine_distance": copy_cosine,
        })
        if target_idx in target_indices:
            selected[target_idx] = {
                "predicted": predicted.detach().cpu().half(),
                "actual": actual.detach().cpu().half(),
                "copy": z_copy.detach().cpu().half(),
            }
        z_history = torch.cat([z_history, z_current.unsqueeze(0)], dim=0)
        action_history = torch.cat(
            [action_history, actions[current_idx].unsqueeze(0)], dim=0
        )
        z_current = predicted

    return steps, selected


@torch.no_grad()
def rollout_entries(
    model,
    entries: list[dict],
    device: torch.device,
    use_bf16: bool,
    action_mode: str = "real",
    collect_embeddings: bool = False,
) -> tuple[list[dict], list[dict]]:
    """Batched rollout for windows sharing one context node count."""
    if not entries:
        return [], []
    context_idx = entries[0]["context_idx"]
    if any(entry["context_idx"] != context_idx for entry in entries):
        raise ValueError("rollout_entries requires a common context_idx")
    max_targets = [max(entry["targets"].values()) for entry in entries]
    t_max = max(max_targets) + 1
    batch_size = len(entries)
    dim = int(
        entries[0]["shard"]["z_state_embeddings"].shape[-1]
    )

    states = torch.zeros(batch_size, t_max, dim, dtype=torch.float16)
    actions = torch.zeros_like(states)
    statics = torch.zeros(batch_size, dim, dtype=torch.float16)
    times = torch.zeros(batch_size, t_max, dtype=torch.float32)
    durations = torch.zeros(batch_size, t_max, dtype=torch.float32)
    for b, entry in enumerate(entries):
        window = entry["window"]
        shard = entry["shard"]
        limit = max_targets[b] + 1
        state_idx = torch.as_tensor(
            window["emb_indices_state"][:limit], dtype=torch.long
        )
        action_idx = torch.as_tensor(
            window["emb_indices_action"][:limit], dtype=torch.long
        )
        states[b, :limit] = shard["z_state_embeddings"][state_idx]
        actions[b, :limit] = shard["z_action_embeddings"][action_idx]
        statics[b] = shard["z_statics"][int(window["static_idx"])]
        times[b, :limit] = torch.as_tensor(
            window["time_hours"][:limit], dtype=torch.float32
        )
        durations[b, :limit] = torch.as_tensor(
            window["duration_hours"][:limit], dtype=torch.float32
        )

    states = states.to(device)
    actions = actions.to(device)
    if action_mode == "removed":
        actions.zero_()
    elif action_mode == "shuffled":
        # Deterministic within-trajectory circular shuffle. It preserves action
        # marginal scale while destroying alignment with the clinical state.
        actions = torch.roll(actions, shifts=1, dims=1)
    elif action_mode != "real":
        raise ValueError(f"unknown action_mode={action_mode!r}")
    statics = statics.to(device)
    times = times.to(device)
    durations = durations.to(device)
    z_history = states[:, :context_idx, :].float()
    action_history = actions[:, :context_idx, :]
    z_current = states[:, context_idx, :].float()
    z_copy = z_current.clone()
    steps_by_window: list[list[dict]] = [[] for _ in entries]
    embeddings_by_window: list[dict[str, dict[str, torch.Tensor]]] = [
        {} for _ in entries
    ]
    stats_by_window: list[dict[int, tuple[float, float]]] = [
        {} for _ in entries
    ]

    for current_idx in range(context_idx, max(max_targets)):
        sequence_end = current_idx + 1
        z_sequence = torch.cat([z_history, z_current.unsqueeze(1)], dim=1)
        action_sequence = torch.cat(
            [action_history, actions[:, current_idx, :].unsqueeze(1)], dim=1
        )
        lengths = torch.full(
            (batch_size,), sequence_end, dtype=torch.long, device=device
        )
        with torch.amp.autocast(
            "cuda", dtype=torch.bfloat16,
            enabled=use_bf16 and device.type == "cuda",
        ):
            predicted = model(
                z_sequence,
                statics,
                action_sequence,
                lengths,
                return_last_only=True,
                time_hours=times[:, :sequence_end],
                duration_hours=durations[:, :sequence_end],
            ).float()
        target_idx = current_idx + 1
        actual = states[:, target_idx, :].float()
        model_l1 = (predicted - actual).abs().sum(dim=-1)
        copy_l1 = (z_copy - actual).abs().sum(dim=-1)
        cosine = 1.0 - F.cosine_similarity(predicted, actual)
        copy_cosine = 1.0 - F.cosine_similarity(z_copy, actual)

        for b, entry in enumerate(entries):
            if target_idx > max_targets[b]:
                continue
            window = entry["window"]
            step = {
                "target_index": target_idx,
                "elapsed_hours": float(
                    window["end_hours"][target_idx]
                    - window["end_hours"][context_idx]
                ),
                "model_l1": float(model_l1[b]),
                "copy_l1": float(copy_l1[b]),
                "cosine_distance": float(cosine[b]),
                "copy_cosine_distance": float(copy_cosine[b]),
            }
            steps_by_window[b].append(step)
            for horizon, requested_idx in entry["targets"].items():
                if target_idx == requested_idx:
                    stats_by_window[b][target_idx] = (
                        float(predicted[b].float().std()),
                        float(actual[b].float().std()),
                    )
                    if collect_embeddings:
                        embeddings_by_window[b][str(horizon)] = {
                            "predicted": predicted[b].detach().cpu().half(),
                            "actual": actual[b].detach().cpu().half(),
                            "copy": z_copy[b].detach().cpu().half(),
                        }

        z_history = torch.cat([z_history, z_current.unsqueeze(1)], dim=1)
        action_history = torch.cat(
            [action_history, actions[:, current_idx, :].unsqueeze(1)], dim=1
        )
        z_current = predicted

    records: list[dict] = []
    trajectories: list[dict] = []
    for b, entry in enumerate(entries):
        window = entry["window"]
        by_target = {
            step["target_index"]: step for step in steps_by_window[b]
        }
        for horizon, target_idx in entry["targets"].items():
            step = by_target[target_idx]
            predicted_std, actual_std = stats_by_window[b][target_idx]
            records.append({
                "window_index": entry["window_index"],
                "subject_id": int(window["subject_id"]),
                "hadm_id": int(window["hadm_id"]),
                "horizon_hours": horizon,
                "actual_elapsed_hours": step["elapsed_hours"],
                "target_index": target_idx,
                "stratum": transition_stratum(
                    window["is_icu"], context_idx, target_idx
                ),
                "target_is_icu": bool(window["is_icu"][target_idx]),
                "target_has_measured": bool(window["has_measured"][target_idx]),
                "last_action_present": bool(window["action_present"][target_idx - 1]),
                "any_action_in_rollout": bool(
                    torch.as_tensor(
                        window["action_present"][context_idx:target_idx],
                        dtype=torch.bool,
                    ).any()
                ),
                "model_l1": step["model_l1"],
                "copy_l1": step["copy_l1"],
                "cosine_distance": step["cosine_distance"],
                "copy_cosine_distance": step["copy_cosine_distance"],
                "predicted_z_std": predicted_std,
                "actual_z_std": actual_std,
                **({"gap_length": int(entry["gap_length"])} if "gap_length" in entry else {}),
                **({"mask_rate": float(entry["mask_rate"])} if "mask_rate" in entry else {}),
            })
        if collect_embeddings:
            trajectories.append({
                "window_index": entry["window_index"],
                "subject_id": int(window["subject_id"]),
                "hadm_id": int(window["hadm_id"]),
                "context_end_index": context_idx,
                "context_end_hour": float(window["end_hours"][context_idx]),
                "steps": steps_by_window[b],
                "horizon_embeddings": embeddings_by_window[b],
            })
    return records, trajectories


def _execute_entry_groups(
    model,
    grouped_entries: dict[int, list[dict]],
    device: torch.device,
    use_bf16: bool,
    batch_size: int,
    action_mode: str = "real",
    max_trajectories: int = 0,
) -> tuple[list[dict], list[dict]]:
    records: list[dict] = []
    trajectories: list[dict] = []
    for context_idx, entries in sorted(grouped_entries.items()):
        for start in range(0, len(entries), batch_size):
            collect = len(trajectories) < max_trajectories
            batch_records, batch_trajectories = rollout_entries(
                model, entries[start:start + batch_size], device, use_bf16,
                action_mode=action_mode,
                collect_embeddings=collect,
            )
            records.extend(batch_records)
            if collect:
                trajectories.extend(
                    batch_trajectories[: max_trajectories - len(trajectories)]
                )
    return records, trajectories


def _gap_recovery_entries(
    windows: list[dict], shards: list[dict], min_context: int = 2,
) -> dict[int, list[dict]]:
    """First natural gap per window, supervised at the next observed endpoint."""
    grouped: dict[int, list[dict]] = defaultdict(list)
    for window_index, window in enumerate(windows):
        observed = torch.as_tensor(window["has_measured"], dtype=torch.bool)
        length = int(window["length"])
        i = min_context
        while i < length and bool(observed[i]):
            i += 1
        if i >= length or not bool(observed[:i].all()):
            continue
        j = i
        while j < length and not bool(observed[j]):
            j += 1
        if j >= length:
            continue
        context_idx = i - 1
        elapsed = float(window["end_hours"][j] - window["end_hours"][context_idx])
        grouped[context_idx].append({
            "window_index": window_index,
            "window": window,
            "shard": shards[int(window["_shard_idx"])],
            "context_idx": context_idx,
            "targets": {elapsed: j},
            "gap_length": j - i,
        })
    return grouped


def _mask_recovery_entries(
    windows: list[dict],
    shards: list[dict],
    rate: float,
    max_windows: int,
    min_context: int = 2,
) -> dict[int, list[dict]]:
    """Deterministic held-out observed targets for artificial-mask recovery."""
    grouped: dict[int, list[dict]] = defaultdict(list)
    used = 0
    for window_index, window in enumerate(windows):
        if used >= max_windows:
            break
        observed = torch.as_tensor(window["has_measured"], dtype=torch.bool)
        length = int(window["length"])
        if not bool(observed[:length].all()):
            continue
        candidates = [i for i in range(min_context, length - 1) if bool(observed[i])]
        if not candidates:
            continue
        n_pick = max(1, int(round(len(candidates) * rate)))
        # Evenly spaced deterministic choices make validation/test reproducible.
        pick_positions = np.linspace(0, len(candidates) - 1, n_pick, dtype=int)
        for pos in sorted(set(int(v) for v in pick_positions)):
            target_idx = candidates[pos]
            context_idx = target_idx - 1
            elapsed = float(
                window["end_hours"][target_idx] - window["end_hours"][context_idx]
            )
            grouped[context_idx].append({
                "window_index": window_index,
                "window": window,
                "shard": shards[int(window["_shard_idx"])],
                "context_idx": context_idx,
                "targets": {elapsed: target_idx},
                "mask_rate": rate,
            })
        used += 1
    return grouped


def _metric_summary(records: list[dict]) -> dict:
    if not records:
        return {"n": 0}
    model_l1 = np.asarray([r["model_l1"] for r in records], dtype=np.float64)
    copy_l1 = np.asarray([r["copy_l1"] for r in records], dtype=np.float64)
    cosine = np.asarray([r["cosine_distance"] for r in records], dtype=np.float64)
    copy_cosine = np.asarray(
        [r["copy_cosine_distance"] for r in records], dtype=np.float64
    )
    return {
        "n": len(records),
        "n_patients": len({int(r["subject_id"]) for r in records}),
        "model_mean_l1": float(model_l1.mean()),
        "copy_forward_mean_l1": float(copy_l1.mean()),
        "mase": float(model_l1.mean() / copy_l1.mean())
        if copy_l1.mean() > 0 else None,
        "model_mean_cosine_distance": float(cosine.mean()),
        "copy_forward_mean_cosine_distance": float(copy_cosine.mean()),
        "predicted_z_std": float(np.mean([r["predicted_z_std"] for r in records])),
        "actual_z_std": float(np.mean([r["actual_z_std"] for r in records])),
    }


def encoder_zstd_by_source(
    shards: list[dict], windows: list[dict], max_samples: int = 5000,
) -> dict[str, float | None]:
    """Mean per-feature std across observed encoder embeddings."""
    values: dict[str, list[torch.Tensor]] = {"all": [], "icu": [], "ward": []}
    counts = {key: 0 for key in values}
    for window in windows:
        shard = shards[int(window["_shard_idx"])]
        indices = torch.as_tensor(window["emb_indices_state"], dtype=torch.long)
        observed = torch.as_tensor(window["has_measured"], dtype=torch.bool)
        is_icu = torch.as_tensor(window["is_icu"], dtype=torch.bool)
        z = shard["z_state_embeddings"][indices].float()
        for source, mask in (
            ("all", observed),
            ("icu", observed & is_icu),
            ("ward", observed & ~is_icu),
        ):
            if counts[source] >= max_samples or not bool(mask.any()):
                continue
            take = z[mask][: max_samples - counts[source]]
            values[source].append(take)
            counts[source] += len(take)
        if all(counts[key] >= max_samples for key in counts):
            break
    result: dict[str, float | None] = {}
    for source, chunks in values.items():
        if not chunks or sum(len(v) for v in chunks) < 2:
            result[source] = None
        else:
            result[source] = float(torch.cat(chunks).std(dim=0).mean())
    return result


def aggregate_records(records: list[dict], horizons: list[float]) -> dict:
    result: dict[str, dict] = {}
    for horizon in horizons:
        h_records = [r for r in records if r["horizon_hours"] == horizon]
        result[str(int(horizon) if horizon.is_integer() else horizon)] = {
            "all": _metric_summary(h_records),
            **{
                stratum: _metric_summary(
                    [r for r in h_records if r["stratum"] == stratum]
                )
                for stratum in ("ward", "icu", "mixed")
            },
            "target_ward": _metric_summary(
                [r for r in h_records if not r["target_is_icu"]]
            ),
            "target_icu": _metric_summary(
                [r for r in h_records if r["target_is_icu"]]
            ),
            "target_measured": _metric_summary(
                [r for r in h_records if r["target_has_measured"]]
            ),
            "ward_measured": _metric_summary([
                r for r in h_records
                if r["stratum"] == "ward" and r["target_has_measured"]
            ]),
            "icu_measured": _metric_summary([
                r for r in h_records
                if r["stratum"] == "icu" and r["target_has_measured"]
            ]),
            "target_unmeasured": _metric_summary(
                [r for r in h_records if not r["target_has_measured"]]
            ),
            "rollout_with_action": _metric_summary(
                [r for r in h_records if r["any_action_in_rollout"]]
            ),
            "rollout_no_action": _metric_summary(
                [r for r in h_records if not r["any_action_in_rollout"]]
            ),
        }
    return result


def patient_bootstrap(
    records: list[dict],
    horizons: list[float],
    n_bootstrap: int,
    seed: int,
) -> dict:
    """Percentile CIs with subject_id as the resampling unit."""
    by_patient: dict[int, list[dict]] = defaultdict(list)
    for record in records:
        by_patient[int(record["subject_id"])].append(record)
    patient_ids = np.asarray(sorted(by_patient))
    if patient_ids.size < 2 or n_bootstrap <= 0:
        return {}
    rng = np.random.default_rng(seed)
    draws: dict[tuple[str, str, str], list[float]] = defaultdict(list)

    for _ in range(n_bootstrap):
        sampled = rng.choice(patient_ids, size=len(patient_ids), replace=True)
        sample_records = [
            record for patient_id in sampled for record in by_patient[int(patient_id)]
        ]
        summary = aggregate_records(sample_records, horizons)
        for horizon, strata in summary.items():
            for stratum, metrics in strata.items():
                for metric in (
                    "model_mean_l1",
                    "copy_forward_mean_l1",
                    "mase",
                    "model_mean_cosine_distance",
                ):
                    value = metrics.get(metric)
                    if value is not None:
                        draws[(horizon, stratum, metric)].append(float(value))

    intervals: dict[str, dict] = {}
    for (horizon, stratum, metric), values in draws.items():
        intervals.setdefault(horizon, {}).setdefault(stratum, {})[metric] = {
            "lower": float(np.percentile(values, 2.5)),
            "upper": float(np.percentile(values, 97.5)),
            "n_bootstrap": n_bootstrap,
            "n_valid_replicates": len(values),
        }
    return intervals


def _plot_case(case: dict, path: Path) -> None:
    """Plot actual/predicted latent PC1 and rollout error over real hours."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    horizons = sorted(
        (float(key), value) for key, value in case["horizon_embeddings"].items()
    )
    if not horizons:
        return
    actual = torch.stack([value["actual"].float() for _, value in horizons])
    predicted = torch.stack([value["predicted"].float() for _, value in horizons])
    joined = torch.cat([actual, predicted], dim=0)
    centered = joined - joined.mean(dim=0, keepdim=True)
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    pc1 = centered @ vh[0]
    hours = [hour for hour, _ in horizons]
    n = len(hours)
    step_by_target = {
        step["target_index"]: step for step in case["steps"]
    }
    target_indices = [
        next(
            step["target_index"] for step in case["steps"]
            if abs(step["elapsed_hours"] - hour) <= 0.51
        )
        for hour in hours
    ]
    errors = [step_by_target[idx]["model_l1"] for idx in target_indices]
    copy_errors = [step_by_target[idx]["copy_l1"] for idx in target_indices]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(hours, pc1[:n].numpy(), "o-", label="Actual")
    axes[0].plot(hours, pc1[n:].numpy(), "s--", label="Predicted")
    axes[0].set(xlabel="Real rollout hours", ylabel="Latent PC1")
    axes[0].legend()
    axes[1].plot(hours, errors, "o-", label="Model L1")
    axes[1].plot(hours, copy_errors, "s--", label="Copy-forward L1")
    axes[1].set(xlabel="Real rollout hours", ylabel="Latent L1")
    axes[1].legend()
    fig.suptitle(
        f"subject={case['subject_id']} hadm={case['hadm_id']} "
        f"window={case['window_index']}"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _case_payload(
    trajectories: list[dict], n_cases: int, output_dir: Path
) -> tuple[list[dict], list[dict]]:
    if not trajectories or n_cases <= 0:
        return [], []
    ordered = sorted(
        trajectories,
        key=lambda x: np.mean([step["model_l1"] for step in x["steps"]]),
    )
    middle = len(ordered) // 2
    selected = ordered[max(0, middle - n_cases // 2): middle + (n_cases + 1) // 2]
    selected += ordered[-n_cases:]
    summaries = []
    tensors = []
    plot_dir = output_dir / "case_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for case in selected:
        plot_path = plot_dir / (
            f"subject_{case['subject_id']}_hadm_{case['hadm_id']}"
            f"_window_{case['window_index']}.png"
        )
        _plot_case(case, plot_path)
        summary = {k: v for k, v in case.items() if k != "horizon_embeddings"}
        summary["trajectory_plot"] = str(plot_path)
        summaries.append(summary)
        tensors.append({
            "subject_id": case["subject_id"],
            "hadm_id": case["hadm_id"],
            "horizon_embeddings": case["horizon_embeddings"],
        })
    return summaries, tensors


def verify_test_selection(plan: str, manifest_path: str | None, plan_cfg: dict) -> dict:
    if not manifest_path:
        raise ValueError(
            "Test evaluation requires evaluation.selection_manifest. Select a "
            "checkpoint on full validation first; test must remain untouched."
        )
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Selection manifest not found: {path}")
    with path.open() as f:
        manifest = json.load(f)
    if manifest.get("selected_plan") != plan:
        raise ValueError(
            f"Selection manifest chose {manifest.get('selected_plan')!r}, "
            f"but --plan={plan!r}"
        )
    if manifest.get("test_has_been_read"):
        raise RuntimeError(
            "The selection manifest records a completed test evaluation. "
            "Refusing to read the one-shot test split again."
        )
    selected = manifest.get("selected_artifacts", {})
    checks = (
        ("predictor_sha256", Path(plan_cfg["predictor_checkpoint"])),
        ("encoder_sha256", Path(plan_cfg["encoder_checkpoint"])),
    )
    for key, path_to_check in checks:
        actual = _sha256_path(path_to_check.resolve())
        if selected.get(key) != actual:
            raise RuntimeError(
                f"Selected checkpoint changed after validation: {path_to_check} "
                f"manifest={selected.get(key)} actual={actual}"
            )
    return manifest


def evaluate(config: dict, plan: str, split: str, device: torch.device) -> dict:
    eval_cfg = config["evaluation"]
    plan_cfg = config["plans"][plan]
    if split == "test":
        verify_test_selection(plan, eval_cfg.get("selection_manifest"), plan_cfg)

    max_windows = eval_cfg.get("max_windows")
    complete_manifest = Path(plan_cfg["embedding_dir"]) / split / "_complete.json"
    if max_windows is None and not complete_manifest.exists():
        raise FileNotFoundError(
            f"Full {split} evaluation requires {complete_manifest}; rerun "
            "precompute_embeddings without shard partitioning to certify it."
        )
    shards, windows = load_test_data(
        plan_cfg["embedding_dir"], split, max_windows=max_windows
    )
    expected = eval_cfg.get("expected_windows", {}).get(split)
    if max_windows is None and expected is not None and len(windows) != int(expected):
        raise ValueError(
            f"Full {split} must contain exactly {expected} windows, "
            f"but embedding cache contains {len(windows)}"
        )
    validate_mixed_schema(windows)
    model = load_predictor(
        plan_cfg["predictor_checkpoint"],
        device,
        model_config=plan_cfg.get("model_config"),
    )
    if not (model.use_continuous_time or model.absolute_time_additive):
        logger.warning("Evaluating explicit no-time ablation")

    context_hours = float(eval_cfg["context_hours"])
    horizons = sorted(float(h) for h in eval_cfg["horizons_hours"])
    tolerance = float(eval_cfg.get("horizon_tolerance_hours", 0.51))
    min_context_fraction = float(eval_cfg.get("min_context_fraction", 0.75))
    use_bf16 = eval_cfg.get("precision", "bf16") == "bf16"
    coverage = {h: 0 for h in horizons}
    context_eligible = 0
    started = time.time()
    grouped_entries: dict[int, list[dict]] = defaultdict(list)
    for window_index, window in enumerate(windows):
        context_idx = context_end_index(
            window, context_hours, min_context_fraction, tolerance
        )
        if context_idx is None:
            continue
        context_eligible += 1
        targets = horizon_targets(window, context_idx, horizons, tolerance)
        if not targets:
            continue
        for horizon in targets:
            coverage[horizon] += 1
        grouped_entries[context_idx].append({
            "window_index": window_index,
            "window": window,
            "shard": shards[int(window["_shard_idx"])],
            "context_idx": context_idx,
            "targets": targets,
        })

    batch_size = int(eval_cfg.get("batch_size", 16))
    action_modes = list(eval_cfg.get("action_modes", ["real"]))
    if "real" not in action_modes:
        action_modes.insert(0, "real")
    records_by_action: dict[str, list[dict]] = {}
    trajectories: list[dict] = []
    for action_mode in action_modes:
        logger.info("Conditional rollout action mode=%s", action_mode)
        mode_records, mode_trajectories = _execute_entry_groups(
            model, grouped_entries, device, use_bf16, batch_size, action_mode,
            max_trajectories=(int(eval_cfg.get("n_cases", 5)) if action_mode == "real" else 0),
        )
        records_by_action[action_mode] = mode_records
        if action_mode == "real":
            trajectories = mode_trajectories

    records = records_by_action["real"]

    n_bootstrap = int(eval_cfg.get("bootstrap_replicates", 500))
    metrics = aggregate_records(records, horizons)
    encoder_zstd = encoder_zstd_by_source(
        shards, windows, int(eval_cfg.get("zstd_max_samples", 5000))
    )
    zstd_min = float(eval_cfg.get("zstd_min", 0.05))
    encoder_gate = {
        "threshold": zstd_min,
        "by_source": encoder_zstd,
        "passed": all(
            encoder_zstd.get(source) is not None
            and float(encoder_zstd[source]) >= zstd_min
            for source in ("icu", "ward")
        ),
    }
    rollout_drift: dict[str, dict] = {}
    for source, baseline_horizon in (("icu_measured", "1"), ("ward_measured", "24")):
        baseline = metrics.get(baseline_horizon, {}).get(source, {}).get("model_mean_l1")
        rollout_drift[source] = {}
        for horizon, strata in metrics.items():
            value = strata.get(source, {}).get("model_mean_l1")
            rollout_drift[source][horizon] = (
                float(value) / float(baseline)
                if value is not None and baseline not in (None, 0) else None
            )
    action_ablation = {
        mode: aggregate_records(mode_records, horizons)
        for mode, mode_records in records_by_action.items()
    }
    gap_records, _ = _execute_entry_groups(
        model, _gap_recovery_entries(windows, shards),
        device, use_bf16, batch_size, "real",
    )
    gap_recovery = {}
    for bucket in ("1", "2", "3", "4plus"):
        bucket_records = [
            r for r in gap_records if (
                (bucket == "1" and r.get("gap_length") == 1)
                or (bucket == "2" and r.get("gap_length") == 2)
                or (bucket == "3" and r.get("gap_length") == 3)
                or (bucket == "4plus" and r.get("gap_length", 0) >= 4)
            )
        ]
        gap_recovery[bucket] = {
            "all": _metric_summary(bucket_records),
            "icu": _metric_summary([r for r in bucket_records if r["stratum"] == "icu"]),
            "ward": _metric_summary([r for r in bucket_records if r["stratum"] == "ward"]),
        }
    mask_recovery = {}
    for rate in [float(v) for v in eval_cfg.get("mask_rates", [0.1, 0.2, 0.3])]:
        mask_records, _ = _execute_entry_groups(
            model,
            _mask_recovery_entries(
                windows, shards, rate,
                int(eval_cfg.get("mask_eval_max_windows", 2000)),
            ),
            device, use_bf16, batch_size, "real",
        )
        mask_recovery[str(rate)] = {
            "all": _metric_summary(mask_records),
            "icu": _metric_summary([r for r in mask_records if r["stratum"] == "icu"]),
            "ward": _metric_summary([r for r in mask_records if r["stratum"] == "ward"]),
        }
    intervals = patient_bootstrap(
        records, horizons, n_bootstrap, int(eval_cfg.get("bootstrap_seed", 42))
    )
    output_dir = Path(plan_cfg["output_dir"]) / split
    output_dir.mkdir(parents=True, exist_ok=True)
    case_summaries, case_tensors = _case_payload(
        trajectories, int(eval_cfg.get("n_cases", 5)), output_dir
    )
    summary = {
        "plan": plan,
        "split": split,
        "n_windows_loaded": len(windows),
        "max_windows": max_windows,
        "is_full_split": max_windows is None,
        "n_context_eligible": context_eligible,
        "context_hours": context_hours,
        "horizons_hours": horizons,
        "horizon_tolerance_hours": tolerance,
        "coverage": {
            str(int(h) if h.is_integer() else h): {
                "n": coverage[h],
                "fraction_of_loaded_windows": coverage[h] / len(windows),
                "fraction_of_context_eligible": (
                    coverage[h] / context_eligible if context_eligible else 0.0
                ),
            }
            for h in horizons
        },
        "metrics": metrics,
        "encoder_collapse_gate": encoder_gate,
        "normalized_rollout_drift": rollout_drift,
        "action_ablation": action_ablation,
        "artificial_mask_recovery": mask_recovery,
        "natural_gap_recovery": gap_recovery,
        "bootstrap_95ci": intervals,
        "elapsed_seconds": time.time() - started,
    }
    _write_json(output_dir / "summary.json", summary)
    _write_json(output_dir / "per_window_metrics.json", records)
    _write_json(output_dir / "case_trajectories.json", case_summaries)
    torch.save(case_tensors, output_dir / "case_embeddings.pt")
    if split == "test":
        manifest_path = Path(eval_cfg["selection_manifest"])
        with manifest_path.open() as f:
            manifest = json.load(f)
        manifest["test_has_been_read"] = True
        manifest["test_summary"] = str((output_dir / "summary.json").resolve())
        _write_json(manifest_path, manifest)
    logger.info("Saved hospital mixed evaluation to %s", output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Real-hour hospital mixed-resolution latent rollout evaluation"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--split", choices=["val", "test"], required=True)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument(
        "--max_windows", type=int, default=None,
        help="Smoke only. Omit for full validation/test.",
    )
    args = parser.parse_args()
    setup_logging()
    config = load_config(args.config)
    if args.plan not in config.get("plans", {}):
        raise KeyError(
            f"Unknown plan {args.plan!r}; available={list(config.get('plans', {}))}"
        )
    if args.max_windows is not None:
        config["evaluation"]["max_windows"] = args.max_windows
    device = torch.device(
        f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    )
    evaluate(config, args.plan, args.split, device)


if __name__ == "__main__":
    main()
