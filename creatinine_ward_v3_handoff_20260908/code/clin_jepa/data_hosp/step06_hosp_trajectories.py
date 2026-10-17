"""Step 06: build per-bin state/action text and write trajectory shards.

ICU counterpart: :mod:`clin_jepa.data.step06_trajectories`. The output format is
deliberately identical — ``schema_version: "clin_jepa_trajectories_v1"``, the
same ``per_hour`` / ``per_hour_labels`` / ``per_stay`` / ``windows`` blocks — so
that :class:`clin_jepa.training.trajectory_dataset.TrajectoryWindowDataset` and
everything built on it consume these shards with no code change.

Field aliasing, since the schema names came from the ICU setting:

===========================  ==================================================
shard field                  hospital meaning
===========================  ==================================================
``per_hour.hours``           bin index; one bin is 6 hours, not 1
``per_stay.stay_ids``        ``hadm_id``
``per_stay.charlson_cis``    prior-admission diagnosis count (compatibility name;
                             no materialised Charlson view)
``per_stay.icu_mortalities`` in-hospital death (alias of hospital_mortalities)
``per_stay.sepsis3s``        suspected infection: antibiotic + culture within
                             24h. Sepsis-3 needs a SOFA trajectory and is
                             ICU-only, so this is the admission-wide analogue.
===========================  ==================================================

Hospital-specific outcomes that have no v1 slot (30/90-day mortality, ICU
transfer, 30-day readmission) are added in a ``per_stay_extra`` block, which
older readers ignore.

Usage:
    python -m clin_jepa.data_hosp.step06_hosp_trajectories
    python -m clin_jepa.data_hosp.step06_hosp_trajectories --buckets 0 1 --n_workers 2
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from clin_jepa.data_hosp import db
from clin_jepa.data_hosp.helpers.hosp_text_templates import (
    format_hosp_event,
    hosp_sort_key,
)

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "clin_jepa_trajectories_v1"

EMPTY_STATE = "No observations recorded."
EMPTY_ACTION = "No active interventions."

# Columns pulled from the cohort for per-stay metadata.
COHORT_COLS = [
    "hadm_id", "subject_id", "admission_age", "gender", "race",
    "hospital_expire_flag", "prolonged_stay",
    "suspected_infection", "mortality_30d", "mortality_90d",
    "has_icu", "readmission_30d", "los_hours",
]


# ── Text assembly ───────────────────────────────────────────────────────


def _priority_array(src: np.ndarray, var: np.ndarray) -> np.ndarray:
    """Within-bin sort priority for each row, memoised per (source, variable)."""
    cache: dict[tuple, int] = {}
    out = np.empty(len(src), dtype=np.int32)
    for i in range(len(src)):
        key = (src[i], var[i])
        p = cache.get(key)
        if p is None:
            p = hosp_sort_key(src[i], var[i])
            cache[key] = p
        out[i] = p
    return out


def build_texts(
    events: pd.DataFrame,
    empty_sentinel: str,
    bin_hours: int = 6,
    time_lookup: dict[tuple[int, int], float] | None = None,
) -> dict[int, dict[int, str]]:
    """Render one text line per (admission, bin) present in ``events``.

    A single linear pass over the sorted frame rather than a groupby per
    admission: at hospital scale this step handles ~197M rows, and the pass
    is what keeps it tractable.

    The leading ``t=Xh`` stamp uses wall-clock hours from admission
    (``step_idx * bin_hours``), not the bin index. With 6h bins, step 1
    therefore renders as ``t=6h``.

    Returns:
        ``{hadm_id: {step_idx: text}}``. Bins absent from the result carry no
        events and are filled with the sentinel by the caller.
    """
    if len(events) == 0:
        return {}

    events = events.assign(
        _p=_priority_array(
            events["source_table"].to_numpy(dtype=object),
            events["variable_name"].to_numpy(dtype=object),
        )
    ).sort_values(["hadm_id", "step_idx", "_p"], kind="stable")

    hadm = events["hadm_id"].to_numpy()
    step = events["step_idx"].to_numpy()
    src = events["source_table"].to_numpy(dtype=object)
    var = events["variable_name"].to_numpy(dtype=object)
    num = events["numeric_value"].to_numpy(dtype=np.float64)
    txt = events["text_value"].to_numpy(dtype=object)
    unit = events["unit"].to_numpy(dtype=object)

    out: dict[int, dict[int, str]] = defaultdict(dict)
    n = len(hadm)
    i = 0
    while i < n:
        h, s = hadm[i], step[i]
        j = i
        fragments: list[str] = []
        while j < n and hadm[j] == h and step[j] == s:
            frag = format_hosp_event(
                src[j], var[j],
                None if np.isnan(num[j]) else float(num[j]),
                txt[j], unit[j] if unit[j] is not None else "",
            )
            if frag:
                fragments.append(frag)
            j += 1
        body = ". ".join(fragments) if fragments else empty_sentinel
        wall_h = (
            time_lookup.get((int(h), int(s)), float(s) * float(bin_hours))
            if time_lookup is not None
            else float(s) * float(bin_hours)
        )
        out[int(h)][int(s)] = f"t={wall_h:g}h | {body}"
        i = j
    return out


def _fill_texts(
    per_stay_texts: dict[int, str],
    n_steps: int,
    empty_sentinel: str,
    bin_hours: int = 6,
    time_hours: list[float] | None = None,
) -> list[str]:
    """Dense per-bin text list, substituting the sentinel for empty bins."""
    return [
        per_stay_texts.get(
            s,
            (
                f"t={float(time_hours[s]):g}h | {empty_sentinel}"
                if time_hours is not None
                else f"t={s * int(bin_hours)}h | {empty_sentinel}"
            ),
        )
        for s in range(n_steps)
    ]


def _strip_machine_time_prefix(text: str) -> str:
    """Remove the leading ``t=...h |`` machine stamp from rendered text."""
    _prefix, separator, body = str(text).partition(" | ")
    return body if separator else str(text)


def _augment_state_texts(
    texts: list[str],
    starts: list[float],
    ends: list[float],
    durations: list[float],
    is_icu: list[bool],
    has_measured: list[bool],
    hours_since: list[float | None],
    observed_groups: list[list[str]],
) -> list[str]:
    """Add machine-derived time and missingness context to mixed nodes."""
    augmented: list[str] = []
    for i, text in enumerate(texts):
        _, separator, body = text.partition(" | ")
        body = body if separator else text
        since = hours_since[i]
        since_text = "never previously measured" if since is None or pd.isna(since) else f"{float(since):g}h"
        groups = observed_groups[i] or []
        groups_text = ", ".join(str(group) for group in groups) if groups else "none"
        context = (
            f"t={float(starts[i]):g}h | "
            f"Interval: [{float(starts[i]):g}h, {float(ends[i]):g}h). "
            f"Interval duration: {float(durations[i]):g}h. "
            f"Care setting resolution: {'ICU 1h' if is_icu[i] else 'ward 24h'}. "
            f"Measured observation available: {'yes' if has_measured[i] else 'no'}. "
            f"Observed groups: {groups_text}. "
            f"Hours since last measured observation: {since_text}. "
        )
        augmented.append(context + body)
    return augmented


# ── Labels ──────────────────────────────────────────────────────────────


def build_label_frames(
    obs: pd.DataFrame,
    label_targets: list[dict],
) -> dict[str, dict[int, dict[int, float]]]:
    """Per-bin numeric label values, keyed label -> hadm_id -> step_idx."""
    out: dict[str, dict[int, dict[int, float]]] = {}
    for target in label_targets:
        mask = (
            (obs["variable_name"] == target["variable_name"])
            & (obs["source_table"] == target["source_table"])
            & obs["numeric_value"].notna()
        )
        rows = obs.loc[mask, ["hadm_id", "step_idx", "numeric_value"]]
        if len(rows) == 0:
            out[target["name"]] = {}
            continue
        agg = "mean" if target.get("agg") == "mean" else "last"
        grouped = (
            rows.groupby(["hadm_id", "step_idx"])["numeric_value"].agg(agg)
        )
        nested: dict[int, dict[int, float]] = defaultdict(dict)
        for (h, s), v in grouped.items():
            nested[int(h)][int(s)] = float(v)
        out[target["name"]] = nested
    return out


# ── Windowing ───────────────────────────────────────────────────────────


def slice_windows(n_steps: int, config: dict) -> list[tuple[int, int]]:
    """Window positions for one admission.

    Identical rule to the ICU pipeline: slide when the stay is at least
    ``t_max`` long, otherwise emit the single available window.
    """
    t_max, min_len, stride = config["t_max"], config["min_len"], config["stride"]
    if n_steps < min_len:
        return []
    starts = (
        range(0, n_steps - t_max + 1, stride) if n_steps >= t_max else [0]
    )
    out = []
    for start in starts:
        length = min(t_max, n_steps - start)
        if length >= min_len:
            out.append((int(start), int(length)))
    return out


def slice_windows_mixed(
    starts: list[float],
    ends: list[float],
    config: dict,
) -> list[tuple[int, int]]:
    """Window variable-resolution nodes by physical time and node count."""
    n_steps = len(starts)
    mixed = config["mixed_time"]
    min_len = int(mixed.get("min_len", 2))
    max_nodes = int(mixed.get("max_nodes", 72))
    max_history_hours = float(mixed.get("max_history_hours", 168))
    stride_hours = float(mixed.get("stride_hours", 12))
    min_history_hours = float(mixed.get("min_history_hours", 1))
    if n_steps < min_len:
        return []

    if n_steps <= max_nodes and ends[-1] - starts[0] <= max_history_hours:
        return [(0, n_steps)]

    candidate_starts = [0]
    last_start_time = starts[0]
    for idx in range(1, n_steps):
        if starts[idx] - last_start_time >= stride_hours:
            candidate_starts.append(idx)
            last_start_time = starts[idx]

    windows: list[tuple[int, int]] = []
    for start_idx in candidate_starts:
        end_idx = start_idx
        while end_idx < n_steps and end_idx - start_idx < max_nodes:
            if ends[end_idx] - starts[start_idx] > max_history_hours + 1e-6:
                break
            end_idx += 1
        length = end_idx - start_idx
        if length < min_len:
            continue
        if ends[end_idx - 1] - starts[start_idx] < min_history_hours:
            continue
        windows.append((start_idx, length))
    return windows


# ── Shard assembly ──────────────────────────────────────────────────────


def _empty_shard(label_names: list[str]) -> dict:
    return {
        "hour_key_to_index": {},
        "per_hour_stay_ids": [],
        "per_hour_hours": [],
        "per_hour_time_hours": [],
        "per_hour_start_hours": [],
        "per_hour_end_hours": [],
        "per_hour_duration_hours": [],
        "per_hour_is_icu": [],
        "per_hour_has_measured": [],
        "per_hour_has_action": [],
        "per_hour_hours_since_last_measurement": [],
        "per_hour_state_texts": [],
        "per_hour_action_texts": [],
        "per_hour_labels": {n: [] for n in label_names},
        "stay_key_to_index": {},
        "per_stay_rows": [],
        "windows": [],
    }


def _add_bundle(shard: dict, bundle: dict) -> None:
    """Append one admission's windows, deduplicating shared bins.

    Sliding windows overlap, so a bin belongs to several windows; storing the
    text once and referencing it by index is what keeps shard size linear in
    bins rather than in window-bins.
    """
    stay_id = bundle["stay_id"]
    if stay_id not in shard["stay_key_to_index"]:
        shard["stay_key_to_index"][stay_id] = len(shard["per_stay_rows"])
        shard["per_stay_rows"].append(bundle["per_stay"])
    stay_index = shard["stay_key_to_index"][stay_id]

    for start_step, length in bundle["window_positions"]:
        hour_indices = np.empty(length, dtype=np.int32)
        for i in range(length):
            abs_bin = start_step + i
            key = (stay_id, abs_bin)
            idx = shard["hour_key_to_index"].get(key)
            if idx is None:
                idx = len(shard["per_hour_state_texts"])
                shard["hour_key_to_index"][key] = idx
                shard["per_hour_stay_ids"].append(stay_id)
                shard["per_hour_hours"].append(abs_bin)
                shard["per_hour_time_hours"].append(
                    float(bundle["time_hours"][abs_bin])
                )
                shard["per_hour_start_hours"].append(
                    float(bundle["start_hours"][abs_bin])
                )
                shard["per_hour_end_hours"].append(
                    float(bundle["end_hours"][abs_bin])
                )
                shard["per_hour_duration_hours"].append(
                    float(bundle["duration_hours"][abs_bin])
                )
                shard["per_hour_is_icu"].append(
                    bool(bundle["is_icu"][abs_bin])
                )
                shard["per_hour_has_measured"].append(
                    bool(bundle["has_measured"][abs_bin])
                )
                shard["per_hour_has_action"].append(
                    bool(bundle["has_action"][abs_bin])
                )
                since = bundle["hours_since_last_measurement"][abs_bin]
                shard["per_hour_hours_since_last_measurement"].append(
                    np.nan if since is None or pd.isna(since) else float(since)
                )
                shard["per_hour_state_texts"].append(bundle["state_texts"][abs_bin])
                shard["per_hour_action_texts"].append(bundle["action_texts"][abs_bin])
                for name, arr in bundle["labels"].items():
                    shard["per_hour_labels"][name].append(float(arr[abs_bin]))
            hour_indices[i] = idx

        shard["windows"].append({
            "stay_index": int(stay_index),
            "start_step": int(start_step),
            "length": int(length),
            "hour_indices": hour_indices,
        })


def _finalize(shard: dict, label_names: list[str], bucket: int,
              shard_idx: int) -> dict:
    rows = shard["per_stay_rows"]

    def col(key, dtype):
        return np.asarray([r[key] for r in rows], dtype=dtype)

    return {
        "per_hour": {
            "stay_ids": np.asarray(shard["per_hour_stay_ids"], dtype=np.int64),
            "hours": np.asarray(shard["per_hour_hours"], dtype=np.int32),
            "time_hours": np.asarray(
                shard["per_hour_time_hours"], dtype=np.float32,
            ),
            "start_hours": np.asarray(
                shard["per_hour_start_hours"], dtype=np.float32,
            ),
            "end_hours": np.asarray(
                shard["per_hour_end_hours"], dtype=np.float32,
            ),
            "duration_hours": np.asarray(
                shard["per_hour_duration_hours"], dtype=np.float32,
            ),
            "is_icu": np.asarray(shard["per_hour_is_icu"], dtype=np.bool_),
            "has_measured": np.asarray(
                shard["per_hour_has_measured"], dtype=np.bool_,
            ),
            "has_action": np.asarray(
                shard["per_hour_has_action"], dtype=np.bool_,
            ),
            "action_present": np.asarray(
                shard["per_hour_has_action"], dtype=np.bool_,
            ),
            "hours_since_last_measurement": np.asarray(
                shard["per_hour_hours_since_last_measurement"], dtype=np.float32,
            ),
            "state_texts": list(shard["per_hour_state_texts"]),
            "action_texts": list(shard["per_hour_action_texts"]),
        },
        "per_hour_labels": {
            n: np.asarray(shard["per_hour_labels"][n], dtype=np.float32)
            for n in label_names
        },
        "per_stay": {
            "stay_ids": col("stay_id", np.int64),
            "subject_ids": col("subject_id", np.int64),
            "ages": col("age", np.float64),
            "genders": [r["gender"] for r in rows],
            "races": [r["race"] for r in rows],
            "charlson_cis": col("charlson_ci", np.int32),
            "icu_mortalities": col("hospital_mortality", np.bool_),
            "hospital_mortalities": col("hospital_mortality", np.bool_),
            "prolonged_stays": col("prolonged_stay", np.bool_),
            "sepsis3s": col("suspected_infection", np.bool_),
        },
        "per_stay_extra": {
            "mortality_30d": col("mortality_30d", np.bool_),
            "mortality_90d": col("mortality_90d", np.bool_),
            "icu_transfer": col("has_icu", np.bool_),
            "readmission_30d": col("readmission_30d", np.bool_),
            "los_hours": col("los_hours", np.float32),
        },
        "windows": shard["windows"],
        "schema_version": SCHEMA_VERSION,
        "shard_idx": int(shard_idx),
        "chunk_idx": int(bucket),
        "n_windows": len(shard["windows"]),
        "n_unique_hours": len(shard["per_hour_state_texts"]),
        "n_unique_stays": len(rows),
    }


# ── Per-bucket worker ───────────────────────────────────────────────────


def process_bucket(args: tuple) -> dict:
    """Build and write all shards for one hadm_id bucket.

    Workers write their own shard files rather than returning bundles: the
    per-hour text for a bucket is hundreds of megabytes, and shipping it back
    through the pool would dominate the runtime.
    """
    bucket, paths, traj_cfg, splits_map = args
    t0 = time.time()

    disc = Path(paths["output"]["discretized"])
    out_dir = Path(paths["output"]["trajectories"])
    cohort_path = Path(paths["output"]["cohort"]) / "cohort.parquet"

    label_targets = traj_cfg["label_targets"]
    class_targets = traj_cfg.get("classification_targets", [])
    label_names = [t["name"] for t in label_targets] + [
        t["name"] for t in class_targets
    ]
    shard_size = int(traj_cfg.get("shard_size", 10000))

    # Directory paths, not globs: pandas passes the path straight to pyarrow,
    # which reads a directory as a dataset but does not expand wildcards.
    meta_dir = disc / "meta" / f"bucket={bucket}"
    obs_dir = disc / "obs" / f"bucket={bucket}"
    act_dir = disc / "act" / f"bucket={bucket}"

    stats = {
        "bucket": bucket, "n_stays": 0, "n_windows": 0,
        "n_hours": 0, "n_shards": 0, "by_split": {},
    }

    if not meta_dir.exists():
        logger.warning("bucket %d: no meta directory at %s", bucket, meta_dir)
        return stats

    meta = pd.read_parquet(meta_dir)
    if len(meta) == 0:
        return stats
    mixed_time = "node_start_hours" in meta.columns

    obs = (
        pd.read_parquet(obs_dir) if obs_dir.exists()
        else pd.DataFrame(columns=[
            "hadm_id", "step_idx", "variable_name", "numeric_value",
            "text_value", "unit", "category", "source_table",
        ])
    )
    act = (
        pd.read_parquet(act_dir) if act_dir.exists()
        else pd.DataFrame(columns=obs.columns)
    )
    prior_diagnosis_counts = (
        obs.loc[obs["source_table"].eq("prior_diagnoses")]
        .groupby("hadm_id")["variable_name"]
        .nunique()
        .to_dict()
    )

    # Diagnoses are the label source for downstream new-onset tasks and must
    # not leak into the state the encoder sees.
    exclude_cats = set(traj_cfg["state_text"].get("exclude_categories", []))
    exclude_vars = set(traj_cfg["state_text"].get("exclude_variables", []))
    if exclude_cats or exclude_vars:
        obs = obs.loc[
            ~obs["category"].isin(exclude_cats)
            & ~obs["variable_name"].isin(exclude_vars)
        ]

    cohort = pd.read_parquet(cohort_path, columns=COHORT_COLS)
    cohort = cohort[cohort["hadm_id"].isin(set(meta["hadm_id"]))].set_index("hadm_id")

    bin_hours = int(traj_cfg.get("bin_hours", 6))
    time_lookup = None
    if mixed_time:
        time_lookup = {}
        for row in meta.itertuples(index=False):
            for step_idx, start_hour in enumerate(row.node_start_hours):
                time_lookup[(int(row.hadm_id), step_idx)] = float(start_hour)
    state_texts = build_texts(
        obs, EMPTY_STATE, bin_hours=bin_hours, time_lookup=time_lookup,
    )
    action_texts = build_texts(
        act, EMPTY_ACTION, bin_hours=bin_hours, time_lookup=time_lookup,
    )
    label_lookup = build_label_frames(obs, label_targets)

    del obs, act

    # ── Build one bundle per admission ──────────────────────────────────
    bundles_by_split: dict[str, list[dict]] = defaultdict(list)
    for meta_row in meta.itertuples(index=False):
        hadm_id, n_steps = int(meta_row.hadm_id), int(meta_row.n_steps)
        if mixed_time and bool(
            traj_cfg.get("state_text", {}).get("include_machine_context", True)
        ):
            starts = [float(v) for v in meta_row.node_start_hours]
            ends = [float(v) for v in meta_row.node_end_hours]
            durations = [float(v) for v in meta_row.duration_hours]
            is_icu = [bool(v) for v in meta_row.is_icu]
            has_measured = [bool(v) for v in meta_row.has_measured]
            hours_since = list(meta_row.hours_since_last_measurement)
            observed_groups = [
                list(groups) if groups is not None else []
                for groups in meta_row.observed_groups
            ]
            windows = slice_windows_mixed(starts, ends, traj_cfg)
        else:
            starts = [float(i * bin_hours) for i in range(n_steps)]
            ends = [float((i + 1) * bin_hours) for i in range(n_steps)]
            durations = [float(bin_hours)] * n_steps
            is_icu = [False] * n_steps
            measured_steps = set(state_texts.get(hadm_id, {}))
            has_measured = [i in measured_steps for i in range(n_steps)]
            hours_since = []
            last_measured = None
            for i, measured in enumerate(has_measured):
                if measured:
                    last_measured = i
                    hours_since.append(0.0)
                else:
                    hours_since.append(
                        np.nan if last_measured is None
                        else float((i - last_measured) * bin_hours)
                    )
            observed_groups = [[] for _ in range(n_steps)]
            windows = slice_windows(n_steps, traj_cfg)
        if not windows:
            continue
        if hadm_id not in cohort.index:
            continue
        row = cohort.loc[hadm_id]

        labels = {}
        for name in [target["name"] for target in label_targets]:
            arr = np.full(n_steps, np.nan, dtype=np.float32)
            for s, v in label_lookup.get(name, {}).get(hadm_id, {}).items():
                if 0 <= s < n_steps:
                    arr[s] = v
            labels[name] = arr

        for target in class_targets:
            source = target["source_label"]
            if source not in labels:
                raise KeyError(
                    f"classification target {target['name']!r} references "
                    f"missing source label {source!r}"
                )
            threshold = float(target["threshold"])
            measured = np.isfinite(labels[source])
            derived = np.full(n_steps, 2.0, dtype=np.float32)
            derived[measured & (labels[source] > threshold)] = 0.0
            derived[measured & (labels[source] <= threshold)] = 1.0
            labels[target["name"]] = derived

        state_list = _fill_texts(
            state_texts.get(hadm_id, {}), n_steps, EMPTY_STATE, bin_hours,
            time_hours=starts if mixed_time else None,
        )
        if mixed_time:
            state_list = _augment_state_texts(
                state_list, starts, ends, durations, is_icu, has_measured,
                hours_since, observed_groups,
            )

        action_list = _fill_texts(
            action_texts.get(hadm_id, {}), n_steps, EMPTY_ACTION, bin_hours,
            time_hours=starts if mixed_time else None,
        )
        action_steps = set(action_texts.get(hadm_id, {}))
        has_action = [step in action_steps for step in range(n_steps)]
        include_machine_context = bool(
            traj_cfg.get("state_text", {}).get("include_machine_context", True)
        )
        if not include_machine_context:
            state_list = [_strip_machine_time_prefix(text) for text in state_list]
            action_list = [_strip_machine_time_prefix(text) for text in action_list]

        time_reference = str(traj_cfg.get("model_time_reference", "start"))
        if time_reference not in {"start", "end"}:
            raise ValueError(
                "model_time_reference must be 'start' or 'end', got "
                f"{time_reference!r}"
            )
        model_times = ends if time_reference == "end" else starts

        bundle = {
            "stay_id": hadm_id,
            "n_steps": n_steps,
            "state_texts": state_list,
            "action_texts": action_list,
            "has_action": has_action,
            "time_hours": model_times,
            "start_hours": starts,
            "end_hours": ends,
            "duration_hours": durations,
            "is_icu": is_icu,
            "has_measured": has_measured,
            "hours_since_last_measurement": hours_since,
            "labels": labels,
            "window_positions": windows,
            "per_stay": {
                "stay_id": hadm_id,
                "subject_id": int(row["subject_id"]),
                "age": float(row["admission_age"]),
                "gender": str(row["gender"]),
                "race": str(row["race"]),
                # Keep the legacy field name for reader compatibility, but do
                # not expose the current admission's discharge ICD count.
                "charlson_ci": int(prior_diagnosis_counts.get(hadm_id, 0)),
                "hospital_mortality": bool(row["hospital_expire_flag"]),
                "prolonged_stay": bool(row["prolonged_stay"]),
                "suspected_infection": bool(row["suspected_infection"]),
                "mortality_30d": bool(row["mortality_30d"]),
                "mortality_90d": bool(row["mortality_90d"]),
                "has_icu": bool(row["has_icu"]),
                "readmission_30d": bool(row["readmission_30d"]),
                "los_hours": float(row["los_hours"]),
            },
        }
        bundles_by_split[splits_map.get(hadm_id, "train")].append(bundle)

    # ── Write shards per split ──────────────────────────────────────────
    for split, bundles in bundles_by_split.items():
        split_dir = out_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        bundles.sort(key=lambda b: b["stay_id"])

        shard_idx = 0
        shard = _empty_shard(label_names)
        n_shards = n_windows = n_hours = 0

        def flush(cur: dict, idx: int) -> tuple[int, int, int]:
            if not cur["windows"]:
                return 0, 0, 0
            final = _finalize(cur, label_names, bucket, idx)
            torch.save(
                final, split_dir / f"trajectories_b{bucket:02d}_{idx:03d}.pt",
            )
            return 1, final["n_windows"], final["n_unique_hours"]

        for bundle in bundles:
            n_w = len(bundle["window_positions"])
            if shard["windows"] and len(shard["windows"]) + n_w > shard_size:
                ds, dw, dh = flush(shard, shard_idx)
                n_shards += ds; n_windows += dw; n_hours += dh
                shard_idx += 1
                shard = _empty_shard(label_names)
            _add_bundle(shard, bundle)

        ds, dw, dh = flush(shard, shard_idx)
        n_shards += ds; n_windows += dw; n_hours += dh

        stats["by_split"][split] = {
            "n_stays": len(bundles), "n_windows": n_windows,
            "n_hours": n_hours, "n_shards": n_shards,
        }
        stats["n_stays"] += len(bundles)
        stats["n_windows"] += n_windows
        stats["n_hours"] += n_hours
        stats["n_shards"] += n_shards

    stats["elapsed_sec"] = round(time.time() - t0, 1)
    return stats


# ── Main ────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 06: hospital trajectories")
    parser.add_argument("--paths_config", default="hosp_paths.yaml")
    parser.add_argument("--traj_config", default="hosp_trajectories.yaml")
    parser.add_argument("--n_workers", type=int, default=None)
    parser.add_argument("--buckets", type=int, nargs="*", default=None,
                        help="Restrict to these buckets (smoke tests)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s",
    )

    paths = db.load_yaml(args.paths_config)
    traj_cfg = db.load_yaml(args.traj_config)
    n_buckets = int(paths["sharding"]["n_buckets"])
    n_workers = args.n_workers or int(traj_cfg.get("n_workers", 8))

    logger.info("=" * 62)
    if traj_cfg.get("mixed_time", {}).get("enabled", False):
        logger.info(
            "Step 06: mixed-time trajectory shards "
            "(max_history=%gh, max_nodes=%d, stride=%gh)",
            traj_cfg["mixed_time"]["max_history_hours"],
            traj_cfg["mixed_time"]["max_nodes"],
            traj_cfg["mixed_time"]["stride_hours"],
        )
    else:
        logger.info(
            "Step 06: trajectory shards (t_max=%d bins, stride=%d, min_len=%d)",
            traj_cfg["t_max"], traj_cfg["stride"], traj_cfg["min_len"],
        )
    logger.info("=" * 62)

    splits_path = Path(paths["output"]["splits"]) / "splits.csv"
    splits_df = pd.read_csv(splits_path)
    splits_map = dict(
        zip(splits_df["hadm_id"].astype(int), splits_df["split"])
    )
    logger.info("Loaded splits for %d admissions", len(splits_map))

    out_dir = Path(paths["output"]["trajectories"])
    for split in ("train", "val", "test"):
        (out_dir / split).mkdir(parents=True, exist_ok=True)
        for stale in (out_dir / split).glob("*.pt"):
            stale.unlink()

    buckets = args.buckets if args.buckets is not None else list(range(n_buckets))
    work = [(b, paths, traj_cfg, splits_map) for b in buckets]

    t0 = time.time()
    results: list[dict] = []
    if n_workers <= 1:
        for w in work:
            r = process_bucket(w)
            results.append(r)
            logger.info("  bucket %02d: %d stays, %d windows, %d shards (%.1fs)",
                        r["bucket"], r["n_stays"], r["n_windows"],
                        r["n_shards"], r.get("elapsed_sec", 0))
    else:
        with Pool(n_workers) as pool:
            for r in pool.imap_unordered(process_bucket, work):
                results.append(r)
                logger.info(
                    "  bucket %02d: %d stays, %d windows, %d shards (%.1fs) "
                    "[%d/%d done]",
                    r["bucket"], r["n_stays"], r["n_windows"], r["n_shards"],
                    r.get("elapsed_sec", 0), len(results), len(work),
                )

    elapsed = time.time() - t0

    totals = {"n_stays": 0, "n_windows": 0, "n_hours": 0, "n_shards": 0}
    per_split: dict[str, dict] = defaultdict(
        lambda: {"n_stays": 0, "n_windows": 0, "n_hours": 0, "n_shards": 0}
    )
    for r in results:
        for k in totals:
            totals[k] += r[k]
        for split, s in r["by_split"].items():
            for k in totals:
                per_split[split][k] += s[k]

    logger.info("-" * 62)
    logger.info("  admissions with windows %14s", f"{totals['n_stays']:,}")
    logger.info("  training windows        %14s", f"{totals['n_windows']:,}")
    logger.info("  unique bins stored      %14s", f"{totals['n_hours']:,}")
    logger.info("  shards written          %14s", f"{totals['n_shards']:,}")
    logger.info("  elapsed                 %13.1fs", elapsed)
    for split in ("train", "val", "test"):
        s = per_split[split]
        logger.info("  %-6s stays=%7d windows=%8d bins=%9d shards=%4d",
                    split, s["n_stays"], s["n_windows"], s["n_hours"],
                    s["n_shards"])

    meta = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "schema_version": SCHEMA_VERSION,
        "scope": "hospital_admission",
        "time_mode": (
            "mixed_ward_24h_icu_1h"
            if traj_cfg.get("mixed_time", {}).get("enabled", False)
            else "fixed"
        ),
        "bin_hours": (
            None
            if traj_cfg.get("mixed_time", {}).get("enabled", False)
            else int(traj_cfg.get("bin_hours", 6))
        ),
        "params": {
            "t_max_bins": traj_cfg["t_max"],
            "min_len_bins": traj_cfg["min_len"],
            "stride_bins": traj_cfg["stride"],
            "model_time_reference": traj_cfg.get(
                "model_time_reference", "start"
            ),
            "shard_size": traj_cfg.get("shard_size", 10000),
            "mixed_time": traj_cfg.get("mixed_time"),
        },
        "counts": totals,
        "per_split": {k: dict(v) for k, v in per_split.items()},
        "label_targets": (
            [t["name"] for t in traj_cfg["label_targets"]]
            + [t["name"] for t in traj_cfg.get("classification_targets", [])]
        ),
        "per_bucket": sorted(results, key=lambda r: r["bucket"]),
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False)
    )
    logger.info("Wrote %s", out_dir / "metadata.json")


if __name__ == "__main__":
    main()
