"""v0.3 care-episode segmentation and windowing.

A mixed-resolution admission is split into separate sequences:

    WARD_ONLY / WARD_PRE_ICU / ICU / WARD_INTER_ICU / WARD_POST_ICU

Each care episode is an independent dynamics sequence. Model time is therefore
the bucket end measured from that episode's start: the first ICU bucket ends at
1h and the first complete ward bucket at 24h. Admission-relative start/end hours
remain metadata for physical-horizon evaluation and provenance.
"""

from __future__ import annotations

import re
from typing import Iterable

import numpy as np

EMPTY_STATE_BODY = "No observations recorded."
EMPTY_ACTION_BODY = "No actions recorded."
CANONICAL_NO_ACTION = (
    "Clinical action:\nNo recorded intervention in the modeled action set."
)


def clinical_text_only(text: str) -> str:
    """Strip legacy machine time/resolution/missingness prose from v0.3 text."""
    value = str(text).strip()
    marker = "Hours since last measured observation:"
    if marker in value:
        start = value.index(marker)
        end = value.find(". ", start)
        if end >= 0:
            return value[end + 2 :].strip()
    if re.match(r"^t=-?\d+(?:\.\d+)?h\s*\|", value):
        return value.split(" | ", 1)[1].strip()
    return value


def normalize_state_text(text: str, observed: bool) -> str:
    """Canonical v0.3 state text without imputation or machine time prose."""
    body = clinical_text_only(text)
    if not observed or body in {"", EMPTY_STATE_BODY}:
        return EMPTY_STATE_BODY
    if body.startswith("Patient state:"):
        return body
    return f"Patient state:\n{body}"


def normalize_action_text(text: str) -> tuple[str, bool]:
    """Return canonical action text and whether a modeled action is present."""
    body = clinical_text_only(text)
    lowered = " ".join(body.lower().split())
    compact = lowered.rstrip(" .;:")
    empty = compact in {
        "",
        "no actions recorded",
        "no action recorded",
        "no actions were recorded",
        "no active interventions",
        "no active intervention",
        "no interventions recorded",
        "no intervention recorded",
        "clinical action: no actions recorded",
        "clinical action: no recorded intervention in the modeled action set",
    }
    if empty:
        return CANONICAL_NO_ACTION, False
    if body.startswith("Clinical action:"):
        return body, True
    return f"Clinical action:\n{body}", True


def split_care_episodes(
    start_hours: Iterable[float],
    end_hours: Iterable[float],
    is_icu: Iterable[bool],
    has_measured: Iterable[bool] | None = None,
) -> list[dict]:
    """Split one admission's ordered nodes into contiguous care episodes."""
    starts = np.asarray(list(start_hours), dtype=np.float64)
    ends = np.asarray(list(end_hours), dtype=np.float64)
    icu = np.asarray(list(is_icu), dtype=np.bool_)
    n = len(starts)
    if n == 0:
        return []
    if has_measured is None:
        measured = np.ones(n, dtype=np.bool_)
    else:
        measured = np.asarray(list(has_measured), dtype=np.bool_)

    runs: list[tuple[int, int, bool]] = []
    run_start = 0
    for i in range(1, n):
        if bool(icu[i]) != bool(icu[run_start]):
            runs.append((run_start, i, bool(icu[run_start])))
            run_start = i
    runs.append((run_start, n, bool(icu[run_start])))

    saw_icu = False
    episodes: list[dict] = []
    for lo, hi, run_icu in runs:
        if run_icu:
            episode_type = "ICU"
            source_type = "ICU"
            saw_icu = True
        elif not saw_icu and len(runs) == 1:
            episode_type = "WARD_ONLY"
            source_type = "WARD"
        elif not saw_icu:
            episode_type = "WARD_PRE_ICU"
            source_type = "WARD"
        else:
            trailing_icu = any(
                flag for a, _b, flag in runs if a >= hi and flag
            )
            episode_type = "WARD_INTER_ICU" if trailing_icu else "WARD_POST_ICU"
            source_type = "WARD"

        episode_start = float(starts[lo])
        episode_time = (ends[lo:hi] - episode_start).astype(np.float32)
        episodes.append({
            "episode_type": episode_type,
            "source_type": source_type,
            "index_lo": int(lo),
            "index_hi": int(hi),
            "start_hours": starts[lo:hi].astype(np.float32),
            "end_hours": ends[lo:hi].astype(np.float32),
            "time_hours": episode_time.copy(),
            "episode_time_hours": episode_time,
            "duration_hours": (ends[lo:hi] - starts[lo:hi]).astype(np.float32),
            "is_icu": icu[lo:hi].copy(),
            "has_measured": measured[lo:hi].copy(),
            "state_observed": measured[lo:hi].copy(),
            "n_steps": int(hi - lo),
        })
    return episodes


def slice_episode_windows(
    n_steps: int,
    start_hours: np.ndarray,
    end_hours: np.ndarray,
    source_type: str,
    t_max: int = 72,
    min_len: int = 3,
    icu_stride_hours: float = 12.0,
    ward_stride_hours: float = 288.0,
) -> list[tuple[int, int]]:
    """One window per episode unless it exceeds T_max; then stride by source."""
    if n_steps < min_len:
        return []
    if n_steps <= t_max:
        return [(0, int(n_steps))]

    stride_hours = icu_stride_hours if source_type == "ICU" else ward_stride_hours
    candidate_starts = [0]
    last_start = float(start_hours[0])
    for idx in range(1, n_steps):
        if float(start_hours[idx]) - last_start >= stride_hours - 1e-6:
            candidate_starts.append(idx)
            last_start = float(start_hours[idx])

    windows: list[tuple[int, int]] = []
    for start_idx in candidate_starts:
        length = min(t_max, n_steps - start_idx)
        if length >= min_len:
            windows.append((int(start_idx), int(length)))
    return windows


def dense_observed_segments(
    state_observed: np.ndarray,
    min_len: int = 4,
) -> list[tuple[int, int]]:
    """Consecutive observed runs used for DENSE_TRAJ / artificial mask."""
    observed = np.asarray(state_observed, dtype=np.bool_)
    n = len(observed)
    segments: list[tuple[int, int]] = []
    i = 0
    while i < n:
        if not observed[i]:
            i += 1
            continue
        j = i + 1
        while j < n and observed[j]:
            j += 1
        if j - i >= min_len:
            segments.append((i, j - i))
        i = j
    return segments
