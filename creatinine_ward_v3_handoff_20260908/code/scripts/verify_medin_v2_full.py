#!/usr/bin/env python3
"""Verify MedIn v0.3 episode / dense / gap trajectory invariants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from clin_jepa.data_hosp.episode_windows import CANONICAL_NO_ACTION, EMPTY_STATE_BODY


def _check_split(root: Path, split: str, pool: str) -> dict:
    files = sorted((root / split).glob("*.pt"))
    if not files:
        raise FileNotFoundError(f"no shards in {root / split}")
    n_windows = n_hours = 0
    sources = {"ICU": 0, "WARD": 0, "MIXED": 0}
    mixed_windows = 0
    time_errors = 0
    empty_windows = 0
    quadrants = {"S1A1": 0, "S1A0": 0, "S0A1": 0, "S0A0": 0}
    for path in files:
        shard = torch.load(path, map_location="cpu", weights_only=False)
        ph = shard["per_hour"]
        required = {
            "state_observed", "action_present", "start_hours", "end_hours",
            "time_hours", "state_texts", "action_texts", "stay_ids",
        }
        missing = required - set(ph)
        if missing:
            raise AssertionError(f"{path} missing migrated fields {sorted(missing)}")
        stay_ids_all = np.asarray(ph["stay_ids"], dtype=np.int64)
        starts_all = np.asarray(ph["start_hours"], dtype=np.float64)
        ends_all = np.asarray(ph["end_hours"], dtype=np.float64)
        expected_time_all = np.empty(len(stay_ids_all), dtype=np.float32)
        for stay_id in np.unique(stay_ids_all):
            stay_idx = np.flatnonzero(stay_ids_all == stay_id)
            expected_time_all[stay_idx] = (
                ends_all[stay_idx] - float(starts_all[stay_idx].min())
            ).astype(np.float32)
        state_all = np.asarray(ph["state_observed"], dtype=np.bool_)
        action_all = np.asarray(ph["action_present"], dtype=np.bool_)
        for state_value, action_value in zip(state_all, action_all):
            quadrants[f"S{int(state_value)}A{int(action_value)}"] += 1
        for idx, (state_value, action_value) in enumerate(zip(state_all, action_all)):
            state_text = str(ph["state_texts"][idx])
            action_text = str(ph["action_texts"][idx])
            if not state_value and state_text != EMPTY_STATE_BODY:
                raise AssertionError(f"{path} node {idx}: missing state has noncanonical text")
            if state_value and not state_text.startswith("Patient state:\n"):
                raise AssertionError(f"{path} node {idx}: observed state lacks role prefix")
            if action_value and action_text == CANONICAL_NO_ACTION:
                raise AssertionError(f"{path} node {idx}: real action uses NO_ACTION text")
            if action_value and not action_text.startswith("Clinical action:\n"):
                raise AssertionError(f"{path} node {idx}: real action lacks role prefix")
            if not action_value and action_text != CANONICAL_NO_ACTION:
                raise AssertionError(f"{path} node {idx}: absent action is not canonical")
            if "Care setting resolution:" in state_text or "Interval duration:" in state_text:
                raise AssertionError(f"machine resolution/time prose leaked into {path}")
        n_hours += len(ph["state_texts"])
        n_windows += len(shard["windows"])
        for w in shard["windows"]:
            hi = np.asarray(w["hour_indices"])
            icu = np.asarray(ph["is_icu"])[hi]
            if icu.any() and (~icu).any():
                mixed_windows += 1
            src = w.get("source_type")
            if src in sources:
                sources[src] += 1
            elif bool(icu.all()):
                sources["ICU"] += 1
            elif not bool(icu.any()):
                sources["WARD"] += 1
            else:
                sources["MIXED"] += 1
            t_abs = np.asarray(ph["time_hours"])[hi]
            starts = np.asarray(ph["start_hours"])[hi]
            ends = np.asarray(ph["end_hours"])[hi]
            measured = np.asarray(ph["has_measured"])[hi]
            if (
                not np.allclose(t_abs, expected_time_all[hi], atol=1e-4)
                or np.any(np.diff(t_abs) <= 0)
                or np.any(ends <= starts)
            ):
                time_errors += 1
            if pool == "trajectories_dense" and not bool(measured.all()):
                empty_windows += 1
    if mixed_windows:
        raise AssertionError(
            f"{root} {split}: {mixed_windows} windows still mix ICU+ward nodes"
        )
    if time_errors:
        raise AssertionError(f"{root} {split}: {time_errors} windows have invalid episode time")
    if empty_windows:
        raise AssertionError(f"{root} {split}: dense pool contains {empty_windows} empty windows")
    return {
        "shards": len(files),
        "windows": n_windows,
        "hours": n_hours,
        "sources": sources,
        "time_errors": time_errors,
        "quadrants": quadrants,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    report = {}
    for pool in ("trajectories_episode", "trajectories_dense", "trajectories_gap"):
        pool_dir = root / pool
        if not pool_dir.exists():
            continue
        report[pool] = {}
        for split in ("train", "val", "test"):
            if (pool_dir / split).exists():
                report[pool][split] = _check_split(pool_dir, split, pool)
    splits = root / "splits" / "splits.csv"
    if not splits.exists():
        raise FileNotFoundError(f"patient split proof missing: {splits}")
    import pandas as pd
    df = pd.read_csv(splits)
    subjects = {
        s: set(df.loc[df.split == s, "subject_id"])
        for s in ("train", "val", "test")
    }
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = subjects[left] & subjects[right]
        if overlap:
            raise AssertionError(f"subject leak {left}∩{right} n={len(overlap)}")
    report["split_patients"] = {
        s: int(df.loc[df.split == s, "subject_id"].nunique())
        for s in ("train", "val", "test")
    }
    migration_report = root / "migration_report.json"
    if not migration_report.exists():
        raise FileNotFoundError(
            f"migrated data requires provenance report {migration_report}"
        )
    migration = json.loads(migration_report.read_text(encoding="utf-8"))
    if migration.get("schema_version") != "medin_v03_existing_shard_migration_v2":
        raise AssertionError(
            f"outdated migration report schema: {migration.get('schema_version')}"
        )
    report["migration_report"] = str(migration_report.resolve())
    train_quadrants = (
        report.get("trajectories_episode", {}).get("train", {}).get("quadrants", {})
    )
    train_no_action = int(train_quadrants.get("S1A0", 0)) + int(
        train_quadrants.get("S0A0", 0)
    )
    report["no_action_distribution_note"] = (
        "No NO_ACTION nodes observed; allowed after structured/text audit."
        if train_no_action == 0 else
        f"Observed {train_no_action} NO_ACTION nodes in train."
    )
    report["schema_version"] = "medin_v03_episode_time_no_action_verify_v2"
    out = root / "verify_v2_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print("VERIFY_OK", out)


if __name__ == "__main__":
    main()
