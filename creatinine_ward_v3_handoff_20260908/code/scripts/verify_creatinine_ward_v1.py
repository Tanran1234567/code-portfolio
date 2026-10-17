"""Fail-fast verifier for the ICU-free creatinine ward experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build_node_manifest(root: Path, cohort: pd.DataFrame) -> pd.DataFrame:
    """Create an auditable structured gold table before reading embeddings."""
    obs_files = sorted((root / "discretized" / "obs").glob("bucket=*/*.parquet"))
    if not obs_files:
        raise FileNotFoundError("no discretized observation parquet files")
    creat_parts = []
    for path in obs_files:
        frame = pd.read_parquet(path)
        mask = (
            frame["source_table"].eq("chemistry")
            & frame["variable_name"].eq("creatinine")
            & frame["numeric_value"].notna()
        )
        if bool(mask.any()):
            creat_parts.append(frame.loc[
                mask, ["hadm_id", "step_idx", "timestamp", "numeric_value", "unit"]
            ])
    if creat_parts:
        creat = pd.concat(creat_parts, ignore_index=True)
        creat = (
            creat.sort_values(["hadm_id", "step_idx", "timestamp"])
            .drop_duplicates(["hadm_id", "step_idx"], keep="last")
            .rename(columns={
                "timestamp": "creatinine_charttime",
                "numeric_value": "creatinine_last_mg_dl",
                "unit": "creatinine_unit",
            })
        )
    else:
        creat = pd.DataFrame(columns=[
            "hadm_id", "step_idx", "creatinine_charttime",
            "creatinine_last_mg_dl", "creatinine_unit",
        ])
    rows = []
    for row in cohort.itertuples(index=False):
        for step_idx in range(int(row.n_complete_bins)):
            rows.append({
                "subject_id": int(row.subject_id),
                "hadm_id": int(row.hadm_id),
                "split": str(row.split),
                "step_idx": step_idx,
                "bucket_start_hours": float(step_idx * 24),
                "bucket_end_hours": float((step_idx + 1) * 24),
                "time_hours": float((step_idx + 1) * 24),
            })
    manifest = pd.DataFrame(rows).merge(
        creat, on=["hadm_id", "step_idx"], how="left", validate="one_to_one"
    )
    value = manifest.creatinine_last_mg_dl.to_numpy(float)
    class_id = np.full(len(manifest), 2, dtype=np.int8)
    class_id[np.isfinite(value) & (value > 1.5)] = 0
    class_id[np.isfinite(value) & (value <= 1.5)] = 1
    manifest["creatinine_class_id"] = class_id
    manifest["creatinine_class"] = np.asarray(
        ["HIGH", "NOT_HIGH", "NOT_MEASURED"], dtype=object
    )[class_id]
    manifest_dir = root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(manifest_dir / "node_labels.parquet", index=False)
    manifest.to_csv(manifest_dir / "node_labels.csv", index=False)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    args = parser.parse_args()
    root = Path(args.data_root)
    cohort_path = root / "cohort" / "cohort.parquet"
    cohort = pd.read_parquet(cohort_path)
    expected_split = {"train": 2100, "val": 450, "test": 450}
    assert cohort.split.value_counts().to_dict() == expected_split
    assert cohort.subject_id.nunique() == 3000
    assert cohort.hadm_id.nunique() == 3000
    assert not cohort.has_icu.any()
    assert (cohort.n_icu_stays == 0).all()
    assert cohort.ever_creatinine.all()
    assert (cohort.n_creatinine > 0).all()
    expected_missing_patients = {"train": 462, "val": 99, "test": 99}
    observed_missing_patients = (
        cohort.groupby("split").has_missing_creatinine_bin.sum().astype(int).to_dict()
    )
    assert observed_missing_patients == expected_missing_patients
    expected_high_patients = {"train": 693, "val": 148, "test": 148}
    observed_high_patients = (
        (cohort.creat_max.astype(float) > 1.5)
        .groupby(cohort.split).sum().astype(int).to_dict()
    )
    assert observed_high_patients == expected_high_patients
    node_manifest = build_node_manifest(root, cohort)
    gold = node_manifest.set_index(["hadm_id", "step_idx"])

    totals = {
        "admissions": 0,
        "windows": 0,
        "nodes": 0,
        "class_counts": {"HIGH": 0, "NOT_HIGH": 0, "NOT_MEASURED": 0},
    }
    text_hashes: dict[tuple[int, int], tuple[str, str]] = {}
    by_split: dict[str, dict] = {}
    for split, expected_n in expected_split.items():
        split_dir = root / "trajectories" / split
        shards = sorted(split_dir.glob("*.pt"))
        if not shards:
            raise FileNotFoundError(f"no trajectory shards in {split_dir}")
        stay_ids: set[int] = set()
        class_patients = {0: set(), 1: set(), 2: set()}
        split_windows = split_nodes = 0
        split_class_counts = np.zeros(3, dtype=np.int64)
        transition_counts = np.zeros((3, 3), dtype=np.int64)
        missing_run_patients = {"one": 0, "two": 0, "three_or_more": 0}
        for path in shards:
            shard = torch.load(path, map_location="cpu", weights_only=False)
            ph = shard["per_hour"]
            labels = shard["per_hour_labels"]
            assert "label_creatinine" in labels
            assert "label_creatinine_class" in labels
            assert not np.asarray(ph["is_icu"], dtype=bool).any()
            assert np.asarray(ph["has_measured"], dtype=bool).all()
            times = np.asarray(ph["time_hours"], dtype=float)
            starts = np.asarray(ph["start_hours"], dtype=float)
            ends = np.asarray(ph["end_hours"], dtype=float)
            durations = np.asarray(ph["duration_hours"], dtype=float)
            assert np.allclose(times, ends)
            assert np.allclose(ends - starts, 24.0)
            assert np.allclose(durations, 24.0)
            assert all(not str(text).lstrip().startswith("t=") for text in ph["state_texts"])
            assert all(not str(text).lstrip().startswith("t=") for text in ph["action_texts"])

            creat = np.asarray(labels["label_creatinine"], dtype=float)
            observed_class = np.asarray(labels["label_creatinine_class"], dtype=int)
            expected_class = np.full(len(creat), 2, dtype=int)
            expected_class[np.isfinite(creat) & (creat > 1.5)] = 0
            expected_class[np.isfinite(creat) & (creat <= 1.5)] = 1
            assert np.array_equal(observed_class, expected_class)
            for node_index, (hadm_id, step_idx) in enumerate(zip(
                np.asarray(ph["stay_ids"], dtype=np.int64),
                np.asarray(ph["hours"], dtype=np.int32),
            )):
                gold_row = gold.loc[(int(hadm_id), int(step_idx))]
                gold_value = float(gold_row.creatinine_last_mg_dl)
                if np.isfinite(gold_value):
                    assert np.isclose(creat[node_index], gold_value)
                else:
                    assert not np.isfinite(creat[node_index])
                assert observed_class[node_index] == int(gold_row.creatinine_class_id)
                text_hashes[(int(hadm_id), int(step_idx))] = (
                    hashlib.sha256(
                        str(ph["state_texts"][node_index]).encode("utf-8")
                    ).hexdigest(),
                    hashlib.sha256(
                        str(ph["action_texts"][node_index]).encode("utf-8")
                    ).hexdigest(),
                )
            counts = np.bincount(observed_class, minlength=3)
            split_class_counts += counts
            for name, value in zip(("HIGH", "NOT_HIGH", "NOT_MEASURED"), counts):
                totals["class_counts"][name] += int(value)

            windows = shard["windows"]
            stay_ids.update(map(int, shard["per_stay"]["stay_ids"]))
            split_windows += len(windows)
            split_nodes += len(ph["state_texts"])
            assert len(windows) == len(shard["per_stay"]["stay_ids"])
            for window in windows:
                assert int(window["start_step"]) == 0
                assert int(window["length"]) >= 4
                subject_id = int(
                    shard["per_stay"]["subject_ids"][int(window["stay_index"])]
                )
                window_classes = observed_class[window["hour_indices"]]
                for class_id in np.unique(window_classes):
                    class_patients[int(class_id)].add(subject_id)
                for source, target in zip(window_classes[:-1], window_classes[1:]):
                    transition_counts[int(source), int(target)] += 1
                longest_missing_run = 0
                current_missing_run = 0
                for class_id in window_classes:
                    if int(class_id) == 2:
                        current_missing_run += 1
                        longest_missing_run = max(
                            longest_missing_run, current_missing_run
                        )
                    else:
                        current_missing_run = 0
                if longest_missing_run == 1:
                    missing_run_patients["one"] += 1
                elif longest_missing_run == 2:
                    missing_run_patients["two"] += 1
                elif longest_missing_run >= 3:
                    missing_run_patients["three_or_more"] += 1
        assert len(stay_ids) == expected_n, (split, len(stay_ids), expected_n)
        assert split_windows == expected_n
        assert all(len(values) >= 45 for values in class_patients.values())
        by_split[split] = {
            "admissions": len(stay_ids),
            "windows": split_windows,
            "nodes": split_nodes,
            "class_patients": {
                name: len(class_patients[index])
                for index, name in enumerate(("HIGH", "NOT_HIGH", "NOT_MEASURED"))
            },
            "class_nodes": {
                name: int(split_class_counts[index])
                for index, name in enumerate(("HIGH", "NOT_HIGH", "NOT_MEASURED"))
            },
            "transition_matrix_rows_current_cols_next": transition_counts.tolist(),
            "transition_class_order": ["HIGH", "NOT_HIGH", "NOT_MEASURED"],
            "longest_consecutive_not_measured_run_patients": missing_run_patients,
        }
        totals["admissions"] += len(stay_ids)
        totals["windows"] += split_windows
        totals["nodes"] += split_nodes

    assert totals["admissions"] == 3000
    assert totals["windows"] == 3000
    node_manifest["state_text_sha256"] = [
        text_hashes[(int(row.hadm_id), int(row.step_idx))][0]
        for row in node_manifest.itertuples(index=False)
    ]
    node_manifest["action_text_sha256"] = [
        text_hashes[(int(row.hadm_id), int(row.step_idx))][1]
        for row in node_manifest.itertuples(index=False)
    ]
    node_manifest.to_parquet(root / "manifests" / "node_labels.parquet", index=False)
    node_manifest.to_csv(root / "manifests" / "node_labels.csv", index=False)
    train_values = node_manifest.loc[
        node_manifest.split.eq("train")
        & node_manifest.creatinine_last_mg_dl.notna(),
        "creatinine_last_mg_dl",
    ].to_numpy(float)
    report = {
        "schema_version": "creatinine_ward_verify_v1",
        "status": "VERIFY_OK",
        "cohort_sha256": sha256(cohort_path),
        "node_manifest_sha256": sha256(root / "manifests" / "node_labels.parquet"),
        "totals": totals,
        "by_split": by_split,
        "train_creatinine_distribution": {
            "n_measured_nodes": int(len(train_values)),
            "percentiles": {
                str(percentile): float(np.percentile(train_values, percentile))
                for percentile in (1, 5, 25, 50, 75, 90, 95, 99)
            },
            "candidate_threshold_counts": {
                str(threshold): {
                    "above": int((train_values > threshold).sum()),
                    "at_or_below": int((train_values <= threshold).sum()),
                }
                for threshold in (1.2, 1.3, 1.5, 2.0)
            },
        },
        "invariants": {
            "icu_admissions": 0,
            "bin_hours": 24,
            "partial_final_bins": 0,
            "state_empty_bins": 0,
            "windows_per_admission": 1,
            "model_time": "bucket_end_hours_from_admission",
            "creatinine_threshold_mg_dl": 1.5,
        },
    }
    output = root / "verify_creatinine_ward_v1.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("CREATININE_WARD_VERIFY_OK", output)


if __name__ == "__main__":
    main()
