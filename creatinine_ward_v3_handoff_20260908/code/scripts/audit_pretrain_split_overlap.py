#!/usr/bin/env python3
"""Audit patient leakage between ICU pretraining and hospital evaluation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import torch


def _as_ints(values: Iterable) -> set[int]:
    result = set()
    for value in values:
        if value is None or str(value).strip() == "":
            continue
        result.add(int(float(value)))
    return result


def _aliases(column: str) -> tuple[str, ...]:
    if column in {"subject_id", "subject_ids"}:
        return ("subject_id", "subject_ids")
    if column in {"hadm_id", "hadm_ids", "stay_id", "stay_ids"}:
        return ("hadm_id", "hadm_ids", "stay_id", "stay_ids")
    return (column,)


def _ids_from_pt(path: Path, column: str) -> set[int]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    for alias in _aliases(column):
        if "per_stay" in payload and alias in payload["per_stay"]:
            return _as_ints(payload["per_stay"][alias])
    if "windows" in payload:
        for alias in _aliases(column):
            values = [
                window[alias] for window in payload["windows"] if alias in window
            ]
            if values:
                return _as_ints(values)
    for alias in _aliases(column):
        if alias in payload:
            return _as_ints(payload[alias])
    raise KeyError(f"{path} has no {column!r} field")


def load_ids(
    source: Path,
    column: str,
    split_value: str | None = None,
) -> set[int]:
    if not source.exists():
        raise FileNotFoundError(source)
    if source.is_dir():
        files = sorted(source.rglob("*.pt"))
        if not files:
            raise FileNotFoundError(f"No .pt shards in {source}")
        ids: set[int] = set()
        for path in files:
            ids.update(_ids_from_pt(path, column))
        return ids
    suffix = source.suffix.lower()
    if suffix in {".pt", ".pth"}:
        return _ids_from_pt(source, column)
    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with source.open(newline="") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            selected = next(
                (name for name in _aliases(column) if name in (reader.fieldnames or [])),
                None,
            )
            if selected is None:
                raise KeyError(
                    f"{source} columns={reader.fieldnames}; required {column!r}"
                )
            rows = list(reader)
            if split_value and rows and "split" in rows[0]:
                rows = [row for row in rows if row["split"] == split_value]
            return _as_ints(row[selected] for row in rows)
    if suffix == ".json":
        with source.open() as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            payload = payload[column]
        if payload and isinstance(payload[0], dict):
            if split_value and "split" in payload[0]:
                payload = [
                    row for row in payload if row["split"] == split_value
                ]
            payload = [row[column] for row in payload]
        return _as_ints(payload)
    if suffix == ".parquet":
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("Reading parquet requires pandas/pyarrow") from exc
        frame = pd.read_parquet(source)
        if split_value and "split" in frame.columns:
            frame = frame.loc[frame["split"] == split_value]
        return _as_ints(frame[column])
    return _as_ints(source.read_text().splitlines())


def overlap_report(
    icu_ids: set[int],
    train_ids: set[int],
    val_ids: set[int],
    test_ids: set[int],
) -> dict:
    def detail(left: set[int], right: set[int]) -> dict:
        overlap = sorted(left & right)
        return {
            "count": len(overlap),
            "sample_ids": overlap[:100],
        }

    return {
        "schema_version": "medin_patient_overlap_audit_v1",
        "counts": {
            "icu_pretrain": len(icu_ids),
            "hospital_train": len(train_ids),
            "hospital_val": len(val_ids),
            "hospital_test": len(test_ids),
        },
        "hospital_internal": {
            "train_val": detail(train_ids, val_ids),
            "train_test": detail(train_ids, test_ids),
            "val_test": detail(val_ids, test_ids),
        },
        "icu_pretrain_vs_hospital": {
            "train": detail(icu_ids, train_ids),
            "val": detail(icu_ids, val_ids),
            "test": detail(icu_ids, test_ids),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--icu_pretrain", required=True)
    parser.add_argument("--hospital_train", required=True)
    parser.add_argument("--hospital_val", required=True)
    parser.add_argument("--hospital_test", required=True)
    parser.add_argument("--column", default="subject_id")
    parser.add_argument(
        "--icu_split_value",
        default="train",
        help="If ICU source has a split column, audit only this split.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--allow_icu_test_overlap",
        action="store_true",
        help="Record but do not fail on ICU-pretrain/test overlap.",
    )
    args = parser.parse_args()
    sources = {
        "icu_pretrain": load_ids(
            Path(args.icu_pretrain), args.column, args.icu_split_value
        ),
        "hospital_train": load_ids(Path(args.hospital_train), args.column),
        "hospital_val": load_ids(Path(args.hospital_val), args.column),
        "hospital_test": load_ids(Path(args.hospital_test), args.column),
    }
    report = overlap_report(
        sources["icu_pretrain"],
        sources["hospital_train"],
        sources["hospital_val"],
        sources["hospital_test"],
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        json.dump(report, f, indent=2)
    internal_overlap = any(
        item["count"] for item in report["hospital_internal"].values()
    )
    icu_test_overlap = report["icu_pretrain_vs_hospital"]["test"]["count"] > 0
    if internal_overlap:
        raise SystemExit("FAIL: hospital train/val/test patient overlap detected")
    if icu_test_overlap and not args.allow_icu_test_overlap:
        raise SystemExit(
            "FAIL: ICU pretraining patients overlap hospital test patients; "
            "use a decontaminated test split or explicitly acknowledge with "
            "--allow_icu_test_overlap"
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
