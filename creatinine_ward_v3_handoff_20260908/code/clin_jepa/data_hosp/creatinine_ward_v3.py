"""Build Creatinine Ward v3 labels and trajectory shards.

This v3 dataset keeps the frozen v1 3000-patient cohort and split, but fixes
two downstream-analysis issues:

* the state text retains every structured serum creatinine result in a 24-hour
  bucket instead of only the last value;
* AKI labels are computed from raw ``mimiciv_hosp.labevents`` rows, using a
  prior 7-365 day median creatinine when available and the admission first
  creatinine as a recorded fallback when it is not.

The module intentionally writes to a new data root and never mutates v1/v2.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch

from clin_jepa.data.helpers.text_templates import format_numeric
from clin_jepa.data_hosp import db

LOG = logging.getLogger(__name__)

SCHEMA_VERSION = "creatinine_ward_v3_kdigo_scr"
CREATININE_ITEMID = 50912
CLASS_NAMES_R = ["R0", "R1", "R2", "RU"]
COURSE_CLASS_NAMES = ["RECOVERED", "PERSISTENT_OR_WORSE", "NOT_CURRENT_AKI", "RU"]
RU_CLASS_ID = 3


def _cohort_with_split(v1_root: Path) -> pd.DataFrame:
    cohort_path = v1_root / "cohort" / "cohort.parquet"
    split_path = v1_root / "splits" / "splits.csv"
    if not cohort_path.exists():
        raise FileNotFoundError(f"missing v1 cohort: {cohort_path}")
    if not split_path.exists():
        raise FileNotFoundError(f"missing v1 splits: {split_path}")
    cohort = pd.read_parquet(cohort_path)
    splits = pd.read_csv(split_path)
    if "split" not in cohort.columns:
        cohort = cohort.merge(
            splits[["subject_id", "hadm_id", "split"]],
            on=["subject_id", "hadm_id"],
            how="left",
            validate="one_to_one",
        )
    if "n_complete_bins" not in cohort.columns:
        cohort["n_complete_bins"] = np.floor(
            cohort["los_hours"].astype(float) / 24.0
        ).astype(int)
    cohort["n_complete_bins"] = cohort["n_complete_bins"].clip(lower=0, upper=30)
    missing = cohort["split"].isna().sum()
    if int(missing) != 0:
        raise ValueError(f"{missing} cohort rows have no split")
    if cohort.subject_id.nunique() != 3000 or cohort.hadm_id.nunique() != 3000:
        raise ValueError(
            "v3 must reuse the frozen 3000-patient/3000-admission cohort, got "
            f"subjects={cohort.subject_id.nunique()} hadm={cohort.hadm_id.nunique()}"
        )
    return cohort.sort_values(["split", "subject_id", "hadm_id"]).reset_index(drop=True)


def _write_frozen_cohort(cohort: pd.DataFrame, output_root: Path) -> None:
    for subdir in ("cohort", "splits", "manifests"):
        (output_root / subdir).mkdir(parents=True, exist_ok=True)
    cohort.to_parquet(output_root / "cohort" / "cohort.parquet", index=False)
    cohort.to_parquet(output_root / "manifests" / "cohort_3000.parquet", index=False)
    cohort.to_csv(output_root / "manifests" / "cohort_3000.csv", index=False)
    cohort[["subject_id", "hadm_id", "split"]].to_csv(
        output_root / "splits" / "splits.csv", index=False
    )


def _query_creatinine_events(
    con,
    cohort_path: Path,
    n_buckets: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    db.register_cohort(con, cohort_path, n_buckets)
    prior = con.execute(f"""
        SELECT
            c.subject_id,
            c.hadm_id AS index_hadm_id,
            pa.hadm_id AS source_hadm_id,
            l.charttime,
            l.valuenum::DOUBLE AS creatinine_mg_dl,
            coalesce(l.valueuom, 'mg/dL') AS unit
        FROM mimiciv_hosp.labevents l
        JOIN cohort c ON c.subject_id = l.subject_id
        JOIN mimiciv_hosp.admissions pa
          ON pa.hadm_id = l.hadm_id
         AND pa.subject_id = c.subject_id
         AND pa.hadm_id <> c.hadm_id
         AND pa.admittime < c.admittime
        WHERE l.itemid = {CREATININE_ITEMID}
          AND l.valuenum IS NOT NULL
          AND isfinite(l.valuenum)
          AND l.valuenum > 0
          AND l.charttime >= c.admittime - INTERVAL 365 DAY
          AND l.charttime <  c.admittime - INTERVAL 7 DAY
    """).df()
    current = con.execute(f"""
        SELECT
            c.subject_id,
            c.hadm_id,
            c.split,
            c.admittime,
            c.dischtime,
            c.n_complete_bins,
            c.bucket,
            l.hadm_id AS source_hadm_id,
            l.charttime,
            date_diff('second', c.admittime, l.charttime) / 3600.0
                AS timestamp_hours,
            floor(date_diff('second', c.admittime, l.charttime) / 3600.0 / 24.0)
                ::INTEGER AS step_idx,
            l.valuenum::DOUBLE AS creatinine_mg_dl,
            coalesce(l.valueuom, 'mg/dL') AS unit
        FROM mimiciv_hosp.labevents l
        JOIN cohort c USING (hadm_id)
        WHERE l.itemid = {CREATININE_ITEMID}
          AND l.valuenum IS NOT NULL
          AND isfinite(l.valuenum)
          AND l.valuenum > 0
          AND l.charttime >= c.admittime
          AND l.charttime < c.admittime + c.n_complete_bins * INTERVAL 24 HOUR
    """).df()
    return prior, current


def _collapse_same_time(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return events.assign(
            same_time_n_rows=[],
            same_time_n_values=[],
            same_time_conflict=[],
        )
    group_cols = ["hadm_id", "charttime"]
    rows = (
        events.groupby(group_cols, dropna=False)
        .size()
        .rename("same_time_n_rows")
        .reset_index()
    )
    values = (
        events.groupby(group_cols, dropna=False)["creatinine_mg_dl"]
        .nunique()
        .rename("same_time_n_values")
        .reset_index()
    )
    merged = events.merge(rows, on=group_cols).merge(values, on=group_cols)
    merged["same_time_conflict"] = merged["same_time_n_values"].astype(int) > 1
    # Exact duplicate rows carry no extra clinical information. If conflicting
    # values share a timestamp, keep all distinct values and mark that time RU.
    merged = merged.drop_duplicates(
        ["hadm_id", "charttime", "creatinine_mg_dl"]
    ).sort_values(["hadm_id", "timestamp_hours", "creatinine_mg_dl"])
    return merged.reset_index(drop=True)


def _build_baseline(prior: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    prior_summary = (
        prior.groupby(["subject_id", "index_hadm_id"], dropna=False)
        .agg(
            baseline_creatinine_mg_dl=("creatinine_mg_dl", "median"),
            n_prior_creatinine=("creatinine_mg_dl", "size"),
            n_prior_source_admissions=("source_hadm_id", "nunique"),
            prior_first_charttime=("charttime", "min"),
            prior_last_charttime=("charttime", "max"),
        )
        .reset_index()
        .rename(columns={"index_hadm_id": "hadm_id"})
    )
    if current.empty:
        raise ValueError("current admission creatinine table is empty")
    first_current = (
        current.sort_values(["hadm_id", "timestamp_hours", "charttime"])
        .groupby(["subject_id", "hadm_id"], as_index=False)
        .first()[["subject_id", "hadm_id", "creatinine_mg_dl", "charttime"]]
        .rename(
            columns={
                "creatinine_mg_dl": "fallback_first_admission_creatinine_mg_dl",
                "charttime": "fallback_first_admission_charttime",
            }
        )
    )
    baseline = first_current.merge(
        prior_summary, on=["subject_id", "hadm_id"], how="left"
    )
    has_prior = baseline["baseline_creatinine_mg_dl"].notna()
    baseline.loc[~has_prior, "baseline_creatinine_mg_dl"] = baseline.loc[
        ~has_prior, "fallback_first_admission_creatinine_mg_dl"
    ]
    baseline["reference_source"] = np.where(
        has_prior, "prior_7_365d_median", "admission_first_scr_fallback"
    )
    baseline["n_prior_creatinine"] = baseline["n_prior_creatinine"].fillna(0).astype(int)
    baseline["n_prior_source_admissions"] = (
        baseline["n_prior_source_admissions"].fillna(0).astype(int)
    )
    return baseline


def _stage_for_event(
    value: float,
    baseline: float,
    prior48_min: float | None,
    same_time_conflict: bool,
) -> tuple[int, str]:
    if same_time_conflict:
        return RU_CLASS_ID, "RU_CONFLICTING_SAME_TIME_RESULTS"
    if not np.isfinite(value) or not np.isfinite(baseline) or baseline <= 0:
        return RU_CLASS_ID, "RU_NO_REFERENCE"
    ratio = value / baseline
    stage = 0
    if ratio >= 3.0:
        stage = 3
    elif ratio >= 2.0:
        stage = 2
    elif ratio >= 1.5:
        stage = 1
    if prior48_min is not None and np.isfinite(prior48_min):
        if value - prior48_min >= 0.3:
            stage = max(stage, 1)
    if stage <= 0:
        return 0, "R0"
    if stage == 1:
        return 1, "R1"
    return 2, "R2"


def _annotate_events(current: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    current = _collapse_same_time(current)
    current = current.merge(
        baseline[
            [
                "hadm_id",
                "baseline_creatinine_mg_dl",
                "reference_source",
                "n_prior_creatinine",
                "n_prior_source_admissions",
            ]
        ],
        on="hadm_id",
        how="left",
        validate="many_to_one",
    )
    annotated = []
    for _hadm, group in current.groupby("hadm_id", sort=False):
        group = group.sort_values(["timestamp_hours", "creatinine_mg_dl"]).copy()
        times = group["timestamp_hours"].to_numpy(float)
        values = group["creatinine_mg_dl"].to_numpy(float)
        prior48 = []
        prior7d = []
        class_ids = []
        reasons = []
        for idx, (timestamp, value) in enumerate(zip(times, values)):
            prev_times = times[:idx]
            prev_values = values[:idx]
            before = prev_times < timestamp
            p48 = prev_values[before & (prev_times >= timestamp - 48.0)]
            p7d = prev_values[before & (prev_times >= timestamp - 168.0)]
            prior48_min = float(np.min(p48)) if len(p48) else np.nan
            prior7d_min = float(np.min(p7d)) if len(p7d) else np.nan
            class_id, reason = _stage_for_event(
                float(value),
                float(group["baseline_creatinine_mg_dl"].iloc[idx]),
                None if not np.isfinite(prior48_min) else prior48_min,
                bool(group["same_time_conflict"].iloc[idx]),
            )
            prior48.append(prior48_min)
            prior7d.append(prior7d_min)
            class_ids.append(class_id)
            reasons.append(reason)
        group["past48_min_creatinine_mg_dl"] = prior48
        group["past7d_min_creatinine_mg_dl"] = prior7d
        group["kdigo_scr_rclass_id"] = np.asarray(class_ids, dtype=np.int8)
        group["kdigo_scr_rclass"] = np.asarray(CLASS_NAMES_R, dtype=object)[class_ids]
        group["kdigo_reason"] = reasons
        annotated.append(group)
    return pd.concat(annotated, ignore_index=True) if annotated else current


def _node_grid(cohort: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in cohort.itertuples(index=False):
        for step_idx in range(int(row.n_complete_bins)):
            rows.append({
                "subject_id": int(row.subject_id),
                "hadm_id": int(row.hadm_id),
                "split": str(row.split),
                "step_idx": int(step_idx),
                "bucket_start_hours": float(step_idx * 24),
                "bucket_end_hours": float((step_idx + 1) * 24),
                "time_hours": float((step_idx + 1) * 24),
                "n_complete_bins": int(row.n_complete_bins),
            })
    return pd.DataFrame(rows)


def _max_rclass(events: pd.DataFrame) -> tuple[int, str]:
    if events.empty:
        return RU_CLASS_ID, "RU_NO_CURRENT_SCR"
    if bool(events["same_time_conflict"].any()):
        return RU_CLASS_ID, "RU_CONFLICTING_SAME_TIME_RESULTS"
    class_id = int(events["kdigo_scr_rclass_id"].max())
    return class_id, CLASS_NAMES_R[class_id]


def _future_rclass(
    events: pd.DataFrame,
    anchor_end: float,
    horizon_hours: float,
    n_complete_bins: int,
) -> tuple[int, str]:
    if anchor_end + horizon_hours > n_complete_bins * 24.0 + 1e-6:
        return RU_CLASS_ID, "RU_INCOMPLETE_FOLLOWUP"
    future = events[
        (events["timestamp_hours"] > anchor_end)
        & (events["timestamp_hours"] <= anchor_end + horizon_hours)
    ]
    if future.empty:
        return RU_CLASS_ID, "RU_NO_FUTURE_SCR"
    if bool(future["same_time_conflict"].any()):
        return RU_CLASS_ID, "RU_CONFLICTING_SAME_TIME_RESULTS"
    class_id = int(future["kdigo_scr_rclass_id"].max())
    return class_id, CLASS_NAMES_R[class_id]


def _course72(
    current_class: int,
    events: pd.DataFrame,
    anchor_end: float,
    n_complete_bins: int,
) -> tuple[int, str]:
    if current_class == RU_CLASS_ID:
        return RU_CLASS_ID, "RU_CURRENT_UNKNOWN"
    if current_class == 0:
        return 2, "NOT_CURRENT_AKI"
    if anchor_end + 72.0 > n_complete_bins * 24.0 + 1e-6:
        return RU_CLASS_ID, "RU_INCOMPLETE_FOLLOWUP"
    future = events[
        (events["timestamp_hours"] > anchor_end)
        & (events["timestamp_hours"] <= anchor_end + 72.0)
    ]
    if future.empty:
        return RU_CLASS_ID, "RU_NO_FUTURE_SCR"
    if bool(future["same_time_conflict"].any()):
        return RU_CLASS_ID, "RU_CONFLICTING_SAME_TIME_RESULTS"
    last = future.sort_values(["timestamp_hours", "creatinine_mg_dl"]).iloc[-1]
    if int(last["kdigo_scr_rclass_id"]) == 0:
        return 0, "RECOVERED"
    return 1, "PERSISTENT_OR_WORSE"


def _build_node_labels(cohort: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    grid = _node_grid(cohort)
    by_hadm = {int(hadm): g.copy() for hadm, g in events.groupby("hadm_id")}
    rows = []
    for row in grid.itertuples(index=False):
        hadm_events = by_hadm.get(
            int(row.hadm_id),
            pd.DataFrame(columns=events.columns),
        )
        current = hadm_events[
            (hadm_events["timestamp_hours"] >= row.bucket_start_hours)
            & (hadm_events["timestamp_hours"] < row.bucket_end_hours)
        ]
        current_class, current_reason = _max_rclass(current)
        future48_class, future48_reason = _future_rclass(
            hadm_events, row.bucket_end_hours, 48.0, int(row.n_complete_bins)
        )
        future72_class, future72_reason = _future_rclass(
            hadm_events, row.bucket_end_hours, 72.0, int(row.n_complete_bins)
        )
        course_class, course_reason = _course72(
            current_class, hadm_events, row.bucket_end_hours, int(row.n_complete_bins)
        )
        last_current = (
            current.sort_values(["timestamp_hours", "creatinine_mg_dl"]).iloc[-1]
            if len(current)
            else None
        )
        rows.append({
            **row._asdict(),
            "n_current_creatinine": int(len(current)),
            "current_last_creatinine_mg_dl": (
                float(last_current["creatinine_mg_dl"]) if last_current is not None else np.nan
            ),
            "current_kdigo_scr_rclass_id": int(current_class),
            "current_kdigo_scr_rclass": CLASS_NAMES_R[int(current_class)],
            "current_ru_reason": current_reason if current_class == RU_CLASS_ID else "",
            "future48_kdigo_scr_rclass_id": int(future48_class),
            "future48_kdigo_scr_rclass": CLASS_NAMES_R[int(future48_class)],
            "future48_ru_reason": future48_reason if future48_class == RU_CLASS_ID else "",
            "future72_kdigo_scr_rclass_id": int(future72_class),
            "future72_kdigo_scr_rclass": CLASS_NAMES_R[int(future72_class)],
            "future72_ru_reason": future72_reason if future72_class == RU_CLASS_ID else "",
            "aki72_course_class_id": int(course_class),
            "aki72_course_class": COURSE_CLASS_NAMES[int(course_class)],
            "aki72_course_ru_reason": course_reason if course_class == RU_CLASS_ID else "",
            "main_anchor_72h_context": bool(row.step_idx >= 2),
        })
    return pd.DataFrame(rows)


def _creatinine_text(events: pd.DataFrame) -> str:
    if events.empty:
        return ""
    pieces = []
    for event in events.itertuples(index=False):
        marker = " conflict" if bool(event.same_time_conflict) else ""
        pieces.append(
            f"{format_numeric(float(event.creatinine_mg_dl))} mg/dL at "
            f"+{float(event.timestamp_hours):g}h{marker}"
        )
    return "Creatinine measurements: " + "; ".join(pieces)


def _replace_creatinine_fragment(text: str, replacement: str) -> str:
    prefix = ""
    body = str(text)
    if " | " in body:
        head, body = body.split(" | ", 1)
        prefix = head + " | "
    parts = [part.strip() for part in re.split(r"\.\s+", body) if part.strip()]
    kept = []
    insert_at = None
    for part in parts:
        if part.startswith("Creatinine:"):
            if insert_at is None:
                insert_at = len(kept)
            continue
        kept.append(part)
    if replacement:
        if insert_at is None:
            insert_at = len(kept)
        kept.insert(insert_at, replacement)
    body_out = ". ".join(kept) if kept else "No observations recorded."
    return prefix + body_out


def _label_lookup(node_labels: pd.DataFrame, column: str) -> dict[tuple[int, int], float]:
    return {
        (int(row.hadm_id), int(row.step_idx)): float(getattr(row, column))
        for row in node_labels[["hadm_id", "step_idx", column]].itertuples(index=False)
    }


def _patch_one_shard(
    src: Path,
    dst: Path,
    creat_texts: dict[tuple[int, int], str],
    node_labels: pd.DataFrame,
) -> dict:
    shard = torch.load(src, map_location="cpu", weights_only=False)
    ph = shard["per_hour"]
    stay_ids = np.asarray(ph["stay_ids"], dtype=np.int64)
    steps = np.asarray(ph["hours"], dtype=np.int32)
    ph["state_texts"] = [
        _replace_creatinine_fragment(
            text, creat_texts.get((int(hadm), int(step)), "")
        )
        for text, hadm, step in zip(ph["state_texts"], stay_ids, steps)
    ]
    added = {
        "label_scr_aki_current_rclass": _label_lookup(
            node_labels, "current_kdigo_scr_rclass_id"
        ),
        "label_future48_scr_aki_rclass": _label_lookup(
            node_labels, "future48_kdigo_scr_rclass_id"
        ),
        "label_future72_scr_aki_rclass": _label_lookup(
            node_labels, "future72_kdigo_scr_rclass_id"
        ),
        "label_aki72_course_class": _label_lookup(
            node_labels, "aki72_course_class_id"
        ),
        "label_creatinine_count": _label_lookup(
            node_labels, "n_current_creatinine"
        ),
    }
    for name, lookup in added.items():
        shard["per_hour_labels"][name] = np.asarray(
            [lookup[(int(hadm), int(step))] for hadm, step in zip(stay_ids, steps)],
            dtype=np.float32,
        )
    shard["schema_version"] = SCHEMA_VERSION
    shard["v3_notes"] = {
        "source_shard": str(src),
        "creatinine_text_semantics": "all raw structured SCr measurements per 24h bucket",
        "reference_a": "prior 7-365 day median creatinine",
        "fallback_reference": "first creatinine in the index admission",
    }
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(shard, dst)
    return {
        "source": str(src),
        "output": str(dst),
        "n_windows": int(shard.get("n_windows", len(shard["windows"]))),
        "n_unique_hours": int(len(ph["state_texts"])),
        "n_multi_creatinine_nodes": int(
            sum(added["label_creatinine_count"][(int(h), int(s))] > 1 for h, s in zip(stay_ids, steps))
        ),
    }


def _iter_trajectory_shards(v1_root: Path) -> Iterable[tuple[str, Path]]:
    for split in ("train", "val", "test"):
        split_dir = v1_root / "trajectories" / split
        for path in sorted(split_dir.glob("*.pt")):
            yield split, path


def build(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root)
    v1_root = Path(args.v1_data_root)
    paths = db.load_yaml(args.paths_config)
    n_buckets = int(paths.get("sharding", {}).get("n_buckets", 16))
    cohort = _cohort_with_split(v1_root)
    output_root.mkdir(parents=True, exist_ok=True)
    _write_frozen_cohort(cohort, output_root)

    con = db.connect(paths)
    prior, current_raw = _query_creatinine_events(
        con, output_root / "cohort" / "cohort.parquet", n_buckets
    )
    baseline = _build_baseline(prior, current_raw)
    annotated = _annotate_events(current_raw, baseline)
    node_labels = _build_node_labels(cohort, annotated)

    manifest_dir = output_root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    prior.to_parquet(manifest_dir / "prior_7_365d_creatinine.parquet", index=False)
    current_raw.to_parquet(manifest_dir / "current_admission_creatinine_raw.parquet", index=False)
    annotated.to_parquet(manifest_dir / "current_admission_creatinine_annotated.parquet", index=False)
    baseline.to_parquet(manifest_dir / "reference_baseline.parquet", index=False)
    baseline.to_csv(manifest_dir / "reference_baseline.csv", index=False)
    node_labels.to_parquet(manifest_dir / "node_labels_v3.parquet", index=False)
    node_labels.to_csv(manifest_dir / "node_labels_v3.csv", index=False)

    annotated_for_text = annotated.sort_values(
        ["hadm_id", "step_idx", "timestamp_hours", "creatinine_mg_dl"]
    )
    creat_texts = {
        (int(hadm), int(step)): _creatinine_text(group)
        for (hadm, step), group in annotated_for_text.groupby(
            ["hadm_id", "step_idx"], sort=False
        )
    }
    out_traj = output_root / "trajectories"
    if out_traj.exists():
        shutil.rmtree(out_traj)
    shard_reports = []
    for split, src in _iter_trajectory_shards(v1_root):
        dst = out_traj / split / src.name
        shard_reports.append(_patch_one_shard(src, dst, creat_texts, node_labels))
    if not shard_reports:
        raise FileNotFoundError(f"no v1 trajectory shards under {v1_root / 'trajectories'}")
    shutil.copy2(v1_root / "trajectories" / "metadata.json", out_traj / "metadata_v1_source.json")

    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "CREATININE_WARD_V3_DATA_OK",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "v1_data_root": str(v1_root),
        "n_patients": int(cohort.subject_id.nunique()),
        "n_admissions": int(cohort.hadm_id.nunique()),
        "split_counts": cohort.split.value_counts().sort_index().to_dict(),
        "baseline_reference_counts": baseline.reference_source.value_counts().to_dict(),
        "node_current_class_counts": node_labels.current_kdigo_scr_rclass.value_counts().to_dict(),
        "node_future48_class_counts": node_labels.future48_kdigo_scr_rclass.value_counts().to_dict(),
        "node_aki72_course_counts": node_labels.aki72_course_class.value_counts().to_dict(),
        "n_current_creatinine_events_raw": int(len(current_raw)),
        "n_current_creatinine_events_distinct": int(len(annotated)),
        "n_nodes_with_multiple_current_creatinine": int((node_labels.n_current_creatinine > 1).sum()),
        "n_same_time_conflict_rows": int(annotated.same_time_conflict.sum()) if "same_time_conflict" in annotated else 0,
        "shards": shard_reports,
    }
    (output_root / "verify_creatinine_ward_v3.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    LOG.info("CREATININE_WARD_V3_DATA_OK %s", output_root)


def verify(args: argparse.Namespace) -> None:
    root = Path(args.output_root)
    report_path = root / "verify_creatinine_ward_v3.json"
    if not report_path.exists():
        raise FileNotFoundError(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "CREATININE_WARD_V3_DATA_OK", report
    assert report["n_patients"] == 3000, report
    assert report["n_admissions"] == 3000, report
    labels = pd.read_parquet(root / "manifests" / "node_labels_v3.parquet")
    assert labels.main_anchor_72h_context.any()
    assert set(labels.future48_kdigo_scr_rclass.unique()).issubset(set(CLASS_NAMES_R))
    for split in ("train", "val", "test"):
        if not list((root / "trajectories" / split).glob("*.pt")):
            raise FileNotFoundError(f"missing v3 trajectories for {split}")
    print("CREATININE_WARD_V3_VERIFY_OK", report_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["build", "verify", "all"])
    parser.add_argument(
        "--paths-config", default="creatinine_ward_paths_v1.yaml",
        help="Hospital paths config; must point at the MIMIC DuckDB.",
    )
    parser.add_argument(
        "--v1-data-root", default=os.environ.get("CREATININE_WARD_DATA_ROOT"),
        help="Frozen v1 3000-patient data root.",
    )
    parser.add_argument(
        "--output-root", default=os.environ.get("CREATININE_WARD_V3_DATA_ROOT"),
        help="New v3 data root.",
    )
    args = parser.parse_args()
    if not args.v1_data_root:
        parser.error("set --v1-data-root or CREATININE_WARD_DATA_ROOT")
    if not args.output_root:
        parser.error("set --output-root or CREATININE_WARD_V3_DATA_ROOT")
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s"
    )
    if args.command in {"build", "all"}:
        build(args)
    if args.command in {"verify", "all"}:
        verify(args)


if __name__ == "__main__":
    main()
