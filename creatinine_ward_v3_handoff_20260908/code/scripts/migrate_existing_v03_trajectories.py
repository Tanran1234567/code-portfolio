#!/usr/bin/env python3
"""Audit and migrate existing v0.3 trajectory shards without re-querying MIMIC.

The source may be old mixed-admission trajectories or already split episode
trajectories. The migration is non-destructive by default and writes three
verified pools below a new MEDIN_DATA_ROOT:

  trajectories_episode/{train,val,test}
  trajectories_dense/{train,val,test}
  trajectories_gap/{train,val,test}

Model time is episode-relative. Missing numeric measurements are never imputed.
State availability and action presence are independent booleans. Canonical
NO_ACTION remains a text condition encoded by the shared Clin-JEPA encoder.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from clin_jepa.data_hosp.episode_windows import (
    normalize_action_text,
    normalize_state_text,
)
from build_dense_segments_v2 import densify_shard
from build_gap_trajectories_v2 import gapify_shard
from convert_mixed_to_episodes_v2 import convert_shard

logger = logging.getLogger(__name__)


def _atomic_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def _is_episode_shard(shard: dict) -> bool:
    ph = shard.get("per_hour", {})
    ps = shard.get("per_stay", {})
    schema = str(shard.get("schema_version", "")).lower()
    return (
        "episode" in schema
        or "episode_types" in ph
        or "episode_types" in ps
        or "source_types" in ps
    )


def _derive_episode_time(ph: dict) -> np.ndarray:
    stay_ids = np.asarray(ph["stay_ids"], dtype=np.int64)
    starts = np.asarray(ph["start_hours"], dtype=np.float64)
    ends = np.asarray(ph["end_hours"], dtype=np.float64)
    result = np.empty(len(stay_ids), dtype=np.float32)
    for stay_id in np.unique(stay_ids):
        idx = np.flatnonzero(stay_ids == stay_id)
        episode_start = float(starts[idx].min())
        result[idx] = (ends[idx] - episode_start).astype(np.float32)
    return result


def _migrate_episode_shard(shard: dict) -> tuple[dict, dict]:
    out = dict(shard)
    ph = dict(shard["per_hour"])
    n = len(ph["state_texts"])
    if "state_observed" not in ph and "has_measured" not in ph:
        raise ValueError(
            "old shard lacks has_measured/state_observed; refusing unsafe migration"
        )
    observed = np.asarray(
        ph.get("state_observed", ph.get("has_measured")), dtype=np.bool_,
    )
    existing_action = ph.get("action_present", ph.get("has_action"))
    existing_action = (
        np.asarray(existing_action, dtype=np.bool_) if existing_action is not None else None
    )
    action_counts = ph.get("action_event_count", ph.get("action_event_counts"))
    action_counts = (
        np.asarray(action_counts, dtype=np.int64) if action_counts is not None else None
    )

    state_texts: list[str] = []
    action_texts: list[str] = []
    action_present = np.zeros(n, dtype=np.bool_)
    action_event_count = np.zeros(n, dtype=np.int32)
    mismatch_examples: list[dict] = []
    text_action_present = np.zeros(n, dtype=np.bool_)
    machine_prose = 0

    for i, (state_text, action_text) in enumerate(
        zip(ph["state_texts"], ph["action_texts"])
    ):
        raw_state = str(state_text)
        raw_action = str(action_text)
        if "Care setting resolution:" in raw_state or "Interval duration:" in raw_state:
            machine_prose += 1
        state_texts.append(normalize_state_text(raw_state, bool(observed[i])))
        normalized_action, text_present = normalize_action_text(raw_action)
        action_texts.append(normalized_action)
        text_action_present[i] = text_present
        if action_counts is not None and int(action_counts[i]) >= 0:
            present = int(action_counts[i]) > 0
            action_event_count[i] = int(action_counts[i])
        else:
            present = text_present
            action_event_count[i] = -1 if present else 0
        action_present[i] = present
        if existing_action is not None and bool(existing_action[i]) != bool(present):
            if len(mismatch_examples) < 100:
                mismatch_examples.append({
                    "index": i,
                    "existing": bool(existing_action[i]),
                    "migrated": bool(present),
                    "raw_action": raw_action[:300],
                    "normalized_action": normalized_action[:300],
                })

    episode_time = _derive_episode_time(ph)
    old_time = np.asarray(ph.get("time_hours", episode_time), dtype=np.float32)
    ph.update({
        "time_hours": episode_time.copy(),
        "episode_time_hours": episode_time.copy(),
        "state_observed": observed.copy(),
        "has_measured": observed.copy(),
        "action_present": action_present.copy(),
        "has_action": action_present.copy(),
        "action_event_count": action_event_count,
        "state_texts": state_texts,
        "action_texts": action_texts,
    })
    out["per_hour"] = ph
    out["schema_version"] = "medin_v2_episode_v4_episode_time_action_sentinel"
    out["time_semantics"] = "bucket_end_hours_from_care_episode_start"
    out["action_presence_semantics"] = "modeled_action_present_in_bucket"

    quadrants = Counter()
    for state_value, action_value in zip(observed, action_present):
        quadrants[f"S{int(state_value)}A{int(action_value)}"] += 1
    report = {
        "nodes": n,
        "quadrants": dict(quadrants),
        "machine_prose_removed": machine_prose,
        "old_vs_new_time_changed": int((np.abs(old_time - episode_time) > 1e-4).sum()),
        "existing_action_mismatches": (
            int((existing_action != action_present).sum())
            if existing_action is not None else None
        ),
        "structured_vs_text_action_mismatches": (
            int((action_present != text_action_present).sum())
            if action_counts is not None else None
        ),
        "mismatch_examples": mismatch_examples,
    }
    return out, report


def _quadrant_report(shard: dict) -> dict:
    ph = shard["per_hour"]
    state = np.asarray(ph["state_observed"], dtype=np.bool_)
    action = np.asarray(ph["action_present"], dtype=np.bool_)
    counter = Counter(
        f"S{int(s)}A{int(a)}" for s, a in zip(state, action)
    )
    return {"nodes": len(state), "quadrants": dict(counter)}


def _source_action_audit(shard: dict) -> dict:
    """Audit old flags before mixed-to-episode conversion discards them."""
    ph = shard["per_hour"]
    text_present = np.asarray(
        [normalize_action_text(text)[1] for text in ph["action_texts"]],
        dtype=np.bool_,
    )
    existing = ph.get("action_present", ph.get("has_action"))
    counts = ph.get("action_event_count", ph.get("action_event_counts"))
    mismatch_examples: list[dict] = []
    if existing is not None:
        existing_array = np.asarray(existing, dtype=np.bool_)
        for idx in np.flatnonzero(existing_array != text_present)[:100]:
            normalized, migrated = normalize_action_text(ph["action_texts"][int(idx)])
            mismatch_examples.append({
                "index": int(idx),
                "existing": bool(existing_array[idx]),
                "migrated": bool(migrated),
                "raw_action": str(ph["action_texts"][int(idx)])[:300],
                "normalized_action": normalized[:300],
            })
    return {
        "source_existing_action_mismatches": (
            int((np.asarray(existing, dtype=np.bool_) != text_present).sum())
            if existing is not None else 0
        ),
        "source_structured_vs_text_action_mismatches": (
            int(((np.asarray(counts) > 0) != text_present).sum())
            if counts is not None and np.all(np.asarray(counts) >= 0) else 0
        ),
        "mismatch_examples": mismatch_examples,
    }


def _merge_stats(total: dict, item: dict) -> None:
    total["shards"] += 1
    total["nodes"] += int(item.get("nodes", 0))
    for key, value in item.get("quadrants", {}).items():
        total["quadrants"][key] += int(value)
    for key in (
        "machine_prose_removed",
        "old_vs_new_time_changed",
        "existing_action_mismatches",
        "structured_vs_text_action_mismatches",
        "source_existing_action_mismatches",
        "source_structured_vs_text_action_mismatches",
    ):
        value = item.get(key)
        if value is not None:
            total[key] += int(value)
    remaining = 100 - len(total["mismatch_examples"])
    if remaining > 0:
        total["mismatch_examples"].extend(item.get("mismatch_examples", [])[:remaining])


def migrate_split(
    src_dir: Path,
    dst_root: Path,
    split: str,
    input_kind: str,
    audit_only: bool,
    resume: bool,
    max_shards: int | None,
    t_max: int,
    min_len: int,
    dense_min_len: int,
) -> dict:
    files = sorted(src_dir.glob("*.pt"))
    if max_shards is not None:
        files = files[:max_shards]
    if not files:
        raise FileNotFoundError(f"no .pt shards in {src_dir}")
    total = {
        "shards": 0,
        "nodes": 0,
        "quadrants": defaultdict(int),
        "machine_prose_removed": 0,
        "old_vs_new_time_changed": 0,
        "existing_action_mismatches": 0,
        "structured_vs_text_action_mismatches": 0,
        "source_existing_action_mismatches": 0,
        "source_structured_vs_text_action_mismatches": 0,
        "mismatch_examples": [],
    }
    for index, path in enumerate(files, start=1):
        episode_path = dst_root / "trajectories_episode" / split / path.name
        dense_path = dst_root / "trajectories_dense" / split / path.name
        dense_empty_marker = dense_path.with_suffix(".dense_empty")
        gap_path = dst_root / "trajectories_gap" / split / path.name
        complete_marker = episode_path.with_suffix(".migration_v4_complete")
        if (
            resume and not audit_only and complete_marker.exists()
            and episode_path.exists() and gap_path.exists()
            and (dense_path.exists() or dense_empty_marker.exists())
        ):
            logger.info("SKIP complete %s", path.name)
            continue
        shard = torch.load(path, map_location="cpu", weights_only=False)
        kind = input_kind
        if kind == "auto":
            kind = "episode" if _is_episode_shard(shard) else "mixed"
        if kind == "mixed":
            source_action_audit = _source_action_audit(shard)
            episode = convert_shard(shard, t_max=t_max, min_len=min_len)
            if episode is None:
                logger.warning("no eligible episode windows in %s", path)
                continue
            episode, report = _migrate_episode_shard(episode)
            report.update(source_action_audit)
        else:
            episode, report = _migrate_episode_shard(shard)
        _merge_stats(total, report)
        if not audit_only:
            dense = densify_shard(episode, min_len=dense_min_len)
            gap = gapify_shard(episode, min_len=min_len)
            if gap is None:
                raise RuntimeError(f"migration produced empty gap pool for {path}")
            _atomic_save(episode, episode_path)
            if dense is not None:
                _atomic_save(dense, dense_path)
            else:
                dense_empty_marker.parent.mkdir(parents=True, exist_ok=True)
                dense_empty_marker.write_text("no observed segment reached dense_min_len\n")
            _atomic_save(gap, gap_path)
            complete_marker.write_text(
                "medin_v2_episode_v4_episode_time_action_sentinel\n",
                encoding="utf-8",
            )
        if index % 10 == 0 or index == len(files):
            logger.info("%s %d/%d shards", split, index, len(files))
    total["quadrants"] = dict(total["quadrants"])
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-root", required=True, help="root containing train/val/test")
    parser.add_argument("--dst-root", required=True, help="new non-destructive MEDIN_DATA_ROOT")
    parser.add_argument("--input-kind", choices=["auto", "mixed", "episode"], default="auto")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-shards", type=int)
    parser.add_argument("--t-max", type=int, default=72)
    parser.add_argument("--min-len", type=int, default=3)
    parser.add_argument("--dense-min-len", type=int, default=4)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    src_root = Path(args.src_root).resolve()
    dst_root = Path(args.dst_root).resolve()
    if src_root == dst_root:
        raise ValueError("src-root and dst-root must differ; in-place migration is forbidden")
    report = {
        "schema_version": "medin_v03_existing_shard_migration_v2",
        "source": str(src_root),
        "destination": str(dst_root),
        "input_kind": args.input_kind,
        "audit_only": args.audit_only,
        "splits": {},
    }
    for split in ("train", "val", "test"):
        split_dir = src_root / split
        if not split_dir.exists():
            logger.warning("missing split %s", split_dir)
            continue
        report["splits"][split] = migrate_split(
            split_dir, dst_root, split, args.input_kind, args.audit_only,
            args.resume, args.max_shards, args.t_max, args.min_len,
            args.dense_min_len,
        )
    missing_splits = {"train", "val", "test"} - set(report["splits"])
    if missing_splits:
        raise FileNotFoundError(f"source trajectory splits missing: {sorted(missing_splits)}")
    if not args.audit_only:
        source_split_dir = src_root.parent / "splits"
        source_split_csv = source_split_dir / "splits.csv"
        if not source_split_csv.exists():
            raise FileNotFoundError(
                f"patient split provenance missing: {source_split_csv}"
            )
        destination_split_dir = dst_root / "splits"
        destination_split_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_split_csv, destination_split_dir / "splits.csv")
        split_stats = source_split_dir / "split_stats.json"
        if split_stats.exists():
            shutil.copy2(split_stats, destination_split_dir / "split_stats.json")
        report["patient_split_source"] = str(source_split_csv.resolve())
    report_path = (
        dst_root / ("migration_audit.json" if args.audit_only else "migration_report.json")
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    structured_mismatches = sum(
        int(item.get("structured_vs_text_action_mismatches", 0))
        + int(item.get("source_structured_vs_text_action_mismatches", 0))
        for item in report["splits"].values()
    )
    if structured_mismatches:
        raise SystemExit(
            f"FAIL: {structured_mismatches} nodes disagree between structured "
            "action counts and normalized action text; inspect audit before training"
        )
    print("MIGRATION_OK", report_path)


if __name__ == "__main__":
    main()
