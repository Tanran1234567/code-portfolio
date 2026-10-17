#!/usr/bin/env python3
"""Convert v0.1 mixed-admission trajectory shards into v0.3 episode shards.

Each mixed hadm sequence is split into WARD_PRE_ICU / ICU / WARD_POST_ICU
(or WARD_ONLY). Model time remains the bucket-end elapsed from admission start;
episode-local time is metadata only. Output shards keep the v0.1
per_hour/per_stay/windows layout plus source_type metadata.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from clin_jepa.data_hosp.episode_windows import (
    normalize_action_text,
    normalize_state_text,
    slice_episode_windows,
    split_care_episodes,
)

logger = logging.getLogger(__name__)

def _hours_for_stay(ph: dict, stay_id: int) -> np.ndarray:
    stay_ids = np.asarray(ph["stay_ids"])
    return np.where(stay_ids == stay_id)[0]


def convert_shard(shard: dict, t_max: int, min_len: int) -> dict | None:
    ph = shard["per_hour"]
    ps = shard["per_stay"]
    phl = shard.get("per_hour_labels", {})
    label_names = list(phl.keys())

    new_ph = {
        "stay_ids": [],
        "hours": [],
        "start_hours": [],
        "end_hours": [],
        "duration_hours": [],
        "time_hours": [],
        "episode_time_hours": [],
        "is_icu": [],
        "has_measured": [],
        "state_observed": [],
        "has_action": [],
        "action_present": [],
        "action_event_count": [],
        "hours_since_last_measurement": [],
        "state_texts": [],
        "action_texts": [],
        "source_types": [],
        "episode_types": [],
    }
    new_phl = {n: [] for n in label_names}
    new_ps_rows: list[dict] = []
    windows: list[dict] = []
    stay_key_to_index: dict[int, int] = {}

    n_stays = len(ps["stay_ids"])
    episode_seq = 0
    for si in range(n_stays):
        stay_id = int(ps["stay_ids"][si])
        idx = _hours_for_stay(ph, stay_id)
        if len(idx) == 0:
            continue
        order = np.argsort(ph["start_hours"][idx] if "start_hours" in ph else ph["hours"][idx])
        idx = idx[order]
        starts = np.asarray(ph.get("start_hours", ph["hours"])[idx], dtype=np.float64)
        if "end_hours" in ph:
            ends = np.asarray(ph["end_hours"][idx], dtype=np.float64)
        elif "duration_hours" in ph:
            ends = starts + np.asarray(ph["duration_hours"][idx], dtype=np.float64)
        else:
            ends = starts + 1.0
        is_icu = np.asarray(ph.get("is_icu", np.zeros(len(ph["hours"]), dtype=np.bool_))[idx])
        if "has_measured" not in ph and "state_observed" not in ph:
            raise ValueError(
                "v0.3 migration requires has_measured/state_observed; "
                "refusing to assume every state is observed"
            )
        measured_all = ph.get("state_observed", ph.get("has_measured"))
        measured = np.asarray(measured_all)[idx]
        episodes = split_care_episodes(starts, ends, is_icu, measured)

        local_ep = 0
        for ep in episodes:
            if ep["source_type"] == "ICU":
                icu_hours = float(ep["end_hours"][-1] - ep["start_hours"][0])
                if icu_hours < 6.0 - 1e-6:
                    continue
            lo, hi = ep["index_lo"], ep["index_hi"]
            local = idx[lo:hi]
            ep_stay_id = stay_id * 1000 + local_ep
            local_ep += 1
            episode_seq += 1
            if ep_stay_id not in stay_key_to_index:
                stay_key_to_index[ep_stay_id] = len(new_ps_rows)
                new_ps_rows.append({
                    "stay_id": ep_stay_id,
                    "hadm_id": stay_id,
                    "subject_id": int(ps["subject_ids"][si]),
                    "age": float(ps["ages"][si]),
                    "gender": ps["genders"][si],
                    "race": ps["races"][si],
                    "charlson_ci": int(ps["charlson_cis"][si]),
                    "hospital_mortality": bool(ps["hospital_mortalities"][si]),
                    "icu_mortality": bool(ps["icu_mortalities"][si]),
                    "prolonged_stay": bool(ps["prolonged_stays"][si]),
                    "suspected_infection": bool(ps["sepsis3s"][si]),
                    "source_type": ep["source_type"],
                    "episode_type": ep["episode_type"],
                })
            stay_index = stay_key_to_index[ep_stay_id]
            hour_indices = []
            for j, src in enumerate(local):
                hour_indices.append(len(new_ph["state_texts"]))
                new_ph["stay_ids"].append(ep_stay_id)
                new_ph["hours"].append(j)
                new_ph["start_hours"].append(float(ep["start_hours"][j]))
                new_ph["end_hours"].append(float(ep["end_hours"][j]))
                new_ph["duration_hours"].append(float(ep["duration_hours"][j]))
                new_ph["time_hours"].append(float(ep["time_hours"][j]))
                new_ph["episode_time_hours"].append(float(ep["episode_time_hours"][j]))
                new_ph["is_icu"].append(bool(ep["is_icu"][j]))
                new_ph["has_measured"].append(bool(ep["has_measured"][j]))
                new_ph["state_observed"].append(bool(ep["has_measured"][j]))
                since = ph.get("hours_since_last_measurement")
                new_ph["hours_since_last_measurement"].append(
                    float(since[src]) if since is not None else np.nan
                )
                state_text = normalize_state_text(
                    ph["state_texts"][src], bool(ep["has_measured"][j])
                )
                action_text, action_present = normalize_action_text(
                    ph["action_texts"][src]
                )
                new_ph["state_texts"].append(state_text)
                new_ph["action_texts"].append(action_text)
                new_ph["has_action"].append(action_present)
                new_ph["action_present"].append(action_present)
                counts = ph.get("action_event_count", ph.get("action_event_counts"))
                new_ph["action_event_count"].append(
                    int(counts[src]) if counts is not None else (-1 if action_present else 0)
                )
                new_ph["source_types"].append(ep["source_type"])
                new_ph["episode_types"].append(ep["episode_type"])
                for name in label_names:
                    new_phl[name].append(float(phl[name][src]))

            for start, length in slice_episode_windows(
                ep["n_steps"], ep["start_hours"], ep["end_hours"],
                ep["source_type"], t_max=t_max, min_len=min_len,
            ):
                windows.append({
                    "stay_index": stay_index,
                    "start_step": int(start),
                    "length": int(length),
                    "hour_indices": np.asarray(hour_indices[start:start + length], dtype=np.int32),
                    "source_type": ep["source_type"],
                    "episode_type": ep["episode_type"],
                })

    if not windows:
        return None

    def col(key, dtype):
        return np.asarray([r[key] for r in new_ps_rows], dtype=dtype)

    return {
        "per_hour": {
            "stay_ids": np.asarray(new_ph["stay_ids"], dtype=np.int64),
            "hours": np.asarray(new_ph["hours"], dtype=np.int32),
            "start_hours": np.asarray(new_ph["start_hours"], dtype=np.float32),
            "end_hours": np.asarray(new_ph["end_hours"], dtype=np.float32),
            "duration_hours": np.asarray(new_ph["duration_hours"], dtype=np.float32),
            "time_hours": np.asarray(new_ph["time_hours"], dtype=np.float32),
            "episode_time_hours": np.asarray(
                new_ph["episode_time_hours"], dtype=np.float32,
            ),
            "is_icu": np.asarray(new_ph["is_icu"], dtype=np.bool_),
            "has_measured": np.asarray(new_ph["has_measured"], dtype=np.bool_),
            "state_observed": np.asarray(new_ph["state_observed"], dtype=np.bool_),
            "has_action": np.asarray(new_ph["has_action"], dtype=np.bool_),
            "action_present": np.asarray(new_ph["action_present"], dtype=np.bool_),
            "action_event_count": np.asarray(
                new_ph["action_event_count"], dtype=np.int32,
            ),
            "hours_since_last_measurement": np.asarray(
                new_ph["hours_since_last_measurement"], dtype=np.float32,
            ),
            "state_texts": list(new_ph["state_texts"]),
            "action_texts": list(new_ph["action_texts"]),
            "source_types": list(new_ph["source_types"]),
            "episode_types": list(new_ph["episode_types"]),
        },
        "per_hour_labels": {
            n: np.asarray(new_phl[n], dtype=np.float32) for n in label_names
        },
        "per_stay": {
            "stay_ids": col("stay_id", np.int64),
            "subject_ids": col("subject_id", np.int64),
            "hadm_ids": col("hadm_id", np.int64),
            "ages": col("age", np.float64),
            "genders": [r["gender"] for r in new_ps_rows],
            "races": [r["race"] for r in new_ps_rows],
            "charlson_cis": col("charlson_ci", np.int32),
            "icu_mortalities": col("icu_mortality", np.bool_),
            "hospital_mortalities": col("hospital_mortality", np.bool_),
            "prolonged_stays": col("prolonged_stay", np.bool_),
            "sepsis3s": col("suspected_infection", np.bool_),
            "source_types": [r["source_type"] for r in new_ps_rows],
            "episode_types": [r["episode_type"] for r in new_ps_rows],
        },
        "windows": windows,
        "schema_version": "medin_v2_episode_v4_episode_time_action_sentinel",
        "time_semantics": "bucket_end_hours_from_care_episode_start",
        "action_presence_semantics": "modeled_action_present_in_bucket",
        "n_windows": len(windows),
        "n_unique_hours": len(new_ph["state_texts"]),
        "n_unique_stays": len(new_ps_rows),
    }


def convert_split(src: Path, dst: Path, t_max: int, min_len: int) -> dict:
    dst.mkdir(parents=True, exist_ok=True)
    for stale in dst.glob("*.pt"):
        stale.unlink()
    stats = defaultdict(int)
    for path in sorted(src.glob("*.pt")):
        shard = torch.load(path, map_location="cpu", weights_only=False)
        converted = convert_shard(shard, t_max=t_max, min_len=min_len)
        if converted is None:
            stats["empty_shards"] += 1
            continue
        torch.save(converted, dst / path.name)
        stats["shards"] += 1
        stats["windows"] += converted["n_windows"]
        stats["hours"] += converted["n_unique_hours"]
        stats["episodes"] += converted["n_unique_stays"]
    return dict(stats)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, help="v0.1 mixed trajectories root")
    parser.add_argument("--dst", required=True, help="v0.3 episode trajectories root")
    parser.add_argument("--t-max", type=int, default=72)
    parser.add_argument("--min-len", type=int, default=3)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    src = Path(args.src)
    dst = Path(args.dst)
    report = {}
    for split in ("train", "val", "test"):
        split_src = src / split
        if not split_src.exists():
            logger.warning("missing split %s", split_src)
            continue
        report[split] = convert_split(split_src, dst / split, args.t_max, args.min_len)
        logger.info("%s %s", split, report[split])
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "convert_report.json").write_text(json.dumps(report, indent=2))
    logger.info("wrote %s", dst / "convert_report.json")


if __name__ == "__main__":
    main()
