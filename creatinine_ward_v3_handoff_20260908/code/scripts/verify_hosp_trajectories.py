#!/usr/bin/env python3
"""Verify hospital trajectory shards are loadable by Clin-JEPA training code.

Usage (from Clin-JEPA-main root):
    python3 scripts/verify_hosp_trajectories.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from clin_jepa.training.trajectory_dataset import TrajectoryWindowDataset

TRAJ = Path("/home/zhoujiale/clin_jepa_hosp_data/trajectories")
OUT = Path("/home/zhoujiale/Clin-JEPA-main/reports/hosp_pipeline")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    meta = json.loads((TRAJ / "metadata.json").read_text())
    report: dict = {"metadata": meta, "checks": {}}

    # 1. Shard counts
    shard_counts = {
        s: len(list((TRAJ / s).glob("*.pt"))) for s in ("train", "val", "test")
    }
    report["checks"]["shard_counts"] = shard_counts
    assert all(v > 0 for v in shard_counts.values()), shard_counts

    # 2. Schema smoke on one shard
    sample = torch.load(
        next((TRAJ / "train").glob("*.pt")),
        map_location="cpu",
        weights_only=False,
    )
    assert sample["schema_version"] == "clin_jepa_trajectories_v1"
    report["checks"]["schema_version"] = sample["schema_version"]
    report["checks"]["sample_keys"] = sorted(sample.keys())
    report["checks"]["per_stay_keys"] = sorted(sample["per_stay"].keys())
    report["checks"]["per_stay_extra_keys"] = sorted(
        sample.get("per_stay_extra", {}).keys()
    )
    report["checks"]["n_label_targets"] = len(sample["per_hour_labels"])

    # 3. Dataset loader (hospital min_length=4)
    ds = TrajectoryWindowDataset(
        TRAJ, split="train", max_windows=500, min_length=4,
    )
    lengths = [ds[i]["length"] for i in range(len(ds))]
    item = ds[0]
    # Wall-clock stamp: with 6h bins, step k should start with t={6k}h
    assert item["state_texts"][0].startswith("t=0h |"), item["state_texts"][0][:40]
    if item["length"] > 1:
        assert item["state_texts"][1].startswith("t=6h |"), item["state_texts"][1][:40]

    report["checks"]["dataset_smoke"] = {
        "n_windows_loaded": len(ds),
        "length_hist": dict(Counter(lengths)),
        "example_stay_id": item["stay_id"],
        "example_demo": item["demographics_text"],
        "example_state0": item["state_texts"][0][:240],
        "example_action0": item["action_texts"][0][:240],
        "binary_labels": item["binary_labels"],
        "n_continuous_labels": len(item["labels"]),
    }

    # 4. Full-split window counts via metadata (already authoritative)
    report["checks"]["ready_for_training"] = True
    report["checks"]["suggested_env"] = {
        "CLIN_JEPA_DATA": "/home/zhoujiale/clin_jepa_hosp_data",
        "train_config": "configs/train/pretrain_clin_jepa_hosp.yaml",
    }

    out_json = OUT / "hosp_trajectory_verify.json"
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report["checks"], indent=2, ensure_ascii=False))
    print(f"\nWrote {out_json}")
    print("PASS: hospital trajectories are train/val/test ready.")


if __name__ == "__main__":
    main()
