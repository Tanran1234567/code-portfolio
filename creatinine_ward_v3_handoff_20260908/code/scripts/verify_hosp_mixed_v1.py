#!/usr/bin/env python3
"""Verify mixed-resolution hospital data and leakage guards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="/home/zhoujiale/clin_jepa_hosp_data_v1_mixed",
    )
    args = parser.parse_args()
    root = Path(args.root)

    stats = json.loads((root / "discretized/discretize_stats.json").read_text())
    assert stats["mode"] == "mixed_ward_24h_icu_1h"
    assert stats["n_retained"] > 0

    node_parts = sorted((root / "discretized/nodes").glob("bucket=*/*.parquet"))
    assert node_parts, "No node parquet files"
    nodes = pd.concat([pd.read_parquet(path) for path in node_parts], ignore_index=True)
    assert (nodes["duration_hours"] > 0).all()
    assert (nodes.loc[nodes["is_icu"], "duration_hours"] <= 1.000001).all()
    assert (nodes.loc[~nodes["is_icu"], "duration_hours"] <= 24.000001).all()

    for _, group in nodes.groupby("hadm_id"):
        group = group.sort_values("step_idx")
        np.testing.assert_allclose(
            group["node_end_hours"].to_numpy()[:-1],
            group["node_start_hours"].to_numpy()[1:],
            atol=1e-5,
        )

    shards = sorted((root / "trajectories/train").glob("*.pt"))
    assert shards, "No trajectory shards"
    shard = torch.load(shards[0], map_location="cpu", weights_only=False)
    ph = shard["per_hour"]
    for key in (
        "start_hours", "end_hours", "duration_hours", "is_icu",
        "has_measured", "hours_since_last_measurement",
    ):
        assert key in ph, f"Missing per_hour.{key}"
        assert len(ph[key]) == len(ph["state_texts"])

    texts = ph["state_texts"]
    assert any("Hours since last measured observation:" in text for text in texts)
    assert any("Admission context:" in text for text in texts)
    assert not any("Diagnosis:" in text for text in texts), (
        "Current-admission discharge ICD leaked into state text"
    )
    assert any("Care setting resolution: ICU 1h" in text for text in texts)
    assert any("Care setting resolution: ward 24h" in text for text in texts)

    print(json.dumps({
        "status": "ok",
        "nodes": len(nodes),
        "icu_nodes": int(nodes["is_icu"].sum()),
        "ward_nodes": int((~nodes["is_icu"]).sum()),
        "checked_shard": str(shards[0]),
        "sample_state": texts[0][:500],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
