#!/usr/bin/env python3
"""Build DENSE_TRAJ shards: consecutive observed segments with length >= 4."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from clin_jepa.data_hosp.episode_windows import dense_observed_segments

logger = logging.getLogger(__name__)


def densify_shard(shard: dict, min_len: int) -> dict | None:
    ph = shard["per_hour"]
    windows = []
    for w in shard["windows"]:
        hi = np.asarray(w["hour_indices"])
        observed = np.asarray(ph["has_measured"])[hi]
        for start, length in dense_observed_segments(observed, min_len=min_len):
            windows.append({
                **w,
                "start_step": int(w["start_step"] + start),
                "length": int(length),
                "hour_indices": hi[start:start + length].astype(np.int32),
                "pool": "dense",
            })
    if not windows:
        return None
    out = dict(shard)
    out["windows"] = windows
    out["n_windows"] = len(windows)
    out["pool"] = "dense"
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, help="episode trajectories root")
    parser.add_argument("--dst", required=True, help="dense trajectories root")
    parser.add_argument("--min-len", type=int, default=4)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    src, dst = Path(args.src), Path(args.dst)
    report = {}
    for split in ("train", "val", "test"):
        split_src = src / split
        if not split_src.exists():
            continue
        split_dst = dst / split
        split_dst.mkdir(parents=True, exist_ok=True)
        for stale in split_dst.glob("*.pt"):
            stale.unlink()
        n_win = n_shard = 0
        for path in sorted(split_src.glob("*.pt")):
            shard = torch.load(path, map_location="cpu", weights_only=False)
            dense = densify_shard(shard, min_len=args.min_len)
            if dense is None:
                continue
            torch.save(dense, split_dst / path.name)
            n_shard += 1
            n_win += dense["n_windows"]
        report[split] = {"shards": n_shard, "windows": n_win}
        logger.info("%s %s", split, report[split])
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "dense_report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
