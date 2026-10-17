#!/usr/bin/env python3
"""Build FULL_GAP_TRAJ shards: full episode windows including natural empty."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


def gapify_shard(shard: dict, min_len: int) -> dict | None:
    windows = []
    for w in shard["windows"]:
        if int(w["length"]) < min_len:
            continue
        windows.append({**w, "pool": "gap"})
    if not windows:
        return None
    out = dict(shard)
    out["windows"] = windows
    out["n_windows"] = len(windows)
    out["pool"] = "gap"
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, help="episode trajectories root")
    parser.add_argument("--dst", required=True, help="gap trajectories root")
    parser.add_argument("--min-len", type=int, default=3)
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
            gap = gapify_shard(shard, min_len=args.min_len)
            if gap is None:
                continue
            torch.save(gap, split_dst / path.name)
            n_shard += 1
            n_win += gap["n_windows"]
        report[split] = {"shards": n_shard, "windows": n_win}
        logger.info("%s %s", split, report[split])
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "gap_report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
