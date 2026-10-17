#!/usr/bin/env python3
"""Build lightweight dense/gap window indices before distributed JEPA."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from clin_jepa.training.trajectory_dataset import TrajectoryWindowDataset
from clin_jepa.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--max_windows", type=int)
    parser.add_argument("--save_dir")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config(args.config)
    data = config["data"]
    save_dir = Path(os.path.expandvars(args.save_dir or str(config["save"]["dir"])))
    index_dir = save_dir / "trajectory_indices"
    suffix = f"smoke{args.max_windows}" if args.max_windows else "full"
    minimum = int(data.get("min_length", 3))
    TrajectoryWindowDataset(
        data["train_dir"], "train", args.max_windows, minimum,
        index_cache=index_dir / f"dense_train_{suffix}.pt",
    )
    TrajectoryWindowDataset(
        data["val_dir"], "val",
        args.max_windows or config["logging"].get("val_windows"), minimum,
        index_cache=index_dir / f"dense_val_{suffix}.pt",
    )
    TrajectoryWindowDataset(
        data.get("gap_train_dir", data["train_dir"]), "train",
        args.max_windows, minimum,
        index_cache=index_dir / f"gap_train_{suffix}.pt",
    )
    print("TRAJECTORY_INDEX_OK", index_dir)


if __name__ == "__main__":
    main()
