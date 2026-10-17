#!/usr/bin/env python3
"""Build lazy SFT indices once before multi-GPU Qwen loading."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from transformers import AutoTokenizer

from clin_jepa.training.encoder_sft import LazyPerHourSFTDataset
from clin_jepa.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--max_samples", type=int)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config(args.config)
    data = config["data"]
    model = config["model"]
    hf = config.get("hf", {})
    tokenizer = AutoTokenizer.from_pretrained(
        model["base_model"], local_files_only=hf.get("offline", False),
        cache_dir=hf.get("cache_dir"),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    save_dir = Path(os.path.expandvars(str(config["save"]["dir"])))
    index_dir = save_dir / "sft_indices"
    suffix = f"smoke{args.max_samples}" if args.max_samples else "full"
    val_max = data.get("val_max_samples")
    if args.max_samples:
        val_max = min(args.max_samples, val_max or args.max_samples)
    common = dict(
        tokenizer=tokenizer,
        max_seq_len=int(data["max_seq_len"]),
        skip_empty=bool(data.get("skip_empty", True)),
        no_action_fraction=float(data.get("no_action_sft_fraction", 0.10)),
        seed=int(config["training"].get("seed", 42)),
    )
    LazyPerHourSFTDataset(
        data["train_dir"], text_source="both", max_samples=args.max_samples,
        index_cache=index_dir / f"train_both_{suffix}.pt", **common,
    )
    LazyPerHourSFTDataset(
        data["val_dir"], text_source="state", max_samples=val_max,
        index_cache=index_dir / f"val_state_{suffix}.pt", **common,
    )
    LazyPerHourSFTDataset(
        data["val_dir"], text_source="action", max_samples=val_max,
        index_cache=index_dir / f"val_action_{suffix}.pt", **common,
    )
    print("SFT_INDEX_OK", index_dir)


if __name__ == "__main__":
    main()
