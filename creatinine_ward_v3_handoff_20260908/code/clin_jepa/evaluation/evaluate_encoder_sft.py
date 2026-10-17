"""Full validation/test language-model evaluation for Hospital-SFT LoRA."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorWithFlattening

from clin_jepa.training.encoder_sft import (
    PerHourSFTDataset,
    create_model,
    evaluate,
)
from clin_jepa.utils import load_config, setup_logging


def _evaluate_adapter(
    config: dict,
    adapter: str,
    split: str,
    device: torch.device,
) -> dict:
    local_config = copy.deepcopy(config)
    local_config["model"]["initial_adapter"] = adapter
    model, tokenizer = create_model(local_config, device)
    model.eval()
    split_dir = Path(config["sft_evaluation"]["trajectory_dir"]) / split
    collator = DataCollatorWithFlattening(return_flash_attn_kwargs=True)
    losses = {}
    counts = {}
    for source in ("state", "action"):
        dataset = PerHourSFTDataset(
            split_dir,
            tokenizer,
            max_seq_len=config["data"]["max_seq_len"],
            text_source=source,
            max_samples=None,
            compute_lengths=False,
        )
        loader = DataLoader(
            dataset,
            batch_size=int(config["sft_evaluation"].get("batch_size", 4)),
            shuffle=False,
            collate_fn=collator,
            num_workers=int(config["sft_evaluation"].get("num_workers", 2)),
        )
        losses[source] = evaluate(model, loader, device, max_batches=None)
        counts[source] = len(dataset)
    combined = (losses["state"] + losses["action"]) / 2.0
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "split": split,
        "is_full_split": True,
        "n_samples": counts,
        "loss_state": losses["state"],
        "loss_action": losses["action"],
        "loss_combined": combined,
        "perplexity_combined": math.exp(combined),
    }


def evaluate_sft(config: dict, split: str, device: torch.device) -> dict:
    eval_cfg = config["sft_evaluation"]
    output_dir = Path(eval_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(eval_cfg["selection_manifest"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    candidates = eval_cfg["candidates"]

    if split == "val":
        results = {
            name: _evaluate_adapter(config, path, split, device)
            for name, path in candidates.items()
        }
        selected = min(
            results, key=lambda name: (results[name]["loss_combined"], name)
        )
        manifest = {
            "schema_version": "medin_hospital_sft_selection_v1",
            "selection_split": "val",
            "selection_metric": "loss_combined",
            "selected_plan": selected,
            "selected_adapter": candidates[selected],
            "results": results,
            "test_has_been_read": False,
        }
        with manifest_path.open("w") as f:
            json.dump(manifest, f, indent=2)
        with (output_dir / "val_metrics.json").open("w") as f:
            json.dump(results, f, indent=2)
        return manifest

    if not manifest_path.exists():
        raise FileNotFoundError(
            "Run full SFT validation selection before SFT test evaluation: "
            f"{manifest_path}"
        )
    with manifest_path.open() as f:
        manifest = json.load(f)
    result = _evaluate_adapter(
        config, manifest["selected_adapter"], "test", device
    )
    payload = {
        "selected_plan": manifest["selected_plan"],
        "selected_adapter": manifest["selected_adapter"],
        "metrics": result,
    }
    with (output_dir / "test_metrics.json").open("w") as f:
        json.dump(payload, f, indent=2)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=["val", "test"], required=True)
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()
    setup_logging()
    device = torch.device(
        f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    )
    result = evaluate_sft(load_config(args.config), args.split, device)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
