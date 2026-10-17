"""Precompute encoder embeddings for one of the three training paradigms.

Loads the selected paradigm's encoder LoRA (Clin-JEPA, V-JEPA 2-AC, or the
SFT baseline) and encodes every (stay_id, hour) state text, every unique
action text, and every demographics text into a 4096-D last-token hidden
state. Output shards use the two-table schema consumed by the rollout and
probe pipelines.

Schema per output shard:

    {
        "z_state_embeddings":  Tensor(N_state_unique,  4096) float16,
        "z_action_embeddings": Tensor(N_action_unique, 4096) float16,
        "z_statics":           Tensor(N_stays,         4096) float16,
        "windows": list[{
            "stay_id": int,
            "start_step": int,
            "length": int,
            "emb_indices_state":  Tensor(length,) int32,  -> z_state_embeddings
            "emb_indices_action": Tensor(length,) int32,  -> z_action_embeddings
            "static_idx": int,                             -> z_statics
            "labels": dict[str, Tensor(length,) float32], # 11 clinical labels
            "binary_labels": dict[str, bool],             # 4 outcome labels
        }],
        "schema_version": "clin_jepa_embeddings_v2",
    }

Usage::

    # All shards in one split (single GPU):
    python -m clin_jepa.evaluation.precompute_embeddings \\
        --config configs/eval/precompute_embeddings.yaml \\
        --plan clin_jepa --split train --gpu_id 0

    # Specific shard indices (for parallel multi-GPU):
    python -m clin_jepa.evaluation.precompute_embeddings \\
        --config configs/eval/precompute_embeddings.yaml \\
        --plan clin_jepa --split train --shard_indices 0,1,2,3 --gpu_id 0

    # Stride mode (4 GPUs each take every 4th shard):
    python -m clin_jepa.evaluation.precompute_embeddings \\
        --config configs/eval/precompute_embeddings.yaml \\
        --plan clin_jepa --split train --shard_stride 4 --shard_offset 0 \\
        --gpu_id 0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np

_HF_ENV_APPLIED = False


def _preconfigure_hf_env() -> None:
    """Set HF_HOME / offline env vars BEFORE any HF-adjacent import.

    Reads --config from sys.argv if present; otherwise a no-op (falls back to
    SLURM-exported HF_HOME). Safe to call multiple times.
    """
    global _HF_ENV_APPLIED
    if _HF_ENV_APPLIED:
        return
    _HF_ENV_APPLIED = True

    import sys as _sys
    cfg_path: Optional[str] = None
    args_iter = iter(_sys.argv[1:])
    for tok in args_iter:
        if tok == "--config":
            cfg_path = next(args_iter, None)
            break
        if tok.startswith("--config="):
            cfg_path = tok.split("=", 1)[1]
            break
    if cfg_path and os.path.exists(cfg_path):
        try:
            import yaml as _yaml
            with open(cfg_path) as _f:
                _cfg = _yaml.safe_load(_f)
            _hf = (_cfg or {}).get("hf", {}) or {}
            if _hf.get("cache_dir") and not os.environ.get("HF_HOME"):
                os.environ["HF_HOME"] = _hf["cache_dir"]
            if _hf.get("offline"):
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
                os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        except Exception:
            # If pre-parse fails, let SLURM env handle it.
            pass


_preconfigure_hf_env()

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from clin_jepa.training.encoder_ops import (
    PROJECT_ROOT,
    atomic_torch_save,
    encode_texts_batched,
    load_sft_initialized_encoder,
)
from clin_jepa.training.trajectory_dataset import _build_demo_text
from clin_jepa.utils import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_base_encoder_for_inference(config: dict, device: torch.device):
    """Load the frozen base Qwen without a PEFT adapter for probe baselines."""
    model_cfg = config["model"]
    base_name = os.path.expandvars(str(model_cfg["base_model"]))
    offline = bool(config.get("hf", {}).get("offline", False))
    dtype = getattr(torch, str(model_cfg.get("torch_dtype", "bfloat16")))
    model = AutoModelForCausalLM.from_pretrained(
        base_name,
        torch_dtype=dtype,
        attn_implementation=model_cfg.get(
            "attn_implementation", "flash_attention_2"
        ),
        local_files_only=offline,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        base_name, local_files_only=offline
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model, tokenizer


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    for file in files:
        name = file.relative_to(path) if path.is_dir() else Path(file.name)
        digest.update(str(name).encode())
        with file.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def should_skip_shard(shard_path: Path, output_dir: Path, provenance: dict) -> bool:
    """Resume only when an existing cache has identical provenance."""
    output_name = shard_path.stem.replace("trajectories_", "embeddings_") + ".pt"
    output = output_dir / output_name
    if not output.exists():
        return False
    try:
        cached = torch.load(output, map_location="cpu", weights_only=False)
    except Exception:
        return False
    return cached.get("provenance") == {
        **provenance,
        "trajectory_size": shard_path.stat().st_size,
        "trajectory_mtime_ns": shard_path.stat().st_mtime_ns,
    }


def process_shard(
    shard_path: Path,
    model,
    tokenizer,
    config: dict,
    device: torch.device,
    output_dir: Path,
    max_windows: Optional[int] = None,
) -> dict:
    """Process one trajectory shard into one embedding shard.

    Pipeline:
      1. Load shard (per_hour state/action texts + per_stay demographics + windows)
      2. Encode unique state texts (per_hour is already deduped)
      3. Dedup + encode unique action texts (string hash dedup)
      4. Encode demographics texts (one per stay)
      5. Assemble per-window output with index arrays
      6. Save deduped .pt
    """
    t0 = time.time()
    inf_cfg = config["inference"]
    max_seq_len = config["data"]["max_seq_len"]
    emb_dtype = getattr(torch, inf_cfg.get("embedding_dtype", "float16"))
    schema_version = config.get("output", {}).get(
        "schema_version", "clin_jepa_embeddings_v1"
    )

    # Step 1: Load shard
    shard = torch.load(shard_path, map_location="cpu", weights_only=False)
    ph = shard["per_hour"]
    phl = shard["per_hour_labels"]
    ps = shard["per_stay"]
    pse = shard.get("per_stay_extra", {})
    windows = shard["windows"]

    if max_windows is not None and len(windows) > max_windows:
        windows = windows[:max_windows]
        # Collect referenced hour and stay indices
        referenced_hours = sorted({int(h) for w in windows for h in w["hour_indices"].tolist()})
        referenced_stays = sorted({int(w["stay_index"]) for w in windows})
        old_to_new_hour = {old: new for new, old in enumerate(referenced_hours)}
        old_to_new_stay = {old: new for new, old in enumerate(referenced_stays)}
        # Re-project tables while preserving mixed-time and hospital metadata.
        import numpy as _np

        def _take(values, indices):
            if values is None:
                return None
            if isinstance(values, _np.ndarray):
                return values[indices]
            if isinstance(values, torch.Tensor):
                return values[indices]
            return [values[i] for i in indices]

        ph = {k: _take(v, referenced_hours) for k, v in ph.items()}
        phl = {k: v[referenced_hours] for k, v in phl.items()}
        ps = {k: _take(v, referenced_stays) for k, v in ps.items()}
        pse = {k: _take(v, referenced_stays) for k, v in pse.items()}
        # Remap window hour_indices + stay_index to the new compact tables
        windows = [
            {
                **{k: v for k, v in w.items() if k not in ("hour_indices", "stay_index")},
                "hour_indices": _np.asarray([old_to_new_hour[int(h)] for h in w["hour_indices"].tolist()], dtype=_np.int32),
                "stay_index":   old_to_new_stay[int(w["stay_index"])],
            }
            for w in windows
        ]
        logger.info("  SMOKE mode: truncated to %d windows, %d unique hours, %d unique stays",
                    len(windows), len(ph["state_texts"]), len(ps["stay_ids"]))

    n_unique_hours = len(ph["state_texts"])
    n_unique_stays = len(ps["stay_ids"])

    logger.info(
        "  Loaded %s: %d windows, %d unique hours, %d unique stays",
        shard_path.name, len(windows), n_unique_hours, n_unique_stays,
    )

    # Step 2: Encode observed state texts only. Natural-empty nodes receive a
    # zero placeholder and are filled by predictor rollout, never by Qwen.
    state_texts = ph["state_texts"]  # list[str]
    has_measured = np.asarray(
        ph.get("has_measured", np.ones(len(state_texts), dtype=np.bool_)),
        dtype=np.bool_,
    )
    observed_indices = np.flatnonzero(has_measured).tolist()
    hidden_size = int(getattr(model.config, "hidden_size", 4096))
    z_state = torch.zeros(len(state_texts), hidden_size, dtype=emb_dtype)
    if observed_indices:
        logger.info("  Encoding %d observed state texts...", len(observed_indices))
        encoded_states = encode_texts_batched(
            model, tokenizer, [state_texts[i] for i in observed_indices],
            device=device, max_seq_len=max_seq_len, train_mode=False,
            output_dtype=emb_dtype,
        )
        z_state[observed_indices] = encoded_states.cpu()

    # Step 3: Dedup action texts by string hash, then encode unique set
    unique_action_texts: list[str] = []
    action_str_to_idx: dict[str, int] = {}
    action_idx_per_hour: list[int] = []  # length N_unique_hours

    has_action = np.asarray(
        ph.get(
            "action_present",
            ph.get(
                "has_action",
                [str(v).strip() != "No actions recorded." for v in ph["action_texts"]],
            ),
        ),
        dtype=np.bool_,
    )
    for at in ph["action_texts"]:
        if at not in action_str_to_idx:
            action_str_to_idx[at] = len(unique_action_texts)
            unique_action_texts.append(at)
        action_idx_per_hour.append(action_str_to_idx[at])

    n_unique_action = len(unique_action_texts)
    dedup_ratio = len(ph["action_texts"]) / max(n_unique_action, 1)
    logger.info(
        "  Action dedup: %d -> %d unique (%.1fx ratio)",
        len(ph["action_texts"]), n_unique_action, dedup_ratio,
    )

    z_action = torch.zeros(n_unique_action, hidden_size, dtype=emb_dtype)
    if unique_action_texts:
        logger.info("  Encoding %d unique action texts...", n_unique_action)
        z_action[:] = encode_texts_batched(
            model, tokenizer, unique_action_texts, device=device,
            max_seq_len=max_seq_len, train_mode=False, output_dtype=emb_dtype,
        ).cpu()

    # Step 4: Encode demographics (one text per stay)
    demo_texts = [
        _build_demo_text(
            age=int(round(float(ps["ages"][i]))),
            gender=ps["genders"][i],
            race=ps["races"][i],
        )
        for i in range(n_unique_stays)
    ]
    logger.info("  Encoding %d demographics texts...", n_unique_stays)
    z_static = encode_texts_batched(
        model, tokenizer, demo_texts, device=device,
        max_seq_len=max_seq_len, train_mode=False, output_dtype=emb_dtype,
    )  # (N_unique_stays, 4096)

    # Step 5: Assemble per-window output
    windows_out = []
    for w in windows:
        hi_arr = w["hour_indices"]  # numpy int32
        hi = hi_arr.tolist()
        si = int(w["stay_index"])
        L = int(w["length"])

        # State embeddings: index via per-hour position
        emb_indices_state = torch.tensor(hi, dtype=torch.int32)

        # Action embeddings: route through action_idx_per_hour mapping
        emb_indices_action = torch.tensor(
            [action_idx_per_hour[h] for h in hi],
            dtype=torch.int32,
        )

        # Per-hour labels (gather from per_hour_labels arrays)
        labels = {
            k: torch.from_numpy(phl[k][hi].copy())  # (L,) float32
            for k in phl
        }

        # Per-stay binary outcomes
        binary_labels = {
            "icu_mortality":      bool(ps["icu_mortalities"][si]),
            "hospital_mortality": bool(ps["hospital_mortalities"][si]),
            "prolonged_stay":     bool(ps["prolonged_stays"][si]),
            "sepsis3":            bool(ps["sepsis3s"][si]),
        }
        extra_labels = {
            key: (
                value[si].item()
                if hasattr(value[si], "item")
                else value[si]
            )
            for key, value in pse.items()
        }

        start_hours_all = ph.get("start_hours", ph["hours"])
        end_hours_all = ph.get("end_hours", start_hours_all)
        if schema_version.startswith("medin_embeddings_v4") and "time_hours" not in ph:
            raise ValueError(
                "Migrated v0.3 evaluation requires episode-relative per_hour.time_hours"
            )
        model_time_hours_all = ph.get("time_hours", end_hours_all)
        duration_hours_all = ph.get(
            "duration_hours",
            np.ones(len(ph["hours"]), dtype=np.float32),
        )
        is_icu_all = ph.get(
            "is_icu", np.zeros(len(ph["hours"]), dtype=np.bool_)
        )
        has_measured_all = ph.get(
            "has_measured", np.ones(len(ph["hours"]), dtype=np.bool_)
        )
        since_measured_all = ph.get(
            "hours_since_last_measurement",
            np.full(len(ph["hours"]), np.nan, dtype=np.float32),
        )

        windows_out.append({
            "stay_id":            int(ps["stay_ids"][si]),
            "hadm_id":            int(
                ps["hadm_ids"][si] if "hadm_ids" in ps else ps["stay_ids"][si]
            ),
            "subject_id":         (
                int(ps["subject_ids"][si])
                if ps.get("subject_ids") is not None
                else int(ps["stay_ids"][si])
            ),
            "start_step":         int(w["start_step"]),
            "length":             L,
            "emb_indices_state":  emb_indices_state,
            "emb_indices_action": emb_indices_action,
            "static_idx":         si,
            "time_hours": torch.as_tensor(
                model_time_hours_all[hi], dtype=torch.float32
            ),
            "start_hours": torch.as_tensor(
                start_hours_all[hi], dtype=torch.float32
            ),
            "end_hours": torch.as_tensor(
                end_hours_all[hi], dtype=torch.float32
            ),
            "duration_hours": torch.as_tensor(
                duration_hours_all[hi], dtype=torch.float32
            ),
            "is_icu": torch.as_tensor(is_icu_all[hi], dtype=torch.bool),
            "has_measured": torch.as_tensor(
                has_measured_all[hi], dtype=torch.bool
            ),
            "has_action": torch.as_tensor(has_action[hi], dtype=torch.bool),
            "action_present": torch.as_tensor(has_action[hi], dtype=torch.bool),
            "hours_since_last_measurement": torch.as_tensor(
                since_measured_all[hi], dtype=torch.float32
            ),
            "labels":             labels,
            "binary_labels":      binary_labels,
            "extra_labels":       extra_labels,
        })

    out_name = shard_path.stem.replace("trajectories_", "embeddings_") + ".pt"
    output_path = output_dir / out_name
    atomic_torch_save({
        "z_state_embeddings":  z_state,
        "z_action_embeddings": z_action,
        "z_statics":           z_static,
        "windows":             windows_out,
        "schema_version":      schema_version,
        "provenance": {
            **config["_provenance"],
            "trajectory_size": shard_path.stat().st_size,
            "trajectory_mtime_ns": shard_path.stat().st_mtime_ns,
        },
    }, output_path)

    elapsed = time.time() - t0
    file_mb = output_path.stat().st_size / 1e6
    stats = {
        "shard": shard_path.name,
        "n_windows": len(windows_out),
        "n_unique_state": n_unique_hours,
        "n_unique_action": n_unique_action,
        "n_unique_stays": n_unique_stays,
        "time_seconds": elapsed,
        "output_mb": file_mb,
    }
    logger.info(
        "  Saved %s (%.0f MB, %d windows, %.0fs)",
        out_name, file_mb, len(windows_out), elapsed,
    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute encoder embeddings for downstream evaluation.",
    )
    parser.add_argument(
        "--config",
        default="configs/eval/precompute_embeddings.yaml",
        help="Path to precompute_embeddings.yaml.",
    )
    parser.add_argument(
        "--plan",
        required=True,
        help="Plan key in cfg['plans']; hospital configs may define best/latest/final.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="test",
        help="Data split to encode.",
    )
    parser.add_argument(
        "--shard_indices",
        type=str,
        default=None,
        help="Comma-separated shard indices to process. None means all.",
    )
    parser.add_argument(
        "--shard_stride",
        type=int,
        default=None,
        help="Process every Nth shard (parallel mode across GPUs).",
    )
    parser.add_argument(
        "--shard_offset",
        type=int,
        default=0,
        help="Offset into the shard-stride sweep.",
    )
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument(
        "--max_windows",
        type=int,
        default=None,
        help="Smoke-test option: truncate each shard to the first N windows. "
             "Do not use for full runs.",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    # Merge plan-specific overrides into top-level config.
    plan_cfg = config.get("plans", {}).get(args.plan)
    if plan_cfg is None:
        raise KeyError(
            f"--plan {args.plan} not found in config['plans']. "
            f"Available: {list(config.get('plans', {}).keys())}"
        )
    config.setdefault("model", {})["checkpoint_dir"] = plan_cfg["checkpoint_dir"]
    config.setdefault("data", {})["output_dir"] = plan_cfg["output_dir"]


    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    logger.info("Precompute embeddings: plan=%s split=%s device=%s", args.plan, args.split, device)

    # Load either the raw frozen base (probe control) or a PEFT encoder.
    if plan_cfg.get("base_only", False):
        model, tokenizer = load_base_encoder_for_inference(config, device)
    else:
        model, tokenizer = load_sft_initialized_encoder(config, device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # Discover input shards
    trajectory_dir = PROJECT_ROOT / config["data"]["trajectory_dir"] / args.split
    all_shards = sorted(trajectory_dir.glob("trajectories_*.pt"))
    logger.info("Found %d shards in %s", len(all_shards), trajectory_dir)
    if not all_shards:
        raise FileNotFoundError(
            f"No trajectory shards found in {trajectory_dir}; check that "
            "MEDIN_DATA_ROOT is exported and points at the mixed dataset"
        )

    if args.shard_indices:
        indices = [int(x) for x in args.shard_indices.split(",")]
        shards = [all_shards[i] for i in indices if i < len(all_shards)]
    elif args.shard_stride:
        shards = all_shards[args.shard_offset::args.shard_stride]
    else:
        shards = all_shards

    output_dir = PROJECT_ROOT / config["data"]["output_dir"] / args.split
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(os.path.expandvars(str(config["model"]["checkpoint_dir"])))
    config["_provenance"] = {
        "schema_version": config.get("output", {}).get("schema_version"),
        "encoder_checkpoint": str(checkpoint_dir.resolve()),
        "encoder_sha256": _sha256_path(checkpoint_dir),
        "time_semantics": "hours_from_care_episode_start_at_bucket_end",
        "empty_state_semantics": "zero_placeholder_predictor_rollout",
        "no_action_semantics": "canonical_text_encoded_by_shared_encoder",
    }

    all_stats: list[dict] = []
    for shard_path in shards:
        if should_skip_shard(shard_path, output_dir, config["_provenance"]):
            logger.info("SKIP (exists): %s", shard_path.name)
            continue
        try:
            stats = process_shard(
                shard_path, model, tokenizer, config, device, output_dir,
                max_windows=args.max_windows,
            )
            all_stats.append(stats)
        except Exception as e:
            logger.error("FAILED on %s: %s", shard_path.name, e)
            raise

    if all_stats:
        total_windows = sum(s["n_windows"] for s in all_stats)
        total_state = sum(s["n_unique_state"] for s in all_stats)
        total_action = sum(s["n_unique_action"] for s in all_stats)
        total_time = sum(s["time_seconds"] for s in all_stats)
        total_mb = sum(s["output_mb"] for s in all_stats)
        logger.info(
            "COMPLETE: %d shards, %d windows, %d state texts, %d action texts, %.0f MB, %.1f min",
            len(all_stats), total_windows, total_state, total_action, total_mb, total_time / 60,
        )

        # Save summary stats
        summary = {
            "split": args.split,
            "n_shards_processed": len(all_stats),
            "total_windows": total_windows,
            "total_unique_state": total_state,
            "total_unique_action": total_action,
            "total_seconds": total_time,
            "total_output_mb": total_mb,
            "per_shard": all_stats,
        }
        summary_path = output_dir / f"_summary_gpu{args.gpu_id}.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
    else:
        logger.info("No shards processed (all already exist).")

    # Only the unpartitioned invocation can certify the whole split. Parallel
    # stride/index jobs intentionally leave certification to a final full pass.
    if not args.shard_indices and not args.shard_stride:
        expected_outputs = [
            output_dir / (p.stem.replace("trajectories_", "embeddings_") + ".pt")
            for p in all_shards
        ]
        missing = [str(p) for p in expected_outputs if not p.exists()]
        if missing:
            raise RuntimeError(
                f"Embedding cache incomplete: {len(missing)} files missing; "
                f"examples={missing[:5]}"
            )
        with (output_dir / "_complete.json").open("w") as f:
            json.dump({
                "split": args.split,
                "n_input_shards": len(all_shards),
                "n_output_shards": len(expected_outputs),
                "provenance": config["_provenance"],
            }, f, indent=2)


if __name__ == "__main__":
    main()
