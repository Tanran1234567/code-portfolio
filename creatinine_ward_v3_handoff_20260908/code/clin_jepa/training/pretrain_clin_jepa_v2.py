"""MedIn v0.3 unified JEPA training: dense mask curriculum then natural-empty gaps.

Reuses the v0.1 encoder/checkpoint/resume machinery. Phase schedule:

    P1 dense warmup → P2 dense+mask co-train → P3 EMA align →
    P4 hard sync → P5A dense native mask → P5B natural-empty gap
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import signal
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.utils.data import DataLoader

from clin_jepa.model.predictor import ACTransformerPredictor
from clin_jepa.training.encoder_ops import (
    PROJECT_ROOT,
    _build_online_target_param_pairs,
    atomic_torch_save,
    collect_lora_adapter_state,
    create_target_encoder_fp32,
    ema_update,
    hard_sync_target_from_online,
    load_sft_initialized_encoder,
    manual_allreduce_grads,
    restore_lora_adapter_state,
    restore_rng_state,
    snapshot_rng_state,
    verify_ema_working,
)
from clin_jepa.training.pretrain_clin_jepa import (
    cleanup_distributed,
    compute_rollout_loss_native,
    encode_batch_via_online,
    save_checkpoint,
    setup_distributed,
    set_encoder_trainable,
)
from clin_jepa.training.pretrain_v2_losses import (
    next_step_masks,
    sample_artificial_mask,
    sequence_normalized_l1,
)
from clin_jepa.training.source_balanced_sampler import SourceBalancedSampler
from clin_jepa.training.trajectory_dataset import (
    TrajectoryWindowDataset,
    trajectory_collate_fn,
)
from clin_jepa.utils import setup_logging

logger = logging.getLogger(__name__)
_PREEMPTED = False


def _require_finite(name: str, tensor: torch.Tensor, phase: str, step: int) -> None:
    finite = torch.isfinite(tensor)
    if bool(finite.all()):
        return
    bad = int((~finite).sum().item())
    good = tensor.detach()[finite]
    bounds = (
        (float(good.min()), float(good.max()))
        if good.numel() else (None, None)
    )
    raise FloatingPointError(
        f"non-finite {name}: phase={phase} step={step} bad={bad}/"
        f"{tensor.numel()} finite_range={bounds}"
    )


def _peak_cpu_rss_gib() -> float:
    try:
        import resource
        return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / (1024 ** 2)
    except (ImportError, AttributeError):
        return 0.0


def _preempt_signal_handler(signum, frame) -> None:  # noqa: ARG001
    global _PREEMPTED
    _PREEMPTED = True


def _expand(path: str | Path) -> Path:
    raw = Path(os.path.expandvars(os.path.expanduser(str(path))))
    return raw if raw.is_absolute() else PROJECT_ROOT / raw


def _sources(dataset: TrajectoryWindowDataset) -> list[str]:
    out = []
    for ref in dataset.window_refs:
        src = ref.get("source_type")
        if not src:
            shard = dataset._load_shard(ref["_shard_idx"])
            stay_i = int(ref["stay_index"])
            if "source_types" in shard.get("per_stay", {}):
                src = shard["per_stay"]["source_types"][stay_i]
            else:
                src = "WARD"
        out.append(src)
    return out


def v2_teacher_forced_loss(
    predictor,
    z_online: dict,
    z_target: dict | None,
    mask_rate: float,
    lambda_mask: float,
    min_context: int,
    native: bool,
) -> torch.Tensor:
    observed = z_online["state_observed"]
    lengths = z_online["lengths"]
    pad = z_online["padding_mask"]
    art = sample_artificial_mask(observed, lengths, mask_rate, min_context)
    z_states = z_online["z_states"]
    base_preds = predictor(
        z_states,
        z_online["z_static"],
        z_online["z_actions"],
        lengths,
        time_hours=z_online["time_hours"],
        duration_hours=z_online["duration_hours"],
    )
    preds = base_preds
    if bool(art.any()):
        # Two-pass masked continuation: pass 1 predicts each hidden state from
        # its preceding context; pass 2 replaces hidden inputs so all later
        # predictions cannot attend to the real masked latent. This bounds the
        # Predictor cost at two full forwards rather than O(T) forwards.
        masked_states = z_states.clone()
        B, T, _ = z_states.shape
        for b in range(B):
            masked_positions = torch.where(art[b, 1:T])[0] + 1
            if masked_positions.numel():
                masked_states[b, masked_positions] = base_preds[
                    b, masked_positions - 1
                ]
        preds = predictor(
            masked_states,
            z_online["z_static"],
            z_online["z_actions"],
            lengths,
            time_hours=z_online["time_hours"],
            duration_hours=z_online["duration_hours"],
        )
    targets = (z_target or z_online)["z_states"][:, 1:, :]
    visible, masked, _gap = next_step_masks(observed, art, pad, min_context)
    loss = sequence_normalized_l1(preds, targets, visible)
    if mask_rate > 0 and bool(masked.any()):
        mask_targets = (z_target or z_online)["z_states"][:, 1:, :]
        loss = loss + lambda_mask * sequence_normalized_l1(preds, mask_targets, masked)
    if native:
        loss = loss + compute_rollout_loss_native(
            predictor,
            z_states,
            z_online["z_static"],
            z_online["z_actions"],
            lengths,
            z_online["time_hours"],
            z_online["duration_hours"],
            rollout_horizon=2,
        )
    return loss


def v2_gap_loss(
    predictor,
    z_online: dict,
    z_target: dict | None,
    min_context: int,
    max_gap_steps: int = 8,
    max_gaps_per_sequence: int = 2,
) -> torch.Tensor:
    """Native rollout through natural empty; supervise the next observed endpoint."""
    z_states = z_online["z_states"]
    observed = z_online["state_observed"]
    lengths = z_online["lengths"]
    pad = z_online["padding_mask"]
    B, T, _ = z_states.shape
    preds = predictor(
        z_states,
        z_online["z_static"],
        z_online["z_actions"],
        lengths,
        time_hours=z_online["time_hours"],
        duration_hours=z_online["duration_hours"],
    )
    targets = (z_target or z_online)["z_states"][:, 1:, :]
    visible, _masked, _gap = next_step_masks(
        observed, torch.zeros_like(observed), pad, min_context
    )
    consecutive = visible & observed[:, :-1]
    normal_loss = sequence_normalized_l1(preds, targets, consecutive)

    gap_endpoint_losses: list[torch.Tensor] = []
    target_states = (z_target or z_online)["z_states"]
    for b in range(B):
        length = int(lengths[b].item())
        i = 1
        gaps_used = 0
        while i < length:
            if bool(observed[b, i]):
                i += 1
                continue
            gap_start = i
            while i < length and not bool(observed[b, i]):
                i += 1
            endpoint = i
            if endpoint >= length:
                break  # trailing natural gap has no supervised endpoint
            if endpoint - gap_start > max_gap_steps:
                continue
            if gaps_used >= max_gaps_per_sequence:
                break
            context_end = gap_start - 1
            context_start = context_end
            while (
                context_start > 0
                and context_end - context_start + 1 < min_context
                and bool(observed[b, context_start - 1])
            ):
                context_start -= 1
            if context_end - context_start + 1 < min_context:
                continue
            history_states = z_states[b : b + 1, context_start : context_end + 1]
            history_actions = z_online["z_actions"][
                b : b + 1, context_start : context_end + 1
            ]
            history_times = z_online["time_hours"][
                b : b + 1, context_start : context_end + 1
            ]
            history_durations = z_online["duration_hours"][
                b : b + 1, context_start : context_end + 1
            ]
            predicted = None
            for target_idx in range(gap_start, endpoint + 1):
                seq_len = history_states.shape[1]
                predicted = predictor(
                    history_states,
                    z_online["z_static"][b : b + 1],
                    history_actions,
                    torch.tensor([seq_len], device=z_states.device, dtype=torch.long),
                    time_hours=history_times,
                    duration_hours=history_durations,
                    return_last_only=True,
                )
                if target_idx == endpoint:
                    break
                history_states = torch.cat([history_states, predicted.unsqueeze(1)], dim=1)
                history_actions = torch.cat([
                    history_actions,
                    z_online["z_actions"][b : b + 1, target_idx : target_idx + 1],
                ], dim=1)
                history_times = torch.cat([
                    history_times,
                    z_online["time_hours"][b : b + 1, target_idx : target_idx + 1],
                ], dim=1)
                history_durations = torch.cat([
                    history_durations,
                    z_online["duration_hours"][b : b + 1, target_idx : target_idx + 1],
                ], dim=1)
            if predicted is not None:
                gap_endpoint_losses.append(
                    (predicted[0] - target_states[b, endpoint]).abs().sum()
                )
                gaps_used += 1
    gap_loss = (
        torch.stack(gap_endpoint_losses).mean()
        if gap_endpoint_losses else z_states.new_zeros(())
    )
    return normal_loss + gap_loss


def _encode(
    raw_online,
    tokenizer,
    windows,
    device,
    max_seq_len,
    train_encoder: bool,
    encode_target: bool,
):
    target = None
    if encode_target:
        # Encode the no-grad EMA target first so its temporary activations are
        # released before the online gradient graph is constructed.
        raw_online.set_adapter("target")
        with torch.no_grad():
            target = encode_batch_via_online(
                raw_online, tokenizer, windows, device, max_seq_len,
                train_mode=False, output_dtype=torch.float32,
            )
    raw_online.set_adapter("online")
    online = encode_batch_via_online(
        raw_online, tokenizer, windows, device, max_seq_len,
        train_mode=train_encoder, output_dtype=torch.bfloat16,
    )
    return online, target


def train(config: dict, args) -> None:
    rank, world_size, is_distributed = setup_distributed()
    is_main = rank == 0
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(rank)
        torch.cuda.reset_peak_memory_stats(rank)
    if is_main:
        setup_logging()
        logger.info("MedIn v0.3 unified JEPA (%d GPUs)", world_size)
    if args.smoke:
        torch.autograd.set_detect_anomaly(True, check_nan=True)

    seed = int(config["training"]["seed"])
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)

    online, tokenizer = load_sft_initialized_encoder(config, device)
    online = create_target_encoder_fp32(online)
    if is_main:
        verify_ema_working(online, n_updates=10)
    if is_distributed:
        dist.barrier()
    ema_pairs = _build_online_target_param_pairs(online)

    pred_cfg = config["predictor"]
    predictor = ACTransformerPredictor(
        state_dim=pred_cfg.get("state_dim", 4096),
        action_dim=pred_cfg.get("action_dim", 4096),
        static_dim=pred_cfg.get("static_dim", 4096),
        hidden_dim=pred_cfg.get("hidden_dim", 1024),
        num_layers=pred_cfg.get("num_layers", 6),
        num_heads=pred_cfg.get("num_heads", 8),
        ffn_dim=pred_cfg.get("ffn_dim", 4096),
        max_timesteps=pred_cfg.get("max_timesteps", 72),
        ffn_dropout=pred_cfg.get("ffn_dropout", 0.15),
        attn_dropout=pred_cfg.get("attn_dropout", 0.1),
        prediction_mode=pred_cfg.get("prediction_mode", "absolute"),
        use_continuous_time=bool(pred_cfg.get("use_continuous_time", False)),
        absolute_time_additive=bool(pred_cfg.get("absolute_time_additive", True)),
    ).to(device)
    predictor_init = pred_cfg.get("checkpoint")
    if isinstance(predictor_init, str):
        predictor_init = os.path.expandvars(os.path.expanduser(predictor_init)).strip()
        if predictor_init.lower() in {"", "null", "none", "false"}:
            predictor_init = None
    if predictor_init:
        predictor_path = _expand(predictor_init)
        if predictor_path.is_dir():
            predictor_path = predictor_path / "predictor.pt"
        state = torch.load(predictor_path, map_location="cpu", weights_only=True)
        incompatible = predictor.load_state_dict(state, strict=False)
        unexpected = list(incompatible.unexpected_keys)
        missing = [
            k for k in incompatible.missing_keys
            if not k.startswith("time_proj.") and not k.startswith("abs_time_proj.")
        ]
        if unexpected or missing:
            raise RuntimeError(
                f"Incompatible predictor checkpoint: missing={missing}, unexpected={unexpected}"
            )
        if is_main:
            logger.info("Loaded predictor weights from %s", predictor_path)
    elif is_main:
        logger.info("Predictor randomly initialized (paper from-scratch, no ICU warm-start)")

    if is_distributed:
        with torch.no_grad():
            for p in predictor.parameters():
                dist.broadcast(p.data, src=0)

    data_cfg = config["data"]
    save_dir = _expand(args.save_dir or config["save"]["dir"])
    index_dir = save_dir / "trajectory_indices"
    index_suffix = f"smoke{args.max_windows}" if args.max_windows else "full"
    gap_dir = data_cfg.get("gap_train_dir", data_cfg["train_dir"])

    def _load_datasets(index_read_only: bool = False):
        return (
            TrajectoryWindowDataset(
                shard_dir=_expand(data_cfg["train_dir"]),
                split="train",
                max_windows=args.max_windows,
                min_length=int(data_cfg.get("min_length", 3)),
                index_cache=index_dir / f"dense_train_{index_suffix}.pt",
                index_read_only=index_read_only,
            ),
            TrajectoryWindowDataset(
                shard_dir=_expand(data_cfg["val_dir"]),
                split="val",
                max_windows=args.max_windows or config["logging"].get("val_windows"),
                min_length=int(data_cfg.get("min_length", 3)),
                index_cache=index_dir / f"dense_val_{index_suffix}.pt",
                index_read_only=index_read_only,
            ),
            TrajectoryWindowDataset(
                shard_dir=_expand(gap_dir),
                split="train",
                max_windows=args.max_windows,
                min_length=int(data_cfg.get("min_length", 3)),
                index_cache=index_dir / f"gap_train_{index_suffix}.pt",
                index_read_only=index_read_only,
            ),
        )

    # Only rank 0 may create or refresh an index. Other ranks wait until the
    # cache is complete and then load it strictly read-only. This avoids a
    # shared temporary-file race and prevents indexing the corpus per GPU.
    if is_distributed:
        if is_main:
            train_dense, val_ds, train_gap = _load_datasets()
        dist.barrier()
        if not is_main:
            train_dense, val_ds, train_gap = _load_datasets(index_read_only=True)
    else:
        train_dense, val_ds, train_gap = _load_datasets()

    train_cfg = config["training"]
    batch_size = int(args.per_gpu_batch_size or train_cfg["per_gpu_batch_size"])
    icu_frac = float(train_cfg.get("icu_frac", 0.5))

    def _loader(dataset, shuffle_sources: bool) -> DataLoader:
        sampler = (
            SourceBalancedSampler(
                _sources(dataset), icu_frac=icu_frac, seed=seed,
                num_replicas=world_size, rank=rank,
                groups=[int(ref["_shard_idx"]) for ref in dataset.window_refs],
            )
            if shuffle_sources else None
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=sampler is None,
            collate_fn=trajectory_collate_fn,
            num_workers=0,
        )

    dense_loader = _loader(train_dense, True)
    gap_loader = _loader(train_gap, True)
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=trajectory_collate_fn, num_workers=0,
    )

    phases_cfg = train_cfg["phases"]
    if args.smoke:
        n = args.smoke_phase_steps or 4
        phase_steps = {"p1": n, "p2": n, "p3": n, "p5a": n, "p5b": n}
    else:
        phase_steps = {
            "p1": int(phases_cfg["phase_1_steps"]),
            "p2": int(phases_cfg["phase_2_steps"]),
            "p3": int(phases_cfg["phase_3_steps"]),
            "p5a": int(phases_cfg["phase_5a_steps"]),
            "p5b": int(phases_cfg["phase_5b_steps"]),
        }
    total_steps = sum(phase_steps.values())
    boundaries = []
    acc = 0
    for name in ("p1", "p2", "p3", "p5a", "p5b"):
        acc += phase_steps[name]
        boundaries.append((acc, name))
    if is_main:
        logger.info("phase steps %s total=%d", phase_steps, total_steps)

    raw_online = online
    lora_params = [p for n, p in raw_online.named_parameters() if p.requires_grad]
    pred_params = list(predictor.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": lora_params, "lr": float(train_cfg["lr_encoder"])},
            {"params": pred_params, "lr": float(train_cfg["lr_predictor"])},
        ],
        weight_decay=float(train_cfg["weight_decay"]),
    )
    warmup = max(1, int(total_steps * float(train_cfg.get("warmup_ratio", 0.02))))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    if is_main:
        save_dir.mkdir(parents=True, exist_ok=True)

    mask_rates = train_cfg.get("mask_rates", {})
    lambda_mask = float(train_cfg.get("lambda_mask", 1.0))
    min_context = int(train_cfg.get("min_context", 2))
    max_gap_steps = int(train_cfg.get("max_gap_steps", 8))
    max_gaps_per_sequence = int(train_cfg.get("max_gaps_per_sequence", 2))
    max_seq_len = int(data_cfg.get("max_seq_len", 4096))
    grad_accum = int(args.grad_accum or train_cfg["grad_accum"])

    def current_phase(step: int) -> str:
        running = 0
        for name in ("p1", "p2", "p3", "p5a", "p5b"):
            running += phase_steps[name]
            if step < running:
                return name
        return "p5b"

    def phase_settings(name: str) -> dict:
        if name == "p1":
            return {"train_enc": False, "mask": 0.0, "native": False, "gap": False, "ema": False}
        if name == "p2":
            return {"train_enc": True, "mask": float(mask_rates.get("phase_2", 0.15)), "native": False, "gap": False, "ema": True}
        if name == "p3":
            return {"train_enc": False, "mask": float(mask_rates.get("phase_3", 0.10)), "native": False, "gap": False, "ema": True}
        if name == "p5a":
            return {"train_enc": False, "mask": float(mask_rates.get("phase_5a", 0.20)), "native": True, "gap": False, "ema": False}
        return {"train_enc": False, "mask": 0.0, "native": True, "gap": True, "ema": False}

    global_step = 0
    micro = 0
    dense_batches_seen = 0
    gap_batches_seen = 0
    best_val = float("inf")
    t0 = time.time()
    synced = False
    phase_name = "p1"
    last_phase_name: str | None = None

    def _save_resume() -> None:
        if not is_main:
            return
        atomic_torch_save(
            {
                "schema_version": "medin_v2_pretrain_v5_exact_loader_resume",
                "lora_adapter_state": collect_lora_adapter_state(raw_online),
                "predictor_state": predictor.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "global_step": global_step,
                "micro": micro,
                "dense_batches_seen": dense_batches_seen,
                "gap_batches_seen": gap_batches_seen,
                "best_val": best_val,
                "synced": synced,
                "rng_state": snapshot_rng_state(),
            },
            save_dir / "resume.pt",
        )

    resume_path = None
    if getattr(args, "resume_from", None) is not None:
        if str(args.resume_from).lower() != "none":
            resume_path = Path(os.path.expandvars(str(args.resume_from)))
    else:
        candidate = save_dir / "resume.pt"
        if candidate.exists():
            resume_path = candidate
    if resume_path is not None and resume_path.exists():
        state = torch.load(resume_path, map_location="cpu", weights_only=False)
        if state.get("schema_version") != "medin_v2_pretrain_v5_exact_loader_resume":
            raise RuntimeError(f"unexpected resume schema {state.get('schema_version')}")
        restore_lora_adapter_state(raw_online, state["lora_adapter_state"])
        predictor.load_state_dict(state["predictor_state"])
        predictor.to(device)
        optimizer.load_state_dict(state["optimizer_state"])
        for st in optimizer.state.values():
            for k, v in st.items():
                if isinstance(v, torch.Tensor):
                    st[k] = v.to(device)
        scheduler.load_state_dict(state["scheduler_state"])
        restore_rng_state(state["rng_state"])
        global_step = int(state["global_step"])
        micro = int(state.get("micro", 0))
        dense_batches_seen = int(state.get("dense_batches_seen", 0))
        gap_batches_seen = int(state.get("gap_batches_seen", 0))
        best_val = float(state.get("best_val", float("inf")))
        synced = bool(state.get("synced", False))
        if is_main:
            logger.info("=== RESUMING v2 from %s at step %d ===", resume_path, global_step)

    def _restore_loader_position(loader: DataLoader, consumed: int):
        cycles, offset = divmod(consumed, max(1, len(loader)))
        sampler = loader.sampler
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(cycles)
        iterator = iter(loader)
        for _ in range(offset):
            next(iterator)
        return iterator

    dense_iter = _restore_loader_position(dense_loader, dense_batches_seen)
    gap_iter = _restore_loader_position(gap_loader, gap_batches_seen)

    signal.signal(signal.SIGUSR1, _preempt_signal_handler)
    signal.signal(signal.SIGTERM, _preempt_signal_handler)

    while global_step < total_steps:
        phase_name = current_phase(global_step)
        if phase_name != last_phase_name:
            if is_main and last_phase_name is not None and device.type == "cuda":
                logger.info(
                    "EXIT phase=%s peak_vram=%.2fGiB peak_cpu_rss=%.2fGiB",
                    last_phase_name,
                    torch.cuda.max_memory_allocated(device) / (1024 ** 3),
                    _peak_cpu_rss_gib(),
                )
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(rank)
            if is_main:
                logger.info("ENTER phase=%s step=%d", phase_name, global_step)
            last_phase_name = phase_name
        settings = phase_settings(phase_name)
        set_encoder_trainable(raw_online, settings["train_enc"])
        # Safety invariant: the frozen Qwen base and EMA target must never be
        # accidentally made trainable by a phase transition.
        for name, param in raw_online.named_parameters():
            should_train = settings["train_enc"] and "lora_" in name and ".online." in name
            if param.requires_grad != should_train:
                raise RuntimeError(
                    f"invalid trainability for {name}: got {param.requires_grad}, "
                    f"expected {should_train}"
                )
        if phase_name == "p5a" and not synced:
            hard_sync_target_from_online(raw_online)
            synced = True
            if is_main:
                logger.info("Phase 4 hard sync complete at step %d", global_step)

        loader_iter = gap_iter if settings["gap"] else dense_iter
        try:
            batch = next(loader_iter)
        except StopIteration:
            if settings["gap"]:
                gap_iter = iter(gap_loader)
                batch = next(gap_iter)
            else:
                dense_iter = iter(dense_loader)
                batch = next(dense_iter)
        if settings["gap"]:
            gap_batches_seen += 1
        else:
            dense_batches_seen += 1

        online_enc, target_enc = _encode(
            raw_online, tokenizer, batch["batch"], device, max_seq_len,
            train_encoder=settings["train_enc"],
            encode_target=settings["train_enc"] or phase_name == "p3",
        )
        for key in ("z_states", "z_actions", "z_static", "time_hours", "duration_hours"):
            _require_finite(f"online.{key}", online_enc[key], phase_name, global_step)
        if target_enc is not None:
            for key in ("z_states", "z_actions", "z_static"):
                _require_finite(f"target.{key}", target_enc[key], phase_name, global_step)
        if settings["gap"]:
            loss = v2_gap_loss(
                predictor, online_enc, target_enc, min_context,
                max_gap_steps=max_gap_steps,
                max_gaps_per_sequence=max_gaps_per_sequence,
            )
        else:
            loss = v2_teacher_forced_loss(
                predictor, online_enc, target_enc,
                mask_rate=settings["mask"],
                lambda_mask=lambda_mask,
                min_context=min_context,
                native=settings["native"],
            )
        _require_finite("loss", loss, phase_name, global_step)
        (loss / grad_accum).backward()
        micro += 1
        if micro % grad_accum != 0:
            continue

        if is_distributed:
            manual_allreduce_grads(list(predictor.parameters()) + lora_params)
        predictor_grad_norm = torch.nn.utils.clip_grad_norm_(
            pred_params,
            float(train_cfg["gradient_clip_predictor"]),
            error_if_nonfinite=True,
        )
        if settings["train_enc"] and lora_params:
            encoder_grad_norm = torch.nn.utils.clip_grad_norm_(
                lora_params,
                float(train_cfg["gradient_clip_encoder"]),
                error_if_nonfinite=True,
            )
            if not any(param.grad is not None for param in lora_params):
                raise RuntimeError(
                    f"P2 online LoRA received no gradients at step={global_step}; "
                    "checkpoint path detached the Encoder graph"
                )
            if float(encoder_grad_norm) <= 0.0:
                raise RuntimeError(
                    f"P2 online LoRA gradient norm is zero at step={global_step}; "
                    "refusing fake co-training success"
                )
        else:
            encoder_grad_norm = torch.zeros((), device=device)
        optimizer.step()
        if args.smoke:
            for idx, param in enumerate(pred_params):
                _require_finite(f"predictor.param[{idx}]", param, phase_name, global_step)
            for idx, param in enumerate(lora_params):
                _require_finite(f"online_lora.param[{idx}]", param, phase_name, global_step)
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        if settings["ema"]:
            tau_key = "tau_phase_2b" if phase_name == "p3" else "tau_phase_2a"
            ema_update(ema_pairs, float(train_cfg["ema"][tau_key]))
        global_step += 1

        log_every = int(args.log_every or config["logging"].get("log_every", 10))
        if is_main and global_step % log_every == 0:
            logger.info(
                "step %d phase=%s loss=%.4f grad_p=%.4f grad_e=%.4f "
                "lr_p=%.2e peak_vram=%.2fGiB peak_cpu_rss=%.2fGiB",
                global_step, phase_name, float(loss.detach()),
                float(predictor_grad_norm), float(encoder_grad_norm),
                optimizer.param_groups[-1]["lr"],
                (
                    torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                    if device.type == "cuda" else 0.0
                ),
                _peak_cpu_rss_gib(),
            )

        eval_every = int(args.eval_every or config["logging"].get("eval_every", 200))
        if is_main and global_step % eval_every == 0:
            _save_resume()
            predictor.eval()
            losses = []
            evaluated_windows = 0
            eval_max_windows = int(config["logging"].get("eval_max_windows", 200))
            with torch.no_grad():
                for vb in val_loader:
                    if evaluated_windows >= eval_max_windows:
                        break
                    enc, _tgt = _encode(
                        raw_online, tokenizer, vb["batch"], device, max_seq_len,
                        False, False,
                    )
                    losses.append(float(v2_teacher_forced_loss(
                        predictor, enc, None, 0.0, 0.0, min_context, False,
                    ).detach()))
                    evaluated_windows += len(vb["batch"])
            predictor.train()
            val_loss = float(np.mean(losses)) if losses else float(loss.detach())
            if not np.isfinite(val_loss):
                raise FloatingPointError(
                    f"non-finite validation loss at step={global_step}: {val_loss}"
                )
            logger.info("val_loss=%.4f", val_loss)
            tag = "latest"
            save_checkpoint(
                raw_online, predictor, tokenizer, save_dir, tag,
                global_step, phase_name, val_loss, 0.0, time.time() - t0,
            )
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(
                    raw_online, predictor, tokenizer, save_dir, "best",
                    global_step, phase_name, val_loss, 0.0, time.time() - t0,
                )

        if _PREEMPTED:
            if is_main:
                logger.info("=== PREEMPTION at step %d — saving resume.pt ===", global_step)
                _save_resume()
            cleanup_distributed(is_distributed)
            return

    if is_main and device.type == "cuda":
        logger.info(
            "EXIT phase=%s peak_vram=%.2fGiB peak_cpu_rss=%.2fGiB",
            phase_name, torch.cuda.max_memory_allocated(device) / (1024 ** 3),
            _peak_cpu_rss_gib(),
        )
    if is_main:
        save_checkpoint(
            raw_online, predictor, tokenizer, save_dir, "final",
            global_step, phase_name, best_val, 0.0, time.time() - t0,
        )
        logger.info("v0.3 training complete best_val=%.4f steps=%d", best_val, global_step)
    cleanup_distributed(is_distributed)


def main() -> None:
    parser = argparse.ArgumentParser(description="MedIn v0.3 unified JEPA")
    parser.add_argument("--config", required=True)
    parser.add_argument("--max_windows", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke_phase_steps", type=int, default=None)
    parser.add_argument("--eval_every", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=None)
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to resume.pt. Default: auto-detect save_dir/resume.pt. "
             "Pass 'none' to start from scratch.",
    )
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--per_gpu_batch_size", type=int, default=None)
    parser.add_argument("--grad_accum", type=int, default=None)
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    hf_cfg = config.get("hf", {})
    if hf_cfg.get("cache_dir"):
        os.environ["HF_HOME"] = os.path.expandvars(os.path.expanduser(str(hf_cfg["cache_dir"])))
    if hf_cfg.get("offline"):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    train(config, args)


if __name__ == "__main__":
    main()
