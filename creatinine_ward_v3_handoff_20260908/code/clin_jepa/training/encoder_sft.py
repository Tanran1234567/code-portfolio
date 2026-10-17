"""Encoder SFT initialisation: next-token prediction on per-hour clinical texts.

Trains a fresh LoRA adapter on top of Qwen3-8B by next-token prediction over
two independent text streams from the trajectory shards: per-hour *state*
texts (vitals, labs, severity scores) and per-hour *action* texts
(medications, ventilator settings, procedures). Each ``(stay, hour)`` pair
contributes two training samples — one state, one action. See paper §3.1.

Usage::

    # Single-GPU smoke test (~100 samples):
    python -m clin_jepa.training.encoder_sft \\
        --config configs/train/encoder_sft.yaml --max_samples 100

    # Multi-GPU production (4x H200 DDP):
    torchrun --nproc_per_node=4 -m clin_jepa.training.encoder_sft \\
        --config configs/train/encoder_sft.yaml
"""

import argparse
import hashlib
import json
import logging
import math
import os
import random
import time
from collections import OrderedDict, deque
from datetime import timedelta
from pathlib import Path
from typing import Literal

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from peft import LoraConfig, PeftModel, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithFlattening

from liger_kernel.transformers import apply_liger_kernel_to_qwen3
apply_liger_kernel_to_qwen3()

from clin_jepa.training.sampler import SourceBalancedTokenBatchSampler
from clin_jepa.data_hosp.episode_windows import CANONICAL_NO_ACTION
from clin_jepa.utils import load_config, setup_logging

logger = logging.getLogger(__name__)

EMPTY_TEXT_MARKERS = frozenset({
    "",
    "No observations recorded.",
    "No actions recorded.",
    "Clinical action:\nNo recorded intervention in the modeled action set.",
})


def _is_empty_text(text: str) -> bool:
    value = str(text).strip()
    if value in EMPTY_TEXT_MARKERS:
        return True
    if " | " in value:
        value = value.split(" | ", 1)[1].strip()
    return value in EMPTY_TEXT_MARKERS


# ---------------------------------------------------------------------------
# Dataset — reads .pt shards, pools state + action texts
# ---------------------------------------------------------------------------

class PerHourSFTDataset(Dataset):
    """Per-hour SFT dataset, reading trajectory .pt shards directly.

    The ``text_source`` argument controls which texts to include:
      - ``"state"``: only state_texts
      - ``"action"``: only action_texts
      - ``"both"``: state_texts + action_texts pooled
    """

    def __init__(
        self,
        shard_dir: str | Path,
        tokenizer: AutoTokenizer,
        max_seq_len: int = 4096,
        text_source: Literal["state", "action", "both"] = "both",
        max_samples: int | None = None,
        compute_lengths: bool = True,
        skip_empty: bool = False,
        no_action_fraction: float = 0.10,
        seed: int = 42,
        index_cache: str | Path | None = None,
        index_read_only: bool = False,
    ) -> None:
        del index_cache, index_read_only  # eager dataset does not persist an index
        if text_source not in ("state", "action", "both"):
            raise ValueError(f"text_source must be state/action/both, got {text_source!r}")

        shard_dir = Path(shard_dir)
        files = sorted(shard_dir.glob("*.pt"))
        if not files:
            raise FileNotFoundError(f"No .pt shards in {shard_dir}")

        logger.info(
            "Loading %d trajectory shards from %s (text_source=%s)",
            len(files), shard_dir, text_source,
        )

        self.texts: list[str] = []
        self.sources: list[str] = []
        self.kinds: list[str] = []
        no_action_candidates: list[tuple[str, str]] = []
        n_state_total = 0
        n_action_total = 0

        for shard_path in files:
            shard = torch.load(shard_path, weights_only=False, map_location="cpu")
            state_texts = shard["per_hour"]["state_texts"]
            action_texts = shard["per_hour"]["action_texts"]
            ph = shard["per_hour"]
            n_texts = len(state_texts)
            measured = ph.get("has_measured", [True] * n_texts)
            has_action = ph.get("has_action")
            source_types = ph.get("source_types")
            if source_types is None:
                is_icu = ph.get("is_icu", [False] * n_texts)
                source_types = ["ICU" if bool(v) else "WARD" for v in is_icu]
            if len(state_texts) != len(action_texts):
                raise ValueError(
                    f"Shard {shard_path.name}: state_texts ({len(state_texts)}) "
                    f"and action_texts ({len(action_texts)}) length mismatch"
                )

            if text_source == "both":
                # Interleave state/action: [s0, a0, s1, a1, ...]
                for i, (s, a) in enumerate(zip(state_texts, action_texts)):
                    state_valid = bool(measured[i]) and not _is_empty_text(s)
                    action_valid = (
                        bool(has_action[i]) if has_action is not None
                        else not _is_empty_text(a)
                    )
                    if not skip_empty or state_valid:
                        self.texts.append(s)
                        self.sources.append(str(source_types[i]))
                        self.kinds.append("state")
                        n_state_total += 1
                    if not skip_empty or action_valid:
                        self.texts.append(a)
                        self.sources.append(str(source_types[i]))
                        self.kinds.append("action_real" if action_valid else "action_no")
                        n_action_total += 1
                    elif skip_empty:
                        no_action_candidates.append((a, str(source_types[i])))
            elif text_source == "state":
                for i, s in enumerate(state_texts):
                    if not skip_empty or (bool(measured[i]) and not _is_empty_text(s)):
                        self.texts.append(s)
                        self.sources.append(str(source_types[i]))
                        self.kinds.append("state")
                        n_state_total += 1
            elif text_source == "action":
                for i, a in enumerate(action_texts):
                    action_valid = (
                        bool(has_action[i]) if has_action is not None
                        else not _is_empty_text(a)
                    )
                    if not skip_empty or action_valid:
                        self.texts.append(a)
                        self.sources.append(str(source_types[i]))
                        self.kinds.append("action_real" if action_valid else "action_no")
                        n_action_total += 1
                    elif skip_empty:
                        no_action_candidates.append((a, str(source_types[i])))

        if skip_empty and text_source in ("action", "both") and no_action_candidates:
            if not 0.0 <= no_action_fraction < 1.0:
                raise ValueError("no_action_fraction must be in [0, 1)")
            n_real = sum(kind == "action_real" for kind in self.kinds)
            max_no_action = int(round(
                n_real * no_action_fraction / max(1e-12, 1.0 - no_action_fraction)
            ))
            rng = random.Random(seed)
            selected: list[tuple[str, str]] = []
            # Keep ICU/WARD represented even when their raw no-action prevalence
            # differs sharply; the batch sampler performs the final balancing.
            for source in ("ICU", "WARD"):
                pool = [item for item in no_action_candidates if item[1] == source]
                rng.shuffle(pool)
                selected.extend(pool[: max_no_action // 2])
            if len(selected) < max_no_action:
                used = {id(item) for item in selected}
                remainder = [item for item in no_action_candidates if id(item) not in used]
                rng.shuffle(remainder)
                selected.extend(remainder[: max_no_action - len(selected)])
            for text, source in selected[:max_no_action]:
                self.texts.append(text)
                self.sources.append(source)
                self.kinds.append("action_no")
                n_action_total += 1

        if max_samples:
            rng = random.Random(seed)
            if text_source == "state":
                targets = {"state": max_samples}
            elif text_source == "action":
                targets = {
                    "action_real": int(round(max_samples * 0.90)),
                    "action_no": max_samples - int(round(max_samples * 0.90)),
                }
            else:
                targets = {
                    "state": int(round(max_samples * 0.50)),
                    "action_real": int(round(max_samples * 0.45)),
                }
                targets["action_no"] = max_samples - sum(targets.values())
            chosen: list[int] = []
            for kind, n_kind in targets.items():
                pool = [i for i, value in enumerate(self.kinds) if value == kind]
                rng.shuffle(pool)
                if not pool and n_kind:
                    logger.warning(
                        "Smoke SFT category %s is empty; its quota will be "
                        "redistributed. Formal training still requires data QA.",
                        kind,
                    )
                    continue
                chosen.extend(pool[:n_kind])
            if len(chosen) < max_samples:
                selected = set(chosen)
                fallback = [i for i in range(len(self.texts)) if i not in selected]
                rng.shuffle(fallback)
                chosen.extend(fallback[: max_samples - len(chosen)])
            rng.shuffle(chosen)
            self.texts = [self.texts[i] for i in chosen]
            self.sources = [self.sources[i] for i in chosen]
            self.kinds = [self.kinds[i] for i in chosen]

        logger.info(
            "Loaded %d samples (text_source=%s; state=%d, action=%d)",
            len(self.texts), text_source, n_state_total, n_action_total,
        )
        logger.info(
            "SFT effective kinds: state=%d real_action=%d no_action=%d",
            sum(kind == "state" for kind in self.kinds),
            sum(kind == "action_real" for kind in self.kinds),
            sum(kind == "action_no" for kind in self.kinds),
        )

        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        if compute_lengths:
            logger.info("Pre-tokenizing %d samples for length estimation...", len(self.texts))
            self.token_lengths: list[int] = []
            for text in self.texts:
                n_tokens = len(tokenizer.encode(text, add_special_tokens=False)) + 2
                n_tokens = min(n_tokens, max_seq_len)
                self.token_lengths.append(n_tokens)

            avg_len = sum(self.token_lengths) / len(self.token_lengths)
            total_tokens = sum(self.token_lengths)
            logger.info(
                "Dataset: %d samples, avg %.0f tokens, total %.1fM tokens",
                len(self.texts), avg_len, total_tokens / 1e6,
            )
        else:
            self.token_lengths = None
            logger.info("Dataset: %d samples (length estimation skipped — DDP mode)", len(self.texts))

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        # tokenize, leaving room for EOS
        ids = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_seq_len - 1,
            add_special_tokens=True,
        )["input_ids"]
        ids.append(self.tokenizer.eos_token_id)
        return {"input_ids": ids, "labels": ids.copy()}


class LazyPerHourSFTDataset(Dataset):
    """Full-data SFT index that keeps shard payloads out of resident memory.

    State and real-action samples are compact references into trajectory shards.
    Repeated NO_ACTION examples are synthetic references to one canonical text.
    A small LRU cache loads shards on demand during training.
    """

    def __init__(
        self,
        shard_dir: str | Path,
        tokenizer: AutoTokenizer,
        max_seq_len: int = 4096,
        text_source: Literal["state", "action", "both"] = "both",
        max_samples: int | None = None,
        compute_lengths: bool = True,
        skip_empty: bool = True,
        no_action_fraction: float = 0.10,
        seed: int = 42,
        shard_cache_size: int = 2,
        index_cache: str | Path | None = None,
        index_read_only: bool = False,
    ) -> None:
        del compute_lengths  # lengths are always built while each shard is open
        self.tokenizer = tokenizer
        self.max_seq_len = int(max_seq_len)
        self.shard_cache_size = max(1, int(shard_cache_size))
        self._shard_cache: OrderedDict[int, dict] = OrderedDict()
        self.shard_paths = sorted(Path(shard_dir).glob("*.pt"))
        if not self.shard_paths:
            raise FileNotFoundError(f"No .pt shards in {shard_dir}")
        signature = hashlib.sha256()
        for path in self.shard_paths:
            stat = path.stat()
            # Verified datasets are immutable. Shared filesystems may report
            # unstable nanosecond mtimes, so use the stable name + size pair.
            signature.update(f"{path.name}:{stat.st_size}".encode())
        shard_signature = signature.hexdigest()
        index_cache = Path(index_cache) if index_cache is not None else None
        if index_cache is not None and index_cache.exists():
            cached = torch.load(index_cache, map_location="cpu", weights_only=False)
            expected = {
                "schema_version": "lazy_sft_index_v1",
                "shard_dir": str(Path(shard_dir).resolve()),
                "text_source": text_source,
                "max_samples": max_samples,
                "no_action_fraction": float(no_action_fraction),
                "shard_signature": shard_signature,
            }
            if cached.get("meta") == expected:
                self.sample_refs = cached["sample_refs"]
                self.sources = cached["sources"]
                self.kinds = cached["kinds"]
                self.token_lengths = cached["token_lengths"]
                self.groups = cached["groups"]
                logger.info("Loaded lazy SFT index %s (%d samples)", index_cache, len(self.sample_refs))
                return
            if index_read_only:
                raise RuntimeError(
                    f"prebuilt SFT index metadata mismatch: {index_cache}; "
                    f"cached={cached.get('meta')} expected={expected}"
                )
        elif index_read_only:
            raise FileNotFoundError(f"required prebuilt SFT index missing: {index_cache}")
        self.sample_refs: list[tuple[int, int, str]] = []
        self.sources: list[str] = []
        self.kinds: list[str] = []
        self.token_lengths: list[int] = []
        self.groups: list[int] = []
        no_action_by_source = {"ICU": 0, "WARD": 0}
        rng = random.Random(seed)

        def add_ref(shard_idx: int, hour_idx: int, kind: str, source: str, text: str) -> None:
            ids = tokenizer.encode(text, add_special_tokens=False)
            self.sample_refs.append((shard_idx, hour_idx, kind))
            self.sources.append(source)
            self.kinds.append(kind)
            self.token_lengths.append(min(len(ids) + 2, self.max_seq_len))
            self.groups.append(shard_idx)

        for shard_idx, shard_path in enumerate(self.shard_paths):
            shard = torch.load(shard_path, weights_only=False, map_location="cpu")
            ph = shard["per_hour"]
            states = ph["state_texts"]
            actions = ph["action_texts"]
            observed = ph.get("state_observed", ph.get("has_measured"))
            present = ph.get("action_present", ph.get("has_action"))
            if observed is None or present is None:
                raise ValueError(
                    f"{shard_path} lacks state_observed/action_present; migrate first"
                )
            source_types = ph.get("source_types")
            if source_types is None:
                source_types = [
                    "ICU" if bool(value) else "WARD" for value in ph["is_icu"]
                ]
            for hour_idx, (state_text, action_text) in enumerate(zip(states, actions)):
                source = str(source_types[hour_idx])
                source = "ICU" if source == "ICU" else "WARD"
                if text_source in ("state", "both") and bool(observed[hour_idx]):
                    if not skip_empty or not _is_empty_text(state_text):
                        add_ref(shard_idx, hour_idx, "state", source, state_text)
                if text_source in ("action", "both"):
                    if bool(present[hour_idx]):
                        add_ref(shard_idx, hour_idx, "action_real", source, action_text)
                    else:
                        no_action_by_source[source] += 1
            del shard
            if (shard_idx + 1) % 10 == 0:
                logger.info(
                    "SFT index progress %d/%d shards, samples=%d, raw_no_action=%d",
                    shard_idx + 1, len(self.shard_paths), len(self.sample_refs),
                    sum(no_action_by_source.values()),
                )
            if max_samples:
                enough_state = sum(k == "state" for k in self.kinds) >= max_samples
                enough_action = sum(k == "action_real" for k in self.kinds) >= max_samples
                if (
                    (text_source == "state" and enough_state)
                    or (text_source == "action" and enough_action)
                    or (text_source == "both" and enough_state and enough_action)
                ):
                    break

        n_real = sum(kind == "action_real" for kind in self.kinds)
        n_no_target = min(
            sum(no_action_by_source.values()),
            int(round(n_real * no_action_fraction / max(1e-12, 1.0 - no_action_fraction))),
        )
        no_action_token_len = min(
            len(tokenizer.encode(CANONICAL_NO_ACTION, add_special_tokens=False)) + 2,
            self.max_seq_len,
        )
        remaining = n_no_target
        for source in ("ICU", "WARD"):
            take = min(no_action_by_source[source], n_no_target // 2)
            for _ in range(take):
                self.sample_refs.append((-1, -1, "action_no"))
                self.sources.append(source)
                self.kinds.append("action_no")
                self.token_lengths.append(no_action_token_len)
                self.groups.append(-1)
            remaining -= take
        fallback_sources = [s for s in ("ICU", "WARD") if no_action_by_source[s] > 0]
        for i in range(max(0, remaining)):
            source = fallback_sources[i % len(fallback_sources)]
            self.sample_refs.append((-1, -1, "action_no"))
            self.sources.append(source)
            self.kinds.append("action_no")
            self.token_lengths.append(no_action_token_len)
            self.groups.append(-1)

        if max_samples and len(self.sample_refs) > max_samples:
            targets = (
                {"state": max_samples}
                if text_source == "state" else
                {"action_real": int(max_samples * 0.9), "action_no": max_samples - int(max_samples * 0.9)}
                if text_source == "action" else
                {"state": int(max_samples * 0.5), "action_real": int(max_samples * 0.45), "action_no": max_samples - int(max_samples * 0.95)}
            )
            chosen: list[int] = []
            for kind, target in targets.items():
                pool = [i for i, value in enumerate(self.kinds) if value == kind]
                rng.shuffle(pool)
                chosen.extend(pool[:target])
            if len(chosen) < max_samples:
                used = set(chosen)
                pool = [i for i in range(len(self.sample_refs)) if i not in used]
                rng.shuffle(pool)
                chosen.extend(pool[: max_samples - len(chosen)])
            self.sample_refs = [self.sample_refs[i] for i in chosen]
            self.sources = [self.sources[i] for i in chosen]
            self.kinds = [self.kinds[i] for i in chosen]
            self.token_lengths = [self.token_lengths[i] for i in chosen]
            self.groups = [self.groups[i] for i in chosen]

        logger.info(
            "Lazy SFT index: samples=%d shards_scanned<=%d state=%d real_action=%d no_action=%d",
            len(self.sample_refs), len(self.shard_paths),
            sum(k == "state" for k in self.kinds),
            sum(k == "action_real" for k in self.kinds),
            sum(k == "action_no" for k in self.kinds),
        )
        if index_cache is not None:
            index_cache.parent.mkdir(parents=True, exist_ok=True)
            meta = {
                "schema_version": "lazy_sft_index_v1",
                "shard_dir": str(Path(shard_dir).resolve()),
                "text_source": text_source,
                "max_samples": max_samples,
                "no_action_fraction": float(no_action_fraction),
                "shard_signature": shard_signature,
            }
            tmp = index_cache.with_suffix(index_cache.suffix + f".{os.getpid()}.tmp")
            torch.save({
                "meta": meta,
                "sample_refs": self.sample_refs,
                "sources": self.sources,
                "kinds": self.kinds,
                "token_lengths": self.token_lengths,
                "groups": self.groups,
            }, tmp)
            tmp.replace(index_cache)
            logger.info("Saved lazy SFT index %s", index_cache)

    def _load_shard(self, shard_idx: int) -> dict:
        if shard_idx in self._shard_cache:
            shard = self._shard_cache.pop(shard_idx)
            self._shard_cache[shard_idx] = shard
            return shard
        shard = torch.load(
            self.shard_paths[shard_idx], map_location="cpu", weights_only=False,
        )
        self._shard_cache[shard_idx] = shard
        while len(self._shard_cache) > self.shard_cache_size:
            self._shard_cache.popitem(last=False)
        return shard

    def __len__(self) -> int:
        return len(self.sample_refs)

    def __getitem__(self, idx: int) -> dict:
        shard_idx, hour_idx, kind = self.sample_refs[idx]
        if kind == "action_no":
            text = CANONICAL_NO_ACTION
        else:
            ph = self._load_shard(shard_idx)["per_hour"]
            text = (
                ph["state_texts"][hour_idx]
                if kind == "state" else ph["action_texts"][hour_idx]
            )
        ids = self.tokenizer(
            text, truncation=True, max_length=self.max_seq_len - 1,
            add_special_tokens=True,
        )["input_ids"]
        ids.append(self.tokenizer.eos_token_id)
        return {"input_ids": ids, "labels": ids.copy()}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def setup_distributed() -> tuple[int, int, bool]:
    """Initialize distributed training if available."""
    if "RANK" in os.environ:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            "nccl", timeout=timedelta(hours=2),
            device_id=torch.device("cuda", local_rank),
        )
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        return rank, world_size, True
    return 0, 1, False


def cleanup_distributed(is_distributed: bool) -> None:
    if is_distributed:
        dist.destroy_process_group()


def create_model(config: dict, device: torch.device) -> tuple:
    """Load Qwen3-8B with a fresh or continued LoRA adapter."""
    model_cfg = config["model"]
    hf_cfg = config.get("hf", {})

    logger.info("Loading base model: %s", model_cfg["base_model"])
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["base_model"],
        torch_dtype=torch.bfloat16,
        attn_implementation=model_cfg["attn_implementation"],
        local_files_only=hf_cfg.get("offline", False),
    )

    # Freeze all base parameters
    for param in model.parameters():
        param.requires_grad = False

    initial_adapter = model_cfg.get("initial_adapter")
    if initial_adapter:
        initial_adapter = os.path.expandvars(os.path.expanduser(initial_adapter))
        if not Path(initial_adapter).exists():
            raise FileNotFoundError(
                f"Initial LoRA adapter not found: {initial_adapter}"
            )
        adapter_config_path = Path(initial_adapter) / "adapter_config.json"
        if not adapter_config_path.exists():
            raise FileNotFoundError(adapter_config_path)
        with adapter_config_path.open() as f:
            adapter_config = json.load(f)
        expected_lora = model_cfg.get("lora")
        if expected_lora and model_cfg.get("strict_adapter_config", True):
            checks = {
                "r": (adapter_config.get("r"), expected_lora.get("r")),
                "lora_alpha": (
                    adapter_config.get("lora_alpha"),
                    expected_lora.get("alpha"),
                ),
                "bias": (adapter_config.get("bias"), expected_lora.get("bias", "none")),
            }
            mismatches = {
                key: {"checkpoint": actual, "expected": expected}
                for key, (actual, expected) in checks.items()
                if expected is not None and actual != expected
            }
            actual_targets = set(adapter_config.get("target_modules") or [])
            expected_targets = set(expected_lora.get("target_modules") or [])
            if expected_targets and actual_targets != expected_targets:
                mismatches["target_modules"] = {
                    "checkpoint": sorted(actual_targets),
                    "expected": sorted(expected_targets),
                }
            if mismatches:
                raise ValueError(
                    "Initial LoRA is incompatible with Hospital-SFT config: "
                    f"{mismatches}"
                )
        logger.info("Continuing SFT from LoRA adapter: %s", initial_adapter)
        model = PeftModel.from_pretrained(
            model, initial_adapter, is_trainable=True
        )
    else:
        logger.info("Creating fresh LoRA on frozen Qwen (paper from-scratch, no ICU adapter)")
        lora_cfg = model_cfg["lora"]
        peft_config = LoraConfig(
            r=lora_cfg["r"],
            lora_alpha=lora_cfg["alpha"],
            target_modules=lora_cfg["target_modules"],
            lora_dropout=lora_cfg["dropout"],
            bias=lora_cfg.get("bias", "none"),
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Enable gradient checkpointing
    if model_cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    model = model.to(device)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        initial_adapter or model_cfg["base_model"],
        local_files_only=hf_cfg.get("offline", False),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def evaluate(model: torch.nn.Module, val_loader: DataLoader, device: torch.device,
             max_batches: int | None = 50) -> float:
    """Run validation and return average loss."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            if max_batches is not None and n_batches >= max_batches:
                break
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item()
            n_batches += 1

    model.train()
    return total_loss / max(n_batches, 1)


def train(config: dict, args: argparse.Namespace) -> None:
    """Main training function."""
    rank, world_size, is_distributed = setup_distributed()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    is_main = rank == 0

    if is_main:
        setup_logging()
        logger.info("Encoder SFT — %d GPUs", world_size)

    # --- Model ---
    model, tokenizer = create_model(config, device)

    if is_distributed:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    raw_model = model.module if is_distributed else model

    # --- Data ---
    data_cfg = config["data"]
    project_root = Path(__file__).resolve().parents[2]

    need_lengths = True
    skip_empty = bool(data_cfg.get("skip_empty", False))
    raw_save_dir = Path(os.path.expandvars(str(config["save"]["dir"])))
    if not raw_save_dir.is_absolute():
        raw_save_dir = project_root / raw_save_dir
    index_dir = raw_save_dir / "sft_indices"
    index_suffix = f"smoke{args.max_samples}" if args.max_samples else "full"

    def _data_dir(key: str) -> Path:
        raw = Path(os.path.expandvars(os.path.expanduser(str(data_cfg[key]))))
        return raw if raw.is_absolute() else project_root / raw

    dataset_cls = (
        LazyPerHourSFTDataset if data_cfg.get("lazy_shards", False)
        else PerHourSFTDataset
    )
    if is_distributed and data_cfg.get("lazy_shards", False):
        required_indices = [
            index_dir / f"train_both_{index_suffix}.pt",
            index_dir / f"val_state_{index_suffix}.pt",
            index_dir / f"val_action_{index_suffix}.pt",
        ]
        missing_indices = [str(path) for path in required_indices if not path.exists()]
        if missing_indices:
            raise RuntimeError(
                "Distributed lazy SFT requires prebuilt indices; run "
                "scripts/build_sft_index_v03.py first. Missing: "
                f"{missing_indices}"
            )
    train_dataset = dataset_cls(
        shard_dir=_data_dir("train_dir"),
        tokenizer=tokenizer,
        max_seq_len=data_cfg["max_seq_len"],
        text_source="both",
        max_samples=args.max_samples,
        compute_lengths=need_lengths,
        skip_empty=skip_empty,
        no_action_fraction=float(data_cfg.get("no_action_sft_fraction", 0.10)),
        seed=int(config["training"].get("seed", 42)),
        index_cache=(index_dir / f"train_both_{index_suffix}.pt") if data_cfg.get("lazy_shards", False) else None,
        index_read_only=is_distributed and data_cfg.get("lazy_shards", False),
    )

    configured_val_max = data_cfg.get("val_max_samples")
    val_max_samples = (
        min(args.max_samples, configured_val_max or args.max_samples)
        if args.max_samples
        else configured_val_max
    )
    val_state_dataset = dataset_cls(
        shard_dir=_data_dir("val_dir"),
        tokenizer=tokenizer,
        max_seq_len=data_cfg["max_seq_len"],
        text_source="state",
        max_samples=val_max_samples,
        compute_lengths=False,
        skip_empty=skip_empty,
        no_action_fraction=float(data_cfg.get("no_action_sft_fraction", 0.10)),
        seed=int(config["training"].get("seed", 42)),
        index_cache=(index_dir / f"val_state_{index_suffix}.pt") if data_cfg.get("lazy_shards", False) else None,
        index_read_only=is_distributed and data_cfg.get("lazy_shards", False),
    )
    val_action_dataset = dataset_cls(
        shard_dir=_data_dir("val_dir"),
        tokenizer=tokenizer,
        max_seq_len=data_cfg["max_seq_len"],
        text_source="action",
        max_samples=val_max_samples,
        compute_lengths=False,
        skip_empty=skip_empty,
        no_action_fraction=float(data_cfg.get("no_action_sft_fraction", 0.10)),
        seed=int(config["training"].get("seed", 42)),
        index_cache=(index_dir / f"val_action_{index_suffix}.pt") if data_cfg.get("lazy_shards", False) else None,
        index_read_only=is_distributed and data_cfg.get("lazy_shards", False),
    )

    collator = DataCollatorWithFlattening(return_flash_attn_kwargs=True)

    train_cfg = config["training"]
    n_real_action = sum(kind == "action_real" for kind in train_dataset.kinds)
    n_no_action = sum(kind == "action_no" for kind in train_dataset.kinds)
    available_no_action_fraction = n_no_action / max(1, n_real_action + n_no_action)
    configured_no_action_cap = float(
        train_cfg.get("no_action_fraction_within_action", 0.10)
    )
    effective_no_action_fraction = min(
        configured_no_action_cap, available_no_action_fraction
    )
    if is_main:
        logger.info(
            "Action SFT availability: real=%d no_action=%d available=%.4f "
            "configured_cap=%.4f effective_sampler_fraction=%.4f",
            n_real_action, n_no_action, available_no_action_fraction,
            configured_no_action_cap, effective_no_action_fraction,
        )

    train_sampler = SourceBalancedTokenBatchSampler(
        lengths=train_dataset.token_lengths,
        sources=train_dataset.sources,
        kinds=train_dataset.kinds,
        groups=getattr(train_dataset, "groups", None),
        max_tokens=train_cfg.get("token_budget", 32768),
        max_batch_size=train_cfg.get("max_batch_size", 128),
        icu_frac=train_cfg.get("icu_frac", 0.5),
        state_fraction=train_cfg.get("state_fraction", 0.5),
        no_action_fraction_within_action=effective_no_action_fraction,
        seed=train_cfg.get("seed", 42),
        num_replicas=world_size,
        rank=rank,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=collator,
        num_workers=0 if data_cfg.get("lazy_shards", False) else 4,
        pin_memory=True,
    )

    # Val loaders: same packing, smaller batch
    val_state_loader = DataLoader(
        val_state_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collator,
        num_workers=0 if data_cfg.get("lazy_shards", False) else 2,
    )
    val_action_loader = DataLoader(
        val_action_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collator,
        num_workers=0 if data_cfg.get("lazy_shards", False) else 2,
    )

    # --- Optimizer & Scheduler ---
    optimizer = torch.optim.AdamW(
        [p for p in raw_model.parameters() if p.requires_grad],
        lr=train_cfg["lr"],
        weight_decay=train_cfg.get("weight_decay", 0.01),
    )

    total_steps = len(train_loader) * train_cfg["epochs"]
    warmup_steps = int(total_steps * train_cfg.get("warmup_ratio", 0.1))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # --- Training Loop ---
    log_cfg = config.get("logging", {})
    save_cfg = config.get("save", {})
    save_dir = project_root / save_cfg.get("dir", "checkpoints/encoder_sft")
    save_dir.mkdir(parents=True, exist_ok=True)

    log_every = log_cfg.get("log_every", 50)
    eval_every = log_cfg.get("eval_every", 500)
    eval_max_batches = log_cfg.get("eval_max_batches", 50)
    grad_clip = train_cfg.get("gradient_clip", 1.0)

    # CLI overrides (for smoke tests)
    if getattr(args, "eval_every", None) is not None:
        eval_every = args.eval_every
        if is_main:
            logger.info("CLI override: eval_every = %d", eval_every)
    if getattr(args, "log_every", None) is not None:
        log_every = args.log_every
        if is_main:
            logger.info("CLI override: log_every = %d", log_every)

    best_val_loss = float("inf")
    best_val_step = 0
    early_cfg = config.get("early_stopping", {})
    early_enabled = bool(early_cfg.get("enabled", False))
    early_patience = int(early_cfg.get("patience", 4))
    early_min_delta = float(early_cfg.get("min_delta_relative", 0.01))
    evaluations_without_improvement = 0
    validation_history: list[dict] = []
    last_eval_step = -1
    stopped_early = False
    rolling_losses = deque(maxlen=20)
    total_train_tokens = 0
    global_step = 0
    t0 = time.time()

    if is_main:
        logger.info("Training: %d epochs, %d steps/epoch, %d total steps",
                     train_cfg["epochs"], len(train_loader), total_steps)
        logger.info("Warmup: %d steps, LR: %s, WD: %s",
                     warmup_steps, train_cfg["lr"], train_cfg.get("weight_decay", 0.01))
        logger.info(
            "Early stopping: enabled=%s patience=%d min_delta_relative=%.4f",
            early_enabled, early_patience, early_min_delta,
        )

    def run_validation(reason: str) -> bool:
        """Evaluate both streams, checkpoint best/latest, and sync stop flag."""
        nonlocal best_val_loss, best_val_step
        nonlocal evaluations_without_improvement, last_eval_step
        stop_now = False
        if is_distributed:
            dist.barrier()
        if is_main:
            val_loss_state = evaluate(
                raw_model, val_state_loader, device,
                max_batches=eval_max_batches,
            )
            val_loss_action = evaluate(
                raw_model, val_action_loader, device,
                max_batches=eval_max_batches,
            )
            val_loss = (val_loss_state + val_loss_action) / 2
            if not math.isfinite(val_loss):
                raise FloatingPointError(
                    f"non-finite SFT validation loss at step={global_step}"
                )
            previous_best = best_val_loss
            improved = (
                not math.isfinite(previous_best)
                or val_loss < previous_best * (1.0 - early_min_delta)
            )
            validation_history.append({
                "step": global_step,
                "reason": reason,
                "val_loss": val_loss,
                "val_loss_state": val_loss_state,
                "val_loss_action": val_loss_action,
                "improved": improved,
            })
            logger.info(
                "Val [%s] step=%d combined=%.4f state=%.4f action=%.4f "
                "best=%.4f improved=%s bad_evals=%d/%d",
                reason, global_step, val_loss, val_loss_state,
                val_loss_action, previous_best, improved,
                evaluations_without_improvement, early_patience,
            )

            latest_path = save_dir / "latest"
            raw_model.save_pretrained(latest_path)
            tokenizer.save_pretrained(latest_path)
            with open(save_dir / "latest_meta.json", "w") as f:
                json.dump({
                    "step": global_step,
                    "reason": reason,
                    "val_loss": val_loss,
                    "val_loss_state": val_loss_state,
                    "val_loss_action": val_loss_action,
                    "elapsed_seconds": time.time() - t0,
                }, f, indent=2)

            if improved:
                best_val_loss = val_loss
                best_val_step = global_step
                evaluations_without_improvement = 0
                if save_cfg.get("save_best", True):
                    best_path = save_dir / "best"
                    raw_model.save_pretrained(best_path)
                    tokenizer.save_pretrained(best_path)
                    with open(save_dir / "best_meta.json", "w") as f:
                        json.dump(validation_history[-1], f, indent=2)
                    logger.info("Saved best checkpoint to %s", best_path)
            else:
                evaluations_without_improvement += 1

            stop_now = (
                early_enabled
                and evaluations_without_improvement >= early_patience
            )
            if stop_now:
                logger.info(
                    "EARLY_STOP step=%d best_step=%d best_val=%.4f",
                    global_step, best_val_step, best_val_loss,
                )
            last_eval_step = global_step

        if is_distributed:
            stop_tensor = torch.tensor(
                [int(stop_now)], device=device, dtype=torch.int32,
            )
            dist.broadcast(stop_tensor, src=0)
            stop_now = bool(stop_tensor.item())
            dist.barrier()
        model.train()
        return stop_now

    model.train()
    for epoch in range(train_cfg["epochs"]):
        train_sampler.set_epoch(epoch)

        epoch_loss = 0.0
        epoch_tokens = 0
        epoch_steps = 0

        for batch in train_loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss

            if not bool(torch.isfinite(loss).all()):
                raise FloatingPointError(
                    f"non-finite SFT loss at step={global_step}: {loss}"
                )

            loss.backward()

            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in raw_model.parameters() if p.requires_grad],
                    max_norm=grad_clip,
                    error_if_nonfinite=True,
                )

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            n_tokens = (batch["labels"] != -100).sum().item()
            epoch_loss += loss.item() * n_tokens
            epoch_tokens += n_tokens
            total_train_tokens += n_tokens
            rolling_losses.append((loss.item() * n_tokens, n_tokens))
            epoch_steps += 1
            global_step += 1

            # Log
            if is_main and global_step % log_every == 0:
                avg_loss = epoch_loss / max(epoch_tokens, 1)
                rolling_loss = (
                    sum(value for value, _tokens in rolling_losses)
                    / max(1, sum(tokens for _value, tokens in rolling_losses))
                )
                lr = scheduler.get_last_lr()[0]
                elapsed = time.time() - t0
                tokens_per_sec = total_train_tokens / elapsed
                mem_gb = torch.cuda.max_memory_allocated(device) / 1e9 if torch.cuda.is_available() else 0.0
                logger.info(
                    "Step %d/%d | batch_loss=%.4f rolling20=%.4f "
                    "epoch_avg=%.4f | lr=%.2e | %.0f tok/s | "
                    "peak VRAM %.1f GB | %.1f min elapsed",
                    global_step, total_steps, float(loss.detach()),
                    rolling_loss, avg_loss, lr, tokens_per_sec, mem_gb,
                    elapsed / 60,
                )

            # Eval (state + action separately, then combined for best-checkpoint selection)
            if global_step % eval_every == 0:
                if run_validation("periodic"):
                    stopped_early = True
                    break

        # End of epoch
        if is_main:
            avg_epoch_loss = epoch_loss / max(epoch_tokens, 1)
            logger.info(
                "Epoch %d complete | avg_loss=%.4f | %d steps | %.1fM tokens",
                epoch, avg_epoch_loss, epoch_steps, epoch_tokens / 1e6,
            )

        # A short epoch must still produce validation evidence even when it
        # never reaches eval_every. Avoid a duplicate evaluation if the final
        # step already triggered the periodic path.
        if not stopped_early and global_step % eval_every != 0:
            if run_validation(f"epoch_{epoch + 1}_end"):
                stopped_early = True

        if stopped_early:
            break

    # --- Save final checkpoint ---
    if is_main and save_cfg.get("save_final", True):
        final_path = save_dir / "final"
        raw_model.save_pretrained(final_path)
        tokenizer.save_pretrained(final_path)
        logger.info("Saved final checkpoint to %s", final_path)

        # Save training metadata
        meta = {
            "schema_version": "encoder_sft_v3_episode_no_action_lazy",
            "total_steps": global_step,
            "total_tokens": epoch_tokens,
            "final_train_loss": epoch_loss / max(epoch_tokens, 1),
            "best_val_loss": best_val_loss,
            "best_val_step": best_val_step,
            "stopped_early": stopped_early,
            "epochs_completed": epoch + 1,
            "validation_history": validation_history,
            "config": config,
            "elapsed_seconds": time.time() - t0,
        }
        with open(save_dir / "training_meta.json", "w") as f:
            json.dump(meta, f, indent=2, default=str)

    cleanup_distributed(is_distributed)
    if is_main:
        logger.info("Encoder SFT complete. Total time: %.1f min", (time.time() - t0) / 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Encoder SFT (NTP on state + action texts)")
    parser.add_argument("--config", required=True, help="Path to configs/train/encoder_sft.yaml")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit samples (testing)")
    parser.add_argument("--eval_every", type=int, default=None, help="Override eval cadence (for smoke)")
    parser.add_argument("--log_every", type=int, default=None, help="Override log cadence (for smoke)")
    args = parser.parse_args()

    config = load_config(args.config)

    # Set HF environment
    hf_cfg = config.get("hf", {})
    if hf_cfg.get("cache_dir"):
        os.environ["HF_HOME"] = hf_cfg["cache_dir"]
    if hf_cfg.get("offline"):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    train(config, args)


if __name__ == "__main__":
    main()
