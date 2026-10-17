"""Trajectory-window dataset.

Yields full trajectory windows with per-hour state and action texts, a
per-stay demographics text, per-hour clinical labels, and per-stay binary
outcomes. Used by the JEPA encoder-refinement step (V-JEPA 2-AC baseline)
and by the Clin-JEPA joint co-training pretraining.
"""

from __future__ import annotations

import hashlib
import logging
import os
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def _build_demo_text(age: int, gender: str, race: str) -> str:
    """Build a demographics text string from cohort fields.

    Produces the same demographics text as ``build_demographics_text`` in
    :mod:`clin_jepa.evaluation.precompute_embeddings`. The race string is kept
    as-is (uppercase cohort form).

    Examples:
        >>> _build_demo_text(53, "M", "WHITE")
        'Age: 53 years. Gender: Male. Race: WHITE.'
        >>> _build_demo_text(71, "F", "BLACK/AFRICAN AMERICAN")
        'Age: 71 years. Gender: Female. Race: BLACK/AFRICAN AMERICAN.'

    Args:
        age: Patient age in years (already cast to int).
        gender: Cohort raw gender code ("M" or "F").
        race: Cohort raw race string (uppercase).

    Returns:
        Single-line demographics text used as the ``d`` token in the predictor.
    """
    gender_text = "Male" if gender == "M" else "Female"
    return f"Age: {age} years. Gender: {gender_text}. Race: {race}."


class TrajectoryWindowDataset(Dataset):
    """Yields trajectory windows with raw text + demographics + labels.

    Each item is one trajectory window referring to a ``.pt`` shard produced
    by the data pipeline (:mod:`clin_jepa.data.step06_trajectories`).

    Args:
        shard_dir: Path to the parent dir holding ``train/``, ``val/``, and
            ``test/`` subdirs of ``.pt`` shards. The split subdir is appended
            internally.
        split: One of ``"train"``, ``"val"``, ``"test"``.
        max_windows: If set, stop loading after this many windows
            (useful for smoke tests).
        min_length: Skip windows shorter than this. The trajectory-extraction
            step already enforces a minimum length, so this is a safety net.
        index_read_only: Require a valid prebuilt ``index_cache`` and never
            rebuild it. Distributed nonzero ranks use this to avoid writers.
    """

    def __init__(
        self,
        shard_dir: str | Path,
        split: str = "train",
        max_windows: Optional[int] = None,
        min_length: int = 6,
        shard_cache_size: int = 2,
        index_cache: str | Path | None = None,
        index_read_only: bool = False,
    ) -> None:
        shard_dir = Path(shard_dir) / split
        shard_files = sorted(shard_dir.glob("*.pt"))
        if not shard_files:
            raise FileNotFoundError(f"No .pt shards found in {shard_dir}")

        self.shard_paths: list[Path] = []
        self.window_refs: list[dict] = []
        self.shard_cache_size = max(1, int(shard_cache_size))
        self._shard_cache: OrderedDict[int, dict] = OrderedDict()
        signature = hashlib.sha256()
        for path in shard_files:
            stat = path.stat()
            signature.update(f"{path.name}:{stat.st_size}".encode())
        index_cache = Path(index_cache) if index_cache is not None else None
        meta = {
            "schema_version": "trajectory_window_index_v1",
            "shard_dir": str(shard_dir.resolve()),
            "max_windows": max_windows,
            "min_length": int(min_length),
            "shard_signature": signature.hexdigest(),
        }
        cache_mismatch: dict | None = None
        if index_cache is not None and index_cache.exists():
            cached = torch.load(index_cache, map_location="cpu", weights_only=False)
            if cached.get("meta") == meta:
                self.shard_paths = [Path(value) for value in cached["shard_paths"]]
                self.window_refs = cached["window_refs"]
                logger.info(
                    "Loaded trajectory index %s (%d windows)",
                    index_cache, len(self.window_refs),
                )
                return
            cache_mismatch = {
                key: {"cached": cached.get("meta", {}).get(key), "expected": value}
                for key, value in meta.items()
                if cached.get("meta", {}).get(key) != value
            }
            logger.warning(
                "Trajectory index metadata mismatch; rebuilding %s: %s",
                index_cache,
                cache_mismatch,
            )

        if index_read_only:
            if index_cache is None:
                raise ValueError("index_read_only=True requires index_cache")
            if not index_cache.exists():
                raise FileNotFoundError(
                    f"Required prebuilt trajectory index does not exist: {index_cache}"
                )
            raise RuntimeError(
                f"Prebuilt trajectory index metadata mismatch for {index_cache}: "
                f"{cache_mismatch}"
            )

        for path in shard_files:
            shard = torch.load(path, map_location="cpu", weights_only=False)
            shard_idx = len(self.shard_paths)
            self.shard_paths.append(path)

            for w in shard["windows"]:
                length = int(w["length"])
                if length < min_length:
                    continue
                self.window_refs.append({
                    "_shard_idx":  shard_idx,
                    "stay_index":  int(w["stay_index"]),
                    "start_step":  int(w["start_step"]),
                    "length":      length,
                    "hour_indices": w["hour_indices"],  # np.int32 array, kept as-is
                    "source_type": w.get("source_type"),
                    "episode_type": w.get("episode_type"),
                })

                if max_windows and len(self.window_refs) >= max_windows:
                    break

            if max_windows and len(self.window_refs) >= max_windows:
                break
            del shard
            if len(self.shard_paths) % 10 == 0:
                logger.info(
                    "Trajectory index progress %d/%d shards, windows=%d",
                    len(self.shard_paths), len(shard_files), len(self.window_refs),
                )

        logger.info(
            "TrajectoryWindowDataset: %d windows from %d shards (%s)",
            len(self.window_refs), len(self.shard_paths), shard_dir,
        )
        if index_cache is not None:
            index_cache.parent.mkdir(parents=True, exist_ok=True)
            # A unique temporary name prevents unrelated single-process index
            # builders from colliding. Distributed training uses read-only
            # caches on nonzero ranks and therefore never writes concurrently.
            tmp = index_cache.with_suffix(index_cache.suffix + f".{os.getpid()}.tmp")
            torch.save({
                "meta": meta,
                "shard_paths": [str(path) for path in self.shard_paths],
                "window_refs": self.window_refs,
            }, tmp)
            tmp.replace(index_cache)
            logger.info("Saved trajectory index %s", index_cache)

    def _load_shard(self, shard_idx: int) -> dict:
        """Load one shard lazily; never retain the full trajectory corpus per rank."""
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
        return len(self.window_refs)

    def __getitem__(self, idx: int) -> dict:
        """Lazy gather: pull texts + demographics + labels on demand.

        Returns:
            Dict with keys:
                stay_id (int)
                start_step (int)
                length (int) — number of hours in this window
                state_texts (list[str], len=length) — one state text per hour
                action_texts (list[str], len=length) — one action text per hour
                demographics_text (str) — single-line demographics descriptor
                labels (dict[str, Tensor(length,) float32]) — clinical-target labels
                binary_labels (dict[str, bool]) — stay-level binary outcomes
        """
        w     = self.window_refs[idx]
        shard = self._load_shard(w["_shard_idx"])
        ph    = shard["per_hour"]
        phl   = shard["per_hour_labels"]
        ps    = shard["per_stay"]

        hi = w["hour_indices"].tolist()  # length L
        si = w["stay_index"]

        # Gather per-hour text lists
        state_texts  = [ph["state_texts"][h]  for h in hi]
        action_texts = [ph["action_texts"][h] for h in hi]

        labels = {
            k: torch.from_numpy(phl[k][hi].copy())
            for k in phl
        }
        time_hours = torch.tensor(
            ph.get("time_hours", ph.get("start_hours", ph["hours"]))[hi].copy(),
            dtype=torch.float32,
        )
        duration_hours = torch.tensor(
            ph.get(
                "duration_hours",
                np.ones(len(ph["hours"]), dtype=np.float32),
            )[hi].copy(),
            dtype=torch.float32,
        )
        is_icu = torch.tensor(
            ph.get(
                "is_icu",
                np.zeros(len(ph["hours"]), dtype=np.bool_),
            )[hi].copy(),
            dtype=torch.bool,
        )
        has_measured = torch.tensor(
            ph.get(
                "state_observed",
                ph.get(
                    "has_measured",
                    np.ones(len(ph["hours"]), dtype=np.bool_),
                ),
            )[hi].copy(),
            dtype=torch.bool,
        )
        has_action = torch.tensor(
            ph.get(
                "action_present",
                ph.get(
                    "has_action",
                np.asarray([
                    str(text).strip() not in {"", "No actions recorded."}
                    for text in ph["action_texts"]
                ], dtype=np.bool_),
                ),
            )[hi].copy(),
            dtype=torch.bool,
        )

        # Demographics text (same format as precompute_embeddings.build_demographics_text)
        demographics_text = _build_demo_text(
            age=int(round(float(ps["ages"][si]))),
            gender=ps["genders"][si],
            race=ps["races"][si],
        )

        # Stay-level binary outcomes
        binary_labels = {
            "icu_mortality":      bool(ps["icu_mortalities"][si]),
            "hospital_mortality": bool(ps["hospital_mortalities"][si]),
            "prolonged_stay":     bool(ps["prolonged_stays"][si]),
            "sepsis3":            bool(ps["sepsis3s"][si]),
        }

        source_type = w.get("source_type")
        if source_type is None and "source_types" in ph:
            source_type = ph["source_types"][hi[0]]
        elif source_type is None:
            source_type = "ICU" if bool(is_icu.any()) and bool(is_icu.all()) else (
                "WARD" if not bool(is_icu.any()) else "MIXED"
            )
        episode_type = w.get("episode_type")
        if episode_type is None and "episode_types" in ph:
            episode_type = ph["episode_types"][hi[0]]

        return {
            "stay_id":           int(ps["stay_ids"][si]),
            "subject_id":        int(ps["subject_ids"][si]),
            "start_step":        w["start_step"],
            "length":            w["length"],
            "state_texts":       state_texts,
            "action_texts":      action_texts,
            "time_hours":        time_hours,
            "duration_hours":    duration_hours,
            "is_icu":            is_icu,
            "has_measured":      has_measured,
            "state_observed":    has_measured.clone(),
            "has_action":        has_action,
            "action_present":    has_action.clone(),
            "source_type":       source_type,
            "episode_type":      episode_type,
            "demographics_text": demographics_text,
            "labels":            labels,
            "binary_labels":     binary_labels,
        }


def trajectory_collate_fn(batch: list[dict]) -> dict:
    """Collate trajectory windows: keep texts as lists, stack lengths.

    Returns:
        Dict with:
            batch (list[dict]): the original list of window dicts
            lengths (LongTensor, B): trajectory length per window
            max_length (int): max length in batch
    """
    lengths = torch.tensor([w["length"] for w in batch], dtype=torch.long)
    return {
        "batch":      batch,
        "lengths":    lengths,
        "max_length": int(lengths.max().item()),
        "batch_size": len(batch),
    }
