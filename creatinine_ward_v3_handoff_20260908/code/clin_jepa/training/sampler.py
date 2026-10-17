"""Token-budget batch sampler for variable-length training.

Groups samples by sequence length so short samples get larger batches and
long samples get smaller batches, all bounded by a target token budget.
"""

import logging
import random
from typing import Iterator

logger = logging.getLogger(__name__)


class TokenBudgetBatchSampler:
    """Creates variable-size batches bounded by a total token budget.

    Two modes:
    - padding mode (packing=False): budget = max_len_in_batch × batch_size
    - packing mode (packing=True): budget = sum of all sample lengths

    Args:
        lengths: Per-sample token lengths.
        max_tokens: Maximum total tokens per micro-batch.
        max_batch_size: Hard cap on batch size (for gradient memory limits).
        shuffle: Whether to shuffle batch order each epoch.
        seed: Random seed for reproducibility.
        drop_last: Drop the last incomplete batch.
        packing: If True, use sum-based budget (no padding waste).
    """

    def __init__(
        self,
        lengths: list[int],
        max_tokens: int = 32768,
        max_batch_size: int = 16,
        shuffle: bool = True,
        seed: int = 42,
        drop_last: bool = True,
        packing: bool = False,
    ) -> None:
        self.lengths = lengths
        self.max_tokens = max_tokens
        self.max_batch_size = max_batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.packing = packing
        self.rng = random.Random(seed)

        self._batches = self._create_batches()
        self._log_stats()

    def _create_batches(self) -> list[list[int]]:
        """Create batches by sorting by length and bin-packing."""
        # Sort by length, longest first
        sorted_indices = sorted(
            range(len(self.lengths)),
            key=lambda i: self.lengths[i],
            reverse=True,
        )

        batches = []
        current_batch: list[int] = []
        current_max_len = 0
        current_sum_len = 0

        for idx in sorted_indices:
            sample_len = self.lengths[idx]

            if self.packing:
                # Packing: budget = sum of lengths
                new_total_tokens = current_sum_len + sample_len
            else:
                # Padding mode: budget = max_len × batch_size
                new_max_len = max(current_max_len, sample_len)
                new_total_tokens = (len(current_batch) + 1) * new_max_len

            if (new_total_tokens > self.max_tokens
                    or len(current_batch) >= self.max_batch_size):
                if current_batch:
                    batches.append(current_batch)
                current_batch = [idx]
                current_max_len = sample_len
                current_sum_len = sample_len
            else:
                current_batch.append(idx)
                current_max_len = max(current_max_len, sample_len)
                current_sum_len += sample_len

        # Handle last batch
        if current_batch:
            if not self.drop_last or len(current_batch) >= 1:
                batches.append(current_batch)

        return batches

    def _log_stats(self) -> None:
        """Log batch statistics."""
        batch_sizes = [len(b) for b in self._batches]
        if not batch_sizes:
            return

        avg_bs = sum(batch_sizes) / len(batch_sizes)
        min_bs = min(batch_sizes)
        max_bs = max(batch_sizes)

        # Compute effective tokens per batch (padded)
        token_counts = []
        for batch in self._batches:
            max_len = max(self.lengths[i] for i in batch)
            token_counts.append(len(batch) * max_len)
        avg_tokens = sum(token_counts) / len(token_counts)

        logger.info(
            "TokenBudgetBatchSampler: %d batches, batch_size min=%d avg=%.1f max=%d, "
            "avg_tokens_per_batch=%.0f (budget=%d)",
            len(self._batches), min_bs, avg_bs, max_bs,
            avg_tokens, self.max_tokens,
        )

    def __iter__(self) -> Iterator[list[int]]:
        """Yield batch index lists, optionally shuffled."""
        batches = self._batches.copy()
        if self.shuffle:
            self.rng.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        return len(self._batches)

    def set_epoch(self, epoch: int) -> None:
        """Reset RNG seed for deterministic shuffling across epochs."""
        self.rng = random.Random(42 + epoch)


class SourceBalancedTokenBatchSampler:
    """Source-balanced, token-budgeted batches with deterministic DDP sharding.

    A global ICU/WARD-balanced sample stream is built first, packed by the
    flattened-token budget, padded to a multiple of ``num_replicas``, and then
    sharded by rank. Every rank therefore executes the same number of optimizer
    steps without giving up the token budget used by the single-GPU path.
    """

    def __init__(
        self,
        lengths: list[int],
        sources: list[str],
        kinds: list[str] | None = None,
        groups: list[int] | None = None,
        max_tokens: int = 32768,
        max_batch_size: int = 128,
        icu_frac: float = 0.5,
        state_fraction: float = 0.5,
        no_action_fraction_within_action: float = 0.10,
        seed: int = 42,
        num_replicas: int = 1,
        rank: int = 0,
    ) -> None:
        if len(lengths) != len(sources):
            raise ValueError("lengths and sources must have identical length")
        if kinds is not None and len(kinds) != len(lengths):
            raise ValueError("kinds and lengths must have identical length")
        if groups is not None and len(groups) != len(lengths):
            raise ValueError("groups and lengths must have identical length")
        if not lengths:
            raise ValueError("cannot build batches from an empty dataset")
        if num_replicas < 1 or not 0 <= rank < num_replicas:
            raise ValueError(f"invalid rank={rank}, num_replicas={num_replicas}")
        self.lengths = lengths
        self.sources = sources
        self.kinds = kinds
        self.groups = groups
        self.max_tokens = int(max_tokens)
        self.max_batch_size = int(max_batch_size)
        self.icu_frac = float(icu_frac)
        self.state_fraction = float(state_fraction)
        self.no_action_fraction_within_action = float(no_action_fraction_within_action)
        if not 0.0 <= self.icu_frac <= 1.0:
            raise ValueError("icu_frac must be in [0, 1]")
        if not 0.0 <= self.state_fraction <= 1.0:
            raise ValueError("state_fraction must be in [0, 1]")
        if not 0.0 <= self.no_action_fraction_within_action <= 1.0:
            raise ValueError("no_action_fraction_within_action must be in [0, 1]")
        self.seed = int(seed)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.epoch = 0
        self._rank_batches = self._build_rank_batches()

    def _draw(self, pool: list[int], n: int, rng: random.Random) -> list[int]:
        return [pool[rng.randrange(len(pool))] for _ in range(n)]

    def _build_rank_batches(self) -> list[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        total = len(self.sources)
        if self.kinds is None:
            category_targets = {"all": total}
        else:
            n_state = int(round(total * self.state_fraction))
            n_action = total - n_state
            n_no_action = int(round(n_action * self.no_action_fraction_within_action))
            category_targets = {
                "state": n_state,
                "action_real": n_action - n_no_action,
                "action_no": n_no_action,
            }
        indices: list[int] = []
        if self.kinds is not None:
            fallback_order = {
                "action_no": ("action_real", "state"),
                "action_real": ("state", "action_no"),
                "state": ("action_real", "action_no"),
            }
            for category, n_category in list(category_targets.items()):
                if n_category <= 0:
                    continue
                present = any(kind == category for kind in self.kinds)
                if present:
                    continue
                replacement = next(
                    (
                        candidate for candidate in fallback_order[category]
                        if any(kind == candidate for kind in self.kinds)
                    ),
                    None,
                )
                if replacement is None:
                    raise ValueError("SFT dataset has no usable modality categories")
                logger.warning(
                    "SFT category %s is empty; redistributing %d samples to %s. "
                    "This is smoke robustness, not a substitute for data QA.",
                    category, n_category, replacement,
                )
                category_targets[replacement] += n_category
                category_targets[category] = 0
        for category, n_category in category_targets.items():
            category_pool = [
                i for i in range(total)
                if category == "all" or self.kinds[i] == category
            ]
            if n_category and not category_pool:
                raise ValueError(f"internal sampler redistribution failed for {category}")
            icu = [i for i in category_pool if self.sources[i] == "ICU"]
            ward = [i for i in category_pool if self.sources[i] != "ICU"]
            n_icu = int(round(n_category * self.icu_frac))
            if not icu:
                n_icu = 0
            if not ward:
                n_icu = n_category
            indices.extend(self._draw(icu, n_icu, rng) if n_icu else [])
            n_ward = n_category - n_icu
            indices.extend(self._draw(ward, n_ward, rng) if n_ward else [])
        if self.groups is None:
            rng.shuffle(indices)
        else:
            # Keep consecutive micro-batches near the same shard so lazy
            # loading does not deserialize a large .pt file per sample.
            grouped: dict[int, list[int]] = {}
            for idx in indices:
                grouped.setdefault(int(self.groups[idx]), []).append(idx)
            group_ids = list(grouped)
            rng.shuffle(group_ids)
            indices = []
            for group_id in group_ids:
                rng.shuffle(grouped[group_id])
                indices.extend(grouped[group_id])

        batches: list[list[int]] = []
        current: list[int] = []
        current_tokens = 0
        for idx in indices:
            sample_tokens = max(1, int(self.lengths[idx]))
            if current and (
                current_tokens + sample_tokens > self.max_tokens
                or len(current) >= self.max_batch_size
            ):
                batches.append(current)
                current = []
                current_tokens = 0
            current.append(idx)
            current_tokens += sample_tokens
        if current:
            batches.append(current)
        while len(batches) % self.num_replicas:
            batches.append(list(batches[len(batches) % max(1, len(batches))]))
        return batches[self.rank::self.num_replicas]

    def __iter__(self) -> Iterator[list[int]]:
        yield from self._rank_batches

    def __len__(self) -> int:
        return len(self._rank_batches)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self._rank_batches = self._build_rank_batches()
