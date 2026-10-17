"""Source-balanced window sampler (ICU vs WARD). Metadata only; not a model input."""

from __future__ import annotations

from collections.abc import Iterator
import math
import random

from torch.utils.data import Sampler


class SourceBalancedSampler(Sampler[int]):
    """Yield indices so that ICU and WARD windows are drawn at a target fraction."""

    def __init__(
        self,
        sources: list[str],
        icu_frac: float = 0.5,
        seed: int = 42,
        replacement: bool = True,
        num_replicas: int = 1,
        rank: int = 0,
        groups: list[int] | None = None,
    ) -> None:
        self.icu = [i for i, src in enumerate(sources) if src == "ICU"]
        self.ward = [i for i, src in enumerate(sources) if src != "ICU"]
        self.icu_frac = float(icu_frac)
        self.seed = int(seed)
        self.replacement = replacement
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.groups = groups
        if groups is not None and len(groups) != len(sources):
            raise ValueError("groups and sources must have identical length")
        if self.num_replicas < 1 or not 0 <= self.rank < self.num_replicas:
            raise ValueError(
                f"invalid distributed sampler rank={rank}, replicas={num_replicas}"
            )
        if not self.icu:
            self.icu = list(range(len(sources)))
        if not self.ward:
            self.ward = list(range(len(sources)))
        self._epoch = 0

    def __len__(self) -> int:
        return math.ceil((len(self.icu) + len(self.ward)) / self.num_replicas)

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed + self._epoch)
        self._epoch += 1
        n = len(self) * self.num_replicas
        n_icu = int(round(n * self.icu_frac))
        n_ward = n - n_icu
        if self.replacement:
            icu = [self.icu[rng.randrange(len(self.icu))] for _ in range(n_icu)]
            ward = [self.ward[rng.randrange(len(self.ward))] for _ in range(n_ward)]
        else:
            icu = (self.icu * math.ceil(n_icu / len(self.icu)))[:n_icu]
            ward = (self.ward * math.ceil(n_ward / len(self.ward)))[:n_ward]
            rng.shuffle(icu)
            rng.shuffle(ward)
        mixed = icu + ward
        if self.groups is None:
            rng.shuffle(mixed)
        else:
            grouped: dict[int, list[int]] = {}
            for idx in mixed:
                grouped.setdefault(int(self.groups[idx]), []).append(idx)
            group_ids = list(grouped)
            rng.shuffle(group_ids)
            mixed = []
            for group_id in group_ids:
                rng.shuffle(grouped[group_id])
                mixed.extend(grouped[group_id])
        return iter(mixed[self.rank::self.num_replicas])
