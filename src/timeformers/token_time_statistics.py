"""`PeriodStatistics`: typed wrapper for the `token@time` extraction caches.

Replaces the untyped `{"counts": Tensor, "sums": {"layer_1": Tensor, ...}}`
dicts written by `scripts/build_token_time_profiles.py`. Implements
`__getitem__` so existing dict-keyed helpers in `relational.py` (and scripts
that still `torch.load` the raw dict, e.g. `evaluate_relational_profile_v2.py`)
keep working unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from .relational import contextual_centroids


@dataclass
class PeriodStatistics:
    counts: Tensor
    sums: dict[str, Tensor]

    def __getitem__(self, key: str):
        if key == "counts":
            return self.counts
        if key == "sums":
            return self.sums
        raise KeyError(key)

    @classmethod
    def load(cls, path: Path) -> "PeriodStatistics":
        data = torch.load(path, map_location="cpu", weights_only=True)
        return cls(counts=data["counts"], sums=data["sums"])

    def save(self, path: Path) -> None:
        torch.save({"counts": self.counts, "sums": self.sums}, path)

    def centroids(self, layer: str) -> Tensor:
        return contextual_centroids(self, layer)
