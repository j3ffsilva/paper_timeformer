"""`PeriodStatistics`: a typed wrapper for the per-period extraction caches
produced by `scripts/build_token_time_profiles.py`.

For a single period, the raw cache stores two tensors:

- `counts`: `(vocab_size,)`, how many times each vocabulary item occurred in
  this period's corpus;
- `sums`: `{"layer_1": (vocab_size, hidden_size), "layer_2": ..., ...}`, the
  elementwise sum of the hidden states of each vocabulary item, at each
  encoder layer, over all its occurrences.

`PeriodStatistics` is a typed dataclass around these two fields. It also
implements `__getitem__("counts"|"sums")`, so the existing dict-keyed helpers
in `relational.py` (`stats["counts"]`, `stats["sums"][layer]`) work unchanged
on a `PeriodStatistics` instance.
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
        """Load `{"counts": Tensor, "sums": {...}}` from a `.pt` cache file."""
        data = torch.load(path, map_location="cpu", weights_only=True)
        return cls(counts=data["counts"], sums=data["sums"])

    def save(self, path: Path) -> None:
        """Write `{"counts": self.counts, "sums": self.sums}` to a `.pt`
        cache file, in the format expected by `load`."""
        torch.save({"counts": self.counts, "sums": self.sums}, path)

    def centroids(self, layer: str) -> Tensor:
        """`(vocab_size, hidden_size)`: the average hidden state of each
        vocabulary item at `layer`, i.e. `sums[layer] / counts` (see
        `relational.contextual_centroids`)."""
        return contextual_centroids(self, layer)
