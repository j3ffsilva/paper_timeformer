"""`PeriodStatistics`: a typed wrapper for the per-period extraction caches
produced by `scripts/token_time/build_profiles.py`.

For a single period, the raw cache stores these tensors:

- `counts`: `(vocab_size,)`, how many times each vocabulary item occurred in
  this period's corpus;
- `sums`: `{"layer_1": (vocab_size, hidden_size), "layer_2": ..., ...}`, the
  elementwise sum of the hidden states of each vocabulary item, at each
  encoder layer, over all its occurrences;
- `sum_sq`: `{"layer_1": (vocab_size,), "layer_2": ..., ...}`, the sum of
  squared L2 norms (`||hidden||^2`) of the hidden states of each vocabulary
  item, at each encoder layer, over all its occurrences. Together with
  `sums` and `counts`, this is enough to recover each word's *dispersion*
  (`trace(Cov(hidden))`) without storing every individual occurrence vector.
- `standalone_counts`: `(vocab_size,)`, how many of those occurrences had
  vocabulary item `v` as the *entire* word (i.e. `v` does not start with
  `"##"`, and the following WordPiece in the same occurrence does not start
  with `"##"` either). A vocabulary item that almost always appears as part
  of a larger word (e.g. "graf" in "graf" + "##t" for "graft") will have a
  low `standalone_counts[v] / counts[v]` ratio even though it passes a naive
  `isalpha()` / no-`"##"` filter -- see `lexical_validity`.

`PeriodStatistics` is a typed dataclass around these fields. It also
implements `__getitem__("counts"|"sums")`, so the existing dict-keyed helpers
in `relational.py` (`stats["counts"]`, `stats["sums"][layer]`) work unchanged
on a `PeriodStatistics` instance.

`sum_sq` and `standalone_counts` are optional for backward compatibility with
caches written before these fields existed: `load` defaults them to `{}` /
an empty tensor, and `dispersion`/`standard_error`/`lexical_validity` raise
`KeyError`/`ValueError` accordingly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import Tensor

from .relational import contextual_centroids


@dataclass
class PeriodStatistics:
    counts: Tensor
    sums: dict[str, Tensor]
    sum_sq: dict[str, Tensor] = field(default_factory=dict)
    standalone_counts: Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.long))

    def __getitem__(self, key: str):
        if key == "counts":
            return self.counts
        if key == "sums":
            return self.sums
        if key == "sum_sq":
            return self.sum_sq
        if key == "standalone_counts":
            return self.standalone_counts
        raise KeyError(key)

    @classmethod
    def load(cls, path: Path) -> "PeriodStatistics":
        """Load `{"counts": Tensor, "sums": {...}, "sum_sq": {...},
        "standalone_counts": Tensor}` from a `.pt` cache file. `sum_sq`
        defaults to `{}` and `standalone_counts` to an empty tensor for
        caches written before these fields existed."""
        data = torch.load(path, map_location="cpu", weights_only=True)
        return cls(
            counts=data["counts"],
            sums=data["sums"],
            sum_sq=data.get("sum_sq", {}),
            standalone_counts=data.get("standalone_counts", torch.empty(0, dtype=torch.long)),
        )

    def save(self, path: Path) -> None:
        """Write `{"counts": ..., "sums": ..., "sum_sq": ..., "standalone_counts":
        ...}` to a `.pt` cache file, in the format expected by `load`."""
        torch.save(
            {
                "counts": self.counts,
                "sums": self.sums,
                "sum_sq": self.sum_sq,
                "standalone_counts": self.standalone_counts,
            },
            path,
        )

    def centroids(self, layer: str) -> Tensor:
        """`(vocab_size, hidden_size)`: the average hidden state of each
        vocabulary item at `layer`, i.e. `sums[layer] / counts` (see
        `relational.contextual_centroids`)."""
        return contextual_centroids(self, layer)

    def dispersion(self, layer: str) -> Tensor:
        """`(vocab_size,)`: `trace(Cov(hidden))` of each vocabulary item's
        occurrences at `layer`, i.e. `E[||x||^2] - ||E[x]||^2`.

        This is the average squared distance of an occurrence from its
        type's centroid -- a measure of how spread out a word's contexts
        are. Items with fewer than 2 occurrences get a dispersion of `0`
        (their centroid is computed from too little data to estimate
        spread, not because their usage is actually uniform).

        Raises `KeyError` if this cache has no `sum_sq` for `layer` (older
        caches written before this field existed).
        """
        counts = self.counts.float().clamp_min(1.0)
        mean_sq_norm = self.sum_sq[layer].float() / counts
        centroid_sq_norm = self.centroids(layer).pow(2).sum(dim=-1)
        spread = (mean_sq_norm - centroid_sq_norm).clamp_min(0.0)
        return torch.where(self.counts >= 2, spread, torch.zeros_like(spread))

    def standard_error(self, layer: str) -> Tensor:
        """`(vocab_size,)`: standard error of each vocabulary item's centroid
        at `layer`, i.e. `sqrt(dispersion(layer) / counts)`.

        This estimates how much the centroid would move if re-estimated from
        a different sample of the same size -- a direct, per-word measure of
        how much to trust a profile built from it (e.g. a rare word like
        "chairman" has a high standard error; a frequent word like "plane"
        has a low one). Items with fewer than 2 occurrences get `inf` (no
        spread can be estimated, so the centroid is maximally uncertain).

        Raises `KeyError` if this cache has no `sum_sq` for `layer`.
        """
        counts = self.counts.float()
        se = (self.dispersion(layer) / counts.clamp_min(1.0)).sqrt()
        return torch.where(self.counts >= 2, se, torch.full_like(se, float("inf")))

    def lexical_validity(self) -> Tensor:
        """`(vocab_size,)`: fraction of `v`'s occurrences where `v` is the
        entire word, i.e. `standalone_counts / counts`.

        Language-agnostic alternative to filtering reference candidates by
        an external wordlist: a WordPiece vocabulary item that almost always
        gets continued by `"##..."` pieces (e.g. "graf" in "graf" + "##t")
        is a sub-word fragment, not an interpretable lexical item, regardless
        of which language/corpus the encoder was trained on. Items with zero
        occurrences get a validity of `0`.

        Raises `ValueError` if this cache has no `standalone_counts` (older
        caches written before this field existed).
        """
        if self.standalone_counts.numel() == 0:
            raise ValueError("this cache has no standalone_counts (older cache format)")
        counts = self.counts.float().clamp_min(1.0)
        validity = self.standalone_counts.float() / counts
        return torch.where(self.counts >= 1, validity, torch.zeros_like(validity))
