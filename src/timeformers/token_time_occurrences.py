"""`OccurrenceCache`: per-occurrence hidden states for target words, for a
single period.

`PeriodStatistics` only stores aggregates (`counts`, `sums`, `sum_sq`) over
*all* vocabulary items, which is enough to build relational profiles but not
enough to ask "what would `D(w)` look like under a different split of `w`'s
occurrences between periods?" -- the question behind the permutation null
(nulo B) and the split-half repeatability diagnostic (nulo A).

Answering that requires the individual hidden-state vector of every
occurrence of every *target* word (there is no need to do this for the whole
vocabulary: `mu_t` and the reference centroids are held fixed under both
nulls, only the target's own centroid is recomputed from a resampled subset
of its occurrences).

For each target word (real vocabulary id, or a "virtual" id for
multi-subtoken targets -- see `scripts/token_time/build_profiles.py`), this
module stores:

- `layers[layer]`: `(n_occurrences, hidden_size)`, the hidden state of each
  occurrence (mean-pooled over subtokens, for multi-subtoken targets);
- `doc_index`: `(n_occurrences,)`, which document (by index into that
  period's corpus file) each occurrence came from.

`doc_index` makes document-level resampling possible: nulo B permutes whole
documents' occurrences between the pseudo-d0/d1 labels (preserving the
correlation between occurrences of the same word within one document),
rather than permuting individual occurrences independently.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor


@dataclass
class OccurrenceCache:
    """Per-occurrence hidden states and document indices for a set of target
    vocabulary ids, for one period.

    `targets[target_id]` is `{"layer_1": Tensor(n, hidden), "layer_2": ...,
    "doc_index": Tensor(n,)}`. `n` varies per target id (its occurrence
    count in this period).
    """

    targets: dict[int, dict[str, Tensor]]

    def __getitem__(self, target_id: int) -> dict[str, Tensor]:
        return self.targets[target_id]

    def __contains__(self, target_id: int) -> bool:
        return target_id in self.targets

    @classmethod
    def load(cls, path: Path) -> "OccurrenceCache":
        """Load `{target_id: {"layer_1": Tensor, ..., "doc_index": Tensor}}`
        from a `.pt` cache file."""
        data = torch.load(path, map_location="cpu", weights_only=True)
        return cls(targets={int(target_id): tensors for target_id, tensors in data.items()})

    def save(self, path: Path) -> None:
        """Write `{target_id: {"layer_1": Tensor, ..., "doc_index": Tensor}}`
        to a `.pt` cache file, in the format expected by `load`."""
        torch.save(self.targets, path)

    def centroid(self, target_id: int, layer: str, *, occurrence_mask: Tensor | None = None) -> Tensor:
        """`(hidden_size,)`: mean hidden state of `target_id`'s occurrences
        at `layer`, optionally restricted to `occurrence_mask` (a boolean
        tensor of shape `(n,)`).

        This is the building block for the permutation null: recompute this
        centroid for a resampled subset of occurrences, then plug it into
        `relational.relational_profile` in place of
        `PeriodStatistics.centroids(layer)[target_id]`, holding `mu` and the
        reference centroids fixed.
        """
        hidden = self.targets[target_id][layer]
        if occurrence_mask is not None:
            hidden = hidden[occurrence_mask]
        return hidden.mean(dim=0)
