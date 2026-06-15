"""Nulo B: document-level permutation null for `TokenTimeDisplacement.score`.

Given a target word `w` observed `n_a` times in period `a` and `n_b` times
in period `b`, `document_permutation_null` answers: "if `w`'s occurrences
had been split between the two periods by chance (in the same proportions,
`n_a`/`n_b`), how large would `score = 1 - cos(R_a(w), R_b(w))` typically be?"

Everything except `w`'s own centroid is held fixed at its *observed* value:
`mu_a`/`mu_b` (the period centers) and the reference words' centroids (used
to build `R_a(w)`/`R_b(w)`) come from the real `d0`/`d1` corpora, not from
the permuted split. Only `centroid(w)` is recomputed, from a resampled
subset of `w`'s pooled occurrences (see `OccurrenceCache`).

Resampling is done at the *document* level (`OccurrenceCache.doc_index`):
all of `w`'s occurrences in the same document move together between the two
pseudo-groups, preserving whatever within-document correlation real
occurrences have (repeated word-sense priming, topic, register, ...) that an
occurrence-level shuffle would destroy.

`D_obs` (the real `TokenTimeDisplacement(w).score`) should then be compared
against the distribution of `document_permutation_null(...)` -- e.g. via
`Z_robusto = (D_obs - median) / (1.4826 * MAD)` and the one-sided p-value
`p = (1 + #(D_null >= D_obs)) / (B + 1)` (see
`tmp/36-claude_token_time_signal_noise_measurement_proposal.md`).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from .relational import displacement


def _profile_for_target_centroid(
    target_centroid: Tensor,
    mu: Tensor,
    reference_centroids_normed: Tensor,
) -> Tensor:
    """`relational.relational_profile`, but with the target's centroid given
    directly instead of looked up in a `centroids` tensor by id.

    `reference_centroids_normed` is `F.normalize(centroids[support_ids] -
    mu, dim=1)`, precomputed once by the caller since it does not change
    across permutations.
    """
    target_centered = target_centroid - mu
    target_normed = F.normalize(target_centered.unsqueeze(0), dim=1)
    return (target_normed @ reference_centroids_normed.T).squeeze(0)


def document_permutation_null(
    occurrences_a: dict[str, Tensor],
    occurrences_b: dict[str, Tensor],
    *,
    centroids_a: Tensor,
    centroids_b: Tensor,
    mu_a: Tensor,
    mu_b: Tensor,
    reference_ids: Tensor,
    layer: str,
    n_permutations: int,
    generator: torch.Generator | None = None,
) -> Tensor:
    """`(n_permutations,)`: `D_null` samples for nulo B.

    `occurrences_a`/`occurrences_b` are `OccurrenceCache[target_id]` for the
    same target word in periods `a`/`b` (i.e. `{"layer_1": ..., "layer_2":
    ..., "doc_index": ...}`). `centroids_a`/`centroids_b` are
    `PeriodStatistics.centroids(layer)` for each period, and
    `mu_a`/`mu_b` are `relational.type_uniform_mean(...)` for each period --
    both used only to provide *reference* centroids and centers, held fixed.

    Raises `ValueError` if either period has zero occurrences of the target
    (no split is possible).
    """
    hidden_a = occurrences_a[layer]
    hidden_b = occurrences_b[layer]
    doc_a = occurrences_a["doc_index"]
    doc_b = occurrences_b["doc_index"]
    n_a = hidden_a.shape[0]
    n_b = hidden_b.shape[0]
    if n_a == 0 or n_b == 0:
        raise ValueError("document_permutation_null requires at least one occurrence in each period")

    hidden_pool = torch.cat([hidden_a, hidden_b], dim=0)
    # doc_index is local to each period's corpus file, so offset period b's
    # indices to keep the two periods' documents distinct in the pooled set.
    offset = int(doc_a.max().item()) + 1 if doc_a.numel() > 0 else 0
    block_ids_pool = torch.cat([doc_a, doc_b + offset])

    unique_blocks, inverse = torch.unique(block_ids_pool, return_inverse=True)
    n_blocks = unique_blocks.numel()
    block_sizes = torch.bincount(inverse, minlength=n_blocks)
    block_member_lists = [torch.nonzero(inverse == block).squeeze(1) for block in range(n_blocks)]

    references_a_normed = F.normalize(centroids_a[reference_ids] - mu_a.unsqueeze(0), dim=1)
    references_b_normed = F.normalize(centroids_b[reference_ids] - mu_b.unsqueeze(0), dim=1)

    d_null = torch.empty(n_permutations, dtype=torch.float32)
    for i in range(n_permutations):
        order = torch.randperm(n_blocks, generator=generator)
        group_a_indices: list[Tensor] = []
        group_b_indices: list[Tensor] = []
        size_a = 0
        for block in order.tolist():
            members = block_member_lists[block]
            size = int(block_sizes[block])
            if size_a + size <= n_a:
                group_a_indices.append(members)
                size_a += size
            else:
                group_b_indices.append(members)

        group_a = torch.cat(group_a_indices) if group_a_indices else torch.empty(0, dtype=torch.long)
        group_b = torch.cat(group_b_indices) if group_b_indices else torch.empty(0, dtype=torch.long)

        centroid_a = hidden_pool[group_a].mean(dim=0)
        centroid_b = hidden_pool[group_b].mean(dim=0)

        profile_a = _profile_for_target_centroid(centroid_a, mu_a, references_a_normed)
        profile_b = _profile_for_target_centroid(centroid_b, mu_b, references_b_normed)
        d_null[i] = displacement(profile_a, profile_b)

    return d_null
