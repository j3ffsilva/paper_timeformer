"""Shared relational-profile and occurrence-APD helpers over hidden-state caches."""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor


def relational_profiles(
    points: Tensor,
    target_ids: list[int],
    reference_ids: list[int],
    *,
    center: bool = False,
) -> Tensor:
    points = points.float()
    if center:
        points = points - points[reference_ids].mean(dim=0, keepdim=True)
    targets = F.normalize(points[target_ids], dim=1)
    references = F.normalize(points[reference_ids], dim=1)
    return targets @ references.T


def occurrence_profiles(
    stats: dict,
    layer: str,
    target_index: int,
    reference_centroids: Tensor,
    reference_ids: list[int],
    *,
    center: bool = False,
) -> Tensor:
    selected = stats["occurrence_targets"] == target_index
    occurrences = stats["occurrence_vectors"][layer][selected].float()
    references = reference_centroids[reference_ids].float()
    if center:
        mean = references.mean(dim=0, keepdim=True)
        occurrences = occurrences - mean
        references = references - mean
    occurrences = F.normalize(occurrences, dim=1)
    references = F.normalize(references, dim=1)
    return F.normalize(occurrences @ references.T, dim=1)


def average_pairwise_cosine_distance(before: Tensor, after: Tensor) -> float:
    return float(1.0 - (before @ after.T).mean())


def occurrences_for_target(stats: dict, layer: str, target_index: int) -> Tensor:
    mask = stats["occurrence_targets"] == target_index
    return stats["occurrence_vectors"][layer][mask].float()


def average_pairwise_distance(vectors_t0: Tensor, vectors_t1: Tensor) -> float:
    """APD(w) = mean over (i in t0, j in t1) of 1 - cos(v_i, v_j)."""
    a = F.normalize(vectors_t0, dim=1)
    b = F.normalize(vectors_t1, dim=1)
    cosines = a @ b.T
    return float((1.0 - cosines).mean())
