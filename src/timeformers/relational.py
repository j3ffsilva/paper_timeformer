from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def cosine_similarity_matrix(points: Tensor) -> Tensor:
    normalized = F.normalize(points, dim=-1)
    return normalized @ normalized.T


def centered_cosine_similarity_matrix(points: Tensor) -> Tensor:
    return cosine_similarity_matrix(points - points.mean(dim=0, keepdim=True))


def normalized_euclidean_similarity_matrix(points: Tensor) -> Tensor:
    distances = torch.cdist(points, points)
    scale = distances.max().clamp_min(torch.finfo(distances.dtype).eps)
    return 1.0 - distances / scale


def jensen_shannon_similarity_matrix(distributions: Tensor) -> Tensor:
    eps = torch.finfo(distributions.dtype).eps
    probabilities = distributions.clamp_min(eps)
    probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
    p = probabilities[:, None, :]
    q = probabilities[None, :, :]
    midpoint = 0.5 * (p + q)
    divergence = 0.5 * (
        (p * (p.log() - midpoint.log())).sum(dim=-1)
        + (q * (q.log() - midpoint.log())).sum(dim=-1)
    )
    return 1.0 - divergence / torch.log(distributions.new_tensor(2.0))


def jensen_shannon_divergence_rows(before: Tensor, after: Tensor) -> Tensor:
    if before.shape != after.shape:
        raise ValueError("distributions must have matching shapes")
    eps = torch.finfo(before.dtype).eps
    p = before.clamp_min(eps)
    q = after.clamp_min(eps)
    p = p / p.sum(dim=-1, keepdim=True)
    q = q / q.sum(dim=-1, keepdim=True)
    midpoint = 0.5 * (p + q)
    divergence = 0.5 * (
        (p * (p.log() - midpoint.log())).sum(dim=-1)
        + (q * (q.log() - midpoint.log())).sum(dim=-1)
    )
    return divergence / torch.log(before.new_tensor(2.0))


def topk_neighbors(similarities: Tensor, k: int) -> Tensor:
    if similarities.ndim != 2 or similarities.size(0) != similarities.size(1):
        raise ValueError("similarities must be a square matrix")
    k = min(k, similarities.size(0) - 1)
    without_self = similarities.clone()
    without_self.fill_diagonal_(-torch.inf)
    return torch.topk(without_self, k=k, dim=1).indices


def relational_delta(similarities_before: Tensor, similarities_after: Tensor) -> Tensor:
    if similarities_before.shape != similarities_after.shape:
        raise ValueError("relational profiles must have matching shapes")
    return similarities_after - similarities_before
