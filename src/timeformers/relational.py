from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from .real_corpus import SPECIAL_TOKENS


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


# ---------------------------------------------------------------------------
# Log-PMI relational profiles
# ---------------------------------------------------------------------------


def log_pmi_profiles(q: Tensor, p: Tensor, eps: float = 1e-9) -> Tensor:
    """Compute log-PMI relational profiles R_t(w) for a batch of words.

    Args:
        q: (n_words, vocab_size) — conditional distributions q_t(w),
           averaged over real occurrences of each word.
        p: (vocab_size,) — marginal distribution p_t, estimated from the
           neutral probe [CLS] [MASK] [SEP].

    Returns:
        (n_words, vocab_size) — log-PMI profiles where
        R_t(w)[v] = log(q_t(w)[v] / p_t[v]).
        Positive values indicate specific association; near-zero indicates
        no selective relationship; negative indicates repulsion.
    """
    q_safe = q.clamp(min=eps)
    p_safe = p.clamp(min=eps)
    return torch.log(q_safe) - torch.log(p_safe).unsqueeze(0)


def pmi_cosine_displacement(R_t0: Tensor, R_t1: Tensor) -> Tensor:
    """Cosine displacement between log-PMI profiles.

    Args:
        R_t0, R_t1: (n_words, vocab_size) — log-PMI profiles at two checkpoints.

    Returns:
        (n_words,) — displacement scores in [0, 2].
        0 = identical relational profile; 2 = completely opposite.
    """
    return 1.0 - F.cosine_similarity(R_t0, R_t1, dim=1)


# ---------------------------------------------------------------------------
# Variant-D centering and relational profiles (capítulo 08, Fase 1)
# ---------------------------------------------------------------------------


def contextual_centroids(stats: dict, layer: str) -> Tensor:
    """centroid_t(w) = mean hidden state of `w` over its occurrences."""
    counts = stats["counts"].float().unsqueeze(1).clamp_min(1.0)
    return stats["sums"][layer].float() / counts


def occurrence_weighted_mean(stats: dict, layer: str, *, support: Tensor | None = None) -> Tensor:
    """mu_t = sum(sums) / sum(counts), i.e. the mean over all individual
    occurrences in the period (not the mean of per-token centroids)."""
    sums = stats["sums"][layer].float()
    counts = stats["counts"].float()
    if support is not None:
        mask = support.float().unsqueeze(1)
        sums = sums * mask
        counts = counts * support.float()
    total_sum = sums.sum(dim=0)
    total_count = counts.sum().clamp_min(1.0)
    return total_sum / total_count


def type_uniform_mean(stats: dict, layer: str, *, support: Tensor) -> Tensor:
    """mu_t = mean of per-token-type centroids over the support, each token
    type weighted equally regardless of its occurrence frequency. Avoids
    letting a handful of ultra-frequent function words dominate mu_t."""
    centroids = contextual_centroids(stats, layer)
    selected = centroids[support]
    return selected.mean(dim=0)


def build_active_support(
    stats_t0: dict,
    stats_t1: dict,
    *,
    vocab: list[str],
    targets: set[str],
    n_min: int,
) -> Tensor:
    """Boolean mask over vocab: tokens with count >= n_min in BOTH periods,
    excluding special tokens and target words (§3: V fixed; targets excluded
    from the support so the profile describes w's relation to *other* words)."""
    mask = (stats_t0["counts"] >= n_min) & (stats_t1["counts"] >= n_min)
    for index, token in enumerate(vocab):
        if token in SPECIAL_TOKENS or token in targets:
            mask[index] = False
    return mask


def relational_profile(
    centroids: Tensor,
    mu: Tensor,
    target_id: int,
    support_ids: Tensor,
) -> Tensor:
    """P_t(w)[v] = cos(centroid(w) - mu, centroid(v) - mu) for v in support."""
    centered = centroids - mu.unsqueeze(0)
    normed = F.normalize(centered, dim=1)
    target = normed[target_id : target_id + 1]
    references = normed[support_ids]
    return (target @ references.T).squeeze(0)


def displacement(profile_t0: Tensor, profile_t1: Tensor) -> float:
    """Delta(w) = 1 - cos(P_t0(w), P_t1(w))."""
    return float(1.0 - F.cosine_similarity(profile_t0.unsqueeze(0), profile_t1.unsqueeze(0)).squeeze())


def standardize(profile: Tensor) -> Tensor:
    std = profile.std(unbiased=False).clamp_min(1e-9)
    return (profile - profile.mean()) / std


def ppmi_jsd_displacement(R_t0: Tensor, R_t1: Tensor, eps: float = 1e-9) -> Tensor:
    """JSD between PPMI-induced distributions.

    Positive PMI mass is normalised to a probability distribution and
    compared with Jensen-Shannon divergence.

    Args:
        R_t0, R_t1: (n_words, vocab_size) — log-PMI profiles at two checkpoints.

    Returns:
        (n_words,) — displacement scores in [0, log(2)] ≈ [0, 0.693].
    """
    ppmi_t0 = R_t0.clamp(min=0.0)
    ppmi_t1 = R_t1.clamp(min=0.0)
    norm_t0 = ppmi_t0 / ppmi_t0.sum(dim=1, keepdim=True).clamp(min=eps)
    norm_t1 = ppmi_t1 / ppmi_t1.sum(dim=1, keepdim=True).clamp(min=eps)
    midpoint = 0.5 * (norm_t0 + norm_t1)
    kl_t0 = (norm_t0 * (norm_t0.clamp(min=eps).log() - midpoint.clamp(min=eps).log())).sum(dim=1)
    kl_t1 = (norm_t1 * (norm_t1.clamp(min=eps).log() - midpoint.clamp(min=eps).log())).sum(dim=1)
    return 0.5 * (kl_t0 + kl_t1)
