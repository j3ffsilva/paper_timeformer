"""Relational profiles: comparing a word's meaning across time without a
shared absolute coordinate system.

The central idea is that we never compare two embedding vectors directly
(embeddings from different training runs or time periods do not live in the
same coordinate system). Instead, each word `w` is described by its
*relation* to a fixed set of reference words: a vector of cosine similarities
between `w` and each reference, after centering everything around a common
"average" point `mu`. Two such relational profiles -- one per period -- can
then be compared even though the underlying raw embeddings cannot.

This module provides:

- generic similarity-matrix helpers (`cosine_similarity_matrix`,
  `centered_cosine_similarity_matrix`, `normalized_euclidean_similarity_matrix`,
  `jensen_shannon_similarity_matrix`, `jensen_shannon_divergence_rows`,
  `topk_neighbors`, `relational_delta`);
- log-PMI based relational profiles (`log_pmi_profiles`,
  `pmi_cosine_displacement`, `ppmi_jsd_displacement`);
- the centroid-based relational profile used by the `token@time` framework
  (`contextual_centroids`, `occurrence_weighted_mean`, `type_uniform_mean`,
  `build_active_support`, `relational_profile`, `displacement`,
  `standardize`).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from .real_corpus import SPECIAL_TOKENS


def cosine_similarity_matrix(points: Tensor) -> Tensor:
    """All pairwise cosine similarities between rows of `points`.

    Each row is first scaled to unit length, so the result only depends on
    the *direction* of each vector, not its magnitude.

    Example: for `points = [[1, 0], [0, 1], [1, 1]]`, the result is::

        [[ 1.0   ,  0.0   ,  0.707 ],
         [ 0.0   ,  1.0   ,  0.707 ],
         [ 0.707 ,  0.707 ,  1.0   ]]

    i.e. row 0 and row 1 are orthogonal (similarity 0), while row 2 is at 45
    degrees from both (similarity cos(45 deg) = 0.707). The diagonal is
    always 1 (a vector is identical to itself).
    """
    normalized = F.normalize(points, dim=-1)
    return normalized @ normalized.T


def centered_cosine_similarity_matrix(points: Tensor) -> Tensor:
    """`cosine_similarity_matrix`, but first subtracting the mean of all
    rows from every row.

    Embedding spaces often have a large shared component common to every
    point (e.g. all word vectors pointing roughly in the same direction).
    Subtracting the mean removes that shared offset, so the similarities
    reflect how points differ *from each other* rather than how similar they
    all are to a common direction.
    """
    return cosine_similarity_matrix(points - points.mean(dim=0, keepdim=True))


def normalized_euclidean_similarity_matrix(points: Tensor) -> Tensor:
    """Pairwise Euclidean distances, rescaled to a `[0, 1]` similarity.

    `distances[i, j] = ||points[i] - points[j]||`. Dividing by the largest
    distance in the set and computing `1 - distance / max_distance` turns
    "0 = identical, large = far apart" into "1 = identical, 0 = as far apart
    as the two most distant points in this set". The scale is therefore
    relative to this particular set of points, not an absolute distance.
    """
    distances = torch.cdist(points, points)
    scale = distances.max().clamp_min(torch.finfo(distances.dtype).eps)
    return 1.0 - distances / scale


def jensen_shannon_similarity_matrix(distributions: Tensor) -> Tensor:
    """All pairwise similarities between rows of `distributions`, where each
    row is treated as a (possibly unnormalized) probability distribution.

    Each row is first clamped away from zero and renormalized to sum to 1.
    For every pair of rows `p`, `q`, the Jensen-Shannon divergence (JSD) is
    computed:

        JSD(p, q) = 0.5 * KL(p || m) + 0.5 * KL(q || m),  where m = (p+q)/2

    JSD is bounded in `[0, log(2)]`, with `0` meaning the two distributions
    are identical. The result is `1 - JSD / log(2)`, so `1` = identical
    distributions and `0` = maximally different (disjoint) distributions.
    """
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
    """Jensen-Shannon divergence between matching rows of `before` and
    `after` (not all pairs, unlike `jensen_shannon_similarity_matrix`).

    Both tensors must have the same shape `(n, vocab_size)`: row `i` of
    `before` and row `i` of `after` are two distributions for the *same*
    item (e.g. word `i`'s neighbor distribution at two time periods), and
    `result[i]` is how much that distribution changed, in `[0, 1]` after
    dividing by `log(2)` (0 = unchanged, 1 = maximally different).
    """
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
    """Indices of the `k` highest values in each row of a square similarity
    matrix, excluding the diagonal (an item's similarity to itself).

    Example: if row `i` of `similarities` is `[1.0, 0.9, 0.2, 0.95]` (the
    `1.0` at position `i` being the self-similarity) and `k=1`, the result
    for row `i` is `[3]` (the index of `0.95`), not `[0]`.
    """
    if similarities.ndim != 2 or similarities.size(0) != similarities.size(1):
        raise ValueError("similarities must be a square matrix")
    k = min(k, similarities.size(0) - 1)
    without_self = similarities.clone()
    without_self.fill_diagonal_(-torch.inf)
    return torch.topk(without_self, k=k, dim=1).indices


def relational_delta(similarities_before: Tensor, similarities_after: Tensor) -> Tensor:
    """Elementwise difference between two relational-profile matrices of the
    same shape: `after - before`. The simplest possible "what changed"
    signal -- positive entries grew more similar, negative entries grew less
    similar."""
    if similarities_before.shape != similarities_after.shape:
        raise ValueError("relational profiles must have matching shapes")
    return similarities_after - similarities_before


# ---------------------------------------------------------------------------
# Log-PMI relational profiles
# ---------------------------------------------------------------------------


def log_pmi_profiles(q: Tensor, p: Tensor, eps: float = 1e-9) -> Tensor:
    """Compute log-PMI relational profiles R_t(w) for a batch of words.

    Pointwise mutual information measures how much more (or less) likely a
    context word `v` is to co-occur with `w` than chance would predict:

        PMI(w, v) = log( P(v | w) / P(v) )

    - `PMI > 0`: `v` co-occurs with `w` more than expected (positive
      association).
    - `PMI = 0`: `v` is statistically independent of `w`.
    - `PMI < 0`: `v` co-occurs with `w` less than expected (repulsion).

    Args:
        q: (n_words, vocab_size) -- conditional distributions `q_t(w)`,
           averaged over real occurrences of each word.
        p: (vocab_size,) -- marginal distribution `p_t`, estimated from the
           neutral probe [CLS] [MASK] [SEP].

    Returns:
        (n_words, vocab_size) -- log-PMI profiles `R_t(w)[v] = log(q_t(w)[v]
        / p_t[v])`.
    """
    q_safe = q.clamp(min=eps)
    p_safe = p.clamp(min=eps)
    return torch.log(q_safe) - torch.log(p_safe).unsqueeze(0)


def pmi_cosine_displacement(R_t0: Tensor, R_t1: Tensor) -> Tensor:
    """Cosine displacement between log-PMI profiles, per word.

    Args:
        R_t0, R_t1: (n_words, vocab_size) -- log-PMI profiles at two
            checkpoints.

    Returns:
        (n_words,) -- displacement scores in `[0, 2]`. `0` means the two
        profiles point in exactly the same direction (no change in how `w`
        relates to its context); `2` means they point in exactly opposite
        directions.
    """
    return 1.0 - F.cosine_similarity(R_t0, R_t1, dim=1)


# ---------------------------------------------------------------------------
# Centroid-based relational profiles ("variant D" centering)
# ---------------------------------------------------------------------------
#
# The functions below implement the relational-profile construction used by
# the `token@time` framework (see `token_time.py`):
#
#   1. `contextual_centroids` turns raw per-occurrence hidden states into one
#      vector per vocabulary item (its average contextual representation).
#   2. `build_active_support` picks a stable subset of the vocabulary,
#      "V_active", that is frequent enough in *every* period to serve as a
#      shared yardstick.
#   3. `type_uniform_mean` computes `mu_t`, the "center" of the embedding
#      space at period `t`, as the unweighted average of V_active centroids
#      (so a handful of very frequent words don't dominate the center).
#   4. `relational_profile` expresses a target word `w` as a vector of
#      cosine similarities -- after centering on `mu_t` -- to a chosen set of
#      reference words. This vector is `P_t(w)`.
#   5. `displacement` compares `P_t0(w)` and `P_t1(w)`: if they point in the
#      same direction, `w`'s relation to the references hasn't changed.


def contextual_centroids(stats: dict, layer: str) -> Tensor:
    """centroid_t(w) = mean hidden state of `w` over all its occurrences in
    period `t`, at the given layer.

    `stats` is a dict-like object with:

    - `stats["counts"]`: `(vocab_size,)` tensor, number of occurrences of
      each vocabulary item;
    - `stats["sums"][layer]`: `(vocab_size, hidden_size)` tensor, the
      elementwise sum of the hidden states of each vocabulary item over all
      its occurrences.

    Dividing sums by counts gives the per-type average ("centroid"). Items
    with zero occurrences get a centroid of all zeros (the count is clamped
    to at least 1 to avoid division by zero).
    """
    counts = stats["counts"].float().unsqueeze(1).clamp_min(1.0)
    return stats["sums"][layer].float() / counts


def occurrence_weighted_mean(stats: dict, layer: str, *, support: Tensor | None = None) -> Tensor:
    """mu_t = mean hidden state over all individual *occurrences* in the
    period (not the mean of per-type centroids).

    Because frequent words contribute one term per occurrence, this mean is
    implicitly weighted by token frequency: a word that occurs 1000 times
    pulls the mean 1000 times as hard as a word that occurs once.

    If `support` (a boolean mask over the vocabulary) is given, only the
    occurrences of the selected vocabulary items are included.
    """
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
    """mu_t = mean of per-token-*type* centroids over `support`, each type
    weighted equally regardless of how often it occurs.

    Unlike `occurrence_weighted_mean`, a word that occurs 1000 times counts
    the same as a word that occurs 10 times: both contribute exactly one
    centroid to the average. This avoids letting a handful of ultra-frequent
    function words dominate `mu_t`.
    """
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
    """Boolean mask over the vocabulary selecting "V_active": vocabulary
    items that occur at least `n_min` times in *both* periods.

    V_active is the shared "yardstick" vocabulary used to compute `mu_t` and
    as the candidate set of reference words for relational profiles -- a
    word that is rare in one period would give an unreliable centroid there,
    so it is excluded from the comparison altogether.

    Special tokens (`[PAD]`, `[CLS]`, ...) and the target words themselves
    are always excluded: a target word's profile describes its relation to
    *other* words, not to itself.
    """
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
    """P_t(w)[v] = cos(centroid(w) - mu, centroid(v) - mu) for each `v` in
    `support_ids`.

    Both the target's centroid and every reference centroid are first
    re-centered by subtracting `mu` (the "center" of the embedding space for
    this period, e.g. from `type_uniform_mean`), then normalized to unit
    length. The result is one cosine similarity per reference: how `w`
    relates to each reference word, relative to the overall center of the
    space.

    Trivial example: with `mu = (0, 0)` (no centering effect), a target
    centroid of `(1, 0)` and three references at `(1, 0)`, `(0, 1)`,
    `(-1, 0)`, the profile is `[1.0, 0.0, -1.0]` -- identical direction,
    orthogonal, and opposite direction respectively.
    """
    centered = centroids - mu.unsqueeze(0)
    normed = F.normalize(centered, dim=1)
    target = normed[target_id : target_id + 1]
    references = normed[support_ids]
    return (target @ references.T).squeeze(0)


def displacement(profile_t0: Tensor, profile_t1: Tensor) -> float:
    """Delta(w) = 1 - cos(P_t0(w), P_t1(w)): how much `w`'s relational
    profile changed between the two periods.

    `0` means the two profiles point in exactly the same direction -- `w`'s
    relation to the reference words is unchanged. Larger values (up to `2`)
    mean the profile rotated towards a different set of references, i.e.
    `w`'s contextual meaning shifted relative to the rest of the vocabulary.
    """
    return float(1.0 - F.cosine_similarity(profile_t0.unsqueeze(0), profile_t1.unsqueeze(0)).squeeze())


def standardize(profile: Tensor) -> Tensor:
    """z-score a 1D profile: `(x - mean(x)) / std(x)`.

    Converts each entry from "raw cosine similarity" to "number of standard
    deviations above or below this profile's own average similarity". This
    makes entries from different profiles (which may have different overall
    scales) comparable: a reference with `z = 2` is unusually similar to `w`
    *relative to w's other references*, regardless of the absolute cosine
    value.
    """
    std = profile.std(unbiased=False).clamp_min(1e-9)
    return (profile - profile.mean()) / std


def ppmi_jsd_displacement(R_t0: Tensor, R_t1: Tensor, eps: float = 1e-9) -> Tensor:
    """Jensen-Shannon divergence between PPMI-induced distributions.

    Each log-PMI profile is first clamped to its positive part (PPMI:
    "positive PMI" -- only keep context words that co-occur with `w` *more*
    than chance, discard repulsion). The positive values are then
    renormalized into a probability distribution and compared with the
    Jensen-Shannon divergence, which is symmetric and bounded.

    Args:
        R_t0, R_t1: (n_words, vocab_size) -- log-PMI profiles at two
            checkpoints.

    Returns:
        (n_words,) -- displacement scores in `[0, log(2)]` ~= `[0, 0.693]`.
    """
    ppmi_t0 = R_t0.clamp(min=0.0)
    ppmi_t1 = R_t1.clamp(min=0.0)
    norm_t0 = ppmi_t0 / ppmi_t0.sum(dim=1, keepdim=True).clamp(min=eps)
    norm_t1 = ppmi_t1 / ppmi_t1.sum(dim=1, keepdim=True).clamp(min=eps)
    midpoint = 0.5 * (norm_t0 + norm_t1)
    kl_t0 = (norm_t0 * (norm_t0.clamp(min=eps).log() - midpoint.clamp(min=eps).log())).sum(dim=1)
    kl_t1 = (norm_t1 * (norm_t1.clamp(min=eps).log() - midpoint.clamp(min=eps).log())).sum(dim=1)
    return 0.5 * (kl_t0 + kl_t1)
