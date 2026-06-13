"""Semantic mode decomposition via SVD of the cohesion matrix (§7-9 of
docs/12-novo_perfil_relacional.md).

The cohesion matrix M_t(w)[v, v'] = P_t(w)[v] * P_t(w)[v'] * cos(e(v), e(v'))
for v, v' in V_w = {v : P_t(w)[v] > tau} is positive semi-definite and equals
(D E)(D E)^T where D = diag(P_t(w)[V_w]) and E are the centralized,
normalized embeddings of V_w. Its eigendecomposition is obtained from the SVD
of D E without ever forming M (§7.5):

    D E = U Sigma W^T  =>  lambda_i = sigma_i^2,  a_i = u_i
"""
from __future__ import annotations

import torch
from torch import Tensor

from .gap_criterion import select_gap_index


def filter_support(profile: Tensor, gamma: float) -> tuple[Tensor, float | None]:
    """Select V_w = {v : P_t(w)[v] > tau} via the gap criterion on the
    positive, descending-sorted components of profile (§8.2).

    Returns (indices into `profile` of V_w, tau), or (empty, None) if no gap
    exceeds gamma.
    """
    positive_mask = profile > 0
    positive_values = profile[positive_mask]
    if positive_values.numel() < 2:
        return profile.new_zeros(0, dtype=torch.long), None
    order = torch.argsort(positive_values, descending=True)
    sorted_values = positive_values[order]
    i_star = select_gap_index(sorted_values, gamma)
    if i_star is None:
        return profile.new_zeros(0, dtype=torch.long), None
    tau = float(sorted_values[i_star - 1])
    positive_indices = torch.nonzero(positive_mask, as_tuple=False).flatten()
    selected = positive_indices[order[:i_star]]
    return selected, tau


def filter_support_topn(profile: Tensor, gamma: float, top_n: int) -> tuple[Tensor, float | None]:
    """Like filter_support, but first restricts the candidate set to the
    top_n components of `profile` by absolute value before applying the gap
    criterion to their positive, descending-sorted subset.

    Motivation: with small d_model, P_t(w) over the full V_active has no
    detectable gap (tau ~ 0, V_w ~ |V_active|), so M_t(w) degenerates to a
    single dominant mode. Restricting to the top_n most related candidates
    first gives the gap criterion a much smaller, less noisy set to work
    with (§8.2 still applies -- only the candidate pool changes).
    """
    if profile.numel() == 0:
        return profile.new_zeros(0, dtype=torch.long), None
    top_n = min(top_n, profile.numel())
    candidate_order = torch.argsort(profile.abs(), descending=True)[:top_n]
    candidate_profile = profile[candidate_order]
    selected_local, tau = filter_support(candidate_profile, gamma)
    if tau is None:
        return profile.new_zeros(0, dtype=torch.long), None
    return candidate_order[selected_local], tau


def cohesion_svd(profile_vw: Tensor, embeddings_vw: Tensor) -> tuple[Tensor, Tensor]:
    """SVD-based eigendecomposition of M_t(w) over V_w (§7.5).

    Args:
        profile_vw: (|V_w|,) -- P_t(w)[v] for v in V_w, all > 0.
        embeddings_vw: (|V_w|, d) -- centralized, L2-normalized embeddings
            of V_w (i.e. e_hat_t(v)).

    Returns:
        eigenvalues (descending, >= 0), eigenvectors (columns = a_i, sign
        convention: largest-magnitude component is positive, §7.5).
    """
    d_matrix = profile_vw.unsqueeze(1) * embeddings_vw
    u, sigma, _ = torch.linalg.svd(d_matrix, full_matrices=False)
    eigenvalues = sigma**2
    eigenvectors = u
    for i in range(eigenvectors.shape[1]):
        column = eigenvectors[:, i]
        argmax = torch.argmax(torch.abs(column))
        if column[argmax] < 0:
            eigenvectors[:, i] = -column
    return eigenvalues, eigenvectors


def select_num_modes(eigenvalues: Tensor, gamma: float) -> int | None:
    """k via the gap criterion on the eigenvalues (§8.3). None if the modes
    are not spectrally distinguishable (treated as monosemous)."""
    positive = eigenvalues[eigenvalues > 0]
    if positive.numel() < 2:
        return None
    return select_gap_index(positive, gamma)


def top_tokens_per_mode(
    eigenvectors: Tensor,
    vw_tokens: list[str],
    k: int,
    *,
    top_n: int = 10,
) -> list[list[tuple[str, float]]]:
    """For each of the first k modes, the top_n tokens of V_w by positive
    loading a_i[v] (§7.6)."""
    modes = []
    for i in range(k):
        loadings = eigenvectors[:, i]
        positive = loadings.clamp_min(0)
        order = torch.argsort(positive, descending=True)[:top_n]
        modes.append(
            [(vw_tokens[index], float(loadings[index])) for index in order if loadings[index] > 0]
        )
    return modes
