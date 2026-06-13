"""Gap criterion for automatic threshold/rank selection (§8 of
docs/12-novo_perfil_relacional.md).

Given any non-negative sequence sorted in descending order
X_1 >= X_2 >= ... > 0, the relative gap

    h_i = (X_i - X_{i+1}) / X_i

is invariant to positive rescaling of the sequence (X -> c*X for c > 0).
The selected index i* is the position of the largest gap, accepted only if
h_{i*} exceeds a validity threshold gamma. If no gap exceeds gamma, the
distribution has no clear structure at that resolution and None is returned.

This single function underlies three uses in the framework:
  - tau (and V_w) from the sorted positive components of P_t(w)            (§8.2)
  - k (number of semantic modes) from the sorted eigenvalues of M_t(w)     (§8.3)
  - eligibility for individual mode tracking, via adjacent eigengaps       (§8.4, §11.6)
"""
from __future__ import annotations

import torch
from torch import Tensor


def relative_gaps(values: Tensor) -> Tensor:
    """h_i = (X_i - X_{i+1}) / X_i for a descending, strictly positive sequence.

    Returns a tensor of length len(values) - 1. Invariant to positive
    rescaling: relative_gaps(c * values) == relative_gaps(values) for c > 0.
    """
    if values.ndim != 1:
        raise ValueError("values must be a 1D tensor")
    if values.numel() < 2:
        return values.new_zeros(0)
    if torch.any(values <= 0):
        raise ValueError("values must be strictly positive")
    if torch.any(values[:-1] < values[1:]):
        raise ValueError("values must be sorted in descending order")
    return (values[:-1] - values[1:]) / values[:-1]


def select_gap_index(values: Tensor, gamma: float) -> int | None:
    """Return i* = argmax(h) (1-indexed count of accepted elements), i.e. the
    number of elements in the "head" of the distribution before the largest
    gap, provided h_{i*} > gamma. Returns None if no gap exceeds gamma
    (no clear structure at this resolution -- §8.2/§8.3).

    Example: values = [0.41, 0.31, 0.18, 0.04, 0.03], gamma=0.3 ->
    gaps = [0.24, 0.42, 0.78, 0.25]; max gap at index 2 (0-indexed) -> i*=3
    (the first 3 elements form the head).
    """
    gaps = relative_gaps(values)
    if gaps.numel() == 0:
        return None
    best_index = int(torch.argmax(gaps))
    best_gap = float(gaps[best_index])
    if best_gap <= gamma:
        return None
    return best_index + 1


def adjacent_gaps_valid(values: Tensor, index: int, gamma: float) -> bool:
    """Whether element `index` (0-indexed) is separated from both neighbors
    by relative gaps > gamma -- the Davis-Kahan stability condition for
    individually tracking a single mode (§8.4, §11.6).

    The first and last elements only need their single adjacent gap to be
    valid.
    """
    gaps = relative_gaps(values)
    if gaps.numel() == 0:
        return True
    left_ok = index == 0 or float(gaps[index - 1]) > gamma
    right_ok = index == values.numel() - 1 or float(gaps[index]) > gamma
    return left_ok and right_ok
