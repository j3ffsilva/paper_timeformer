"""Comparing `TokenTimeDisplacement`s between *different* words.

`TokenTimeDisplacement` (see `token_time.py`) captures how a single word's
relational profile changed between two periods. The functions here compare
two such displacements -- for two different words -- to ask: "did `w` and
`u` change in the *same way*?".

With only two periods (before/after), a word's "trajectory" is just a single
displacement vector, so "similar trajectories" reduces to "similar
displacement directions": `cos(Delta(w), Delta(u))`.
"""

from __future__ import annotations

import torch.nn.functional as F

from .token_time import TokenTimeDisplacement


def _check_same_references(disp_a: TokenTimeDisplacement, disp_b: TokenTimeDisplacement) -> None:
    if disp_a.reference_vocab != disp_b.reference_vocab:
        raise ValueError("displacements must share the same reference_vocab (coordinate system)")


def displacement_similarity(disp_a: TokenTimeDisplacement, disp_b: TokenTimeDisplacement) -> float:
    """`cos(Delta(w), Delta(u))`: how similar the *direction* of change is
    between two words' displacements, over the same references.

    `1` means `w` and `u` shifted their relation to the references in
    exactly the same way (e.g. both became more similar to "queen" and less
    similar to "knight"); `0` means the two shifts are unrelated; `-1` means
    they shifted in exactly opposite ways.
    """
    _check_same_references(disp_a, disp_b)
    return float(F.cosine_similarity(disp_a.delta.unsqueeze(0), disp_b.delta.unsqueeze(0)).squeeze())


def displacement_contributions(
    disp_a: TokenTimeDisplacement, disp_b: TokenTimeDisplacement
) -> list[tuple[str, float]]:
    """Per-reference contribution to `displacement_similarity(disp_a, disp_b)`.

    `displacement_similarity` is the cosine similarity between the two
    (unit-normalized) delta vectors, which is just the sum of the
    elementwise products of their entries:
    `cos(a, b) = sum_v normalize(a)[v] * normalize(b)[v]`.

    This function returns that per-reference product, sorted descending, so
    the contributions sum (up to floating-point error) to
    `displacement_similarity(disp_a, disp_b)`. A large positive entry for
    reference `v` means: "both `w` and `u` changed their relation to `v` in
    the same direction and by a comparable (normalized) amount, so `v` is a
    big part of why `w` and `u` look similar". A large negative entry means
    `v` is pulling the two displacements *apart*.

    Trivial example: if `disp_a.delta` and `disp_b.delta` were both exactly
    `[1, 0]` after normalization, the contributions would be `[1*1, 0*0] =
    [1, 0]`, summing to `1` -- the displacement similarity for two identical
    unit vectors.
    """
    _check_same_references(disp_a, disp_b)
    normed_a = F.normalize(disp_a.delta, dim=0)
    normed_b = F.normalize(disp_b.delta, dim=0)
    contributions = normed_a * normed_b
    order = contributions.argsort(descending=True)
    return [(disp_a.reference_vocab[index], float(contributions[index])) for index in order.tolist()]
