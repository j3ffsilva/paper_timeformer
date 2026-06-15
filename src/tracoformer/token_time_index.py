"""Searching for words whose meaning shifted in a similar direction.

`nearest_displacements` implements a "nearest trajectories" query restricted
to the direction-similarity mode described in `token_time_metrics.py`: given
a word `w` and a precomputed displacement for every candidate word, find the
candidates whose displacement points most in the same direction as `w`'s.
"""

from __future__ import annotations

from .token_time import TokenTimeDisplacement
from .token_time_metrics import displacement_similarity


def nearest_displacements(
    word: str,
    displacements: dict[str, TokenTimeDisplacement],
    k: int,
) -> list[tuple[str, float]]:
    """Words whose displacement most resembles `displacements[word]` in
    direction, sorted descending by `displacement_similarity`.

    `displacements` maps every candidate word (including `word` itself) to
    its `TokenTimeDisplacement` over a shared set of references. The result
    is the top `k` *other* words, e.g. for `word="attack"` it might return
    `[("assault", 0.91), ("raid", 0.85), ...]`: words whose meaning shifted
    in a similar way to "attack"'s between the two periods.
    """
    if word not in displacements:
        raise KeyError(word)
    target = displacements[word]
    scored = [
        (other_word, displacement_similarity(target, disp))
        for other_word, disp in displacements.items()
        if other_word != word
    ]
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:k]
