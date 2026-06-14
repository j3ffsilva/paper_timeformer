"""Fase B (docs/39): buscar deslocamentos de direção semelhante.

`nearest_displacements` implementa `nearest_trajectories(w, k, mode="direction")`
restrito ao único modo disponível com dois períodos.
"""

from __future__ import annotations

from .token_time import TokenTimeDisplacement
from .token_time_metrics import displacement_similarity


def nearest_displacements(
    word: str,
    displacements: dict[str, TokenTimeDisplacement],
    k: int,
) -> list[tuple[str, float]]:
    """Words whose `Delta` most resembles `Delta(word)` in direction, sorted
    descending by `displacement_similarity`. Excludes `word` itself."""
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
