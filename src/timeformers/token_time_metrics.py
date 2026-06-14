"""Fase B (docs/39): comparar deslocamentos `TokenTimeDisplacement`.

Com apenas dois períodos (D0/D1), "trajetória" é um único vetor de
deslocamento -- por isso esta fase compara `Delta(w)` entre palavras
("similaridade de direção", docs/39 operação 4), não formas de trajetória
(Fase C).
"""

from __future__ import annotations

import torch.nn.functional as F

from .token_time import TokenTimeDisplacement


def _check_same_references(disp_a: TokenTimeDisplacement, disp_b: TokenTimeDisplacement) -> None:
    if disp_a.reference_vocab != disp_b.reference_vocab:
        raise ValueError("displacements must share the same reference_vocab (coordinate system)")


def displacement_similarity(disp_a: TokenTimeDisplacement, disp_b: TokenTimeDisplacement) -> float:
    """cos(Delta(w), Delta(u)): direction similarity of two displacements
    over the same references (docs/39, "similaridade de direção")."""
    _check_same_references(disp_a, disp_b)
    return float(F.cosine_similarity(disp_a.delta.unsqueeze(0), disp_b.delta.unsqueeze(0)).squeeze())


def displacement_contributions(
    disp_a: TokenTimeDisplacement, disp_b: TokenTimeDisplacement
) -> list[tuple[str, float]]:
    """Per-reference contribution to `displacement_similarity(disp_a, disp_b)`,
    sorted descending. The contributions sum to that similarity, so the top
    entries are the references whose shared change drives the two
    displacements together; the bottom entries pull them apart (docs/39 Fase
    B item 3, "explicar cada resultado pelas referências que mais
    contribuíram")."""
    _check_same_references(disp_a, disp_b)
    normed_a = F.normalize(disp_a.delta, dim=0)
    normed_b = F.normalize(disp_b.delta, dim=0)
    contributions = normed_a * normed_b
    order = contributions.argsort(descending=True)
    return [(disp_a.reference_vocab[index], float(contributions[index])) for index in order.tolist()]
