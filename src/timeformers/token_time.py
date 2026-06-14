"""`token@time` core objects (docs/39).

`TokenTimeProfile` wraps `relational_profile` (variant-D centering,
capítulo 08, Fase 1): `R_t(w)[v] = cos(centroid_t(w) - mu_t, centroid_t(v) -
mu_t)`. `TokenTimeDisplacement` wraps `displacement`:
`Delta(w, a, b) = R_b(w) - R_a(w)`, with `score = 1 - cos(R_a(w), R_b(w))` as
its scalar summary. `TokenTimeTrajectory` chains profiles across periods.

Both objects only carry the references they were built over -- comparing two
profiles requires the same `reference_vocab` (docs/39, "isolamento do sistema
de coordenadas").
"""

from __future__ import annotations

from dataclasses import dataclass, field

from torch import Tensor

from .relational import displacement, relational_profile, standardize


@dataclass
class TokenTimeProfile:
    """R_t(w)[v]: relational profile of `word` at `period`."""

    word: str
    period: str
    checkpoint: str
    layer: str
    reference_ids: Tensor
    reference_vocab: list[str]
    vector: Tensor
    count: int
    seed: int | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.vector.shape != self.reference_ids.shape:
            raise ValueError("vector and reference_ids must have the same shape")
        if len(self.reference_vocab) != self.reference_ids.numel():
            raise ValueError("reference_vocab must have one entry per reference_id")

    def as_dict(self) -> dict[str, float]:
        """{reference_token: R_t(w)[v]}."""
        return {token: float(value) for token, value in zip(self.reference_vocab, self.vector)}


def build_profile(
    centroids: Tensor,
    mu: Tensor,
    target_id: int,
    reference_ids: Tensor,
    vocab: list[str],
    *,
    word: str,
    period: str,
    checkpoint: str,
    layer: str,
    count: int,
    seed: int | None = None,
    metadata: dict | None = None,
) -> TokenTimeProfile:
    vector = relational_profile(centroids, mu, target_id, reference_ids)
    reference_vocab = [vocab[index] for index in reference_ids.tolist()]
    return TokenTimeProfile(
        word=word,
        period=period,
        checkpoint=checkpoint,
        layer=layer,
        reference_ids=reference_ids,
        reference_vocab=reference_vocab,
        vector=vector,
        count=count,
        seed=seed,
        metadata=metadata or {},
    )


@dataclass
class TokenTimeDisplacement:
    """Delta(w, a, b) = R_b(w) - R_a(w), with `score = 1 - cos(R_a(w), R_b(w))`."""

    word: str
    period_a: str
    period_b: str
    reference_vocab: list[str]
    profile_a: Tensor
    profile_b: Tensor
    delta: Tensor
    score: float
    metadata: dict = field(default_factory=dict)

    def standardized_delta(self) -> Tensor:
        """z_b(w)[v] - z_a(w)[v], standardizing each profile over its own
        references first -- comparable across references of differing scale
        (history/27, relatório de vizinhanças)."""
        return standardize(self.profile_b) - standardize(self.profile_a)

    def _ranked(self, *, descending: bool, k: int) -> list[tuple[str, float]]:
        delta_z = self.standardized_delta()
        order = delta_z.argsort(descending=descending)[:k]
        return [(self.reference_vocab[index], float(delta_z[index])) for index in order.tolist()]

    def top_gains(self, k: int = 20) -> list[tuple[str, float]]:
        """References with the largest increase in standardized similarity."""
        return self._ranked(descending=True, k=k)

    def top_losses(self, k: int = 20) -> list[tuple[str, float]]:
        """References with the largest decrease in standardized similarity."""
        return self._ranked(descending=False, k=k)


def compare_profiles(profile_a: TokenTimeProfile, profile_b: TokenTimeProfile) -> TokenTimeDisplacement:
    if profile_a.word != profile_b.word:
        raise ValueError("compare_profiles requires the same word at both periods")
    if profile_a.reference_vocab != profile_b.reference_vocab:
        raise ValueError("compare_profiles requires the same reference_vocab (coordinate system)")
    return TokenTimeDisplacement(
        word=profile_a.word,
        period_a=profile_a.period,
        period_b=profile_b.period,
        reference_vocab=profile_a.reference_vocab,
        profile_a=profile_a.vector,
        profile_b=profile_b.vector,
        delta=profile_b.vector - profile_a.vector,
        score=displacement(profile_a.vector, profile_b.vector),
    )


@dataclass
class TokenTimeTrajectory:
    """T(w) = [R_0(w), R_1(w), ..., R_n(w)].

    With two periods this is a single displacement, not a trajectory shape
    (docs/39): `displacements` has length `len(profiles) - 1`, and no
    shape/signature metrics are exposed here -- those require 3+ periods
    (Fase C) and live in `structural_metrics.py`.
    """

    word: str
    profiles: list[TokenTimeProfile]

    def __post_init__(self) -> None:
        if len(self.profiles) < 2:
            raise ValueError("a trajectory requires at least two profiles")
        if any(profile.word != self.word for profile in self.profiles):
            raise ValueError("all profiles in a trajectory must be for the same word")

    @property
    def periods(self) -> list[str]:
        return [profile.period for profile in self.profiles]

    @property
    def displacements(self) -> list[TokenTimeDisplacement]:
        return [
            compare_profiles(before, after)
            for before, after in zip(self.profiles, self.profiles[1:])
        ]
