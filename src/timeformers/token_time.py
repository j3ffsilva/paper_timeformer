"""Core `token@time` objects: relational profiles, displacements and
trajectories of a word across time periods.

These classes wrap the centroid-based relational profile from
`relational.py` (`relational_profile`, `displacement`, `standardize`) in
typed, self-describing objects:

- `TokenTimeProfile`: `R_t(w)[v]`, the relational profile of word `w` at
  period `t`, i.e. its cosine similarity to each reference word `v` after
  centering on `mu_t`.
- `TokenTimeDisplacement`: `Delta(w, a, b) = R_b(w) - R_a(w)`, the
  per-reference change in `w`'s profile between periods `a` and `b`, plus a
  single scalar `score = 1 - cos(R_a(w), R_b(w))` summarizing how much the
  *direction* of the profile changed overall.
- `TokenTimeTrajectory`: a sequence of profiles for the same word across
  more than two periods.

Comparing two profiles only makes sense if they were built over the exact
same set of reference words in the same order -- otherwise position `i` in
one profile and position `i` in the other would refer to different
references. This is enforced by checking `reference_vocab` equality before
any comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from torch import Tensor

from .relational import displacement, relational_profile, standardize


@dataclass
class TokenTimeProfile:
    """R_t(w)[v]: relational profile of `word` at `period`.

    `vector[i]` is the cosine similarity between `word` and
    `reference_vocab[i]` (both centered on `mu_t`), so `vector` and
    `reference_vocab`/`reference_ids` always have matching lengths and
    ordering.
    """

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
        """{reference_token: R_t(w)[v]}, e.g. `{"king": 0.42, "queen": 0.38, ...}`."""
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
    """Build a `TokenTimeProfile` for `word` from precomputed `centroids`
    (`relational.contextual_centroids`) and center `mu`
    (`relational.type_uniform_mean`).

    `target_id` and `reference_ids` are vocabulary indices: `target_id`
    identifies `word` in `centroids`, and `reference_ids` selects which
    vocabulary items to use as references (their tokens, in the same order,
    become `reference_vocab`).
    """
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
    """Delta(w, a, b) = R_b(w) - R_a(w), with `score = 1 - cos(R_a(w), R_b(w))`.

    `delta[i]` is how much `w`'s similarity to `reference_vocab[i]` changed
    between period `a` and period `b` (positive = `w` became more similar to
    that reference, negative = less similar). `score` is a single number
    summarizing the overall change in direction: `0` = no change, up to `2`
    = the profile pointed in the exact opposite direction afterwards.
    """

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
        """`z_b(w)[v] - z_a(w)[v]`, where each profile is z-scored
        (`relational.standardize`) against its *own* references first.

        Raw cosine similarities from `profile_a` and `profile_b` can have
        different overall scales (e.g. if `mu` shifted between periods), so
        subtracting them directly can be misleading. Standardizing first
        expresses each entry as "how unusual is this reference's similarity,
        relative to this profile's own average and spread" -- making the two
        periods comparable on the same scale.
        """
        return standardize(self.profile_b) - standardize(self.profile_a)

    def _ranked(self, *, descending: bool, k: int) -> list[tuple[str, float]]:
        delta_z = self.standardized_delta()
        order = delta_z.argsort(descending=descending)[:k]
        return [(self.reference_vocab[index], float(delta_z[index])) for index in order.tolist()]

    def top_gains(self, k: int = 20) -> list[tuple[str, float]]:
        """The `k` references whose standardized similarity to `w` increased
        the most from period `a` to period `b`, e.g.
        `[("queen", 1.8), ("crown", 1.5), ...]`."""
        return self._ranked(descending=True, k=k)

    def top_losses(self, k: int = 20) -> list[tuple[str, float]]:
        """The `k` references whose standardized similarity to `w` decreased
        the most from period `a` to period `b` (the mirror of
        `top_gains`)."""
        return self._ranked(descending=False, k=k)


def compare_profiles(profile_a: TokenTimeProfile, profile_b: TokenTimeProfile) -> TokenTimeDisplacement:
    """Build a `TokenTimeDisplacement` from two profiles of the same word
    over the same set of references.

    Raises `ValueError` if the profiles describe different words, or were
    built over different (or differently ordered) reference vocabularies --
    in either case, comparing the underlying vectors entry-by-entry would be
    meaningless.
    """
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
    """T(w) = [R_0(w), R_1(w), ..., R_n(w)]: the relational profile of `w`
    across more than two periods, in chronological order.

    `displacements` pairs up consecutive profiles, so a trajectory of `n+1`
    profiles yields `n` displacements: `Delta(w, 0, 1), Delta(w, 1, 2), ...,
    Delta(w, n-1, n)`. With only two periods, a trajectory degenerates to a
    single displacement -- there is no separate notion of trajectory "shape"
    to analyze in that case.
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
        """The period labels in order, e.g. `["1950", "1960", "1970"]`."""
        return [profile.period for profile in self.profiles]

    @property
    def displacements(self) -> list[TokenTimeDisplacement]:
        """One `TokenTimeDisplacement` per consecutive pair of periods."""
        return [
            compare_profiles(before, after)
            for before, after in zip(self.profiles, self.profiles[1:])
        ]
