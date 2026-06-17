"""`TokenTimeIndex`: a facade over a "profile directory" produced by the
`token@time` extraction pipeline (see `scripts/build_token_time_profiles.py`).

A profile directory contains:

- `vocab.json`: the vocabulary, as a list of tokens (vocabulary index ->
  token string);
- `targets.json`: the list of target words being analyzed;
- `target_ids.json`: `{target_word: vocabulary_index}`;
- `metadata.json` (optional): `{"checkpoint": ..., "period_files": [...]}`;
- `cache/theta_d0.pt`, `cache/theta_d1.pt`, ...: one `PeriodStatistics` cache
  per period (see `token_time_statistics.py`).

`TokenTimeIndex.load` reads all of this once, and the resulting object
exposes the `token@time` operations as a fluent interface:

```python
idx = TokenTimeIndex.load("outputs/seed1000", seed=1000)
refs = idx.reference_set()
idx.displacement("prop", refs).top_gains(10)
idx.nearest("attack", reference_ids=idx.active_support().nonzero().flatten(), k=5)
```

This module introduces no new formulas: `profile`/`displacement` delegate to
`build_profile`/`compare_profiles` (`token_time.py`), `active_support` to
`build_active_support` (`relational.py`), `reference_set` to
`build_reference_set` (below), and `nearest` to `nearest_displacements`
(`token_time_index.py`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from .real_corpus import SPECIAL_TOKENS
from .relational import build_active_support, type_uniform_mean
from .token_time import TokenTimeDisplacement, TokenTimeProfile, build_profile, compare_profiles
from .token_time_index import nearest_displacements
from .token_time_null import document_permutation_null, split_half_displacement
from .token_time_occurrences import OccurrenceCache
from .token_time_statistics import PeriodStatistics


def build_reference_set(
    vocab: list[str],
    active_support_mask: Tensor,
    *,
    targets: set[str],
    counts_d0: Tensor,
    counts_d1: Tensor,
    max_references: int,
    lexical_validity_d0: Tensor | None = None,
    lexical_validity_d1: Tensor | None = None,
    min_lexical_validity: float = 0.0,
) -> list[int]:
    """Whole-word, alphabetic, non-special tokens from V_active (see
    `relational.build_active_support`).

    V_active itself (used for `mu_t` and `displacement`) may include
    WordPiece fragments such as `"##ing"`, which are not meaningful entries
    in a human-readable neighbor table. This function returns a filtered
    subset -- whole alphabetic words, excluding special tokens and the
    target words themselves -- ordered by how frequent each candidate is in
    its *less frequent* period (so the reported references are reliably
    well-estimated in both periods), and capped at `max_references`.

    If `lexical_validity_d0`/`lexical_validity_d1` are given (see
    `PeriodStatistics.lexical_validity`), candidates whose validity in
    either period is below `min_lexical_validity` are also excluded. This
    catches WordPiece *fragments* that pass the `isalpha()`/no-`"##"` check
    but rarely occur as a whole word on their own (e.g. "graf" in
    "graf" + "##t" for "graft"), without depending on an external wordlist.
    """
    counts_min = torch.minimum(counts_d0, counts_d1).float()
    candidates = []
    for index, token in enumerate(vocab):
        if not active_support_mask[index]:
            continue
        if token in SPECIAL_TOKENS or token in targets:
            continue
        if token.startswith("##"):
            continue
        if not token.isalpha():
            continue
        if min_lexical_validity > 0.0:
            if lexical_validity_d0 is not None and lexical_validity_d0[index] < min_lexical_validity:
                continue
            if lexical_validity_d1 is not None and lexical_validity_d1[index] < min_lexical_validity:
                continue
        candidates.append(index)
    candidates.sort(key=lambda index: counts_min[index].item(), reverse=True)
    return candidates[:max_references]


@dataclass
class TokenTimeIndex:
    """In-memory view of a profile directory: vocabulary, targets, and one
    `PeriodStatistics` per period.

    `periods[0]` and `periods[1]` correspond to `period_files[0]` and
    `period_files[1]` respectively (e.g. `["d0", "d1"]` for a two-period
    before/after comparison).
    """

    vocab: list[str]
    targets: list[str]
    target_ids: dict[str, int]
    checkpoint: str
    period_files: list[str]
    periods: list[PeriodStatistics]
    occurrences: list[OccurrenceCache] | None = None
    seed: int | None = None

    @classmethod
    def load(
        cls,
        profile_dir: Path,
        *,
        cache_paths: list[Path] | None = None,
        seed: int | None = None,
    ) -> "TokenTimeIndex":
        """Load a profile directory written by `build_token_time_profiles.py`.

        If `cache_paths` is not given, defaults to
        `<profile_dir>/cache/theta_<period_file>.pt` for each entry in
        `metadata.json`'s `period_files` (or `["d0", "d1"]` if there is no
        `metadata.json`).

        Also loads `<profile_dir>/cache/occurrences_<period_file>.pt` into
        `occurrences`, if present (older profile directories, written before
        `OccurrenceCache` existed, have none -- `occurrences` stays `None`
        and `null_b` is unavailable).
        """
        profile_dir = Path(profile_dir)
        vocab = json.loads((profile_dir / "vocab.json").read_text(encoding="utf-8"))
        targets = json.loads((profile_dir / "targets.json").read_text(encoding="utf-8"))
        target_ids = json.loads((profile_dir / "target_ids.json").read_text(encoding="utf-8"))
        metadata_path = profile_dir / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        checkpoint = metadata.get("checkpoint", "")
        period_files = metadata.get("period_files", ["d0", "d1"])

        if cache_paths is None:
            cache_paths = [profile_dir / "cache" / f"theta_{name}.pt" for name in period_files]
        periods = [PeriodStatistics.load(path) for path in cache_paths]

        occurrences: list[OccurrenceCache] | None = []
        for name in period_files:
            occurrence_path = profile_dir / "cache" / f"occurrences_{name}.pt"
            if not occurrence_path.exists():
                occurrences = None
                break
            occurrences.append(OccurrenceCache.load(occurrence_path))

        return cls(
            vocab=vocab,
            targets=targets,
            target_ids=target_ids,
            checkpoint=checkpoint,
            period_files=period_files,
            periods=periods,
            occurrences=occurrences,
            seed=seed,
        )

    def active_support(self, n_min: int = 10) -> Tensor:
        """V_active: boolean mask over the vocabulary, tokens occurring at
        least `n_min` times in both periods (see
        `relational.build_active_support`)."""
        return build_active_support(
            self.periods[0], self.periods[1], vocab=self.vocab, targets=set(self.targets), n_min=n_min
        )

    def reference_set(self, max_references: int = 3216, *, min_lexical_validity: float = 0.5) -> Tensor:
        """Whole-word subset of V_active, as vocabulary-index tensor, for
        human-readable neighbor tables (see `build_reference_set`).

        `min_lexical_validity` excludes WordPiece fragments that pass the
        no-`"##"`/`isalpha()` check but rarely occur as a whole word on
        their own (e.g. "graf" in "graf" + "##t", `lexical_validity < 0.1`
        in practice) -- see `PeriodStatistics.lexical_validity`.
        """
        active_mask = self.active_support()
        reference_ids = build_reference_set(
            self.vocab,
            active_mask,
            targets=set(self.targets),
            counts_d0=self.periods[0].counts,
            counts_d1=self.periods[1].counts,
            max_references=max_references,
            lexical_validity_d0=self.periods[0].lexical_validity(),
            lexical_validity_d1=self.periods[1].lexical_validity(),
            min_lexical_validity=min_lexical_validity,
        )
        return torch.tensor(reference_ids, dtype=torch.long)

    def profile(
        self,
        word: str,
        period_index: int,
        reference_ids: Tensor,
        *,
        layer: str = "layer_2",
        n_min_active: int = 10,
    ) -> TokenTimeProfile:
        """`r_t(w)[v]` for `word` at `self.periods[period_index]`, over
        `reference_ids` (see `relational.relational_profile`).

        Internally computes `centroids` (`PeriodStatistics.centroids`) and
        `mu` (`relational.type_uniform_mean` over `active_support`) for the
        requested period, then calls `token_time.build_profile`.
        """
        stats = self.periods[period_index]
        active_mask = self.active_support(n_min_active)
        centroids = stats.centroids(layer)
        mu = type_uniform_mean(stats, layer, support=active_mask)
        target_id = self.target_ids[word]
        standard_error = None
        if stats.sum_sq:
            standard_error = float(stats.standard_error(layer)[target_id])
        return build_profile(
            centroids,
            mu,
            target_id,
            reference_ids,
            self.vocab,
            word=word,
            period=self.period_files[period_index],
            checkpoint=self.checkpoint,
            layer=layer,
            count=int(stats.counts[target_id]),
            seed=self.seed,
            standard_error=standard_error,
        )

    def displacement(
        self,
        word: str,
        reference_ids: Tensor,
        *,
        layer: str = "layer_2",
        n_min_active: int = 10,
    ) -> TokenTimeDisplacement:
        """`Delta(w) = r_1(w) - r_0(w)`, with `score = δ(w) = 1 - cos(r_0(w), r_1(w))`
        (see `token_time.compare_profiles`). Builds `profile(word, 0, ...)`
        and `profile(word, 1, ...)` and compares them."""
        profile_a = self.profile(word, 0, reference_ids, layer=layer, n_min_active=n_min_active)
        profile_b = self.profile(word, 1, reference_ids, layer=layer, n_min_active=n_min_active)
        return compare_profiles(profile_a, profile_b)

    def null_b(
        self,
        word: str,
        reference_ids: Tensor,
        *,
        layer: str = "layer_2",
        n_min_active: int = 10,
        n_permutations: int = 200,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """`(n_permutations,)`: `{δ_b(w)}` samples for `word` under nulo B
        (`token_time_null.document_permutation_null`).

        Compare against `displacement(word, reference_ids, ...).score` (`δ(w)`)
        -- e.g. `ζ(w) = (δ(w) - median) / (1.4826 * MAD)` and the one-sided
        p-value `(1 + sum({δ_b(w)} >= δ(w))) / (n_permutations + 1)`.

        Raises `ValueError` if this index has no `occurrences` (profile
        directory written before `OccurrenceCache` existed) or if `word` has
        zero occurrences in either period.
        """
        if self.occurrences is None:
            raise ValueError("this index has no occurrence caches (older profile directory)")
        target_id = self.target_ids[word]
        active_mask = self.active_support(n_min_active)
        centroids_a = self.periods[0].centroids(layer)
        centroids_b = self.periods[1].centroids(layer)
        mu_a = type_uniform_mean(self.periods[0], layer, support=active_mask)
        mu_b = type_uniform_mean(self.periods[1], layer, support=active_mask)
        return document_permutation_null(
            self.occurrences[0][target_id],
            self.occurrences[1][target_id],
            centroids_a=centroids_a,
            centroids_b=centroids_b,
            mu_a=mu_a,
            mu_b=mu_b,
            reference_ids=reference_ids,
            layer=layer,
            n_permutations=n_permutations,
            generator=generator,
        )

    def split_half(
        self,
        word: str,
        reference_ids: Tensor,
        *,
        period_index: int = 1,
        layer: str = "layer_2",
        n_min_active: int = 10,
        generator: torch.Generator | None = None,
    ) -> tuple[float, float]:
        """`(D_half1, D_half2)`: split-half repeatability diagnostic
        (`token_time_null.split_half_displacement`) for `word` at
        `self.periods[period_index]`, against the *other* period's profile.

        Not a null -- compare against `displacement(word, ...).score` to see
        whether `δ(w)` is stable across two independent random halves of
        `period_index`'s documents.

        Raises `ValueError` if this index has no `occurrences` or `word` has
        fewer than 2 distinct documents in `period_index`.
        """
        if self.occurrences is None:
            raise ValueError("this index has no occurrence caches (older profile directory)")
        target_id = self.target_ids[word]
        other_index = 1 - period_index
        active_mask = self.active_support(n_min_active)
        centroids = self.periods[period_index].centroids(layer)
        mu = type_uniform_mean(self.periods[period_index], layer, support=active_mask)
        reference_profile = self.profile(word, other_index, reference_ids, layer=layer, n_min_active=n_min_active).vector
        return split_half_displacement(
            self.occurrences[period_index][target_id],
            reference_profile,
            centroids=centroids,
            mu=mu,
            reference_ids=reference_ids,
            layer=layer,
            generator=generator,
        )

    def nearest(
        self,
        word: str,
        *,
        reference_ids: Tensor,
        k: int = 5,
        layer: str = "layer_2",
        n_min_active: int = 10,
    ) -> list[tuple[str, float]]:
        """The `k` target words whose displacement most resembles `word`'s
        in direction (see `token_time_index.nearest_displacements`).

        Computes `displacement(...)` for every word in `self.targets` (this
        is recomputed on every call, not cached)."""
        displacements = {
            target: self.displacement(target, reference_ids, layer=layer, n_min_active=n_min_active)
            for target in self.targets
        }
        return nearest_displacements(word, displacements, k)
