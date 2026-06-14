"""`TokenTimeIndex`: facade over a `--profile-dir` written by
`scripts/build_token_time_profiles.py`.

Loads `vocab.json`, `targets.json`, `target_ids.json`, `metadata.json` and the
per-period caches (`PeriodStatistics`) once, then exposes the `token@time`
core operations (docs/39) as a fluent interface:

```python
idx = TokenTimeIndex.load("outputs/token_time_fase_a/seed1000", seed=1000)
refs = idx.reference_set()
idx.displacement("prop", refs).top_gains(10)
idx.nearest("attack", reference_ids=idx.active_support().nonzero().flatten(), k=5)
```

No new formulas: `profile`/`displacement` delegate to `build_profile`/
`compare_profiles` (`token_time.py`), `active_support`/`reference_set` to
`build_active_support`/`build_reference_set`, `nearest` to
`nearest_displacements` (`token_time_index.py`).
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
from .token_time_statistics import PeriodStatistics


def build_reference_set(
    vocab: list[str],
    active_support_mask: Tensor,
    *,
    targets: set[str],
    counts_d0: Tensor,
    counts_d1: Tensor,
    max_references: int,
) -> list[int]:
    """Whole-word, alphabetic, non-special tokens from V_ativo.

    A subset of V_ativo restricted to human-readable lexical references for
    neighborhood reports. V_ativo itself (used for `mu_t` and `displacement`)
    keeps WordPiece fragments -- only the *reported* references are filtered
    (history/27 §"detalhe antes de implementar").
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
        candidates.append(index)
    candidates.sort(key=lambda index: counts_min[index].item(), reverse=True)
    return candidates[:max_references]


@dataclass
class TokenTimeIndex:
    vocab: list[str]
    targets: list[str]
    target_ids: dict[str, int]
    checkpoint: str
    period_files: list[str]
    periods: list[PeriodStatistics]
    seed: int | None = None

    @classmethod
    def load(
        cls,
        profile_dir: Path,
        *,
        cache_paths: list[Path] | None = None,
        seed: int | None = None,
    ) -> "TokenTimeIndex":
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

        return cls(
            vocab=vocab,
            targets=targets,
            target_ids=target_ids,
            checkpoint=checkpoint,
            period_files=period_files,
            periods=periods,
            seed=seed,
        )

    def active_support(self, n_min: int = 10) -> Tensor:
        """V_ativo: tokens with count >= n_min in both periods (docs/39 §3)."""
        return build_active_support(
            self.periods[0], self.periods[1], vocab=self.vocab, targets=set(self.targets), n_min=n_min
        )

    def reference_set(self, max_references: int = 3216) -> Tensor:
        """Whole-word subset of V_ativo for human-readable neighbor tables."""
        active_mask = self.active_support()
        reference_ids = build_reference_set(
            self.vocab,
            active_mask,
            targets=set(self.targets),
            counts_d0=self.periods[0].counts,
            counts_d1=self.periods[1].counts,
            max_references=max_references,
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
        """R_t(w)[v] over `reference_ids`, variant-D centering (docs/39, capítulo 08 Fase 1)."""
        stats = self.periods[period_index]
        active_mask = self.active_support(n_min_active)
        centroids = stats.centroids(layer)
        mu = type_uniform_mean(stats, layer, support=active_mask)
        target_id = self.target_ids[word]
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
        )

    def displacement(
        self,
        word: str,
        reference_ids: Tensor,
        *,
        layer: str = "layer_2",
        n_min_active: int = 10,
    ) -> TokenTimeDisplacement:
        """Delta(w) = R_1(w) - R_0(w), with score = 1 - cos(R_0(w), R_1(w))."""
        profile_a = self.profile(word, 0, reference_ids, layer=layer, n_min_active=n_min_active)
        profile_b = self.profile(word, 1, reference_ids, layer=layer, n_min_active=n_min_active)
        return compare_profiles(profile_a, profile_b)

    def nearest(
        self,
        word: str,
        *,
        reference_ids: Tensor,
        k: int = 5,
        layer: str = "layer_2",
        n_min_active: int = 10,
    ) -> list[tuple[str, float]]:
        """Words whose `Delta` most resembles `Delta(word)` in direction (Fase B)."""
        displacements = {
            target: self.displacement(target, reference_ids, layer=layer, n_min_active=n_min_active)
            for target in self.targets
        }
        return nearest_displacements(word, displacements, k)
