#!/usr/bin/env python3
"""Split-half repeatability diagnostic for `token@time` (step 8 of the
revised priority order, `tmp/36-claude_token_time_signal_noise_measurement_proposal.md`).

This is a diagnostic, *not* a null: it splits d1's occurrences of each
target word `w` into two random halves by document (using
`OccurrenceCache.doc_index`), recomputes `R_1(w)` from each half with
`mu_1`/reference centroids held fixed at their full-d1 values (same
fixed-reference-system idea as nulo B, see `token_time_null.py`), and
compares `D_half(w) = 1 - cos(R_0(w), R_half(w))` between the two halves and
against the full-sample `D_obs(w)`.

If `token@time` measurements are dominated by a handful of documents (an
artifact), the two halves should disagree; if the signal is a corpus-wide
property of `w`, the two halves should agree with each other and with the
full-sample `D_obs`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from timeformers.token_time_repository import TokenTimeIndex  # noqa: E402


def split_half_displacements(
    idx: TokenTimeIndex, reference_ids: torch.Tensor, *, layer: str, generator: torch.Generator
) -> list[dict]:
    results = []
    for word in idx.targets:
        target_id = idx.target_ids[word]
        if idx.occurrences[1][target_id][layer].shape[0] < 4:
            continue
        try:
            d_half1, d_half2 = idx.split_half(word, reference_ids, layer=layer, generator=generator)
        except ValueError:
            continue
        disp_full = idx.displacement(word, reference_ids, layer=layer)
        results.append({"word": word, "D_obs": disp_full.score, "D_half1": d_half1, "D_half2": d_half2})
    return results


def main() -> None:
    layer = "layer_2"
    for profile_dir in ["outputs/token_time_fase_a/seed1000", "outputs/token_time_fase_a/seed1001"]:
        idx = TokenTimeIndex.load(Path(profile_dir))
        reference_ids = idx.reference_set()
        generator = torch.Generator().manual_seed(0)
        results = split_half_displacements(idx, reference_ids, layer=layer, generator=generator)

        for row in results:
            print(json.dumps(row))

        d_obs = [row["D_obs"] for row in results]
        d_half1 = [row["D_half1"] for row in results]
        d_half2 = [row["D_half2"] for row in results]
        d_half_mean = [(a + b) / 2 for a, b in zip(d_half1, d_half2)]

        rho_halves, p_halves = spearmanr(d_half1, d_half2)
        rho_full, p_full = spearmanr(d_obs, d_half_mean)
        print(f"\n{profile_dir}")
        print(f"  n_words = {len(results)}")
        print(f"  Spearman(D_half1, D_half2) = {rho_halves:.3f} (p={p_halves:.4f})")
        print(f"  Spearman(D_obs, mean(D_half1, D_half2)) = {rho_full:.3f} (p={p_full:.4f})")


if __name__ == "__main__":
    main()
