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
import torch.nn.functional as F
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from timeformers.relational import displacement, type_uniform_mean  # noqa: E402
from timeformers.token_time_null import _profile_for_target_centroid  # noqa: E402
from timeformers.token_time_repository import TokenTimeIndex  # noqa: E402


def split_half_displacements(
    idx: TokenTimeIndex, reference_ids: torch.Tensor, *, layer: str, generator: torch.Generator
) -> list[dict]:
    active_mask = idx.active_support()
    centroids_b = idx.periods[1].centroids(layer)
    mu_b = type_uniform_mean(idx.periods[1], layer, support=active_mask)
    references_b_normed = F.normalize(centroids_b[reference_ids] - mu_b.unsqueeze(0), dim=1)

    results = []
    for word in idx.targets:
        target_id = idx.target_ids[word]
        occ_b = idx.occurrences[1][target_id]
        doc_b = occ_b["doc_index"]
        hidden_b = occ_b[layer]
        if hidden_b.shape[0] < 4:
            continue

        unique_docs = torch.unique(doc_b)
        if unique_docs.numel() < 2:
            continue
        perm = unique_docs[torch.randperm(unique_docs.numel(), generator=generator)]
        midpoint = unique_docs.numel() // 2
        docs_1, docs_2 = perm[:midpoint], perm[midpoint:]
        mask_1 = torch.isin(doc_b, docs_1)
        mask_2 = torch.isin(doc_b, docs_2)
        if mask_1.sum() == 0 or mask_2.sum() == 0:
            continue

        profile_a = idx.profile(word, 0, reference_ids, layer=layer)

        d_halves = []
        for mask in (mask_1, mask_2):
            centroid_half = hidden_b[mask].mean(dim=0)
            profile_half = _profile_for_target_centroid(centroid_half, mu_b, references_b_normed)
            d_halves.append(displacement(profile_a.vector, profile_half))

        disp_full = idx.displacement(word, reference_ids, layer=layer)
        results.append(
            {
                "word": word,
                "D_obs": disp_full.score,
                "D_half1": d_halves[0],
                "D_half2": d_halves[1],
                "n_half1": int(mask_1.sum()),
                "n_half2": int(mask_2.sum()),
            }
        )
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
