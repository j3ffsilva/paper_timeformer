#!/usr/bin/env python3
"""Calibrate the nulo B (document-permutation null, `token_time_null.
document_permutation_null`) on pseudo-periods, before it is used to judge
the 37 SemEval target words.

A pseudo-period pair is built by randomly splitting a single real period's
documents in half (see `data/processed/.../corpus_pseudo/pseudo_{a,b}.txt`,
produced once and reused across seeds for comparability). There is no real
temporal change between `pseudo_a` and `pseudo_b` -- both halves are random
samples of the *same* period -- so `D_obs(w)` between them is, by
construction, a draw from (something close to) the null itself.

This script checks the three calibration properties from
`tmp/36-claude_token_time_signal_noise_measurement_proposal.md` (step 5):

1. **Percentile distribution**: `percentile(w) = P[D_null(w) <= D_obs(w)]`
   should be roughly uniform on `[0, 1]` across words, if the null is
   well-calibrated (no real change exists between pseudo_a/pseudo_b).
2. **False-positive rate**: the fraction of words with one-sided
   `p < alpha` should be close to `alpha`.
3. **Null narrows with more data**: `MAD(D_null(w))` computed from the
   pseudo-periods (each ~half the documents of the real d0) should be larger
   than `MAD(D_null(w))` computed from the real d0-vs-d1 split (full-sized
   periods), for the same word -- pass `--compare-profile-dir` to check this.

Run on `outputs/token_time_fase_a/seed{1000,1001}_pseudo` and compare results
across seeds for repeatability.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from timeformers.token_time_repository import TokenTimeIndex  # noqa: E402


def evaluate(idx: TokenTimeIndex, reference_ids: torch.Tensor, *, layer: str, n_permutations: int, generator: torch.Generator) -> list[dict]:
    results = []
    for word in idx.targets:
        target_id = idx.target_ids[word]
        n0 = int(idx.periods[0].counts[target_id])
        n1 = int(idx.periods[1].counts[target_id])
        if n0 < 2 or n1 < 2:
            continue
        disp = idx.displacement(word, reference_ids, layer=layer)
        d_null = idx.null_b(word, reference_ids, layer=layer, n_permutations=n_permutations, generator=generator)
        median = d_null.median().item()
        mad = (d_null - median).abs().median().item()
        percentile = (d_null <= disp.score).float().mean().item()
        p = (1 + int((d_null >= disp.score).sum())) / (len(d_null) + 1)
        z = (disp.score - median) / (1.4826 * mad) if mad > 0 else float("nan")
        results.append(
            {
                "word": word,
                "n0": n0,
                "n1": n1,
                "D_obs": disp.score,
                "median_null": median,
                "mad_null": mad,
                "percentile": percentile,
                "p": p,
                "Z": z,
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile-dir", type=Path, required=True, help="pseudo-period profile dir (pseudo_a/pseudo_b)")
    parser.add_argument("--compare-profile-dir", type=Path, default=None, help="real d0/d1 profile dir, for MAD comparison")
    parser.add_argument("--layer", default="layer_2")
    parser.add_argument("--n-permutations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, nargs="+", default=[0.05, 0.1])
    args = parser.parse_args()

    idx = TokenTimeIndex.load(args.profile_dir)
    reference_ids = idx.reference_set()
    generator = torch.Generator().manual_seed(args.seed)
    results = evaluate(idx, reference_ids, layer=args.layer, n_permutations=args.n_permutations, generator=generator)

    compare = None
    if args.compare_profile_dir is not None:
        compare_idx = TokenTimeIndex.load(args.compare_profile_dir)
        compare_reference_ids = compare_idx.reference_set()
        compare_generator = torch.Generator().manual_seed(args.seed)
        compare = {
            row["word"]: row
            for row in evaluate(
                compare_idx, compare_reference_ids, layer=args.layer, n_permutations=args.n_permutations, generator=compare_generator
            )
        }

    for row in results:
        line = dict(row)
        if compare is not None and row["word"] in compare:
            line["mad_null_real"] = compare[row["word"]]["mad_null"]
        print(json.dumps(line))

    percentiles = [row["percentile"] for row in results]
    print(f"\nn_words = {len(results)}")
    print(f"mean percentile = {statistics.mean(percentiles):.3f} (expect ~0.5 if calibrated)")
    print(f"stdev percentile = {statistics.pstdev(percentiles):.3f} (expect ~0.29 for uniform[0,1])")
    for alpha in args.alpha:
        fp_rate = sum(1 for row in results if row["p"] < alpha) / len(results)
        print(f"false-positive rate at alpha={alpha}: {fp_rate:.3f} (expect ~{alpha})")

    if compare is not None:
        ratios = [
            row["mad_null"] / compare[row["word"]]["mad_null"]
            for row in results
            if row["word"] in compare and compare[row["word"]]["mad_null"] > 0
        ]
        if ratios:
            print(
                f"\nmean MAD(pseudo)/MAD(real d0-vs-d1) = {statistics.mean(ratios):.2f} "
                "(expect > 1: pseudo-periods have ~half the documents, so a wider null)"
            )


if __name__ == "__main__":
    main()
