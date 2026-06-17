#!/usr/bin/env python3
"""Interactive-ish demo of the `TokenTimeIndex` facade (`token_time_repository.py`).

Loads a profile directory produced by `build_profiles.py` and walks through
the `token@time` operations end to end for a handful of words: the
relational profile at each period, the displacement between periods (top
gains/losses), and the nearest trajectories (other target words whose
meaning shifted in a similar direction).

This is meant to be read and re-run with different `--words`/`--profile-dir`
to get a feel for the API and sanity-check results -- it is not a
correctness test (see `tests/` for those).

Example:
    ./venv/bin/python scripts/token_time/explore_index.py \\
        --profile-dir outputs/token_time_fase_a/seed1000 \\
        --words plane chairman graft
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tracoformer.token_time_repository import TokenTimeIndex  # noqa: E402


def show_profile_top(idx: TokenTimeIndex, word: str, reference_ids: torch.Tensor, *, period_index: int, k: int) -> None:
    profile = idx.profile(word, period_index, reference_ids)
    top_values, top_indices = torch.topk(profile.vector, k)
    period_label = idx.period_files[period_index]
    se_label = f", standard_error={profile.standard_error:.4f}" if profile.standard_error is not None else ""
    print(f"  [{period_label}] count={profile.count}{se_label}")
    print(f"  [{period_label}] top-{k} nearest references:")
    for value, index in zip(top_values.tolist(), top_indices.tolist()):
        print(f"    {profile.reference_vocab[index]:<20} {value:+.4f}")


def show_displacement(idx: TokenTimeIndex, word: str, reference_ids: torch.Tensor, *, k: int) -> None:
    disp = idx.displacement(word, reference_ids)
    print(f"  displacement score (1 - cos) = {disp.score:.4f}")
    print(f"  top-{k} gains (became more similar to):")
    for token, value in disp.top_gains(k):
        print(f"    {token:<20} {value:+.4f}")
    print(f"  top-{k} losses (became less similar to):")
    for token, value in disp.top_losses(k):
        print(f"    {token:<20} {value:+.4f}")


def show_nearest(idx: TokenTimeIndex, word: str, reference_ids: torch.Tensor, *, k: int) -> None:
    print(f"  nearest trajectories (most similar displacement direction):")
    for other_word, similarity in idx.nearest(word, reference_ids=reference_ids, k=k):
        print(f"    {other_word:<20} {similarity:+.4f}")


def show_trust(
    idx: TokenTimeIndex,
    word: str,
    reference_ids: torch.Tensor,
    *,
    n_permutations: int,
    seed: int,
) -> None:
    """How much should we trust `δ(w)`? Three checks, all from
    `tmp/36-claude_token_time_signal_noise_measurement_proposal.md` (steps
    4/8):

    - nulo B (`null_b`): is `δ(w)` larger than what document-permutation
      alone would produce by chance (`ζ(w)`, one-sided `p`)?
    - split-half (`split_half`): does `δ(w)` hold up on two independent
      random halves of d1's documents, or is it driven by a few of them?

    Both require `idx.occurrences` (profile dirs written after the
    `OccurrenceCache` change); older profile dirs print a note and skip this.
    """
    if idx.occurrences is None:
        print("  (sem OccurrenceCache nesta pasta -- nulo B / split-half indisponiveis)")
        return

    disp = idx.displacement(word, reference_ids)
    d_null = idx.null_b(word, reference_ids, n_permutations=n_permutations, generator=torch.Generator().manual_seed(seed))
    median = d_null.median().item()
    mad = (d_null - median).abs().median().item()
    z = (disp.score - median) / (1.4826 * mad) if mad > 0 else float("nan")
    p = (1 + int((d_null >= disp.score).sum())) / (len(d_null) + 1)
    print(f"  delta(w) = {disp.score:.4f}")
    print(f"  nulo B (B={n_permutations}): median={median:.4f} MAD={mad:.4f} -> zeta(w)={z:+.2f}, p={p:.4f}")

    try:
        d_half1, d_half2 = idx.split_half(word, reference_ids, generator=torch.Generator().manual_seed(seed))
        print(f"  split-half: D_half1={d_half1:.4f}, D_half2={d_half2:.4f} (vs delta(w)={disp.score:.4f})")
    except ValueError as exc:
        print(f"  split-half: indisponivel ({exc})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Explore a token@time profile directory interactively.")
    parser.add_argument("--profile-dir", type=Path, default=ROOT / "outputs" / "token_time_fase_a" / "seed1000")
    parser.add_argument("--words", nargs="*", default=None, help="Target words to inspect (default: all targets)")
    parser.add_argument("--max-references", type=int, default=3216)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--nearest-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n-permutations", type=int, default=200, help="nulo B permutations for the trust section")
    parser.add_argument("--null-seed", type=int, default=0, help="generator seed for nulo B / split-half")
    parser.add_argument("--skip-trust", action="store_true", help="skip the nulo B / split-half trust section")
    args = parser.parse_args()

    idx = TokenTimeIndex.load(args.profile_dir, seed=args.seed)
    print(f"Loaded profile dir: {args.profile_dir}")
    print(f"  vocab size       = {len(idx.vocab)}")
    print(f"  targets          = {idx.targets}")
    print(f"  periods          = {idx.period_files}")

    reference_ids = idx.reference_set(max_references=args.max_references)
    print(f"  reference set    = {reference_ids.numel()} words")

    words = args.words or idx.targets
    for word in words:
        if word not in idx.target_ids:
            print(f"\n=== {word} === (skipped: not a target word)")
            continue
        print(f"\n=== {word} ===")
        show_profile_top(idx, word, reference_ids, period_index=0, k=args.top_k)
        show_profile_top(idx, word, reference_ids, period_index=1, k=args.top_k)
        show_displacement(idx, word, reference_ids, k=args.top_k)
        show_nearest(idx, word, reference_ids, k=args.nearest_k)
        if not args.skip_trust:
            print("  --- confiabilidade (nulo B / split-half) ---")
            show_trust(idx, word, reference_ids, n_permutations=args.n_permutations, seed=args.null_seed)


if __name__ == "__main__":
    main()
