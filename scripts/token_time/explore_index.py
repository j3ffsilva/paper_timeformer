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

from timeformers.token_time_repository import TokenTimeIndex  # noqa: E402


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Explore a token@time profile directory interactively.")
    parser.add_argument("--profile-dir", type=Path, default=ROOT / "outputs" / "token_time_fase_a" / "seed1000")
    parser.add_argument("--words", nargs="*", default=None, help="Target words to inspect (default: all targets)")
    parser.add_argument("--max-references", type=int, default=3216)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--nearest-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=None)
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


if __name__ == "__main__":
    main()
