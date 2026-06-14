#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.research.common.io import read_truth, write_csv  # noqa: E402
from scripts.research.common.profiles import (  # noqa: E402
    average_pairwise_cosine_distance,
    occurrence_profiles,
    relational_profiles,
)
from timeformers.relational import contextual_centroids  # noqa: E402


def balanced_apd(
    before: torch.Tensor,
    after: torch.Tensor,
    *,
    sample_size: int,
    seeds: int,
    seed: int,
) -> tuple[float, float]:
    size = min(sample_size, before.size(0), after.size(0))
    scores = []
    for offset in range(seeds):
        generator = torch.Generator().manual_seed(seed + offset)
        before_indices = torch.randperm(before.size(0), generator=generator)[:size]
        after_indices = torch.randperm(after.size(0), generator=generator)[:size]
        scores.append(
            average_pairwise_cosine_distance(
                before[before_indices],
                after[after_indices],
            )
        )
    return float(np.mean(scores)), float(np.std(scores, ddof=1))


def evaluate(rows: list[dict], truth: dict[str, dict[str, float]]) -> dict:
    rows = sorted(rows, key=lambda row: row["target"])
    scores = np.array([row["balanced_apd_mean"] for row in rows])
    graded = np.array([truth[row["target"]]["graded"] for row in rows])
    binary = np.array([truth[row["target"]]["binary"] for row in rows])
    rho, p_value = spearmanr(graded, scores)
    return {
        "n_targets": len(rows),
        "spearman": float(rho),
        "spearman_p": float(p_value),
        "roc_auc": float(roc_auc_score(binary, scores)),
        "average_precision": float(average_precision_score(binary, scores)),
    }


def top_centroid_neighbors(
    centroids: torch.Tensor,
    target_id: int,
    reference_ids: list[int],
    references: list[str],
    *,
    top_k: int,
) -> list[dict]:
    profile = relational_profiles(
        centroids,
        [target_id],
        reference_ids,
        center=True,
    )[0]
    values, indices = torch.topk(profile, k=top_k)
    return [
        {
            "rank": rank + 1,
            "token": references[int(index)],
            "similarity": float(value),
        }
        for rank, (value, index) in enumerate(zip(values, indices))
    ]


def recurring_occurrence_neighbors(
    stats: dict,
    layer: str,
    target_index: int,
    centroids: torch.Tensor,
    reference_ids: list[int],
    references: list[str],
    *,
    top_k: int,
) -> list[dict]:
    profiles = occurrence_profiles(
        stats,
        layer,
        target_index,
        centroids,
        reference_ids,
        center=True,
    )
    indices = torch.topk(profiles, k=top_k, dim=1).indices.flatten()
    counts = torch.bincount(indices, minlength=len(references))
    values, top_indices = torch.topk(counts, k=min(30, len(references)))
    total_occurrences = profiles.size(0)
    return [
        {
            "token": references[int(index)],
            "top_k_occurrence_count": int(value),
            "top_k_occurrence_fraction": float(value / total_occurrences),
        }
        for value, index in zip(values, top_indices)
        if int(value) > 0
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Balanced APD and neighbor audit from cached hidden states.")
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--layer", default="layer_2")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--audit-targets", nargs="+", default=["plane_nn", "chairman_nn", "graft_nn"])
    args = parser.parse_args()

    vocab = json.loads((args.experiment_dir / "vocab.json").read_text(encoding="utf-8"))
    targets = json.loads((args.experiment_dir / "targets.json").read_text(encoding="utf-8"))
    references = json.loads((args.profile_dir / "references.json").read_text(encoding="utf-8"))
    token_to_id = {token: index for index, token in enumerate(vocab)}
    target_ids = [token_to_id[target] for target in targets]
    reference_ids = [token_to_id[token] for token in references]

    before_stats = torch.load(args.profile_dir / "cache" / "theta0_d0.pt", weights_only=True)
    after_stats = torch.load(args.profile_dir / "cache" / "theta1_d1.pt", weights_only=True)
    before_centroids = contextual_centroids(before_stats, args.layer)
    after_centroids = contextual_centroids(after_stats, args.layer)

    rows = []
    audit = {}
    for target_index, target in enumerate(targets):
        before = occurrence_profiles(
            before_stats,
            args.layer,
            target_index,
            before_centroids,
            reference_ids,
            center=False,
        )
        after = occurrence_profiles(
            after_stats,
            args.layer,
            target_index,
            after_centroids,
            reference_ids,
            center=False,
        )
        mean, std = balanced_apd(
            before,
            after,
            sample_size=args.sample_size,
            seeds=args.seeds,
            seed=args.seed + 10_000 * target_index,
        )
        rows.append(
            {
                "target": target,
                "count_d0": before.size(0),
                "count_d1": after.size(0),
                "sample_size": min(args.sample_size, before.size(0), after.size(0)),
                "balanced_apd_mean": mean,
                "balanced_apd_std": std,
                "full_apd": average_pairwise_cosine_distance(before, after),
            }
        )

        if target in args.audit_targets:
            audit[target] = {}
            for label, stats, centroids in (
                ("theta0_d0", before_stats, before_centroids),
                ("theta1_d1", after_stats, after_centroids),
            ):
                audit[target][label] = {
                    "centroid_neighbors": top_centroid_neighbors(
                        centroids,
                        target_ids[target_index],
                        reference_ids,
                        references,
                        top_k=20,
                    ),
                    "recurring_occurrence_neighbors": recurring_occurrence_neighbors(
                        stats,
                        args.layer,
                        target_index,
                        centroids,
                        reference_ids,
                        references,
                        top_k=args.top_k,
                    ),
                }

    truth = read_truth(args.truth)
    metrics = evaluate(rows, truth)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.output_dir / "balanced_apd_scores.csv")
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (args.output_dir / "neighbor_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    ranking = sorted(rows, key=lambda row: row["balanced_apd_mean"], reverse=True)
    rank_by_target = {row["target"]: rank + 1 for rank, row in enumerate(ranking)}
    summary = {
        **metrics,
        "layer": args.layer,
        "sample_size": args.sample_size,
        "seeds": args.seeds,
        "audited_targets": {
            target: {
                "rank": rank_by_target[target],
                "score": next(row["balanced_apd_mean"] for row in rows if row["target"] == target),
            }
            for target in args.audit_targets
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
