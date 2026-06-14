#!/usr/bin/env python3
"""Decompose temporal centroid change into sense composition and within-sense drift."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from scripts.evaluate_consec_gate3 import jensen_shannon, spearman
    from scripts.evaluate_consec_timeformer_occurrences import (
        bootstrap_mean_ci,
        collect_unique_rows,
        sign_flip_p,
    )
except ModuleNotFoundError:
    from evaluate_consec_gate3 import jensen_shannon, spearman
    from evaluate_consec_timeformer_occurrences import (
        bootstrap_mean_ci,
        collect_unique_rows,
        sign_flip_p,
    )


LAYERS = ("layer_1", "layer_2")


def vector_cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0:
        return float("nan")
    return float(np.dot(left, right) / denominator)


def decomposition_components(
    probabilities: np.ndarray,
    vectors: np.ndarray,
    periods: np.ndarray,
    *,
    vectors_are_normalized: bool = False,
) -> dict[str, np.ndarray | float]:
    vectors = np.asarray(vectors, dtype=np.float64)
    normalized = (
        vectors
        if vectors_are_normalized
        else vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    )
    period_values = sorted(set(periods.tolist()))
    if len(period_values) != 2:
        raise ValueError(f"Expected two periods, got {period_values}")
    mixtures = []
    centroids = []
    global_centroids = []
    for period in period_values:
        selected = periods == period
        period_probabilities = probabilities[selected]
        period_vectors = normalized[selected]
        mixtures.append(period_probabilities.mean(axis=0))
        masses = period_probabilities.sum(axis=0)
        centroids.append(
            (period_probabilities.T @ period_vectors) / masses[:, None]
        )
        global_centroids.append(period_vectors.mean(axis=0))
    p0, p1 = mixtures
    mu0, mu1 = centroids
    m0, m1 = global_centroids
    total = m1 - m0
    composition = np.sum(
        (p1 - p0)[:, None] * (mu1 + mu0) / 2.0,
        axis=0,
    )
    drift = np.sum(
        ((p1 + p0) / 2.0)[:, None] * (mu1 - mu0),
        axis=0,
    )
    total_squared = float(np.dot(total, total))
    if total_squared == 0:
        raise ValueError("Zero temporal centroid displacement")
    reconstruction_error = float(
        np.linalg.norm(total - composition - drift) / np.linalg.norm(total)
    )
    return {
        "total": total,
        "composition": composition,
        "drift": drift,
        "composition_share": float(np.dot(composition, total) / total_squared),
        "drift_share": float(np.dot(drift, total) / total_squared),
        "total_norm": float(np.linalg.norm(total)),
        "composition_norm": float(np.linalg.norm(composition)),
        "drift_norm": float(np.linalg.norm(drift)),
        "composition_total_cosine": vector_cosine(composition, total),
        "drift_total_cosine": vector_cosine(drift, total),
        "reconstruction_error": reconstruction_error,
        "sense_jsd": jensen_shannon(p0, p1),
    }


def decompose_with_null(
    rows: list[dict],
    vectors: np.ndarray,
    *,
    n_permutations: int,
    seed: int,
) -> dict[str, float]:
    probabilities_by_row = [
        json.loads(row["sense_probabilities"]) for row in rows
    ]
    sensekeys = sorted(probabilities_by_row[0])
    if any(sorted(values) != sensekeys for values in probabilities_by_row):
        raise ValueError(f"Sense inventory mismatch for {rows[0]['target']}")
    probabilities = np.asarray(
        [[values[key] for key in sensekeys] for values in probabilities_by_row],
        dtype=np.float64,
    )
    periods = np.asarray([row["period"] for row in rows])
    normalized_vectors = np.asarray(vectors, dtype=np.float64)
    normalized_vectors /= np.linalg.norm(
        normalized_vectors, axis=1, keepdims=True
    )
    observed = decomposition_components(
        probabilities,
        normalized_vectors,
        periods,
        vectors_are_normalized=True,
    )
    rng = np.random.default_rng(seed)
    null_shares = np.empty(n_permutations, dtype=np.float64)
    period_masks = [periods == period for period in sorted(set(periods))]
    for permutation_index in range(n_permutations):
        shuffled = probabilities.copy()
        for mask in period_masks:
            shuffled[mask] = probabilities[mask][rng.permutation(mask.sum())]
        null_shares[permutation_index] = decomposition_components(
            shuffled,
            normalized_vectors,
            periods,
            vectors_are_normalized=True,
        )["composition_share"]
    null_mean = float(null_shares.mean())
    null_std = float(null_shares.std(ddof=1))
    return {
        "sense_jsd": float(observed["sense_jsd"]),
        "total_norm": float(observed["total_norm"]),
        "composition_norm": float(observed["composition_norm"]),
        "drift_norm": float(observed["drift_norm"]),
        "composition_share": float(observed["composition_share"]),
        "drift_share": float(observed["drift_share"]),
        "composition_total_cosine": float(
            observed["composition_total_cosine"]
        ),
        "drift_total_cosine": float(observed["drift_total_cosine"]),
        "reconstruction_error": float(observed["reconstruction_error"]),
        "null_composition_share_mean": null_mean,
        "null_composition_share_std": null_std,
        "excess_composition_share": float(
            observed["composition_share"] - null_mean
        ),
        "composition_share_null_z": float(
            (observed["composition_share"] - null_mean) / null_std
            if null_std
            else 0.0
        ),
        "composition_share_p_upper": float(
            (np.sum(null_shares >= observed["composition_share"]) + 1)
            / (n_permutations + 1)
        ),
    }


def read_embedding_cache(path: Path, expected_ids: list[str]) -> dict[str, np.ndarray]:
    cached = dict(np.load(path, allow_pickle=False))
    sample_ids = cached.pop("sample_ids").astype(str).tolist()
    if sample_ids != expected_ids:
        raise ValueError("Embedding cache sample IDs do not match predictions")
    return cached


def evaluate(args: argparse.Namespace) -> None:
    unique_rows, per_file = collect_unique_rows(args.prediction_files)
    expected_ids = [row["sample_id"] for row in unique_rows]
    embeddings = read_embedding_cache(args.embedding_cache, expected_ids)
    index_by_id = {
        row["sample_id"]: index for index, row in enumerate(unique_rows)
    }
    detailed = []
    for consec_index, source_rows in enumerate(per_file):
        grouped = defaultdict(list)
        for row in source_rows:
            if row["role"] == "confirmatory":
                grouped[row["target"]].append(row)
        for model_index, model_seed in enumerate(args.model_seeds):
            for layer_index, layer in enumerate(LAYERS):
                all_vectors = embeddings[f"model_{model_index}_{layer}"]
                for target_index, target in enumerate(sorted(grouped)):
                    rows = grouped[target]
                    indices = [index_by_id[row["sample_id"]] for row in rows]
                    metrics = decompose_with_null(
                        rows,
                        all_vectors[indices],
                        n_permutations=args.null_permutations,
                        seed=(
                            args.null_seed
                            + consec_index * 100_000
                            + model_index * 10_000
                            + layer_index * 1_000
                            + target_index
                        ),
                    )
                    detailed.append(
                        {
                            "consec_seed": args.consec_seeds[consec_index],
                            "model_seed": model_seed,
                            "layer": layer,
                            "target": target,
                            **metrics,
                        }
                    )

    targets = sorted({row["target"] for row in detailed})
    per_target = []
    for target in targets:
        output = {"target": target}
        for layer in LAYERS:
            selected = [
                row for row in detailed
                if row["target"] == target and row["layer"] == layer
            ]
            for metric in [
                "sense_jsd",
                "total_norm",
                "composition_share",
                "drift_share",
                "excess_composition_share",
                "composition_share_null_z",
                "reconstruction_error",
            ]:
                output[f"{layer}_{metric}"] = float(
                    np.mean([row[metric] for row in selected])
                )
        per_target.append(output)

    combination_summaries = []
    for consec_seed in args.consec_seeds:
        for model_seed in args.model_seeds:
            selected = [
                row for row in detailed
                if row["layer"] == "layer_2"
                and row["consec_seed"] == consec_seed
                and row["model_seed"] == model_seed
            ]
            values = np.asarray(
                [row["excess_composition_share"] for row in selected]
            )
            combination_summaries.append(
                {
                    "consec_seed": consec_seed,
                    "model_seed": model_seed,
                    "mean_excess_composition_share": float(np.mean(values)),
                    "median_excess_composition_share": float(np.median(values)),
                    "positive_targets": int(np.sum(values > 0)),
                }
            )

    aggregate = np.asarray(
        [row["layer_2_excess_composition_share"] for row in per_target]
    )
    maximum_error = max(row["reconstruction_error"] for row in detailed)
    checks = {
        "mean_positive_all_six_combinations": bool(
            all(
                row["mean_excess_composition_share"] > 0
                for row in combination_summaries
            )
        ),
        "aggregate_mean_positive": bool(np.mean(aggregate) > 0),
        "sign_flip_p_below_0_05": bool(
            sign_flip_p(
                aggregate,
                args.aggregate_permutations,
                args.aggregate_seed,
            )
            < 0.05
        ),
        "at_least_15_positive_targets": bool(np.sum(aggregate > 0) >= 15),
        "reconstruction_error_below_1e_6": bool(maximum_error < 1e-6),
    }
    layer_summaries = {}
    for layer_index, layer in enumerate(LAYERS):
        selected_targets = [
            row[f"{layer}_excess_composition_share"] for row in per_target
        ]
        selected_details = [row for row in detailed if row["layer"] == layer]
        layer_summaries[layer] = {
            "mean_composition_share": float(
                np.mean([row["composition_share"] for row in selected_details])
            ),
            "mean_drift_share": float(
                np.mean([row["drift_share"] for row in selected_details])
            ),
            "mean_excess_composition_share": float(
                np.mean(selected_targets)
            ),
            "median_excess_composition_share": float(
                np.median(selected_targets)
            ),
            "positive_excess_targets": int(
                np.sum(np.asarray(selected_targets) > 0)
            ),
            "bootstrap_mean_ci_95": bootstrap_mean_ci(
                selected_targets,
                args.aggregate_bootstrap,
                args.aggregate_seed + layer_index,
            ),
            "sign_flip_p_one_sided": sign_flip_p(
                selected_targets,
                args.aggregate_permutations,
                args.aggregate_seed + 10 + layer_index,
            ),
            "excess_share_vs_sense_jsd_spearman": spearman(
                selected_targets,
                [row[f"{layer}_sense_jsd"] for row in per_target],
            ),
        }
    summary = {
        "n_targets": len(targets),
        "n_combinations_per_layer": (
            len(args.consec_seeds) * len(args.model_seeds)
        ),
        "null_permutations_per_target_combination": args.null_permutations,
        "maximum_reconstruction_error": float(maximum_error),
        "primary_layer": "layer_2",
        "primary": {
            "mean_excess_composition_share": float(np.mean(aggregate)),
            "median_excess_composition_share": float(np.median(aggregate)),
            "positive_targets": int(np.sum(aggregate > 0)),
            "bootstrap_mean_ci_95": bootstrap_mean_ci(
                aggregate,
                args.aggregate_bootstrap,
                args.aggregate_seed,
            ),
            "sign_flip_p_one_sided": sign_flip_p(
                aggregate,
                args.aggregate_permutations,
                args.aggregate_seed,
            ),
            "combinations": combination_summaries,
        },
        "layer_summaries": layer_summaries,
        "checks": checks,
        "composition_geometry_passed": bool(all(checks.values())),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_combination_target.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detailed[0]))
        writer.writeheader()
        writer.writerows(detailed)
    with (args.output_dir / "per_target.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_target[0]))
        writer.writeheader()
        writer.writerows(per_target)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-files", type=Path, nargs=3, required=True)
    parser.add_argument("--consec-seeds", nargs=3, required=True)
    parser.add_argument("--model-seeds", nargs=2, required=True)
    parser.add_argument("--embedding-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--null-permutations", type=int, default=2000)
    parser.add_argument("--null-seed", type=int, default=20260621)
    parser.add_argument("--aggregate-permutations", type=int, default=20000)
    parser.add_argument("--aggregate-bootstrap", type=int, default=20000)
    parser.add_argument("--aggregate-seed", type=int, default=20260622)
    return parser


if __name__ == "__main__":
    evaluate(build_parser().parse_args())
