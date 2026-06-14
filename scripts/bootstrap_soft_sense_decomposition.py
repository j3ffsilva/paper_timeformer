#!/usr/bin/env python3
"""Bootstrap occurrence uncertainty for the soft sense-vector decomposition."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from scripts.decompose_consec_timeformer_change import (
        LAYERS,
        decomposition_components,
    )
    from scripts.evaluate_consec_gate3 import spearman
    from scripts.evaluate_consec_timeformer_occurrences import collect_unique_rows
except ModuleNotFoundError:
    from decompose_consec_timeformer_change import LAYERS, decomposition_components
    from evaluate_consec_gate3 import spearman
    from evaluate_consec_timeformer_occurrences import collect_unique_rows


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def classify_interval(low: float, high: float) -> str:
    if low > 0:
        return "robust_positive"
    if high < 0:
        return "robust_negative"
    return "uncertain"


def stratified_bootstrap_indices(
    periods: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    period_values = sorted(set(periods.tolist()))
    sampled_parts = []
    for period in period_values:
        indices = np.flatnonzero(periods == period)
        sampled_parts.append(
            indices[
                rng.integers(
                    0,
                    len(indices),
                    size=(n_bootstrap, len(indices)),
                )
            ]
        )
    return np.concatenate(sampled_parts, axis=1)


def probability_matrix(rows: list[dict]) -> np.ndarray:
    probabilities_by_row = [
        json.loads(row["sense_probabilities"]) for row in rows
    ]
    sensekeys = sorted(probabilities_by_row[0])
    if any(sorted(values) != sensekeys for values in probabilities_by_row):
        raise ValueError(f"Sense inventory mismatch for {rows[0]['target']}")
    return np.asarray(
        [[values[key] for key in sensekeys] for values in probabilities_by_row],
        dtype=np.float64,
    )


def load_embedding_cache(
    path: Path,
    expected_ids: list[str],
) -> dict[str, np.ndarray]:
    cached = dict(np.load(path, allow_pickle=False))
    sample_ids = cached.pop("sample_ids").astype(str).tolist()
    if sample_ids != expected_ids:
        raise ValueError("Embedding cache sample IDs do not match predictions")
    return cached


def load_null_means(path: Path) -> dict[tuple[str, str, str, str], float]:
    return {
        (
            row["consec_seed"],
            row["model_seed"],
            row["layer"],
            row["target"],
        ): float(row["null_composition_share_mean"])
        for row in read_csv(path)
    }


def load_observed(path: Path) -> dict[str, dict[str, float]]:
    result = {}
    for row in read_csv(path):
        result[row["target"]] = {
            layer: float(row[f"{layer}_excess_composition_share"])
            for layer in LAYERS
        }
    return result


def bootstrap_target(
    *,
    target: str,
    grouped_by_seed: list[list[dict]],
    consec_seeds: list[str],
    model_seeds: list[str],
    embeddings: dict[str, np.ndarray],
    index_by_id: dict[str, int],
    null_means: dict[tuple[str, str, str, str], float],
    n_bootstrap: int,
    seed: int,
) -> dict[str, np.ndarray]:
    layer_estimates = {
        layer: np.zeros(n_bootstrap, dtype=np.float64) for layer in LAYERS
    }
    n_combinations = len(consec_seeds) * len(model_seeds)
    for consec_index, rows in enumerate(grouped_by_seed):
        periods = np.asarray([row["period"] for row in rows])
        probabilities = probability_matrix(rows)
        bootstrap_indices = stratified_bootstrap_indices(
            periods,
            n_bootstrap,
            seed + consec_index * 10_000,
        )
        for model_index, model_seed in enumerate(model_seeds):
            for layer in LAYERS:
                vectors = np.asarray(
                    embeddings[f"model_{model_index}_{layer}"][
                        [index_by_id[row["sample_id"]] for row in rows]
                    ],
                    dtype=np.float64,
                )
                vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
                null_mean = null_means[
                    (consec_seeds[consec_index], model_seed, layer, target)
                ]
                for bootstrap_index, selected in enumerate(bootstrap_indices):
                    components = decomposition_components(
                        probabilities[selected],
                        vectors[selected],
                        periods[selected],
                        vectors_are_normalized=True,
                    )
                    layer_estimates[layer][bootstrap_index] += (
                        components["composition_share"] - null_mean
                    ) / n_combinations
    return layer_estimates


def evaluate(args: argparse.Namespace) -> None:
    unique_rows, per_file = collect_unique_rows(args.prediction_files)
    expected_ids = [row["sample_id"] for row in unique_rows]
    embeddings = load_embedding_cache(args.embedding_cache, expected_ids)
    index_by_id = {
        row["sample_id"]: index for index, row in enumerate(unique_rows)
    }
    null_means = load_null_means(args.decomposition_details)
    observed = load_observed(args.decomposition_targets)
    grouped_per_seed = []
    for source_rows in per_file:
        grouped = defaultdict(list)
        for row in source_rows:
            if row["role"] == "confirmatory":
                grouped[row["target"]].append(row)
        grouped_per_seed.append(grouped)
    targets = sorted(observed)
    samples = {
        layer: np.empty((len(targets), args.n_bootstrap), dtype=np.float32)
        for layer in LAYERS
    }
    for target_index, target in enumerate(targets):
        result = bootstrap_target(
            target=target,
            grouped_by_seed=[
                grouped[target] for grouped in grouped_per_seed
            ],
            consec_seeds=args.consec_seeds,
            model_seeds=args.model_seeds,
            embeddings=embeddings,
            index_by_id=index_by_id,
            null_means=null_means,
            n_bootstrap=args.n_bootstrap,
            seed=args.bootstrap_seed + target_index * 100_000,
        )
        for layer in LAYERS:
            samples[layer][target_index] = result[layer]

    per_target = []
    for target_index, target in enumerate(targets):
        row = {"target": target}
        for layer in LAYERS:
            values = samples[layer][target_index].astype(np.float64)
            low, high = np.quantile(values, [0.025, 0.975])
            row.update(
                {
                    f"{layer}_observed_excess": observed[target][layer],
                    f"{layer}_bootstrap_median": float(np.median(values)),
                    f"{layer}_ci_95_low": float(low),
                    f"{layer}_ci_95_high": float(high),
                    f"{layer}_probability_positive": float(
                        np.mean(values > 0)
                    ),
                    f"{layer}_classification": classify_interval(low, high),
                }
            )
        per_target.append(row)

    primary_samples = samples["layer_2"].astype(np.float64)
    aggregate_means = primary_samples.mean(axis=0)
    aggregate_low, aggregate_high = np.quantile(
        aggregate_means, [0.025, 0.975]
    )
    observed_values = [
        row["layer_2_observed_excess"] for row in per_target
    ]
    bootstrap_medians = [
        row["layer_2_bootstrap_median"] for row in per_target
    ]
    robust_positive = [
        row["target"]
        for row in per_target
        if row["layer_2_classification"] == "robust_positive"
    ]
    robust_negative = [
        row["target"]
        for row in per_target
        if row["layer_2_classification"] == "robust_negative"
    ]
    probability_positive_count = sum(
        row["layer_2_probability_positive"] > 0.5
        for row in per_target
    )
    rank_stability = spearman(observed_values, bootstrap_medians)
    plane_row = next(row for row in per_target if row["target"] == "plane_nn")
    checks = {
        "aggregate_ci_above_zero": bool(aggregate_low > 0),
        "at_least_15_probability_positive": bool(
            probability_positive_count >= 15
        ),
        "plane_robust_positive": bool(
            plane_row["layer_2_classification"] == "robust_positive"
        ),
        "observed_bootstrap_rank_spearman_above_0_8": bool(
            rank_stability > 0.8
        ),
    }
    layer_summaries = {}
    for layer in LAYERS:
        layer_means = samples[layer].astype(np.float64).mean(axis=0)
        low, high = np.quantile(layer_means, [0.025, 0.975])
        layer_summaries[layer] = {
            "observed_mean_excess": float(
                np.mean(
                    [row[f"{layer}_observed_excess"] for row in per_target]
                )
            ),
            "bootstrap_mean_median": float(np.median(layer_means)),
            "bootstrap_mean_ci_95": [float(low), float(high)],
            "robust_positive_targets": [
                row["target"]
                for row in per_target
                if row[f"{layer}_classification"] == "robust_positive"
            ],
            "robust_negative_targets": [
                row["target"]
                for row in per_target
                if row[f"{layer}_classification"] == "robust_negative"
            ],
        }
    summary = {
        "n_targets": len(targets),
        "n_bootstrap": args.n_bootstrap,
        "primary_layer": "layer_2",
        "primary": {
            "observed_mean_excess": float(np.mean(observed_values)),
            "bootstrap_aggregate_median": float(
                np.median(aggregate_means)
            ),
            "bootstrap_aggregate_ci_95": [
                float(aggregate_low),
                float(aggregate_high),
            ],
            "targets_probability_positive_above_0_5": (
                probability_positive_count
            ),
            "robust_positive_targets": robust_positive,
            "robust_negative_targets": robust_negative,
            "uncertain_targets": [
                row["target"]
                for row in per_target
                if row["layer_2_classification"] == "uncertain"
            ],
            "observed_vs_bootstrap_median_spearman": rank_stability,
        },
        "layer_summaries": layer_summaries,
        "checks": checks,
        "global_conclusion_stable": bool(all(checks.values())),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_target.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_target[0]))
        writer.writeheader()
        writer.writerows(per_target)
    np.savez_compressed(
        args.output_dir / "bootstrap_samples.npz",
        targets=np.asarray(targets),
        layer_1=samples["layer_1"],
        layer_2=samples["layer_2"],
    )
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
    parser.add_argument("--decomposition-details", type=Path, required=True)
    parser.add_argument("--decomposition-targets", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260624)
    return parser


if __name__ == "__main__":
    evaluate(build_parser().parse_args())
