#!/usr/bin/env python3
"""Audit per-sense contributions to the soft temporal decomposition."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from scripts.research.common.consec import collect_unique_rows  # noqa: E402


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sense_contributions(
    probabilities: np.ndarray,
    vectors: np.ndarray,
    periods: np.ndarray,
) -> dict[str, np.ndarray]:
    vectors = np.asarray(vectors, dtype=np.float64)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    mixtures = []
    centroids = []
    globals_ = []
    for period in sorted(set(periods.tolist())):
        selected = periods == period
        period_probabilities = probabilities[selected]
        period_vectors = vectors[selected]
        mixtures.append(period_probabilities.mean(axis=0))
        masses = period_probabilities.sum(axis=0)
        centroids.append(
            (period_probabilities.T @ period_vectors) / masses[:, None]
        )
        globals_.append(period_vectors.mean(axis=0))
    p0, p1 = mixtures
    mu0, mu1 = centroids
    total = globals_[1] - globals_[0]
    total_squared = float(np.dot(total, total))
    composition_vectors = (
        (p1 - p0)[:, None] * (mu1 + mu0) / 2.0
    )
    drift_vectors = (
        ((p1 + p0) / 2.0)[:, None] * (mu1 - mu0)
    )
    return {
        "p0": p0,
        "p1": p1,
        "composition_share": composition_vectors @ total / total_squared,
        "drift_share": drift_vectors @ total / total_squared,
        "centroid_cosine": np.asarray(
            [
                float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right)))
                for left, right in zip(mu0, mu1)
            ]
        ),
    }


def evaluate(args: argparse.Namespace) -> None:
    unique_rows, per_file = collect_unique_rows(args.prediction_files)
    expected_ids = [row["sample_id"] for row in unique_rows]
    cached = dict(np.load(args.embedding_cache, allow_pickle=False))
    sample_ids = cached.pop("sample_ids").astype(str).tolist()
    if sample_ids != expected_ids:
        raise ValueError("Embedding cache sample IDs do not match predictions")
    index_by_id = {
        row["sample_id"]: index for index, row in enumerate(unique_rows)
    }
    inventory = {
        (row["target"], row["sensekey"]): row
        for row in read_csv(args.sense_inventory)
    }
    grouped_per_seed = []
    for source_rows in per_file:
        grouped = defaultdict(list)
        for row in source_rows:
            if row["role"] == "confirmatory":
                grouped[row["target"]].append(row)
        grouped_per_seed.append(grouped)

    raw = defaultdict(list)
    validation_errors = []
    for target in args.targets:
        for consec_index, grouped in enumerate(grouped_per_seed):
            rows = grouped[target]
            probability_dicts = [
                json.loads(row["sense_probabilities"]) for row in rows
            ]
            sensekeys = sorted(probability_dicts[0])
            probabilities = np.asarray(
                [[values[key] for key in sensekeys] for values in probability_dicts],
                dtype=np.float64,
            )
            periods = np.asarray([row["period"] for row in rows])
            indices = [index_by_id[row["sample_id"]] for row in rows]
            for model_index, model_seed in enumerate(args.model_seeds):
                vectors = cached[f"model_{model_index}_{args.layer}"][indices]
                result = sense_contributions(probabilities, vectors, periods)
                validation_errors.append(
                    abs(
                        float(result["composition_share"].sum())
                        + float(result["drift_share"].sum())
                        - 1.0
                    )
                )
                for sense_index, sensekey in enumerate(sensekeys):
                    raw[(target, sensekey)].append(
                        {
                            "p0": result["p0"][sense_index],
                            "p1": result["p1"][sense_index],
                            "composition_share": result["composition_share"][
                                sense_index
                            ],
                            "drift_share": result["drift_share"][sense_index],
                            "centroid_cosine": result["centroid_cosine"][
                                sense_index
                            ],
                        }
                    )

    rows = []
    for (target, sensekey), values in sorted(raw.items()):
        metadata = inventory[(target, sensekey)]
        mean_p0 = float(np.mean([row["p0"] for row in values]))
        mean_p1 = float(np.mean([row["p1"] for row in values]))
        rows.append(
            {
                "target": target,
                "sensekey": sensekey,
                "synset": metadata["synset"],
                "definition": metadata["definition"],
                "mean_p0": mean_p0,
                "mean_p1": mean_p1,
                "delta_probability": mean_p1 - mean_p0,
                "mean_composition_share": float(
                    np.mean([row["composition_share"] for row in values])
                ),
                "mean_drift_share": float(
                    np.mean([row["drift_share"] for row in values])
                ),
                "mean_centroid_cosine": float(
                    np.mean([row["centroid_cosine"] for row in values])
                ),
            }
        )
    summary = {
        "layer": args.layer,
        "targets": args.targets,
        "maximum_share_sum_error": max(validation_errors),
        "top_positive_composition_senses": [
            {
                "target": row["target"],
                "sensekey": row["sensekey"],
                "definition": row["definition"],
                "delta_probability": row["delta_probability"],
                "mean_composition_share": row["mean_composition_share"],
            }
            for row in sorted(
                rows,
                key=lambda row: row["mean_composition_share"],
                reverse=True,
            )[:15]
        ],
        "top_negative_composition_senses": [
            {
                "target": row["target"],
                "sensekey": row["sensekey"],
                "definition": row["definition"],
                "delta_probability": row["delta_probability"],
                "mean_composition_share": row["mean_composition_share"],
            }
            for row in sorted(
                rows,
                key=lambda row: row["mean_composition_share"],
            )[:15]
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "sense_contributions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-files", type=Path, nargs=3, required=True)
    parser.add_argument("--model-seeds", nargs=2, required=True)
    parser.add_argument("--embedding-cache", type=Path, required=True)
    parser.add_argument("--sense-inventory", type=Path, required=True)
    parser.add_argument("--targets", nargs="+", required=True)
    parser.add_argument("--layer", choices=["layer_1", "layer_2"], default="layer_2")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


if __name__ == "__main__":
    evaluate(build_parser().parse_args())
