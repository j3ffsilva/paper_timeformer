#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.evaluate_hidden_relational_profiles import read_truth, write_csv  # noqa: E402


SCORE_COLUMNS = (
    "natural_jsd",
    "corpus_theta0_jsd",
    "corpus_theta1_jsd",
    "checkpoint_d0_jsd",
    "checkpoint_d1_jsd",
)


def eligible_settings(
    stability_rows: list[dict],
    *,
    min_ami: float,
    max_singleton_fraction: float,
    max_community_fraction: float,
) -> dict[tuple[int, float], dict]:
    return {
        (int(row["k"]), float(row["resolution"])): row
        for row in stability_rows
        if float(row["mean_ami"]) >= min_ami
        and float(row["singleton_fraction"]) <= max_singleton_fraction
        and float(row["max_community_fraction"]) <= max_community_fraction
    }


def aggregate_rows(
    score_rows: list[dict],
    settings: dict[tuple[int, float], dict],
    *,
    weighting: str,
) -> list[dict]:
    grouped = defaultdict(list)
    for row in score_rows:
        key = (int(row["k"]), float(row["resolution"]))
        if key in settings:
            grouped[(row["target"], float(row["temperature"]))].append(row)

    output = []
    for (target, temperature), rows in grouped.items():
        if weighting == "equal":
            weights = np.ones(len(rows), dtype=float)
        elif weighting == "ami":
            weights = np.array(
                [
                    float(settings[(int(row["k"]), float(row["resolution"]))]["mean_ami"])
                    for row in rows
                ]
            )
        else:
            raise ValueError("weighting must be 'equal' or 'ami'")
        weights /= weights.sum()
        aggregated = {
            column: float(
                np.dot(weights, np.array([float(row[column]) for row in rows]))
            )
            for column in SCORE_COLUMNS
        }
        output.append(
            {
                "target": target,
                "temperature": temperature,
                "weighting": weighting,
                "n_levels": len(rows),
                **aggregated,
                "frozen_corpus_mean": 0.5
                * (
                    aggregated["corpus_theta0_jsd"]
                    + aggregated["corpus_theta1_jsd"]
                ),
            }
        )
    return output


def evaluate(rows: list[dict], truth: dict[str, dict[str, float]]) -> list[dict]:
    metrics = []
    keys = sorted({(row["temperature"], row["weighting"]) for row in rows})
    for temperature, weighting in keys:
        selected = sorted(
            [
                row
                for row in rows
                if row["temperature"] == temperature and row["weighting"] == weighting
            ],
            key=lambda row: row["target"],
        )
        graded = np.array([truth[row["target"]]["graded"] for row in selected])
        binary = np.array([truth[row["target"]]["binary"] for row in selected])
        for score_name in ("natural_jsd", "frozen_corpus_mean"):
            scores = np.array([row[score_name] for row in selected])
            rho, p_value = spearmanr(graded, scores)
            metrics.append(
                {
                    "temperature": temperature,
                    "weighting": weighting,
                    "score": score_name,
                    "spearman": float(rho),
                    "spearman_p": float(p_value),
                    "roc_auc": float(roc_auc_score(binary, scores)),
                    "average_precision": float(average_precision_score(binary, scores)),
                }
            )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate stable community partitions across resolutions.")
    parser.add_argument("--community-dir", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-ami", type=float, default=0.8)
    parser.add_argument("--max-singleton-fraction", type=float, default=0.1)
    parser.add_argument("--max-community-fraction", type=float, default=0.25)
    args = parser.parse_args()

    with (args.community_dir / "partition_stability.csv").open(newline="", encoding="utf-8") as handle:
        stability_rows = list(csv.DictReader(handle))
    with (args.community_dir / "scores.csv").open(newline="", encoding="utf-8") as handle:
        score_rows = list(csv.DictReader(handle))

    settings = eligible_settings(
        stability_rows,
        min_ami=args.min_ami,
        max_singleton_fraction=args.max_singleton_fraction,
        max_community_fraction=args.max_community_fraction,
    )
    rows = []
    for weighting in ("equal", "ami"):
        rows.extend(aggregate_rows(score_rows, settings, weighting=weighting))
    truth = read_truth(args.truth)
    metrics = evaluate(rows, truth)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.output_dir / "scores.csv")
    write_csv(metrics, args.output_dir / "metrics.csv")
    (args.output_dir / "levels.json").write_text(
        json.dumps(
            [
                {
                    "k": key[0],
                    "resolution": key[1],
                    "mean_ami": float(row["mean_ami"]),
                    "n_communities": int(row["n_communities"]),
                }
                for key, row in sorted(settings.items())
            ],
            indent=2,
        ),
        encoding="utf-8",
    )

    diagnostic = {}
    for temperature in sorted({row["temperature"] for row in rows}):
        for weighting in ("equal", "ami"):
            selected = [
                row
                for row in rows
                if row["temperature"] == temperature and row["weighting"] == weighting
            ]
            ranking = sorted(selected, key=lambda row: row["natural_jsd"], reverse=True)
            ranks = {row["target"]: index + 1 for index, row in enumerate(ranking)}
            diagnostic[f"{weighting}_{temperature}"] = {
                target: {
                    "rank": ranks[target],
                    **next(row for row in selected if row["target"] == target),
                }
                for target in ("plane_nn", "chairman_nn", "graft_nn", "tree_nn")
            }
    summary = {
        "n_levels": len(settings),
        "metrics": metrics,
        "diagnostic": diagnostic,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
