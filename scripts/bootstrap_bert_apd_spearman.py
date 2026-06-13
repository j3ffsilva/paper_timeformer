#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


def parse_condition(value: str) -> tuple[str, Path]:
    try:
        label, path = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "conditions must use LABEL=EVALUATION_DIR"
        ) from exc
    if not label or not path:
        raise argparse.ArgumentTypeError(
            "conditions must use non-empty LABEL=EVALUATION_DIR"
        )
    return label, Path(path)


def read_truth(path: Path) -> dict[str, float]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {
            row["target"]: float(row["graded"])
            for row in csv.DictReader(handle, delimiter="\t")
        }


def read_scores(path: Path) -> dict[str, float]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {
            row["target"]: float(row["apd"])
            for row in csv.DictReader(handle)
        }


def correlation(gold: np.ndarray, scores: np.ndarray) -> float:
    rho = spearmanr(gold, scores).statistic
    return float(rho) if np.isfinite(rho) else 0.0


def summarize(samples: np.ndarray, point: float) -> dict[str, float]:
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return {
        "point": point,
        "bootstrap_mean": float(samples.mean()),
        "ci_95_low": float(lower),
        "ci_95_high": float(upper),
        "probability_gt_zero": float(np.mean(samples > 0.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Bootstrap target-level Spearman correlations and paired "
            "differences between aligned BERT APD evaluations."
        )
    )
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument(
        "--condition",
        action="append",
        type=parse_condition,
        required=True,
        help="Repeat as LABEL=EVALUATION_DIR.",
    )
    parser.add_argument(
        "--readout",
        action="append",
        default=None,
        help="Repeat to limit readouts; defaults to layer_1 and layer_2.",
    )
    parser.add_argument("--n-bootstrap", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260613)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.n_bootstrap < 1:
        parser.error("--n-bootstrap must be positive")
    readouts = args.readout or ["layer_1", "layer_2"]
    truth = read_truth(args.truth)
    condition_dirs = dict(args.condition)
    if len(condition_dirs) != len(args.condition):
        parser.error("condition labels must be unique")

    result = {
        "method": "paired nonparametric bootstrap over target words",
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "readouts": {},
    }
    rng = np.random.default_rng(args.seed)

    for readout in readouts:
        scores_by_condition = {
            label: read_scores(directory / f"rows_{readout}.csv")
            for label, directory in condition_dirs.items()
        }
        shared_targets = sorted(
            set(truth).intersection(
                *(set(scores) for scores in scores_by_condition.values())
            )
        )
        if len(shared_targets) < 3:
            raise ValueError(f"fewer than three shared targets for {readout}")

        gold = np.array([truth[target] for target in shared_targets])
        score_arrays = {
            label: np.array([scores[target] for target in shared_targets])
            for label, scores in scores_by_condition.items()
        }
        indices = rng.integers(
            0,
            len(shared_targets),
            size=(args.n_bootstrap, len(shared_targets)),
        )
        bootstrap_rhos = {
            label: np.array([
                correlation(gold[index], scores[index])
                for index in indices
            ])
            for label, scores in score_arrays.items()
        }
        condition_summaries = {
            label: summarize(
                bootstrap_rhos[label],
                correlation(gold, scores),
            )
            for label, scores in score_arrays.items()
        }
        differences = {}
        for left, right in combinations(score_arrays, 2):
            samples = bootstrap_rhos[left] - bootstrap_rhos[right]
            point = (
                condition_summaries[left]["point"]
                - condition_summaries[right]["point"]
            )
            differences[f"{left}_minus_{right}"] = summarize(samples, point)

        result["readouts"][readout] = {
            "n_targets": len(shared_targets),
            "targets": shared_targets,
            "conditions": condition_summaries,
            "paired_differences": differences,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
