#!/usr/bin/env python3
"""Measure convergent validity between ConSeC and TimeFormer word scores."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from scripts.evaluate_consec_gate3 import rankdata, spearman
except ModuleNotFoundError:
    from evaluate_consec_gate3 import rankdata, spearman


def read_csv(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def read_apd(path: Path) -> dict[str, float]:
    return {row["target"]: float(row["apd"]) for row in read_csv(path)}


def partial_spearman_multi(values_a, values_b, controls) -> float:
    ranks_a = rankdata(values_a)
    ranks_b = rankdata(values_b)
    control_array = np.asarray(controls, dtype=np.float64)
    if control_array.ndim == 1:
        control_array = control_array[:, None]
    ranked_controls = np.column_stack(
        [rankdata(control_array[:, index]) for index in range(control_array.shape[1])]
    )
    design = np.column_stack(
        [np.ones(len(ranks_a), dtype=np.float64), ranked_controls]
    )
    residual_a = ranks_a - design.dot(
        np.linalg.lstsq(design, ranks_a, rcond=None)[0]
    )
    residual_b = ranks_b - design.dot(
        np.linalg.lstsq(design, ranks_b, rcond=None)[0]
    )
    denominator = np.sqrt(
        np.sum(residual_a ** 2) * np.sum(residual_b ** 2)
    )
    return float(np.sum(residual_a * residual_b) / denominator)


def correlation_summary(
    values_a,
    values_b,
    n_bootstrap: int,
    n_permutations: int,
    seed: int,
) -> dict:
    a = np.asarray(values_a, dtype=np.float64)
    b = np.asarray(values_b, dtype=np.float64)
    rho = spearman(a, b)
    rng = np.random.default_rng(seed)
    bootstrap = []
    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(a), size=len(a))
        value = spearman(a[indices], b[indices])
        if not math.isnan(value):
            bootstrap.append(value)
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    extreme = sum(
        abs(spearman(a, rng.permutation(b))) >= abs(rho)
        for _ in range(n_permutations)
    )
    return {
        "n_targets": len(a),
        "spearman": rho,
        "bootstrap_ci_95_low": float(low),
        "bootstrap_ci_95_high": float(high),
        "permutation_p_two_sided": float(
            (extreme + 1) / (n_permutations + 1)
        ),
    }


def aggregate_consec(
    score_files: list[Path],
    null_file: Path,
) -> dict[str, dict]:
    raw_by_target = defaultdict(list)
    metadata = {}
    for path in score_files:
        for row in read_csv(path):
            raw_by_target[row["target"]].append(float(row["jsd"]))
            metadata[row["target"]] = {
                "role": row["role"],
                "coverage_confidence": row["coverage_confidence"],
                "n_senses": int(row["n_senses"]),
                "graded": float(row["graded"]),
            }
    null_by_target = defaultdict(lambda: defaultdict(list))
    for row in read_csv(null_file):
        null_by_target[row["target"]]["excess"].append(
            float(row["excess_jsd"])
        )
        null_by_target[row["target"]]["z"].append(float(row["null_z"]))
    result = {}
    for target, values in raw_by_target.items():
        result[target] = {
            **metadata[target],
            "consec_jsd": float(np.mean(values)),
            "consec_excess": float(
                np.mean(null_by_target[target]["excess"])
            ),
            "consec_z": float(np.mean(null_by_target[target]["z"])),
        }
    return result


def load_condition(directory: Path) -> dict[str, dict[str, float]]:
    layer_1 = read_apd(directory / "rows_layer_1.csv")
    layer_2 = read_apd(directory / "rows_layer_2.csv")
    if set(layer_1) != set(layer_2):
        raise ValueError(f"Layer target mismatch in {directory}")
    return {
        target: {
            "layer_1": layer_1[target],
            "layer_2": layer_2[target],
            "layer_delta": layer_2[target] - layer_1[target],
        }
        for target in layer_1
    }


def evaluate(args: argparse.Namespace) -> None:
    consec = aggregate_consec(args.consec_score_files, args.null_file)
    conditions = {
        name: load_condition(path)
        for name, path in [
            ("full_s1000", args.full_seed1000),
            ("full_s1001", args.full_seed1001),
            ("pseudo_s1000", args.pseudo_seed1000),
            ("lower_lr_s1000", args.lower_lr_seed1000),
        ]
    }
    targets = sorted(
        target
        for target, row in consec.items()
        if row["role"] == "confirmatory"
    )
    for name, condition in conditions.items():
        missing = set(targets) - set(condition)
        if missing:
            raise ValueError(f"{name} is missing targets: {sorted(missing)}")

    rows = []
    for target in targets:
        full_layer_1 = float(
            np.mean(
                [
                    conditions["full_s1000"][target]["layer_1"],
                    conditions["full_s1001"][target]["layer_1"],
                ]
            )
        )
        full_layer_2 = float(
            np.mean(
                [
                    conditions["full_s1000"][target]["layer_2"],
                    conditions["full_s1001"][target]["layer_2"],
                ]
            )
        )
        rows.append(
            {
                "target": target,
                **consec[target],
                "full_mean_layer_1": full_layer_1,
                "full_mean_layer_2": full_layer_2,
                "full_mean_layer_delta": full_layer_2 - full_layer_1,
                "full_s1000_layer_1": conditions["full_s1000"][target][
                    "layer_1"
                ],
                "full_s1001_layer_1": conditions["full_s1001"][target][
                    "layer_1"
                ],
                "pseudo_s1000_layer_1": conditions["pseudo_s1000"][target][
                    "layer_1"
                ],
                "pseudo_s1000_layer_delta": conditions[
                    "pseudo_s1000"
                ][target]["layer_delta"],
                "chronology_specific_delta_s1000": (
                    conditions["full_s1000"][target]["layer_delta"]
                    - conditions["pseudo_s1000"][target]["layer_delta"]
                ),
                "lower_lr_s1000_layer_1": conditions[
                    "lower_lr_s1000"
                ][target]["layer_1"],
            }
        )

    seed = args.seed
    main = {}
    for offset, score_name in enumerate(["consec_jsd", "consec_excess"]):
        main[score_name] = correlation_summary(
            [row["full_mean_layer_1"] for row in rows],
            [row[score_name] for row in rows],
            args.n_bootstrap,
            args.n_permutations,
            seed + offset,
        )

    controls = {}
    for score_name in ["consec_jsd", "consec_excess", "consec_z"]:
        layer = [row["full_mean_layer_1"] for row in rows]
        score = [row[score_name] for row in rows]
        controls[score_name] = {
            "partial_controlling_gold": partial_spearman_multi(
                layer, score, [row["graded"] for row in rows]
            ),
            "partial_controlling_n_senses": partial_spearman_multi(
                layer, score, [row["n_senses"] for row in rows]
            ),
            "partial_controlling_gold_and_n_senses": (
                partial_spearman_multi(
                    layer,
                    score,
                    [
                        [row["graded"], row["n_senses"]]
                        for row in rows
                    ],
                )
            ),
        }

    sensitivity = {}
    comparisons = [
        "full_s1000_layer_1",
        "full_s1001_layer_1",
        "pseudo_s1000_layer_1",
        "lower_lr_s1000_layer_1",
        "full_mean_layer_2",
        "full_mean_layer_delta",
        "pseudo_s1000_layer_delta",
        "chronology_specific_delta_s1000",
    ]
    for predictor in comparisons:
        sensitivity[predictor] = {
            score: spearman(
                [row[predictor] for row in rows],
                [row[score] for row in rows],
            )
            for score in ["consec_jsd", "consec_excess", "consec_z"]
        }

    high_rows = [
        row for row in rows if row["coverage_confidence"] == "high"
    ]
    high_coverage = {
        score: spearman(
            [row["full_mean_layer_1"] for row in high_rows],
            [row[score] for row in high_rows],
        )
        for score in ["consec_jsd", "consec_excess", "consec_z"]
    }
    benchmark_context = {}
    for label, subset in [("confirmatory", rows), ("high_coverage", high_rows)]:
        benchmark_context[label] = {
            score: spearman(
                [row[score] for row in subset],
                [row["graded"] for row in subset],
            )
            for score in [
                "consec_jsd",
                "consec_excess",
                "consec_z",
                "full_mean_layer_1",
                "full_mean_layer_2",
                "full_mean_layer_delta",
            ]
        }
    checks = {
        "raw_positive": bool(main["consec_jsd"]["spearman"] > 0),
        "excess_positive": bool(main["consec_excess"]["spearman"] > 0),
        "excess_permutation_p_below_0_05": bool(
            main["consec_excess"]["permutation_p_two_sided"] < 0.05
        ),
    }
    summary = {
        "n_confirmatory_targets": len(rows),
        "n_high_coverage_targets": len(high_rows),
        "main": main,
        "partial_controls": controls,
        "sensitivity": sensitivity,
        "high_coverage": high_coverage,
        "posthoc_benchmark_context": benchmark_context,
        "checks": checks,
        "convergent_validity_passed": bool(all(checks.values())),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_target.csv").open(
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
    parser.add_argument(
        "--consec-score-files", type=Path, nargs=3, required=True
    )
    parser.add_argument("--null-file", type=Path, required=True)
    parser.add_argument("--full-seed1000", type=Path, required=True)
    parser.add_argument("--full-seed1001", type=Path, required=True)
    parser.add_argument("--pseudo-seed1000", type=Path, required=True)
    parser.add_argument("--lower-lr-seed1000", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-bootstrap", type=int, default=20000)
    parser.add_argument("--n-permutations", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260618)
    return parser


if __name__ == "__main__":
    evaluate(build_parser().parse_args())
