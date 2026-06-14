#!/usr/bin/env python3
"""Evaluate a within-target period-label permutation null for ConSeC JSD."""

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

from scripts.research.common.stats import jensen_shannon, spearman  # noqa: E402


def read_csv(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=np.float64)
    running = 1.0
    for reverse_index in range(len(values) - 1, -1, -1):
        index = order[reverse_index]
        rank = reverse_index + 1
        running = min(running, values[index] * len(values) / rank)
        adjusted[index] = running
    return adjusted.tolist()


def target_null(rows, n_permutations, seed):
    sensekeys = sorted(json.loads(rows[0]["sense_probabilities"]))
    matrix = np.asarray(
        [
            [
                json.loads(row["sense_probabilities"])[sensekey]
                for sensekey in sensekeys
            ]
            for row in rows
        ],
        dtype=np.float64,
    )
    periods = np.asarray([row["period"] for row in rows])
    d0_mask = periods == "1810-1860"
    n_d0 = int(d0_mask.sum())
    observed = jensen_shannon(
        matrix[d0_mask].mean(axis=0),
        matrix[~d0_mask].mean(axis=0),
    )
    rng = np.random.default_rng(seed)
    null_values = np.empty(n_permutations, dtype=np.float64)
    indices = np.arange(len(rows))
    for index in range(n_permutations):
        rng.shuffle(indices)
        left = matrix[indices[:n_d0]].mean(axis=0)
        right = matrix[indices[n_d0:]].mean(axis=0)
        null_values[index] = jensen_shannon(left, right)
    null_mean = float(null_values.mean())
    null_std = float(null_values.std(ddof=1))
    return {
        "observed_jsd": float(observed),
        "null_mean": null_mean,
        "null_std": null_std,
        "null_p95": float(np.quantile(null_values, 0.95)),
        "excess_jsd": float(observed - null_mean),
        "null_z": float(
            (observed - null_mean) / null_std if null_std else 0.0
        ),
        "p_upper": float(
            (np.sum(null_values >= observed) + 1)
            / (n_permutations + 1)
        ),
    }


def parse_cached_rows(path: Path) -> list[dict]:
    integer_fields = {"n_senses"}
    float_fields = {
        "observed_jsd",
        "null_mean",
        "null_std",
        "null_p95",
        "excess_jsd",
        "null_z",
        "p_upper",
        "p_fdr_bh",
    }
    rows = read_csv(path)
    for row in rows:
        for field in integer_fields:
            row[field] = int(row[field])
        for field in float_fields:
            row[field] = float(row[field])
    return rows


def summarize(all_seed_rows, args):
    seed_summaries = []
    truth = {
        row["target"]: row for row in read_csv(args.truth, delimiter="\t")
    }
    for seed in args.seeds:
        target_rows = [
            row for row in all_seed_rows if str(row["seed"]) == str(seed)
        ]
        if not target_rows:
            raise ValueError(f"No cached null rows found for seed {seed}")
        confirmatory = [
            row for row in target_rows if row["role"] == "confirmatory"
        ]
        gold = [float(truth[row["target"]]["graded"]) for row in confirmatory]
        n_senses = [row["n_senses"] for row in confirmatory]
        raw = [row["observed_jsd"] for row in confirmatory]
        excess = [row["excess_jsd"] for row in confirmatory]
        z_scores = [row["null_z"] for row in confirmatory]
        seed_summaries.append(
            {
                "seed": seed,
                "raw_gold_spearman": spearman(raw, gold),
                "excess_gold_spearman": spearman(excess, gold),
                "z_gold_spearman": spearman(z_scores, gold),
                "raw_n_senses_spearman": spearman(raw, n_senses),
                "excess_n_senses_spearman": spearman(excess, n_senses),
                "z_n_senses_spearman": spearman(z_scores, n_senses),
                "confirmatory_fdr_below_0_05": sum(
                    row["p_fdr_bh"] < 0.05 for row in confirmatory
                ),
            }
        )

    checks = {
        "excess_gold_positive_all_seeds": bool(
            all(row["excess_gold_spearman"] > 0 for row in seed_summaries)
        ),
        "mean_excess_gold_positive": bool(
            np.mean([row["excess_gold_spearman"] for row in seed_summaries])
            > 0
        ),
        "inventory_association_reduced_on_average": bool(
            np.mean(
                [
                    abs(row["excess_n_senses_spearman"])
                    for row in seed_summaries
                ]
            )
            < np.mean(
                [
                    abs(row["raw_n_senses_spearman"])
                    for row in seed_summaries
                ]
            )
        ),
    }
    return {
        "seeds": args.seeds,
        "n_permutations_per_target": args.n_permutations,
        "per_seed": seed_summaries,
        "mean_raw_gold_spearman": float(
            np.mean([row["raw_gold_spearman"] for row in seed_summaries])
        ),
        "mean_excess_gold_spearman": float(
            np.mean([row["excess_gold_spearman"] for row in seed_summaries])
        ),
        "mean_z_gold_spearman": float(
            np.mean([row["z_gold_spearman"] for row in seed_summaries])
        ),
        "checks": checks,
        "null_useful": bool(all(checks.values())),
    }


def evaluate(args: argparse.Namespace) -> None:
    if len(args.seeds) != len(args.prediction_files):
        raise ValueError("--seeds and --prediction-files must have equal length")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.output_dir / "per_target_null.csv"
    if args.resume and cache_path.exists():
        all_seed_rows = parse_cached_rows(cache_path)
    else:
        all_seed_rows = []
        for seed_offset, (seed, path) in enumerate(
            zip(args.seeds, args.prediction_files)
        ):
            grouped = defaultdict(list)
            for row in read_csv(path):
                grouped[row["target"]].append(row)
            target_rows = []
            for target_index, target in enumerate(sorted(grouped)):
                source = grouped[target]
                metric = target_null(
                    source,
                    args.n_permutations,
                    args.permutation_seed + seed_offset * 1000 + target_index,
                )
                first = source[0]
                target_rows.append(
                    {
                        "seed": seed,
                        "target": target,
                        "role": first["role"],
                        "n_senses": int(first["n_wordnet_senses"]),
                        **metric,
                    }
                )
            adjusted = benjamini_hochberg(
                [row["p_upper"] for row in target_rows]
            )
            for row, value in zip(target_rows, adjusted):
                row["p_fdr_bh"] = value
            all_seed_rows.extend(target_rows)
        with cache_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(all_seed_rows[0]))
            writer.writeheader()
            writer.writerows(all_seed_rows)

    summary = summarize(all_seed_rows, args)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n"
    )
    print(json.dumps(summary, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-files", type=Path, nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-permutations", type=int, default=20000)
    parser.add_argument("--permutation-seed", type=int, default=20260617)
    parser.add_argument("--resume", action="store_true")
    return parser


if __name__ == "__main__":
    evaluate(build_parser().parse_args())
