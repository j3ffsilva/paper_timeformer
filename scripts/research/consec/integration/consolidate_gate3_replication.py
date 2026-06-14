#!/usr/bin/env python3
"""Consolidate preregistered ConSeC Gate 3 sampling replications."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from scripts.research.common.stats import partial_spearman, spearman  # noqa: E402


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def consolidate(args: argparse.Namespace) -> None:
    seed_rows = []
    score_maps = []
    shared_targets = None
    for seed, directory in zip(args.seeds, args.result_dirs):
        summary = json.loads((directory / "summary.json").read_text())
        rows = [
            row
            for row in read_csv(directory / "target_scores.csv")
            if row["role"] == "confirmatory"
        ]
        by_target = {row["target"]: row for row in rows}
        targets = set(by_target)
        shared_targets = targets if shared_targets is None else shared_targets & targets
        score_maps.append(by_target)
        seed_rows.append(
            {
                "seed": seed,
                "spearman": summary["primary_confirmatory"]["spearman"],
                "permutation_p": summary["primary_confirmatory"][
                    "permutation_p_two_sided"
                ],
                "partial_spearman": summary["inventory_size_control"][
                    "gold_partial_spearman_controlling_n_senses"
                ],
                "partial_permutation_p": summary["inventory_size_control"][
                    "partial_permutation_p_two_sided"
                ],
                "score_vs_n_senses": summary["inventory_size_control"][
                    "score_vs_n_senses_spearman"
                ],
            }
        )

    targets = sorted(shared_targets)
    gold = np.asarray(
        [float(score_maps[0][target]["graded"]) for target in targets]
    )
    n_senses = np.asarray(
        [int(score_maps[0][target]["n_senses"]) for target in targets]
    )
    score_matrix = np.asarray(
        [
            [float(score_map[target]["jsd"]) for target in targets]
            for score_map in score_maps
        ]
    )
    raw_rhos = np.asarray(
        [spearman(scores, gold) for scores in score_matrix]
    )
    partial_rhos = np.asarray(
        [
            partial_spearman(scores, gold, n_senses)
            for scores in score_matrix
        ]
    )

    rng = np.random.default_rng(args.permutation_seed)
    observed_mean = float(raw_rhos.mean())
    extreme = 0
    for _ in range(args.n_permutations):
        permuted = rng.permutation(gold)
        permuted_mean = np.mean(
            [spearman(scores, permuted) for scores in score_matrix]
        )
        if abs(permuted_mean) >= abs(observed_mean):
            extreme += 1
    joint_p = (extreme + 1) / (args.n_permutations + 1)

    target_rows = []
    for index, target in enumerate(targets):
        values = score_matrix[:, index]
        target_rows.append(
            {
                "target": target,
                "graded": gold[index],
                "n_senses": int(n_senses[index]),
                "mean_jsd": float(values.mean()),
                "std_jsd": float(values.std(ddof=1)),
                "min_jsd": float(values.min()),
                "max_jsd": float(values.max()),
                **{
                    "jsd_seed_{}".format(seed): float(value)
                    for seed, value in zip(args.seeds, values)
                },
            }
        )

    pairwise = []
    for left in range(len(args.seeds)):
        for right in range(left + 1, len(args.seeds)):
            pairwise.append(
                {
                    "seed_a": args.seeds[left],
                    "seed_b": args.seeds[right],
                    "score_spearman": spearman(
                        score_matrix[left], score_matrix[right]
                    ),
                }
            )

    checks = {
        "new_seed_spearmans_positive": all(
            row["spearman"] > 0 for row in seed_rows[1:]
        ),
        "mean_raw_spearman_positive": observed_mean > 0,
        "mean_partial_spearman_positive": float(partial_rhos.mean()) > 0,
        "joint_permutation_p_below_0_05": joint_p < 0.05,
    }
    summary = {
        "seeds": args.seeds,
        "n_targets": len(targets),
        "per_seed": seed_rows,
        "mean_raw_spearman": observed_mean,
        "std_raw_spearman": float(raw_rhos.std(ddof=1)),
        "mean_partial_spearman": float(partial_rhos.mean()),
        "std_partial_spearman": float(partial_rhos.std(ddof=1)),
        "joint_permutation_p_two_sided": float(joint_p),
        "pairwise_score_rank_stability": pairwise,
        "checks": checks,
        "replication_success": all(checks.values()),
        "n_permutations": args.n_permutations,
        "permutation_seed": args.permutation_seed,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_seed.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(seed_rows[0]))
        writer.writeheader()
        writer.writerows(seed_rows)
    with (args.output_dir / "per_target_stability.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(target_rows[0]))
        writer.writeheader()
        writer.writerows(target_rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n"
    )
    print(json.dumps(summary, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-permutations", type=int, default=20000)
    parser.add_argument("--permutation-seed", type=int, default=20260616)
    return parser


if __name__ == "__main__":
    consolidate(build_parser().parse_args())
