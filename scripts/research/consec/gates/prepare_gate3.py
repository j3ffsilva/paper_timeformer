#!/usr/bin/env python3
"""Freeze balanced Gate 3 occurrence samples after the WordNet coverage audit."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path


PERIODS = ("1810-1860", "1960-2010")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def strip_pos_suffix(token: str) -> str:
    if token.endswith(("_nn", "_vb")):
        return token[:-3]
    return token


def display_context(tokens: list[str], index: int, radius: int) -> str:
    start = max(0, index - radius)
    end = min(len(tokens), index + radius + 1)
    shown = [strip_pos_suffix(token) for token in tokens[start:end]]
    shown[index - start] = f"[{strip_pos_suffix(tokens[index])}]"
    return " ".join(shown)


def assign_role(row: dict[str, str]) -> str:
    n_senses = int(row["n_wordnet_senses"])
    if row["coverage_status"] == "sufficient" and n_senses > 1:
        return "confirmatory"
    if row["coverage_status"] == "partial":
        return "partial_diagnostic"
    if row["coverage_status"] == "monosemous_covered":
        return "monosemous_control"
    return "excluded"


def collect_occurrences(
    corpus_dir: Path,
    reviews: list[dict[str, str]],
    *,
    max_per_period: int,
    radius: int,
    seed: int,
) -> list[dict[str, str]]:
    review_by_target = {row["target"]: row for row in reviews}
    target_set = set(review_by_target)
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for period in PERIODS:
        with (corpus_dir / f"{period}.txt").open(encoding="utf-8") as handle:
            for document_index, line in enumerate(handle):
                tokens = line.split()
                for token_index, target in enumerate(tokens):
                    if target not in target_set:
                        continue
                    grouped.setdefault((target, period), []).append(
                        {
                            "sample_id": (
                                f"{period}-{target}-{document_index}-{token_index}"
                            ),
                            "target": target,
                            "period": period,
                            "document_index": str(document_index),
                            "token_index": str(token_index),
                            "context": display_context(tokens, token_index, radius),
                        }
                    )

    output = []
    for target in sorted(target_set):
        counts = [len(grouped[(target, period)]) for period in PERIODS]
        sample_size = min(max_per_period, *counts)
        for period in PERIODS:
            rows = list(grouped[(target, period)])
            random.Random(f"{seed}:{target}:{period}:gate3").shuffle(rows)
            rows = sorted(rows[:sample_size], key=lambda row: row["sample_id"])
            for row in rows:
                review = review_by_target[target]
                row.update(
                    {
                        "role": assign_role(review),
                        "coverage_status": review["coverage_status"],
                        "coverage_confidence": review["confidence"],
                        "n_wordnet_senses": review["n_wordnet_senses"],
                    }
                )
                output.append(row)
    return sorted(output, key=lambda row: row["sample_id"])


def prepare(args: argparse.Namespace) -> None:
    reviews = read_csv(args.coverage_review)
    required = {"coverage_status", "gate3_decision", "confidence"}
    incomplete = [
        row["target"]
        for row in reviews
        if any(not row[field].strip() for field in required)
    ]
    if incomplete:
        raise ValueError(f"Incomplete coverage review: {incomplete}")

    occurrences = collect_occurrences(
        args.corpus_dir,
        reviews,
        max_per_period=args.max_per_period,
        radius=args.context_radius,
        seed=args.seed,
    )
    roles = Counter(assign_role(row) for row in reviews)
    role_targets = {
        role: sorted(
            row["target"] for row in reviews if assign_role(row) == role
        )
        for role in sorted(roles)
    }
    high_confidence = sorted(
        row["target"]
        for row in reviews
        if assign_role(row) == "confirmatory" and row["confidence"] == "high"
    )
    sample_counts = Counter(
        (row["role"], row["target"], row["period"]) for row in occurrences
    )
    summary = {
        "seed": args.seed,
        "max_per_period": args.max_per_period,
        "context_radius": args.context_radius,
        "number_of_targets": len(reviews),
        "number_of_occurrences": len(occurrences),
        "role_counts": dict(sorted(roles.items())),
        "targets_by_role": role_targets,
        "high_confidence_confirmatory_targets": high_confidence,
        "sample_counts": [
            {
                "role": role,
                "target": target,
                "period": period,
                "count": count,
            }
            for (role, target, period), count in sorted(sample_counts.items())
        ],
        "primary_metric": (
            "Spearman correlation between per-target sense-distribution JSD "
            "and SemEval graded gold on confirmatory targets"
        ),
        "decision_rule": (
            "GO iff primary Spearman is positive and a two-sided target-label "
            "permutation test has p < 0.05."
        ),
        "sensitivity_analysis": (
            "Repeat the primary metric on high-confidence confirmatory targets."
        ),
        "diagnostics": [
            "partial_diagnostic targets never enter the primary score",
            "monosemous controls never enter the primary score",
            "binary ROC-AUC and average precision are secondary",
        ],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "occurrences.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(occurrences[0]))
        writer.writeheader()
        writer.writerows(occurrences)
    (args.output_dir / "frozen_design.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage-review", type=Path, required=True)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-per-period", type=int, default=100)
    parser.add_argument("--context-radius", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260613)
    return parser


if __name__ == "__main__":
    prepare(build_parser().parse_args())
