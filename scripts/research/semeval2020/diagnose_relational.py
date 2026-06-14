#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.research.common.io import read_scores, read_truth  # noqa: E402


def entropy(distribution: torch.Tensor) -> float:
    values = distribution.double()
    values = values / values.sum().clamp_min(torch.finfo(values.dtype).eps)
    return float(-(values * values.clamp_min(torch.finfo(values.dtype).eps).log()).sum())


def normalized_entropy(distribution: torch.Tensor) -> float:
    if distribution.numel() <= 1:
        return 0.0
    return entropy(distribution) / math.log(distribution.numel())


def ppmi_distribution(profile: torch.Tensor) -> torch.Tensor:
    positive = profile.clamp(min=0.0)
    mass = positive.sum().clamp_min(torch.finfo(positive.dtype).eps)
    return positive / mass


def load_profile_entropies(profile_dir: Path) -> dict[str, list[float]]:
    profile_paths = sorted(profile_dir.glob("t*.pt"))
    if not profile_paths:
        return {}
    entropies: dict[str, list[float]] = {}
    for path in profile_paths:
        profile = torch.load(path, map_location="cpu", weights_only=True)
        targets = profile["targets"]
        distributions = profile.get("distributions")
        if distributions is None and profile.get("pmi_profiles") is not None:
            distributions = torch.stack(
                [ppmi_distribution(row) for row in profile["pmi_profiles"]]
            )
        elif distributions is None and profile.get("full_distributions") is not None:
            distributions = profile["full_distributions"]
        if distributions is None:
            continue
        for index, target in enumerate(targets):
            entropies.setdefault(target, []).append(normalized_entropy(distributions[index]))
    return entropies


def safe_spearman(rows: list[dict], left: str, right: str) -> dict[str, float | None]:
    pairs = [
        (float(row[left]), float(row[right]))
        for row in rows
        if row.get(left) not in (None, "") and row.get(right) not in (None, "")
    ]
    if len(pairs) < 2:
        return {"rho": None, "p": None}
    x = np.array([pair[0] for pair in pairs], dtype=float)
    y = np.array([pair[1] for pair in pairs], dtype=float)
    if len(set(x.tolist())) < 2 or len(set(y.tolist())) < 2:
        return {"rho": None, "p": None}
    rho, p_value = spearmanr(x, y)
    return {"rho": float(rho), "p": float(p_value)}


def read_metadata(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_rows(
    truth: dict[str, dict[str, float]],
    scores: dict[str, float],
    metadata: dict,
    entropies: dict[str, list[float]],
) -> list[dict]:
    rows = []
    coverage = metadata.get("target_coverage", {})
    for target in sorted(set(truth) & set(scores)):
        counts = [int(value) for value in coverage.get(target, [])]
        entropy_values = entropies.get(target, [])
        row = {
            "target": target,
            "predicted_score": scores[target],
            "binary": truth[target]["binary"],
            "graded": truth[target]["graded"],
            "freq_t0": counts[0] if len(counts) > 0 else "",
            "freq_t1": counts[1] if len(counts) > 1 else "",
            "freq_min": min(counts) if counts else "",
            "freq_max": max(counts) if counts else "",
            "freq_sum": sum(counts) if counts else "",
            "freq_abs_delta": abs(counts[1] - counts[0]) if len(counts) > 1 else "",
            "freq_ratio": (max(counts) / max(min(counts), 1)) if counts else "",
            "entropy_t0": entropy_values[0] if len(entropy_values) > 0 else "",
            "entropy_t1": entropy_values[1] if len(entropy_values) > 1 else "",
            "entropy_abs_delta": (
                abs(entropy_values[1] - entropy_values[0]) if len(entropy_values) > 1 else ""
            ),
        }
        rows.append(row)
    rows.sort(key=lambda row: row["predicted_score"], reverse=True)
    return rows


def group_means(rows: list[dict], key: str) -> dict[str, dict[str, float]]:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(str(row[key]), []).append(row)
    result = {}
    for value, group in groups.items():
        result[value] = {
            "n": len(group),
            "predicted_score_mean": float(np.mean([float(row["predicted_score"]) for row in group])),
            "graded_mean": float(np.mean([float(row["graded"]) for row in group])),
            "freq_min_mean": float(np.mean([float(row["freq_min"]) for row in group if row["freq_min"] != ""])),
            "entropy_abs_delta_mean": float(
                np.mean([float(row["entropy_abs_delta"]) for row in group if row["entropy_abs_delta"] != ""])
            ),
        }
    return result


def summarize(rows: list[dict], args: argparse.Namespace) -> dict:
    correlations = {}
    for field in [
        "graded",
        "binary",
        "freq_t0",
        "freq_t1",
        "freq_min",
        "freq_sum",
        "freq_abs_delta",
        "freq_ratio",
        "entropy_t0",
        "entropy_t1",
        "entropy_abs_delta",
    ]:
        correlations[f"predicted_vs_{field}"] = safe_spearman(rows, "predicted_score", field)
    return {
        "n_targets": len(rows),
        "predictions": str(args.predictions),
        "truth": str(args.truth),
        "metadata": str(args.metadata),
        "profile_dir": str(args.profile_dir),
        "comparison": args.comparison,
        "score_column": args.score_column,
        "correlations": correlations,
        "group_means_by_binary": group_means(rows, "binary"),
        "top_10": rows[:10],
        "bottom_10": rows[-10:],
    }


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "target",
        "predicted_score",
        "binary",
        "graded",
        "freq_t0",
        "freq_t1",
        "freq_min",
        "freq_max",
        "freq_sum",
        "freq_abs_delta",
        "freq_ratio",
        "entropy_t0",
        "entropy_t1",
        "entropy_abs_delta",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose SemEval relational scores against frequency and entropy.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--truth", type=Path, default=Path("data/processed/semeval2020_task1/eng_lemma/truth.tsv"))
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/processed/semeval2020_task1/eng_lemma/metadata.json"),
    )
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--comparison", default="from_t0")
    parser.add_argument(
        "--score-column",
        choices=["pmi_cosine", "ppmi_jsd"],
        default="pmi_cosine",
    )
    args = parser.parse_args()

    truth = read_truth(args.truth)
    scores = read_scores(args.predictions, comparison=args.comparison, score_column=args.score_column)
    metadata = read_metadata(args.metadata)
    entropies = load_profile_entropies(args.profile_dir)
    rows = build_rows(truth, scores, metadata, entropies)
    summary = summarize(rows, args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.output_dir / "diagnostics.csv")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["correlations"], indent=2))


if __name__ == "__main__":
    main()
