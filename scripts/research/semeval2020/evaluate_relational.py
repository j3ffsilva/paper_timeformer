#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.research.common.io import read_scores, read_truth  # noqa: E402


def best_f1(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    best = {"f1": 0.0, "threshold": float("nan")}
    for threshold in sorted(set(float(value) for value in y_score)):
        predicted = (y_score >= threshold).astype(int)
        score = f1_score(y_true, predicted, zero_division=0)
        if score > best["f1"]:
            best = {"f1": float(score), "threshold": threshold}
    return best


def evaluate(truth: dict[str, dict[str, float]], scores: dict[str, float]) -> tuple[dict, list[dict]]:
    shared_targets = sorted(set(truth) & set(scores))
    if not shared_targets:
        raise ValueError("No overlapping targets between truth and predicted scores")

    y_score = np.array([scores[target] for target in shared_targets], dtype=float)
    y_binary = np.array([truth[target]["binary"] for target in shared_targets], dtype=int)
    y_graded = np.array([truth[target]["graded"] for target in shared_targets], dtype=float)

    rho, rho_p = spearmanr(y_graded, y_score)
    result = {
        "n_targets": len(shared_targets),
        "spearman_graded": float(rho),
        "spearman_graded_p": float(rho_p),
        "roc_auc_binary": float(roc_auc_score(y_binary, y_score)) if len(set(y_binary.tolist())) > 1 else None,
        "average_precision_binary": (
            float(average_precision_score(y_binary, y_score)) if len(set(y_binary.tolist())) > 1 else None
        ),
        "best_f1_binary": best_f1(y_binary, y_score),
    }
    joined = [
        {
            "target": target,
            "predicted_score": scores[target],
            "binary": truth[target]["binary"],
            "graded": truth[target]["graded"],
        }
        for target in shared_targets
    ]
    joined.sort(key=lambda row: row["predicted_score"], reverse=True)
    return result, joined


def write_joined(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["target", "predicted_score", "binary", "graded"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate relational change scores against SemEval-2020 Task 1 gold.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--truth", type=Path, default=Path("data/processed/semeval2020_task1/eng_lemma/truth.tsv"))
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
    result, joined = evaluate(truth, scores)
    result = {
        **result,
        "predictions": str(args.predictions),
        "truth": str(args.truth),
        "comparison": args.comparison,
        "score_column": args.score_column,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_joined(joined, args.output_dir / "joined_scores.csv")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
