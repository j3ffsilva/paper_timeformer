"""Shared CSV/TSV I/O helpers used across the `research/` evaluation scripts."""

from __future__ import annotations

import csv
from pathlib import Path


def read_truth(path: Path) -> dict[str, dict[str, float]]:
    """Read a SemEval-2020-style truth TSV with `target`, `binary`, `graded` columns."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {
            row["target"]: {"binary": float(row["binary"]), "graded": float(row["graded"])}
            for row in csv.DictReader(handle, delimiter="\t")
        }


def read_scores(path: Path, *, comparison: str, score_column: str) -> dict[str, float]:
    """Read a scores CSV, keeping the max `score_column` per target for one `comparison`."""
    scores: dict[str, float] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["comparison"] != comparison:
                continue
            target = row["target"]
            score = float(row[score_column])
            if target not in scores or score > scores[target]:
                scores[target] = score
    return scores


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
