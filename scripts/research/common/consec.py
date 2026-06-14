"""Shared helpers for working with ConSeC occurrence/prediction CSVs."""

from __future__ import annotations

import csv
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def prepare_context(context: str, target: str) -> tuple[list[str], int, str]:
    """Split a `[lemma]`-marked context into tokens, marker position and lemma.

    `target` is a sense-inventory key like `graft_nn`; the POS suffix
    (`_nn`/`_vb`/...) is stripped to recover the bracketed lemma marker.
    """
    lemma = target[:-3]
    marker = f"[{lemma}]"
    tokens = context.split()
    positions = [index for index, token in enumerate(tokens) if token == marker]
    if len(positions) != 1:
        raise ValueError(f"Expected one {marker} marker")
    tokens[positions[0]] = lemma
    return tokens, positions[0], lemma


def collect_unique_rows(prediction_files: list[Path]) -> tuple[list[dict], list[list[dict]]]:
    """Deduplicate `confirmatory` rows by `sample_id` across prediction files.

    Returns the deduplicated rows (sorted by `sample_id`) plus the raw
    per-file rows, so callers can also inspect non-confirmatory rows.
    """
    per_file = [read_csv(path) for path in prediction_files]
    unique: dict[str, dict] = {}
    for rows in per_file:
        for row in rows:
            if row["role"] != "confirmatory":
                continue
            sample_id = row["sample_id"]
            if sample_id in unique:
                previous = unique[sample_id]
                if (
                    previous["context"] != row["context"]
                    or previous["sense_probabilities"] != row["sense_probabilities"]
                ):
                    raise ValueError(f"Inconsistent duplicate: {sample_id}")
            else:
                unique[sample_id] = row
    return [unique[key] for key in sorted(unique)], per_file
