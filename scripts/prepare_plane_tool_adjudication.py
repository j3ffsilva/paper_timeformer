#!/usr/bin/env python3
"""Prepare and consolidate the blind adjudication of ambiguous plane contexts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Iterable


ALLOWED_LABELS = ("tool", "geometry", "aircraft", "botanical", "unclear")
ALLOWED_CONFIDENCE = ("low", "medium", "high")
ANNOTATION_FIELDS = ("item_id", "context", "label", "confidence", "notes")
MANIFEST_FIELDS = ("item_id", "corpus", "occurrence_index", "context")

ANNOTATOR_INSTRUCTIONS = """# Blind adjudication of `plane`

Annotate each context independently. The target occurrence is marked as
`[plane]`. Do not consult model predictions, another annotator, or project
results before completing the sheet.

Use exactly one of these labels:

- `tool`: the woodworking tool used to smooth or shape wood;
- `geometry`: a flat surface or mathematical/physical plane, including an
  inclined plane;
- `aircraft`: an airplane;
- `botanical`: a plane tree;
- `unclear`: the context is insufficient, belongs to another sense, or cannot
  be assigned safely to one of the four senses above.

Use `low`, `medium`, or `high` in `confidence`. The `notes` field is optional.
Do not change `item_id` or `context`.
"""


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(
    path: Path, fieldnames: Iterable[str], rows: Iterable[dict[str, str]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_package(
    predictions_path: Path,
    output_dir: Path,
    *,
    seed: int = 20260613,
    corpus: str = "1810-1860",
    heuristic_label: str = "tool",
) -> int:
    rows = read_csv(predictions_path)
    required = {"corpus", "occurrence_index", "gold_sense", "context"}
    missing = required.difference(rows[0] if rows else ())
    if missing:
        raise ValueError(f"Predictions file is missing columns: {sorted(missing)}")

    selected = [
        row
        for row in rows
        if row["corpus"] == corpus and row["gold_sense"] == heuristic_label
    ]
    if not selected:
        raise ValueError("No occurrences matched the adjudication selection rule")

    selected.sort(key=lambda row: int(row["occurrence_index"]))
    rng = random.Random(seed)
    rng.shuffle(selected)

    manifest = []
    annotation_rows = []
    for number, row in enumerate(selected, start=1):
        item_id = f"PT{number:02d}"
        context = row["context"].replace("[plane]", "[plane]")
        manifest.append(
            {
                "item_id": item_id,
                "corpus": row["corpus"],
                "occurrence_index": row["occurrence_index"],
                "context": context,
            }
        )
        annotation_rows.append(
            {
                "item_id": item_id,
                "context": context,
                "label": "",
                "confidence": "",
                "notes": "",
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "README_ANNOTATORS.md").write_text(
        ANNOTATOR_INSTRUCTIONS, encoding="utf-8"
    )
    write_csv(output_dir / "manifest.csv", MANIFEST_FIELDS, manifest)
    write_csv(output_dir / "annotator_a.csv", ANNOTATION_FIELDS, annotation_rows)

    annotation_rows_b = annotation_rows.copy()
    random.Random(seed + 1).shuffle(annotation_rows_b)
    write_csv(output_dir / "annotator_b.csv", ANNOTATION_FIELDS, annotation_rows_b)

    metadata = {
        "source_file": str(predictions_path),
        "source_sha256": file_sha256(predictions_path),
        "generation_seed": seed,
        "number_of_items": len(selected),
        "allowed_labels": list(ALLOWED_LABELS),
        "allowed_confidence": list(ALLOWED_CONFIDENCE),
        "blinding": {
            "prediction_columns_in_forms": False,
            "heuristic_label_in_forms": False,
            "annotator_orders_identical": False,
        },
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return len(selected)


def validate_annotations(
    rows: list[dict[str, str]], *, source_name: str
) -> dict[str, dict[str, str]]:
    by_id: dict[str, dict[str, str]] = {}
    for row in rows:
        item_id = row.get("item_id", "").strip()
        label = row.get("label", "").strip()
        confidence = row.get("confidence", "").strip()
        if not item_id:
            raise ValueError(f"{source_name}: blank item_id")
        if item_id in by_id:
            raise ValueError(f"{source_name}: duplicate item_id {item_id}")
        if label not in ALLOWED_LABELS:
            raise ValueError(f"{source_name}: invalid label for {item_id}: {label!r}")
        if confidence and confidence not in ALLOWED_CONFIDENCE:
            raise ValueError(
                f"{source_name}: invalid confidence for {item_id}: {confidence!r}"
            )
        by_id[item_id] = row
    return by_id


def cohen_kappa(labels_a: list[str], labels_b: list[str]) -> float:
    if len(labels_a) != len(labels_b) or not labels_a:
        raise ValueError("Kappa requires two non-empty label lists of equal length")
    total = len(labels_a)
    observed = sum(a == b for a, b in zip(labels_a, labels_b)) / total
    counts_a = Counter(labels_a)
    counts_b = Counter(labels_b)
    expected = sum(
        (counts_a[label] / total) * (counts_b[label] / total)
        for label in ALLOWED_LABELS
    )
    if math.isclose(expected, 1.0):
        return 1.0 if math.isclose(observed, 1.0) else 0.0
    return (observed - expected) / (1.0 - expected)


def summarize_annotations(
    annotator_a_path: Path,
    annotator_b_path: Path,
    output_dir: Path,
) -> dict[str, float | int]:
    rows_a = validate_annotations(read_csv(annotator_a_path), source_name="annotator A")
    rows_b = validate_annotations(read_csv(annotator_b_path), source_name="annotator B")
    if set(rows_a) != set(rows_b):
        missing_a = sorted(set(rows_b).difference(rows_a))
        missing_b = sorted(set(rows_a).difference(rows_b))
        raise ValueError(
            f"Annotation item sets differ; missing A={missing_a}, missing B={missing_b}"
        )

    item_ids = sorted(rows_a)
    labels_a = [rows_a[item_id]["label"].strip() for item_id in item_ids]
    labels_b = [rows_b[item_id]["label"].strip() for item_id in item_ids]
    agreements = sum(a == b for a, b in zip(labels_a, labels_b))
    summary: dict[str, float | int] = {
        "number_of_items": len(item_ids),
        "agreements": agreements,
        "disagreements": len(item_ids) - agreements,
        "raw_agreement": agreements / len(item_ids),
        "cohen_kappa": cohen_kappa(labels_a, labels_b),
    }

    adjudication_rows = []
    for item_id in item_ids:
        row_a = rows_a[item_id]
        row_b = rows_b[item_id]
        label_a = row_a["label"].strip()
        label_b = row_b["label"].strip()
        adjudication_rows.append(
            {
                "item_id": item_id,
                "context": row_a["context"],
                "annotator_a_label": label_a,
                "annotator_b_label": label_b,
                "consensus_label": label_a if label_a == label_b else "",
                "adjudicator_notes": "",
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "agreement_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    write_csv(
        output_dir / "adjudication.csv",
        (
            "item_id",
            "context",
            "annotator_a_label",
            "annotator_b_label",
            "consensus_label",
            "adjudicator_notes",
        ),
        adjudication_rows,
    )
    return summary


def summarize_single_annotator(
    annotation_path: Path,
    manifest_path: Path,
    predictions_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    annotations = validate_annotations(
        read_csv(annotation_path), source_name="single annotator"
    )
    manifest_rows = read_csv(manifest_path)
    manifest = {row["item_id"]: row for row in manifest_rows}
    if set(annotations) != set(manifest):
        raise ValueError("Annotation and manifest item sets differ")

    prediction_rows = read_csv(predictions_path)
    predictions = {
        (row["corpus"], row["occurrence_index"]): row for row in prediction_rows
    }

    label_counts = Counter()
    confidence_counts = Counter()
    confusion = Counter()
    comparison_rows = []
    correct = 0
    evaluable = 0
    tool_correct = 0
    tool_total = 0

    for item_id in sorted(annotations):
        annotation = annotations[item_id]
        source = manifest[item_id]
        key = (source["corpus"], source["occurrence_index"])
        if key not in predictions:
            raise ValueError(f"No model prediction found for {item_id}: {key}")
        model_prediction = predictions[key]["prediction"].strip()
        human_label = annotation["label"].strip()
        confidence = annotation["confidence"].strip()
        is_evaluable = human_label != "unclear"
        is_correct = is_evaluable and model_prediction == human_label

        label_counts[human_label] += 1
        if confidence:
            confidence_counts[confidence] += 1
        confusion[(human_label, model_prediction)] += 1
        evaluable += int(is_evaluable)
        correct += int(is_correct)
        tool_total += int(human_label == "tool")
        tool_correct += int(human_label == "tool" and is_correct)
        comparison_rows.append(
            {
                "item_id": item_id,
                "human_label": human_label,
                "confidence": confidence,
                "model_prediction": model_prediction,
                "correct": str(is_correct).lower() if is_evaluable else "",
                "context": annotation["context"],
            }
        )

    summary: dict[str, object] = {
        "annotation_design": "single_blind_annotator",
        "number_of_items": len(annotations),
        "label_counts": dict(sorted(label_counts.items())),
        "confidence_counts": dict(sorted(confidence_counts.items())),
        "heuristic_tool_precision": tool_total / len(annotations),
        "model_accuracy_excluding_unclear": correct / evaluable if evaluable else None,
        "model_correct_excluding_unclear": correct,
        "model_evaluable_items": evaluable,
        "model_tool_accuracy": tool_correct / tool_total if tool_total else None,
        "model_tool_correct": tool_correct,
        "model_tool_items": tool_total,
        "confusion": [
            {
                "human_label": human_label,
                "model_prediction": prediction,
                "count": count,
            }
            for (human_label, prediction), count in sorted(confusion.items())
        ],
        "limitations": [
            "No inter-annotator agreement can be estimated.",
            "The annotator used Google Translate to understand historical English.",
            "Results are diagnostic and do not replace the frozen original Gate 1.",
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "single_annotator_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    write_csv(
        output_dir / "single_annotator_model_comparison.csv",
        (
            "item_id",
            "human_label",
            "confidence",
            "model_prediction",
            "correct",
            "context",
        ),
        comparison_rows,
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Generate blind annotation forms")
    prepare.add_argument("--predictions", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--seed", type=int, default=20260613)
    prepare.add_argument("--corpus", default="1810-1860")
    prepare.add_argument("--heuristic-label", default="tool")

    summarize = subparsers.add_parser(
        "summarize", help="Validate annotations and prepare adjudication"
    )
    summarize.add_argument("--annotator-a", type=Path, required=True)
    summarize.add_argument("--annotator-b", type=Path, required=True)
    summarize.add_argument("--output-dir", type=Path, required=True)

    single = subparsers.add_parser(
        "single-summary",
        help="Compare one completed blind annotation sheet with model predictions",
    )
    single.add_argument("--annotations", type=Path, required=True)
    single.add_argument("--manifest", type=Path, required=True)
    single.add_argument("--predictions", type=Path, required=True)
    single.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare":
        count = prepare_package(
            args.predictions,
            args.output_dir,
            seed=args.seed,
            corpus=args.corpus,
            heuristic_label=args.heuristic_label,
        )
        print(f"Prepared {count} blind adjudication items in {args.output_dir}")
    elif args.command == "summarize":
        summary = summarize_annotations(
            args.annotator_a, args.annotator_b, args.output_dir
        )
        print(json.dumps(summary, indent=2))
    else:
        summary = summarize_single_annotator(
            args.annotations,
            args.manifest,
            args.predictions,
            args.output_dir,
        )
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
