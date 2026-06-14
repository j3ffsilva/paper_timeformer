import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.prepare_plane_tool_adjudication import (
    ANNOTATION_FIELDS,
    cohen_kappa,
    prepare_package,
    summarize_annotations,
    summarize_single_annotator,
)


def write_csv(path: Path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class PlaneToolAdjudicationTests(unittest.TestCase):
    def test_prepare_filters_and_blinds_forms(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            predictions = root / "predictions.csv"
            write_csv(
                predictions,
                (
                    "corpus",
                    "occurrence_index",
                    "gold_sense",
                    "prediction",
                    "margin",
                    "score_tool",
                    "context",
                ),
                [
                    {
                        "corpus": "1810-1860",
                        "occurrence_index": "7",
                        "gold_sense": "tool",
                        "prediction": "geometry",
                        "margin": "0.1",
                        "score_tool": "0.2",
                        "context": "a [plane] for wood",
                    },
                    {
                        "corpus": "1810-1860",
                        "occurrence_index": "8",
                        "gold_sense": "geometry",
                        "prediction": "geometry",
                        "margin": "0.9",
                        "score_tool": "0.0",
                        "context": "an inclined [plane]",
                    },
                ],
            )

            output = root / "package"
            count = prepare_package(predictions, output, seed=3)
            self.assertEqual(count, 1)
            with (output / "annotator_a.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(tuple(reader.fieldnames), ANNOTATION_FIELDS)
                rows = list(reader)
            self.assertEqual(rows[0]["context"], "a [plane] for wood")
            serialized = (output / "annotator_a.csv").read_text(encoding="utf-8")
            for forbidden in ("prediction", "margin", "score_tool", "gold_sense"):
                self.assertNotIn(forbidden, serialized)

            metadata = json.loads((output / "metadata.json").read_text())
            self.assertEqual(metadata["number_of_items"], 1)
            self.assertFalse(metadata["blinding"]["prediction_columns_in_forms"])

    def test_cohen_kappa(self):
        self.assertEqual(cohen_kappa(["tool", "tool"], ["tool", "tool"]), 1.0)
        value = cohen_kappa(
            ["tool", "tool", "geometry", "geometry"],
            ["tool", "geometry", "geometry", "tool"],
        )
        self.assertAlmostEqual(value, 0.0)

    def test_summarize_prefills_only_agreements(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fields = ANNOTATION_FIELDS
            write_csv(
                root / "a.csv",
                fields,
                [
                    {
                        "item_id": "PT01",
                        "context": "one",
                        "label": "tool",
                        "confidence": "high",
                        "notes": "",
                    },
                    {
                        "item_id": "PT02",
                        "context": "two",
                        "label": "geometry",
                        "confidence": "medium",
                        "notes": "",
                    },
                ],
            )
            write_csv(
                root / "b.csv",
                fields,
                [
                    {
                        "item_id": "PT02",
                        "context": "two",
                        "label": "tool",
                        "confidence": "low",
                        "notes": "",
                    },
                    {
                        "item_id": "PT01",
                        "context": "one",
                        "label": "tool",
                        "confidence": "high",
                        "notes": "",
                    },
                ],
            )
            summary = summarize_annotations(
                root / "a.csv", root / "b.csv", root / "summary"
            )
            self.assertEqual(summary["agreements"], 1)
            with (root / "summary" / "adjudication.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["consensus_label"], "tool")
            self.assertEqual(rows[1]["consensus_label"], "")

    def test_single_annotator_summary_compares_with_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            annotations = root / "annotations.csv"
            manifest = root / "manifest.csv"
            predictions = root / "predictions.csv"
            write_csv(
                annotations,
                ANNOTATION_FIELDS,
                [
                    {
                        "item_id": "PT01",
                        "context": "a [plane]",
                        "label": "tool",
                        "confidence": "high",
                        "notes": "",
                    },
                    {
                        "item_id": "PT02",
                        "context": "another [plane]",
                        "label": "unclear",
                        "confidence": "low",
                        "notes": "",
                    },
                ],
            )
            write_csv(
                manifest,
                ("item_id", "corpus", "occurrence_index", "context"),
                [
                    {
                        "item_id": "PT01",
                        "corpus": "D0",
                        "occurrence_index": "1",
                        "context": "a [plane]",
                    },
                    {
                        "item_id": "PT02",
                        "corpus": "D0",
                        "occurrence_index": "2",
                        "context": "another [plane]",
                    },
                ],
            )
            write_csv(
                predictions,
                ("corpus", "occurrence_index", "prediction"),
                [
                    {
                        "corpus": "D0",
                        "occurrence_index": "1",
                        "prediction": "tool",
                    },
                    {
                        "corpus": "D0",
                        "occurrence_index": "2",
                        "prediction": "geometry",
                    },
                ],
            )
            summary = summarize_single_annotator(
                annotations, manifest, predictions, root / "output"
            )
            self.assertEqual(summary["model_accuracy_excluding_unclear"], 1.0)
            self.assertEqual(summary["model_tool_accuracy"], 1.0)
            self.assertEqual(summary["heuristic_tool_precision"], 0.5)


if __name__ == "__main__":
    unittest.main()
