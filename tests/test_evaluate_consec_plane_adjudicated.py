import unittest

from scripts.evaluate_consec_plane_adjudicated import (
    bootstrap_accuracy,
    compute_summary,
    gate_summary,
    prepare_context,
)


class ConsecPlaneAdjudicatedTests(unittest.TestCase):
    def test_prepare_context(self):
        tokens, target_index = prepare_context("use the [plane] on wood")
        self.assertEqual(tokens, ["use", "the", "plane", "on", "wood"])
        self.assertEqual(target_index, 2)

    def test_compute_summary(self):
        summary = compute_summary(
            [
                {"human_label": "tool", "prediction": "tool"},
                {"human_label": "tool", "prediction": "geometry"},
                {"human_label": "unclear", "prediction": "tool"},
            ]
        )
        self.assertEqual(summary["model_accuracy_excluding_unclear"], 0.5)
        self.assertEqual(summary["model_tool_accuracy"], 0.5)
        self.assertEqual(summary["model_evaluable_items"], 2)

    def test_bootstrap_accuracy(self):
        metric = bootstrap_accuracy([True, True, False, True], 1000, 3)
        self.assertEqual(metric["n"], 4)
        self.assertEqual(metric["accuracy"], 0.75)

    def test_gate_summary_passes_clear_case(self):
        rows = []
        for corpus, label, count in (
            ("1810-1860", "geometry", 8),
            ("1810-1860", "tool", 8),
            ("1960-2010", "aircraft", 8),
        ):
            for index in range(count):
                rows.append(
                    {
                        "corpus": corpus,
                        "original_gold_sense": label,
                        "prediction": label,
                        "anchor": "True" if label == "geometry" and index == 0 else "False",
                    }
                )
        summary = gate_summary(rows, "original_gold_sense", 1000, 7)
        self.assertTrue(summary["passed"])


if __name__ == "__main__":
    unittest.main()
