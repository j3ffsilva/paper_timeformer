import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import torch

from scripts.evaluate_external_wsd_plane import (
    PLANE_SENSES,
    bootstrap_accuracy,
    bootstrap_macro_accuracy,
    combine_hidden_states,
    gate_decision,
    load_plane_vectors,
    load_layer_weights,
    pool_target_subwords,
)


class ExternalWSDPlaneTests(unittest.TestCase):
    def test_load_plane_vectors_filters_and_normalizes_zip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vectors.zip"
            with zipfile.ZipFile(path, "w") as archive:
                lines = [
                    f"{sensekey} 3 4\n"
                    for sensekey in PLANE_SENSES
                ]
                lines.append("other%1:00:00:: 1 0\n")
                archive.writestr(
                    "lmms-sp-wsd.bert-large-cased.vectors.txt",
                    "".join(lines),
                )
            vectors = load_plane_vectors(path)
        self.assertEqual(set(vectors), set(PLANE_SENSES))
        np.testing.assert_allclose(vectors["plane%1:06:00::"], [0.6, 0.8])

    def test_combine_hidden_states_uses_reverse_lmms_order(self):
        hidden_states = (
            torch.full((1, 2, 1), 1.0),
            torch.full((1, 2, 1), 2.0),
            torch.full((1, 2, 1), 4.0),
        )
        combined = combine_hidden_states(
            hidden_states,
            torch.tensor([0.5, 0.3, 0.2]),
        )
        torch.testing.assert_close(combined, torch.full((1, 2, 1), 2.8))

    def test_embedded_layer_weights_are_valid(self):
        weights = load_layer_weights(None)
        self.assertEqual(len(weights), 25)
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=3)

    def test_pool_target_subwords_averages_aligned_pieces(self):
        embeddings = torch.tensor([[0.0], [2.0], [4.0], [8.0]])
        pooled = pool_target_subwords(
            embeddings,
            [None, 0, 0, 1],
            target_index=0,
        )
        torch.testing.assert_close(pooled, torch.tensor([3.0]))

    def test_bootstrap_accuracy_is_deterministic(self):
        values = np.array([1, 1, 0, 1], dtype=np.float32)
        left = bootstrap_accuracy(values, n_bootstrap=200, seed=7)
        right = bootstrap_accuracy(values, n_bootstrap=200, seed=7)
        self.assertEqual(left, right)
        self.assertEqual(left["accuracy"], 0.75)

    def test_gate_decision_applies_predefined_thresholds(self):
        summaries = [
            {
                "corpus": "1810-1860",
                "gold_sense": "geometry",
                "accuracy": 0.75,
                "ci_95_low": 0.70,
            },
            {
                "corpus": "1810-1860",
                "gold_sense": "tool",
                "accuracy": 0.70,
                "ci_95_low": 0.40,
            },
            {
                "corpus": "1960-2010",
                "gold_sense": "aircraft",
                "accuracy": 0.80,
                "ci_95_low": 0.75,
            },
        ]
        rows = [
            {
                "corpus": "1810-1860",
                "anchor": True,
                "prediction": "geometry",
            },
            {
                "corpus": "1960-2010",
                "anchor": False,
                "prediction": "aircraft",
            },
        ]
        macro = {"accuracy": 0.75, "ci_95_low": 0.60}
        self.assertTrue(gate_decision(summaries, rows, macro)["passed"])

    def test_macro_accuracy_weights_senses_equally(self):
        rows = [
            *[
                {
                    "corpus": "1810-1860",
                    "gold_sense": "geometry",
                    "prediction": "geometry",
                }
                for _ in range(10)
            ],
            {
                "corpus": "1810-1860",
                "gold_sense": "tool",
                "prediction": "other",
            },
            {
                "corpus": "1960-2010",
                "gold_sense": "aircraft",
                "prediction": "aircraft",
            },
        ]
        result = bootstrap_macro_accuracy(
            rows,
            periods=["1810-1860", "1960-2010"],
            n_bootstrap=50,
            seed=3,
        )
        self.assertAlmostEqual(result["accuracy"], 2.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
