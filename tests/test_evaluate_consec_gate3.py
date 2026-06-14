import unittest

from scripts.evaluate_consec_gate3 import (
    average_precision,
    partial_spearman,
    jensen_shannon,
    prepare_context,
    roc_auc,
    spearman,
)


class EvaluateConsecGate3Tests(unittest.TestCase):
    def test_jensen_shannon(self):
        self.assertAlmostEqual(jensen_shannon([1, 0], [1, 0]), 0.0)
        self.assertAlmostEqual(
            jensen_shannon([1, 0], [0, 1]), 0.6931471805599453
        )

    def test_prepare_context_verb(self):
        tokens, index, lemma = prepare_context("they [tip] the table", "tip_vb")
        self.assertEqual(tokens, ["they", "tip", "the", "table"])
        self.assertEqual(index, 1)
        self.assertEqual(lemma, "tip")

    def test_metrics(self):
        self.assertAlmostEqual(spearman([1, 2, 3], [4, 5, 6]), 1.0)
        self.assertAlmostEqual(roc_auc([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8]), 1.0)
        self.assertAlmostEqual(
            average_precision([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8]), 1.0
        )
        self.assertGreater(
            partial_spearman(
                [1, 2, 4, 5],
                [1, 3, 4, 6],
                [1, 1, 2, 2],
            ),
            0.9,
        )


if __name__ == "__main__":
    unittest.main()
