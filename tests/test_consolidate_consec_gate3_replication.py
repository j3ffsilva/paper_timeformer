import unittest

from scripts.evaluate_consec_gate3 import partial_spearman, spearman


class ConsolidateConsecGate3ReplicationTests(unittest.TestCase):
    def test_shared_metrics_are_available(self):
        self.assertAlmostEqual(spearman([1, 2, 3], [3, 2, 1]), -1.0)
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
