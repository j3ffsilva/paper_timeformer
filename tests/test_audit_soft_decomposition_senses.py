import unittest

import numpy as np

from scripts.audit_soft_decomposition_senses import sense_contributions


class AuditSoftDecompositionSensesTest(unittest.TestCase):
    def test_per_sense_shares_reconstruct_total_share(self):
        probabilities = np.asarray(
            [
                [1.0, 0.0],
                [0.8, 0.2],
                [0.2, 0.8],
                [0.0, 1.0],
            ]
        )
        vectors = np.asarray(
            [[1.0, 0.0], [0.9, 0.1], [0.1, 0.9], [0.0, 1.0]]
        )
        periods = np.asarray(["d0", "d0", "d1", "d1"])
        result = sense_contributions(probabilities, vectors, periods)
        self.assertAlmostEqual(
            float(
                result["composition_share"].sum()
                + result["drift_share"].sum()
            ),
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
