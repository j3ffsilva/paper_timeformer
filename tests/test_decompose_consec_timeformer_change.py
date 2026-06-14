import unittest

import numpy as np

from scripts.decompose_consec_timeformer_change import decomposition_components


class SoftSenseDecompositionTest(unittest.TestCase):
    def test_symmetric_decomposition_reconstructs_total(self):
        probabilities = np.asarray(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
            ]
        )
        vectors = np.asarray(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
            ]
        )
        periods = np.asarray(["d0", "d0", "d0", "d1", "d1", "d1"])
        result = decomposition_components(probabilities, vectors, periods)
        np.testing.assert_allclose(
            result["total"],
            result["composition"] + result["drift"],
            atol=1e-12,
        )
        self.assertAlmostEqual(result["composition_share"], 1.0)
        self.assertAlmostEqual(result["drift_share"], 0.0)
        self.assertLess(result["reconstruction_error"], 1e-12)


if __name__ == "__main__":
    unittest.main()
