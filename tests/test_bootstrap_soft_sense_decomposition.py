import unittest

import numpy as np

from scripts.bootstrap_soft_sense_decomposition import (
    classify_interval,
    stratified_bootstrap_indices,
)


class BootstrapSoftSenseDecompositionTest(unittest.TestCase):
    def test_classify_interval(self):
        self.assertEqual(classify_interval(0.1, 0.4), "robust_positive")
        self.assertEqual(classify_interval(-0.4, -0.1), "robust_negative")
        self.assertEqual(classify_interval(-0.1, 0.2), "uncertain")

    def test_stratified_indices_preserve_period_counts(self):
        periods = np.asarray(["d0", "d0", "d1", "d1", "d1"])
        samples = stratified_bootstrap_indices(periods, 10, 5)
        self.assertEqual(samples.shape, (10, 5))
        for sample in samples:
            selected = periods[sample]
            self.assertEqual(int(np.sum(selected == "d0")), 2)
            self.assertEqual(int(np.sum(selected == "d1")), 3)


if __name__ == "__main__":
    unittest.main()
