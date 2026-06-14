import unittest

from scripts.integrate_consec_timeformer import (
    correlation_summary,
    partial_spearman_multi,
)


class IntegrateConsecTimeformerTest(unittest.TestCase):
    def test_partial_spearman_removes_shared_control(self):
        control = [1, 2, 3, 4, 5, 6]
        values_a = [1, 3, 2, 6, 4, 5]
        values_b = [2, 1, 4, 3, 6, 5]
        value = partial_spearman_multi(
            values_a,
            values_b,
            [[item, item * item] for item in control],
        )
        self.assertTrue(-1.0 <= value <= 1.0)

    def test_correlation_summary_detects_identical_ranking(self):
        result = correlation_summary(
            [1, 2, 3, 4, 5],
            [10, 20, 30, 40, 50],
            n_bootstrap=100,
            n_permutations=200,
            seed=7,
        )
        self.assertEqual(result["spearman"], 1.0)
        self.assertLess(result["permutation_p_two_sided"], 0.05)


if __name__ == "__main__":
    unittest.main()
