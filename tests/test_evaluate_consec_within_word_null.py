import unittest

from scripts.evaluate_consec_within_word_null import benjamini_hochberg


class EvaluateConsecWithinWordNullTests(unittest.TestCase):
    def test_benjamini_hochberg_is_monotone_in_rank_order(self):
        adjusted = benjamini_hochberg([0.01, 0.04, 0.03])
        self.assertAlmostEqual(adjusted[0], 0.03)
        self.assertAlmostEqual(adjusted[1], 0.04)
        self.assertAlmostEqual(adjusted[2], 0.04)


if __name__ == "__main__":
    unittest.main()
