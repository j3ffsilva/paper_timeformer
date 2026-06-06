import unittest

import torch

from scripts.report_temporal_relational_neighborhoods import (
    neighborhood_rows,
    rank_descending,
    standardize,
)


class TemporalRelationalNeighborhoodTests(unittest.TestCase):
    def test_standardize_has_zero_mean_and_unit_scale(self):
        values = torch.tensor([1.0, 2.0, 4.0])
        standardized = standardize(values)
        self.assertTrue(
            torch.isclose(standardized.mean(), torch.tensor(0.0), atol=1e-6)
        )
        self.assertTrue(
            torch.isclose(
                standardized.std(unbiased=False), torch.tensor(1.0), atol=1e-6
            )
        )

    def test_rank_descending_assigns_one_to_largest_value(self):
        ranks = rank_descending(torch.tensor([0.2, 0.8, 0.4]))
        self.assertEqual(ranks.tolist(), [3, 1, 2])

    def test_neighborhood_rows_report_relative_gain_and_rank_gain(self):
        rows = neighborhood_rows(
            target="word_nn",
            references=["old", "new", "stable"],
            before=torch.tensor([0.9, 0.1, 0.5]),
            after=torch.tensor([0.1, 0.9, 0.5]),
        )
        by_reference = {row["reference"]: row for row in rows}
        self.assertGreater(by_reference["new"]["delta_z"], 0)
        self.assertEqual(by_reference["new"]["rank_gain"], 2)
        self.assertLess(by_reference["old"]["delta_z"], 0)
        self.assertEqual(by_reference["old"]["rank_gain"], -2)


if __name__ == "__main__":
    unittest.main()
