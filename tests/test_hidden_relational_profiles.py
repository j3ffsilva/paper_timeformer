import unittest

import torch

from scripts.evaluate_hidden_relational_profiles import (
    average_pairwise_cosine_distance,
    relational_energy_distance,
    relational_profiles,
    relational_separation_ratio,
)


class HiddenRelationalProfilesTest(unittest.TestCase):
    def test_relational_profiles_are_invariant_to_shared_rotation(self) -> None:
        points = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        rotation = torch.tensor([[0.0, -1.0], [1.0, 0.0]])

        before = relational_profiles(points, [0], [1, 2])
        after = relational_profiles(points @ rotation, [0], [1, 2])

        self.assertTrue(torch.allclose(before, after, atol=1e-6))

    def test_distribution_scores_are_zero_for_identical_profiles(self) -> None:
        profiles = torch.nn.functional.normalize(
            torch.tensor([[1.0, 0.0], [0.8, 0.2]]),
            dim=1,
        )

        self.assertAlmostEqual(
            average_pairwise_cosine_distance(profiles, profiles),
            0.0149287,
            places=5,
        )
        self.assertAlmostEqual(relational_energy_distance(profiles, profiles), 0.0, places=6)
        self.assertAlmostEqual(relational_separation_ratio(profiles, profiles), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
