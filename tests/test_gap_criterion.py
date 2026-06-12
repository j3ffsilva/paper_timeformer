import unittest

import torch

from timeformers.gap_criterion import adjacent_gaps_valid, relative_gaps, select_gap_index
from timeformers.semantic_modes import (
    cohesion_svd,
    filter_support,
    filter_support_topn,
    select_num_modes,
)


class RelativeGapsTest(unittest.TestCase):
    def test_invariant_to_positive_rescaling(self) -> None:
        values = torch.tensor([0.41, 0.31, 0.18, 0.04, 0.03])
        scaled = values * 7.0
        torch.testing.assert_close(relative_gaps(values), relative_gaps(scaled))

    def test_rejects_unsorted_input(self) -> None:
        with self.assertRaises(ValueError):
            relative_gaps(torch.tensor([0.1, 0.5]))

    def test_rejects_non_positive_input(self) -> None:
        with self.assertRaises(ValueError):
            relative_gaps(torch.tensor([0.5, 0.0]))


class SelectGapIndexTest(unittest.TestCase):
    def test_clear_gap(self) -> None:
        # From the worked example in §8.3 of novo_perfil_relacional.md.
        values = torch.tensor([0.41, 0.31, 0.18, 0.04, 0.03])
        self.assertEqual(select_gap_index(values, gamma=0.3), 3)

    def test_no_gap_above_gamma_returns_none(self) -> None:
        values = torch.tensor([0.40, 0.39, 0.38, 0.37])
        self.assertIsNone(select_gap_index(values, gamma=0.3))

    def test_single_value_returns_none(self) -> None:
        self.assertIsNone(select_gap_index(torch.tensor([0.5]), gamma=0.3))


class AdjacentGapsValidTest(unittest.TestCase):
    def test_isolated_value_is_valid(self) -> None:
        values = torch.tensor([0.9, 0.1, 0.09])
        self.assertTrue(adjacent_gaps_valid(values, index=0, gamma=0.3))

    def test_quasi_degenerate_pair_is_invalid(self) -> None:
        values = torch.tensor([0.9, 0.5, 0.49, 0.1])
        # gap(0->1) = 0.44 > gamma; gap(1->2) = 0.02 < gamma; gap(2->3) large.
        self.assertFalse(adjacent_gaps_valid(values, index=1, gamma=0.3))
        self.assertFalse(adjacent_gaps_valid(values, index=2, gamma=0.3))


class FilterSupportTest(unittest.TestCase):
    def test_selects_head_of_positive_profile(self) -> None:
        profile = torch.tensor([0.05, 0.41, -0.2, 0.31, 0.18, 0.04, 0.03, 0.0])
        indices, tau = filter_support(profile, gamma=0.3)
        self.assertIsNotNone(tau)
        selected = set(int(i) for i in indices)
        self.assertEqual(selected, {1, 3, 4})  # values 0.41, 0.31, 0.18
        self.assertAlmostEqual(tau, 0.18)

    def test_no_structure_returns_empty(self) -> None:
        profile = torch.tensor([0.40, 0.39, 0.38, 0.37, -0.1])
        indices, tau = filter_support(profile, gamma=0.3)
        self.assertEqual(indices.numel(), 0)
        self.assertIsNone(tau)


class FilterSupportTopNTest(unittest.TestCase):
    def test_restricts_candidates_by_absolute_value_then_applies_gap(self) -> None:
        profile = torch.tensor([0.05, 0.41, -0.2, 0.31, 0.18, 0.04, 0.03, 0.0])
        # Top-4 by |value|: indices 1 (0.41), 3 (0.31), 2 (0.20), 4 (0.18).
        # Among their positive values [0.41, 0.31, 0.18], the gap criterion
        # picks the first 2.
        indices, tau = filter_support_topn(profile, gamma=0.3, top_n=4)
        self.assertIsNotNone(tau)
        selected = set(int(i) for i in indices)
        self.assertEqual(selected, {1, 3})
        self.assertAlmostEqual(tau, 0.31)

    def test_empty_profile_returns_none(self) -> None:
        indices, tau = filter_support_topn(torch.tensor([]), gamma=0.3, top_n=5)
        self.assertEqual(indices.numel(), 0)
        self.assertIsNone(tau)


class CohesionSvdTest(unittest.TestCase):
    def test_two_orthogonal_clusters_give_two_modes(self) -> None:
        torch.manual_seed(0)
        d = 8
        # Two well-separated clusters of unit vectors in orthogonal subspaces.
        cluster_a = torch.nn.functional.normalize(
            torch.eye(d)[0:1].repeat(4, 1) + 0.01 * torch.randn(4, d), dim=1
        )
        cluster_b = torch.nn.functional.normalize(
            torch.eye(d)[4:5].repeat(4, 1) + 0.01 * torch.randn(4, d), dim=1
        )
        embeddings = torch.cat([cluster_a, cluster_b], dim=0)
        # P_t(w)[v] high and roughly uniform for all v in V_w.
        profile = torch.full((8,), 0.5)

        eigenvalues, eigenvectors = cohesion_svd(profile, embeddings)
        self.assertTrue(torch.all(eigenvalues >= -1e-6))
        k = select_num_modes(eigenvalues, gamma=0.3)
        self.assertEqual(k, 2)
        # Each top mode should load on exactly one cluster.
        mode0 = eigenvectors[:, 0].clamp_min(0)
        mode1 = eigenvectors[:, 1].clamp_min(0)
        self.assertTrue(
            (mode0[:4].sum() > mode0[4:].sum()) != (mode1[:4].sum() > mode1[4:].sum())
        )

    def test_single_cluster_gives_one_mode_or_none(self) -> None:
        torch.manual_seed(1)
        d = 8
        embeddings = torch.nn.functional.normalize(
            torch.eye(d)[0:1].repeat(8, 1) + 0.01 * torch.randn(8, d), dim=1
        )
        profile = torch.full((8,), 0.5)
        eigenvalues, _ = cohesion_svd(profile, embeddings)
        k = select_num_modes(eigenvalues, gamma=0.3)
        self.assertIn(k, (None, 1))


if __name__ == "__main__":
    unittest.main()
