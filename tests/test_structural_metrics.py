import unittest

import torch

from timeformers.structural_metrics import (
    temporal_directional_advantage,
    temporal_directional_fidelity,
    temporal_event_metrics,
    temporal_path_metrics,
    temporal_shape_error,
)


def profiles_from_relation(values: list[float]) -> list[torch.Tensor]:
    profiles = []
    for value in values:
        profile = torch.eye(3)
        profile[0, 1] = value
        profile[1, 0] = value
        profiles.append(profile)
    return profiles


class StructuralMetricsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.gradual = profiles_from_relation([0.0, 0.25, 0.5, 0.75, 1.0])
        self.abrupt = profiles_from_relation([0.0, 0.0, 0.0, 1.0, 1.0])
        self.transient = profiles_from_relation([0.0, 0.0, 1.0, 1.0, 0.0])
        self.oscillating = profiles_from_relation([0.0, 1.0, 0.0, 1.0, 0.0])
        self.stable = profiles_from_relation([0.0] * 5)

    def test_same_destination_has_same_final_magnitude(self) -> None:
        gradual = temporal_path_metrics(self.gradual)
        abrupt = temporal_path_metrics(self.abrupt)

        self.assertTrue(torch.allclose(gradual["final_magnitude"], abrupt["final_magnitude"]))
        self.assertTrue(torch.allclose(gradual["path_length"], abrupt["path_length"]))
        self.assertTrue(
            torch.allclose(gradual["displacement_efficiency"], abrupt["displacement_efficiency"])
        )

    def test_oscillation_has_activity_without_final_displacement(self) -> None:
        gradual = temporal_path_metrics(self.gradual)
        oscillating = temporal_path_metrics(self.oscillating)

        self.assertGreater(float(oscillating["path_length"][0]), float(gradual["path_length"][0]))
        self.assertEqual(float(oscillating["final_magnitude"][0]), 0.0)
        self.assertEqual(float(oscillating["displacement_efficiency"][0]), 0.0)
        self.assertEqual(float(oscillating["recovery"][0]), 1.0)

    def test_transient_recovers_after_intermediate_peak(self) -> None:
        transient = temporal_path_metrics(self.transient)
        abrupt = temporal_path_metrics(self.abrupt)

        self.assertEqual(int(transient["peak_period"][0]), 2)
        self.assertEqual(float(transient["recovery"][0]), 1.0)
        self.assertEqual(float(abrupt["recovery"][0]), 0.0)

    def test_directional_fidelity_ignores_oracle_stable_steps(self) -> None:
        fidelity = temporal_directional_fidelity(self.abrupt, self.abrupt)

        self.assertEqual(int(fidelity["active_step_count"][0]), 1)
        self.assertEqual(int(fidelity["stable_step_count"][0]), 3)
        self.assertAlmostEqual(float(fidelity["step_fidelity"][0]), 1.0, places=6)
        self.assertAlmostEqual(float(fidelity["accumulated_fidelity"][0]), 1.0, places=6)

    def test_directional_advantage_compares_observed_with_placebo(self) -> None:
        advantage = temporal_directional_advantage(self.gradual, self.stable, self.gradual)

        self.assertAlmostEqual(float(advantage["step_fidelity_advantage"][0]), 1.0, places=6)
        self.assertAlmostEqual(float(advantage["accumulated_fidelity_advantage"][0]), 1.0, places=6)

    def test_shape_error_distinguishes_gradual_and_abrupt_paths(self) -> None:
        matching = temporal_shape_error(self.gradual, self.gradual)
        mismatched = temporal_shape_error(self.gradual, self.abrupt)

        self.assertTrue(torch.allclose(matching, torch.zeros_like(matching)))
        self.assertGreater(float(mismatched[0]), 0.0)

    def test_event_metrics_identify_abrupt_step(self) -> None:
        metrics = temporal_event_metrics(self.abrupt, self.abrupt)

        self.assertEqual(int(metrics["event_period"][0]), 3)
        self.assertEqual(int(metrics["observed_peak_period"][0]), 3)
        self.assertEqual(int(metrics["event_period_error"][0]), 0)
        self.assertAlmostEqual(float(metrics["event_concentration"][0]), 1.0, places=6)
        self.assertAlmostEqual(float(metrics["pre_event_drift"][0]), 0.0, places=6)
        self.assertAlmostEqual(float(metrics["post_event_drift"][0]), 0.0, places=6)
        self.assertAlmostEqual(float(metrics["event_fidelity"][0]), 1.0, places=6)

    def test_event_metrics_penalize_spread_change(self) -> None:
        metrics = temporal_event_metrics(self.gradual, self.abrupt)

        self.assertEqual(int(metrics["event_period"][0]), 3)
        self.assertLess(float(metrics["event_concentration"][0]), 1.0)
        self.assertGreater(float(metrics["pre_event_drift"][0]), 0.0)
        self.assertGreater(float(metrics["post_event_drift"][0]), 0.0)

    def test_invalid_profile_sequences_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            temporal_path_metrics([torch.eye(3)])
        with self.assertRaises(ValueError):
            temporal_path_metrics([torch.ones(2, 3), torch.ones(2, 4)])
        with self.assertRaises(ValueError):
            temporal_shape_error(self.gradual, self.gradual[:-1])

    def test_rectangular_target_anchor_profiles_are_supported(self) -> None:
        profiles = [
            torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            torch.tensor([[0.5, 0.0, 0.0], [0.0, -0.5, 0.0]]),
            torch.tensor([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]),
        ]
        metrics = temporal_path_metrics(profiles)

        self.assertEqual(metrics["final_magnitude"].shape, (2,))
        self.assertTrue(torch.allclose(metrics["displacement_efficiency"], torch.ones(2)))


if __name__ == "__main__":
    unittest.main()
