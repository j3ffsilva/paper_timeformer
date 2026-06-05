import unittest

import torch

from timeformers.corpus import SUBJECTS
from timeformers.structural_experiment import structural_metric_rows, summarize_structural_rows


def profiles_with_shared_change(values: list[float]) -> list[torch.Tensor]:
    profiles = []
    for value in values:
        profile = torch.eye(len(SUBJECTS))
        profile[0, 1] = value
        profile[1, 0] = value
        profiles.append(profile)
    return profiles


class StructuralExperimentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.metadata = {
            subject: {
                "condition": "gradual" if index % 2 == 0 else "abrupt_persistent",
                "quartet": index // 4,
                "direction": "n1_to_n2",
                "start": 0.9,
                "alternate": 0.1,
            }
            for index, subject in enumerate(SUBJECTS)
        }

    def test_metric_rows_preserve_subject_metadata_and_series(self) -> None:
        profiles = profiles_with_shared_change([0.0, 0.5, 1.0])
        placebo = profiles_with_shared_change([0.0, 0.0, 0.0])
        rows, series = structural_metric_rows(
            "continual_real",
            profiles,
            profiles,
            self.metadata,
            placebo_profiles=placebo,
        )

        self.assertEqual(len(rows), len(SUBJECTS))
        self.assertEqual(len(series), len(SUBJECTS) * len(profiles))
        self.assertEqual(rows[0]["condition"], "gradual")
        self.assertAlmostEqual(rows[0]["accumulated_fidelity"], 1.0, places=6)
        self.assertAlmostEqual(rows[0]["accumulated_fidelity_advantage"], 1.0, places=6)
        self.assertEqual([row["period"] for row in series if row["subject"] == "S1"], [0, 1, 2])

    def test_summary_groups_by_regime_and_condition(self) -> None:
        profiles = profiles_with_shared_change([0.0, 1.0])
        rows, _ = structural_metric_rows("continual_real", profiles, profiles, self.metadata)
        summary = summarize_structural_rows(rows)

        self.assertEqual({row["condition"] for row in summary}, {"gradual", "abrupt_persistent"})
        self.assertTrue(all(row["n"] == len(SUBJECTS) // 2 for row in summary))


if __name__ == "__main__":
    unittest.main()
