import unittest

from scripts.evaluate_multiresolution_community_profiles import aggregate_rows


class MultiresolutionCommunityProfilesTest(unittest.TestCase):
    def test_equal_weighting_averages_levels(self) -> None:
        rows = [
            {
                "target": "word",
                "temperature": "0.1",
                "k": "10",
                "resolution": "1.0",
                "natural_jsd": "0.2",
                "corpus_theta0_jsd": "0.1",
                "corpus_theta1_jsd": "0.3",
                "checkpoint_d0_jsd": "0.4",
                "checkpoint_d1_jsd": "0.2",
            },
            {
                "target": "word",
                "temperature": "0.1",
                "k": "20",
                "resolution": "1.0",
                "natural_jsd": "0.4",
                "corpus_theta0_jsd": "0.3",
                "corpus_theta1_jsd": "0.1",
                "checkpoint_d0_jsd": "0.2",
                "checkpoint_d1_jsd": "0.4",
            },
        ]
        settings = {
            (10, 1.0): {"mean_ami": "0.8"},
            (20, 1.0): {"mean_ami": "0.9"},
        }

        result = aggregate_rows(rows, settings, weighting="equal")[0]

        self.assertAlmostEqual(result["natural_jsd"], 0.3)
        self.assertAlmostEqual(result["frozen_corpus_mean"], 0.2)


if __name__ == "__main__":
    unittest.main()
