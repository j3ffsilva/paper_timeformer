import unittest

import numpy as np

from scripts.evaluate_contextual_usage_clusters import (
    aggregate_runs,
    balanced_sample,
    cluster_period_distributions,
    relational_occurrence_vectors,
)


class ContextualUsageClusterTests(unittest.TestCase):
    def test_balanced_sample_uses_equal_period_sizes(self):
        before = np.arange(60, dtype=float).reshape(20, 3)
        after = np.arange(90, dtype=float).reshape(30, 3)
        left, right = balanced_sample(before, after, max_per_period=12, seed=0)
        self.assertEqual(left.shape, (12, 3))
        self.assertEqual(right.shape, (12, 3))

    def test_usage_cluster_jsd_separates_shift_from_stability(self):
        rng = np.random.default_rng(4)
        sense_a = rng.normal(loc=[3.0, 0.0], scale=0.15, size=(80, 2))
        sense_b = rng.normal(loc=[0.0, 3.0], scale=0.15, size=(80, 2))
        stable_before = np.concatenate([sense_a[:40], sense_b[:40]])
        stable_after = np.concatenate([sense_a[40:], sense_b[40:]])
        shifted_before = sense_a
        shifted_after = sense_b

        stable = cluster_period_distributions(
            stable_before, stable_after, n_clusters=2, seed=0
        )
        shifted = cluster_period_distributions(
            shifted_before, shifted_after, n_clusters=2, seed=0
        )
        self.assertLess(stable["jsd"], 0.05)
        self.assertGreater(shifted["jsd"], 0.8)

    def test_aggregate_runs_uses_median(self):
        rows = [
            {
                "target": "word_nn",
                "checkpoint": 0,
                "jsd": score,
                "silhouette": 0.5,
                "count_d0": 10,
                "count_d1": 12,
                "sampled_per_period": 10,
            }
            for score in [0.1, 0.2, 0.9]
        ]
        aggregated = aggregate_runs(rows)
        self.assertAlmostEqual(aggregated[0]["score"], 0.2)

    def test_relational_occurrence_vectors_preserve_period_sizes(self):
        rng = np.random.default_rng(8)
        before = rng.normal(size=(20, 8))
        after = rng.normal(size=(30, 8))
        anchors = rng.normal(size=(40, 8))
        anchors /= np.linalg.norm(anchors, axis=1, keepdims=True)
        left, right = relational_occurrence_vectors(
            before,
            after,
            anchors,
            pca_components=6,
            seed=0,
        )
        self.assertEqual(left.shape, (20, 6))
        self.assertEqual(right.shape, (30, 6))


if __name__ == "__main__":
    unittest.main()
