import unittest

import numpy as np

from scripts.evaluate_consec_timeformer_occurrences import (
    pair_metrics,
    sign_flip_p,
)


class ConsecTimeformerOccurrenceTest(unittest.TestCase):
    def test_pair_metrics_detect_semantic_geometry_alignment(self):
        rows = []
        for index, probability in enumerate([0.99, 0.9, 0.1, 0.01]):
            rows.append(
                {
                    "target": "plane_nn",
                    "period": "1810-1860" if index % 2 == 0 else "1960-2010",
                    "prediction_sensekey": "a" if probability > 0.5 else "b",
                    "sense_probabilities": (
                        '{"a": %.8f, "b": %.8f}'
                        % (probability, 1.0 - probability)
                    ),
                }
            )
        vectors = np.asarray(
            [[1.0, 0.0], [0.9, 0.1], [0.1, 0.9], [0.0, 1.0]],
            dtype=np.float32,
        )
        result = pair_metrics(rows, vectors)
        self.assertGreater(result["semantic_geometry_partial_period"], 0.8)

    def test_sign_flip_p_is_small_for_consistent_positive_values(self):
        value = sign_flip_p([0.2] * 20, 5000, 9)
        self.assertLess(value, 0.01)


if __name__ == "__main__":
    unittest.main()
