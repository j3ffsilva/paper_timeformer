import random
import unittest

from scripts.prepare_wordnet_coverage_gate3 import (
    automatic_priority,
    centered_context,
    sample_occurrences,
    split_target,
)


class PrepareWordNetCoverageGate3Tests(unittest.TestCase):
    def test_split_target_maps_noun_and_verb(self):
        self.assertEqual(split_target("tree_nn"), ("tree", "n"))
        self.assertEqual(split_target("tip_vb"), ("tip", "v"))

    def test_centered_context_marks_target_and_strips_suffixes(self):
        context = centered_context(
            ["the_dt", "family_nn", "tree_nn", "grow_vb"], 2, 2
        )
        self.assertEqual(context, "the_dt family [tree] grow")

    def test_automatic_priority_is_not_coverage_decision(self):
        self.assertEqual(automatic_priority(0), "no_inventory")
        self.assertEqual(automatic_priority(1), "monosemous_control")
        self.assertEqual(automatic_priority(3), "low_review_burden")
        self.assertEqual(automatic_priority(8), "medium_review_burden")
        self.assertEqual(automatic_priority(9), "high_review_burden")

    def test_sampling_is_deterministic_and_does_not_change_global_rng(self):
        occurrences = {
            "tree_nn": {
                "1810-1860": [
                    {
                        "target": "tree_nn",
                        "period": "1810-1860",
                        "document_index": index,
                        "token_index": 2,
                        "context": "context {}".format(index),
                    }
                    for index in range(10)
                ],
                "1960-2010": [],
            }
        }
        random.seed(44)
        expected = random.random()
        random.seed(44)
        first = sample_occurrences(occurrences, 4, 7)
        observed = random.random()
        second = sample_occurrences(occurrences, 4, 7)
        self.assertEqual(first, second)
        self.assertEqual(observed, expected)
        self.assertEqual(len(first), 4)


if __name__ == "__main__":
    unittest.main()
