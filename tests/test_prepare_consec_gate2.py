import unittest

from scripts.prepare_consec_gate2 import (
    classify_graft,
    classify_tree,
    display_context,
)


class PrepareConsecGate2Tests(unittest.TestCase):
    def test_classify_graft(self):
        self.assertEqual(
            classify_graft(["government", "graft_nn", "bribe"])[0],
            "corruption",
        )
        self.assertEqual(
            classify_graft(["skin", "graft_nn", "patient"])[0],
            "medical",
        )
        self.assertEqual(
            classify_graft(
                [
                    "graft_nn",
                    "jewel",
                    "chain",
                    "battery",
                    "plastic",
                    "hair",
                    "skin",
                    "and",
                    "bone",
                ]
            )[0],
            "unlabeled",
        )
        self.assertEqual(
            classify_graft(["stock", "graft_nn", "scion"])[0],
            "botanical_inventory_gap",
        )

    def test_classify_tree_diagram_phrase(self):
        document = ["the", "family", "tree_nn", "show", "ancestor"]
        self.assertEqual(classify_tree(document, 2), ("diagram", 1))

    def test_classify_tree_plant_requires_two_clues(self):
        document = ["forest", "branch", "tree_nn", "shade"]
        self.assertEqual(classify_tree(document, 2)[0], "plant")
        self.assertEqual(classify_tree(["green", "tree_nn"], 1)[0], "unlabeled")

    def test_display_context_marks_target(self):
        self.assertEqual(
            display_context(["use", "graft_nn", "with", "skin"], "graft_nn"),
            "use [graft] with skin",
        )


if __name__ == "__main__":
    unittest.main()
