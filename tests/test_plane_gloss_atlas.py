import unittest

from scripts.evaluate_plane_gloss_atlas import high_confidence_sense


class PlaneGlossAtlasTests(unittest.TestCase):
    def test_high_confidence_sense(self):
        self.assertEqual(
            high_confidence_sense(["parallel", "line", "plane_nn"]),
            "geometry",
        )
        self.assertEqual(
            high_confidence_sense(["pilot", "flight", "plane_nn"]),
            "aircraft",
        )
        self.assertEqual(
            high_confidence_sense(["carpenter", "tool", "plane_nn", "wood"]),
            "tool",
        )
        self.assertEqual(high_confidence_sense(["ordinary", "plane_nn"]), "unlabeled")

    def test_tied_keywords_are_unlabeled(self):
        self.assertEqual(
            high_confidence_sense(["line", "airport", "plane_nn"]),
            "unlabeled",
        )


if __name__ == "__main__":
    unittest.main()
