import unittest

from scripts.prepare_consec_gate3 import assign_role, display_context


class PrepareConsecGate3Tests(unittest.TestCase):
    def test_assign_role(self):
        self.assertEqual(
            assign_role(
                {
                    "coverage_status": "sufficient",
                    "n_wordnet_senses": "2",
                }
            ),
            "confirmatory",
        )
        self.assertEqual(
            assign_role(
                {"coverage_status": "partial", "n_wordnet_senses": "9"}
            ),
            "partial_diagnostic",
        )
        self.assertEqual(
            assign_role(
                {
                    "coverage_status": "monosemous_covered",
                    "n_wordnet_senses": "1",
                }
            ),
            "monosemous_control",
        )

    def test_display_context(self):
        self.assertEqual(
            display_context(["family_nn", "tree_nn", "diagram_nn"], 1, 2),
            "family [tree] diagram",
        )


if __name__ == "__main__":
    unittest.main()
