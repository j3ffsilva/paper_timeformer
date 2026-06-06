import unittest

from scripts.audit_plane_checkpoint_semantics import (
    heuristic_sense,
    transition_counts,
)


class PlaneCheckpointSemanticTests(unittest.TestCase):
    def test_heuristic_sense_separates_geometry_and_aviation(self):
        self.assertEqual(heuristic_sense(["parallel", "line", "plane_nn"]), "geometry")
        self.assertEqual(heuristic_sense(["pilot", "flight", "plane_nn"]), "aviation")
        self.assertEqual(heuristic_sense(["ordinary", "plane_nn"]), "unlabeled")

    def test_transition_counts(self):
        rows = [
            {"theta0_label": "geometry", "theta1_label": "transport"},
            {"theta0_label": "geometry", "theta1_label": "transport"},
            {"theta0_label": "geometry", "theta1_label": "geometry"},
        ]
        result = {
            (row["theta0_label"], row["theta1_label"]): row["count"]
            for row in transition_counts(rows)
        }
        self.assertEqual(result[("geometry", "transport")], 2)
        self.assertEqual(result[("geometry", "geometry")], 1)


if __name__ == "__main__":
    unittest.main()
