import unittest

from scripts.diagnose_paired_field_controls import (
    FIELD_CONTROLS,
    TARGET_FIELDS,
    field_adjusted_scores,
    run_level_adjustments,
)


class PairedFieldControlTests(unittest.TestCase):
    def test_adjusted_score_uses_field_control_median(self):
        rows = [
            {
                "token": "chairman_nn",
                "score": 0.5,
                "theta0_score": 0.4,
                "theta1_score": 0.6,
                "checkpoint_disagreement": 0.2,
                "count_d0": 10,
                "count_d1": 12,
            },
            {
                "token": "plane_nn",
                "score": 0.4,
                "theta0_score": 0.3,
                "theta1_score": 0.5,
                "checkpoint_disagreement": 0.2,
                "count_d0": 10,
                "count_d1": 12,
            },
            {
                "token": "graft_nn",
                "score": 0.3,
                "theta0_score": 0.3,
                "theta1_score": 0.3,
                "checkpoint_disagreement": 0.0,
                "count_d0": 10,
                "count_d1": 12,
            },
            {
                "token": "tree_nn",
                "score": 0.2,
                "theta0_score": 0.2,
                "theta1_score": 0.2,
                "checkpoint_disagreement": 0.0,
                "count_d0": 10,
                "count_d1": 12,
            },
        ]
        for field, controls in FIELD_CONTROLS.items():
            for index, token in enumerate(controls):
                rows.append(
                    {
                        "token": token,
                        "score": 0.1 + index * 0.01,
                        "theta0_score": 0.1,
                        "theta1_score": 0.1,
                        "checkpoint_disagreement": 0.0,
                        "count_d0": 10,
                        "count_d1": 12,
                    }
                )
        fields, adjusted = field_adjusted_scores(rows)
        baselines = {row["field"]: row["median_score"] for row in fields}
        chairman = next(row for row in adjusted if row["target"] == "chairman_nn")
        self.assertAlmostEqual(
            chairman["adjusted_score"],
            0.5 - baselines["leadership"],
        )

    def test_run_level_adjustment_compares_matching_configuration(self):
        rows = []
        tokens = list(TARGET_FIELDS)
        for controls in FIELD_CONTROLS.values():
            tokens.extend(controls)
        for token in dict.fromkeys(tokens):
            for checkpoint in range(2):
                rows.append(
                    {
                        "target": token,
                        "checkpoint": checkpoint,
                        "n_clusters": 2,
                        "seed": 0,
                        "jsd": 0.5 if token == "chairman_nn" else 0.1,
                    }
                )
        adjusted, robustness = run_level_adjustments(rows)
        chairman = next(row for row in adjusted if row["target"] == "chairman_nn")
        self.assertAlmostEqual(chairman["adjusted_score"], 0.4)
        chairman_robustness = next(
            row for row in robustness if row["target"] == "chairman_nn"
        )
        self.assertEqual(chairman_robustness["positive_fraction"], 1.0)


if __name__ == "__main__":
    unittest.main()
