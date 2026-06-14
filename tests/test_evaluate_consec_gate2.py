import unittest

from scripts.evaluate_consec_gate2 import (
    apply_audit,
    gate_summary,
    prepare_context,
)


class EvaluateConsecGate2Tests(unittest.TestCase):
    def test_prepare_context_uses_target_marker(self):
        tokens, index, lemma = prepare_context(
            "the family [tree] show ancestors", "tree_nn"
        )
        self.assertEqual(tokens, ["the", "family", "tree", "show", "ancestors"])
        self.assertEqual(index, 2)
        self.assertEqual(lemma, "tree")

    def test_apply_audit_preserves_human_disagreement(self):
        candidates = [
            {
                "sample_id": "sample-1",
                "target": "tree_nn",
                "context": "a genealogical [tree]",
                "heuristic_sense": "diagram",
            },
            {
                "sample_id": "sample-2",
                "target": "tree_nn",
                "context": "a leafy [tree]",
                "heuristic_sense": "plant",
            },
        ]
        annotations = [
            {
                "item_id": "G2-001",
                "target": "tree_nn",
                "context": "a genealogical [tree]",
                "label": "plant",
                "confidence": "high",
                "notes": "",
            }
        ]
        manifest = [
            {
                "item_id": "G2-001",
                "sample_id": "sample-1",
                "target": "tree_nn",
                "context": "a genealogical [tree]",
            }
        ]
        rows, summary = apply_audit(candidates, annotations, manifest)
        self.assertEqual(rows[0]["post_audit_gold_sense"], "plant")
        self.assertEqual(rows[1]["post_audit_gold_sense"], "plant")
        self.assertEqual(summary["agreements"], 0)
        self.assertEqual(len(summary["disagreements"]), 1)

    def test_apply_audit_excludes_unclear(self):
        candidates = [
            {
                "sample_id": "sample-1",
                "target": "graft_nn",
                "context": "ambiguous [graft]",
                "heuristic_sense": "corruption",
            }
        ]
        annotations = [
            {
                "item_id": "G2-001",
                "target": "graft_nn",
                "context": "ambiguous [graft]",
                "label": "unclear",
                "confidence": "high",
                "notes": "",
            }
        ]
        manifest = [
            {
                "item_id": "G2-001",
                "sample_id": "sample-1",
                "target": "graft_nn",
                "context": "ambiguous [graft]",
            }
        ]
        rows, _ = apply_audit(candidates, annotations, manifest)
        self.assertEqual(rows[0]["post_audit_evaluable"], "False")

    def test_gate_summary_passes_clear_case(self):
        rows = []
        for target, label in (
            ("graft_nn", "corruption"),
            ("graft_nn", "medical"),
            ("tree_nn", "diagram"),
            ("tree_nn", "plant"),
        ):
            for _ in range(20):
                rows.append(
                    {
                        "target": target,
                        "gold": label,
                        "prediction": label,
                    }
                )
        summary = gate_summary(rows, "gold", 1000, 7)
        self.assertTrue(summary["passed"])


if __name__ == "__main__":
    unittest.main()
