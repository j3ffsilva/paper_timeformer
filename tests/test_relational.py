import unittest

import torch

from timeformers.corpus import Example, SUBJECTS, generate_subject_probe_examples
from timeformers.dataset import ContextPairMLMDataset, MASK_ID, POS_OBJECT, POS_VERB
from timeformers.models import build_model
from timeformers.relational import (
    centered_cosine_similarity_matrix,
    cosine_similarity_matrix,
    jensen_shannon_similarity_matrix,
    normalized_euclidean_similarity_matrix,
)
from timeformers.relational_metrics import (
    placebo_reference_relational_change,
    relational_change_by_subject,
    representation_cka,
)
from timeformers.representations import extract_occurrence_representations


class RelationalMetricsTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.points = torch.randn(12, 8)

    def test_orthogonal_coordinate_change_is_not_semantic_change(self) -> None:
        q, _ = torch.linalg.qr(torch.randn(8, 8))
        before = cosine_similarity_matrix(self.points)
        after = cosine_similarity_matrix(self.points @ q)

        changes = relational_change_by_subject(before, after, k=4)

        self.assertTrue(torch.allclose(before, after, atol=1e-5))
        for values in changes.values():
            self.assertTrue(torch.allclose(values, torch.zeros_like(values), atol=1e-5))
        self.assertAlmostEqual(representation_cka(self.points, self.points @ q), 1.0, places=5)

    def test_centered_cosine_ignores_translation_rotation_and_scale(self) -> None:
        q, _ = torch.linalg.qr(torch.randn(8, 8))
        transformed = 3.0 * (self.points @ q) + torch.randn(8)

        before = centered_cosine_similarity_matrix(self.points)
        after = centered_cosine_similarity_matrix(transformed)

        self.assertTrue(torch.allclose(before, after, atol=1e-5))

    def test_normalized_euclidean_ignores_translation_rotation_and_scale(self) -> None:
        q, _ = torch.linalg.qr(torch.randn(8, 8))
        transformed = 3.0 * (self.points @ q) + torch.randn(8)

        before = normalized_euclidean_similarity_matrix(self.points)
        after = normalized_euclidean_similarity_matrix(transformed)

        self.assertTrue(torch.allclose(before, after, atol=1e-5))

    def test_jensen_shannon_similarity_uses_probability_distributions(self) -> None:
        distributions = torch.tensor([[0.9, 0.1], [0.9, 0.1], [0.1, 0.9]])
        similarities = jensen_shannon_similarity_matrix(distributions)

        self.assertTrue(torch.allclose(similarities, similarities.T))
        self.assertTrue(torch.allclose(similarities.diag(), torch.ones(3)))
        self.assertAlmostEqual(float(similarities[0, 1]), 1.0, places=6)
        self.assertLess(float(similarities[0, 2]), float(similarities[0, 1]))

    def test_changed_relation_is_detected(self) -> None:
        changed = self.points.clone()
        changed[0] = changed[1]
        before = cosine_similarity_matrix(self.points)
        after = cosine_similarity_matrix(changed)

        changes = relational_change_by_subject(before, after, k=4)

        self.assertGreater(float(changes["mean_abs_similarity_delta"][0]), 0.01)
        self.assertGreater(float(changes["spearman_change"][0]), 0.01)

    def test_placebo_reference_reports_known_residual_direction(self) -> None:
        before = cosine_similarity_matrix(self.points)
        placebo_after = before + torch.randn_like(before) * 0.01
        oracle_after = before.clone()
        oracle_after[0, 1] += 0.2
        oracle_after[1, 0] += 0.2
        observed_after = placebo_after + (oracle_after - before)

        changes = placebo_reference_relational_change(
            before,
            observed_after,
            before,
            placebo_after,
            before,
            oracle_after,
        )

        self.assertAlmostEqual(float(changes["excess_oracle_direction_cosine"][0]), 1.0, places=5)
        self.assertGreater(float(changes["excess_mean_abs_similarity_delta"][0]), 0.0)

    def test_context_pair_masking_hides_both_context_markers(self) -> None:
        row = Example(0, "S1", "V1", "O1", 0, "train", 1.0, "stable")
        item = ContextPairMLMDataset([row])[0]

        self.assertEqual(int(item["input_ids"][POS_VERB]), MASK_ID)
        self.assertEqual(int(item["input_ids"][POS_OBJECT]), MASK_ID)
        self.assertNotEqual(int(item["labels"][POS_VERB]), -100)
        self.assertNotEqual(int(item["labels"][POS_OBJECT]), -100)

    def test_subject_probes_have_one_neutral_example_per_subject(self) -> None:
        rows = generate_subject_probe_examples()

        self.assertEqual(len(rows), len(SUBJECTS))
        self.assertEqual({row.subject for row in rows}, set(SUBJECTS))
        self.assertEqual(len({(row.verb, row.obj) for row in rows}), 1)

    def test_prediction_probe_extracts_normalized_distributions(self) -> None:
        dataset = ContextPairMLMDataset(generate_subject_probe_examples())
        model = build_model("Static", d_model=16, n_heads=4, n_layers=1, d_ff=32, dropout=0.0)

        reps = extract_occurrence_representations(
            model,
            dataset,
            batch_size=len(dataset),
            target="prediction_distribution",
        )
        verb_distribution, object_distribution = reps["h"].chunk(2, dim=1)

        self.assertEqual(reps["h"].shape[1], 16)
        self.assertTrue(torch.allclose(verb_distribution.sum(dim=1), torch.ones(len(dataset)), atol=1e-5))
        self.assertTrue(torch.allclose(object_distribution.sum(dim=1), torch.ones(len(dataset)), atol=1e-5))


if __name__ == "__main__":
    unittest.main()
