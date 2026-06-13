import unittest

import torch
from transformers import BertConfig, BertForMaskedLM

from timeformers.bert_continual import (
    CheckpointDiagnostic,
    assert_weight_tying,
    random_pseudo_periods,
    relative_l2_sp_penalty,
    select_checkpoint,
    snapshot_named_parameters,
    split_documents,
    strip_pos_suffix,
)


class BertContinualTest(unittest.TestCase):
    def test_strip_pos_suffix(self) -> None:
        self.assertEqual(strip_pos_suffix("plane_nn"), "plane")
        self.assertEqual(strip_pos_suffix("circle_vb"), "circle")
        self.assertEqual(strip_pos_suffix("ordinary"), "ordinary")

    def test_document_split_is_reproducible_and_disjoint(self) -> None:
        documents = [[str(index)] for index in range(20)]
        train_a, validation_a = split_documents(
            documents,
            validation_fraction=0.2,
            seed=7,
        )
        train_b, validation_b = split_documents(
            documents,
            validation_fraction=0.2,
            seed=7,
        )
        self.assertEqual(train_a, train_b)
        self.assertEqual(validation_a, validation_b)
        self.assertFalse(set(map(tuple, train_a)) & set(map(tuple, validation_a)))

    def test_random_pseudo_periods_preserve_sizes_and_documents(self) -> None:
        periods = [
            [["a"], ["b"]],
            [["c"], ["d"], ["e"]],
        ]
        shuffled = random_pseudo_periods(periods, seed=3)
        self.assertEqual([len(period) for period in shuffled], [2, 3])
        self.assertEqual(
            sorted(document[0] for period in shuffled for document in period),
            ["a", "b", "c", "d", "e"],
        )

    def test_assert_weight_tying(self) -> None:
        model = BertForMaskedLM(BertConfig(
            vocab_size=32,
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
        ))
        assert_weight_tying(model)
        model.cls.predictions.decoder.weight = torch.nn.Parameter(
            model.cls.predictions.decoder.weight.detach().clone()
        )
        with self.assertRaisesRegex(ValueError, "not tied"):
            assert_weight_tying(model)

    def test_selection_prefers_retention_within_loss_tolerance(self) -> None:
        diagnostics = [
            CheckpointDiagnostic("best-loss", 0, 1.0, 10, 1.0, 1.0, 0.01, [1.0], 1.0, 1.0, 0.2, 0.1, 0.8, 0.0),
            CheckpointDiagnostic("retained", 0, 0.5, 5, 1.1, 1.0, 0.01, [1.005], 1.005, 1.005, 0.1, 0.05, 0.95, 0.0),
            CheckpointDiagnostic("too-high-loss", 0, 0.25, 2, 1.2, 1.0, 0.01, [1.02], 1.02, 1.02, 0.05, 0.02, 0.99, 0.0),
        ]
        self.assertEqual(select_checkpoint(diagnostics).name, "retained")

    def test_relative_l2_sp_penalty_is_normalized(self) -> None:
        model = BertForMaskedLM(BertConfig(
            vocab_size=32,
            hidden_size=8,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=16,
        ))
        reference = snapshot_named_parameters(
            model,
            prefix="bert.encoder.layer.1.",
        )
        self.assertEqual(float(relative_l2_sp_penalty(model, reference)), 0.0)
        with torch.no_grad():
            model.bert.encoder.layer[1].output.dense.weight.add_(0.1)
        self.assertGreater(float(relative_l2_sp_penalty(model, reference)), 0.0)


if __name__ == "__main__":
    unittest.main()
