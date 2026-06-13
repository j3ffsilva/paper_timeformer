import types
import unittest

import torch
import torch.nn as nn

from timeformers.pretrained_init import initialize_from_bert
from timeformers.real_models import RealStaticMLM


class FakeTokenizer:
    unk_token_id = 0
    unk_token = "[UNK]"

    def __init__(self) -> None:
        self.ids = {
            "[PAD]": [1],
            "[CLS]": [2],
            "[SEP]": [3],
            "[MASK]": [4],
            "[UNK]": [0],
            "plane": [5, 6],
            "tree": [7],
        }

    def encode(self, token: str, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        return self.ids.get(token, [self.unk_token_id])


def fake_bert_layer(d_model: int, d_ff: int) -> nn.Module:
    return nn.ModuleDict({
        "attention": nn.ModuleDict({
            "self": nn.ModuleDict({
                "query": nn.Linear(d_model, d_model),
                "key": nn.Linear(d_model, d_model),
                "value": nn.Linear(d_model, d_model),
            }),
            "output": nn.ModuleDict({
                "dense": nn.Linear(d_model, d_model),
                "LayerNorm": nn.LayerNorm(d_model, eps=1e-12),
            }),
        }),
        "intermediate": nn.ModuleDict({"dense": nn.Linear(d_model, d_ff)}),
        "output": nn.ModuleDict({
            "dense": nn.Linear(d_ff, d_model),
            "LayerNorm": nn.LayerNorm(d_model, eps=1e-12),
        }),
    })


class FakeBert(nn.Module):
    def __init__(self, d_model: int = 4, n_layers: int = 2, n_heads: int = 2, d_ff: int = 8) -> None:
        super().__init__()
        self.config = types.SimpleNamespace(
            hidden_size=d_model,
            num_hidden_layers=n_layers,
            num_attention_heads=n_heads,
            intermediate_size=d_ff,
            layer_norm_eps=1e-12,
        )
        self.embeddings = nn.Module()
        self.embeddings.word_embeddings = nn.Embedding(16, d_model)
        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList(
            [fake_bert_layer(d_model, d_ff) for _ in range(n_layers)]
        )


class PretrainedInitTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.vocab = ["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]", "plane_nn", "tree", "missing"]
        self.model = RealStaticMLM(
            vocab_size=len(self.vocab),
            seq_len=8,
            d_model=4,
            n_layers=2,
            n_heads=2,
            d_ff=8,
            norm_first=False,
            activation="gelu",
            layer_norm_eps=1e-12,
        )
        self.bert = FakeBert()

    def test_copies_encoder_parameters_and_averages_wordpieces(self) -> None:
        original_missing = self.model.token_emb.weight[7].detach().clone()

        report = initialize_from_bert(
            self.model,
            self.bert,
            FakeTokenizer(),
            self.vocab,
            model_name="fake/bert",
        )

        source = self.bert.encoder.layer[0]
        expected_qkv = torch.cat([
            source.attention.self.query.weight,
            source.attention.self.key.weight,
            source.attention.self.value.weight,
        ])
        self.assertTrue(torch.equal(self.model.encoder.layers[0].self_attn.in_proj_weight, expected_qkv))
        expected_plane = self.bert.embeddings.word_embeddings.weight[[5, 6]].mean(dim=0)
        self.assertTrue(torch.equal(self.model.token_emb.weight[5], expected_plane))
        self.assertTrue(torch.equal(self.model.token_emb.weight[7], original_missing))
        self.assertEqual(report.encoder_layers_copied, 2)
        self.assertEqual(report.token_embeddings_copied, 7)
        self.assertEqual(report.token_embeddings_skipped, 1)

    def test_rejects_pre_norm_target(self) -> None:
        model = RealStaticMLM(
            vocab_size=len(self.vocab),
            d_model=4,
            n_layers=2,
            n_heads=2,
            d_ff=8,
            norm_first=True,
            activation="gelu",
        )
        with self.assertRaisesRegex(ValueError, "norm order"):
            initialize_from_bert(
                model,
                self.bert,
                FakeTokenizer(),
                self.vocab,
                model_name="fake/bert",
            )


if __name__ == "__main__":
    unittest.main()
