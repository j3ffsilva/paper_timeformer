"""Shared `RealStaticMLM` encoding helpers for hidden-state evaluation scripts."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from timeformers.real_corpus import encode_document  # noqa: E402
from timeformers.real_models import RealStaticMLM  # noqa: E402


class ContextChunkDataset(Dataset):
    def __init__(self, corpus, token_to_id: dict[str, int], *, seq_len: int) -> None:
        self.seq_len = seq_len
        self.pad_id = token_to_id["[PAD]"]
        self.cls_id = token_to_id["[CLS]"]
        self.sep_id = token_to_id["[SEP]"]
        content_len = seq_len - 2
        self.chunks = []
        for document in corpus.documents:
            encoded = encode_document(document, token_to_id)
            self.chunks.extend(
                encoded[start : start + content_len]
                for start in range(0, len(encoded), content_len)
                if encoded[start : start + content_len]
            )

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        chunk = self.chunks[index]
        input_ids = [self.cls_id] + chunk + [self.sep_id]
        input_ids += [self.pad_id] * (self.seq_len - len(input_ids))
        lexical_ids = [-1] + chunk + [-1]
        lexical_ids += [-1] * (self.seq_len - len(lexical_ids))
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "lexical_ids": torch.tensor(lexical_ids, dtype=torch.long),
        }


def build_model(config: dict, vocab_size: int, pad_id: int) -> RealStaticMLM:
    return RealStaticMLM(
        vocab_size=vocab_size,
        seq_len=int(config["seq_len"]),
        d_model=int(config["d_model"]),
        n_layers=int(config["layers"]),
        n_heads=int(config["heads"]),
        d_ff=int(config["d_ff"]),
        dropout=float(config["dropout"]),
        pad_id=pad_id,
        norm_first=config.get("encoder_norm_order", "pre") == "pre",
        activation=config.get("activation", "relu"),
        layer_norm_eps=float(config.get("layer_norm_eps", 1e-5)),
        mask_padding=bool(config.get("mask_padding", False)),
    )


def encode_layers(model: RealStaticMLM, input_ids: Tensor) -> dict[str, Tensor]:
    hidden = model.embed(input_ids)
    padding_mask = input_ids.eq(model.token_emb.padding_idx)
    layers = []
    for layer in model.encoder.layers:
        hidden = layer(hidden, src_key_padding_mask=padding_mask)
        layers.append(hidden)
    if model.encoder.norm is not None:
        layers[-1] = model.encoder.norm(layers[-1])
    outputs = {f"layer_{index + 1}": value for index, value in enumerate(layers)}
    if len(layers) >= 2:
        outputs["mean_last_2"] = 0.5 * (layers[-2] + layers[-1])
    return outputs
