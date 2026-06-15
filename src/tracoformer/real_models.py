"""A small from-scratch Transformer encoder with an MLM head, for masked
language modeling over a custom (non-BERT) vocabulary built by
`real_corpus.build_vocabulary`.

This is independent of the pretrained BERT-tiny encoder used elsewhere
(`bert_continual.py`): it has its own token/position embeddings and its own
vocabulary, and is trained from random initialization.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class RealMLMHead(nn.Module):
    """MLM prediction head: `LayerNorm -> GELU -> Linear(d_model, vocab_size)`,
    applied to every position's hidden state to produce per-token logits
    over the vocabulary."""

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, hidden: Tensor) -> Tensor:
        return self.proj(F.gelu(self.norm(hidden)))


class RealStaticMLM(nn.Module):
    """Token + position embeddings, a stack of standard Transformer encoder
    layers, and an `RealMLMHead`.

    "Static" refers to `needs_time = False`: this model has no
    period-conditional component, so `epoch_idx` (the period index passed by
    `real_corpus` datasets) is accepted but ignored everywhere it appears.
    """

    needs_time = False

    def __init__(
        self,
        *,
        vocab_size: int,
        seq_len: int = 32,
        d_model: int = 96,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 192,
        dropout: float = 0.1,
        pad_id: int = 0,
        norm_first: bool = True,
        activation: str = "relu",
        layer_norm_eps: float = 1e-5,
        mask_padding: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.norm_first = norm_first
        self.activation_name = activation
        self.layer_norm_eps = layer_norm_eps
        self.mask_padding = mask_padding
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=norm_first,
            layer_norm_eps=layer_norm_eps,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.drop = nn.Dropout(dropout)
        self.mlm_head = RealMLMHead(d_model, vocab_size)
        nn.init.normal_(self.token_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

    def embed(self, input_ids: Tensor, epoch_idx: Tensor | None = None) -> Tensor:
        """Token embedding + position embedding (by absolute position
        `0..seq_len-1`), with dropout. `epoch_idx` is accepted for interface
        compatibility with period-conditional models but unused here."""
        del epoch_idx
        positions = torch.arange(input_ids.size(1), device=input_ids.device)
        return self.drop(self.token_emb(input_ids) + self.pos_emb(positions))

    def forward(self, input_ids: Tensor, epoch_idx: Tensor | None = None) -> dict[str, Tensor]:
        """Returns `logits` (`(batch, seq_len, vocab_size)`, MLM predictions
        for every position), `hidden` (`(batch, seq_len, d_model)`, the
        encoder's final hidden states), and `h_subj` (`hidden[:, 1, :]`, the
        hidden state at position 1 -- the first content token after
        `[CLS]`).

        If `mask_padding` is set, padding positions (`input_ids ==
        padding_idx`) are excluded from self-attention via
        `src_key_padding_mask`.
        """
        padding_mask = input_ids.eq(self.token_emb.padding_idx) if self.mask_padding else None
        hidden = self.encoder(
            self.embed(input_ids, epoch_idx),
            src_key_padding_mask=padding_mask,
        )
        return {
            "logits": self.mlm_head(hidden),
            "hidden": hidden,
            "h_subj": hidden[:, 1, :],
        }
