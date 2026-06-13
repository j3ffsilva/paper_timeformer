from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class RealMLMHead(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, hidden: Tensor) -> Tensor:
        return self.proj(F.gelu(self.norm(hidden)))


class RealStaticMLM(nn.Module):
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
        del epoch_idx
        positions = torch.arange(input_ids.size(1), device=input_ids.device)
        return self.drop(self.token_emb(input_ids) + self.pos_emb(positions))

    def forward(self, input_ids: Tensor, epoch_idx: Tensor | None = None) -> dict[str, Tensor]:
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
