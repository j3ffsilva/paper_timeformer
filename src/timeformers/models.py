from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .dataset import POS_SUBJECT, SEQ_LEN, VOCAB_SIZE


class TimeEncoding(nn.Module):
    def __init__(self, d_model: int, d_sin: int = 32, n_epochs: int = 10) -> None:
        super().__init__()
        if d_sin % 2:
            raise ValueError("d_sin must be even")
        self.t_scale = max(n_epochs - 1, 1)
        freqs = torch.exp(-math.log(10000.0) * torch.arange(0, d_sin, 2).float() / d_sin)
        self.register_buffer("freqs", freqs)
        self.mlp = nn.Sequential(nn.Linear(d_sin, d_sin), nn.GELU(), nn.Linear(d_sin, d_model))

    def forward(self, epoch_idx: Tensor) -> Tensor:
        t = epoch_idx.float() / self.t_scale
        angles = t.unsqueeze(-1) * self.freqs.unsqueeze(0)
        feats = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return self.mlp(feats)


class MLMHead(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, VOCAB_SIZE)

    def forward(self, hidden: Tensor) -> Tensor:
        return self.proj(F.gelu(self.norm(hidden)))


class BaseModel(nn.Module):
    needs_time = False

    def __init__(self, d_model: int = 48, n_heads: int = 4, n_layers: int = 2, d_ff: int = 96, dropout: float = 0.1) -> None:
        super().__init__()
        self.d_model = d_model
        self.token_emb = nn.Embedding(VOCAB_SIZE, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(SEQ_LEN, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.drop = nn.Dropout(dropout)
        self.mlm_head = MLMHead(d_model)
        nn.init.normal_(self.token_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

    def token_inputs(self, input_ids: Tensor, epoch_idx: Tensor | None = None) -> Tensor:
        return self.token_emb(input_ids)

    def embed(self, input_ids: Tensor, epoch_idx: Tensor | None = None) -> Tensor:
        positions = torch.arange(input_ids.size(1), device=input_ids.device)
        return self.drop(self.token_inputs(input_ids, epoch_idx) + self.pos_emb(positions))

    def forward(self, input_ids: Tensor, epoch_idx: Tensor | None = None) -> dict[str, Tensor]:
        hidden = self.encoder(self.embed(input_ids, epoch_idx))
        return {
            "logits": self.mlm_head(hidden),
            "hidden": hidden,
            "h_subj": hidden[:, POS_SUBJECT, :],
        }


class Static(BaseModel):
    pass


class Additive(BaseModel):
    needs_time = True

    def __init__(self, *args, d_sin: int = 32, n_epochs: int = 10, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.time_enc = TimeEncoding(self.d_model, d_sin=d_sin, n_epochs=n_epochs)

    def token_inputs(self, input_ids: Tensor, epoch_idx: Tensor | None = None) -> Tensor:
        tok = self.token_emb(input_ids)
        if epoch_idx is None:
            return tok
        return tok + self.time_enc(epoch_idx).unsqueeze(1)


class TokenTime(BaseModel):
    needs_time = True

    def __init__(self, *args, d_sin: int = 32, n_epochs: int = 10, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.time_enc = TimeEncoding(self.d_model, d_sin=d_sin, n_epochs=n_epochs)
        self.proj = nn.Linear(2 * self.d_model, self.d_model)

    def token_inputs(self, input_ids: Tensor, epoch_idx: Tensor | None = None) -> Tensor:
        tok = self.token_emb(input_ids)
        if epoch_idx is None:
            return tok
        t = self.time_enc(epoch_idx).unsqueeze(1).expand_as(tok)
        return self.proj(torch.cat([tok, t], dim=-1))


class FiLM(BaseModel):
    needs_time = True

    def __init__(self, *args, d_sin: int = 32, n_epochs: int = 10, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.time_enc = TimeEncoding(self.d_model, d_sin=d_sin, n_epochs=n_epochs)
        self.gamma = nn.Linear(self.d_model, self.d_model)
        self.beta = nn.Linear(self.d_model, self.d_model)
        nn.init.zeros_(self.gamma.weight)
        nn.init.ones_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def token_inputs(self, input_ids: Tensor, epoch_idx: Tensor | None = None) -> Tensor:
        tok = self.token_emb(input_ids)
        if epoch_idx is None:
            return tok
        t = self.time_enc(epoch_idx)
        gamma = self.gamma(t).unsqueeze(1)
        beta = self.beta(t).unsqueeze(1)
        return gamma * tok + beta


MODEL_REGISTRY = {
    "Static": Static,
    "Additive": Additive,
    "TokenTime": TokenTime,
    "FiLM": FiLM,
}


def build_model(name: str, **kwargs) -> nn.Module:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model {name!r}. Choose from {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**kwargs)

