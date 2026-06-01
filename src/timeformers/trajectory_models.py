from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class TemporalEncoder(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 96,
        max_len: int = 32,
        variant: str = "bidirectional",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if variant not in {"bidirectional", "causal", "linear"}:
            raise ValueError("variant must be bidirectional, causal, or linear")
        self.variant = variant
        self.proj = nn.Linear(d_in, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        if variant == "linear":
            self.encoder = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU())
        else:
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_ff,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, values: Tensor, valid_mask: Tensor | None = None) -> Tensor:
        positions = torch.arange(values.size(1), device=values.device)
        x = self.proj(values) + self.pos_emb(positions).unsqueeze(0)
        if self.variant == "linear":
            return self.encoder(x)

        key_padding_mask = None if valid_mask is None else ~valid_mask
        attn_mask = None
        if self.variant == "causal":
            attn_mask = torch.triu(
                torch.ones(values.size(1), values.size(1), device=values.device, dtype=torch.bool),
                diagonal=1,
            )
        return self.encoder(x, mask=attn_mask, src_key_padding_mask=key_padding_mask)


class TrajectoryTeacher(nn.Module):
    def __init__(self, d_in: int, d_traj: int = 32, encoder_variant: str = "linear", max_len: int = 32) -> None:
        super().__init__()
        self.encoder = TemporalEncoder(d_in, d_model=d_traj, variant=encoder_variant, max_len=max_len)
        self.decoder = nn.Sequential(nn.LayerNorm(d_traj), nn.Linear(d_traj, d_in))

    def forward(self, values: Tensor, valid_mask: Tensor | None = None) -> dict[str, Tensor]:
        m = self.encoder(values, valid_mask)
        return {"M": m, "recon": self.decoder(m)}


class TrajectoryStudent(nn.Module):
    def __init__(self, d_in: int, d_traj: int = 32, encoder_variant: str = "bidirectional", max_len: int = 32) -> None:
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(d_in))
        nn.init.normal_(self.mask_token, std=0.02)
        self.encoder = TemporalEncoder(d_in, d_model=d_traj, variant=encoder_variant, max_len=max_len)

    def forward(self, values: Tensor, mask_positions: Tensor, valid_mask: Tensor | None = None) -> dict[str, Tensor]:
        masked = values.clone()
        masked[mask_positions] = self.mask_token.to(values.dtype)
        return {"M": self.encoder(masked, valid_mask)}
