from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn as nn
from torch import Tensor

from .representations import group_representations_by_subject_epoch


AGG_KEYS = ("R", "subject_idx", "epoch_idx", "p_n1", "class_id", "n_occurrences")


class MeanAggregator(nn.Module):
    def forward(self, occurrences: Tensor) -> dict[str, Tensor]:
        return {"R": occurrences.mean(dim=0)}


class AttentionPoolingAggregator(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.score = nn.Linear(d_model, 1)

    def forward(self, occurrences: Tensor) -> dict[str, Tensor]:
        weights = torch.softmax(self.score(occurrences).squeeze(-1), dim=0)
        return {"R": torch.sum(weights.unsqueeze(-1) * occurrences, dim=0), "weights": weights}


class SetTransformerAggregator(nn.Module):
    """Permutation-invariant set encoder with per-occurrence U outputs."""

    def __init__(self, d_model: int, n_heads: int = 4, d_ff: int | None = None, dropout: float = 0.1) -> None:
        super().__init__()
        d_ff = d_ff or 2 * d_model
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))

    def forward(self, occurrences: Tensor) -> dict[str, Tensor]:
        x = occurrences.unsqueeze(0)
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        u = x.squeeze(0)
        return {"R": u.mean(dim=0), "U": u}


class SetSlotsAggregator(SetTransformerAggregator):
    """Set encoder that keeps multiple period-level slots instead of one mean."""

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        d_ff: int | None = None,
        dropout: float = 0.1,
        num_slots: int = 2,
    ) -> None:
        super().__init__(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout)
        self.num_slots = num_slots
        self.slot_queries = nn.Parameter(torch.randn(num_slots, d_model) * 0.02)
        self.slot_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.slot_norm = nn.LayerNorm(d_model)

    def forward(self, occurrences: Tensor) -> dict[str, Tensor]:
        encoded = super().forward(occurrences)
        u = encoded["U"]
        queries = self.slot_queries.unsqueeze(0)
        slots, _ = self.slot_attn(queries, u.unsqueeze(0), u.unsqueeze(0), need_weights=False)
        slots = self.slot_norm(slots.squeeze(0))
        return {"R": slots.reshape(-1), "U": u, "slots": slots}


def build_aggregator(name: str, d_model: int, n_heads: int = 4, num_slots: int = 2) -> nn.Module:
    if name == "mean":
        return MeanAggregator()
    if name == "attention":
        return AttentionPoolingAggregator(d_model)
    if name == "set":
        return SetTransformerAggregator(d_model, n_heads=n_heads)
    if name == "set_slots":
        return SetSlotsAggregator(d_model, n_heads=n_heads, num_slots=num_slots)
    raise ValueError("Unknown aggregator. Choose mean, attention, set, or set_slots.")


@torch.no_grad()
def aggregate_subject_periods(
    reps: dict[str, Tensor],
    aggregator: nn.Module | None = None,
    device: str = "cpu",
) -> dict[str, Tensor]:
    """Aggregate occurrence-level h_s^i(t) into period-level R_s(t)."""
    device_t = torch.device(device)
    aggregator = aggregator or MeanAggregator()
    aggregator.eval()
    aggregator.to(device_t)

    grouped = group_representations_by_subject_epoch(reps)
    out: dict[str, list[Tensor]] = defaultdict(list)

    for (subject, epoch), group in sorted(grouped.items()):
        h = group["h"].to(device_t)
        aggregated = aggregator(h)
        out["R"].append(aggregated["R"].detach().cpu())
        out["subject_idx"].append(torch.tensor(subject, dtype=torch.long))
        out["epoch_idx"].append(torch.tensor(epoch, dtype=torch.long))
        out["p_n1"].append(group["p_n1"].float().mean())
        out["class_id"].append(torch.mode(group["class_id"]).values)
        out["n_occurrences"].append(torch.tensor(len(h), dtype=torch.long))

    return {key: torch.stack(out[key], dim=0) for key in AGG_KEYS}
