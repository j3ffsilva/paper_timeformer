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
