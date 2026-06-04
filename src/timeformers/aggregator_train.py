from __future__ import annotations

import json
from itertools import chain
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .representations import group_representations_by_subject_epoch
from .train import _should_log


def supervised_contrastive_loss(points: torch.Tensor, labels: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    if points.size(0) < 3 or len(torch.unique(labels)) < 2:
        return points.new_zeros(())
    z = F.normalize(points, dim=-1)
    logits = z @ z.T / temperature
    eye = torch.eye(points.size(0), device=points.device, dtype=torch.bool)
    same = labels.unsqueeze(0) == labels.unsqueeze(1)
    positives = same & ~eye
    logits = logits.masked_fill(eye, -1e9)
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    valid = positives.any(dim=1)
    if not valid.any():
        return points.new_zeros(())
    return -(log_prob[valid] * positives[valid]).sum(dim=1).div(positives[valid].sum(dim=1)).mean()


class OccurrenceContextHead(nn.Module):
    def __init__(self, d_model: int, n_classes: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_set_aggregator_context(
    reps: dict[str, torch.Tensor],
    aggregator: nn.Module,
    output_dir: Path,
    d_model: int,
    device: str = "cpu",
    n_epochs: int = 50,
    lr: float = 1e-3,
    contrastive_weight: float = 1.0,
    verbose: bool = True,
) -> list[dict[str, float]]:
    """Synthetic-only sanity training for occurrence-level N1/N2 separation."""
    output_dir.mkdir(parents=True, exist_ok=True)
    device_t = torch.device(device)
    grouped = list(group_representations_by_subject_epoch(reps).items())
    aggregator.to(device_t)
    head = OccurrenceContextHead(d_model).to(device_t)
    opt = torch.optim.AdamW(list(aggregator.parameters()) + list(head.parameters()), lr=lr, weight_decay=1e-2)
    history = []

    for epoch in range(n_epochs):
        aggregator.train()
        head.train()
        totals = {"loss": 0.0, "ce": 0.0, "contrastive": 0.0, "acc": 0.0}
        n_groups = 0
        for _, group in grouped:
            labels = group["true_context"].to(device_t)
            if len(torch.unique(labels)) < 2:
                continue
            h = group["h"].to(device_t)
            out = aggregator(h)
            u = out.get("U", h)
            logits = head(u)
            ce = F.cross_entropy(logits, labels)
            contrastive = supervised_contrastive_loss(u, labels)
            loss = ce + contrastive_weight * contrastive
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(chain(aggregator.parameters(), head.parameters()), 1.0)
            opt.step()

            totals["loss"] += float(loss.detach())
            totals["ce"] += float(ce.detach())
            totals["contrastive"] += float(contrastive.detach())
            totals["acc"] += float((logits.argmax(dim=-1) == labels).float().mean().detach())
            n_groups += 1

        record = {"epoch": epoch, **{key: value / max(n_groups, 1) for key, value in totals.items()}}
        history.append(record)
        if verbose and _should_log(epoch, n_epochs):
            print(
                f"  aggregator epoch {epoch:03d} loss={record['loss']:.4f} "
                f"acc={record['acc']:.4f}"
            )

    torch.save(aggregator.state_dict(), output_dir / "aggregator.pt")
    torch.save(head.state_dict(), output_dir / "context_head.pt")
    (output_dir / "aggregator_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return history
