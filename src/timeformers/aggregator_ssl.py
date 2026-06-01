from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .representations import group_representations_by_subject_epoch
from .trajectory_losses import variance_regularizer


class OccurrenceDecoder(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _sample_subset(h: torch.Tensor, min_size: int = 2) -> torch.Tensor:
    if h.size(0) <= min_size:
        return h
    keep = max(min_size, int(round(0.75 * h.size(0))))
    idx = torch.randperm(h.size(0), device=h.device)[:keep]
    return h[idx]


def context_similarity_contrastive_loss(
    points: torch.Tensor,
    context: torch.Tensor,
    temperature: float = 0.2,
    topk: int = 2,
) -> torch.Tensor:
    if points.size(0) < 3:
        return points.new_zeros(())
    k = min(topk, points.size(0) - 1)
    z = F.normalize(points, dim=-1)
    c = F.normalize(context, dim=-1)
    context_sim = c @ c.T
    eye = torch.eye(points.size(0), device=points.device, dtype=torch.bool)
    context_sim = context_sim.masked_fill(eye, -1e9)
    positive_idx = torch.topk(context_sim, k=k, dim=1).indices
    positives = torch.zeros(points.size(0), points.size(0), device=points.device, dtype=torch.bool)
    positives.scatter_(1, positive_idx, True)

    logits = z @ z.T / temperature
    logits = logits.masked_fill(eye, -1e9)
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    return -(log_prob * positives).sum(dim=1).div(positives.sum(dim=1).clamp_min(1)).mean()


def train_set_aggregator_ssl(
    reps: dict[str, torch.Tensor],
    aggregator: nn.Module,
    output_dir: Path,
    d_model: int,
    device: str = "cpu",
    n_epochs: int = 50,
    lr: float = 1e-3,
    variance_weight: float = 1.0,
    consistency_weight: float = 1.0,
    context_weight: float = 1.0,
    context_temperature: float = 0.2,
    context_topk: int = 2,
    verbose: bool = True,
) -> list[dict[str, float]]:
    """Self-supervised set training with local-context similarity as signal."""
    output_dir.mkdir(parents=True, exist_ok=True)
    device_t = torch.device(device)
    grouped = list(group_representations_by_subject_epoch(reps).items())
    aggregator.to(device_t)
    decoder = OccurrenceDecoder(d_model).to(device_t)
    opt = torch.optim.AdamW(list(aggregator.parameters()) + list(decoder.parameters()), lr=lr, weight_decay=1e-2)
    history = []

    for epoch in range(n_epochs):
        aggregator.train()
        decoder.train()
        totals = {"loss": 0.0, "recon": 0.0, "variance": 0.0, "consistency": 0.0, "context": 0.0}
        n_groups = 0
        for _, group in grouped:
            h = group["h"].to(device_t)
            context = group["context"].to(device_t)
            if h.size(0) < 2:
                continue
            out = aggregator(h)
            u = out.get("U", h)
            recon = F.mse_loss(decoder(u), h)
            variance = variance_regularizer(u)
            context_loss = context_similarity_contrastive_loss(
                u,
                context,
                temperature=context_temperature,
                topk=context_topk,
            )

            view_a = _sample_subset(h)
            view_b = _sample_subset(h)
            pool_a = aggregator(view_a)["R"]
            pool_b = aggregator(view_b)["R"]
            consistency = 1.0 - F.cosine_similarity(pool_a, pool_b, dim=0)

            loss = (
                recon
                + variance_weight * variance
                + consistency_weight * consistency
                + context_weight * context_loss
            )
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(list(aggregator.parameters()) + list(decoder.parameters()), 1.0)
            opt.step()

            totals["loss"] += float(loss.detach())
            totals["recon"] += float(recon.detach())
            totals["variance"] += float(variance.detach())
            totals["consistency"] += float(consistency.detach())
            totals["context"] += float(context_loss.detach())
            n_groups += 1

        record = {"epoch": epoch, **{key: value / max(n_groups, 1) for key, value in totals.items()}}
        history.append(record)
        if verbose and (epoch == 0 or epoch == n_epochs - 1 or (epoch + 1) % 10 == 0):
            print(
                f"  aggregator-ssl epoch {epoch:03d} loss={record['loss']:.4f} "
                f"recon={record['recon']:.4f} var={record['variance']:.4f} ctx={record['context']:.4f}"
            )

    torch.save(aggregator.state_dict(), output_dir / "aggregator_ssl.pt")
    torch.save(decoder.state_dict(), output_dir / "decoder.pt")
    (output_dir / "aggregator_ssl_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return history
