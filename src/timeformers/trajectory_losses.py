from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def masked_mse(pred: Tensor, target: Tensor, mask: Tensor | None = None) -> Tensor:
    loss = (pred - target).pow(2).mean(dim=-1)
    if mask is None:
        return loss.mean()
    selected = loss[mask]
    if selected.numel() == 0:
        return loss.new_zeros(())
    return selected.mean()


def linear_cka(x: Tensor, y: Tensor, mask: Tensor | None = None, eps: float = 1e-8) -> Tensor:
    if mask is not None:
        x = x[mask]
        y = y[mask]
    else:
        x = x.reshape(-1, x.size(-1))
        y = y.reshape(-1, y.size(-1))
    if x.size(0) < 2:
        return x.new_zeros(())
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    xy = x.T @ y
    xx = x.T @ x
    yy = y.T @ y
    return xy.pow(2).sum() / (torch.sqrt(xx.pow(2).sum() * yy.pow(2).sum()) + eps)


def variance_regularizer(x: Tensor, mask: Tensor | None = None, gamma: float = 1.0, eps: float = 1e-4) -> Tensor:
    if mask is not None:
        x = x[mask]
    else:
        x = x.reshape(-1, x.size(-1))
    if x.size(0) < 2:
        return x.new_zeros(())
    std = torch.sqrt(x.var(dim=0, unbiased=False) + eps)
    return F.relu(gamma - std).mean()


def anti_identity_loss(
    representation: Tensor,
    source: Tensor,
    mask: Tensor | None = None,
    tau_cka: float = 0.7,
    variance_weight: float = 1.0,
) -> tuple[Tensor, dict[str, float]]:
    cka = linear_cka(representation, source, mask)
    copy_penalty = F.relu(cka - tau_cka)
    var_penalty = variance_regularizer(representation, mask)
    loss = copy_penalty + variance_weight * var_penalty
    return loss, {
        "cka": float(cka.detach()),
        "anti_copy": float(copy_penalty.detach()),
        "variance": float(var_penalty.detach()),
    }
