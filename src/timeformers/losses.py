from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def mlm_loss(logits: Tensor, labels: Tensor) -> Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100)


def trajectory_axis_loss(
    h_subj: Tensor,
    true_context: Tensor,
    p_n1: Tensor,
    prototypes: tuple[Tensor, Tensor] | None = None,
) -> Tensor:
    """Align subject representation with the planted N1->N2 trajectory.

    Prototypes are detached so the loss trains each example against the current
    batch axis without letting the axis itself collapse into a shortcut.
    """
    if prototypes is None:
        n1_mask = true_context == 0
        n2_mask = true_context == 1
        if int(n1_mask.sum()) < 2 or int(n2_mask.sum()) < 2:
            return h_subj.new_zeros(())
        detached = h_subj.detach()
        proto_n1 = detached[n1_mask].mean(dim=0, keepdim=True)
        proto_n2 = detached[n2_mask].mean(dim=0, keepdim=True)
    else:
        proto_n1, proto_n2 = prototypes
        proto_n1 = proto_n1.to(device=h_subj.device, dtype=h_subj.dtype).view(1, -1)
        proto_n2 = proto_n2.to(device=h_subj.device, dtype=h_subj.dtype).view(1, -1)

    score = F.cosine_similarity(h_subj, proto_n1, dim=-1) - F.cosine_similarity(h_subj, proto_n2, dim=-1)
    target = 2.0 * p_n1 - 1.0
    return F.mse_loss(score, target)


def trajectory_ranking_loss(
    h_subj: Tensor,
    true_context: Tensor,
    p_n1: Tensor,
    subject_idx: Tensor,
    prototypes: tuple[Tensor, Tensor] | None = None,
    margin: float = 0.05,
) -> Tensor:
    """Rank same-subject representations by their planted trajectory value."""
    n1_mask = true_context == 0
    n2_mask = true_context == 1
    if prototypes is None:
        if int(n1_mask.sum()) < 2 or int(n2_mask.sum()) < 2:
            return h_subj.new_zeros(())
        detached = h_subj.detach()
        proto_n1 = detached[n1_mask].mean(dim=0, keepdim=True)
        proto_n2 = detached[n2_mask].mean(dim=0, keepdim=True)
    else:
        proto_n1, proto_n2 = prototypes
        proto_n1 = proto_n1.to(device=h_subj.device, dtype=h_subj.dtype).view(1, -1)
        proto_n2 = proto_n2.to(device=h_subj.device, dtype=h_subj.dtype).view(1, -1)

    score = F.cosine_similarity(h_subj, proto_n1, dim=-1) - F.cosine_similarity(h_subj, proto_n2, dim=-1)
    p_delta = p_n1.unsqueeze(1) - p_n1.unsqueeze(0)
    same_subject = subject_idx.unsqueeze(1) == subject_idx.unsqueeze(0)
    ordered = same_subject & (p_delta > 1e-6)
    if not ordered.any():
        return h_subj.new_zeros(())

    score_delta = score.unsqueeze(1) - score.unsqueeze(0)
    adaptive_margin = margin * p_delta[ordered].clamp_min(0.0)
    return F.relu(adaptive_margin - score_delta[ordered]).mean()
