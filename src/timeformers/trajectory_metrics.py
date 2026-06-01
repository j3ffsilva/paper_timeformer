from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.metrics import silhouette_score
from scipy import stats

from .representations import group_representations_by_subject_epoch
from .trajectory_losses import linear_cka


CLASS_NAMES = {0: "stable", 1: "drift", 2: "bifurcating", 3: "abrupt"}


def flatten_valid(values: torch.Tensor, valid_mask: torch.Tensor) -> np.ndarray:
    return values[valid_mask].detach().cpu().numpy()


def flattened_targets(p_n1: torch.Tensor, valid_mask: torch.Tensor) -> np.ndarray:
    return p_n1[valid_mask].detach().cpu().numpy()


def probe_p_n1_r2(values: torch.Tensor, p_n1: torch.Tensor, valid_mask: torch.Tensor) -> float:
    x = flatten_valid(values, valid_mask)
    y = flattened_targets(p_n1, valid_mask)
    if len(y) < 4:
        return float("nan")
    split = max(1, int(0.8 * len(y)))
    model = Ridge(alpha=1.0)
    model.fit(x[:split], y[:split])
    pred = model.predict(x[split:])
    return float(r2_score(y[split:], pred))


def teacher_sanity_metrics(encoded: dict[str, torch.Tensor]) -> dict[str, float]:
    cka = linear_cka(encoded["M"], encoded["R"], encoded["valid_mask"])
    return {
        "cka_M_R": float(cka),
        "probe_r2_M": probe_p_n1_r2(encoded["M"], encoded["p_n1"], encoded["valid_mask"]),
        "probe_r2_R": probe_p_n1_r2(encoded["R"], encoded["p_n1"], encoded["valid_mask"]),
    }


def cosine_axis_scores(values: torch.Tensor, p_n1: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    flat = values[valid_mask]
    target = p_n1[valid_mask]
    n1 = flat[target >= 0.5]
    n2 = flat[target < 0.5]
    if n1.numel() == 0 or n2.numel() == 0:
        return torch.full_like(p_n1, float("nan"), dtype=torch.float32)
    proto_n1 = torch.nn.functional.normalize(n1.mean(dim=0), dim=0)
    proto_n2 = torch.nn.functional.normalize(n2.mean(dim=0), dim=0)
    normed = torch.nn.functional.normalize(values, dim=-1)
    return normed @ proto_n1 - normed @ proto_n2


def d2_context_drift_metrics(
    values: torch.Tensor,
    p_n1: torch.Tensor,
    class_id: torch.Tensor,
    valid_mask: torch.Tensor,
    prefix: str,
) -> dict[str, float]:
    scores = cosine_axis_scores(values, p_n1, valid_mask)
    metrics: dict[str, float] = {}
    for cid, name in CLASS_NAMES.items():
        subject_mask = class_id == cid
        if not subject_mask.any():
            continue
        deltas = []
        rhos = []
        for idx in torch.nonzero(subject_mask, as_tuple=False).flatten():
            valid = valid_mask[idx]
            if int(valid.sum()) < 3:
                continue
            s = scores[idx][valid].detach().cpu().numpy()
            target = p_n1[idx][valid].detach().cpu().numpy()
            if np.isnan(s).any():
                continue
            deltas.append(float(s[-1] - s[0]))
            if len(set(np.round(target, 6))) > 1:
                rho, _ = stats.spearmanr(s, target)
                if not np.isnan(rho):
                    rhos.append(float(rho))
        metrics[f"{prefix}_d2_delta_{name}"] = float(np.mean(deltas)) if deltas else float("nan")
        metrics[f"{prefix}_d2_spearman_{name}"] = float(np.mean(rhos)) if rhos else float("nan")
    return metrics


def cka_metric(a: torch.Tensor, b: torch.Tensor, valid_mask: torch.Tensor, prefix: str) -> dict[str, float]:
    return {f"{prefix}_cka": float(linear_cka(a, b, valid_mask))}


def masked_reconstruction_summary(result: dict[str, torch.Tensor], prefix: str = "d5a") -> dict[str, float]:
    losses = result["loss"].detach().cpu()
    class_ids = result["class_id"].detach().cpu()
    metrics = {f"{prefix}_loss": float(losses.mean())}
    for class_id, name in CLASS_NAMES.items():
        mask = class_ids == class_id
        if mask.any():
            metrics[f"{prefix}_loss_{name}"] = float(losses[mask].mean())
    return metrics


@torch.no_grad()
def d6_bimodality_silhouette(
    reps: dict[str, torch.Tensor],
    aggregator: torch.nn.Module | None = None,
    class_id: int = 2,
    min_occurrences: int = 4,
    late_half_only: bool = True,
    device: str = "cpu",
) -> dict[str, float]:
    """Measure Bifurcating bimodality using U for set aggregators or h for mean."""
    grouped = group_representations_by_subject_epoch(reps)
    device_t = torch.device(device)
    if aggregator is not None:
        aggregator.eval()
        aggregator.to(device_t)

    all_scores = []
    all_variances = []
    by_epoch: dict[int, list[float]] = {}
    epochs = sorted({epoch for _, epoch in grouped})
    cutoff = epochs[len(epochs) // 2] if late_half_only and epochs else None

    for (_, epoch), group in grouped.items():
        if cutoff is not None and epoch < cutoff:
            continue
        if int(torch.mode(group["class_id"]).values) != class_id:
            continue
        labels = group["true_context"].detach().cpu().numpy()
        if len(labels) < min_occurrences or len(set(labels.tolist())) < 2:
            continue

        h = group["h"].to(device_t)
        if aggregator is not None:
            out = aggregator(h)
            points = out.get("U", h).detach().cpu().numpy()
        else:
            points = h.detach().cpu().numpy()
        if len(points) <= len(set(labels.tolist())):
            continue
        score = float(silhouette_score(points, labels, metric="cosine"))
        all_scores.append(score)
        all_variances.append(float(np.mean(np.var(points, axis=0))))
        by_epoch.setdefault(epoch, []).append(score)

    metrics = {
        "d6_silhouette": float(np.mean(all_scores)) if all_scores else float("nan"),
        "d6_point_variance": float(np.mean(all_variances)) if all_variances else float("nan"),
        "d6_n_groups": float(len(all_scores)),
    }
    for epoch, scores in sorted(by_epoch.items()):
        metrics[f"d6_silhouette_t{epoch}"] = float(np.mean(scores))
    return metrics
