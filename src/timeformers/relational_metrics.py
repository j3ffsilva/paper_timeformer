from __future__ import annotations

import math

import torch
from scipy import stats
from torch import Tensor

from .relational import relational_delta, topk_neighbors
from .trajectory_losses import linear_cka


def relational_change_by_subject(similarities_before: Tensor, similarities_after: Tensor, k: int = 5) -> dict[str, Tensor]:
    if similarities_before.shape != similarities_after.shape:
        raise ValueError("similarity matrices must have matching shapes")

    neighbors_before = topk_neighbors(similarities_before, k)
    neighbors_after = topk_neighbors(similarities_after, k)
    delta = relational_delta(similarities_before, similarities_after)
    jaccard_change = []
    spearman_change = []
    mean_abs_delta = []

    for subject in range(similarities_before.size(0)):
        before_set = set(neighbors_before[subject].tolist())
        after_set = set(neighbors_after[subject].tolist())
        union = before_set | after_set
        overlap = len(before_set & after_set) / max(len(union), 1)
        jaccard_change.append(1.0 - overlap)

        mask = torch.ones(similarities_before.size(0), dtype=torch.bool)
        mask[subject] = False
        before = similarities_before[subject, mask].detach().cpu().numpy()
        after = similarities_after[subject, mask].detach().cpu().numpy()
        rho = float(stats.spearmanr(before, after).statistic)
        spearman_change.append((1.0 - rho) / 2.0 if not math.isnan(rho) else 1.0)
        mean_abs_delta.append(float(delta[subject, mask].abs().mean()))

    return {
        "jaccard_change": torch.tensor(jaccard_change),
        "spearman_change": torch.tensor(spearman_change),
        "mean_abs_similarity_delta": torch.tensor(mean_abs_delta),
    }


def representation_cka(points_before: Tensor, points_after: Tensor) -> float:
    return float(linear_cka(points_before, points_after))


def placebo_reference_relational_change(
    observed_before: Tensor,
    observed_after: Tensor,
    placebo_before: Tensor,
    placebo_after: Tensor,
    oracle_before: Tensor,
    oracle_after: Tensor,
) -> dict[str, Tensor]:
    shapes = {
        observed_before.shape,
        observed_after.shape,
        placebo_before.shape,
        placebo_after.shape,
        oracle_before.shape,
        oracle_after.shape,
    }
    if len(shapes) != 1:
        raise ValueError("all relational profiles must have matching shapes")

    excess = relational_delta(observed_before, observed_after) - relational_delta(placebo_before, placebo_after)
    oracle = relational_delta(oracle_before, oracle_after)
    n_subjects = excess.size(0)
    mask = ~torch.eye(n_subjects, dtype=torch.bool, device=excess.device)
    excess_without_self = excess[mask].reshape(n_subjects, n_subjects - 1)
    observed_without_self = relational_delta(observed_before, observed_after)[mask].reshape(n_subjects, n_subjects - 1)
    placebo_without_self = relational_delta(placebo_before, placebo_after)[mask].reshape(n_subjects, n_subjects - 1)
    oracle_without_self = oracle[mask].reshape(n_subjects, n_subjects - 1)

    return {
        "observed_mean_abs_similarity_delta": observed_without_self.abs().mean(dim=1),
        "placebo_mean_abs_similarity_delta": placebo_without_self.abs().mean(dim=1),
        "observed_minus_placebo_magnitude": (
            observed_without_self.abs().mean(dim=1) - placebo_without_self.abs().mean(dim=1)
        ),
        "excess_mean_abs_similarity_delta": excess_without_self.abs().mean(dim=1),
        "oracle_mean_abs_similarity_delta": oracle_without_self.abs().mean(dim=1),
        "observed_oracle_direction_cosine": torch.nn.functional.cosine_similarity(
            observed_without_self, oracle_without_self, dim=1
        ),
        "placebo_oracle_direction_cosine": torch.nn.functional.cosine_similarity(
            placebo_without_self, oracle_without_self, dim=1
        ),
        "oracle_direction_advantage": (
            torch.nn.functional.cosine_similarity(observed_without_self, oracle_without_self, dim=1)
            - torch.nn.functional.cosine_similarity(placebo_without_self, oracle_without_self, dim=1)
        ),
        "excess_oracle_direction_cosine": torch.nn.functional.cosine_similarity(
            excess_without_self, oracle_without_self, dim=1
        ),
    }


counterfactual_relational_change = placebo_reference_relational_change
