"""Hand-rolled rank statistics and resampling helpers (no scipy dependency).

Kept dependency-free (numpy only) so they can also be imported from the
Python-3.7 ConSeC evaluation environment.
"""

from __future__ import annotations

import numpy as np


def rankdata(values):
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def spearman(values_a, values_b):
    ranks_a = rankdata(values_a)
    ranks_b = rankdata(values_b)
    centered_a = ranks_a - ranks_a.mean()
    centered_b = ranks_b - ranks_b.mean()
    denominator = np.sqrt(
        np.sum(centered_a ** 2) * np.sum(centered_b ** 2)
    )
    return float(np.sum(centered_a * centered_b) / denominator)


def partial_spearman(values_a, values_b, controls):
    ranks_a = rankdata(values_a)
    ranks_b = rankdata(values_b)
    control_ranks = rankdata(controls)
    design = np.column_stack(
        [np.ones(len(control_ranks), dtype=np.float64), control_ranks]
    )
    residual_a = ranks_a - design.dot(
        np.linalg.lstsq(design, ranks_a, rcond=None)[0]
    )
    residual_b = ranks_b - design.dot(
        np.linalg.lstsq(design, ranks_b, rcond=None)[0]
    )
    denominator = np.sqrt(
        np.sum(residual_a ** 2) * np.sum(residual_b ** 2)
    )
    if denominator == 0:
        return float("nan")
    return float(np.sum(residual_a * residual_b) / denominator)


def jensen_shannon(values_a, values_b):
    a = np.asarray(values_a, dtype=np.float64)
    b = np.asarray(values_b, dtype=np.float64)
    a = a / a.sum()
    b = b / b.sum()
    midpoint = 0.5 * (a + b)

    def kl_divergence(values, reference):
        selected = values > 0
        return float(
            np.sum(values[selected] * np.log(values[selected] / reference[selected]))
        )

    return 0.5 * (
        kl_divergence(a, midpoint) + kl_divergence(b, midpoint)
    )


def sign_flip_p(values, n_permutations: int, seed: int) -> float:
    values = np.asarray(values, dtype=np.float64)
    observed = float(values.mean())
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(n_permutations):
        signs = rng.choice([-1.0, 1.0], size=len(values))
        if float(np.mean(values * signs)) >= observed:
            extreme += 1
    return float((extreme + 1) / (n_permutations + 1))


def bootstrap_mean_ci(values, n_bootstrap: int, seed: int) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[~np.isnan(values)]
    rng = np.random.default_rng(seed)
    estimates = np.asarray(
        [
            np.mean(values[rng.integers(0, len(values), size=len(values))])
            for _ in range(n_bootstrap)
        ]
    )
    return [float(value) for value in np.quantile(estimates, [0.025, 0.975])]
