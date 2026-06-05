from __future__ import annotations

from collections import defaultdict

import torch
from torch import Tensor

from .corpus import SUBJECTS
from .structural_metrics import (
    temporal_directional_advantage,
    temporal_directional_fidelity,
    temporal_event_metrics,
    temporal_path_metrics,
    temporal_shape_error,
)


SCALAR_METRICS = (
    "final_magnitude",
    "path_length",
    "displacement_efficiency",
    "recovery",
    "peak_magnitude",
    "peak_period",
    "step_fidelity",
    "accumulated_fidelity",
    "active_step_count",
    "stable_step_count",
    "active_accumulated_count",
    "shape_error",
    "event_period",
    "observed_peak_period",
    "event_period_error",
    "event_oracle_strength",
    "event_step_magnitude",
    "event_concentration",
    "pre_event_drift",
    "pre_event_drift_ratio",
    "post_event_drift",
    "post_event_drift_ratio",
    "event_fidelity",
    "step_fidelity_advantage",
    "accumulated_fidelity_advantage",
)


def structural_metric_rows(
    regime: str,
    profiles: list[Tensor],
    oracle_profiles: list[Tensor],
    metadata: dict[str, dict[str, float | int | str]],
    *,
    placebo_profiles: list[Tensor] | None = None,
    subjects: list[str] | None = None,
) -> tuple[list[dict], list[dict]]:
    path = temporal_path_metrics(profiles)
    fidelity = temporal_directional_fidelity(profiles, oracle_profiles)
    shape_error = temporal_shape_error(profiles, oracle_profiles)
    event = temporal_event_metrics(profiles, oracle_profiles)
    advantage = (
        temporal_directional_advantage(profiles, placebo_profiles, oracle_profiles)
        if placebo_profiles is not None
        else {}
    )
    scalar_values = {**path, **fidelity, **event, **advantage, "shape_error": shape_error}

    subjects = SUBJECTS if subjects is None else subjects
    if len(subjects) != profiles[0].size(0):
        raise ValueError("subject names must match the number of relational profile rows")
    rows = []
    series_rows = []
    for subject_index, subject in enumerate(subjects):
        base = {
            "regime": regime,
            "subject": subject,
            "subject_idx": subject_index,
            **metadata[subject],
        }
        rows.append(
            {
                **base,
                **{
                    metric: float(scalar_values[metric][subject_index])
                    for metric in SCALAR_METRICS
                    if metric in scalar_values
                },
            }
        )
        for period in range(len(profiles)):
            series_rows.append(
                {
                    **base,
                    "period": period,
                    "accumulated_magnitude": float(path["accumulated_magnitudes"][period, subject_index]),
                    "step_magnitude": (
                        0.0 if period == 0 else float(path["step_magnitudes"][period - 1, subject_index])
                    ),
                }
            )
    return rows, series_rows


def summarize_structural_rows(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["regime"], row["condition"])].append(row)

    summary = []
    for (regime, condition), items in sorted(grouped.items()):
        record = {"regime": regime, "condition": condition, "n": len(items)}
        for metric in SCALAR_METRICS:
            values = [item[metric] for item in items if metric in item]
            if not values:
                continue
            tensor = torch.tensor(values, dtype=torch.float32)
            record[f"{metric}_mean"] = float(tensor.mean())
            record[f"{metric}_median"] = float(tensor.median())
            record[f"{metric}_sd"] = float(tensor.std(unbiased=True)) if len(values) > 1 else 0.0
        summary.append(record)
    return summary
