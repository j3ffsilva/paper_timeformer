from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def _stack_profiles(profiles: list[Tensor]) -> Tensor:
    if len(profiles) < 2:
        raise ValueError("at least two relational profiles are required")
    shape = profiles[0].shape
    if len(shape) != 2:
        raise ValueError("relational profiles must be matrices")
    if any(profile.shape != shape for profile in profiles):
        raise ValueError("all relational profiles must have matching shapes")
    return torch.stack(profiles)


def _relational_vectors(stacked: Tensor) -> Tensor:
    if stacked.size(1) != stacked.size(2):
        return stacked
    n_subjects = stacked.size(1)
    mask = ~torch.eye(n_subjects, dtype=torch.bool, device=stacked.device)
    return stacked[:, mask].reshape(stacked.size(0), n_subjects, n_subjects - 1)


def _trajectory_deltas(profiles: list[Tensor]) -> tuple[Tensor, Tensor]:
    relational_vectors = _relational_vectors(_stack_profiles(profiles))
    accumulated = relational_vectors - relational_vectors[0:1]
    steps = relational_vectors[1:] - relational_vectors[:-1]
    return accumulated, steps


def temporal_path_metrics(profiles: list[Tensor]) -> dict[str, Tensor]:
    """Measure final displacement, path activity, efficiency, and recovery."""
    accumulated, steps = _trajectory_deltas(profiles)
    accumulated_magnitudes = accumulated.abs().mean(dim=-1)
    step_magnitudes = steps.abs().mean(dim=-1)
    final_magnitude = accumulated_magnitudes[-1]
    path_length = step_magnitudes.sum(dim=0)
    eps = torch.finfo(path_length.dtype).eps
    peak_magnitude, peak_period = accumulated_magnitudes.max(dim=0)

    return {
        "accumulated_magnitudes": accumulated_magnitudes,
        "step_magnitudes": step_magnitudes,
        "final_magnitude": final_magnitude,
        "path_length": path_length,
        "displacement_efficiency": final_magnitude / path_length.clamp_min(eps),
        "recovery": 1.0 - final_magnitude / peak_magnitude.clamp_min(eps),
        "peak_magnitude": peak_magnitude,
        "peak_period": peak_period,
    }


def _mean_valid_cosine(observed: Tensor, oracle: Tensor) -> tuple[Tensor, Tensor]:
    eps = torch.finfo(oracle.dtype).eps
    valid = oracle.norm(dim=-1) > eps
    cosine = F.cosine_similarity(observed, oracle, dim=-1)
    valid_count = valid.sum(dim=0)
    fidelity = (cosine * valid).sum(dim=0) / valid_count.clamp_min(1)
    return fidelity, valid_count


def temporal_directional_fidelity(
    observed_profiles: list[Tensor],
    oracle_profiles: list[Tensor],
) -> dict[str, Tensor]:
    """Compare observed and oracle directions for stepwise and accumulated change."""
    if len(observed_profiles) != len(oracle_profiles):
        raise ValueError("observed and oracle trajectories must have the same length")
    observed_accumulated, observed_steps = _trajectory_deltas(observed_profiles)
    oracle_accumulated, oracle_steps = _trajectory_deltas(oracle_profiles)
    step_fidelity, active_step_count = _mean_valid_cosine(observed_steps, oracle_steps)
    accumulated_fidelity, active_accumulated_count = _mean_valid_cosine(
        observed_accumulated[1:],
        oracle_accumulated[1:],
    )
    n_steps = observed_steps.size(0)

    return {
        "step_fidelity": step_fidelity,
        "accumulated_fidelity": accumulated_fidelity,
        "active_step_count": active_step_count,
        "stable_step_count": n_steps - active_step_count,
        "active_accumulated_count": active_accumulated_count,
    }


def temporal_directional_advantage(
    observed_profiles: list[Tensor],
    placebo_profiles: list[Tensor],
    oracle_profiles: list[Tensor],
) -> dict[str, Tensor]:
    observed = temporal_directional_fidelity(observed_profiles, oracle_profiles)
    placebo = temporal_directional_fidelity(placebo_profiles, oracle_profiles)
    return {
        "step_fidelity_advantage": observed["step_fidelity"] - placebo["step_fidelity"],
        "accumulated_fidelity_advantage": (
            observed["accumulated_fidelity"] - placebo["accumulated_fidelity"]
        ),
    }


def temporal_shape_error(
    observed_profiles: list[Tensor],
    oracle_profiles: list[Tensor],
) -> Tensor:
    """Compare normalized accumulated-magnitude trajectories per subject."""
    if len(observed_profiles) != len(oracle_profiles):
        raise ValueError("observed and oracle trajectories must have the same length")
    observed = temporal_path_metrics(observed_profiles)["accumulated_magnitudes"]
    oracle = temporal_path_metrics(oracle_profiles)["accumulated_magnitudes"]
    eps = torch.finfo(observed.dtype).eps
    observed_shape = observed / observed.max(dim=0).values.clamp_min(eps)
    oracle_shape = oracle / oracle.max(dim=0).values.clamp_min(eps)
    return (observed_shape - oracle_shape).abs().mean(dim=0)


def temporal_event_metrics(
    observed_profiles: list[Tensor],
    oracle_profiles: list[Tensor],
) -> dict[str, Tensor]:
    """Measure local event recovery around the strongest oracle step."""
    if len(observed_profiles) != len(oracle_profiles):
        raise ValueError("observed and oracle trajectories must have the same length")
    observed_accumulated, observed_steps = _trajectory_deltas(observed_profiles)
    _, oracle_steps = _trajectory_deltas(oracle_profiles)

    observed_step_magnitudes = observed_steps.abs().mean(dim=-1)
    oracle_step_magnitudes = oracle_steps.abs().mean(dim=-1)
    accumulated_magnitudes = observed_accumulated.abs().mean(dim=-1)
    path_length = observed_step_magnitudes.sum(dim=0)
    eps = torch.finfo(path_length.dtype).eps

    event_step = oracle_step_magnitudes.argmax(dim=0)
    observed_peak_step = observed_step_magnitudes.argmax(dim=0)
    subject_index = torch.arange(observed_steps.size(1), device=observed_steps.device)
    event_step_magnitude = observed_step_magnitudes[event_step, subject_index]
    event_oracle_strength = oracle_step_magnitudes[event_step, subject_index]

    pre_event_drift = torch.zeros_like(path_length)
    post_event_drift = torch.zeros_like(path_length)
    for subject in range(observed_steps.size(1)):
        step = int(event_step[subject])
        pre_event_drift[subject] = accumulated_magnitudes[step, subject]
        if step + 1 < observed_step_magnitudes.size(0):
            post_event_drift[subject] = observed_step_magnitudes[step + 1 :, subject].sum()

    event_observed = observed_steps[event_step, subject_index]
    event_oracle = oracle_steps[event_step, subject_index]
    event_fidelity = F.cosine_similarity(event_observed, event_oracle, dim=-1)

    return {
        "event_period": event_step + 1,
        "observed_peak_period": observed_peak_step + 1,
        "event_period_error": (observed_peak_step - event_step).abs(),
        "event_oracle_strength": event_oracle_strength,
        "event_step_magnitude": event_step_magnitude,
        "event_concentration": event_step_magnitude / path_length.clamp_min(eps),
        "pre_event_drift": pre_event_drift,
        "pre_event_drift_ratio": pre_event_drift / path_length.clamp_min(eps),
        "post_event_drift": post_event_drift,
        "post_event_drift_ratio": post_event_drift / path_length.clamp_min(eps),
        "event_fidelity": event_fidelity,
    }
