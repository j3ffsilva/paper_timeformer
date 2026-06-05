from __future__ import annotations

import random

from .corpus import (
    N_EPOCHS,
    OBJS_N1,
    OBJS_N2,
    STRUCTURAL_CONDITIONS,
    SUBJECTS,
    SUBJECT_CLASSES,
    VERBS_N1,
    VERBS_N2,
    Example,
)


N_STRUCTURAL_ANCHORS = 16
STRUCTURAL_ANCHORS = SUBJECTS[:N_STRUCTURAL_ANCHORS]
STRUCTURAL_TARGETS = SUBJECTS[N_STRUCTURAL_ANCHORS:]
N_STRUCTURAL_QUARTETS = len(STRUCTURAL_TARGETS) // len(STRUCTURAL_CONDITIONS)
STRUCTURAL_CLASS_NAMES = {
    len(SUBJECT_CLASSES) + index: condition
    for index, condition in enumerate(STRUCTURAL_CONDITIONS)
}
STRUCTURAL_CLASS_NAMES[len(SUBJECT_CLASSES) + len(STRUCTURAL_CONDITIONS)] = "structural_anchor"


def structural_condition(subject_index: int) -> str:
    if subject_index < N_STRUCTURAL_ANCHORS:
        return "structural_anchor"
    return STRUCTURAL_CONDITIONS[(subject_index - N_STRUCTURAL_ANCHORS) % len(STRUCTURAL_CONDITIONS)]


def structural_quartet(subject_index: int) -> int:
    if subject_index < N_STRUCTURAL_ANCHORS:
        return -1
    return (subject_index - N_STRUCTURAL_ANCHORS) // len(STRUCTURAL_CONDITIONS)


def _gradual(start: float, alternate: float, n_periods: int) -> list[float]:
    return [
        start + (alternate - start) * period / (n_periods - 1)
        for period in range(n_periods)
    ]


def _validate_period_index(value: int, *, n_periods: int, name: str) -> None:
    if value < 1 or value >= n_periods:
        raise ValueError(f"{name} must be in [1, n_periods - 1]")


def _abrupt_persistent(
    start: float,
    alternate: float,
    n_periods: int,
    *,
    switch_period: int | None = None,
) -> list[float]:
    switch = n_periods // 2 if switch_period is None else switch_period
    _validate_period_index(switch, n_periods=n_periods, name="switch_period")
    return [start if period < switch else alternate for period in range(n_periods)]


def _transient(
    start: float,
    alternate: float,
    n_periods: int,
    *,
    onset_period: int | None = None,
    width: int = 2,
) -> list[float]:
    onset = max(1, n_periods // 2 - 1) if onset_period is None else onset_period
    if width < 1:
        raise ValueError("transient_width must be at least 1")
    _validate_period_index(onset, n_periods=n_periods, name="transient_onset_period")
    if onset + width > n_periods:
        raise ValueError("transient_onset_period + transient_width must be <= n_periods")
    return [alternate if onset <= period < onset + width else start for period in range(n_periods)]


def _oscillating(start: float, alternate: float, n_periods: int) -> list[float]:
    return [
        alternate if period < n_periods - 1 and period % 2 == 1 else start
        for period in range(n_periods)
    ]


TRAJECTORY_BUILDERS = {
    "gradual": _gradual,
    "abrupt_persistent": _abrupt_persistent,
    "transient": _transient,
    "oscillating": _oscillating,
}


def generate_structural_trajectories(
    seed: int,
    n_periods: int = N_EPOCHS,
    abrupt_switch_period: int | None = None,
    transient_onset_period: int | None = None,
    transient_width: int = 2,
) -> tuple[dict[str, list[float]], dict[str, dict[str, float | int | str]]]:
    """Generate paired temporal forms with shared endpoints inside each quartet."""
    if n_periods < 6:
        raise ValueError("n_periods must be at least 6")
    if len(STRUCTURAL_TARGETS) % len(STRUCTURAL_CONDITIONS) != 0:
        raise ValueError("structural targets must divide evenly into quartets")

    rng = random.Random(seed)
    trajectories: dict[str, list[float]] = {}
    metadata: dict[str, dict[str, float | int | str]] = {}

    anchor_values = [0.05 + index * (0.90 / (N_STRUCTURAL_ANCHORS - 1)) for index in range(N_STRUCTURAL_ANCHORS)]
    for subject, value in zip(STRUCTURAL_ANCHORS, anchor_values):
        rounded = round(value, 4)
        trajectories[subject] = [rounded] * n_periods
        metadata[subject] = {
            "condition": "structural_anchor",
            "role": "anchor",
            "quartet": -1,
            "direction": "stable",
            "start": rounded,
            "alternate": rounded,
        }

    for quartet in range(N_STRUCTURAL_QUARTETS):
        high = rng.uniform(0.88, 0.98)
        low = rng.uniform(0.02, 0.12)
        direction = "n1_to_n2" if quartet < N_STRUCTURAL_QUARTETS // 2 else "n2_to_n1"
        start, alternate = (high, low) if direction == "n1_to_n2" else (low, high)

        for offset, condition in enumerate(STRUCTURAL_CONDITIONS):
            subject_index = N_STRUCTURAL_ANCHORS + quartet * len(STRUCTURAL_CONDITIONS) + offset
            subject = SUBJECTS[subject_index]
            if condition == "abrupt_persistent":
                values = _abrupt_persistent(
                    start,
                    alternate,
                    n_periods,
                    switch_period=abrupt_switch_period,
                )
            elif condition == "transient":
                values = _transient(
                    start,
                    alternate,
                    n_periods,
                    onset_period=transient_onset_period,
                    width=transient_width,
                )
            else:
                values = TRAJECTORY_BUILDERS[condition](start, alternate, n_periods)
            trajectories[subject] = [round(value, 4) for value in values]
            metadata[subject] = {
                "condition": condition,
                "role": "target",
                "quartet": quartet,
                "direction": direction,
                "start": round(start, 4),
                "alternate": round(alternate, 4),
                "abrupt_switch_period": abrupt_switch_period,
                "transient_onset_period": transient_onset_period,
                "transient_width": transient_width,
            }

    return trajectories, metadata


def _draw_marker(rng: random.Random, true_context: int, fidelity: float) -> tuple[str, str]:
    marker_context = true_context if rng.random() < fidelity else 1 - true_context
    if marker_context == 0:
        return rng.choice(VERBS_N1), rng.choice(OBJS_N1)
    return rng.choice(VERBS_N2), rng.choice(OBJS_N2)


def generate_structural_examples(
    seed: int,
    fidelity: float = 0.75,
    examples_per_subject_period: int = 12,
    test_fraction: float = 0.2,
    n_periods: int = N_EPOCHS,
    abrupt_switch_period: int | None = None,
    transient_onset_period: int | None = None,
    transient_width: int = 2,
) -> tuple[
    list[Example],
    dict[str, list[float]],
    dict[str, dict[str, float | int | str]],
]:
    trajectories, metadata = generate_structural_trajectories(
        seed,
        n_periods=n_periods,
        abrupt_switch_period=abrupt_switch_period,
        transient_onset_period=transient_onset_period,
        transient_width=transient_width,
    )
    rows = sample_structural_examples(
        trajectories,
        metadata,
        sampling_seed=seed,
        fidelity=fidelity,
        examples_per_subject_period=examples_per_subject_period,
        test_fraction=test_fraction,
    )
    return rows, trajectories, metadata


def sample_structural_examples(
    trajectories: dict[str, list[float]],
    metadata: dict[str, dict[str, float | int | str]],
    *,
    sampling_seed: int,
    fidelity: float = 0.75,
    examples_per_subject_period: int = 12,
    test_fraction: float = 0.2,
) -> list[Example]:
    if set(trajectories) != set(SUBJECTS) or set(metadata) != set(SUBJECTS):
        raise ValueError("trajectories and metadata must contain every subject")
    n_periods = {len(values) for values in trajectories.values()}
    if len(n_periods) != 1:
        raise ValueError("all subject trajectories must have the same length")

    rng = random.Random(sampling_seed)
    rows: list[Example] = []

    for subject in SUBJECTS:
        condition = str(metadata[subject]["condition"])
        for period, p_n1 in enumerate(trajectories[subject]):
            n_test = max(1, round(examples_per_subject_period * test_fraction))
            split_slots = ["test"] * n_test + ["train"] * max(0, examples_per_subject_period - n_test)
            rng.shuffle(split_slots)
            for split in split_slots:
                true_context = 0 if rng.random() < p_n1 else 1
                verb, obj = _draw_marker(rng, true_context, fidelity)
                rows.append(Example(period, subject, verb, obj, true_context, split, p_n1, condition))

    rng.shuffle(rows)
    return rows


def generate_structural_null_examples(
    seed: int,
    fidelity: float = 0.75,
    examples_per_subject_period: int = 12,
    test_fraction: float = 0.2,
    n_periods: int = N_EPOCHS,
    sampling_seed: int | None = None,
    abrupt_switch_period: int | None = None,
    transient_onset_period: int | None = None,
    transient_width: int = 2,
) -> tuple[
    list[Example],
    dict[str, list[float]],
    dict[str, dict[str, float | int | str]],
]:
    _, metadata = generate_structural_trajectories(
        seed,
        n_periods=n_periods,
        abrupt_switch_period=abrupt_switch_period,
        transient_onset_period=transient_onset_period,
        transient_width=transient_width,
    )
    trajectories = {
        subject: [float(metadata[subject]["start"])] * n_periods
        for subject in SUBJECTS
    }
    rows = sample_structural_examples(
        trajectories,
        metadata,
        sampling_seed=seed + 100_000 if sampling_seed is None else sampling_seed,
        fidelity=fidelity,
        examples_per_subject_period=examples_per_subject_period,
        test_fraction=test_fraction,
    )
    return rows, trajectories, metadata
