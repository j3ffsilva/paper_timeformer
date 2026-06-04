from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path


N_EPOCHS = 10
N_SUBJECTS = 40
N_PER_CLASS = 10
SUBJECT_CLASSES = ("stable", "drift", "bifurcating", "abrupt")
CLASS_NAMES = {i: name for i, name in enumerate(SUBJECT_CLASSES)}

SUBJECTS = [f"S{i}" for i in range(1, N_SUBJECTS + 1)]
VERBS_N1 = [f"V{i}" for i in range(1, 5)]
VERBS_N2 = [f"V{i}" for i in range(5, 9)]
OBJS_N1 = [f"O{i}" for i in range(1, 5)]
OBJS_N2 = [f"O{i}" for i in range(5, 9)]


@dataclass(frozen=True)
class Example:
    epoch: int
    subject: str
    verb: str
    obj: str
    true_context: int
    split: str
    p_n1: float
    subject_class: str

    @property
    def sentence(self) -> str:
        return f"{self.subject} {self.verb} {self.obj}"


def subject_class(subject_index: int) -> str:
    class_idx = subject_index // N_PER_CLASS
    return SUBJECT_CLASSES[min(class_idx, len(SUBJECT_CLASSES) - 1)]


def generate_trajectories(seed: int, n_epochs: int = N_EPOCHS) -> dict[str, list[float]]:
    if n_epochs < 5:
        raise ValueError("n_epochs must be at least 5 to generate all trajectory classes")
    rng = random.Random(seed)
    trajectories: dict[str, list[float]] = {}
    stable_values = [0.62 + i * (0.36 / max(N_PER_CLASS - 1, 1)) for i in range(N_PER_CLASS)]
    rng.shuffle(stable_values)

    for i, subject in enumerate(SUBJECTS):
        cls = subject_class(i)
        if cls == "stable":
            p = stable_values[i]
            traj = [p] * n_epochs
        elif cls == "drift":
            start = rng.uniform(0.88, 0.98)
            end = rng.uniform(0.02, 0.18)
            traj = [start + (end - start) * t / (n_epochs - 1) for t in range(n_epochs)]
        elif cls == "bifurcating":
            start = rng.uniform(0.88, 0.98)
            plateau = rng.uniform(0.43, 0.57)
            onset = rng.randint(2, 4)
            transition = rng.randint(2, 4)
            traj = []
            for t in range(n_epochs):
                if t < onset:
                    traj.append(start)
                elif t < onset + transition:
                    alpha = (t - onset + 1) / transition
                    traj.append(start + (plateau - start) * alpha)
                else:
                    traj.append(plateau)
        else:
            start = rng.uniform(0.88, 0.98)
            end = rng.uniform(0.02, 0.12)
            switch = rng.randint(3, min(7, n_epochs - 2))
            traj = [start if t <= switch else end for t in range(n_epochs)]
        trajectories[subject] = [round(max(0.0, min(1.0, v)), 4) for v in traj]
    return trajectories


def _draw_marker(rng: random.Random, true_context: int, fidelity: float) -> tuple[str, str]:
    use_canonical = rng.random() < fidelity
    marker_context = true_context if use_canonical else 1 - true_context
    if marker_context == 0:
        return rng.choice(VERBS_N1), rng.choice(OBJS_N1)
    return rng.choice(VERBS_N2), rng.choice(OBJS_N2)


def generate_examples(
    seed: int,
    fidelity: float = 0.75,
    examples_per_subject_epoch: int = 12,
    test_fraction: float = 0.2,
    n_epochs: int = N_EPOCHS,
) -> tuple[list[Example], dict[str, list[float]]]:
    rng = random.Random(seed)
    trajectories = generate_trajectories(seed, n_epochs=n_epochs)
    rows: list[Example] = []

    for subject_index, subject in enumerate(SUBJECTS):
        cls = subject_class(subject_index)
        for epoch in range(n_epochs):
            p_n1 = trajectories[subject][epoch]
            n_test = max(1, round(examples_per_subject_epoch * test_fraction))
            split_slots = ["test"] * n_test + ["train"] * max(0, examples_per_subject_epoch - n_test)
            rng.shuffle(split_slots)
            for split in split_slots:
                true_context = 0 if rng.random() < p_n1 else 1
                verb, obj = _draw_marker(rng, true_context, fidelity)
                rows.append(Example(epoch, subject, verb, obj, true_context, split, p_n1, cls))
    rng.shuffle(rows)
    return rows, trajectories


def examples_for_epoch(rows: list[Example], epoch: int, split: str | None = None) -> list[Example]:
    return [row for row in rows if row.epoch == epoch and (split is None or row.split == split)]


def generate_fixed_probe_examples(epoch: int = 0) -> list[Example]:
    """Build balanced, deterministic contexts shared by every model checkpoint."""
    rows = []
    contexts = list(zip(VERBS_N1, OBJS_N1)) + list(zip(VERBS_N2, OBJS_N2))
    for subject_index, subject in enumerate(SUBJECTS):
        cls = subject_class(subject_index)
        for context_index, (verb, obj) in enumerate(contexts):
            true_context = 0 if context_index < len(VERBS_N1) else 1
            rows.append(Example(epoch, subject, verb, obj, true_context, "probe", 0.5, cls))
    return rows


def generate_subject_probe_examples(epoch: int = 0) -> list[Example]:
    """Build one neutral context-masked probe per subject."""
    return [
        Example(epoch, subject, VERBS_N1[0], OBJS_N1[0], 0, "probe", 0.5, subject_class(subject_index))
        for subject_index, subject in enumerate(SUBJECTS)
    ]


def write_examples(rows: list[Example], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["epoch", "subject", "verb", "obj", "sentence", "true_context", "split", "p_n1", "subject_class"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            record = asdict(row)
            record["sentence"] = row.sentence
            writer.writerow(record)


def write_trajectories(trajectories: dict[str, list[float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"p_n1": trajectories}, indent=2), encoding="utf-8")
