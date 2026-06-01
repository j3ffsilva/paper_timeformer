#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from timeformers.aggregators import MeanAggregator, aggregate_subject_periods
from timeformers.corpus import generate_examples, write_examples, write_trajectories
from timeformers.dataset import MLMDataset
from timeformers.models import build_model
from timeformers.representations import extract_occurrence_representations, save_representations
from timeformers.train import Trainer
from timeformers.trajectories import TrajectoryDataset, build_trajectory_sequences
from timeformers.trajectory_metrics import teacher_sanity_metrics
from timeformers.trajectory_models import TrajectoryTeacher
from timeformers.trajectory_train import TrajectoryTeacherTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Sanity 0 for masked trajectory distillation.")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--fidelity", type=float, default=0.75)
    parser.add_argument("--examples-per-subject-epoch", type=int, default=6)
    parser.add_argument("--semantic-epochs", type=int, default=5)
    parser.add_argument("--teacher-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--teacher-lr", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--tau-cka", type=float, default=0.7)
    parser.add_argument("--d-model", type=int, default=48)
    parser.add_argument("--d-traj", type=int, default=16)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=96)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/sanity_0_teacher"))
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, trajectories = generate_examples(
        seed=args.seed,
        fidelity=args.fidelity,
        examples_per_subject_epoch=args.examples_per_subject_epoch,
    )
    write_examples(rows, args.output_dir / "data" / "corpus.tsv")
    write_trajectories(trajectories, args.output_dir / "data" / "trajectories.json")

    train_ds = MLMDataset(rows, split="train", seed=args.seed)
    test_ds = MLMDataset(rows, split="test", seed=args.seed + 99)
    eval_ds = MLMDataset(rows, split=None, seed=args.seed + 123)

    semantic = build_model(
        "TokenTime",
        d_model=args.d_model,
        n_layers=args.layers,
        n_heads=args.heads,
        d_ff=args.d_ff,
    )
    Trainer(semantic, args.output_dir / "semantic_encoder", device=args.device).train(
        train_ds,
        test_ds,
        n_epochs=args.semantic_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        lambda_traj=0.0,
        verbose=not args.quiet,
    )

    reps = extract_occurrence_representations(semantic, eval_ds, batch_size=args.batch_size, device=args.device)
    save_representations(reps, args.output_dir / "occurrence_representations.pt")
    aggregated = aggregate_subject_periods(reps, MeanAggregator(), device=args.device)
    sequences = build_trajectory_sequences(aggregated)
    trajectory_ds = TrajectoryDataset(sequences)

    teacher = TrajectoryTeacher(
        d_in=args.d_model,
        d_traj=args.d_traj,
        encoder_variant="linear",
        max_len=sequences.values.size(1),
    )
    teacher_trainer = TrajectoryTeacherTrainer(teacher, args.output_dir / "teacher", device=args.device)
    teacher_trainer.train(
        trajectory_ds,
        n_epochs=args.teacher_epochs,
        batch_size=args.batch_size,
        lr=args.teacher_lr,
        beta=args.beta,
        tau_cka=args.tau_cka,
        verbose=not args.quiet,
    )

    encoded = teacher_trainer.encode(trajectory_ds, batch_size=args.batch_size)
    metrics = teacher_sanity_metrics(encoded)
    metrics["passes_cka"] = metrics["cka_M_R"] < args.tau_cka
    metrics["passes_probe"] = metrics["probe_r2_M"] > metrics["probe_r2_R"]
    (args.output_dir / "sanity_0_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
