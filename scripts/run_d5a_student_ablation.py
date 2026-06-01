#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from timeformers.aggregator_train import train_set_aggregator_context
from timeformers.aggregators import MeanAggregator, aggregate_subject_periods, build_aggregator
from timeformers.corpus import generate_examples
from timeformers.dataset import MLMDataset
from timeformers.models import build_model
from timeformers.representations import extract_occurrence_representations
from timeformers.train import Trainer
from timeformers.trajectories import TrajectoryDataset, build_trajectory_sequences
from timeformers.trajectory_metrics import masked_reconstruction_summary, teacher_sanity_metrics
from timeformers.trajectory_models import TrajectoryStudent, TrajectoryTeacher
from timeformers.trajectory_train import TrajectoryStudentTrainer, TrajectoryTeacherTrainer


def parse_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def write_csv(rows: list[dict], path: Path) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_aggregated_sequences(args, reps: dict[str, torch.Tensor], output_dir: Path) -> TrajectoryDataset:
    if args.aggregator == "mean":
        aggregator = MeanAggregator()
    else:
        aggregator = build_aggregator(args.aggregator, args.d_model, n_heads=args.heads)
        if args.aggregator == "set" and not args.skip_set_training:
            train_set_aggregator_context(
                reps,
                aggregator,
                output_dir / "set_aggregator",
                d_model=args.d_model,
                device=args.device,
                n_epochs=args.aggregator_epochs,
                lr=args.aggregator_lr,
                contrastive_weight=args.contrastive_weight,
                verbose=not args.quiet,
            )
    aggregated = aggregate_subject_periods(reps, aggregator, device=args.device)
    return TrajectoryDataset(build_trajectory_sequences(aggregated))


def main() -> None:
    parser = argparse.ArgumentParser(description="Fair D5a ablation: same teacher, multiple student variants.")
    parser.add_argument("--students", default="bidirectional,causal,linear")
    parser.add_argument("--aggregator", choices=["mean", "attention", "set"], default="set")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--fidelity", type=float, default=0.75)
    parser.add_argument("--examples-per-subject-epoch", type=int, default=8)
    parser.add_argument("--semantic-epochs", type=int, default=5)
    parser.add_argument("--teacher-epochs", type=int, default=20)
    parser.add_argument("--student-epochs", type=int, default=20)
    parser.add_argument("--aggregator-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--teacher-lr", type=float, default=1e-3)
    parser.add_argument("--student-lr", type=float, default=1e-3)
    parser.add_argument("--aggregator-lr", type=float, default=1e-3)
    parser.add_argument("--contrastive-weight", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--tau-cka", type=float, default=0.7)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--d-traj", type=int, default=16)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/d5a_student_ablation"))
    parser.add_argument("--skip-set-training", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, _ = generate_examples(
        seed=args.seed,
        fidelity=args.fidelity,
        examples_per_subject_epoch=args.examples_per_subject_epoch,
    )
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
    trajectory_ds = build_aggregated_sequences(args, reps, args.output_dir)

    teacher = TrajectoryTeacher(
        d_in=args.d_model,
        d_traj=args.d_traj,
        encoder_variant="bidirectional",
        max_len=trajectory_ds.sequences.values.size(1),
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
    teacher_metrics = teacher_sanity_metrics(teacher_trainer.encode(trajectory_ds, batch_size=args.batch_size))

    rows_out = []
    for student_variant in parse_list(args.students):
        student = TrajectoryStudent(
            d_in=args.d_model,
            d_traj=args.d_traj,
            encoder_variant=student_variant,
            max_len=trajectory_ds.sequences.values.size(1),
        )
        trainer = TrajectoryStudentTrainer(
            student,
            teacher,
            args.output_dir / f"student_{student_variant}",
            device=args.device,
        )
        trainer.train(
            trajectory_ds,
            n_epochs=args.student_epochs,
            batch_size=args.batch_size,
            lr=args.student_lr,
            verbose=not args.quiet,
        )
        result = trainer.evaluate_all_masked_reconstruction(trajectory_ds, batch_size=args.batch_size)
        metrics = {
            "seed": args.seed,
            "fidelity": args.fidelity,
            "aggregator": args.aggregator,
            "student": student_variant,
            **{f"teacher_{key}": value for key, value in teacher_metrics.items()},
            **masked_reconstruction_summary(result, prefix="d5a_all"),
        }
        rows_out.append(metrics)
        (args.output_dir / f"metrics_{student_variant}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    write_csv(rows_out, args.output_dir / "d5a_student_ablation.csv")
    (args.output_dir / "d5a_student_ablation.json").write_text(json.dumps(rows_out, indent=2), encoding="utf-8")
    print(json.dumps(rows_out, indent=2))


if __name__ == "__main__":
    main()
