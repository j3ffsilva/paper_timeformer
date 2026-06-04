#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from timeformers.experiment import (
    build_trajectory_dataset,
    parse_comma_list,
    prepare_synthetic_representations,
    write_csv,
)
from timeformers.trajectory_metrics import masked_reconstruction_summary, teacher_sanity_metrics
from timeformers.trajectory_models import TrajectoryStudent, TrajectoryTeacher
from timeformers.trajectory_train import TrajectoryStudentTrainer, TrajectoryTeacherTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Fair D5a ablation: same teacher, multiple student variants.")
    parser.add_argument("--students", default="bidirectional,causal,linear")
    parser.add_argument("--aggregator", choices=["mean", "attention", "set", "set_slots"], default="set")
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
    parser.add_argument("--set-training", choices=["supervised", "ssl", "none"], default="supervised")
    parser.add_argument("--ssl-variance-weight", type=float, default=1.0)
    parser.add_argument("--ssl-consistency-weight", type=float, default=1.0)
    parser.add_argument("--ssl-context-weight", type=float, default=1.0)
    parser.add_argument("--ssl-context-temperature", type=float, default=0.2)
    parser.add_argument("--ssl-context-topk", type=int, default=2)
    parser.add_argument("--ssl-subset-fraction", type=float, default=0.75)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--tau-cka", type=float, default=0.7)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--d-traj", type=int, default=16)
    parser.add_argument("--num-slots", type=int, default=2)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/d5a_student_ablation"))
    parser.add_argument("--skip-set-training", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if args.skip_set_training:
        args.set_training = "none"

    reps = prepare_synthetic_representations(args, args.output_dir)
    _, trajectory_ds = build_trajectory_dataset(args, reps, args.aggregator, args.output_dir)
    d_aggregated = trajectory_ds.sequences.values.size(-1)

    teacher = TrajectoryTeacher(
        d_in=d_aggregated,
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
    for student_variant in parse_comma_list(args.students):
        student = TrajectoryStudent(
            d_in=d_aggregated,
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
            "set_training": args.set_training,
            "d_aggregated": d_aggregated,
            "num_slots": args.num_slots if args.aggregator == "set_slots" else 1,
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
