#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from timeformers.experiment import build_trajectory_dataset, encode_student, prepare_synthetic_representations, write_csv
from timeformers.trajectory_metrics import (
    cka_metric,
    d2_context_drift_metrics,
    d6_bimodality_silhouette,
    masked_reconstruction_summary,
    teacher_sanity_metrics,
)
from timeformers.trajectory_models import TrajectoryStudent, TrajectoryTeacher
from timeformers.trajectory_train import TrajectoryStudentTrainer, TrajectoryTeacherTrainer


def parse_configs(value: str) -> list[tuple[str, str]]:
    configs = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Config must look like AGGREGATOR:TEMPORAL, got {item!r}")
        aggregator, temporal = item.split(":", 1)
        configs.append((aggregator.strip(), temporal.strip()))
    return configs


def run_config(args, reps: dict[str, torch.Tensor], aggregator_name: str, temporal_name: str) -> dict[str, float | str]:
    config_name = f"{aggregator_name}_{temporal_name}"
    config_dir = args.output_dir / config_name
    config_dir.mkdir(parents=True, exist_ok=True)

    aggregator, trajectory_ds = build_trajectory_dataset(args, reps, aggregator_name, config_dir)
    d6 = d6_bimodality_silhouette(
        reps,
        aggregator=None if aggregator_name == "mean" else aggregator,
        device=args.device,
    )
    sequences = trajectory_ds.sequences
    d_aggregated = sequences.values.size(-1)

    teacher = TrajectoryTeacher(
        d_in=d_aggregated,
        d_traj=args.d_traj,
        encoder_variant=temporal_name,
        max_len=sequences.values.size(1),
    )
    teacher_trainer = TrajectoryTeacherTrainer(teacher, config_dir / "teacher", device=args.device)
    teacher_trainer.train(
        trajectory_ds,
        n_epochs=args.teacher_epochs,
        batch_size=args.batch_size,
        lr=args.teacher_lr,
        beta=args.beta,
        tau_cka=args.tau_cka,
        verbose=not args.quiet,
    )
    teacher_encoded = teacher_trainer.encode(trajectory_ds, batch_size=args.batch_size)

    student = TrajectoryStudent(
        d_in=d_aggregated,
        d_traj=args.d_traj,
        encoder_variant=temporal_name,
        max_len=sequences.values.size(1),
    )
    student_trainer = TrajectoryStudentTrainer(student, teacher, config_dir / "student", device=args.device)
    student_trainer.train(
        trajectory_ds,
        n_epochs=args.student_epochs,
        batch_size=args.batch_size,
        lr=args.student_lr,
        verbose=not args.quiet,
    )
    d5a = masked_reconstruction_summary(
        student_trainer.evaluate_masked_reconstruction(trajectory_ds, batch_size=args.batch_size),
        prefix="d5a",
    )
    m_student = encode_student(student, trajectory_ds, args.device, args.batch_size)
    token_time = torch.cat([sequences.values, m_student], dim=-1)

    metrics: dict[str, float | str] = {
        "config": config_name,
        "aggregator": aggregator_name,
        "temporal_encoder": temporal_name,
        "seed": args.seed,
        "fidelity": args.fidelity,
        **{f"teacher_{k}": v for k, v in teacher_sanity_metrics(teacher_encoded).items()},
        **d6,
        **d5a,
        **d2_context_drift_metrics(sequences.values, sequences.p_n1, sequences.class_id, sequences.valid_mask, "R"),
        **d2_context_drift_metrics(m_student, sequences.p_n1, sequences.class_id, sequences.valid_mask, "m"),
        **d2_context_drift_metrics(token_time, sequences.p_n1, sequences.class_id, sequences.valid_mask, "token_time"),
        **cka_metric(sequences.values, m_student, sequences.valid_mask, "R_m"),
        "d_aggregated": d_aggregated,
    }
    (config_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an end-to-end synthetic trajectory pipeline.")
    parser.add_argument("--configs", default="mean:linear,mean:bidirectional,set:bidirectional")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--fidelity", type=float, default=0.75)
    parser.add_argument("--examples-per-subject-epoch", type=int, default=12)
    parser.add_argument("--semantic-epochs", type=int, default=5)
    parser.add_argument("--teacher-epochs", type=int, default=50)
    parser.add_argument("--student-epochs", type=int, default=50)
    parser.add_argument("--aggregator-epochs", type=int, default=50)
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
    parser.add_argument("--d-model", type=int, default=48)
    parser.add_argument("--d-traj", type=int, default=16)
    parser.add_argument("--num-slots", type=int, default=2)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=96)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/synthetic_pipeline"))
    parser.add_argument("--skip-set-training", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if args.skip_set_training:
        args.set_training = "none"

    reps = prepare_synthetic_representations(args, args.output_dir, save_artifacts=True)

    results = [run_config(args, reps, aggregator, temporal) for aggregator, temporal in parse_configs(args.configs)]
    write_csv(results, args.output_dir / "pipeline_results.csv")
    (args.output_dir / "pipeline_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
