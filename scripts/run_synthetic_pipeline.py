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
from timeformers.aggregator_ssl import train_set_aggregator_ssl
from timeformers.aggregators import MeanAggregator, aggregate_subject_periods, build_aggregator
from timeformers.corpus import generate_examples, write_examples, write_trajectories
from timeformers.dataset import MLMDataset
from timeformers.models import build_model
from timeformers.representations import extract_occurrence_representations, save_representations
from timeformers.train import Trainer
from timeformers.trajectories import TrajectoryDataset, build_trajectory_sequences
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


@torch.no_grad()
def encode_student(student: TrajectoryStudent, dataset: TrajectoryDataset, device: str, batch_size: int) -> torch.Tensor:
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    device_t = torch.device(device)
    student.eval()
    student.to(device_t)
    outputs = []
    for batch in loader:
        values = batch["values"].to(device_t)
        valid_mask = batch["valid_mask"].to(device_t)
        outputs.append(student.encoder(values, valid_mask).cpu())
    return torch.cat(outputs, dim=0)


def build_and_train_aggregator(args, reps: dict[str, torch.Tensor], name: str, config_dir: Path):
    if name == "mean":
        return MeanAggregator()
    aggregator = build_aggregator(name, args.d_model, n_heads=args.heads, num_slots=args.num_slots)
    if name in {"set", "set_slots"} and args.set_training == "supervised":
        train_set_aggregator_context(
            reps,
            aggregator,
            config_dir / "set_aggregator",
            d_model=args.d_model,
            device=args.device,
            n_epochs=args.aggregator_epochs,
            lr=args.aggregator_lr,
            contrastive_weight=args.contrastive_weight,
            verbose=not args.quiet,
        )
    elif name in {"set", "set_slots"} and args.set_training == "ssl":
        train_set_aggregator_ssl(
            reps,
            aggregator,
            config_dir / "set_aggregator_ssl",
            d_model=args.d_model,
            device=args.device,
            n_epochs=args.aggregator_epochs,
            lr=args.aggregator_lr,
            variance_weight=args.ssl_variance_weight,
            consistency_weight=args.ssl_consistency_weight,
            context_weight=args.ssl_context_weight,
            context_temperature=args.ssl_context_temperature,
            context_topk=args.ssl_context_topk,
            verbose=not args.quiet,
        )
    return aggregator


def run_config(args, reps: dict[str, torch.Tensor], aggregator_name: str, temporal_name: str) -> dict[str, float | str]:
    config_name = f"{aggregator_name}_{temporal_name}"
    config_dir = args.output_dir / config_name
    config_dir.mkdir(parents=True, exist_ok=True)

    aggregator = build_and_train_aggregator(args, reps, aggregator_name, config_dir)
    d6 = d6_bimodality_silhouette(
        reps,
        aggregator=None if aggregator_name == "mean" else aggregator,
        device=args.device,
    )
    aggregated = aggregate_subject_periods(reps, aggregator, device=args.device)
    sequences = build_trajectory_sequences(aggregated)
    trajectory_ds = TrajectoryDataset(sequences)
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
    }
    (config_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def write_csv(rows: list[dict[str, float | str]], path: Path) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


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

    torch.manual_seed(args.seed)
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

    results = [run_config(args, reps, aggregator, temporal) for aggregator, temporal in parse_configs(args.configs)]
    write_csv(results, args.output_dir / "pipeline_results.csv")
    (args.output_dir / "pipeline_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
