from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import torch

from .aggregator_ssl import train_set_aggregator_ssl
from .aggregator_train import train_set_aggregator_context
from .aggregators import MeanAggregator, aggregate_subject_periods, build_aggregator
from .corpus import generate_examples, write_examples, write_trajectories
from .dataset import MLMDataset
from .models import build_model
from .representations import extract_occurrence_representations, save_representations
from .train import Trainer
from .trajectories import TrajectoryDataset, build_trajectory_sequences
from .trajectory_models import TrajectoryStudent


def parse_comma_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def write_csv(rows: list[dict], path: Path) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _arg(args: Any, name: str, default: Any = None) -> Any:
    return getattr(args, name, default)


def prepare_synthetic_representations(
    args: Any,
    output_dir: Path,
    *,
    save_artifacts: bool = False,
) -> dict[str, torch.Tensor]:
    torch.manual_seed(args.seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, trajectories = generate_examples(
        seed=args.seed,
        fidelity=args.fidelity,
        examples_per_subject_epoch=args.examples_per_subject_epoch,
    )
    if save_artifacts:
        write_examples(rows, output_dir / "data" / "corpus.tsv")
        write_trajectories(trajectories, output_dir / "data" / "trajectories.json")

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
    Trainer(semantic, output_dir / "semantic_encoder", device=args.device).train(
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
    if save_artifacts:
        save_representations(reps, output_dir / "occurrence_representations.pt")
    return reps


def build_and_train_aggregator(
    args: Any,
    reps: dict[str, torch.Tensor],
    name: str,
    output_dir: Path,
    *,
    artifact_dir: Path | None = None,
):
    if name == "mean":
        return MeanAggregator()

    aggregator = build_aggregator(name, args.d_model, n_heads=args.heads, num_slots=_arg(args, "num_slots", 2))
    set_training = _arg(args, "set_training", "supervised")
    if name in {"set", "set_slots"} and set_training == "supervised":
        train_set_aggregator_context(
            reps,
            aggregator,
            artifact_dir or output_dir / "set_aggregator",
            d_model=args.d_model,
            device=args.device,
            n_epochs=args.aggregator_epochs,
            lr=args.aggregator_lr,
            contrastive_weight=args.contrastive_weight,
            verbose=not args.quiet,
        )
    elif name in {"set", "set_slots"} and set_training == "ssl":
        train_set_aggregator_ssl(
            reps,
            aggregator,
            artifact_dir or output_dir / "set_aggregator_ssl",
            d_model=args.d_model,
            device=args.device,
            n_epochs=args.aggregator_epochs,
            lr=args.aggregator_lr,
            variance_weight=args.ssl_variance_weight,
            consistency_weight=args.ssl_consistency_weight,
            context_weight=args.ssl_context_weight,
            context_temperature=args.ssl_context_temperature,
            context_topk=args.ssl_context_topk,
            subset_fraction=_arg(args, "ssl_subset_fraction", 0.75),
            verbose=not args.quiet,
        )
    return aggregator


def build_trajectory_dataset(
    args: Any,
    reps: dict[str, torch.Tensor],
    aggregator_name: str,
    output_dir: Path,
) -> tuple[torch.nn.Module, TrajectoryDataset]:
    aggregator = build_and_train_aggregator(args, reps, aggregator_name, output_dir)
    aggregated = aggregate_subject_periods(reps, aggregator, device=args.device)
    return aggregator, TrajectoryDataset(build_trajectory_sequences(aggregated))


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
