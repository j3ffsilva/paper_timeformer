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

from timeformers.aggregator_ssl import train_set_aggregator_ssl
from timeformers.aggregator_train import train_set_aggregator_context
from timeformers.aggregators import build_aggregator
from timeformers.corpus import generate_examples
from timeformers.dataset import MLMDataset
from timeformers.models import build_model
from timeformers.representations import extract_occurrence_representations
from timeformers.train import Trainer
from timeformers.trajectory_metrics import d6_bimodality_silhouette


def write_csv(rows: list[dict], path: Path) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare mean, raw Set, SSL Set, and supervised Set on D6.")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--fidelity", type=float, default=0.75)
    parser.add_argument("--examples-per-subject-epoch", type=int, default=8)
    parser.add_argument("--semantic-epochs", type=int, default=5)
    parser.add_argument("--aggregator-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--aggregator-lr", type=float, default=1e-3)
    parser.add_argument("--contrastive-weight", type=float, default=1.0)
    parser.add_argument("--ssl-variance-weight", type=float, default=1.0)
    parser.add_argument("--ssl-consistency-weight", type=float, default=1.0)
    parser.add_argument("--ssl-context-weight", type=float, default=1.0)
    parser.add_argument("--ssl-context-temperature", type=float, default=0.2)
    parser.add_argument("--ssl-context-topk", type=int, default=2)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ssl_aggregator_sanity"))
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

    result_rows = []
    mean_metrics = d6_bimodality_silhouette(reps, aggregator=None, device=args.device)
    result_rows.append({"regime": "mean", "seed": args.seed, **mean_metrics})

    set_raw = build_aggregator("set", args.d_model, n_heads=args.heads)
    result_rows.append({"regime": "set_raw", "seed": args.seed, **d6_bimodality_silhouette(reps, set_raw, device=args.device)})

    set_ssl = build_aggregator("set", args.d_model, n_heads=args.heads)
    train_set_aggregator_ssl(
        reps,
        set_ssl,
        args.output_dir / "set_ssl",
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
    result_rows.append({"regime": "set_ssl", "seed": args.seed, **d6_bimodality_silhouette(reps, set_ssl, device=args.device)})

    set_supervised = build_aggregator("set", args.d_model, n_heads=args.heads)
    train_set_aggregator_context(
        reps,
        set_supervised,
        args.output_dir / "set_supervised",
        d_model=args.d_model,
        device=args.device,
        n_epochs=args.aggregator_epochs,
        lr=args.aggregator_lr,
        contrastive_weight=args.contrastive_weight,
        verbose=not args.quiet,
    )
    result_rows.append(
        {"regime": "set_supervised", "seed": args.seed, **d6_bimodality_silhouette(reps, set_supervised, device=args.device)}
    )

    write_csv(result_rows, args.output_dir / "ssl_aggregator_sanity.csv")
    (args.output_dir / "ssl_aggregator_sanity.json").write_text(json.dumps(result_rows, indent=2), encoding="utf-8")
    print(json.dumps(result_rows, indent=2))


if __name__ == "__main__":
    main()
