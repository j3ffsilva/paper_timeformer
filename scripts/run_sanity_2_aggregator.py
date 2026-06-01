#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from timeformers.aggregator_train import train_set_aggregator_context
from timeformers.aggregators import build_aggregator
from timeformers.corpus import generate_examples
from timeformers.dataset import MLMDataset
from timeformers.models import build_model
from timeformers.representations import extract_occurrence_representations
from timeformers.train import Trainer
from timeformers.trajectory_metrics import d6_bimodality_silhouette


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Sanity 2: D6 bimodality for aggregators.")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--fidelity", type=float, default=0.75)
    parser.add_argument("--examples-per-subject-epoch", type=int, default=12)
    parser.add_argument("--semantic-epochs", type=int, default=5)
    parser.add_argument("--aggregator-epochs", type=int, default=50)
    parser.add_argument("--aggregator-lr", type=float, default=1e-3)
    parser.add_argument("--contrastive-weight", type=float, default=1.0)
    parser.add_argument("--skip-aggregator-training", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--d-model", type=int, default=48)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=96)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/sanity_2_aggregator"))
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

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

    metrics = {"mean": d6_bimodality_silhouette(reps, aggregator=None, device=args.device)}
    for name in ("attention",):
        aggregator = build_aggregator(name, args.d_model, n_heads=args.heads)
        metrics[name] = d6_bimodality_silhouette(reps, aggregator=aggregator, device=args.device)

    set_aggregator = build_aggregator("set", args.d_model, n_heads=args.heads)
    if not args.skip_aggregator_training:
        train_set_aggregator_context(
            reps,
            set_aggregator,
            args.output_dir / "set_aggregator",
            d_model=args.d_model,
            device=args.device,
            n_epochs=args.aggregator_epochs,
            lr=args.aggregator_lr,
            contrastive_weight=args.contrastive_weight,
            verbose=not args.quiet,
        )
    metrics["set"] = d6_bimodality_silhouette(reps, aggregator=set_aggregator, device=args.device)

    metrics["passes_set_over_mean"] = metrics["set"]["d6_silhouette"] > metrics["mean"]["d6_silhouette"]
    (args.output_dir / "sanity_2_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
