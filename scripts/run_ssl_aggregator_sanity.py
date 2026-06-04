#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from timeformers.aggregators import build_aggregator
from timeformers.experiment import build_and_train_aggregator, prepare_synthetic_representations, write_csv
from timeformers.trajectory_metrics import d6_bimodality_silhouette


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
    parser.add_argument("--ssl-subset-fraction", type=float, default=0.75)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--num-slots", type=int, default=2)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ssl_aggregator_sanity"))
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    reps = prepare_synthetic_representations(args, args.output_dir)

    result_rows = []
    mean_metrics = d6_bimodality_silhouette(reps, aggregator=None, device=args.device)
    result_rows.append({"regime": "mean", "seed": args.seed, **mean_metrics})

    set_raw = build_aggregator("set", args.d_model, n_heads=args.heads)
    result_rows.append({"regime": "set_raw", "seed": args.seed, **d6_bimodality_silhouette(reps, set_raw, device=args.device)})

    set_slots_raw = build_aggregator("set_slots", args.d_model, n_heads=args.heads, num_slots=args.num_slots)
    result_rows.append(
        {"regime": "set_slots_raw", "seed": args.seed, **d6_bimodality_silhouette(reps, set_slots_raw, device=args.device)}
    )

    args.set_training = "ssl"
    set_ssl = build_and_train_aggregator(
        args,
        reps,
        "set",
        args.output_dir,
        artifact_dir=args.output_dir / "set_ssl",
    )
    result_rows.append({"regime": "set_ssl", "seed": args.seed, **d6_bimodality_silhouette(reps, set_ssl, device=args.device)})

    set_slots_ssl = build_and_train_aggregator(
        args,
        reps,
        "set_slots",
        args.output_dir,
        artifact_dir=args.output_dir / "set_slots_ssl",
    )
    result_rows.append(
        {"regime": "set_slots_ssl", "seed": args.seed, **d6_bimodality_silhouette(reps, set_slots_ssl, device=args.device)}
    )

    args.set_training = "supervised"
    set_supervised = build_and_train_aggregator(
        args,
        reps,
        "set",
        args.output_dir,
        artifact_dir=args.output_dir / "set_supervised",
    )
    result_rows.append(
        {"regime": "set_supervised", "seed": args.seed, **d6_bimodality_silhouette(reps, set_supervised, device=args.device)}
    )

    write_csv(result_rows, args.output_dir / "ssl_aggregator_sanity.csv")
    (args.output_dir / "ssl_aggregator_sanity.json").write_text(json.dumps(result_rows, indent=2), encoding="utf-8")
    print(json.dumps(result_rows, indent=2))


if __name__ == "__main__":
    main()
