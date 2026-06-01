#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from timeformers.corpus import generate_examples, write_examples, write_trajectories
from timeformers.dataset import MLMDataset
from timeformers.metrics import extract_representations, mlm_accuracy, trajectory_metrics
from timeformers.models import build_model
from timeformers.train import Trainer


MODEL_SPECS = {
    "Static": ("Static", 0.0),
    "StaticTraj": ("Static", 1.0),
    "Standard": ("Static", 0.0),
    "StandardTraj": ("Static", 1.0),
    "Additive": ("Additive", 0.0),
    "TokenTime": ("TokenTime", 0.0),
    "FiLM": ("FiLM", 0.0),
    "TokenTimeTraj": ("TokenTime", 1.0),
    "FiLMTraj": ("FiLM", 1.0),
}


def parse_csv(value: str, cast):
    return [cast(v.strip()) for v in value.split(",") if v.strip()]


def mean_sd(values: list[float]) -> tuple[float, float]:
    arr = np.array([v for v in values if not math.isnan(float(v))], dtype=float)
    if len(arr) == 0:
        return float("nan"), float("nan")
    sd = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    return float(arr.mean()), sd


def run_one(args, fidelity: float, seed: int, model_spec: str) -> dict:
    base_model, default_lambda = MODEL_SPECS[model_spec]
    lambda_traj = args.lambda_traj if model_spec.endswith("Traj") else default_lambda
    run_dir = Path(args.output_dir) / f"f{fidelity:.3f}".replace(".", "p") / f"seed_{seed:04d}" / model_spec

    rows, trajectories = generate_examples(
        seed=seed,
        fidelity=fidelity,
        examples_per_subject_epoch=args.examples_per_subject_epoch,
    )
    data_dir = run_dir.parent / "data"
    if not (data_dir / "corpus.tsv").exists():
        write_examples(rows, data_dir / "corpus.tsv")
        write_trajectories(trajectories, data_dir / "trajectories.json")

    train_ds = MLMDataset(rows, split="train", seed=seed)
    test_ds = MLMDataset(rows, split="test", seed=seed + 99)
    eval_ds = MLMDataset(rows, split=None, seed=seed + 123)
    model = build_model(base_model, d_model=args.d_model, n_layers=args.layers, n_heads=args.heads, d_ff=args.d_ff)
    proto_mode = args.traj_prototypes if lambda_traj else "none"
    loss_mode = args.traj_loss if lambda_traj else "none"
    if not args.quiet:
        print(
            f"\n=== fidelity={fidelity:.3f} seed={seed} model={model_spec} "
            f"lambda={lambda_traj:g} loss={loss_mode} proto={proto_mode} ==="
        )
    Trainer(model, run_dir, device=args.device).train(
        train_ds,
        test_ds,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=seed,
        lambda_traj=lambda_traj,
        traj_prototypes=args.traj_prototypes,
        traj_loss=args.traj_loss,
        traj_margin=args.traj_margin,
        verbose=not args.quiet,
    )

    reps = extract_representations(model, eval_ds, device=args.device)
    metrics = trajectory_metrics(reps)
    metrics["mlm_accuracy"] = mlm_accuracy(model, test_ds, device=args.device)
    row = {
        "fidelity": fidelity,
        "noise": 1.0 - fidelity,
        "seed": seed,
        "model": model_spec,
        "base_model": base_model,
        "lambda_traj": lambda_traj,
        "traj_loss": loss_mode,
        "traj_prototypes": proto_mode,
        **metrics,
    }
    (run_dir / "metrics.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the synthetic Timeformer paper2 pilot.")
    parser.add_argument("--models", default="TokenTime,FiLM,TokenTimeTraj,FiLMTraj")
    parser.add_argument("--fidelities", default="0.75")
    parser.add_argument("--seeds", default="1000")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--examples-per-subject-epoch", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda-traj", type=float, default=1.0)
    parser.add_argument("--traj-loss", choices=["axis", "rank"], default="axis")
    parser.add_argument("--traj-margin", type=float, default=0.05)
    parser.add_argument("--traj-prototypes", choices=["batch", "global"], default="batch")
    parser.add_argument("--d-model", type=int, default=48)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=96)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="outputs/pilot")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    models = parse_csv(args.models, str)
    fidelities = parse_csv(args.fidelities, float)
    seeds = parse_csv(args.seeds, int)
    unknown = [m for m in models if m not in MODEL_SPECS]
    if unknown:
        raise SystemExit(f"Unknown model specs: {unknown}. Choose from {sorted(MODEL_SPECS)}")

    rows = []
    for fidelity in fidelities:
        for seed in seeds:
            for model_spec in models:
                if args.quiet:
                    print(f"running fidelity={fidelity:.3f} seed={seed} model={model_spec}")
                rows.append(run_one(args, fidelity, seed, model_spec))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "pilot_results.csv"
    fields = sorted({key for row in rows for key in row})
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "pilot_results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    metrics = [
        "mlm_accuracy",
        "spearman_drift",
        "spearman_bifurcating",
        "path_contrast_drift_minus_stable",
        "directed_contrast_drift_minus_stable",
    ]
    summary_rows = []
    for fidelity in sorted({row["fidelity"] for row in rows}):
        for model in models:
            subset = [row for row in rows if row["fidelity"] == fidelity and row["model"] == model]
            if not subset:
                continue
            summary = {"fidelity": fidelity, "model": model, "n": len(subset)}
            for metric in metrics:
                mean, sd = mean_sd([float(row[metric]) for row in subset])
                summary[f"{metric}_mean"] = mean
                summary[f"{metric}_sd"] = sd
            summary_rows.append(summary)

    summary_fields = ["fidelity", "model", "n"] + [f"{metric}_{stat}" for metric in metrics for stat in ("mean", "sd")]
    summary_csv = out_dir / "pilot_summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    (out_dir / "pilot_summary.json").write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")

    focus = ["mlm_accuracy", "spearman_drift", "path_contrast_drift_minus_stable", "directed_contrast_drift_minus_stable", "spearman_bifurcating"]
    print("\n=== Summary ===")
    for row in rows:
        values = " ".join(f"{key}={row.get(key, float('nan')):.4f}" for key in focus)
        print(f"{row['model']:14s} f={row['fidelity']:.3f} seed={row['seed']} {values}")
    print(f"\nWrote {summary_path}")
    print(f"Wrote {summary_csv}")


if __name__ == "__main__":
    main()
