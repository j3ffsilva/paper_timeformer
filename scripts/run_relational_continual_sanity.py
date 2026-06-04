#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from timeformers.corpus import (
    CLASS_NAMES,
    SUBJECTS,
    examples_for_epoch,
    generate_examples,
    generate_fixed_probe_examples,
    generate_subject_probe_examples,
)
from timeformers.dataset import ContextPairMLMDataset, MLMDataset, RepresentationDataset
from timeformers.models import build_model
from timeformers.relational import (
    centered_cosine_similarity_matrix,
    cosine_similarity_matrix,
    jensen_shannon_similarity_matrix,
    normalized_euclidean_similarity_matrix,
)
from timeformers.relational_metrics import (
    placebo_reference_relational_change,
    relational_change_by_subject,
    representation_cka,
)
from timeformers.representations import extract_occurrence_representations, subject_centroids
from timeformers.train import ContinualPeriodTrainer


RELATION_BUILDERS = {
    "cosine": cosine_similarity_matrix,
    "centered_cosine": centered_cosine_similarity_matrix,
    "normalized_euclidean": normalized_euclidean_similarity_matrix,
    "jensen_shannon": jensen_shannon_similarity_matrix,
}

MODE_SPECS = {
    "prediction_distribution_js": ("prediction_distribution", "jensen_shannon"),
    "subject_prediction_hidden_cosine": ("masked_context", "cosine"),
    "subject_prediction_hidden_centered_cosine": ("masked_context", "centered_cosine"),
    "subject_prediction_hidden_euclidean": ("masked_context", "normalized_euclidean"),
    "subject_only_probes": ("subject", "cosine"),
    "fixed_probes": ("subject", "cosine"),
    "in_corpus": ("subject", "cosine"),
}


def write_csv(rows: list[dict], path: Path) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_static(args) -> torch.nn.Module:
    return build_model(
        "Static",
        d_model=args.d_model,
        n_layers=args.layers,
        n_heads=args.heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
    )


def train_regime(args, period_datasets: list[MLMDataset], val_period_datasets: list[MLMDataset], name: str) -> Path:
    torch.manual_seed(args.seed)
    model = build_static(args)
    output_dir = args.output_dir / name
    ContinualPeriodTrainer(model, output_dir, device=args.device).train(
        period_datasets,
        val_period_datasets=val_period_datasets,
        n_epochs_per_period=args.epochs_per_period,
        n_epochs_first_period=args.base_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        early_stopping_patience=(
            args.early_stopping_patience if args.checkpoint_selection == "best_validation" else None
        ),
        early_stopping_min_delta=args.early_stopping_min_delta,
        restore_best_model=args.checkpoint_selection == "best_validation",
        verbose=not args.quiet,
    )
    return output_dir


def prepare_regime(
    args,
    period_datasets: list[MLMDataset],
    val_period_datasets: list[MLMDataset],
    name: str,
) -> Path:
    output_dir = args.output_dir / name
    checkpoints = [output_dir / f"checkpoint_t{period:02d}.pt" for period in range(args.n_periods)]
    if args.reuse_checkpoints:
        missing = [str(path) for path in checkpoints if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Cannot reuse checkpoints; missing: {missing}")
        return output_dir
    return train_regime(args, period_datasets, val_period_datasets, name)


def build_training_dataset(args, rows, period: int, split: str) -> MLMDataset:
    dataset_type = ContextPairMLMDataset if args.masking == "context_pair" else MLMDataset
    return dataset_type(examples_for_epoch(rows, period, split=split), seed=args.seed + period)


def extract_profile(
    args,
    model: torch.nn.Module,
    dataset: RepresentationDataset,
    output_path: Path,
    target: str = "subject",
    relation: str = "cosine",
) -> dict[str, torch.Tensor]:
    reps = extract_occurrence_representations(
        model,
        dataset,
        batch_size=args.batch_size,
        device=args.device,
        target=target,
    )
    centroids = subject_centroids(reps)
    profile = {
        **centroids,
        "similarities": RELATION_BUILDERS[relation](centroids["h"]),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(profile, output_path)
    return profile


def load_checkpoint_model(args, checkpoint: Path) -> torch.nn.Module:
    model = build_static(args)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    return model


def collect_profiles(
    args,
    rows,
    checkpoint_dir: Path,
    regime: str,
    *,
    frozen_checkpoint: Path | None = None,
    placebo_contexts: bool = False,
) -> dict[str, list[dict[str, torch.Tensor]]]:
    fixed_dataset = RepresentationDataset(generate_fixed_probe_examples())
    subject_only_dataset = ContextPairMLMDataset(generate_subject_probe_examples())
    profiles = {mode: [] for mode in MODE_SPECS}

    for period in range(args.n_periods):
        checkpoint = frozen_checkpoint or checkpoint_dir / f"checkpoint_t{period:02d}.pt"
        model = load_checkpoint_model(args, checkpoint)
        context_period = 0 if placebo_contexts else period
        in_corpus_dataset = RepresentationDataset(examples_for_epoch(rows, context_period))
        datasets = {
            "subject_prediction": subject_only_dataset,
            "subject_only": subject_only_dataset,
            "fixed": fixed_dataset,
            "in_corpus": in_corpus_dataset,
        }
        for mode, (target, relation) in MODE_SPECS.items():
            if mode.startswith("prediction_distribution") or mode.startswith("subject_prediction_hidden"):
                dataset = datasets["subject_prediction"]
            elif mode == "subject_only_probes":
                dataset = datasets["subject_only"]
            elif mode == "fixed_probes":
                dataset = datasets["fixed"]
            else:
                dataset = datasets["in_corpus"]
            profiles[mode].append(
                extract_profile(
                    args,
                    model,
                    dataset,
                    args.output_dir / "profiles" / regime / mode / f"t{period:02d}.pt",
                    target=target,
                    relation=relation,
                )
            )
    return profiles


def comparison_rows(regime: str, mode: str, profiles: list[dict[str, torch.Tensor]], k: int) -> list[dict]:
    rows = []
    comparisons = []
    for period in range(1, len(profiles)):
        comparisons.append(("consecutive", period - 1, period))
        comparisons.append(("from_t0", 0, period))

    for comparison, period_a, period_b in comparisons:
        before = profiles[period_a]
        after = profiles[period_b]
        changes = relational_change_by_subject(before["similarities"], after["similarities"], k=k)
        cka = representation_cka(before["h"], after["h"])
        for index, subject in enumerate(before["subject_idx"].tolist()):
            class_id = int(before["class_id"][index])
            rows.append(
                {
                    "regime": regime,
                    "mode": mode,
                    "comparison": comparison,
                    "from_period": period_a,
                    "to_period": period_b,
                    "subject_idx": subject,
                    "class_id": class_id,
                    "class_name": CLASS_NAMES[class_id],
                    "representation_cka": cka,
                    **{key: float(value[index]) for key, value in changes.items()},
                }
            )
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        key = (row["regime"], row["mode"], row["comparison"], row["to_period"], row["class_name"])
        grouped[key].append(row)

    output = []
    metrics = ("jaccard_change", "spearman_change", "mean_abs_similarity_delta", "representation_cka")
    for key, items in sorted(grouped.items()):
        record = dict(zip(("regime", "mode", "comparison", "to_period", "class_name"), key))
        record["n"] = len(items)
        for metric in metrics:
            values = torch.tensor([item[metric] for item in items])
            record[f"{metric}_mean"] = float(values.mean())
            record[f"{metric}_sd"] = float(values.std(unbiased=True)) if len(values) > 1 else 0.0
        output.append(record)
    return output


def oracle_profiles(trajectories: dict[str, list[float]], n_periods: int) -> dict[str, list[torch.Tensor]]:
    points = [
        torch.tensor([[trajectories[subject][period], 1.0 - trajectories[subject][period]] for subject in SUBJECTS])
        for period in range(n_periods)
    ]
    return {
        mode: [RELATION_BUILDERS[relation](period_points) for period_points in points]
        for mode, (_, relation) in MODE_SPECS.items()
    }


def placebo_reference_rows(
    real_modes: dict[str, list[dict[str, torch.Tensor]]],
    placebo_modes: dict[str, list[dict[str, torch.Tensor]]],
    oracle_modes: dict[str, list[torch.Tensor]],
) -> list[dict]:
    rows = []
    for mode, real_profiles in real_modes.items():
        placebo_profiles = placebo_modes[mode]
        oracle = oracle_modes[mode]
        for period in range(1, len(real_profiles)):
            for comparison, period_a in (("consecutive", period - 1), ("from_t0", 0)):
                changes = placebo_reference_relational_change(
                    real_profiles[period_a]["similarities"],
                    real_profiles[period]["similarities"],
                    placebo_profiles[period_a]["similarities"],
                    placebo_profiles[period]["similarities"],
                    oracle[period_a],
                    oracle[period],
                )
                for index, subject in enumerate(real_profiles[period_a]["subject_idx"].tolist()):
                    class_id = int(real_profiles[period_a]["class_id"][index])
                    rows.append(
                        {
                            "mode": mode,
                            "comparison": comparison,
                            "from_period": period_a,
                            "to_period": period,
                            "subject_idx": subject,
                            "class_id": class_id,
                            "class_name": CLASS_NAMES[class_id],
                            **{key: float(value[index]) for key, value in changes.items()},
                        }
                    )
    return rows


def summarize_placebo_reference(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        key = (row["mode"], row["comparison"], row["to_period"], row["class_name"])
        grouped[key].append(row)

    output = []
    metrics = (
        "observed_mean_abs_similarity_delta",
        "placebo_mean_abs_similarity_delta",
        "observed_minus_placebo_magnitude",
        "excess_mean_abs_similarity_delta",
        "oracle_mean_abs_similarity_delta",
        "observed_oracle_direction_cosine",
        "placebo_oracle_direction_cosine",
        "oracle_direction_advantage",
        "excess_oracle_direction_cosine",
    )
    for key, items in sorted(grouped.items()):
        record = dict(zip(("mode", "comparison", "to_period", "class_name"), key))
        record["n"] = len(items)
        for metric in metrics:
            values = torch.tensor([item[metric] for item in items])
            record[f"{metric}_mean"] = float(values.mean())
            record[f"{metric}_sd"] = float(values.std(unbiased=True)) if len(values) > 1 else 0.0
        output.append(record)
    return output


def print_focus_summary(summary: list[dict], final_period: int) -> None:
    print(f"\nRelational change from t0 to t{final_period} (mean absolute similarity delta):")
    for row in summary:
        if (
            row["comparison"] == "from_t0"
            and row["to_period"] == final_period
            and row["mode"] == "prediction_distribution_js"
        ):
            print(
                f"  {row['regime']:18s} {row['class_name']:12s} "
                f"{row['mean_abs_similarity_delta_mean']:.4f}"
            )


def print_direction_summary(summary: list[dict], final_period: int) -> None:
    print(f"\nOracle direction from t0 to t{final_period} (prediction distribution + Jensen-Shannon):")
    for row in summary:
        if (
            row["comparison"] == "from_t0"
            and row["to_period"] == final_period
            and row["mode"] == "prediction_distribution_js"
        ):
            print(
                f"  {row['class_name']:12s} "
                f"observed={row['observed_oracle_direction_cosine_mean']:+.3f} "
                f"placebo={row['placebo_oracle_direction_cosine_mean']:+.3f} "
                f"advantage={row['oracle_direction_advantage_mean']:+.3f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal continual relational-change falsification experiment.")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--fidelity", type=float, default=0.75)
    parser.add_argument("--examples-per-subject-epoch", type=int, default=12)
    parser.add_argument("--n-periods", type=int, default=10)
    parser.add_argument(
        "--base-epochs",
        type=int,
        default=50,
        help="Warm-up epochs for t0 before temporal comparisons begin.",
    )
    parser.add_argument("--epochs-per-period", type=int, default=5)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument(
        "--checkpoint-selection",
        choices=("best_validation", "final"),
        default="best_validation",
        help="Use best validation checkpoint or a fixed-update final checkpoint.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=48)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--masking",
        choices=("context_pair", "single"),
        default="context_pair",
        help="Synthetic MLM corruption. context_pair forces the model to use the subject.",
    )
    parser.add_argument("--k-neighbors", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/relational_continual_sanity"))
    parser.add_argument("--reuse-checkpoints", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / "config.json"
    if args.reuse_checkpoints:
        (args.output_dir / "analysis_config.json").write_text(
            json.dumps(vars(args), indent=2, default=str),
            encoding="utf-8",
        )
    else:
        config_path.write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")

    rows, trajectories = generate_examples(
        seed=args.seed,
        fidelity=args.fidelity,
        examples_per_subject_epoch=args.examples_per_subject_epoch,
        n_epochs=args.n_periods,
    )
    (args.output_dir / "trajectories.json").write_text(json.dumps(trajectories, indent=2), encoding="utf-8")

    real_datasets = [build_training_dataset(args, rows, period, "train") for period in range(args.n_periods)]
    real_val_datasets = [build_training_dataset(args, rows, period, "test") for period in range(args.n_periods)]
    placebo_base = build_training_dataset(args, rows, 0, "train")
    placebo_val_base = build_training_dataset(args, rows, 0, "test")
    placebo_datasets = [placebo_base for _ in range(args.n_periods)]
    placebo_val_datasets = [placebo_val_base for _ in range(args.n_periods)]

    real_dir = prepare_regime(args, real_datasets, real_val_datasets, "continual_real")
    placebo_dir = prepare_regime(args, placebo_datasets, placebo_val_datasets, "continual_placebo")
    frozen_checkpoint = real_dir / "checkpoint_t00.pt"

    profile_groups = {
        "continual_real": collect_profiles(args, rows, real_dir, "continual_real"),
        "continual_placebo": collect_profiles(
            args,
            rows,
            placebo_dir,
            "continual_placebo",
            placebo_contexts=True,
        ),
        "frozen": collect_profiles(
            args,
            rows,
            real_dir,
            "frozen",
            frozen_checkpoint=frozen_checkpoint,
        ),
    }

    result_rows = []
    for regime, modes in profile_groups.items():
        for mode, profiles in modes.items():
            result_rows.extend(comparison_rows(regime, mode, profiles, args.k_neighbors))

    summary = summarize(result_rows)
    placebo_reference = placebo_reference_rows(
        profile_groups["continual_real"],
        profile_groups["continual_placebo"],
        oracle_profiles(trajectories, args.n_periods),
    )
    placebo_reference_summary = summarize_placebo_reference(placebo_reference)
    write_csv(result_rows, args.output_dir / "relational_results.csv")
    write_csv(summary, args.output_dir / "relational_summary.csv")
    write_csv(placebo_reference, args.output_dir / "counterfactual_results.csv")
    write_csv(placebo_reference_summary, args.output_dir / "counterfactual_summary.csv")
    write_csv(placebo_reference, args.output_dir / "placebo_reference_results.csv")
    write_csv(placebo_reference_summary, args.output_dir / "placebo_reference_summary.csv")
    (args.output_dir / "relational_results.json").write_text(json.dumps(result_rows, indent=2), encoding="utf-8")
    (args.output_dir / "relational_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output_dir / "counterfactual_results.json").write_text(
        json.dumps(placebo_reference, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "counterfactual_summary.json").write_text(
        json.dumps(placebo_reference_summary, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "placebo_reference_results.json").write_text(
        json.dumps(placebo_reference, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "placebo_reference_summary.json").write_text(
        json.dumps(placebo_reference_summary, indent=2),
        encoding="utf-8",
    )
    print_focus_summary(summary, args.n_periods - 1)
    print_direction_summary(placebo_reference_summary, args.n_periods - 1)
    print(f"\nWrote results to {args.output_dir}")


if __name__ == "__main__":
    main()
