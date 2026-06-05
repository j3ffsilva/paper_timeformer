#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_relational_continual_sanity import (  # noqa: E402
    build_static,
    build_training_dataset,
    extract_profile,
    prepare_regime,
    write_csv,
)
from timeformers.corpus import SUBJECTS, generate_subject_probe_examples  # noqa: E402
from timeformers.dataset import ContextPairMLMDataset  # noqa: E402
from timeformers.relational import jensen_shannon_similarity_matrix  # noqa: E402
from timeformers.structural_corpus import (  # noqa: E402
    STRUCTURAL_ANCHORS,
    STRUCTURAL_TARGETS,
    generate_structural_examples,
    generate_structural_null_examples,
)
from timeformers.structural_experiment import (  # noqa: E402
    structural_metric_rows,
    summarize_structural_rows,
)
from timeformers.train import ContinualPeriodTrainer, _forward_loss  # noqa: E402


def collect_prediction_profiles(
    args,
    checkpoint_dir: Path,
    regime: str,
    *,
    frozen_checkpoint: Path | None = None,
) -> list[dict[str, torch.Tensor]]:
    probe = ContextPairMLMDataset(generate_subject_probe_examples())
    profiles = []
    for period in range(args.n_periods):
        checkpoint = frozen_checkpoint or checkpoint_dir / f"checkpoint_t{period:02d}.pt"
        from run_relational_continual_sanity import load_checkpoint_model

        model = load_checkpoint_model(args, checkpoint)
        profiles.append(
            extract_profile(
                args,
                model,
                probe,
                args.output_dir / "profiles" / regime / "prediction_distribution_js" / f"t{period:02d}.pt",
                target="prediction_distribution",
                relation="jensen_shannon",
            )
        )
    return profiles


def oracle_profiles(trajectories: dict[str, list[float]], n_periods: int) -> list[torch.Tensor]:
    return [
        jensen_shannon_similarity_matrix(
            torch.tensor(
                [[trajectories[subject][period], 1.0 - trajectories[subject][period]] for subject in SUBJECTS],
                dtype=torch.float32,
            )
        )
        for period in range(n_periods)
    ]


def target_anchor_profiles(profiles: list[torch.Tensor]) -> list[torch.Tensor]:
    target_indices = torch.tensor([SUBJECTS.index(subject) for subject in STRUCTURAL_TARGETS])
    anchor_indices = torch.tensor([SUBJECTS.index(subject) for subject in STRUCTURAL_ANCHORS])
    return [profile[target_indices][:, anchor_indices] for profile in profiles]


def build_period_datasets(args, rows):
    train = [build_training_dataset(args, rows, period, "train") for period in range(args.n_periods)]
    validation = [build_training_dataset(args, rows, period, "test") for period in range(args.n_periods)]
    return train, validation


def train_independent_period_models(args, period_datasets, val_period_datasets) -> Path:
    output_dir = args.output_dir / "independent_period"
    checkpoints = [output_dir / f"checkpoint_t{period:02d}.pt" for period in range(args.n_periods)]
    if args.reuse_checkpoints and all(path.exists() for path in checkpoints):
        return output_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    all_history = []
    for period in range(args.n_periods):
        model = build_static(args)
        period_dir = output_dir / f"period_{period:02d}"
        epochs = args.base_epochs if period == 0 else args.epochs_per_period
        history = ContinualPeriodTrainer(model, period_dir, device=args.device).train(
            [period_datasets[period]],
            val_period_datasets=[val_period_datasets[period]],
            n_epochs_per_period=epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed + period,
            early_stopping_patience=(
                args.early_stopping_patience if args.checkpoint_selection == "best_validation" else None
            ),
            early_stopping_min_delta=args.early_stopping_min_delta,
            restore_best_model=args.checkpoint_selection == "best_validation",
            verbose=not args.quiet,
        )
        source = period_dir / "checkpoint_t00.pt"
        state_source = period_dir / "training_state_t00.pt"
        target = output_dir / f"checkpoint_t{period:02d}.pt"
        state_target = output_dir / f"training_state_t{period:02d}.pt"
        target.write_bytes(source.read_bytes())
        state_target.write_bytes(state_source.read_bytes())
        period_summary = json.loads((period_dir / "period_summaries.json").read_text(encoding="utf-8"))[0]
        period_summary["period"] = period
        period_summary["independent_seed"] = args.seed + period
        summaries.append(period_summary)
        for record in history:
            adjusted = dict(record)
            adjusted["period"] = period
            adjusted["independent_seed"] = args.seed + period
            all_history.append(adjusted)

    (output_dir / "period_summaries.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    (output_dir / "continual_history.json").write_text(json.dumps(all_history, indent=2), encoding="utf-8")
    return output_dir


def _steps_for_dataset(dataset, batch_size: int) -> int:
    return (len(dataset) + batch_size - 1) // batch_size


def _continuous_steps_through_period(args, period_datasets, period: int) -> int:
    steps = _steps_for_dataset(period_datasets[0], args.batch_size) * args.base_epochs
    for prior_period in range(1, period + 1):
        steps += _steps_for_dataset(period_datasets[prior_period], args.batch_size) * args.epochs_per_period
    return steps


def _train_model_for_exact_steps(args, model, dataset, max_steps: int, seed: int, output_dir: Path) -> list[dict]:
    torch.manual_seed(seed)
    device = torch.device(args.device)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    history = []
    steps_done = 0
    epoch = 0

    while steps_done < max_steps:
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        total_loss = 0.0
        n_batches = 0
        for batch in loader:
            if steps_done >= max_steps:
                break
            optimizer.zero_grad()
            _, loss, _ = _forward_loss(model, batch, device, lambda_traj=0.0)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach())
            n_batches += 1
            steps_done += 1
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / max(n_batches, 1),
                "n_examples": len(dataset),
                "gradient_steps_this_epoch": n_batches,
                "cumulative_gradient_steps": steps_done,
            }
        )
        epoch += 1

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "checkpoint_t00.pt")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "period": 0,
            "selected_epoch": epoch - 1,
            "selected_cumulative_gradient_steps": steps_done,
        },
        output_dir / "training_state_t00.pt",
    )
    (output_dir / "continual_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output_dir / "period_summaries.json").write_text(
        json.dumps(
            [
                {
                    "period": 0,
                    "epochs_run": epoch,
                    "gradient_steps_computed": steps_done,
                    "cumulative_gradient_steps_computed": steps_done,
                    "selected_epoch": epoch - 1,
                    "selected_cumulative_gradient_steps": steps_done,
                    "best_val_loss": None,
                }
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    return history


def train_cumulative_retrain_models(args, period_datasets) -> Path:
    output_dir = args.output_dir / "cumulative_retrain"
    checkpoints = [output_dir / f"checkpoint_t{period:02d}.pt" for period in range(args.n_periods)]
    if args.reuse_checkpoints and all(path.exists() for path in checkpoints):
        return output_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    all_history = []
    for period in range(args.n_periods):
        model = build_static(args)
        period_dir = output_dir / f"period_{period:02d}"
        cumulative_dataset = ConcatDataset(period_datasets[: period + 1])
        target_steps = _continuous_steps_through_period(args, period_datasets, period)
        history = _train_model_for_exact_steps(
            args,
            model,
            cumulative_dataset,
            target_steps,
            seed=args.seed + 10_000 + period,
            output_dir=period_dir,
        )
        source = period_dir / "checkpoint_t00.pt"
        state_source = period_dir / "training_state_t00.pt"
        target = output_dir / f"checkpoint_t{period:02d}.pt"
        state_target = output_dir / f"training_state_t{period:02d}.pt"
        target.write_bytes(source.read_bytes())
        state_target.write_bytes(state_source.read_bytes())
        summaries.append(
            {
                "period": period,
                "epochs_run": len(history),
                "gradient_steps_computed": target_steps,
                "cumulative_gradient_steps_computed": target_steps,
                "selected_epoch": len(history) - 1,
                "selected_cumulative_gradient_steps": target_steps,
                "best_val_loss": None,
                "cumulative_seed": args.seed + 10_000 + period,
                "n_examples": len(cumulative_dataset),
            }
        )
        for record in history:
            adjusted = dict(record)
            adjusted["period"] = period
            adjusted["cumulative_seed"] = args.seed + 10_000 + period
            all_history.append(adjusted)

    (output_dir / "period_summaries.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    (output_dir / "continual_history.json").write_text(json.dumps(all_history, indent=2), encoding="utf-8")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment P: temporal form of relational semantic change.")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--fidelity", type=float, default=0.75)
    parser.add_argument("--examples-per-subject-period", type=int, default=100)
    parser.add_argument("--n-periods", type=int, default=10)
    parser.add_argument(
        "--abrupt-switch-period",
        type=int,
        default=None,
        help="Period where abrupt_persistent targets switch to the alternate state.",
    )
    parser.add_argument(
        "--transient-onset-period",
        type=int,
        default=None,
        help="First period where transient targets enter the alternate state.",
    )
    parser.add_argument(
        "--transient-width",
        type=int,
        default=2,
        help="Number of periods transient targets remain in the alternate state.",
    )
    parser.add_argument("--base-epochs", type=int, default=60)
    parser.add_argument("--epochs-per-period", type=int, default=30)
    parser.add_argument("--checkpoint-selection", choices=("best_validation", "final"), default="final")
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=48)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--masking", choices=("context_pair", "single"), default="context_pair")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/structural_relational_experiment"))
    parser.add_argument("--reuse-checkpoints", action="store_true")
    parser.add_argument(
        "--include-independent-period",
        action="store_true",
        help="Train independent per-period models for Experiment A.",
    )
    parser.add_argument(
        "--include-cumulative-retrain",
        action="store_true",
        help="Train cumulative-from-scratch models for Experiment A2.",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_name = "analysis_config.json" if args.reuse_checkpoints else "config.json"
    (args.output_dir / config_name).write_text(
        json.dumps(vars(args), indent=2, default=str),
        encoding="utf-8",
    )

    real_rows, trajectories, metadata = generate_structural_examples(
        seed=args.seed,
        fidelity=args.fidelity,
        examples_per_subject_period=args.examples_per_subject_period,
        n_periods=args.n_periods,
        abrupt_switch_period=args.abrupt_switch_period,
        transient_onset_period=args.transient_onset_period,
        transient_width=args.transient_width,
    )
    null_rows, null_trajectories, _ = generate_structural_null_examples(
        seed=args.seed,
        fidelity=args.fidelity,
        examples_per_subject_period=args.examples_per_subject_period,
        n_periods=args.n_periods,
        abrupt_switch_period=args.abrupt_switch_period,
        transient_onset_period=args.transient_onset_period,
        transient_width=args.transient_width,
    )
    (args.output_dir / "trajectories.json").write_text(json.dumps(trajectories, indent=2), encoding="utf-8")
    (args.output_dir / "null_trajectories.json").write_text(
        json.dumps(null_trajectories, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "subject_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    real_train, real_val = build_period_datasets(args, real_rows)
    null_train, null_val = build_period_datasets(args, null_rows)
    placebo_train = [real_train[0] for _ in range(args.n_periods)]
    placebo_val = [real_val[0] for _ in range(args.n_periods)]

    real_dir = prepare_regime(args, real_train, real_val, "continual_real")
    null_dir = prepare_regime(args, null_train, null_val, "resampled_null")
    placebo_dir = prepare_regime(args, placebo_train, placebo_val, "continual_placebo")
    independent_dir = (
        train_independent_period_models(args, real_train, real_val)
        if args.include_independent_period
        else None
    )
    cumulative_dir = (
        train_cumulative_retrain_models(args, real_train)
        if args.include_cumulative_retrain
        else None
    )
    frozen_checkpoint = real_dir / "checkpoint_t00.pt"

    profile_groups = {
        "continual_real": collect_prediction_profiles(args, real_dir, "continual_real"),
        "resampled_null": collect_prediction_profiles(args, null_dir, "resampled_null"),
        "continual_placebo": collect_prediction_profiles(args, placebo_dir, "continual_placebo"),
        "frozen": collect_prediction_profiles(
            args,
            real_dir,
            "frozen",
            frozen_checkpoint=frozen_checkpoint,
        ),
    }
    if independent_dir is not None:
        profile_groups["independent_period"] = collect_prediction_profiles(
            args,
            independent_dir,
            "independent_period",
        )
    if cumulative_dir is not None:
        profile_groups["cumulative_retrain"] = collect_prediction_profiles(
            args,
            cumulative_dir,
            "cumulative_retrain",
        )
    oracle = target_anchor_profiles(oracle_profiles(trajectories, args.n_periods))
    placebo_similarities = target_anchor_profiles(
        [profile["similarities"] for profile in profile_groups["continual_placebo"]]
    )

    metric_rows = []
    series_rows = []
    for regime, profiles in profile_groups.items():
        similarities = target_anchor_profiles([profile["similarities"] for profile in profiles])
        rows, series = structural_metric_rows(
            regime,
            similarities,
            oracle,
            metadata,
            placebo_profiles=placebo_similarities if regime == "continual_real" else None,
            subjects=STRUCTURAL_TARGETS,
        )
        metric_rows.extend(rows)
        series_rows.extend(series)

    summary = summarize_structural_rows(metric_rows)
    write_csv(metric_rows, args.output_dir / "structural_metrics.csv")
    write_csv(series_rows, args.output_dir / "structural_series.csv")
    write_csv(summary, args.output_dir / "structural_summary.csv")
    (args.output_dir / "structural_metrics.json").write_text(json.dumps(metric_rows, indent=2), encoding="utf-8")
    (args.output_dir / "structural_series.json").write_text(json.dumps(series_rows, indent=2), encoding="utf-8")
    (args.output_dir / "structural_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nPrediction-distribution temporal metrics:")
    for row in summary:
        if row["regime"] != "continual_real":
            continue
        print(
            f"  {row['condition']:20s} "
            f"final={row['final_magnitude_mean']:.4f} "
            f"path={row['path_length_mean']:.4f} "
            f"recovery={row['recovery_mean']:.3f} "
            f"shape_error={row['shape_error_mean']:.3f}"
        )
    print(f"\nWrote results to {args.output_dir}")


if __name__ == "__main__":
    main()
