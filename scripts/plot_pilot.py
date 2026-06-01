#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path

_cache_dir = Path(".matplotlib-cache").resolve()
_cache_dir.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_cache_dir))
os.environ.setdefault("XDG_CACHE_HOME", str(_cache_dir))

import matplotlib.pyplot as plt


MODEL_ORDER = ["Standard", "StandardTraj", "TokenTime", "FiLM", "TokenTimeTraj", "FiLMTraj"]
MODEL_LABELS = {
    "Static": "Static",
    "StaticTraj": "Static + L_traj",
    "Standard": "Standard",
    "StandardTraj": "Standard + L_traj",
    "TokenTime": "Token-Time",
    "FiLM": "FiLM",
    "TokenTimeTraj": "Token-Time + L_traj",
    "FiLMTraj": "FiLM + L_traj",
}
COLORS = {
    "Static": "#9D9DA1",
    "StaticTraj": "#BAB0AC",
    "Standard": "#9D9DA1",
    "StandardTraj": "#BAB0AC",
    "TokenTime": "#4C78A8",
    "FiLM": "#F58518",
    "TokenTimeTraj": "#54A24B",
    "FiLMTraj": "#B279A2",
}
MARKERS = {
    "Static": "v",
    "StaticTraj": "P",
    "Standard": "v",
    "StandardTraj": "P",
    "TokenTime": "o",
    "FiLM": "s",
    "TokenTimeTraj": "^",
    "FiLMTraj": "D",
}


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def fnum(value: str | float) -> float:
    return float(value)


def rows_by_model(rows: list[dict[str, str]], model: str) -> list[dict[str, str]]:
    return sorted([row for row in rows if row["model"] == model], key=lambda r: fnum(r["fidelity"]))


def available_models(rows: list[dict[str, str]]) -> list[str]:
    present = {row["model"] for row in rows}
    ordered = [model for model in MODEL_ORDER if model in present]
    return ordered + sorted(present - set(ordered))


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_metric_curves(summary_rows: list[dict[str, str]], out_dir: Path) -> None:
    specs = [
        ("spearman_drift_mean", "Drift trajectory Spearman", "spearman_drift_by_fidelity"),
        ("spearman_bifurcating_mean", "Bifurcating trajectory Spearman", "spearman_bifurcating_by_fidelity"),
        ("mlm_accuracy_mean", "Masked-token accuracy", "mlm_accuracy_by_fidelity"),
    ]
    for metric, ylabel, stem in specs:
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        for model in available_models(summary_rows):
            model_rows = rows_by_model(summary_rows, model)
            xs = [fnum(row["fidelity"]) for row in model_rows]
            ys = [fnum(row[metric]) for row in model_rows]
            ax.plot(
                xs,
                ys,
                label=MODEL_LABELS.get(model, model),
                color=COLORS.get(model),
                marker=MARKERS.get(model, "o"),
                linewidth=2.0,
                markersize=5.5,
            )
        ax.set_xlabel("Marker fidelity")
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0.0)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, ncols=2)
        save_figure(fig, out_dir, stem)


def plot_paired_deltas(stats_rows: list[dict[str, str]], out_dir: Path) -> None:
    focus = [
        ("spearman_drift", "Drift Spearman delta"),
        ("spearman_bifurcating", "Bifurcating Spearman delta"),
        ("mlm_accuracy", "MLM accuracy delta"),
    ]
    comparisons = ["FiLMTraj-FiLM", "FiLMTraj-TokenTime", "StandardTraj-Standard", "FiLMTraj-StandardTraj"]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.1), sharex=True)
    for ax, (metric, ylabel) in zip(axes, focus, strict=True):
        for comparison in comparisons:
            subset = sorted(
                [
                    row
                    for row in stats_rows
                    if row["comparison"] == comparison and row["metric"] == metric
                ],
                key=lambda r: fnum(r["fidelity"]),
            )
            xs = [fnum(row["fidelity"]) for row in subset]
            means = [fnum(row["mean"]) for row in subset]
            lows = [fnum(row["ci95_low"]) for row in subset]
            highs = [fnum(row["ci95_high"]) for row in subset]
            yerr = [
                [mean - low for mean, low in zip(means, lows, strict=True)],
                [high - mean for mean, high in zip(means, highs, strict=True)],
            ]
            if not subset:
                continue
            label = (
                comparison.replace("FiLMTraj", "FiLM + L_traj")
                .replace("TokenTime", "Token-Time")
                .replace("StandardTraj", "Standard + L_traj")
            )
            color = {
                "FiLMTraj-FiLM": "#B279A2",
                "FiLMTraj-TokenTime": "#79706E",
                "StandardTraj-Standard": "#BAB0AC",
                "FiLMTraj-StandardTraj": "#8CD17D",
            }.get(comparison)
            ax.errorbar(
                xs,
                means,
                yerr=yerr,
                label=label,
                color=color,
                marker="o",
                linewidth=2.0,
                capsize=3,
            )
        ax.axhline(0.0, color="#333333", linewidth=0.9)
        ax.set_xlabel("Marker fidelity")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].legend(frameon=False, loc="lower right")
    save_figure(fig, out_dir, "paired_deltas_filmtraj")


def plot_tradeoff(summary_rows: list[dict[str, str]], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    for model in available_models(summary_rows):
        model_rows = rows_by_model(summary_rows, model)
        xs = [fnum(row["mlm_accuracy_mean"]) for row in model_rows]
        ys = [
            (fnum(row["spearman_drift_mean"]) + fnum(row["spearman_bifurcating_mean"])) / 2.0
            for row in model_rows
        ]
        ax.plot(
            xs,
            ys,
            label=MODEL_LABELS.get(model, model),
            color=COLORS.get(model),
            marker=MARKERS.get(model, "o"),
            linewidth=1.7,
            markersize=5.5,
            alpha=0.95,
        )
        for row, x, y in zip(model_rows, xs, ys, strict=True):
            fidelity = fnum(row["fidelity"])
            if math.isclose(fidelity, 0.5) or math.isclose(fidelity, 0.75):
                ax.annotate(f"{fidelity:.2f}", (x, y), xytext=(4, 3), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Masked-token accuracy")
    ax.set_ylabel("Mean trajectory Spearman")
    ax.grid(color="#D9D9D9", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    save_figure(fig, out_dir, "mlm_vs_trajectory_tradeoff")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Timeformer pilot results.")
    parser.add_argument("run_dir", type=Path, help="Directory containing pilot_summary.csv and pilot_paired_stats.csv.")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    run_dir = args.run_dir
    out_dir = args.output_dir or run_dir / "figures"
    summary_rows = load_csv(run_dir / "pilot_summary.csv")
    stats_rows = load_csv(run_dir / "pilot_paired_stats.csv")

    plot_metric_curves(summary_rows, out_dir)
    plot_paired_deltas(stats_rows, out_dir)
    plot_tradeoff(summary_rows, out_dir)

    print(f"Wrote figures to {out_dir}")


if __name__ == "__main__":
    main()
