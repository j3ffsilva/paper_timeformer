#!/usr/bin/env python3
"""Build reproducible figures, tables, and context examples for the paper."""

from __future__ import annotations

import argparse
import csv
import json
import os
import textwrap
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-paper-timeformer")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


EXAMPLE_TARGETS = ("plane_nn", "multitude_nn", "gas_nn", "record_nn")
ILLUSTRATIVE_SENSES = {
    ("plane_nn", "1810-1860"): "plane%1:25:00::",
    ("plane_nn", "1960-2010"): "plane%1:06:01::",
    ("multitude_nn", "1810-1860"): "multitude%1:14:00::",
    ("multitude_nn", "1960-2010"): "multitude%1:23:00::",
    ("gas_nn", "1810-1860"): "gas%1:27:00::",
    ("gas_nn", "1960-2010"): "gas%1:27:02::",
    ("record_nn", "1810-1860"): "record%1:10:03::",
    ("record_nn", "1960-2010"): "record%1:06:00::",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in text)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_latex_table(
    path: Path,
    rows: list[dict],
    columns: list[tuple[str, str]],
    alignment: str,
) -> None:
    header = " & ".join(latex_escape(label) for _, label in columns)
    lines = [
        rf"\begin{{tabular}}{{{alignment}}}",
        r"\toprule",
        header + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            " & ".join(latex_escape(row[key]) for key, _ in columns) + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def save_figure(figure, output_dir: Path, stem: str) -> None:
    figure.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    figure.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(figure)


def forest_plot(rows: list[dict], output_dir: Path) -> None:
    ordered = sorted(
        rows,
        key=lambda row: float(row["layer_2_bootstrap_median"]),
    )
    labels = [row["target"].replace("_nn", "").replace("_vb", "") for row in ordered]
    median = np.asarray([float(row["layer_2_bootstrap_median"]) for row in ordered])
    low = np.asarray([float(row["layer_2_ci_95_low"]) for row in ordered])
    high = np.asarray([float(row["layer_2_ci_95_high"]) for row in ordered])
    robust = np.asarray(
        [row["layer_2_classification"] == "robust_positive" for row in ordered]
    )
    y = np.arange(len(ordered))
    figure, axis = plt.subplots(figsize=(7.2, 9.0))
    axis.axvline(0, color="#555555", linewidth=1, linestyle="--")
    for selected, color, label in [
        (~robust, "#9b9b9b", "uncertain"),
        (robust, "#1f6f8b", "robust positive"),
    ]:
        axis.errorbar(
            median[selected],
            y[selected],
            xerr=np.vstack(
                [
                    median[selected] - low[selected],
                    high[selected] - median[selected],
                ]
            ),
            fmt="o",
            color=color,
            ecolor=color,
            capsize=2,
            markersize=4,
            linewidth=1,
            label=label,
        )
    axis.set_yticks(y, labels)
    axis.set_xlabel("Excess composition share (layer 2)")
    axis.set_title("Stratified bootstrap by target word")
    axis.legend(loc="lower right", frameon=False)
    axis.grid(axis="x", alpha=0.2)
    save_figure(figure, output_dir, "figure1_bootstrap_forest")


def scatter_plot(
    bootstrap_rows: list[dict],
    decomposition_rows: list[dict],
    output_dir: Path,
) -> None:
    bootstrap = {row["target"]: row for row in bootstrap_rows}
    points = []
    for row in decomposition_rows:
        target = row["target"]
        points.append(
            (
                float(row["layer_2_sense_jsd"]),
                float(bootstrap[target]["layer_2_bootstrap_median"]),
                target,
                bootstrap[target]["layer_2_classification"]
                == "robust_positive",
            )
        )
    figure, axis = plt.subplots(figsize=(7.2, 5.2))
    for robust, color, label in [
        (False, "#9b9b9b", "uncertain"),
        (True, "#1f6f8b", "robust positive"),
    ]:
        selected = [point for point in points if point[3] == robust]
        axis.scatter(
            [point[0] for point in selected],
            [point[1] for point in selected],
            color=color,
            s=34,
            label=label,
            alpha=0.9,
        )
    for x, y, target, _ in points:
        if target in EXAMPLE_TARGETS:
            axis.annotate(
                target[:-3],
                (x, y),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
            )
    axis.axhline(0, color="#555555", linewidth=1, linestyle="--")
    axis.set_xlabel("Mean ConSeC sense-distribution JSD")
    axis.set_ylabel("Bootstrap median excess composition share")
    axis.set_title("Sense-mixture change and geometric composition")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    save_figure(figure, output_dir, "figure2_jsd_vs_composition")


def sense_shift_plot(
    sense_rows: list[dict],
    output_dir: Path,
) -> None:
    by_target = {
        target: [row for row in sense_rows if row["target"] == target]
        for target in EXAMPLE_TARGETS
    }
    figure, axes = plt.subplots(2, 2, figsize=(11.0, 7.2))
    for axis, target in zip(axes.flat, EXAMPLE_TARGETS):
        rows = sorted(
            by_target[target],
            key=lambda row: max(float(row["mean_p0"]), float(row["mean_p1"])),
            reverse=True,
        )[:4]
        labels = [
            "\n".join(textwrap.wrap(row["definition"], width=32)[:2])
            for row in rows
        ]
        y = np.arange(len(rows))
        height = 0.34
        axis.barh(
            y - height / 2,
            [float(row["mean_p0"]) for row in rows],
            height=height,
            label="1810-1860",
            color="#b8c4ce",
        )
        axis.barh(
            y + height / 2,
            [float(row["mean_p1"]) for row in rows],
            height=height,
            label="1960-2010",
            color="#1f6f8b",
        )
        axis.set_yticks(y, labels, fontsize=8)
        axis.invert_yaxis()
        axis.set_xlim(0, 1)
        axis.set_title(target[:-3])
        axis.grid(axis="x", alpha=0.2)
    axes[0, 0].legend(frameon=False, fontsize=9)
    figure.suptitle("Posterior sense mixtures in selected examples")
    figure.tight_layout()
    save_figure(figure, output_dir, "figure3_selected_sense_shifts")


def representative_contexts(
    prediction_files: list[Path],
    inventory_rows: list[dict],
) -> list[dict]:
    definitions = {
        (row["target"], row["sensekey"]): row["definition"]
        for row in inventory_rows
    }
    unique = {}
    for path in prediction_files:
        for row in read_csv(path):
            if row["target"] in EXAMPLE_TARGETS:
                unique[row["sample_id"]] = row
    selected = []
    for target in EXAMPLE_TARGETS:
        target_rows = [row for row in unique.values() if row["target"] == target]
        sensekeys = sorted(
            {
                sensekey
                for row in target_rows
                for sensekey in json.loads(row["sense_probabilities"])
            }
        )
        means = {}
        for period in ("1810-1860", "1960-2010"):
            period_rows = [row for row in target_rows if row["period"] == period]
            means[period] = {
                sensekey: float(
                    np.mean(
                        [
                            json.loads(row["sense_probabilities"])[sensekey]
                            for row in period_rows
                        ]
                    )
                )
                for sensekey in sensekeys
            }
        for period in ("1810-1860", "1960-2010"):
            period_rows = [row for row in target_rows if row["period"] == period]
            selected_sense = ILLUSTRATIVE_SENSES[(target, period)]
            best = max(
                period_rows,
                key=lambda row: json.loads(row["sense_probabilities"])[
                    selected_sense
                ],
            )
            posterior = json.loads(best["sense_probabilities"])[selected_sense]
            selected.append(
                {
                    "target": target,
                    "period": period,
                    "sensekey": selected_sense,
                    "definition": definitions[(target, selected_sense)],
                    "posterior": f"{posterior:.3f}",
                    "context": best["context"],
                    "sample_id": best["sample_id"],
                }
            )
    return selected


def build_tables(
    bootstrap_rows: list[dict],
    decomposition_rows: list[dict],
    summaries: dict[str, dict],
    output_dir: Path,
) -> None:
    gate3 = summaries["gate3"]
    occurrence = summaries["occurrence"]
    decomposition_summary = summaries["decomposition"]
    bootstrap_summary = summaries["bootstrap"]
    layer_2_alignment = occurrence["secondary_layer_means"]["layer_2"][
        "semantic_geometry_partial_period"
    ]
    core_rows = [
        {
            "stage": "Gate 3 replication",
            "estimate": f"{gate3['mean_raw_spearman']:.3f}",
            "uncertainty": (
                "joint permutation "
                f"p={gate3['joint_permutation_p_two_sided']:.4f}"
            ),
        },
        {
            "stage": "Occurrence alignment, layer 1",
            "estimate": f"{occurrence['primary']['mean_target_rho']:.3f}",
            "uncertainty": (
                f"[{occurrence['primary']['bootstrap_mean_ci_95'][0]:.3f}, "
                f"{occurrence['primary']['bootstrap_mean_ci_95'][1]:.3f}]"
            ),
        },
        {
            "stage": "Occurrence alignment, layer 2",
            "estimate": f"{layer_2_alignment['mean_across_targets']:.3f}",
            "uncertainty": (
                f"[{layer_2_alignment['bootstrap_mean_ci_95'][0]:.3f}, "
                f"{layer_2_alignment['bootstrap_mean_ci_95'][1]:.3f}]"
            ),
        },
        {
            "stage": "Excess composition share, layer 2",
            "estimate": (
                f"{decomposition_summary['primary']['mean_excess_composition_share']:.3f}"
            ),
            "uncertainty": (
                f"[{decomposition_summary['primary']['bootstrap_mean_ci_95'][0]:.3f}, "
                f"{decomposition_summary['primary']['bootstrap_mean_ci_95'][1]:.3f}]"
            ),
        },
        {
            "stage": "Stratified bootstrap aggregate",
            "estimate": (
                f"{bootstrap_summary['primary']['bootstrap_aggregate_median']:.3f}"
            ),
            "uncertainty": (
                f"[{bootstrap_summary['primary']['bootstrap_aggregate_ci_95'][0]:.3f}, "
                f"{bootstrap_summary['primary']['bootstrap_aggregate_ci_95'][1]:.3f}]"
            ),
        },
    ]
    write_csv(output_dir / "table1_core_results.csv", core_rows)
    write_latex_table(
        output_dir / "table1_core_results.tex",
        core_rows,
        [
            ("stage", "Analysis"),
            ("estimate", "Estimate"),
            ("uncertainty", "Uncertainty"),
        ],
        "lrr",
    )

    robust_rows = [
        {
            "target": row["target"],
            "observed": f"{float(row['layer_2_observed_excess']):.3f}",
            "median": f"{float(row['layer_2_bootstrap_median']):.3f}",
            "ci": (
                f"[{float(row['layer_2_ci_95_low']):.3f}, "
                f"{float(row['layer_2_ci_95_high']):.3f}]"
            ),
        }
        for row in sorted(
            bootstrap_rows,
            key=lambda row: float(row["layer_2_bootstrap_median"]),
            reverse=True,
        )
        if row["layer_2_classification"] == "robust_positive"
    ]
    write_csv(output_dir / "table2_robust_targets.csv", robust_rows)
    write_latex_table(
        output_dir / "table2_robust_targets.tex",
        robust_rows,
        [
            ("target", "Target"),
            ("observed", "Observed"),
            ("median", "Bootstrap median"),
            ("ci", "95% CI"),
        ],
        "lrrr",
    )

    decomposition = {row["target"]: row for row in decomposition_rows}
    example_rows = []
    for row in bootstrap_rows:
        if row["target"] not in EXAMPLE_TARGETS:
            continue
        target = row["target"]
        example_rows.append(
            {
                "target": target,
                "jsd": f"{float(decomposition[target]['layer_2_sense_jsd']):.3f}",
                "composition": f"{float(row['layer_2_bootstrap_median']):.3f}",
                "ci": (
                    f"[{float(row['layer_2_ci_95_low']):.3f}, "
                    f"{float(row['layer_2_ci_95_high']):.3f}]"
                ),
            }
        )
    write_csv(output_dir / "table3_selected_examples.csv", example_rows)
    write_latex_table(
        output_dir / "table3_selected_examples.tex",
        example_rows,
        [
            ("target", "Target"),
            ("jsd", "Sense JSD"),
            ("composition", "Composition"),
            ("ci", "95% CI"),
        ],
        "lrrr",
    )


def evaluate(args: argparse.Namespace) -> None:
    bootstrap_rows = read_csv(args.bootstrap_targets)
    decomposition_rows = read_csv(args.decomposition_targets)
    sense_rows = read_csv(args.sense_contributions)
    inventory_rows = read_csv(args.sense_inventory)
    summaries = {
        "gate3": json.loads(args.gate3_summary.read_text(encoding="utf-8")),
        "occurrence": json.loads(
            args.occurrence_summary.read_text(encoding="utf-8")
        ),
        "decomposition": json.loads(
            args.decomposition_summary.read_text(encoding="utf-8")
        ),
        "bootstrap": json.loads(
            args.bootstrap_summary.read_text(encoding="utf-8")
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)

    forest_plot(bootstrap_rows, args.output_dir)
    scatter_plot(bootstrap_rows, decomposition_rows, args.output_dir)
    sense_shift_plot(sense_rows, args.output_dir)
    build_tables(bootstrap_rows, decomposition_rows, summaries, args.output_dir)

    contexts = representative_contexts(args.prediction_files, inventory_rows)
    write_csv(args.output_dir / "context_audit_examples.csv", contexts)
    manifest = {
        "figures": [
            "figure1_bootstrap_forest",
            "figure2_jsd_vs_composition",
            "figure3_selected_sense_shifts",
        ],
        "tables": [
            "table1_core_results",
            "table2_robust_targets",
            "table3_selected_examples",
        ],
        "context_examples": len(contexts),
        "selected_targets": list(EXAMPLE_TARGETS),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "README.md").write_text(
        "# ConSeC-TimeFormer paper assets\n\n"
        "Generated by `scripts/build_consec_timeformer_paper_assets.py`.\n\n"
        "## Figures\n\n"
        "- `figure1_bootstrap_forest`: target-level uncertainty for the "
        "layer-2 excess composition share.\n"
        "- `figure2_jsd_vs_composition`: ConSeC sense-distribution change "
        "against geometric composition.\n"
        "- `figure3_selected_sense_shifts`: posterior sense mixtures for "
        "four audited examples.\n\n"
        "## Tables and audit data\n\n"
        "- `table1_core_results`: confirmatory result chain.\n"
        "- `table2_robust_targets`: ten targets with positive individual "
        "95% intervals.\n"
        "- `table3_selected_examples`: quantitative values for the four "
        "qualitative examples.\n"
        "- `context_audit_examples.csv`: one representative context per "
        "period and example target.\n\n"
        "PNG files are for inspection; PDF files are vector assets for the "
        "manuscript. CSV files are source tables and `.tex` files are "
        "booktabs-compatible fragments.\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-targets", type=Path, required=True)
    parser.add_argument("--decomposition-targets", type=Path, required=True)
    parser.add_argument("--sense-contributions", type=Path, required=True)
    parser.add_argument("--sense-inventory", type=Path, required=True)
    parser.add_argument("--gate3-summary", type=Path, required=True)
    parser.add_argument("--occurrence-summary", type=Path, required=True)
    parser.add_argument("--decomposition-summary", type=Path, required=True)
    parser.add_argument("--bootstrap-summary", type=Path, required=True)
    parser.add_argument("--prediction-files", type=Path, nargs=3, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


if __name__ == "__main__":
    evaluate(build_parser().parse_args())
