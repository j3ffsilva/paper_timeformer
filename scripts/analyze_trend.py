#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from scipy import stats


DEFAULT_METRICS = ["spearman_drift", "spearman_bifurcating", "mlm_accuracy"]
DEFAULT_COMPARISONS = ["FiLMTraj-FiLM", "FiLMTraj-TokenTime", "StandardTraj-Standard"]


def parse_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def analyze(rows: list[dict[str, str]], metrics: list[str], comparisons: list[str]) -> list[dict]:
    output = []
    for comparison in comparisons:
        for metric in metrics:
            subset = sorted(
                [row for row in rows if row["comparison"] == comparison and row["metric"] == metric],
                key=lambda row: float(row["fidelity"]),
            )
            if len(subset) < 3:
                continue
            fidelity = [float(row["fidelity"]) for row in subset]
            noise = [1.0 - x for x in fidelity]
            deltas = [float(row["mean"]) for row in subset]
            slope, intercept, r_value, p_value, stderr = stats.linregress(noise, deltas)
            spearman_r, spearman_p = stats.spearmanr(noise, deltas)
            kendall_tau, kendall_p = stats.kendalltau(noise, deltas)
            output.append(
                {
                    "comparison": comparison,
                    "metric": metric,
                    "n_levels": len(subset),
                    "slope_per_noise": slope,
                    "intercept": intercept,
                    "pearson_r": r_value,
                    "linear_p_two_sided": p_value,
                    "slope_stderr": stderr,
                    "spearman_r": spearman_r,
                    "spearman_p_two_sided": spearman_p,
                    "kendall_tau": kendall_tau,
                    "kendall_p_two_sided": kendall_p,
                    "fidelities": ",".join(f"{x:.7g}" for x in fidelity),
                    "mean_deltas": ",".join(f"{x:.7g}" for x in deltas),
                }
            )
    return output


def write_csv(rows: list[dict], path: Path) -> None:
    fields = [
        "comparison",
        "metric",
        "n_levels",
        "slope_per_noise",
        "intercept",
        "pearson_r",
        "linear_p_two_sided",
        "slope_stderr",
        "spearman_r",
        "spearman_p_two_sided",
        "kendall_tau",
        "kendall_p_two_sided",
        "fidelities",
        "mean_deltas",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict], path: Path) -> None:
    lines = [
        "# Fidelity Trend Analysis",
        "",
        "Trend is computed over mean paired deltas at each fidelity level.",
        "`slope_per_noise` uses noise = 1 - fidelity, so a positive slope means the gap grows as markers degrade.",
        "",
        "| Comparison | Metric | Slope/noise | Linear p | Spearman rho | Spearman p | Kendall tau | Kendall p |",
        "|------------|--------|-------------|----------|--------------|------------|-------------|-----------|",
    ]
    for row in rows:
        lines.append(
            "| {comparison} | {metric} | {slope_per_noise:+.4f} | {linear_p_two_sided:.4g} | "
            "{spearman_r:+.4f} | {spearman_p_two_sided:.4g} | "
            "{kendall_tau:+.4f} | {kendall_p_two_sided:.4g} |".format(**row)
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze trend of paired deltas across fidelity/noise levels.")
    parser.add_argument("paired_stats_csv", type=Path)
    parser.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    parser.add_argument("--comparisons", default=",".join(DEFAULT_COMPARISONS))
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    rows = load_rows(args.paired_stats_csv)
    trend_rows = analyze(rows, parse_list(args.metrics), parse_list(args.comparisons))
    out_dir = args.output_dir or args.paired_stats_csv.parent
    out_csv = out_dir / "pilot_trend_stats.csv"
    out_json = out_dir / "pilot_trend_stats.json"
    out_md = out_dir / "pilot_trend_stats.md"
    write_csv(trend_rows, out_csv)
    out_json.write_text(json.dumps(trend_rows, indent=2), encoding="utf-8")
    write_markdown(trend_rows, out_md)
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
