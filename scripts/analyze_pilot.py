#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

from scipy import stats


DEFAULT_METRICS = [
    "spearman_drift",
    "spearman_bifurcating",
    "mlm_accuracy",
    "path_contrast_drift_minus_stable",
    "directed_contrast_drift_minus_stable",
]

DEFAULT_COMPARISONS = [
    ("FiLMTraj", "FiLM"),
    ("FiLMTraj", "TokenTime"),
    ("TokenTimeTraj", "TokenTime"),
    ("FiLMTraj", "TokenTimeTraj"),
    ("StandardTraj", "Standard"),
    ("FiLMTraj", "StandardTraj"),
]


def parse_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_comparisons(value: str) -> list[tuple[str, str]]:
    comparisons = []
    for item in parse_list(value):
        if "-" not in item:
            raise ValueError(f"Comparison must look like LEFT-RIGHT, got {item!r}")
        left, right = item.split("-", 1)
        comparisons.append((left.strip(), right.strip()))
    return comparisons


def mean_ci(values: list[float]) -> dict[str, float]:
    clean = [v for v in values if not math.isnan(v)]
    if not clean:
        return {"n": 0, "mean": math.nan, "sd": math.nan, "ci95_low": math.nan, "ci95_high": math.nan}
    mean = statistics.mean(clean)
    sd = statistics.stdev(clean) if len(clean) > 1 else 0.0
    se = sd / math.sqrt(len(clean)) if clean else math.nan
    if len(clean) > 1:
        tcrit = stats.t.ppf(0.975, df=len(clean) - 1)
    else:
        tcrit = math.nan
    return {
        "n": len(clean),
        "mean": mean,
        "sd": sd,
        "ci95_low": mean - tcrit * se if len(clean) > 1 else math.nan,
        "ci95_high": mean + tcrit * se if len(clean) > 1 else math.nan,
    }


def paired_tests(diffs: list[float]) -> dict[str, float]:
    clean = [v for v in diffs if not math.isnan(v)]
    if len(clean) < 2:
        return {
            "t_stat": math.nan,
            "t_p_two_sided": math.nan,
            "wilcoxon_stat": math.nan,
            "wilcoxon_p_two_sided": math.nan,
        }
    t_stat, t_p = stats.ttest_1samp(clean, popmean=0.0)
    try:
        w_stat, w_p = stats.wilcoxon(clean, zero_method="wilcox", alternative="two-sided")
    except ValueError:
        w_stat, w_p = math.nan, math.nan
    return {
        "t_stat": float(t_stat),
        "t_p_two_sided": float(t_p),
        "wilcoxon_stat": float(w_stat),
        "wilcoxon_p_two_sided": float(w_p),
    }


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def analyze(rows: list[dict[str, str]], metrics: list[str], comparisons: list[tuple[str, str]]) -> list[dict]:
    fidelities = sorted({row["fidelity"] for row in rows}, key=float)
    output = []
    for fidelity in fidelities:
        subset = [row for row in rows if row["fidelity"] == fidelity]
        seeds = sorted({row["seed"] for row in subset}, key=int)
        by_key = {(row["seed"], row["model"]): row for row in subset}
        for left, right in comparisons:
            for metric in metrics:
                diffs = []
                used_seeds = []
                for seed in seeds:
                    a = by_key.get((seed, left))
                    b = by_key.get((seed, right))
                    if not a or not b:
                        continue
                    diffs.append(float(a[metric]) - float(b[metric]))
                    used_seeds.append(seed)
                row = {
                    "fidelity": float(fidelity),
                    "comparison": f"{left}-{right}",
                    "left": left,
                    "right": right,
                    "metric": metric,
                    **mean_ci(diffs),
                    **paired_tests(diffs),
                    "seeds": ",".join(used_seeds),
                }
                output.append(row)
    return output


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "fidelity",
        "comparison",
        "left",
        "right",
        "metric",
        "n",
        "mean",
        "sd",
        "ci95_low",
        "ci95_high",
        "t_stat",
        "t_p_two_sided",
        "wilcoxon_stat",
        "wilcoxon_p_two_sided",
        "seeds",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict], path: Path) -> None:
    focus = [r for r in rows if r["comparison"] in {"FiLMTraj-FiLM", "FiLMTraj-TokenTime"}]
    lines = [
        "# Pilot Paired Statistics",
        "",
        "Delta = left model minus right model. Positive Spearman deltas favor the left model.",
        "",
    ]
    for fidelity in sorted({r["fidelity"] for r in focus}):
        lines.append(f"## Fidelity {fidelity:.3f}")
        for metric in ["spearman_drift", "spearman_bifurcating", "mlm_accuracy"]:
            lines.append(f"### {metric}")
            for row in [r for r in focus if r["fidelity"] == fidelity and r["metric"] == metric]:
                lines.append(
                    f"- {row['comparison']}: mean={row['mean']:+.4f}, "
                    f"95% CI [{row['ci95_low']:+.4f}, {row['ci95_high']:+.4f}], "
                    f"t-p={row['t_p_two_sided']:.4g}, W-p={row['wilcoxon_p_two_sided']:.4g}"
                )
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze paired deltas from a Timeformer pilot_results.csv file.")
    parser.add_argument("results_csv", type=Path)
    parser.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    parser.add_argument(
        "--comparisons",
        default=",".join(f"{left}-{right}" for left, right in DEFAULT_COMPARISONS),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    metrics = parse_list(args.metrics)
    comparisons = parse_comparisons(args.comparisons)
    rows = load_rows(args.results_csv)
    stats_rows = analyze(rows, metrics, comparisons)

    out_dir = args.output_dir or args.results_csv.parent
    out_csv = out_dir / "pilot_paired_stats.csv"
    out_json = out_dir / "pilot_paired_stats.json"
    out_md = out_dir / "pilot_paired_stats.md"
    write_csv(stats_rows, out_csv)
    out_json.write_text(json.dumps(stats_rows, indent=2), encoding="utf-8")
    write_markdown(stats_rows, out_md)

    print(f"Wrote {out_csv}")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
