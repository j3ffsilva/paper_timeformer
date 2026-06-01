#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


PIPELINE_METRICS = [
    "d6_silhouette",
    "teacher_cka_M_R",
    "teacher_probe_r2_M",
    "teacher_probe_r2_R",
    "R_m_cka",
    "m_d2_spearman_drift",
    "m_d2_spearman_bifurcating",
    "m_d2_spearman_abrupt",
    "token_time_d2_spearman_drift",
    "token_time_d2_spearman_bifurcating",
    "token_time_d2_spearman_abrupt",
]

D5A_METRICS = [
    "teacher_cka_M_R",
    "teacher_probe_r2_M",
    "teacher_probe_r2_R",
    "d5a_all_loss",
    "d5a_all_loss_stable",
    "d5a_all_loss_drift",
    "d5a_all_loss_bifurcating",
    "d5a_all_loss_abrupt",
]


def load_json_list(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(loaded, list):
            rows.extend(loaded)
        else:
            rows.append(loaded)
    return rows


def mean_sd(values: list[float]) -> tuple[float, float]:
    clean = [v for v in values if not math.isnan(v)]
    if not clean:
        return math.nan, math.nan
    mean = sum(clean) / len(clean)
    if len(clean) < 2:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in clean) / (len(clean) - 1)
    return mean, math.sqrt(var)


def summarize(rows: list[dict], group_key: str, metrics: list[str]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row[group_key])].append(row)

    summary = []
    for group, items in sorted(grouped.items()):
        record = {group_key: group, "n": len(items)}
        for metric in metrics:
            mean, sd = mean_sd([float(item.get(metric, math.nan)) for item in items])
            record[f"{metric}_mean"] = mean
            record[f"{metric}_sd"] = sd
        summary.append(record)
    return summary


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict], group_key: str, metrics: list[str], path: Path) -> None:
    lines = ["# Synthetic Summary", ""]
    header = [group_key, "n"] + metrics
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for row in rows:
        values = [str(row[group_key]), str(row["n"])]
        for metric in metrics:
            mean = row[f"{metric}_mean"]
            sd = row[f"{metric}_sd"]
            values.append(f"{mean:.4f} +/- {sd:.4f}")
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize synthetic pipeline or D5a JSON outputs.")
    parser.add_argument("--kind", choices=["pipeline", "d5a"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("json_files", nargs="+", type=Path)
    args = parser.parse_args()

    rows = load_json_list(args.json_files)
    if args.kind == "pipeline":
        group_key = "config"
        metrics = PIPELINE_METRICS
    else:
        group_key = "student"
        metrics = D5A_METRICS

    summary = summarize(rows, group_key, metrics)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(summary, args.output_dir / f"{args.kind}_summary.csv")
    (args.output_dir / f"{args.kind}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown(summary, group_key, metrics, args.output_dir / f"{args.kind}_summary.md")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
