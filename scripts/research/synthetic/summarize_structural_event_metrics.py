#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


EVENT_METRICS = (
    "event_period",
    "observed_peak_period",
    "event_period_error",
    "event_step_magnitude",
    "event_concentration",
    "pre_event_drift",
    "pre_event_drift_ratio",
    "post_event_drift",
    "post_event_drift_ratio",
    "event_fidelity",
    "final_magnitude",
    "path_length",
    "shape_error",
)


CONTROLS = ("resampled_null", "continual_placebo", "independent_period", "cumulative_retrain")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_group(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("groups must use the form name=path")
    name, path = value.split("=", 1)
    if not name:
        raise argparse.ArgumentTypeError("group name cannot be empty")
    return name, Path(path)


def load_group_rows(group_name: str, root: Path) -> list[dict]:
    rows = []
    for metrics_path in sorted(root.glob("seed_*/structural_metrics.csv")):
        seed = metrics_path.parent.name.removeprefix("seed_")
        for row in read_csv(metrics_path):
            enriched = {
                "group": group_name,
                "seed": seed,
                **row,
            }
            rows.append(enriched)
    if not rows:
        raise FileNotFoundError(f"No seed_*/structural_metrics.csv files found under {root}")
    return rows


def finite_float(row: dict, metric: str) -> float | None:
    value = row.get(metric)
    if value in (None, ""):
        return None
    result = float(value)
    return result if np.isfinite(result) else None


def summarize_rows(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["group"], row["regime"], row["condition"])].append(row)

    summary = []
    for (group, regime, condition), items in sorted(grouped.items()):
        record = {
            "group": group,
            "regime": regime,
            "condition": condition,
            "n": len(items),
        }
        for metric in EVENT_METRICS:
            values = [value for row in items if (value := finite_float(row, metric)) is not None]
            if not values:
                continue
            array = np.array(values, dtype=float)
            record[f"{metric}_mean"] = float(array.mean())
            record[f"{metric}_median"] = float(np.median(array))
            record[f"{metric}_sd"] = float(array.std(ddof=1)) if len(array) > 1 else 0.0
        summary.append(record)
    return summary


def paired_control_deltas(rows: list[dict]) -> list[dict]:
    by_key = defaultdict(dict)
    for row in rows:
        key = (row["group"], row["seed"], row["condition"], row["subject"])
        by_key[key][row["regime"]] = row

    deltas = []
    for (group, seed, condition, subject), regimes in sorted(by_key.items()):
        real = regimes.get("continual_real")
        if real is None:
            continue
        for control in CONTROLS:
            control_row = regimes.get(control)
            if control_row is None:
                continue
            out = {
                "group": group,
                "seed": seed,
                "condition": condition,
                "subject": subject,
                "control": control,
            }
            for metric in EVENT_METRICS:
                real_value = finite_float(real, metric)
                control_value = finite_float(control_row, metric)
                if real_value is None or control_value is None:
                    continue
                out[f"{metric}_real"] = real_value
                out[f"{metric}_control"] = control_value
                out[f"{metric}_delta"] = real_value - control_value
            deltas.append(out)
    return deltas


def summarize_control_deltas(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["group"], row["control"], row["condition"])].append(row)

    summary = []
    for (group, control, condition), items in sorted(grouped.items()):
        record = {
            "group": group,
            "control": control,
            "condition": condition,
            "n": len(items),
        }
        for metric in EVENT_METRICS:
            key = f"{metric}_delta"
            values = [float(row[key]) for row in items if key in row]
            if not values:
                continue
            array = np.array(values, dtype=float)
            record[f"{key}_mean"] = float(array.mean())
            record[f"{key}_median"] = float(np.median(array))
            record[f"{key}_sd"] = float(array.std(ddof=1)) if len(array) > 1 else 0.0
        summary.append(record)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize structural local-event metrics.")
    parser.add_argument(
        "--group",
        action="append",
        type=parse_group,
        required=True,
        help="Experiment group in the form name=path.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for group_name, root in args.group:
        rows.extend(load_group_rows(group_name, root))

    absolute_summary = summarize_rows(rows)
    deltas = paired_control_deltas(rows)
    delta_summary = summarize_control_deltas(deltas)

    write_csv(absolute_summary, args.output_dir / "structural_event_metric_summary.csv")
    write_csv(deltas, args.output_dir / "structural_event_metric_control_deltas.csv")
    write_csv(delta_summary, args.output_dir / "structural_event_metric_control_delta_summary.csv")

    print(f"Wrote {args.output_dir / 'structural_event_metric_summary.csv'}")
    print(f"Wrote {args.output_dir / 'structural_event_metric_control_deltas.csv'}")
    print(f"Wrote {args.output_dir / 'structural_event_metric_control_delta_summary.csv'}")


if __name__ == "__main__":
    main()
