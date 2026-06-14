#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(rows: list[dict], path: Path) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize relational sensitivity across trajectory scales.")
    parser.add_argument("root", type=Path)
    args = parser.parse_args()

    rows = []
    raw_rows = []
    for scale_dir in sorted(args.root.glob("scale_*")):
        config = json.loads((scale_dir / "config.json").read_text(encoding="utf-8"))
        summary = read_csv(scale_dir / "placebo_reference_summary.csv")
        raw = read_csv(scale_dir / "placebo_reference_results.csv")
        for row in raw:
            if row["mode"] == "prediction_distribution_js" and row["comparison"] == "from_t0":
                raw_rows.append(
                    {
                        "trajectory_scale": float(config["trajectory_scale"]),
                        "seed": config["seed"],
                        **row,
                    }
                )
        for row in summary:
            if row["mode"] != "prediction_distribution_js" or row["comparison"] != "from_t0":
                continue
            rows.append(
                {
                    "trajectory_scale": config["trajectory_scale"],
                    "seed": config["seed"],
                    "to_period": row["to_period"],
                    "class_name": row["class_name"],
                    "observed_direction": row["observed_oracle_direction_cosine_mean"],
                    "placebo_direction": row["placebo_oracle_direction_cosine_mean"],
                    "direction_advantage": row["oracle_direction_advantage_mean"],
                    "observed_magnitude": row["observed_mean_abs_similarity_delta_mean"],
                    "placebo_magnitude": row["placebo_mean_abs_similarity_delta_mean"],
                    "magnitude_advantage": row["observed_minus_placebo_magnitude_mean"],
                }
            )
    if not rows:
        raise FileNotFoundError(f"No scale_*/placebo_reference_summary.csv files found under {args.root}")
    write_csv(rows, args.root / "sensitivity_summary.csv")
    final_period = max(int(row["to_period"]) for row in raw_rows)
    null = [
        float(row["observed_mean_abs_similarity_delta"])
        for row in raw_rows
        if row["trajectory_scale"] == 0.0 and int(row["to_period"]) == final_period
    ]
    thresholds = {quantile: float(np.quantile(null, quantile)) for quantile in (0.90, 0.95)}
    detection_rows = []
    for scale in sorted({row["trajectory_scale"] for row in raw_rows}):
        scale_rows = [
            row for row in raw_rows if row["trajectory_scale"] == scale and int(row["to_period"]) == final_period
        ]
        for quantile, threshold in thresholds.items():
            detected = [
                float(row["observed_mean_abs_similarity_delta"]) > threshold
                and (scale == 0.0 or float(row["observed_oracle_direction_cosine"]) > 0)
                for row in scale_rows
            ]
            detection_rows.append(
                {
                    "trajectory_scale": scale,
                    "to_period": final_period,
                    "null_quantile": quantile,
                    "null_magnitude_threshold": threshold,
                    "n": len(detected),
                    "detected_fraction": sum(detected) / len(detected),
                }
            )
    write_csv(detection_rows, args.root / "sensitivity_detection_summary.csv")
    print(f"Wrote {args.root / 'sensitivity_summary.csv'}")
    print(f"Wrote {args.root / 'sensitivity_detection_summary.csv'}")


if __name__ == "__main__":
    main()
