#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import torch


METRICS = (
    "observed_mean_abs_similarity_delta",
    "placebo_mean_abs_similarity_delta",
    "observed_minus_placebo_magnitude",
    "observed_oracle_direction_cosine",
    "placebo_oracle_direction_cosine",
    "oracle_direction_advantage",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(rows: list[dict], path: Path) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def seed_level_rows(root: Path) -> list[dict]:
    output = []
    for seed_dir in sorted(root.glob("seed_*")):
        path = seed_dir / "placebo_reference_results.csv"
        if not path.exists():
            continue
        grouped = defaultdict(list)
        for row in read_csv(path):
            key = (row["mode"], row["comparison"], int(row["to_period"]), row["class_name"])
            grouped[key].append(row)
        for key, rows in grouped.items():
            record = dict(zip(("mode", "comparison", "to_period", "class_name"), key))
            record["seed"] = seed_dir.name.removeprefix("seed_")
            for metric in METRICS:
                record[metric] = sum(float(row[metric]) for row in rows) / len(rows)
            output.append(record)
    return output


def aggregate(seed_rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in seed_rows:
        key = (row["mode"], row["comparison"], row["to_period"], row["class_name"])
        grouped[key].append(row)

    output = []
    for key, rows in sorted(grouped.items()):
        record = dict(zip(("mode", "comparison", "to_period", "class_name"), key))
        record["n_seeds"] = len(rows)
        for metric in METRICS:
            values = torch.tensor([row[metric] for row in rows])
            record[f"{metric}_mean"] = float(values.mean())
            record[f"{metric}_sd"] = float(values.std(unbiased=True)) if len(values) > 1 else 0.0
            record[f"{metric}_positive_fraction"] = float((values > 0).float().mean())
        output.append(record)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate paired relational/placebo results across seeds.")
    parser.add_argument("root", type=Path)
    args = parser.parse_args()

    seed_rows = seed_level_rows(args.root)
    if not seed_rows:
        raise FileNotFoundError(f"No seed_*/placebo_reference_results.csv files found under {args.root}")
    summary = aggregate(seed_rows)
    write_csv(seed_rows, args.root / "multiseed_seed_level.csv")
    write_csv(summary, args.root / "multiseed_summary.csv")
    (args.root / "multiseed_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Aggregated {len({row['seed'] for row in seed_rows})} seeds into {args.root / 'multiseed_summary.csv'}")


if __name__ == "__main__":
    main()
