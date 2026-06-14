#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path


LEXICAL_TOKEN = re.compile(r"^[a-z]+(?:'[a-z]+)?$")


def count_tokens(path: Path) -> Counter:
    counts = Counter()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            counts.update(line.split())
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select unique non-benchmark words matched to target corpus frequencies."
    )
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--period-files", nargs=2, default=["1810-1860.txt", "1960-2010.txt"])
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, default=None)
    parser.add_argument("--min-count-per-period", type=int, default=100)
    args = parser.parse_args()

    counts = [
        count_tokens(args.corpus_dir / filename)
        for filename in args.period_files
    ]
    targets = [
        line.strip()
        for line in args.targets.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    excluded = set(targets)
    excluded.update(target.rsplit("_", 1)[0] for target in targets)
    candidates = [
        token
        for token in counts[0].keys() & counts[1].keys()
        if (
            LEXICAL_TOKEN.fullmatch(token)
            and token not in excluded
            and counts[0][token] >= args.min_count_per_period
            and counts[1][token] >= args.min_count_per_period
        )
    ]
    selected = []
    used = set()
    for target in targets:
        target_frequencies = (
            math.log1p(counts[0][target]),
            math.log1p(counts[1][target]),
        )
        candidate = min(
            (token for token in candidates if token not in used),
            key=lambda token: math.dist(
                target_frequencies,
                (
                    math.log1p(counts[0][token]),
                    math.log1p(counts[1][token]),
                ),
            ),
        )
        used.add(candidate)
        selected.append({
            "target": target,
            "control": candidate,
            "target_d0": counts[0][target],
            "target_d1": counts[1][target],
            "control_d0": counts[0][candidate],
            "control_d1": counts[1][candidate],
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(row["control"] for row in selected) + "\n",
        encoding="utf-8",
    )
    metadata_output = args.metadata_output or args.output.with_suffix(".json")
    metadata_output.write_text(json.dumps(selected, indent=2), encoding="utf-8")
    print(f"Wrote {len(selected)} controls to {args.output}")


if __name__ == "__main__":
    main()
