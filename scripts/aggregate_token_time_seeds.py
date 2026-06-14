#!/usr/bin/env python3
"""Aggregate `token@time` reports across seeds and report stability.

Implements Fase A items 3 and 4 of `docs/39-token_time_analysis_framework.md`:
"agregar as duas seeds" and "reportar estabilidade".

Input: two or more output directories produced by
`report_token_time_neighborhoods.py` (one per seed), each containing
`rankings.csv` and `neighborhoods.csv` for the same targets/references.

Output: combined rankings (mean displacement/turnover, range across seeds,
pairwise Spearman agreement) and combined gains/losses (mean `delta_z`,
range across seeds).
"""

from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

from scipy.stats import spearmanr


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rankings(rankings_per_seed: list[list[dict]]) -> tuple[list[dict], dict]:
    targets = [row["target"] for row in rankings_per_seed[0]]
    for rankings in rankings_per_seed[1:]:
        if [row["target"] for row in rankings] != targets:
            raise ValueError("seed rankings.csv files must list the same targets in the same order")

    by_seed_displacement = [
        [float(row["displacement"]) for row in rankings] for rankings in rankings_per_seed
    ]
    by_seed_turnover = [
        [float(row["turnover"]) for row in rankings] for rankings in rankings_per_seed
    ]

    combined = []
    for index, target in enumerate(targets):
        displacements = [seed[index] for seed in by_seed_displacement]
        turnovers = [seed[index] for seed in by_seed_turnover]
        combined.append(
            {
                "target": target,
                "displacement_mean": sum(displacements) / len(displacements),
                "displacement_range": max(displacements) - min(displacements),
                "turnover_mean": sum(turnovers) / len(turnovers),
                "turnover_range": max(turnovers) - min(turnovers),
                "count_d0": rankings_per_seed[0][index]["count_d0"],
                "count_d1": rankings_per_seed[0][index]["count_d1"],
            }
        )
    combined.sort(key=lambda row: row["displacement_mean"], reverse=True)

    pairwise_spearman = []
    for seed_a, seed_b in combinations(range(len(rankings_per_seed)), 2):
        rho = float(spearmanr(by_seed_displacement[seed_a], by_seed_displacement[seed_b]).statistic)
        pairwise_spearman.append({"seed_a": seed_a, "seed_b": seed_b, "spearman_displacement": rho})

    stability = {
        "n_seeds": len(rankings_per_seed),
        "n_targets": len(targets),
        "pairwise_spearman_displacement": pairwise_spearman,
    }
    return combined, stability


def aggregate_neighborhoods(neighborhoods_per_seed: list[list[dict]]) -> list[dict]:
    by_key: dict[tuple[str, str], dict] = {}
    for rows in neighborhoods_per_seed:
        for row in rows:
            key = (row["target"], row["reference"])
            entry = by_key.setdefault(
                key,
                {"target": row["target"], "reference": row["reference"], "delta_z": [], "rank_d0": [], "rank_d1": []},
            )
            entry["delta_z"].append(float(row["delta_z"]))
            entry["rank_d0"].append(int(row["rank_d0"]))
            entry["rank_d1"].append(int(row["rank_d1"]))

    combined = []
    for (target, reference), entry in by_key.items():
        delta_z = entry["delta_z"]
        combined.append(
            {
                "target": target,
                "reference": reference,
                "delta_z_mean": sum(delta_z) / len(delta_z),
                "delta_z_range": max(delta_z) - min(delta_z),
                "rank_d0_mean": sum(entry["rank_d0"]) / len(entry["rank_d0"]),
                "rank_d1_mean": sum(entry["rank_d1"]) / len(entry["rank_d1"]),
                "n_seeds": len(delta_z),
            }
        )
    return combined


def format_table(rows: list[dict]) -> list[str]:
    lines = ["| reference | mean delta_z | range across seeds |", "|---|---:|---:|"]
    for row in rows:
        lines.append(f"| `{row['reference']}` | {row['delta_z_mean']:.3f} | {row['delta_z_range']:.3f} |")
    return lines


def write_report(path: Path, *, rankings: list[dict], stability: dict, neighborhoods_by_target: dict[str, list[dict]], top_k: int, salience_rank: int) -> None:
    lines = [
        "# token@time: agregação entre seeds e estabilidade",
        "",
        "Combina relatórios por seed de `report_token_time_neighborhoods.py`",
        "(mesmo encoder fixo, mesmo corpus, mesma lista de alvos; só a seed de",
        "treino contínuo muda).",
        "",
        "## Estabilidade do ranking de deslocamento entre seeds",
        "",
        "| seed A | seed B | Spearman (displacement) |",
        "|---|---|---:|",
    ]
    for entry in stability["pairwise_spearman_displacement"]:
        lines.append(f"| {entry['seed_a']} | {entry['seed_b']} | {entry['spearman_displacement']:.3f} |")
    lines.extend(
        [
            "",
            "Um Spearman baixo entre seeds indica que o ranking de deslocamento não",
            "é robusto à seed de treino -- nesse caso, 'menos mudou' não pode ser",
            "lido como evidência de estabilidade (docs/39, operação 6).",
            "",
            "## Ranking combinado (deslocamento final e turnover)",
            "",
            "| target | displacement (média) | range entre seeds | turnover (média) | range entre seeds | n_d0 | n_d1 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rankings:
        lines.append(
            f"| `{row['target']}` | {row['displacement_mean']:.3f} | {row['displacement_range']:.3f} | "
            f"{row['turnover_mean']:.3f} | {row['turnover_range']:.3f} | {row['count_d0']} | {row['count_d1']} |"
        )
    lines.append("")

    for target, rows in neighborhoods_by_target.items():
        salient = [row for row in rows if min(row["rank_d0_mean"], row["rank_d1_mean"]) <= salience_rank]
        gains = sorted(salient, key=lambda row: row["delta_z_mean"], reverse=True)[:top_k]
        losses = sorted(salient, key=lambda row: row["delta_z_mean"])[:top_k]
        lines.extend([f"## `{target}`", "", "### Ganhos consistentes entre seeds (maior `delta_z` médio)", ""])
        lines.extend(format_table(gains))
        lines.extend(["", "### Perdas consistentes entre seeds (menor `delta_z` médio)", ""])
        lines.extend(format_table(losses))
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--salience-rank", type=int, default=50)
    args = parser.parse_args()

    if len(args.seed_dirs) < 2:
        raise ValueError("--seed-dirs requires at least two directories")

    rankings_per_seed = [
        sorted(read_csv(seed_dir / "rankings.csv"), key=lambda row: row["target"])
        for seed_dir in args.seed_dirs
    ]
    neighborhoods_per_seed = [read_csv(seed_dir / "neighborhoods.csv") for seed_dir in args.seed_dirs]

    combined_rankings, stability = aggregate_rankings(rankings_per_seed)
    combined_neighborhoods = aggregate_neighborhoods(neighborhoods_per_seed)

    neighborhoods_by_target: dict[str, list[dict]] = {}
    for row in combined_neighborhoods:
        neighborhoods_by_target.setdefault(row["target"], []).append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(combined_rankings, args.output_dir / "rankings.csv")
    write_csv(combined_neighborhoods, args.output_dir / "neighborhoods.csv")
    (args.output_dir / "stability.json").write_text(json.dumps(stability, indent=2), encoding="utf-8")
    write_report(
        args.output_dir / "report.md",
        rankings=combined_rankings,
        stability=stability,
        neighborhoods_by_target=neighborhoods_by_target,
        top_k=args.top_k,
        salience_rank=args.salience_rank,
    )
    print(json.dumps(stability, indent=2))


if __name__ == "__main__":
    main()
