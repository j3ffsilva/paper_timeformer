#!/usr/bin/env python3
"""Find words whose `token@time` displacement points in a similar direction.

Implements Fase B of `docs/39-token_time_analysis_framework.md`:

1. compara deslocamentos (`compare_profiles` -> `TokenTimeDisplacement`,
   já em `report_token_time_neighborhoods.py`);
2. busca por deslocamentos de direção semelhante
   (`nearest_displacements`, `src/timeformers/token_time_index.py`);
3. explica cada resultado pelas referências que mais contribuíram
   (`displacement_contributions`, `src/timeformers/token_time_metrics.py`).

Com apenas D0/D1, "direção do deslocamento" é o único modo de comparação
disponível (docs/39): não há forma/assinatura de trajetória sem 3+ períodos.

Generic over corpus/targets (same `--profile-dir` layout as
`report_token_time_neighborhoods.py`, written by `build_token_time_profiles.py`).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from timeformers.token_time import TokenTimeDisplacement  # noqa: E402
from timeformers.token_time_index import nearest_displacements  # noqa: E402
from timeformers.token_time_metrics import displacement_contributions  # noqa: E402
from timeformers.token_time_repository import TokenTimeIndex  # noqa: E402


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def format_contributions(rows: list[tuple[str, float]]) -> list[str]:
    lines = ["| reference | contribution |", "|---|---:|"]
    for token, value in rows:
        lines.append(f"| `{token}` | {value:.4f} |")
    return lines


def write_report(
    path: Path,
    *,
    neighbors_by_target: dict[str, list[tuple[str, float]]],
    reference_displacements: dict[str, TokenTimeDisplacement],
    explain_k: int,
) -> None:
    lines = [
        "# token@time: deslocamentos de direção semelhante (Fase B)",
        "",
        "`similarity(w, u) = cos(Delta(w), Delta(u))`, sobre o mesmo V_ativo",
        "usado em `report_token_time_neighborhoods.py`. Com apenas D0/D1, esta",
        "é a única noção de \"similaridade de trajetória\" disponível",
        "(similaridade de direção; forma/assinatura exigem 3+ períodos).",
        "",
    ]
    for target, neighbors in neighbors_by_target.items():
        lines.extend([f"## `{target}`", "", "| vizinho | similaridade de direção |", "|---|---:|"])
        for neighbor, similarity in neighbors:
            lines.append(f"| `{neighbor}` | {similarity:.3f} |")
        lines.append("")

        for neighbor, similarity in neighbors:
            contributions = displacement_contributions(
                reference_displacements[target], reference_displacements[neighbor]
            )
            lines.extend(
                [
                    f"### Por que `{target}` e `{neighbor}` se aproximam (similaridade {similarity:.3f})",
                    "",
                    "Referências cuja mudança em ambas as palavras mais contribui para a",
                    "similaridade (valores positivos altos) ou mais a reduz (valores",
                    "negativos):",
                    "",
                ]
            )
            lines.extend(format_contributions(contributions[:explain_k]))
            lines.append("")
            lines.extend(format_contributions(contributions[-explain_k:]))
            lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--cache-d0", type=Path, default=None, help="default: <profile-dir>/cache/theta_d0.pt")
    parser.add_argument("--cache-d1", type=Path, default=None, help="default: <profile-dir>/cache/theta_d1.pt")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layer", default="layer_2")
    parser.add_argument("--n-min-active", type=int, default=10, help="min count in both periods for V_ativo")
    parser.add_argument("--max-references", type=int, default=3216)
    parser.add_argument("--top-k", type=int, default=5, help="nearest displacements per target")
    parser.add_argument("--explain-k", type=int, default=10, help="top/bottom contributing references shown")
    parser.add_argument("--seed", type=int, default=None, help="training seed, recorded in TokenTimeProfile metadata")
    args = parser.parse_args()

    cache_paths = None
    if args.cache_d0 or args.cache_d1:
        cache_paths = [
            args.cache_d0 or (args.profile_dir / "cache" / "theta_d0.pt"),
            args.cache_d1 or (args.profile_dir / "cache" / "theta_d1.pt"),
        ]
    idx = TokenTimeIndex.load(args.profile_dir, cache_paths=cache_paths, seed=args.seed)

    active_mask = idx.active_support(args.n_min_active)
    active_ids = torch.nonzero(active_mask, as_tuple=False).flatten()
    reference_ids_t = idx.reference_set(args.max_references)

    active_displacements: dict[str, TokenTimeDisplacement] = {}
    reference_displacements: dict[str, TokenTimeDisplacement] = {}
    for target in idx.targets:
        active_displacements[target] = idx.displacement(target, active_ids, layer=args.layer, n_min_active=args.n_min_active)
        reference_displacements[target] = idx.displacement(
            target, reference_ids_t, layer=args.layer, n_min_active=args.n_min_active
        )

    rows = []
    neighbors_by_target: dict[str, list[tuple[str, float]]] = {}
    for target in idx.targets:
        neighbors = nearest_displacements(target, active_displacements, k=args.top_k)
        neighbors_by_target[target] = neighbors
        for rank, (neighbor, similarity) in enumerate(neighbors, start=1):
            rows.append({"target": target, "neighbor": neighbor, "rank": rank, "similarity": similarity})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.output_dir / "similar_displacements.csv")
    write_report(
        args.output_dir / "report.md",
        neighbors_by_target=neighbors_by_target,
        reference_displacements=reference_displacements,
        explain_k=args.explain_k,
    )
    summary = {"n_targets": len(idx.targets), "top_k": args.top_k, "n_active_support": int(active_mask.sum())}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
