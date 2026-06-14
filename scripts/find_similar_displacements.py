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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from timeformers.relational import build_active_support, contextual_centroids, type_uniform_mean  # noqa: E402
from timeformers.token_time import TokenTimeDisplacement, build_profile, compare_profiles  # noqa: E402
from timeformers.token_time_index import nearest_displacements  # noqa: E402
from timeformers.token_time_metrics import displacement_contributions  # noqa: E402

try:
    from scripts.report_token_time_neighborhoods import build_reference_set
except ModuleNotFoundError:
    from report_token_time_neighborhoods import build_reference_set


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

    cache_d0 = args.cache_d0 or (args.profile_dir / "cache" / "theta_d0.pt")
    cache_d1 = args.cache_d1 or (args.profile_dir / "cache" / "theta_d1.pt")

    vocab = json.loads((args.profile_dir / "vocab.json").read_text(encoding="utf-8"))
    targets = json.loads((args.profile_dir / "targets.json").read_text(encoding="utf-8"))
    target_ids = json.loads((args.profile_dir / "target_ids.json").read_text(encoding="utf-8"))
    metadata_path = args.profile_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    checkpoint = metadata.get("checkpoint", "")
    period_files = metadata.get("period_files", ["d0", "d1"])

    stats_d0 = torch.load(cache_d0, map_location="cpu", weights_only=True)
    stats_d1 = torch.load(cache_d1, map_location="cpu", weights_only=True)

    active_mask = build_active_support(
        stats_d0, stats_d1, vocab=vocab, targets=set(targets), n_min=args.n_min_active
    )
    active_ids = torch.nonzero(active_mask, as_tuple=False).flatten()

    reference_ids = build_reference_set(
        vocab,
        active_mask,
        targets=set(targets),
        counts_d0=stats_d0["counts"],
        counts_d1=stats_d1["counts"],
        max_references=args.max_references,
    )
    reference_ids_t = torch.tensor(reference_ids, dtype=torch.long)

    centroids_d0 = contextual_centroids(stats_d0, args.layer)
    centroids_d1 = contextual_centroids(stats_d1, args.layer)
    mu_d0 = type_uniform_mean(stats_d0, args.layer, support=active_mask)
    mu_d1 = type_uniform_mean(stats_d1, args.layer, support=active_mask)

    active_displacements: dict[str, TokenTimeDisplacement] = {}
    reference_displacements: dict[str, TokenTimeDisplacement] = {}
    for target in targets:
        target_id = target_ids[target]
        count_d0 = int(stats_d0["counts"][target_id])
        count_d1 = int(stats_d1["counts"][target_id])

        for ids, displacements in ((active_ids, active_displacements), (reference_ids_t, reference_displacements)):
            profile_d0 = build_profile(
                centroids_d0, mu_d0, target_id, ids, vocab,
                word=target, period=period_files[0], checkpoint=checkpoint,
                layer=args.layer, count=count_d0, seed=args.seed,
            )
            profile_d1 = build_profile(
                centroids_d1, mu_d1, target_id, ids, vocab,
                word=target, period=period_files[1], checkpoint=checkpoint,
                layer=args.layer, count=count_d1, seed=args.seed,
            )
            displacements[target] = compare_profiles(profile_d0, profile_d1)

    rows = []
    neighbors_by_target: dict[str, list[tuple[str, float]]] = {}
    for target in targets:
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
    summary = {"n_targets": len(targets), "top_k": args.top_k, "n_active_support": int(active_mask.sum())}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
