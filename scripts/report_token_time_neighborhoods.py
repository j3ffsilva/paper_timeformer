#!/usr/bin/env python3
"""Report `token@time` neighborhoods, gains/losses and rankings between two
periods, from caches produced by `build_token_time_profiles.py`.

Implements Fase A items 2 and 5 of `docs/39-token_time_analysis_framework.md`:
"produzir vizinhos, ganhos e perdas" and "gerar rankings separados de
deslocamento e turnover".

Reuses, without duplicating formulas:

- variant-D centering, `relational_profile` and `displacement` (the
  `Delta(w) = 1 - cos(P_t0(w), P_t1(w))` of capítulo 08) from
  `evaluate_relational_profile_v2.py`;
- `neighborhood_rows`/`standardize`/`rank_descending`/`format_table` from
  `report_temporal_relational_neighborhoods.py`.

Generic over corpus/targets (Fase A.6): everything needed is read from the
`--profile-dir` written by `build_token_time_profiles.py` plus two cache
files `--cache-d0`/`--cache-d1`.
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

from timeformers.token_time_repository import TokenTimeIndex  # noqa: E402

try:
    from scripts.report_temporal_relational_neighborhoods import (
        format_table,
        neighborhood_rows,
        top_rows,
    )
except ModuleNotFoundError:
    from report_temporal_relational_neighborhoods import (
        format_table,
        neighborhood_rows,
        top_rows,
    )


def turnover_at_k(rows: list[dict], k: int) -> float:
    top_d0 = {row["reference"] for row in rows if row["rank_d0"] <= k}
    top_d1 = {row["reference"] for row in rows if row["rank_d1"] <= k}
    union = top_d0 | top_d1
    if not union:
        return 0.0
    return 1.0 - len(top_d0 & top_d1) / len(union)


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    path: Path,
    *,
    rows_by_target: dict[str, list[dict]],
    rankings: list[dict],
    layer: str,
    top_k: int,
    salience_rank: int,
    turnover_k: int,
) -> None:
    lines = [
        "# token@time neighborhoods (Fase A)",
        "",
        "Encoder fixo (capítulo 08/09): o mesmo checkpoint mediu D0 e D1.",
        "Centralização variante D (capítulo 08, Fase 1): `mu_t` é a média não",
        "ponderada dos centróides de tipo sobre V_ativo.",
        "",
        "```text",
        "P_t(w)[v] = cos(centroid_t(w) - mu_t, centroid_t(v) - mu_t)",
        "Delta(w) = 1 - cos(P_t0(w), P_t1(w))",
        "z_t(w)[v] = standardize_v(P_t(w)[v])",
        "delta_z(w)[v] = z_1(w)[v] - z_0(w)[v]",
        "```",
        "",
        f"- layer: `{layer}`",
        f"- references shown per target: `{len(next(iter(rows_by_target.values())))}`",
        f"- gain/loss salience cutoff: top `{salience_rank}` in D0 or D1",
        f"- turnover@{turnover_k}: fração de troca entre os top-{turnover_k}",
        "  vizinhos de D0 e D1 (Jaccard distance)",
        "",
        "## Rankings",
        "",
        "| target | displacement (1-cos) | turnover@%d | n_d0 | n_d1 |" % turnover_k,
        "|---|---:|---:|---:|---:|",
    ]
    for row in rankings:
        lines.append(
            f"| `{row['target']}` | {row['displacement']:.3f} | "
            f"{row['turnover']:.3f} | {row['count_d0']} | {row['count_d1']} |"
        )
    lines.append("")

    for target, rows in rows_by_target.items():
        lines.extend([f"## `{target}`", "", "### Nearest in D0", ""])
        lines.extend(format_table(top_rows(rows, "similarity_d0", top_k), "similarity_d0"))
        lines.extend(["", "### Nearest in D1", ""])
        lines.extend(format_table(top_rows(rows, "similarity_d1", top_k), "similarity_d1"))
        salient_rows = [
            row for row in rows if min(row["rank_d0"], row["rank_d1"]) <= salience_rank
        ]
        lines.extend(["", "### Largest relative gains", ""])
        lines.extend(format_table(top_rows(salient_rows, "delta_z", top_k), "delta_z"))
        lines.extend(["", "### Largest relative losses", ""])
        lines.extend(
            format_table(top_rows(salient_rows, "delta_z", top_k, reverse=False), "delta_z")
        )
        lines.append("")

    lines.extend(
        [
            "## Interpretation limits",
            "",
            "1. Uma referência é uma coordenada lexical (WordPiece de palavra inteira),",
            "   não um sentido atribuído manualmente.",
            "2. `delta_z` é relativo às outras referências do mesmo alvo; uma",
            "   referência que mudou pode contribuir para `delta_z` sem que o alvo",
            "   tenha mudado.",
            "3. `displacement` é calculado sobre V_ativo completo (inclui WordPieces",
            "   fragmentários), enquanto as tabelas de vizinhos mostram só referências",
            "   de palavra inteira -- os dois números não são diretamente comparáveis.",
            "",
        ]
    )
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
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--salience-rank", type=int, default=50)
    parser.add_argument("--turnover-k", type=int, default=20)
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
    reference_tokens = [idx.vocab[index] for index in reference_ids_t.tolist()]

    rows_by_target: dict[str, list[dict]] = {}
    all_rows: list[dict] = []
    rankings: list[dict] = []
    for target in idx.targets:
        target_id = idx.target_ids[target]
        count_d0 = int(idx.periods[0].counts[target_id])
        count_d1 = int(idx.periods[1].counts[target_id])

        disp = idx.displacement(target, active_ids, layer=args.layer, n_min_active=args.n_min_active).score

        displacement_ref = idx.displacement(target, reference_ids_t, layer=args.layer, n_min_active=args.n_min_active)
        rows = neighborhood_rows(
            target=target,
            references=reference_tokens,
            before=displacement_ref.profile_a,
            after=displacement_ref.profile_b,
        )
        rows_by_target[target] = rows
        all_rows.extend(rows)

        rankings.append(
            {
                "target": target,
                "displacement": disp,
                "turnover": turnover_at_k(rows, args.turnover_k),
                "count_d0": count_d0,
                "count_d1": count_d1,
            }
        )

    rankings.sort(key=lambda row: row["displacement"], reverse=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(all_rows, args.output_dir / "neighborhoods.csv")
    write_csv(rankings, args.output_dir / "rankings.csv")
    (args.output_dir / "references.json").write_text(
        json.dumps(reference_tokens, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(
        args.output_dir / "report.md",
        rows_by_target=rows_by_target,
        rankings=rankings,
        layer=args.layer,
        top_k=args.top_k,
        salience_rank=args.salience_rank,
        turnover_k=args.turnover_k,
    )
    summary = {
        "layer": args.layer,
        "n_targets": len(idx.targets),
        "n_active_support": int(active_mask.sum()),
        "n_references": len(reference_tokens),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
