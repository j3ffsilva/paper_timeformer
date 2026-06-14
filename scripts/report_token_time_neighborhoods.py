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

from timeformers.real_corpus import SPECIAL_TOKENS  # noqa: E402
from timeformers.relational import build_active_support, contextual_centroids, type_uniform_mean  # noqa: E402
from timeformers.token_time import build_profile, compare_profiles  # noqa: E402

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


def build_reference_set(
    vocab: list[str],
    active_support_mask: torch.Tensor,
    *,
    targets: set[str],
    counts_d0: torch.Tensor,
    counts_d1: torch.Tensor,
    max_references: int,
) -> list[int]:
    """Whole-word, alphabetic, non-special tokens from V_ativo.

    A subset of V_ativo restricted to human-readable lexical references for
    the neighborhood report. V_ativo itself (used for `mu_t` and
    `displacement`) keeps WordPiece fragments -- only the *reported*
    references are filtered (history/27 §"detalhe antes de implementar").
    """
    counts_min = torch.minimum(counts_d0, counts_d1).float()
    candidates = []
    for index, token in enumerate(vocab):
        if not active_support_mask[index]:
            continue
        if token in SPECIAL_TOKENS or token in targets:
            continue
        if token.startswith("##"):
            continue
        if not token.isalpha():
            continue
        candidates.append(index)
    candidates.sort(key=lambda index: counts_min[index].item(), reverse=True)
    return candidates[:max_references]


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
    reference_tokens = [vocab[index] for index in reference_ids]
    reference_ids_t = torch.tensor(reference_ids, dtype=torch.long)

    centroids_d0 = contextual_centroids(stats_d0, args.layer)
    centroids_d1 = contextual_centroids(stats_d1, args.layer)
    mu_d0 = type_uniform_mean(stats_d0, args.layer, support=active_mask)
    mu_d1 = type_uniform_mean(stats_d1, args.layer, support=active_mask)

    rows_by_target: dict[str, list[dict]] = {}
    all_rows: list[dict] = []
    rankings: list[dict] = []
    for target in targets:
        target_id = target_ids[target]
        count_d0 = int(stats_d0["counts"][target_id])
        count_d1 = int(stats_d1["counts"][target_id])

        profile_active_d0 = build_profile(
            centroids_d0, mu_d0, target_id, active_ids, vocab,
            word=target, period=period_files[0], checkpoint=checkpoint,
            layer=args.layer, count=count_d0, seed=args.seed,
        )
        profile_active_d1 = build_profile(
            centroids_d1, mu_d1, target_id, active_ids, vocab,
            word=target, period=period_files[1], checkpoint=checkpoint,
            layer=args.layer, count=count_d1, seed=args.seed,
        )
        disp = compare_profiles(profile_active_d0, profile_active_d1).score

        profile_ref_d0 = build_profile(
            centroids_d0, mu_d0, target_id, reference_ids_t, vocab,
            word=target, period=period_files[0], checkpoint=checkpoint,
            layer=args.layer, count=count_d0, seed=args.seed,
        )
        profile_ref_d1 = build_profile(
            centroids_d1, mu_d1, target_id, reference_ids_t, vocab,
            word=target, period=period_files[1], checkpoint=checkpoint,
            layer=args.layer, count=count_d1, seed=args.seed,
        )
        rows = neighborhood_rows(
            target=target,
            references=reference_tokens,
            before=profile_ref_d0.vector,
            after=profile_ref_d1.vector,
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
        "n_targets": len(targets),
        "n_active_support": int(active_mask.sum()),
        "n_references": len(reference_tokens),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
