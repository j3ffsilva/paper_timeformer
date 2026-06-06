#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor


DEFAULT_AUDIT_TARGETS = [
    "plane_nn",
    "graft_nn",
    "chairman_nn",
    "tree_nn",
    "attack_nn",
    "record_nn",
    "stab_nn",
]


def contextual_centroids(stats: dict, layer: str) -> Tensor:
    counts = stats["counts"].float().unsqueeze(1).clamp_min(1.0)
    return stats["sums"][layer].float() / counts


def relational_profile(
    centroids: Tensor,
    target_id: int,
    reference_ids: list[int],
    *,
    center: bool,
) -> Tensor:
    points = centroids.float()
    references = points[reference_ids]
    target = points[target_id : target_id + 1]
    if center:
        reference_mean = references.mean(dim=0, keepdim=True)
        references = references - reference_mean
        target = target - reference_mean
    target = F.normalize(target, dim=1)
    references = F.normalize(references, dim=1)
    return (target @ references.T).squeeze(0)


def standardize(profile: Tensor) -> Tensor:
    std = profile.std(unbiased=False).clamp_min(1e-9)
    return (profile - profile.mean()) / std


def rank_descending(values: Tensor) -> Tensor:
    order = torch.argsort(values, descending=True)
    ranks = torch.empty_like(order)
    ranks[order] = torch.arange(1, values.numel() + 1, device=values.device)
    return ranks


def neighborhood_rows(
    *,
    target: str,
    references: list[str],
    before: Tensor,
    after: Tensor,
) -> list[dict]:
    before_z = standardize(before)
    after_z = standardize(after)
    before_rank = rank_descending(before)
    after_rank = rank_descending(after)
    rows = []
    for index, reference in enumerate(references):
        rows.append(
            {
                "target": target,
                "reference": reference,
                "similarity_d0": float(before[index]),
                "similarity_d1": float(after[index]),
                "delta_similarity": float(after[index] - before[index]),
                "z_d0": float(before_z[index]),
                "z_d1": float(after_z[index]),
                "delta_z": float(after_z[index] - before_z[index]),
                "rank_d0": int(before_rank[index]),
                "rank_d1": int(after_rank[index]),
                "rank_gain": int(before_rank[index] - after_rank[index]),
            }
        )
    return rows


def top_rows(rows: list[dict], key: str, n: int, *, reverse: bool = True) -> list[dict]:
    return sorted(rows, key=lambda row: row[key], reverse=reverse)[:n]


def format_table(rows: list[dict], value_key: str) -> list[str]:
    lines = ["| reference | value | rank D0 | rank D1 |", "|---|---:|---:|---:|"]
    for row in rows:
        lines.append(
            f"| `{row['reference']}` | {row[value_key]:.3f} | "
            f"{row['rank_d0']} | {row['rank_d1']} |"
        )
    return lines


def write_report(
    path: Path,
    *,
    rows_by_target: dict[str, list[dict]],
    audit_targets: list[str],
    layer: str,
    center: bool,
    top_k: int,
    salience_rank: int,
) -> None:
    lines = [
        "# Temporal relational neighborhoods",
        "",
        "This report characterizes what each target gained and lost in relational",
        "proximity. It is not a supervised sense classification.",
        "",
        "## Definition",
        "",
        "```text",
        "r_t(w)[v] = cos(centroid_t(w), centroid_t(v))",
        "z_t(w)[v] = standardize_v(r_t(w)[v])",
        "delta_z(w)[v] = z_1(w)[v] - z_0(w)[v]",
        "```",
        "",
        f"- layer: `{layer}`",
        f"- centered geometry: `{center}`",
        f"- references: `{len(next(iter(rows_by_target.values())))}`",
        f"- gain/loss salience cutoff: top `{salience_rank}` in D0 or D1",
        f"- detailed targets were fixed before inspecting this report: "
        f"`{', '.join(audit_targets)}`",
        "",
        "Raw similarities describe each local checkpoint geometry. `delta_z` is",
        "the primary gain/loss diagnostic because it removes target-wise shifts in",
        "the mean and scale of similarities. It does not remove semantic drift of",
        "the reference tokens themselves.",
        "",
    ]
    for target in audit_targets:
        rows = rows_by_target.get(target)
        if rows is None:
            continue
        lines.extend([f"## `{target}`", "", "### Nearest in D0", ""])
        lines.extend(format_table(top_rows(rows, "similarity_d0", top_k), "similarity_d0"))
        lines.extend(["", "### Nearest in D1", ""])
        lines.extend(format_table(top_rows(rows, "similarity_d1", top_k), "similarity_d1"))
        salient_rows = [
            row
            for row in rows
            if min(row["rank_d0"], row["rank_d1"]) <= salience_rank
        ]
        lines.extend(["", "### Largest relative gains", ""])
        lines.extend(
            format_table(top_rows(salient_rows, "delta_z", top_k), "delta_z")
        )
        lines.extend(["", "### Largest relative losses", ""])
        lines.extend(
            format_table(
                top_rows(salient_rows, "delta_z", top_k, reverse=False), "delta_z"
            )
        )
        lines.append("")

    lines.extend(
        [
            "## Interpretation limits",
            "",
            "1. A reference is a lexical coordinate, not a manually assigned sense.",
            "2. A changing reference can contribute to `delta_z`; gains are relational.",
            "3. Centroid neighborhoods summarize usages and can blur polysemy.",
            "4. The detailed audit targets are illustrative; all targets are present",
            "   in `neighborhoods.csv` to prevent selective computation.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report temporal relational neighborhoods from cached centroids."
    )
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layer", default="layer_2")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--salience-rank", type=int, default=50)
    parser.add_argument("--center", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--audit-targets", nargs="*", default=DEFAULT_AUDIT_TARGETS)
    args = parser.parse_args()

    vocab = json.loads((args.experiment_dir / "vocab.json").read_text(encoding="utf-8"))
    targets = json.loads((args.experiment_dir / "targets.json").read_text(encoding="utf-8"))
    references = json.loads((args.profile_dir / "references.json").read_text(encoding="utf-8"))
    token_to_id = {token: index for index, token in enumerate(vocab)}
    reference_ids = [token_to_id[token] for token in references]

    before_stats = torch.load(
        args.profile_dir / "cache" / "theta0_d0.pt",
        map_location="cpu",
        weights_only=True,
    )
    after_stats = torch.load(
        args.profile_dir / "cache" / "theta1_d1.pt",
        map_location="cpu",
        weights_only=True,
    )
    before_centroids = contextual_centroids(before_stats, args.layer)
    after_centroids = contextual_centroids(after_stats, args.layer)

    rows_by_target = {}
    all_rows = []
    for target in targets:
        target_id = token_to_id[target]
        before = relational_profile(
            before_centroids, target_id, reference_ids, center=args.center
        )
        after = relational_profile(
            after_centroids, target_id, reference_ids, center=args.center
        )
        rows = neighborhood_rows(
            target=target,
            references=references,
            before=before,
            after=after,
        )
        rows_by_target[target] = rows
        all_rows.extend(rows)

    unknown_targets = sorted(set(args.audit_targets) - set(targets))
    if unknown_targets:
        raise ValueError(f"Unknown audit targets: {', '.join(unknown_targets)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "neighborhoods.csv", all_rows)
    write_report(
        args.output_dir / "report.md",
        rows_by_target=rows_by_target,
        audit_targets=args.audit_targets,
        layer=args.layer,
        center=args.center,
        top_k=args.top_k,
        salience_rank=args.salience_rank,
    )
    summary = {
        "layer": args.layer,
        "center": args.center,
        "n_targets": len(targets),
        "n_references": len(references),
        "audit_targets": args.audit_targets,
        "top_k": args.top_k,
        "salience_rank": args.salience_rank,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
