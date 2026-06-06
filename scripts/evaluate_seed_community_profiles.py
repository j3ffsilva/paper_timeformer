#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.evaluate_hidden_relational_profiles import (  # noqa: E402
    contextual_centroids,
    read_truth,
    write_csv,
)
from timeformers.relational import jensen_shannon_divergence_rows  # noqa: E402


DEFAULT_FIELDS = {
    "geometry": ["line", "angle", "surface", "plate", "column"],
    "transport": ["boat", "ship", "rail", "route", "engine"],
    "leadership": ["secretary", "director", "president", "commissioner", "governor"],
    "botanical": ["soil", "stock", "vine", "plant", "seed"],
    "medicine": ["cell", "bone", "skin", "patient", "organ"],
    "corruption": ["corruption", "fraud", "payment", "investigation", "political"],
}


def community_memberships(
    stats: dict,
    layer: str,
    target_index: int,
    centroids: torch.Tensor,
    reference_ids: list[int],
    field_ids: list[list[int]],
    *,
    temperature: float,
) -> torch.Tensor:
    selected = stats["occurrence_targets"] == target_index
    occurrences = stats["occurrence_vectors"][layer][selected].float()
    global_mean = centroids[reference_ids].mean(dim=0, keepdim=True)
    occurrences = F.normalize(occurrences - global_mean, dim=1)
    affinities = []
    for ids in field_ids:
        seeds = F.normalize(centroids[ids].float() - global_mean, dim=1)
        affinities.append((occurrences @ seeds.T).mean(dim=1))
    return torch.softmax(torch.stack(affinities, dim=1) / temperature, dim=1)


def evaluate(rows: list[dict], truth: dict[str, dict[str, float]]) -> list[dict]:
    metrics = []
    for temperature in sorted({row["temperature"] for row in rows}):
        selected = sorted(
            [row for row in rows if row["temperature"] == temperature],
            key=lambda row: row["target"],
        )
        scores = np.array([row["community_jsd"] for row in selected])
        graded = np.array([truth[row["target"]]["graded"] for row in selected])
        binary = np.array([truth[row["target"]]["binary"] for row in selected])
        rho, p_value = spearmanr(graded, scores)
        metrics.append(
            {
                "temperature": temperature,
                "spearman": float(rho),
                "spearman_p": float(p_value),
                "roc_auc": float(roc_auc_score(binary, scores)),
                "average_precision": float(average_precision_score(binary, scores)),
            }
        )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate fixed lexical seed communities.")
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--layer", default="layer_2")
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.5, 1.0])
    args = parser.parse_args()

    vocab = json.loads((args.experiment_dir / "vocab.json").read_text(encoding="utf-8"))
    targets = json.loads((args.experiment_dir / "targets.json").read_text(encoding="utf-8"))
    references = json.loads((args.profile_dir / "references.json").read_text(encoding="utf-8"))
    token_to_id = {token: index for index, token in enumerate(vocab)}
    reference_ids = [token_to_id[token] for token in references]
    fields = {
        name: [token for token in tokens if token in token_to_id]
        for name, tokens in DEFAULT_FIELDS.items()
    }
    field_names = list(fields)
    field_ids = [[token_to_id[token] for token in fields[name]] for name in field_names]

    stats_by_cell = {
        (checkpoint, corpus): torch.load(
            args.profile_dir / "cache" / f"theta{checkpoint}_d{corpus}.pt",
            weights_only=True,
        )
        for checkpoint in range(2)
        for corpus in range(2)
    }
    centroids = {
        cell: contextual_centroids(stats, args.layer)
        for cell, stats in stats_by_cell.items()
    }

    rows = []
    profiles = {}
    for temperature in args.temperatures:
        for cell, stats in stats_by_cell.items():
            cell_profiles = []
            for target_index in range(len(targets)):
                memberships = community_memberships(
                    stats,
                    args.layer,
                    target_index,
                    centroids[cell],
                    reference_ids,
                    field_ids,
                    temperature=temperature,
                )
                cell_profiles.append(memberships.mean(dim=0))
            profiles[(temperature, *cell)] = torch.stack(cell_profiles)

        natural = jensen_shannon_divergence_rows(
            profiles[(temperature, 0, 0)],
            profiles[(temperature, 1, 1)],
        )
        corpus_theta0 = jensen_shannon_divergence_rows(
            profiles[(temperature, 0, 0)],
            profiles[(temperature, 0, 1)],
        )
        corpus_theta1 = jensen_shannon_divergence_rows(
            profiles[(temperature, 1, 0)],
            profiles[(temperature, 1, 1)],
        )
        checkpoint_d0 = jensen_shannon_divergence_rows(
            profiles[(temperature, 0, 0)],
            profiles[(temperature, 1, 0)],
        )
        checkpoint_d1 = jensen_shannon_divergence_rows(
            profiles[(temperature, 0, 1)],
            profiles[(temperature, 1, 1)],
        )
        for target_index, target in enumerate(targets):
            row = {
                "target": target,
                "temperature": temperature,
                "community_jsd": float(natural[target_index]),
                "corpus_theta0_jsd": float(corpus_theta0[target_index]),
                "corpus_theta1_jsd": float(corpus_theta1[target_index]),
                "checkpoint_d0_jsd": float(checkpoint_d0[target_index]),
                "checkpoint_d1_jsd": float(checkpoint_d1[target_index]),
            }
            for period, cell in (("d0", (0, 0)), ("d1", (1, 1))):
                profile = profiles[(temperature, *cell)][target_index]
                for field_index, field in enumerate(field_names):
                    row[f"{period}_{field}"] = float(profile[field_index])
            rows.append(row)

    truth = read_truth(args.truth)
    metrics = evaluate(rows, truth)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.output_dir / "scores.csv")
    write_csv(metrics, args.output_dir / "metrics.csv")
    (args.output_dir / "fields.json").write_text(json.dumps(fields, indent=2), encoding="utf-8")

    diagnostic = {}
    for temperature in args.temperatures:
        selected = sorted(
            [row for row in rows if row["temperature"] == temperature],
            key=lambda row: row["community_jsd"],
            reverse=True,
        )
        rank = {row["target"]: index + 1 for index, row in enumerate(selected)}
        diagnostic[str(temperature)] = {
            target: {
                "rank": rank[target],
                **next(row for row in selected if row["target"] == target),
            }
            for target in ("plane_nn", "chairman_nn", "graft_nn", "tree_nn")
        }
    summary = {"layer": args.layer, "metrics": metrics, "diagnostic": diagnostic}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
