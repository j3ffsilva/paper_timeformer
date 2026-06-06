#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from itertools import combinations
from pathlib import Path

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_mutual_info_score, average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.evaluate_hidden_relational_profiles import (  # noqa: E402
    contextual_centroids,
    read_truth,
    write_csv,
)
from timeformers.relational import jensen_shannon_divergence_rows  # noqa: E402


def centered_normalized(points: torch.Tensor, reference_ids: list[int]) -> torch.Tensor:
    points = points.float()
    mean = points[reference_ids].mean(dim=0, keepdim=True)
    return F.normalize(points - mean, dim=1)


def knn_graph(points: torch.Tensor, *, k: int, mode: str) -> nx.Graph:
    if mode not in {"mutual", "union"}:
        raise ValueError("mode must be 'mutual' or 'union'")
    similarities = points @ points.T
    similarities.fill_diagonal_(-torch.inf)
    neighbor_indices = torch.topk(similarities, k=min(k, points.size(0) - 1), dim=1).indices
    neighbor_sets = [set(row.tolist()) for row in neighbor_indices]
    graph = nx.Graph()
    graph.add_nodes_from(range(points.size(0)))
    for source, neighbors in enumerate(neighbor_sets):
        for target in neighbors:
            reciprocal = source in neighbor_sets[target]
            if source < target and (mode == "union" or reciprocal):
                weight = float(similarities[source, target])
                if weight > 0.0:
                    graph.add_edge(source, target, weight=weight)
            elif mode == "union" and target < source and not graph.has_edge(source, target):
                weight = float(similarities[source, target])
                if weight > 0.0:
                    graph.add_edge(source, target, weight=weight)
    return graph


def mutual_knn_graph(points: torch.Tensor, *, k: int) -> nx.Graph:
    return knn_graph(points, k=k, mode="mutual")


def labels_from_communities(communities: list[set[int]], n_nodes: int) -> np.ndarray:
    labels = np.full(n_nodes, -1, dtype=int)
    for label, community in enumerate(communities):
        labels[list(community)] = label
    if np.any(labels < 0):
        raise ValueError("Every graph node must belong to a community")
    return labels


def partition_diagnostics(
    graph: nx.Graph,
    partitions: list[list[set[int]]],
    *,
    k: int,
    resolution: float,
) -> dict:
    labels = [labels_from_communities(partition, graph.number_of_nodes()) for partition in partitions]
    stability = [
        adjusted_mutual_info_score(left, right)
        for left, right in combinations(labels, 2)
    ]
    modularities = [
        nx.community.modularity(graph, partition, weight="weight", resolution=resolution)
        for partition in partitions
    ]
    representative = partitions[0]
    sizes = np.array([len(community) for community in representative])
    return {
        "k": k,
        "resolution": resolution,
        "mean_ami": float(np.mean(stability)) if stability else 1.0,
        "min_ami": float(np.min(stability)) if stability else 1.0,
        "mean_modularity": float(np.mean(modularities)),
        "n_communities": len(representative),
        "median_community_size": float(np.median(sizes)),
        "max_community_fraction": float(sizes.max() / graph.number_of_nodes()),
        "singleton_fraction": float(np.mean(sizes == 1)),
        "n_edges": graph.number_of_edges(),
    }


def choose_setting(settings: list[dict]) -> dict:
    eligible = [
        setting
        for setting in settings
        if 10 <= setting["n_communities"] <= 200
        and setting["max_community_fraction"] <= 0.25
        and setting["singleton_fraction"] <= 0.10
    ]
    candidates = eligible or settings
    return max(
        candidates,
        key=lambda setting: (
            setting["mean_ami"],
            setting["min_ami"],
            setting["mean_modularity"],
        ),
    )


def community_prototypes(
    normalized_centroids: torch.Tensor,
    reference_ids: list[int],
    communities: list[set[int]],
) -> torch.Tensor:
    references = normalized_centroids[reference_ids]
    return torch.stack(
        [references[list(community)].mean(dim=0) for community in communities]
    )


def target_community_profiles(
    stats: dict,
    layer: str,
    normalized_centroids: torch.Tensor,
    reference_ids: list[int],
    communities: list[set[int]],
    *,
    temperature: float,
) -> torch.Tensor:
    prototypes = community_prototypes(
        normalized_centroids,
        reference_ids,
        communities,
    )
    global_mean = contextual_centroids(stats, layer)[reference_ids].mean(dim=0, keepdim=True)
    profiles = []
    n_targets = int(stats["occurrence_targets"].max()) + 1
    for target_index in range(n_targets):
        selected = stats["occurrence_targets"] == target_index
        occurrences = F.normalize(
            stats["occurrence_vectors"][layer][selected].float() - global_mean,
            dim=1,
        )
        affinities = occurrences @ prototypes.T
        profiles.append(torch.softmax(affinities / temperature, dim=1).mean(dim=0))
    return torch.stack(profiles)


def evaluate_rows(rows: list[dict], truth: dict[str, dict[str, float]]) -> list[dict]:
    metrics = []
    keys = sorted({(row["k"], row["resolution"], row["temperature"]) for row in rows})
    for k, resolution, temperature in keys:
        selected = sorted(
            [
                row
                for row in rows
                if row["k"] == k
                and row["resolution"] == resolution
                and row["temperature"] == temperature
            ],
            key=lambda row: row["target"],
        )
        graded = np.array([truth[row["target"]]["graded"] for row in selected])
        binary = np.array([truth[row["target"]]["binary"] for row in selected])
        scores = np.array([row["natural_jsd"] for row in selected])
        frozen = np.array(
            [
                0.5 * (row["corpus_theta0_jsd"] + row["corpus_theta1_jsd"])
                for row in selected
            ]
        )
        rho, p_value = spearmanr(graded, scores)
        frozen_rho, frozen_p = spearmanr(graded, frozen)
        metrics.append(
            {
                "k": k,
                "resolution": resolution,
                "temperature": temperature,
                "spearman_natural": float(rho),
                "spearman_natural_p": float(p_value),
                "roc_auc_natural": float(roc_auc_score(binary, scores)),
                "average_precision_natural": float(average_precision_score(binary, scores)),
                "spearman_frozen_corpus": float(frozen_rho),
                "spearman_frozen_corpus_p": float(frozen_p),
                "roc_auc_frozen_corpus": float(roc_auc_score(binary, frozen)),
            }
        )
    return metrics


def write_community_report(
    path: Path,
    communities: list[set[int]],
    references: list[str],
    *,
    top_n: int = 30,
) -> None:
    ordered = sorted(communities, key=len, reverse=True)
    lines = ["# Frozen D0 communities", ""]
    for index, community in enumerate(ordered):
        members = [references[node] for node in sorted(community)]
        lines.append(f"## Community {index} (n={len(members)})")
        lines.append("")
        lines.append(", ".join(members[:top_n]))
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover frozen D0 semantic communities and evaluate movement.")
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--layer", default="layer_2")
    parser.add_argument("--graph-mode", choices=["mutual", "union"], default="mutual")
    parser.add_argument("--k-values", type=int, nargs="+", default=[10, 20, 40])
    parser.add_argument("--resolutions", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0.02, 0.05, 0.1, 0.2])
    args = parser.parse_args()

    vocab = json.loads((args.experiment_dir / "vocab.json").read_text(encoding="utf-8"))
    targets = json.loads((args.experiment_dir / "targets.json").read_text(encoding="utf-8"))
    references = json.loads((args.profile_dir / "references.json").read_text(encoding="utf-8"))
    token_to_id = {token: index for index, token in enumerate(vocab)}
    reference_ids = [token_to_id[token] for token in references]
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
    normalized = {
        cell: centered_normalized(points, reference_ids)
        for cell, points in centroids.items()
    }
    d0_reference_points = normalized[(0, 0)][reference_ids]

    settings = []
    partitions_by_setting = {}
    graphs = {}
    for k in args.k_values:
        print(f"[graph] building {args.graph_mode}-kNN k={k}", flush=True)
        graph = knn_graph(d0_reference_points, k=k, mode=args.graph_mode)
        graphs[k] = graph
        for resolution in args.resolutions:
            partitions = [
                nx.community.louvain_communities(
                    graph,
                    weight="weight",
                    resolution=resolution,
                    seed=seed,
                )
                for seed in args.seeds
            ]
            diagnostic = partition_diagnostics(
                graph,
                partitions,
                k=k,
                resolution=resolution,
            )
            settings.append(diagnostic)
            partitions_by_setting[(k, resolution)] = partitions
            print(f"[partition] {diagnostic}", flush=True)

    selected = choose_setting(settings)
    selected_key = (selected["k"], selected["resolution"])
    selected_partition = partitions_by_setting[selected_key][0]
    print(f"[selected] {selected}", flush=True)

    rows = []
    all_partition_scores = {}
    for setting in settings:
        key = (setting["k"], setting["resolution"])
        partition = partitions_by_setting[key][0]
        for temperature in args.temperatures:
            profiles = {
                cell: target_community_profiles(
                    stats,
                    args.layer,
                    normalized[cell],
                    reference_ids,
                    partition,
                    temperature=temperature,
                )
                for cell, stats in stats_by_cell.items()
            }
            scores = {
                "natural_jsd": jensen_shannon_divergence_rows(profiles[(0, 0)], profiles[(1, 1)]),
                "corpus_theta0_jsd": jensen_shannon_divergence_rows(profiles[(0, 0)], profiles[(0, 1)]),
                "corpus_theta1_jsd": jensen_shannon_divergence_rows(profiles[(1, 0)], profiles[(1, 1)]),
                "checkpoint_d0_jsd": jensen_shannon_divergence_rows(profiles[(0, 0)], profiles[(1, 0)]),
                "checkpoint_d1_jsd": jensen_shannon_divergence_rows(profiles[(0, 1)], profiles[(1, 1)]),
            }
            for target_index, target in enumerate(targets):
                rows.append(
                    {
                        "target": target,
                        "k": setting["k"],
                        "resolution": setting["resolution"],
                        "temperature": temperature,
                        **{
                            name: float(values[target_index])
                            for name, values in scores.items()
                        },
                    }
                )
            all_partition_scores[(key, temperature)] = scores

    truth = read_truth(args.truth)
    metrics = evaluate_rows(rows, truth)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(settings, args.output_dir / "partition_stability.csv")
    write_csv(rows, args.output_dir / "scores.csv")
    write_csv(metrics, args.output_dir / "metrics.csv")
    (args.output_dir / "selected_setting.json").write_text(
        json.dumps(selected, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "selected_communities.json").write_text(
        json.dumps(
            [[references[node] for node in sorted(community)] for community in selected_partition],
            indent=2,
        ),
        encoding="utf-8",
    )
    write_community_report(
        args.output_dir / "communities.md",
        selected_partition,
        references,
    )

    selected_metrics = [
        metric
        for metric in metrics
        if metric["k"] == selected["k"] and metric["resolution"] == selected["resolution"]
    ]
    diagnostic = {}
    for metric in selected_metrics:
        temperature = metric["temperature"]
        selected_rows = [
            row
            for row in rows
            if row["k"] == selected["k"]
            and row["resolution"] == selected["resolution"]
            and row["temperature"] == temperature
        ]
        natural_ranking = sorted(selected_rows, key=lambda row: row["natural_jsd"], reverse=True)
        frozen_ranking = sorted(
            selected_rows,
            key=lambda row: 0.5 * (row["corpus_theta0_jsd"] + row["corpus_theta1_jsd"]),
            reverse=True,
        )
        natural_rank = {row["target"]: index + 1 for index, row in enumerate(natural_ranking)}
        frozen_rank = {row["target"]: index + 1 for index, row in enumerate(frozen_ranking)}
        diagnostic[str(temperature)] = {
            target: {
                "natural_rank": natural_rank[target],
                "frozen_corpus_rank": frozen_rank[target],
                **next(row for row in selected_rows if row["target"] == target),
            }
            for target in ("plane_nn", "chairman_nn", "graft_nn", "tree_nn")
        }
    summary = {
        "selected_by_structure": selected,
        "selected_metrics": selected_metrics,
        "diagnostic": diagnostic,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
