#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_auc_score, silhouette_score

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.research.common.io import read_truth, write_csv  # noqa: E402


AUDIT_TARGETS = ("plane_nn", "chairman_nn", "graft_nn", "tree_nn")


def target_occurrences(stats: dict, layer: str, target_index: int) -> np.ndarray:
    selected = stats["occurrence_targets"] == target_index
    return stats["occurrence_vectors"][layer][selected].float().numpy()


def balanced_sample(
    before: np.ndarray,
    after: np.ndarray,
    *,
    max_per_period: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    n = min(len(before), len(after), max_per_period)
    if n < 2:
        raise ValueError("Each period needs at least two occurrences")
    rng = np.random.default_rng(seed)
    left = before[rng.choice(len(before), n, replace=False)]
    right = after[rng.choice(len(after), n, replace=False)]
    return left, right


def centered_unit_rows(points: np.ndarray) -> np.ndarray:
    centered = points - points.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    return centered / np.maximum(norms, 1e-12)


def pooled_reference_anchors(
    before_stats: dict,
    after_stats: dict,
    layer: str,
    reference_ids: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    sums = before_stats["sums"][layer][reference_ids] + after_stats["sums"][layer][reference_ids]
    counts = (
        before_stats["counts"][reference_ids] + after_stats["counts"][reference_ids]
    ).float().unsqueeze(1).clamp_min(1.0)
    anchors = (sums / counts).float().numpy()
    anchor_mean = anchors.mean(axis=0, keepdims=True)
    return centered_unit_rows(anchors), anchor_mean


def relational_occurrence_vectors(
    before: np.ndarray,
    after: np.ndarray,
    anchors: np.ndarray,
    *,
    anchor_mean: np.ndarray | None = None,
    pca_components: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.concatenate([before, after], axis=0)
    if anchor_mean is None:
        anchor_mean = np.zeros((1, points.shape[1]), dtype=points.dtype)
    point_directions = points - anchor_mean
    point_directions /= np.maximum(
        np.linalg.norm(point_directions, axis=1, keepdims=True),
        1e-12,
    )
    profiles = point_directions @ anchors.T
    profiles = centered_unit_rows(profiles)
    n_components = min(pca_components, profiles.shape[0] - 1, profiles.shape[1])
    reduced = PCA(
        n_components=n_components,
        svd_solver="randomized",
        random_state=seed,
    ).fit_transform(profiles)
    return reduced[: len(before)], reduced[len(before) :]


def cluster_period_distributions(
    before: np.ndarray,
    after: np.ndarray,
    *,
    n_clusters: int,
    seed: int,
) -> dict:
    points = centered_unit_rows(np.concatenate([before, after], axis=0))
    labels = KMeans(
        n_clusters=n_clusters,
        n_init=10,
        random_state=seed,
        algorithm="lloyd",
    ).fit_predict(points)
    n_before = len(before)
    before_counts = np.bincount(labels[:n_before], minlength=n_clusters).astype(float)
    after_counts = np.bincount(labels[n_before:], minlength=n_clusters).astype(float)
    before_mass = before_counts / before_counts.sum()
    after_mass = after_counts / after_counts.sum()
    return {
        "jsd": float(jensenshannon(before_mass, after_mass, base=2.0) ** 2),
        "silhouette": float(
            silhouette_score(
                points,
                labels,
                metric="cosine",
                sample_size=min(1000, len(points)),
                random_state=seed,
            )
        ),
        "before_mass": before_mass,
        "after_mass": after_mass,
    }


def aggregate_runs(run_rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, int], list[dict]] = {}
    for row in run_rows:
        grouped.setdefault((row["target"], row["checkpoint"]), []).append(row)

    output = []
    for (target, checkpoint), rows in sorted(grouped.items()):
        scores = np.array([row["jsd"] for row in rows])
        silhouettes = np.array([row["silhouette"] for row in rows])
        output.append(
            {
                "target": target,
                "checkpoint": checkpoint,
                "score": float(np.median(scores)),
                "score_mean": float(scores.mean()),
                "score_std": float(scores.std(ddof=1)) if len(scores) > 1 else 0.0,
                "score_min": float(scores.min()),
                "score_max": float(scores.max()),
                "silhouette_mean": float(silhouettes.mean()),
                "n_runs": len(rows),
                "count_d0": rows[0]["count_d0"],
                "count_d1": rows[0]["count_d1"],
                "sampled_per_period": rows[0]["sampled_per_period"],
            }
        )
    return output


def combine_checkpoints(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["target"], []).append(row)
    output = []
    for target, target_rows in sorted(grouped.items()):
        by_checkpoint = {row["checkpoint"]: row for row in target_rows}
        scores = [by_checkpoint[index]["score"] for index in sorted(by_checkpoint)]
        output.append(
            {
                "target": target,
                "score": float(np.mean(scores)),
                "theta0_score": by_checkpoint[0]["score"],
                "theta1_score": by_checkpoint[1]["score"],
                "checkpoint_disagreement": float(abs(scores[0] - scores[1])),
                "count_d0": by_checkpoint[0]["count_d0"],
                "count_d1": by_checkpoint[0]["count_d1"],
            }
        )
    return output


def evaluate(rows: list[dict], truth: dict[str, dict[str, float]]) -> dict:
    selected = sorted(rows, key=lambda row: row["target"])
    scores = np.array([row["score"] for row in selected])
    graded = np.array([truth[row["target"]]["graded"] for row in selected])
    binary = np.array([truth[row["target"]]["binary"] for row in selected])
    rho, p_value = spearmanr(graded, scores)
    return {
        "n_targets": len(selected),
        "spearman": float(rho),
        "spearman_p": float(p_value),
        "roc_auc": float(roc_auc_score(binary, scores)),
        "average_precision": float(average_precision_score(binary, scores)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure temporal semantic change through target-specific contextual usage clusters."
    )
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layer", default="layer_2")
    parser.add_argument("--representation", choices=["raw", "relational"], default="raw")
    parser.add_argument("--pca-components", type=int, default=32)
    parser.add_argument("--clusters", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--max-per-period", type=int, default=500)
    args = parser.parse_args()

    targets = json.loads((args.experiment_dir / "targets.json").read_text(encoding="utf-8"))
    vocab = json.loads((args.experiment_dir / "vocab.json").read_text(encoding="utf-8"))
    token_to_id = {token: index for index, token in enumerate(vocab)}
    references = json.loads((args.profile_dir / "references.json").read_text(encoding="utf-8"))
    reference_ids = [token_to_id[token] for token in references]
    stats = {
        (checkpoint, corpus): torch.load(
            args.profile_dir / "cache" / f"theta{checkpoint}_d{corpus}.pt",
            map_location="cpu",
            weights_only=True,
        )
        for checkpoint in range(2)
        for corpus in range(2)
    }

    run_rows = []
    for checkpoint in range(2):
        anchors = None
        anchor_mean = None
        if args.representation == "relational":
            anchors, anchor_mean = pooled_reference_anchors(
                stats[(checkpoint, 0)],
                stats[(checkpoint, 1)],
                args.layer,
                reference_ids,
            )
        for target_index, target in enumerate(targets):
            before_all = target_occurrences(stats[(checkpoint, 0)], args.layer, target_index)
            after_all = target_occurrences(stats[(checkpoint, 1)], args.layer, target_index)
            sampled_n = min(len(before_all), len(after_all), args.max_per_period)
            for seed in args.seeds:
                before, after = balanced_sample(
                    before_all,
                    after_all,
                    max_per_period=args.max_per_period,
                    seed=seed,
                )
                if anchors is not None:
                    before, after = relational_occurrence_vectors(
                        before,
                        after,
                        anchors,
                        anchor_mean=anchor_mean,
                        pca_components=args.pca_components,
                        seed=seed,
                    )
                for n_clusters in args.clusters:
                    if n_clusters >= len(before) + len(after):
                        continue
                    result = cluster_period_distributions(
                        before,
                        after,
                        n_clusters=n_clusters,
                        seed=seed,
                    )
                    run_rows.append(
                        {
                            "target": target,
                            "checkpoint": checkpoint,
                            "n_clusters": n_clusters,
                            "seed": seed,
                            "jsd": result["jsd"],
                            "silhouette": result["silhouette"],
                            "count_d0": len(before_all),
                            "count_d1": len(after_all),
                            "sampled_per_period": sampled_n,
                            "mass_d0": json.dumps(result["before_mass"].tolist()),
                            "mass_d1": json.dumps(result["after_mass"].tolist()),
                        }
                    )
            print(
                f"[usage-clusters] theta={checkpoint} target={target} "
                f"counts={len(before_all)}/{len(after_all)}",
                flush=True,
            )

    checkpoint_rows = aggregate_runs(run_rows)
    combined_rows = combine_checkpoints(checkpoint_rows)
    truth = read_truth(args.truth)
    metrics = evaluate(combined_rows, truth)
    ranking = sorted(combined_rows, key=lambda row: row["score"], reverse=True)
    ranks = {row["target"]: index + 1 for index, row in enumerate(ranking)}
    audit = {
        target: {"rank": ranks[target], **next(row for row in ranking if row["target"] == target)}
        for target in AUDIT_TARGETS
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(run_rows, args.output_dir / "runs.csv")
    write_csv(checkpoint_rows, args.output_dir / "checkpoint_scores.csv")
    write_csv(combined_rows, args.output_dir / "scores.csv")
    summary = {
        "method": "target-specific contextual usage clustering",
        "layer": args.layer,
        "representation": args.representation,
        "pca_components": args.pca_components if args.representation == "relational" else None,
        "clusters": args.clusters,
        "seeds": args.seeds,
        "max_per_period": args.max_per_period,
        "score": "mean checkpoint-frozen median JSD across cluster counts and seeds",
        "metrics": metrics,
        "audit": audit,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
