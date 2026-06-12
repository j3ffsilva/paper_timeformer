#!/usr/bin/env python3
"""Tarefa 1 (adendo encoder-fixo) -- mede mudanca com o encoder FIXO.

A grade 2x2 (7.14) mostrou que o eixo que separa t0 de t1 e' quase todo
drift de checkpoint (theta0 -> theta1), nao conteudo do corpus. A
configuracao usada em todas as rodadas anteriores (theta0_d0 vs theta1_d1,
"diagonal") mistura mudanca semantica com essa deformacao.

Este script repete as duas metricas principais (APD sobre ocorrencias,
Delta do perfil relacional) mas com o encoder FIXO: para um theta fixo em
{theta0, theta1}, compara d0 vs d1 (em vez de comparar theta0_d0 vs
theta1_d1). Tambem reporta NMI(cluster, lado) por palavra nesta mesma
configuracao -- sob encoder fixo esse NMI deixa de ser artefato de drift e
passa a medir separabilidade de uso entre periodos.

Nao requer reextracao: opera sobre
outputs/<experiment>/hidden_relational_profiles/cache/theta{0,1}_d{0,1}.pt
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import (
    average_precision_score,
    normalized_mutual_info_score,
    roc_auc_score,
)
from torch import Tensor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_occurrence_apd_v2 import (  # noqa: E402
    average_pairwise_distance,
    occurrences_for_target,
    read_truth,
)
from evaluate_relational_profile_v2 import (  # noqa: E402
    build_active_support,
    contextual_centroids,
    displacement,
    relational_profile,
    type_uniform_mean,
)


def cluster_period_nmi(vectors_a: Tensor, vectors_b: Tensor, *, seed: int = 0) -> float | None:
    n_a, n_b = vectors_a.shape[0], vectors_b.shape[0]
    if n_a < 2 or n_b < 2:
        return None
    combined = F.normalize(torch.cat([vectors_a, vectors_b], dim=0), dim=1).numpy()
    side_labels = np.array([0] * n_a + [1] * n_b)
    cluster_labels = KMeans(n_clusters=2, random_state=seed, n_init=10).fit_predict(combined)
    return float(normalized_mutual_info_score(side_labels, cluster_labels))


def evaluate_score(rows: list[dict], truth: dict[str, dict[str, float]], score_key: str) -> dict:
    rows = [row for row in rows if row[score_key] is not None]
    rows = sorted(rows, key=lambda row: row["target"])
    graded = np.array([truth[row["target"]]["graded"] for row in rows])
    binary = np.array([truth[row["target"]]["binary"] for row in rows])
    scores = np.array([row[score_key] for row in rows])
    rho, p_value = spearmanr(graded, scores)
    return {
        "n_targets": len(rows),
        "spearman": float(rho),
        "spearman_p": float(p_value),
        "roc_auc": float(roc_auc_score(binary, scores)),
        "average_precision": float(average_precision_score(binary, scores)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Tarefa 1: APD/Delta/NMI com encoder fixo (theta0 ou theta1).")
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--truth", type=Path, default=Path("data/processed/semeval2020_task1/eng_lemma/truth.tsv"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layers", nargs="*", default=["layer_2", "mean_last_2"])
    parser.add_argument("--n-min", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    vocab = json.loads((args.experiment_dir / "vocab.json").read_text(encoding="utf-8"))
    targets = json.loads((args.experiment_dir / "targets.json").read_text(encoding="utf-8"))
    token_to_id = {token: index for index, token in enumerate(vocab)}
    target_to_index = {target: index for index, target in enumerate(targets)}
    target_set = set(targets)
    truth = read_truth(args.truth)

    cache = {
        name: torch.load(args.cache_dir / f"{name}.pt", map_location="cpu", weights_only=True)
        for name in ("theta0_d0", "theta0_d1", "theta1_d0", "theta1_d1")
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)

    metrics: dict = {}
    for theta in ("theta0", "theta1"):
        stats_d0 = cache[f"{theta}_d0"]
        stats_d1 = cache[f"{theta}_d1"]

        support_mask = build_active_support(stats_d0, stats_d1, vocab=vocab, targets=target_set, n_min=args.n_min)
        support_ids = torch.nonzero(support_mask, as_tuple=False).flatten()

        metrics[theta] = {}
        for layer in args.layers:
            centroids_d0 = contextual_centroids(stats_d0, layer)
            centroids_d1 = contextual_centroids(stats_d1, layer)
            mu_d0 = type_uniform_mean(stats_d0, layer, support=support_mask)
            mu_d1 = type_uniform_mean(stats_d1, layer, support=support_mask)

            rows = []
            for target in targets:
                target_id = token_to_id[target]
                target_index = target_to_index[target]

                vectors_d0 = occurrences_for_target(stats_d0, layer, target_index)
                vectors_d1 = occurrences_for_target(stats_d1, layer, target_index)
                apd = average_pairwise_distance(vectors_d0, vectors_d1)
                nmi = cluster_period_nmi(vectors_d0, vectors_d1, seed=args.seed)

                profile_d0 = relational_profile(centroids_d0, mu_d0, target_id, support_ids)
                profile_d1 = relational_profile(centroids_d1, mu_d1, target_id, support_ids)
                delta = displacement(profile_d0, profile_d1)

                rows.append({
                    "target": target,
                    "apd": apd,
                    "delta": delta,
                    "nmi": nmi,
                    "n_d0": int(vectors_d0.shape[0]),
                    "n_d1": int(vectors_d1.shape[0]),
                })

            metrics[theta][layer] = {
                "apd": evaluate_score(rows, truth, "apd"),
                "delta": evaluate_score(rows, truth, "delta"),
                "nmi": evaluate_score(rows, truth, "nmi"),
                "n_support": int(support_ids.numel()),
            }
            path = args.output_dir / f"rows_{theta}_{layer}.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

    (args.output_dir / "fixed_encoder_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
