#!/usr/bin/env python3
"""Adendo ao Passo 0: APD_ratio + associacao cluster x periodo.

Duas extensoes baratas sobre o mesmo cache de ocorrencias
(occurrence_vectors/occurrence_targets), respondendo a critica de
2026-06-12 ao adendo anterior:

1. APD_ratio(w) = APD_inter / mean(APD_intra_t0, APD_intra_t1)
   Normaliza palavras naturalmente difusas (ex.: tree_nn, com milhares de
   ocorrencias e alta dispersao mesmo dentro de um periodo) para que a
   distancia inter-periodo seja comparada ao "ruido" intra-periodo da
   propria palavra, em vez de a uma escala absoluta.

2. Associacao cluster x periodo: particiona a nuvem combinada (t0+t1) de
   ocorrencias de w em 2 clusters (KMeans sobre vetores L2-normalizados) e
   mede a associacao entre o rotulo de cluster e o rotulo de periodo via
   normalized mutual information (NMI) e adjusted mutual information (AMI).
   Isso e robusto ao artefato do bimodality_check anterior (delta_bic
   positivo para TODAS as palavras, inclusive as estaveis): uma estrutura
   de densidade generica nao tem motivo para se alinhar com o periodo, mas
   uma mudanca de sentido sim.

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
    adjusted_mutual_info_score,
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


def average_pairwise_distance_within(vectors: Tensor) -> float | None:
    """Mean 1 - cos over all unordered pairs i < j within a single cloud."""
    n = vectors.shape[0]
    if n < 2:
        return None
    normalized = F.normalize(vectors, dim=1)
    cosines = normalized @ normalized.T
    mask = ~torch.eye(n, dtype=torch.bool)
    return float((1.0 - cosines[mask]).mean())


def cluster_period_association(vectors_t0: Tensor, vectors_t1: Tensor, *, seed: int = 0) -> dict:
    n0, n1 = vectors_t0.shape[0], vectors_t1.shape[0]
    if n0 < 2 or n1 < 2:
        return {"n0": n0, "n1": n1, "nmi": None, "ami": None}
    combined = F.normalize(torch.cat([vectors_t0, vectors_t1], dim=0), dim=1).numpy()
    period_labels = np.array([0] * n0 + [1] * n1)
    cluster_labels = KMeans(n_clusters=2, random_state=seed, n_init=10).fit_predict(combined)
    return {
        "n0": n0,
        "n1": n1,
        "nmi": float(normalized_mutual_info_score(period_labels, cluster_labels)),
        "ami": float(adjusted_mutual_info_score(period_labels, cluster_labels)),
    }


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
    parser = argparse.ArgumentParser(description="APD_ratio + cluster x periodo sobre ocorrencias.")
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--truth", type=Path, default=Path("data/processed/semeval2020_task1/eng_lemma/truth.tsv"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layers", nargs="*", default=["layer_2", "mean_last_2"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    targets = json.loads((args.experiment_dir / "targets.json").read_text(encoding="utf-8"))
    target_to_index = {target: index for index, target in enumerate(targets)}
    truth = read_truth(args.truth)

    stats_t0 = torch.load(args.cache_dir / "theta0_d0.pt", map_location="cpu", weights_only=True)
    stats_t1 = torch.load(args.cache_dir / "theta1_d1.pt", map_location="cpu", weights_only=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    metrics = {}
    rows_all = {}
    for layer in args.layers:
        rows = []
        for target in targets:
            index = target_to_index[target]
            vectors_t0 = occurrences_for_target(stats_t0, layer, index)
            vectors_t1 = occurrences_for_target(stats_t1, layer, index)

            apd_inter = average_pairwise_distance(vectors_t0, vectors_t1)
            apd_intra_t0 = average_pairwise_distance_within(vectors_t0)
            apd_intra_t1 = average_pairwise_distance_within(vectors_t1)

            apd_ratio = None
            if apd_intra_t0 is not None and apd_intra_t1 is not None:
                denom = (apd_intra_t0 + apd_intra_t1) / 2.0
                if denom > 0:
                    apd_ratio = apd_inter / denom

            assoc = cluster_period_association(vectors_t0, vectors_t1, seed=args.seed)

            rows.append({
                "target": target,
                "layer": layer,
                "apd_inter": apd_inter,
                "apd_intra_t0": apd_intra_t0,
                "apd_intra_t1": apd_intra_t1,
                "apd_ratio": apd_ratio,
                "nmi": assoc["nmi"],
                "ami": assoc["ami"],
                "n_t0": int(vectors_t0.shape[0]),
                "n_t1": int(vectors_t1.shape[0]),
            })

        metrics[layer] = {
            "apd_ratio": evaluate_score(rows, truth, "apd_ratio"),
            "nmi": evaluate_score(rows, truth, "nmi"),
            "ami": evaluate_score(rows, truth, "ami"),
        }
        rows_all[layer] = rows

    result = {"metrics": metrics}
    (args.output_dir / "cluster_period_apd_ratio_results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    for layer, rows in rows_all.items():
        path = args.output_dir / f"rows_{layer}.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
