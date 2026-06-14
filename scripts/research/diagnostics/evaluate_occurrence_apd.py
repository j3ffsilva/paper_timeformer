#!/usr/bin/env python3
"""Passo 0 (teste discriminante): APD + bimodalidade sobre ocorrências.

Decide se o problema da Fase 1/1.5 (sinal fraco de Delta, M_t(w) sempre
rank-1) está na agregação por centroide ou no encoder em si, usando
`occurrence_vectors`/`occurrence_targets` já presentes no cache (sem
reextração).

1. APD(w) = distância média par-a-par (1 - cos) entre TODAS as ocorrências
   de w em t0 e TODAS as ocorrências de w em t1 (baseline clássico do
   SemEval-2020, "Average Pairwise Distance").
   -> correlaciona com truth.tsv (graded/binary). Se APD funcionar bem,
   o encoder presta e o problema é a agregação em centroide único.

2. Bimodalidade: para a nuvem de ocorrências de w em t1 (e em t0), ajusta
   GMM com k=1 e k=2 componentes (cosseno -> embeddings L2-normalizados,
   GMM euclidiano sobre eles), compara BIC e reporta silhouette para k=2.
   -> se nem plane_nn/graft_nn mostrarem preferência por k=2, a
   distinção de sentidos não está separável nesta representação.

Não requer reextração: opera sobre
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
from sklearn.metrics import average_precision_score, roc_auc_score, silhouette_score
from sklearn.mixture import GaussianMixture
from torch import Tensor

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.research.common.io import read_truth  # noqa: E402
from scripts.research.common.profiles import (  # noqa: E402
    average_pairwise_distance,
    occurrences_for_target,
)


def bimodality_check(vectors: Tensor, *, seed: int = 0) -> dict:
    """GMM(k=1) vs GMM(k=2) BIC on L2-normalized vectors, plus silhouette
    for the k=2 assignment."""
    if vectors.shape[0] < 6:
        return {"n": int(vectors.shape[0]), "bic_k1": None, "bic_k2": None, "delta_bic": None, "silhouette_k2": None}
    points = F.normalize(vectors, dim=1).numpy()
    gmm1 = GaussianMixture(n_components=1, random_state=seed, covariance_type="diag").fit(points)
    gmm2 = GaussianMixture(n_components=2, random_state=seed, covariance_type="diag").fit(points)
    bic1, bic2 = gmm1.bic(points), gmm2.bic(points)
    labels = gmm2.predict(points)
    sil = None
    if len(set(labels.tolist())) == 2:
        sil = float(silhouette_score(points, labels))
    return {
        "n": int(vectors.shape[0]),
        "bic_k1": float(bic1),
        "bic_k2": float(bic2),
        "delta_bic": float(bic1 - bic2),  # positive -> k=2 preferred
        "silhouette_k2": sil,
    }


def evaluate(rows: list[dict], truth: dict[str, dict[str, float]]) -> dict:
    rows = sorted(rows, key=lambda row: row["target"])
    graded = np.array([truth[row["target"]]["graded"] for row in rows])
    binary = np.array([truth[row["target"]]["binary"] for row in rows])
    scores = np.array([row["apd"] for row in rows])
    rho, p_value = spearmanr(graded, scores)
    return {
        "n_targets": len(rows),
        "spearman": float(rho),
        "spearman_p": float(p_value),
        "roc_auc": float(roc_auc_score(binary, scores)),
        "average_precision": float(average_precision_score(binary, scores)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Passo 0: APD + bimodalidade sobre ocorrências.")
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--truth", type=Path, default=Path("data/processed/semeval2020_task1/eng_lemma/truth.tsv"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layers", nargs="*", default=["layer_2", "mean_last_2"])
    parser.add_argument("--bimodality-words", nargs="*", default=["plane_nn", "graft_nn", "chairman_nn", "tree_nn"])
    args = parser.parse_args()

    targets = json.loads((args.experiment_dir / "targets.json").read_text(encoding="utf-8"))
    target_to_index = {target: index for index, target in enumerate(targets)}
    truth = read_truth(args.truth)

    stats_t0 = torch.load(args.cache_dir / "theta0_d0.pt", map_location="cpu", weights_only=True)
    stats_t1 = torch.load(args.cache_dir / "theta1_d1.pt", map_location="cpu", weights_only=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    apd_metrics = {}
    apd_rows_all = {}
    for layer in args.layers:
        rows = []
        for target in targets:
            index = target_to_index[target]
            vectors_t0 = occurrences_for_target(stats_t0, layer, index)
            vectors_t1 = occurrences_for_target(stats_t1, layer, index)
            apd = average_pairwise_distance(vectors_t0, vectors_t1)
            rows.append({"target": target, "layer": layer, "apd": apd, "n_t0": int(vectors_t0.shape[0]), "n_t1": int(vectors_t1.shape[0])})
        apd_metrics[layer] = evaluate(rows, truth)
        apd_rows_all[layer] = rows

    bimodality = {}
    for word in args.bimodality_words:
        index = target_to_index[word]
        bimodality[word] = {}
        for layer in args.layers:
            vectors_t0 = occurrences_for_target(stats_t0, layer, index)
            vectors_t1 = occurrences_for_target(stats_t1, layer, index)
            bimodality[word][layer] = {
                "t0": bimodality_check(vectors_t0),
                "t1": bimodality_check(vectors_t1),
                "combined": bimodality_check(torch.cat([vectors_t0, vectors_t1], dim=0)),
            }

    result = {"apd_metrics": apd_metrics, "bimodality": bimodality}
    (args.output_dir / "occurrence_apd_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    for layer, rows in apd_rows_all.items():
        path = args.output_dir / f"apd_rows_{layer}.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
