#!/usr/bin/env python3
"""Adendo 3 -- de-confunde a origem do "eixo de epoca" achado em 7.11.

Usa a grade 2x2 ja presente no cache (theta{0,1}_d{0,1}.pt: cada checkpoint
aplicado a CADA recorte de corpus) para separar duas fontes possiveis do
NMI(cluster, periodo) ~= 1 observado em mean_last_2 para quase toda palavra:

  A) drift de checkpoint (theta0 -> theta1 via MLM continuado)
  B) separabilidade do corpus em si (d0 = 1810-1860, d1 = 1960-2010; genero,
     ortografia, topico mudam independente do encoder)

Teste 1 (2x2 NMI por palavra, k-means k=2 na nuvem combinada):
  - diagonal (original, Passo 0/7.10): theta0_d0 vs theta1_d1 -> mistura A+B
  - encoder fixo, dados variam:        theta1_d0 vs theta1_d1 -> isola B
  - dados fixos, encoder varia:        theta0_d0 vs theta1_d0 -> isola A
                                        theta0_d1 vs theta1_d1 -> isola A (d1)

Teste 2 (centralizacao por periodo no nivel de ocorrencia):
  Para a combinacao diagonal, subtrai de cada ocorrencia a media global de
  TODAS as ocorrencias (todas as palavras) do seu "lado" (theta0_d0 vs
  theta1_d1), renormaliza, e re-roda NMI/APD. Se o NMI das palavras estaveis
  colapsar, o eixo era majoritariamente um shift aditivo (facil de remover);
  se sobreviver, e mais provavel ser rotacao/reescala anisotropica (7.11).

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


def cluster_association(vectors_a: Tensor, vectors_b: Tensor, *, seed: int = 0) -> dict:
    n_a, n_b = vectors_a.shape[0], vectors_b.shape[0]
    if n_a < 2 or n_b < 2:
        return {"n_a": n_a, "n_b": n_b, "nmi": None, "ami": None}
    combined = F.normalize(torch.cat([vectors_a, vectors_b], dim=0), dim=1).numpy()
    side_labels = np.array([0] * n_a + [1] * n_b)
    cluster_labels = KMeans(n_clusters=2, random_state=seed, n_init=10).fit_predict(combined)
    return {
        "n_a": n_a,
        "n_b": n_b,
        "nmi": float(normalized_mutual_info_score(side_labels, cluster_labels)),
        "ami": float(adjusted_mutual_info_score(side_labels, cluster_labels)),
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


def per_word_nmi(stats_a: dict, stats_b: dict, *, targets: list[str], target_to_index: dict, layer: str, seed: int) -> list[dict]:
    rows = []
    for target in targets:
        index = target_to_index[target]
        vectors_a = occurrences_for_target(stats_a, layer, index)
        vectors_b = occurrences_for_target(stats_b, layer, index)
        assoc = cluster_association(vectors_a, vectors_b, seed=seed)
        rows.append({"target": target, "layer": layer, **assoc})
    return rows


def recenter_by_global_mean(stats: dict, layer: str) -> Tensor:
    """Subtract the mean over ALL occurrences (all words) of this cache, then L2-normalize."""
    vectors = stats["occurrence_vectors"][layer].float()
    mean = vectors.mean(dim=0, keepdim=True)
    return F.normalize(vectors - mean, dim=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Adendo 3: de-confunde o eixo de epoca (checkpoint vs corpus).")
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--truth", type=Path, default=Path("data/processed/semeval2020_task1/eng_lemma/truth.tsv"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layers", nargs="*", default=["layer_2", "mean_last_2"])
    parser.add_argument("--anchors", nargs="*", default=["plane_nn", "graft_nn", "chairman_nn", "tree_nn"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    targets = json.loads((args.experiment_dir / "targets.json").read_text(encoding="utf-8"))
    target_to_index = {target: index for index, target in enumerate(targets)}
    truth = read_truth(args.truth)

    cache = {
        name: torch.load(args.cache_dir / f"{name}.pt", map_location="cpu", weights_only=True)
        for name in ("theta0_d0", "theta0_d1", "theta1_d0", "theta1_d1")
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Teste 1: grade 2x2 de NMI por palavra ----
    combos = {
        "diagonal_theta0d0_vs_theta1d1": ("theta0_d0", "theta1_d1"),  # mistura A+B (original)
        "encoder_fixo_theta1_d0_vs_d1": ("theta1_d0", "theta1_d1"),   # isola B (dados)
        "dados_fixos_d0_theta0_vs_theta1": ("theta0_d0", "theta1_d0"),  # isola A (checkpoint), d0
        "dados_fixos_d1_theta0_vs_theta1": ("theta0_d1", "theta1_d1"),  # isola A (checkpoint), d1
    }

    test1: dict = {}
    for combo_name, (name_a, name_b) in combos.items():
        test1[combo_name] = {}
        for layer in args.layers:
            rows = per_word_nmi(cache[name_a], cache[name_b], targets=targets, target_to_index=target_to_index, layer=layer, seed=args.seed)
            anchors = {row["target"]: row for row in rows if row["target"] in args.anchors}
            nmis = [row["nmi"] for row in rows if row["nmi"] is not None]
            test1[combo_name][layer] = {
                "mean_nmi": float(np.mean(nmis)) if nmis else None,
                "median_nmi": float(np.median(nmis)) if nmis else None,
                "anchors": {target: row["nmi"] for target, row in anchors.items()},
                "discrimination_vs_truth": evaluate_score(
                    [{"target": row["target"], "nmi": row["nmi"]} for row in rows], truth, "nmi"
                ),
            }
            path = args.output_dir / f"test1_{combo_name}_{layer}.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

    # ---- Teste 2: centralizacao por "lado" no nivel de ocorrencia (diagonal) ----
    test2: dict = {}
    for layer in args.layers:
        recentered_a = recenter_by_global_mean(cache["theta0_d0"], layer)
        recentered_b = recenter_by_global_mean(cache["theta1_d1"], layer)
        mean_a = cache["theta0_d0"]["occurrence_vectors"][layer].float().mean(dim=0)
        mean_b = cache["theta1_d1"]["occurrence_vectors"][layer].float().mean(dim=0)
        shift_cosine = float(F.cosine_similarity(mean_a.unsqueeze(0), mean_b.unsqueeze(0)).item())
        shift_norm_ratio = float((mean_a - mean_b).norm() / mean_a.norm())

        rows = []
        for target in targets:
            index = target_to_index[target]
            mask_a = cache["theta0_d0"]["occurrence_targets"] == index
            mask_b = cache["theta1_d1"]["occurrence_targets"] == index
            vectors_a = recentered_a[mask_a]
            vectors_b = recentered_b[mask_b]
            apd = average_pairwise_distance(vectors_a, vectors_b)
            assoc = cluster_association(vectors_a, vectors_b, seed=args.seed)
            rows.append({"target": target, "layer": layer, "apd_recentered": apd, **assoc})

        anchors = {row["target"]: row for row in rows if row["target"] in args.anchors}
        nmis = [row["nmi"] for row in rows if row["nmi"] is not None]
        test2[layer] = {
            "shift_cosine_mean_a_vs_mean_b": shift_cosine,
            "shift_norm_ratio": shift_norm_ratio,
            "mean_nmi_after_recentering": float(np.mean(nmis)) if nmis else None,
            "median_nmi_after_recentering": float(np.median(nmis)) if nmis else None,
            "anchors_nmi": {target: row["nmi"] for target, row in anchors.items()},
            "apd_discrimination_vs_truth": evaluate_score(rows, truth, "apd_recentered"),
        }
        path = args.output_dir / f"test2_recentered_{layer}.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    result = {"test1_grade_2x2": test1, "test2_recentering": test2}
    (args.output_dir / "period_axis_diagnosis.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
