#!/usr/bin/env python3
"""Fase 0A + Fase 1 (ablação A/B/C de centralização) + piso de drift.

Reaproveita os caches já extraídos (sums/counts por token, todos os tokens
do vocabulário) para comparar três formas de centralização do perfil
relacional P_t(w)[v] = cos(ê_t(w), ê_t(v)):

  A: centra na média de um conjunto pequeno de referências frequentes
     (reproduz a abordagem v1 de report_temporal_relational_neighborhoods.py)
  B: centra em mu_t global, ponderado por ocorrência, sobre TODOS os tokens
     observados no período (sum(sums) / sum(counts) sobre o vocabulário)
  C: como B, mas mu_t calculado apenas sobre o suporte V_ativo (tokens com
     count_t >= n_min em ambos os períodos) -- exclui ruído de tokens raros

Em todos os casos, P_t(w)[v] é calculado sobre o MESMO suporte V_ativo, de
forma que apenas a centralização varia entre A/B/C.

Não requer reextração: opera sobre
outputs/<experiment>/hidden_relational_profiles/cache/theta{0,1}_d{0,1}.pt
(usar --reuse-cache).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from timeformers.relational import (  # noqa: E402
    build_active_support,
    contextual_centroids,
    displacement,
    occurrence_weighted_mean,
    relational_profile,
    type_uniform_mean,
)


def read_truth(path: Path) -> dict[str, dict[str, float]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {
            row["target"]: {"binary": float(row["binary"]), "graded": float(row["graded"])}
            for row in csv.DictReader(handle, delimiter="\t")
        }


def evaluate(rows: list[dict], truth: dict[str, dict[str, float]]) -> list[dict]:
    metrics = []
    for (variant, layer), group in group_by(rows, keys=("variant", "layer")).items():
        group = sorted(group, key=lambda row: row["target"])
        graded = np.array([truth[row["target"]]["graded"] for row in group])
        binary = np.array([truth[row["target"]]["binary"] for row in group])
        scores = np.array([row["delta"] for row in group])
        rho, p_value = spearmanr(graded, scores)
        stable_scores = scores[binary == 0]
        changed_scores = scores[binary == 1]
        metrics.append(
            {
                "variant": variant,
                "layer": layer,
                "n_targets": len(group),
                "spearman": float(rho),
                "spearman_p": float(p_value),
                "roc_auc": float(roc_auc_score(binary, scores)),
                "average_precision": float(average_precision_score(binary, scores)),
                "stable_delta_mean": float(stable_scores.mean()),
                "stable_delta_p95": float(np.percentile(stable_scores, 95)),
                "changed_delta_mean": float(changed_scores.mean()),
                "changed_above_stable_p95": float(
                    (changed_scores > np.percentile(stable_scores, 95)).mean()
                ),
            }
        )
    return sorted(metrics, key=lambda row: row["spearman"], reverse=True)


def group_by(rows: list[dict], *, keys: tuple[str, ...]) -> dict[tuple, list[dict]]:
    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[key] for key in keys), []).append(row)
    return grouped


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fase 0A/1: ablação de centralização (A/B/C) + piso de drift, sobre caches existentes."
    )
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True, help="dir with theta{0,1}_d{0,1}.pt")
    parser.add_argument("--truth", type=Path, default=Path("data/processed/semeval2020_task1/eng_lemma/truth.tsv"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-min-b", type=int, default=10, help="min count for V_active (variant B reuses this support)")
    parser.add_argument("--n-min-reference", type=int, default=100, help="min count for variant A's reference set")
    parser.add_argument("--n-min-c", type=int, default=50, help="stricter min count for variant C support and mu_t")
    parser.add_argument("--max-references", type=int, default=3216, help="size of variant A's reference set")
    parser.add_argument("--layers", nargs="*", default=["layer_2", "mean_last_2"])
    args = parser.parse_args()

    vocab = json.loads((args.experiment_dir / "vocab.json").read_text(encoding="utf-8"))
    targets = json.loads((args.experiment_dir / "targets.json").read_text(encoding="utf-8"))
    token_to_id = {token: index for index, token in enumerate(vocab)}
    target_ids = [token_to_id[target] for target in targets]
    target_set = set(targets)
    truth = read_truth(args.truth)

    stats_t0 = torch.load(args.cache_dir / "theta0_d0.pt", map_location="cpu", weights_only=True)
    stats_t1 = torch.load(args.cache_dir / "theta1_d1.pt", map_location="cpu", weights_only=True)

    # Supports -------------------------------------------------------------
    support_b_mask = build_active_support(stats_t0, stats_t1, vocab=vocab, targets=target_set, n_min=args.n_min_b)
    support_c_mask = build_active_support(stats_t0, stats_t1, vocab=vocab, targets=target_set, n_min=args.n_min_c)
    support_b_ids = torch.nonzero(support_b_mask, as_tuple=False).flatten()
    support_c_ids = torch.nonzero(support_c_mask, as_tuple=False).flatten()

    # Variant A reference set: top tokens by min(count_t0, count_t1), capped.
    counts_min = torch.minimum(stats_t0["counts"], stats_t1["counts"]).float()
    ref_eligible = build_active_support(stats_t0, stats_t1, vocab=vocab, targets=target_set, n_min=args.n_min_reference)
    ref_ids_all = torch.nonzero(ref_eligible, as_tuple=False).flatten()
    ref_order = torch.argsort(counts_min[ref_ids_all], descending=True)
    reference_ids = ref_ids_all[ref_order][: args.max_references]

    # Variant A's profile is evaluated on the same support as B for a fair
    # delta comparison (only centering changes), but centered on the small
    # reference set's mean (v1-style).
    support_a_ids = support_b_ids

    print(
        json.dumps(
            {
                "n_vocab": len(vocab),
                "n_support_b (n_min={})".format(args.n_min_b): int(support_b_ids.numel()),
                "n_support_c (n_min={})".format(args.n_min_c): int(support_c_ids.numel()),
                "n_reference_a (n_min={}, capped at {})".format(
                    args.n_min_reference, args.max_references
                ): int(reference_ids.numel()),
            },
            indent=2,
        )
    )

    rows = []
    for layer in args.layers:
        centroids_t0 = contextual_centroids(stats_t0, layer)
        centroids_t1 = contextual_centroids(stats_t1, layer)

        # mu for each variant
        mu_a_t0 = centroids_t0[reference_ids].mean(dim=0)
        mu_a_t1 = centroids_t1[reference_ids].mean(dim=0)
        mu_b_t0 = occurrence_weighted_mean(stats_t0, layer)
        mu_b_t1 = occurrence_weighted_mean(stats_t1, layer)
        mu_c_t0 = occurrence_weighted_mean(stats_t0, layer, support=support_c_mask)
        mu_c_t1 = occurrence_weighted_mean(stats_t1, layer, support=support_c_mask)
        mu_d_t0 = type_uniform_mean(stats_t0, layer, support=support_b_mask)
        mu_d_t1 = type_uniform_mean(stats_t1, layer, support=support_b_mask)

        variants = {
            "A_reference_mean": (mu_a_t0, mu_a_t1, support_a_ids),
            "B_global_mu": (mu_b_t0, mu_b_t1, support_b_ids),
            "C_global_mu_active_support": (mu_c_t0, mu_c_t1, support_c_ids),
            "D_type_uniform_mu": (mu_d_t0, mu_d_t1, support_b_ids),
        }

        for variant, (mu_t0, mu_t1, support_ids) in variants.items():
            for target_index, (target, target_id) in enumerate(zip(targets, target_ids)):
                profile_t0 = relational_profile(centroids_t0, mu_t0, target_id, support_ids)
                profile_t1 = relational_profile(centroids_t1, mu_t1, target_id, support_ids)
                rows.append(
                    {
                        "variant": variant,
                        "layer": layer,
                        "target": target,
                        "delta": displacement(profile_t0, profile_t1),
                        "n_support": int(support_ids.numel()),
                        "count_d0": int(stats_t0["counts"][target_id]),
                        "count_d1": int(stats_t1["counts"][target_id]),
                    }
                )

    metrics = evaluate(rows, truth)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.output_dir / "deltas.csv")
    write_csv(metrics, args.output_dir / "metrics.csv")
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
