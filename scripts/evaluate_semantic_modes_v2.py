#!/usr/bin/env python3
"""Fase 1.5 -- go/no-go espectral barato.

Para um conjunto pequeno de palavras auditadas (alvos com mudança esperada e
controles estáveis), aplica o critério de gap (§8) + SVD da matriz de coesão
(§7.5) sobre o perfil relacional v2 (centralização D: média não-ponderada dos
centróides por tipo sobre V_ativo) e mede a estabilidade de k e dos modos sob:

  - variação de gamma (0.2 / 0.3 / 0.4)
  - variação de n_min do suporte V_ativo (10 / 20 / 50)
  - bootstrap de V_w (subamostragem 80%, N repetições) -> overlap dos top
    tokens do modo 1 (e modo 2, se k>=2) entre repetições

Critério go: para plane_nn/graft_nn, k>=2 com modos interpretáveis e
relativamente estáveis sob bootstrap; para chairman_nn/tree_nn (campo
estável), k=1 ou modos espúrios instáveis.

Não requer reextração: opera sobre
outputs/<experiment>/hidden_relational_profiles/cache/theta{0,1}_d{0,1}.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from timeformers.real_corpus import SPECIAL_TOKENS  # noqa: E402
from timeformers.semantic_modes import (  # noqa: E402
    cohesion_svd,
    filter_support_topn,
    select_num_modes,
    top_tokens_per_mode,
)

DEFAULT_TARGETS = ["plane_nn", "graft_nn", "chairman_nn", "tree_nn"]
DEFAULT_STABLE_CONTROLS = ["ball_nn", "face_nn", "lane_nn", "multitude_nn"]


def contextual_centroids(stats: dict, layer: str) -> Tensor:
    counts = stats["counts"].float().unsqueeze(1).clamp_min(1.0)
    return stats["sums"][layer].float() / counts


def build_active_support(
    stats_t0: dict,
    stats_t1: dict,
    *,
    vocab: list[str],
    excluded: set[str],
    n_min: int,
) -> Tensor:
    mask = (stats_t0["counts"] >= n_min) & (stats_t1["counts"] >= n_min)
    for index, token in enumerate(vocab):
        if token in SPECIAL_TOKENS or token in excluded:
            mask[index] = False
    return mask


def centralized_embeddings(centroids: Tensor, mu: Tensor) -> Tensor:
    return F.normalize(centroids - mu.unsqueeze(0), dim=1)


def decompose(
    word_embedding: Tensor,
    support_embeddings: Tensor,
    support_tokens: list[str],
    *,
    gamma: float,
    top_n: int,
) -> dict:
    profile = support_embeddings @ word_embedding
    vw_indices, tau = filter_support_topn(profile, gamma, top_n)
    if tau is None:
        return {"tau": None, "k": None, "n_vw": 0, "modes": []}
    vw_tokens = [support_tokens[i] for i in vw_indices.tolist()]
    profile_vw = profile[vw_indices]
    embeddings_vw = support_embeddings[vw_indices]
    eigenvalues, eigenvectors = cohesion_svd(profile_vw, embeddings_vw)
    k = select_num_modes(eigenvalues, gamma)
    modes = top_tokens_per_mode(eigenvectors, vw_tokens, k or 1, top_n=10)
    return {
        "tau": tau,
        "k": k,
        "n_vw": len(vw_tokens),
        "eigenvalues": eigenvalues[: min(6, eigenvalues.numel())].tolist(),
        "modes": modes,
        "vw_indices": vw_indices,
        "vw_tokens": vw_tokens,
        "profile_vw": profile_vw,
        "embeddings_vw": embeddings_vw,
    }


def bootstrap_stability(
    profile_vw: Tensor,
    embeddings_vw: Tensor,
    vw_tokens: list[str],
    *,
    k: int,
    gamma: float,
    n_repeats: int,
    fraction: float,
    seed: int,
) -> dict:
    """Subsample V_w (without replacement) and recompute the top-k modes;
    report mean Jaccard overlap of each mode's top-10 tokens against the
    full-data mode, and how often k itself is recovered."""
    if k is None or k < 1 or len(vw_tokens) < 4:
        return {"k_recovery_rate": None, "mode_overlap": []}

    full_modes = top_tokens_per_mode(
        cohesion_svd(profile_vw, embeddings_vw)[1], vw_tokens, k, top_n=10
    )
    full_sets = [{token for token, _ in mode} for mode in full_modes]

    generator = torch.Generator().manual_seed(seed)
    n = len(vw_tokens)
    sample_size = max(int(n * fraction), k + 2)
    k_matches = 0
    overlaps = [[] for _ in range(k)]
    for _ in range(n_repeats):
        perm = torch.randperm(n, generator=generator)[:sample_size]
        sub_profile = profile_vw[perm]
        sub_embeddings = embeddings_vw[perm]
        sub_tokens = [vw_tokens[i] for i in perm.tolist()]
        eigenvalues, eigenvectors = cohesion_svd(sub_profile, sub_embeddings)
        sub_k = select_num_modes(eigenvalues, gamma)
        if sub_k == k:
            k_matches += 1
        sub_modes = top_tokens_per_mode(eigenvectors, sub_tokens, min(sub_k or 1, k), top_n=10)
        for i in range(min(len(sub_modes), k)):
            sub_set = {token for token, _ in sub_modes[i]}
            union = full_sets[i] | sub_set
            inter = full_sets[i] & sub_set
            overlaps[i].append(len(inter) / len(union) if union else 0.0)

    return {
        "k_recovery_rate": k_matches / n_repeats,
        "mode_overlap": [sum(o) / len(o) if o else None for o in overlaps],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fase 1.5: go/no-go espectral.")
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--targets", nargs="*", default=DEFAULT_TARGETS)
    parser.add_argument("--stable-controls", nargs="*", default=DEFAULT_STABLE_CONTROLS)
    parser.add_argument("--layer", default="mean_last_2")
    parser.add_argument("--n-min", type=int, default=10)
    parser.add_argument("--n-min-grid", nargs="*", type=int, default=[10, 20, 50])
    parser.add_argument("--gamma", type=float, default=0.3)
    parser.add_argument("--gamma-grid", nargs="*", type=float, default=[0.2, 0.3, 0.4])
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--top-n-grid", nargs="*", type=int, default=[50, 100, 200])
    parser.add_argument("--bootstrap-repeats", type=int, default=30)
    parser.add_argument("--bootstrap-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    vocab = json.loads((args.experiment_dir / "vocab.json").read_text(encoding="utf-8"))
    token_to_id = {token: index for index, token in enumerate(vocab)}
    all_words = list(dict.fromkeys(args.targets + args.stable_controls))
    excluded = set(all_words)

    stats_t0 = torch.load(args.cache_dir / "theta0_d0.pt", map_location="cpu", weights_only=True)
    stats_t1 = torch.load(args.cache_dir / "theta1_d1.pt", map_location="cpu", weights_only=True)

    results: dict = {
        "layer": args.layer,
        "gamma": args.gamma,
        "n_min": args.n_min,
        "top_n": args.top_n,
        "words": {},
    }

    # Primary support / mu (centralization D), at the default n_min.
    support_mask = build_active_support(stats_t0, stats_t1, vocab=vocab, excluded=excluded, n_min=args.n_min)
    support_ids = torch.nonzero(support_mask, as_tuple=False).flatten()
    support_tokens = [vocab[i] for i in support_ids.tolist()]

    centroids_t0 = contextual_centroids(stats_t0, args.layer)
    centroids_t1 = contextual_centroids(stats_t1, args.layer)
    mu_t0 = centroids_t0[support_ids].mean(dim=0)
    mu_t1 = centroids_t1[support_ids].mean(dim=0)
    support_emb_t0 = centralized_embeddings(centroids_t0[support_ids], mu_t0)
    support_emb_t1 = centralized_embeddings(centroids_t1[support_ids], mu_t1)

    for word in all_words:
        word_id = token_to_id[word]
        word_emb_t0 = F.normalize(centroids_t0[word_id : word_id + 1] - mu_t0, dim=1).squeeze(0)
        word_emb_t1 = F.normalize(centroids_t1[word_id : word_id + 1] - mu_t1, dim=1).squeeze(0)

        word_result: dict = {"is_target": word in args.targets, "periods": {}}
        for period_name, word_emb, support_emb in (
            ("t0", word_emb_t0, support_emb_t0),
            ("t1", word_emb_t1, support_emb_t1),
        ):
            decomposition = decompose(
                word_emb, support_emb, support_tokens, gamma=args.gamma, top_n=args.top_n
            )
            entry = {
                "tau": decomposition["tau"],
                "k": decomposition["k"],
                "n_vw": decomposition["n_vw"],
                "eigenvalues": decomposition.get("eigenvalues"),
                "modes": [
                    [(token, round(weight, 3)) for token, weight in mode[:6]]
                    for mode in decomposition["modes"]
                ],
            }
            if decomposition["tau"] is not None:
                entry["bootstrap"] = bootstrap_stability(
                    decomposition["profile_vw"],
                    decomposition["embeddings_vw"],
                    decomposition["vw_tokens"],
                    k=decomposition["k"] or 1,
                    gamma=args.gamma,
                    n_repeats=args.bootstrap_repeats,
                    fraction=args.bootstrap_fraction,
                    seed=args.seed,
                )
                # gamma sensitivity (k only, reusing the same V_w)
                entry["gamma_sensitivity"] = {}
                for gamma in args.gamma_grid:
                    eigenvalues, _ = cohesion_svd(
                        decomposition["profile_vw"], decomposition["embeddings_vw"]
                    )
                    entry["gamma_sensitivity"][str(gamma)] = select_num_modes(eigenvalues, gamma)
            # top_n sensitivity (re-filter and re-decompose with each candidate pool size)
            entry["top_n_sensitivity"] = {}
            for top_n in args.top_n_grid:
                alt = decompose(word_emb, support_emb, support_tokens, gamma=args.gamma, top_n=top_n)
                entry["top_n_sensitivity"][str(top_n)] = {"tau": alt["tau"], "n_vw": alt["n_vw"], "k": alt["k"]}
            word_result["periods"][period_name] = entry

        # n_min sensitivity (recompute support + decomposition for t1 only, cheaper)
        word_result["n_min_sensitivity"] = {}
        for n_min in args.n_min_grid:
            alt_mask = build_active_support(stats_t0, stats_t1, vocab=vocab, excluded=excluded, n_min=n_min)
            alt_ids = torch.nonzero(alt_mask, as_tuple=False).flatten()
            alt_tokens = [vocab[i] for i in alt_ids.tolist()]
            alt_mu_t1 = centroids_t1[alt_ids].mean(dim=0)
            alt_emb_t1 = centralized_embeddings(centroids_t1[alt_ids], alt_mu_t1)
            alt_word_emb_t1 = F.normalize(centroids_t1[word_id : word_id + 1] - alt_mu_t1, dim=1).squeeze(0)
            alt_decomposition = decompose(
                alt_word_emb_t1, alt_emb_t1, alt_tokens, gamma=args.gamma, top_n=args.top_n
            )
            word_result["n_min_sensitivity"][str(n_min)] = {
                "n_vw": alt_decomposition["n_vw"],
                "k": alt_decomposition["k"],
            }

        results["words"][word] = word_result

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "modes.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
