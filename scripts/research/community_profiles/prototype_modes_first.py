#!/usr/bin/env python3
"""Tarefa 3 (adendo encoder-fixo) -- protótipo "modos primeiro, perfil depois".

Em palavras simples: a Fase 1.5 (NO-GO, §3) tentou achar "modos de sentido"
dentro do PERFIL RELACIONAL (a lista de cossenos contra o vocabulário) de
cada palavra. Isso não funcionou -- o perfil já chega "achatado" (um modo só
domina sempre). Aqui inverte-se a ordem: primeiro agrupa-se a NUVEM DE
OCORRÊNCIAS de cada palavra (sob o encoder fixo theta1, §7.19) em 2-5
grupos; cada grupo (modo) é então descrito pelo SEU PRÓPRIO perfil
relacional (vizinhos mais próximos no vocabulário).

Para 8 palavras auditadas (4 que mudaram, 4 controles estáveis):

1. Junta as ocorrências de d0+d1 de w sob theta1 (encoder fixo).
2. Remove as top-D componentes principais estimadas sobre a nuvem combinada
   das OUTRAS palavras-alvo (proxy de "direção compartilhada" / eixo de
   época residual, §7.15-7.16), e renormaliza.
3. Agrupa com k-means para k em {2..5}; escolhe k pelo silhouette (reporta
   a curva inteira).
4. Para cada modo, calcula o perfil relacional do CENTRÓIDE do modo contra
   o vocabulário de suporte (mesma centralização da Fase 1, variante D, mas
   com centróides agrupados de d0+d1 sob theta1) e lista os top-15
   vizinhos.
5. Reporta a composição temporal de cada modo (fração de ocorrências de d0
   vs d1) e a divergência Jensen-Shannon entre a distribuição de modos em
   d0 vs em d1 -- candidata a métrica de "mudança por reorganização de
   sentidos".

Não requer reextração: opera sobre
outputs/<experiment>/hidden_relational_profiles/cache/theta1_d{0,1}.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.distance import jensenshannon
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from torch import Tensor

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.research.common.profiles import occurrences_for_target  # noqa: E402
from timeformers.relational import build_active_support  # noqa: E402

DEFAULT_TARGETS = ["plane_nn", "graft_nn", "chairman_nn", "tree_nn"]
DEFAULT_STABLE_CONTROLS = ["ball_nn", "face_nn", "lane_nn", "multitude_nn"]


def pooled_centroids(stats_d0: dict, stats_d1: dict, layer: str) -> Tensor:
    """Occurrence-weighted centroid per type, pooling d0 and d1 under the
    same (fixed) encoder."""
    sums = stats_d0["sums"][layer].float() + stats_d1["sums"][layer].float()
    counts = (stats_d0["counts"].float() + stats_d1["counts"].float()).unsqueeze(1).clamp_min(1.0)
    return sums / counts


def fit_pc_removal(other_vectors: Tensor, *, top_d: int) -> tuple[Tensor, Tensor]:
    mean = other_vectors.mean(dim=0)
    centered = (other_vectors - mean).numpy()
    pca = PCA(n_components=top_d).fit(centered)
    return mean, torch.from_numpy(pca.components_).float()


def remove_components(vectors: Tensor, mean: Tensor, components: Tensor) -> Tensor:
    centered = vectors - mean.unsqueeze(0)
    proj = centered @ components.T
    residual = centered - proj @ components
    return F.normalize(residual, dim=1)


def choose_k_by_silhouette(points: np.ndarray, *, k_grid: list[int], seed: int) -> tuple[int, dict, dict]:
    curve = {}
    labels_by_k = {}
    for k in k_grid:
        kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(points)
        labels = kmeans.labels_
        curve[k] = float(silhouette_score(points, labels)) if len(set(labels.tolist())) > 1 else None
        labels_by_k[k] = labels
    valid = {k: v for k, v in curve.items() if v is not None}
    best_k = max(valid, key=valid.get) if valid else k_grid[0]
    return best_k, curve, labels_by_k


def mode_profile(mode_centroid: Tensor, mu: Tensor, support_centroids: Tensor, support_tokens: list[str], top_n: int) -> list[tuple[str, float]]:
    target_emb = F.normalize((mode_centroid - mu).unsqueeze(0), dim=1)
    support_emb = F.normalize(support_centroids - mu.unsqueeze(0), dim=1)
    profile = (target_emb @ support_emb.T).squeeze(0)
    top_values, top_indices = torch.topk(profile, min(top_n, profile.numel()))
    return [(support_tokens[i], round(float(v), 4)) for v, i in zip(top_values.tolist(), top_indices.tolist())]


def main() -> None:
    parser = argparse.ArgumentParser(description="Tarefa 3: protótipo modos-primeiro sob encoder fixo.")
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--targets", nargs="*", default=DEFAULT_TARGETS)
    parser.add_argument("--stable-controls", nargs="*", default=DEFAULT_STABLE_CONTROLS)
    parser.add_argument("--encoder", default="theta1", choices=["theta0", "theta1"])
    parser.add_argument("--layer", default="mean_last_2")
    parser.add_argument("--n-min", type=int, default=10)
    parser.add_argument("--top-d-pcs", type=int, default=2)
    parser.add_argument("--k-grid", nargs="*", type=int, default=[2, 3, 4, 5])
    parser.add_argument("--top-n-neighbors", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    vocab = json.loads((args.experiment_dir / "vocab.json").read_text(encoding="utf-8"))
    token_to_id = {token: index for index, token in enumerate(vocab)}
    all_words = list(dict.fromkeys(args.targets + args.stable_controls))
    excluded = set(all_words)

    stats_d0 = torch.load(args.cache_dir / f"{args.encoder}_d0.pt", map_location="cpu", weights_only=True)
    stats_d1 = torch.load(args.cache_dir / f"{args.encoder}_d1.pt", map_location="cpu", weights_only=True)
    targets_json = json.loads((args.experiment_dir / "targets.json").read_text(encoding="utf-8"))
    target_to_index = {target: index for index, target in enumerate(targets_json)}

    support_mask = build_active_support(stats_d0, stats_d1, vocab=vocab, targets=excluded, n_min=args.n_min)
    support_ids = torch.nonzero(support_mask, as_tuple=False).flatten()
    support_tokens = [vocab[i] for i in support_ids.tolist()]

    centroids_pooled = pooled_centroids(stats_d0, stats_d1, args.layer)

    # PCs estimated on the pooled occurrence cloud of the OTHER target words
    # (under the fixed encoder), to project out a shared "epoch residual"
    # direction (§7.15-7.16) without using w's own occurrences.
    other_vectors_by_word: dict[str, Tensor] = {}
    for word in all_words:
        index = target_to_index[word]
        v_d0 = occurrences_for_target(stats_d0, args.layer, index)
        v_d1 = occurrences_for_target(stats_d1, args.layer, index)
        other_vectors_by_word[word] = torch.cat([v_d0, v_d1], dim=0)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: dict = {"encoder": args.encoder, "layer": args.layer, "top_d_pcs": args.top_d_pcs, "words": {}}

    for word in all_words:
        index = target_to_index[word]
        v_d0 = occurrences_for_target(stats_d0, args.layer, index)
        v_d1 = occurrences_for_target(stats_d1, args.layer, index)

        other_pooled = torch.cat([other_vectors_by_word[w] for w in all_words if w != word], dim=0)
        pc_mean, pc_components = fit_pc_removal(other_pooled, top_d=args.top_d_pcs)

        combined = torch.cat([v_d0, v_d1], dim=0)
        side = np.array([0] * v_d0.shape[0] + [1] * v_d1.shape[0])
        points = remove_components(combined, pc_mean, pc_components).numpy()

        best_k, curve, labels_by_k = choose_k_by_silhouette(points, k_grid=args.k_grid, seed=args.seed)
        labels = labels_by_k[best_k]

        # Mode profiles: centroid of each mode in the ORIGINAL (pre-removal)
        # embedding space, related to the support via the same PC-removal
        # + mu_t (type-uniform over support, pooled d0+d1).
        support_centroids = remove_components(centroids_pooled[support_ids], pc_mean, pc_components)
        mu = support_centroids.mean(dim=0)

        modes = []
        for mode_id in range(best_k):
            mask = labels == mode_id
            mode_centroid_raw = combined[mask].mean(dim=0)
            mode_centroid = remove_components(mode_centroid_raw.unsqueeze(0), pc_mean, pc_components).squeeze(0)
            frac_d0 = float((side[mask] == 0).mean())
            frac_d1 = float((side[mask] == 1).mean())
            neighbors = mode_profile(mode_centroid, mu, support_centroids, support_tokens, args.top_n_neighbors)
            modes.append({
                "mode": mode_id,
                "n": int(mask.sum()),
                "frac_d0": frac_d0,
                "frac_d1": frac_d1,
                "top_neighbors": neighbors,
            })

        # JSD between mode-distribution of d0 occurrences vs d1 occurrences.
        dist_d0 = np.array([(labels[side == 0] == j).mean() for j in range(best_k)])
        dist_d1 = np.array([(labels[side == 1] == j).mean() for j in range(best_k)])
        jsd = float(jensenshannon(dist_d0, dist_d1, base=2))

        results["words"][word] = {
            "is_target": word in args.targets,
            "n_d0": int(v_d0.shape[0]),
            "n_d1": int(v_d1.shape[0]),
            "silhouette_curve": curve,
            "k": best_k,
            "modes": modes,
            "dist_d0": dist_d0.tolist(),
            "dist_d1": dist_d1.tolist(),
            "jsd_d0_d1": jsd,
        }
        print(f"{word}: k={best_k} jsd={jsd:.4f} silhouette={curve}")

    (args.output_dir / "modes_first_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
