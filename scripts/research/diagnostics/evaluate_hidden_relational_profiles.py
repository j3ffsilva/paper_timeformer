#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import Tensor
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.research.common.encoders import (  # noqa: E402
    ContextChunkDataset,
    build_model,
    encode_layers,
)
from scripts.research.common.io import read_truth, write_csv  # noqa: E402
from scripts.research.common.profiles import (  # noqa: E402
    average_pairwise_cosine_distance,
    occurrence_profiles,
    relational_profiles,
)
from timeformers.real_corpus import SPECIAL_TOKENS, read_period_corpora  # noqa: E402
from timeformers.real_models import RealStaticMLM  # noqa: E402
from timeformers.relational import contextual_centroids  # noqa: E402


@torch.no_grad()
def extract_context_statistics(
    model: RealStaticMLM,
    corpus,
    token_to_id: dict[str, int],
    target_ids: list[int],
    *,
    seq_len: int,
    batch_size: int,
    device: str,
) -> dict:
    model.eval().to(device)
    dataset = ContextChunkDataset(corpus, token_to_id, seq_len=seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    vocab_size = len(token_to_id)
    d_model = model.d_model
    layer_names = [f"layer_{index + 1}" for index in range(len(model.encoder.layers))]
    if len(model.encoder.layers) >= 2:
        layer_names.append("mean_last_2")
    sums = {
        name: torch.zeros(vocab_size, d_model, dtype=torch.float32, device=device)
        for name in layer_names
    }
    counts = torch.zeros(vocab_size, dtype=torch.long, device=device)
    target_lookup = torch.full((vocab_size,), -1, dtype=torch.long, device=device)
    target_lookup[torch.tensor(target_ids, device=device)] = torch.arange(len(target_ids), device=device)
    occurrence_vectors = {name: [] for name in layer_names}
    occurrence_targets = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        lexical_ids = batch["lexical_ids"].to(device)
        valid = lexical_ids.ge(0)
        flat_ids = lexical_ids[valid]
        layers = encode_layers(model, input_ids)
        counts += torch.bincount(flat_ids, minlength=vocab_size)
        for name, hidden in layers.items():
            sums[name].index_add_(0, flat_ids, hidden[valid].float())

        target_indices = target_lookup[flat_ids]
        is_target = target_indices.ge(0)
        if is_target.any():
            occurrence_targets.append(target_indices[is_target].cpu())
            for name, hidden in layers.items():
                occurrence_vectors[name].append(hidden[valid][is_target].float().cpu())

    return {
        "counts": counts.cpu(),
        "sums": {name: value.cpu() for name, value in sums.items()},
        "occurrence_targets": torch.cat(occurrence_targets),
        "occurrence_vectors": {
            name: torch.cat(chunks) for name, chunks in occurrence_vectors.items()
        },
    }


def profile_cosine(before: Tensor, after: Tensor) -> Tensor:
    return 1.0 - F.cosine_similarity(before, after, dim=1)


def mean_off_diagonal_cosine_distance(profiles: Tensor) -> Tensor:
    if profiles.size(0) < 2:
        return profiles.new_zeros(())
    similarities = profiles @ profiles.T
    n = profiles.size(0)
    off_diagonal_sum = similarities.sum() - similarities.diagonal().sum()
    return 1.0 - off_diagonal_sum / (n * (n - 1))


def relational_energy_distance(before: Tensor, after: Tensor) -> float:
    cross = 1.0 - (before @ after.T).mean()
    within_before = mean_off_diagonal_cosine_distance(before)
    within_after = mean_off_diagonal_cosine_distance(after)
    return float((2.0 * cross - within_before - within_after).clamp_min(0.0))


def relational_separation_ratio(before: Tensor, after: Tensor) -> float:
    cross = 1.0 - (before @ after.T).mean()
    within = 0.5 * (
        mean_off_diagonal_cosine_distance(before)
        + mean_off_diagonal_cosine_distance(after)
    )
    return float(((cross - within) / cross.clamp_min(1e-9)).clamp_min(0.0))


def evaluate_scores(rows: list[dict], truth: dict[str, dict[str, float]]) -> list[dict]:
    metrics = []
    keys = sorted({(row["method"], row["layer"]) for row in rows})
    for method, layer in keys:
        selected = [row for row in rows if row["method"] == method and row["layer"] == layer]
        selected.sort(key=lambda row: row["target"])
        graded = np.array([truth[row["target"]]["graded"] for row in selected])
        binary = np.array([truth[row["target"]]["binary"] for row in selected])
        scores = np.array([row["score"] for row in selected])
        rho, p_value = spearmanr(graded, scores)
        metrics.append(
            {
                "method": method,
                "layer": layer,
                "n_targets": len(selected),
                "spearman": float(rho),
                "spearman_p": float(p_value),
                "roc_auc": float(roc_auc_score(binary, scores)),
                "average_precision": float(average_precision_score(binary, scores)),
            }
        )
    return sorted(metrics, key=lambda row: row["spearman"], reverse=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate rotation-invariant hidden-state relational profiles."
    )
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--truth", type=Path, default=Path("data/processed/semeval2020_task1/eng_lemma/truth.tsv"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-reference-count", type=int, default=100)
    parser.add_argument("--max-references", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--reuse-cache", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=None)
    args = parser.parse_args()

    config = json.loads((args.experiment_dir / "config.json").read_text(encoding="utf-8"))
    vocab = json.loads((args.experiment_dir / "vocab.json").read_text(encoding="utf-8"))
    targets = json.loads((args.experiment_dir / "targets.json").read_text(encoding="utf-8"))
    token_to_id = {token: index for index, token in enumerate(vocab)}
    target_ids = [token_to_id[target] for target in targets]
    corpora = read_period_corpora(Path(config["input_dir"]))
    if len(corpora) != 2:
        raise ValueError("This diagnostic currently expects exactly two periods")

    corpus_counts = [
        Counter(token for document in corpus.documents for token in document)
        for corpus in corpora
    ]
    candidates = [
        token
        for token in vocab
        if token not in SPECIAL_TOKENS
        and token not in set(targets)
        and min(counts.get(token, 0) for counts in corpus_counts) >= args.min_reference_count
    ]
    candidates.sort(
        key=lambda token: min(counts.get(token, 0) for counts in corpus_counts),
        reverse=True,
    )
    references = candidates[: args.max_references]
    reference_ids = [token_to_id[token] for token in references]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "references.json").write_text(json.dumps(references, indent=2), encoding="utf-8")

    stats_by_cell = {}
    cache_dir = args.cache_dir or (args.output_dir / "cache")
    for checkpoint_index in range(2):
        checkpoint = (
            args.experiment_dir
            / "continual_real"
            / f"checkpoint_t{checkpoint_index:02d}.pt"
        )
        model = build_model(config, len(vocab), token_to_id["[PAD]"])
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        for corpus_index, corpus in enumerate(corpora):
            cache = cache_dir / f"theta{checkpoint_index}_d{corpus_index}.pt"
            if args.reuse_cache and cache.exists():
                print(f"[cache] loading {cache}", flush=True)
                stats = torch.load(cache, map_location="cpu", weights_only=True)
            else:
                started = time.perf_counter()
                print(
                    f"[extract] theta={checkpoint_index} corpus={corpus.period}",
                    flush=True,
                )
                stats = extract_context_statistics(
                    model,
                    corpus,
                    token_to_id,
                    target_ids,
                    seq_len=int(config["seq_len"]),
                    batch_size=args.batch_size,
                    device=args.device,
                )
                cache.parent.mkdir(parents=True, exist_ok=True)
                torch.save(stats, cache)
                print(f"[extract] finished in {time.perf_counter() - started:.1f}s", flush=True)
            stats_by_cell[(checkpoint_index, corpus_index)] = stats
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    truth = read_truth(args.truth)
    score_rows = []
    diagnostic_rows = []
    natural_before = stats_by_cell[(0, 0)]
    natural_after = stats_by_cell[(1, 1)]
    layer_names = list(natural_before["sums"])

    checkpoint_models = []
    for checkpoint_index in range(2):
        model = build_model(config, len(vocab), token_to_id["[PAD]"])
        model.load_state_dict(
            torch.load(
                args.experiment_dir / "continual_real" / f"checkpoint_t{checkpoint_index:02d}.pt",
                map_location="cpu",
                weights_only=True,
            )
        )
        checkpoint_models.append(model)
    embedding_profiles = [
        relational_profiles(model.token_emb.weight.detach(), target_ids, reference_ids)
        for model in checkpoint_models
    ]
    centered_embedding_profiles = [
        relational_profiles(
            model.token_emb.weight.detach(),
            target_ids,
            reference_ids,
            center=True,
        )
        for model in checkpoint_models
    ]
    for target, score in zip(targets, profile_cosine(*embedding_profiles).tolist()):
        target_index = targets.index(target)
        score_rows.append(
            {
                "method": "input_embedding_profile",
                "layer": "embedding",
                "target": target,
                "score": score,
                "count_d0": int(natural_before["occurrence_targets"].eq(target_index).sum()),
                "count_d1": int(natural_after["occurrence_targets"].eq(target_index).sum()),
            }
        )
    for target_index, (target, score) in enumerate(
        zip(targets, profile_cosine(*centered_embedding_profiles).tolist())
    ):
        score_rows.append(
            {
                "method": "centered_input_embedding_profile",
                "layer": "embedding",
                "target": target,
                "score": score,
                "count_d0": int(natural_before["occurrence_targets"].eq(target_index).sum()),
                "count_d1": int(natural_after["occurrence_targets"].eq(target_index).sum()),
            }
        )

    for layer in layer_names:
        centroids = {
            cell: contextual_centroids(stats, layer)
            for cell, stats in stats_by_cell.items()
        }
        profiles = {
            cell: relational_profiles(points, target_ids, reference_ids)
            for cell, points in centroids.items()
        }
        centered_profiles = {
            cell: relational_profiles(points, target_ids, reference_ids, center=True)
            for cell, points in centroids.items()
        }
        natural_scores = profile_cosine(profiles[(0, 0)], profiles[(1, 1)])
        corpus_scores = profile_cosine(profiles[(0, 0)], profiles[(0, 1)])
        checkpoint_scores = profile_cosine(profiles[(0, 0)], profiles[(1, 0)])
        centered_natural_scores = profile_cosine(
            centered_profiles[(0, 0)],
            centered_profiles[(1, 1)],
        )
        for target_index, target in enumerate(targets):
            score_rows.append(
                {
                    "method": "contextual_centroid_profile",
                    "layer": layer,
                    "target": target,
                    "score": float(natural_scores[target_index]),
                    "count_d0": int(natural_before["occurrence_targets"].eq(target_index).sum()),
                    "count_d1": int(natural_after["occurrence_targets"].eq(target_index).sum()),
                }
            )
            diagnostic_rows.append(
                {
                    "layer": layer,
                    "target": target,
                    "natural_theta0d0_theta1d1": float(natural_scores[target_index]),
                    "corpus_theta0d0_theta0d1": float(corpus_scores[target_index]),
                    "checkpoint_theta0d0_theta1d0": float(checkpoint_scores[target_index]),
                }
            )

            before_occurrences = occurrence_profiles(
                natural_before, layer, target_index, centroids[(0, 0)], reference_ids
            )
            after_occurrences = occurrence_profiles(
                natural_after, layer, target_index, centroids[(1, 1)], reference_ids
            )
            score_rows.append(
                {
                    "method": "relational_apd",
                    "layer": layer,
                    "target": target,
                    "score": average_pairwise_cosine_distance(before_occurrences, after_occurrences),
                    "count_d0": int(before_occurrences.size(0)),
                    "count_d1": int(after_occurrences.size(0)),
                }
            )
            score_rows.append(
                {
                    "method": "relational_energy",
                    "layer": layer,
                    "target": target,
                    "score": relational_energy_distance(before_occurrences, after_occurrences),
                    "count_d0": int(before_occurrences.size(0)),
                    "count_d1": int(after_occurrences.size(0)),
                }
            )
            score_rows.append(
                {
                    "method": "relational_separation_ratio",
                    "layer": layer,
                    "target": target,
                    "score": relational_separation_ratio(before_occurrences, after_occurrences),
                    "count_d0": int(before_occurrences.size(0)),
                    "count_d1": int(after_occurrences.size(0)),
                }
            )
            score_rows.append(
                {
                    "method": "centered_contextual_centroid_profile",
                    "layer": layer,
                    "target": target,
                    "score": float(centered_natural_scores[target_index]),
                    "count_d0": int(before_occurrences.size(0)),
                    "count_d1": int(after_occurrences.size(0)),
                }
            )
            centered_before = occurrence_profiles(
                natural_before,
                layer,
                target_index,
                centroids[(0, 0)],
                reference_ids,
                center=True,
            )
            centered_after = occurrence_profiles(
                natural_after,
                layer,
                target_index,
                centroids[(1, 1)],
                reference_ids,
                center=True,
            )
            score_rows.append(
                {
                    "method": "centered_relational_apd",
                    "layer": layer,
                    "target": target,
                    "score": average_pairwise_cosine_distance(centered_before, centered_after),
                    "count_d0": int(centered_before.size(0)),
                    "count_d1": int(centered_after.size(0)),
                }
            )
            score_rows.append(
                {
                    "method": "centered_relational_energy",
                    "layer": layer,
                    "target": target,
                    "score": relational_energy_distance(centered_before, centered_after),
                    "count_d0": int(centered_before.size(0)),
                    "count_d1": int(centered_after.size(0)),
                }
            )

    metrics = evaluate_scores(score_rows, truth)
    write_csv(score_rows, args.output_dir / "scores.csv")
    write_csv(diagnostic_rows, args.output_dir / "factorial_diagnostics.csv")
    write_csv(metrics, args.output_dir / "metrics.csv")
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    best = metrics[0]
    print(json.dumps({"n_references": len(references), "best": best}, indent=2), flush=True)


if __name__ == "__main__":
    main()
