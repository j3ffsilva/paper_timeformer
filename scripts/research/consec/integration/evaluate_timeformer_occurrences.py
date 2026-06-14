#!/usr/bin/env python3
"""Align ConSeC posteriors and TimeFormer vectors on identical occurrences."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from scripts.research.common.consec import (  # noqa: E402
    collect_unique_rows,
    prepare_context,
)
from scripts.research.common.stats import (  # noqa: E402
    bootstrap_mean_ci,
    jensen_shannon,
    partial_spearman,
    sign_flip_p,
    spearman,
)


LAYERS = ("layer_1", "layer_2")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def safe_spearman(values_a, values_b) -> float:
    if len(np.unique(values_a)) < 2 or len(np.unique(values_b)) < 2:
        return float("nan")
    return spearman(values_a, values_b)


def centered_wordpiece_encoding(tokenizer, context: str, target: str, max_length: int):
    tokens, target_index, _ = prepare_context(context, target)
    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        add_special_tokens=False,
        truncation=False,
    )
    word_ids = encoding.word_ids()
    positions = [
        index for index, word_id in enumerate(word_ids)
        if word_id == target_index
    ]
    if not positions:
        raise ValueError(f"Tokenizer lost target in context: {context}")
    content_length = max_length - tokenizer.num_special_tokens_to_add(False)
    if len(positions) > content_length:
        raise ValueError(f"Target exceeds content window: {target}")
    midpoint = (positions[0] + positions[-1]) // 2
    start = max(0, midpoint - content_length // 2)
    start = min(start, max(0, len(encoding["input_ids"]) - content_length))
    if positions[0] < start:
        start = positions[0]
    if positions[-1] >= start + content_length:
        start = positions[-1] - content_length + 1
    end = min(len(encoding["input_ids"]), start + content_length)
    content_ids = encoding["input_ids"][start:end]
    relative = [
        index - start for index in positions
        if start <= index < end
    ]
    if tokenizer.cls_token_id is None or tokenizer.sep_token_id is None:
        raise ValueError("BERT-style CLS and SEP tokens are required")
    input_ids = [tokenizer.cls_token_id, *content_ids, tokenizer.sep_token_id]
    target_positions = [index + 1 for index in relative]
    return input_ids, target_positions


def encode_checkpoint(
    checkpoint: Path,
    rows: list[dict],
    *,
    batch_size: int,
    max_length: int,
    device: str,
) -> dict[str, np.ndarray]:
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForMaskedLM.from_pretrained(checkpoint).to(device)
    model.eval()
    vectors = {layer: [] for layer in LAYERS}
    examples = [
        centered_wordpiece_encoding(
            tokenizer,
            row["context"],
            row["target"],
            max_length,
        )
        for row in rows
    ]
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            batch = examples[start : start + batch_size]
            longest = max(len(input_ids) for input_ids, _ in batch)
            input_ids = []
            attention_mask = []
            for ids, _ in batch:
                padding = longest - len(ids)
                input_ids.append(ids + [tokenizer.pad_token_id] * padding)
                attention_mask.append([1] * len(ids) + [0] * padding)
            outputs = model.bert(
                input_ids=torch.tensor(input_ids, device=device),
                attention_mask=torch.tensor(attention_mask, device=device),
                output_hidden_states=True,
                return_dict=True,
            )
            for layer_index, layer in enumerate(LAYERS, start=1):
                hidden = outputs.hidden_states[layer_index]
                for batch_index, (_, positions) in enumerate(batch):
                    vectors[layer].append(
                        hidden[batch_index, positions].mean(dim=0).cpu().numpy()
                    )
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return {
        layer: np.asarray(values, dtype=np.float32)
        for layer, values in vectors.items()
    }


def load_or_create_embeddings(args, rows: list[dict]) -> dict[str, np.ndarray]:
    expected_ids = np.asarray([row["sample_id"] for row in rows])
    if args.embedding_cache.exists():
        cached = dict(np.load(args.embedding_cache, allow_pickle=False))
        cached_ids = cached.pop("sample_ids").astype(str)
        if not np.array_equal(cached_ids, expected_ids):
            raise ValueError("Embedding cache sample IDs do not match inputs")
        return cached

    result = {}
    for index, checkpoint in enumerate(args.checkpoints):
        encoded = encode_checkpoint(
            checkpoint,
            rows,
            batch_size=args.batch_size,
            max_length=args.max_length,
            device=args.device,
        )
        for layer, values in encoded.items():
            result[f"model_{index}_{layer}"] = values
    args.embedding_cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.embedding_cache,
        sample_ids=expected_ids,
        **result,
    )
    return result


def pair_metrics(rows: list[dict], vectors: np.ndarray) -> dict[str, float]:
    probabilities_by_row = [
        json.loads(row["sense_probabilities"]) for row in rows
    ]
    sensekeys = sorted(probabilities_by_row[0])
    if any(sorted(values) != sensekeys for values in probabilities_by_row):
        raise ValueError(f"Sense inventory mismatch for {rows[0]['target']}")
    probabilities = np.asarray(
        [[values[key] for key in sensekeys] for values in probabilities_by_row],
        dtype=np.float64,
    )
    normalized = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    geometric = []
    semantic = []
    cross_period = []
    same_sense = []
    periods = [row["period"] for row in rows]
    predictions = [row["prediction_sensekey"] for row in rows]
    for left in range(len(rows)):
        for right in range(left + 1, len(rows)):
            geometric.append(
                1.0 - float(np.dot(normalized[left], normalized[right]))
            )
            semantic.append(
                jensen_shannon(probabilities[left], probabilities[right])
            )
            cross_period.append(int(periods[left] != periods[right]))
            same_sense.append(predictions[left] == predictions[right])
    geometric = np.asarray(geometric)
    semantic = np.asarray(semantic)
    cross_period = np.asarray(cross_period)
    same_sense = np.asarray(same_sense)
    within = cross_period == 0
    between = cross_period == 1
    same = same_sense
    same_within = same & within
    same_between = same & between
    hard_drift = float("nan")
    if same_within.any() and same_between.any():
        hard_drift = float(
            geometric[same_between].mean() - geometric[same_within].mean()
        )
    return {
        "n_pairs": len(geometric),
        "semantic_geometry_partial_period": partial_spearman(
            semantic, geometric, cross_period
        ),
        "semantic_geometry_within_period": safe_spearman(
            semantic[within], geometric[within]
        ),
        "semantic_geometry_between_period": safe_spearman(
            semantic[between], geometric[between]
        ),
        "period_geometry_partial_semantic": partial_spearman(
            cross_period, geometric, semantic
        ),
        "hard_same_sense_period_drift": hard_drift,
        "n_same_sense_within_pairs": int(same_within.sum()),
        "n_same_sense_between_pairs": int(same_between.sum()),
    }


def evaluate(args: argparse.Namespace) -> None:
    unique_rows, per_file = collect_unique_rows(args.prediction_files)
    embeddings = load_or_create_embeddings(args, unique_rows)
    index_by_id = {
        row["sample_id"]: index for index, row in enumerate(unique_rows)
    }
    detailed = []
    for consec_index, source_rows in enumerate(per_file):
        grouped = defaultdict(list)
        for row in source_rows:
            if row["role"] == "confirmatory":
                grouped[row["target"]].append(row)
        for model_index in range(len(args.checkpoints)):
            for layer in LAYERS:
                all_vectors = embeddings[f"model_{model_index}_{layer}"]
                for target in sorted(grouped):
                    rows = grouped[target]
                    if len(rows) != 50:
                        raise ValueError(
                            f"Expected 50 rows for {target}, got {len(rows)}"
                        )
                    indices = [index_by_id[row["sample_id"]] for row in rows]
                    detailed.append(
                        {
                            "consec_seed": args.consec_seeds[consec_index],
                            "model_seed": args.model_seeds[model_index],
                            "layer": layer,
                            "target": target,
                            **pair_metrics(rows, all_vectors[indices]),
                        }
                    )

    primary_rows = [row for row in detailed if row["layer"] == "layer_1"]
    combination_summaries = []
    for consec_seed in args.consec_seeds:
        for model_seed in args.model_seeds:
            selected = [
                row for row in primary_rows
                if row["consec_seed"] == consec_seed
                and row["model_seed"] == model_seed
            ]
            values = np.asarray(
                [row["semantic_geometry_partial_period"] for row in selected]
            )
            combination_summaries.append(
                {
                    "consec_seed": consec_seed,
                    "model_seed": model_seed,
                    "mean_target_rho": float(np.nanmean(values)),
                    "median_target_rho": float(np.nanmedian(values)),
                    "positive_targets": int(np.sum(values > 0)),
                }
            )

    targets = sorted({row["target"] for row in primary_rows})
    aggregate_targets = []
    for target in targets:
        selected = [row for row in primary_rows if row["target"] == target]
        aggregate_targets.append(
            {
                "target": target,
                "mean_partial_rho_layer_1": float(
                    np.nanmean(
                        [
                            row["semantic_geometry_partial_period"]
                            for row in selected
                        ]
                    )
                ),
                "mean_within_period_rho_layer_1": float(
                    np.nanmean(
                        [
                            row["semantic_geometry_within_period"]
                            for row in selected
                        ]
                    )
                ),
                "mean_between_period_rho_layer_1": float(
                    np.nanmean(
                        [
                            row["semantic_geometry_between_period"]
                            for row in selected
                        ]
                    )
                ),
                "mean_period_residual_layer_1": float(
                    np.nanmean(
                        [
                            row["period_geometry_partial_semantic"]
                            for row in selected
                        ]
                    )
                ),
                "mean_hard_same_sense_drift_layer_1": float(
                    np.nanmean(
                        [
                            row["hard_same_sense_period_drift"]
                            for row in selected
                        ]
                    )
                ),
            }
        )
    aggregate_values = np.asarray(
        [row["mean_partial_rho_layer_1"] for row in aggregate_targets]
    )
    checks = {
        "median_positive_all_six_combinations": bool(
            all(row["median_target_rho"] > 0 for row in combination_summaries)
        ),
        "aggregate_mean_positive": bool(np.mean(aggregate_values) > 0),
        "sign_flip_p_below_0_05": bool(
            sign_flip_p(
                aggregate_values,
                args.n_permutations,
                args.permutation_seed,
            )
            < 0.05
        ),
        "at_least_15_positive_targets": bool(
            np.sum(aggregate_values > 0) >= 15
        ),
    }
    p_value = sign_flip_p(
        aggregate_values,
        args.n_permutations,
        args.permutation_seed,
    )
    layer_secondary = {}
    metrics = [
        "semantic_geometry_partial_period",
        "semantic_geometry_within_period",
        "semantic_geometry_between_period",
        "period_geometry_partial_semantic",
        "hard_same_sense_period_drift",
    ]
    for layer_index, layer in enumerate(LAYERS):
        layer_secondary[layer] = {}
        for metric_index, metric in enumerate(metrics):
            target_values = []
            for target in targets:
                selected = [
                    row for row in detailed
                    if row["layer"] == layer and row["target"] == target
                ]
                target_values.append(
                    float(np.nanmean([row[metric] for row in selected]))
                )
            valid = np.asarray(target_values)
            valid = valid[~np.isnan(valid)]
            layer_secondary[layer][metric] = {
                "mean_across_targets": float(np.mean(valid)),
                "median_across_targets": float(np.median(valid)),
                "positive_targets": int(np.sum(valid > 0)),
                "bootstrap_mean_ci_95": bootstrap_mean_ci(
                    valid,
                    args.n_bootstrap,
                    args.permutation_seed + 100 + layer_index * 10 + metric_index,
                ),
                "sign_flip_p_one_sided": sign_flip_p(
                    valid,
                    args.n_permutations,
                    args.permutation_seed + 200 + layer_index * 10 + metric_index,
                ),
            }
    summary = {
        "unique_confirmatory_occurrences": len(unique_rows),
        "n_targets": len(targets),
        "n_consec_seeds": len(args.consec_seeds),
        "n_model_seeds": len(args.model_seeds),
        "primary": {
            "mean_target_rho": float(np.mean(aggregate_values)),
            "median_target_rho": float(np.median(aggregate_values)),
            "positive_targets": int(np.sum(aggregate_values > 0)),
            "bootstrap_mean_ci_95": bootstrap_mean_ci(
                aggregate_values,
                args.n_bootstrap,
                args.permutation_seed,
            ),
            "sign_flip_p_one_sided": p_value,
            "combinations": combination_summaries,
        },
        "secondary_layer_means": layer_secondary,
        "checks": checks,
        "occurrence_alignment_passed": bool(all(checks.values())),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_combination_target.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detailed[0]))
        writer.writeheader()
        writer.writerows(detailed)
    with (args.output_dir / "per_target.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate_targets[0]))
        writer.writeheader()
        writer.writerows(aggregate_targets)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-files", type=Path, nargs=3, required=True)
    parser.add_argument("--consec-seeds", nargs=3, required=True)
    parser.add_argument("--checkpoints", type=Path, nargs=2, required=True)
    parser.add_argument("--model-seeds", nargs=2, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--embedding-cache", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-permutations", type=int, default=20000)
    parser.add_argument("--n-bootstrap", type=int, default=20000)
    parser.add_argument("--permutation-seed", type=int, default=20260619)
    return parser


if __name__ == "__main__":
    evaluate(build_parser().parse_args())
