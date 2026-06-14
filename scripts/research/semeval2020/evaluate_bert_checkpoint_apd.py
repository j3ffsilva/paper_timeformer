#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import Tensor

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from timeformers.bert_continual import strip_pos_suffix  # noqa: E402


def read_targets(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def find_occurrences(
    corpus_path: Path,
    targets: set[str],
) -> dict[str, list[tuple[list[str], int]]]:
    occurrences = {target: [] for target in targets}
    with corpus_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            tokens = line.split()
            for index, token in enumerate(tokens):
                if token in targets:
                    normalized = [strip_pos_suffix(item) for item in tokens]
                    occurrences[token].append((normalized, index))
    return occurrences


def average_pairwise_distance(vectors_a: Tensor, vectors_b: Tensor) -> float:
    a = F.normalize(vectors_a, dim=1)
    b = F.normalize(vectors_b, dim=1)
    return float((1.0 - a @ b.T).mean())


def read_truth(path: Path) -> dict[str, dict[str, float]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {
            row["target"]: {
                "binary": float(row["binary"]),
                "graded": float(row["graded"]),
            }
            for row in csv.DictReader(handle, delimiter="\t")
        }


def score_rows(rows: list[dict], truth: dict[str, dict[str, float]]) -> dict:
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
    parser = argparse.ArgumentParser(
        description="Evaluate aligned BERT hidden-state readouts with fixed-encoder occurrence APD."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--period-files", nargs=2, default=["1810-1860.txt", "1960-2010.txt"])
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--truth", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-occurrences", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from transformers import AutoModelForMaskedLM, AutoTokenizer

    targets = read_targets(args.targets)
    target_set = set(targets)
    occurrences = [
        find_occurrences(args.corpus_dir / filename, target_set)
        for filename in args.period_files
    ]
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForMaskedLM.from_pretrained(args.checkpoint).to(args.device)
    model.eval()
    rng = random.Random(args.seed)
    layer_count = model.config.num_hidden_layers
    readout_names = (
        ["embedding"]
        + [f"layer_{index}" for index in range(1, layer_count + 1)]
        + ["mean_transformer_layers", "mean_embedding_and_layers"]
    )
    rows_by_readout = {name: [] for name in readout_names}

    @torch.no_grad()
    def encode_batch(examples: list[tuple[list[str], int]]) -> dict[str, Tensor]:
        encoding = tokenizer(
            [tokens for tokens, _ in examples],
            is_split_into_words=True,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        model_inputs = {
            key: value.to(args.device)
            for key, value in encoding.items()
        }
        outputs = model.bert(
            **model_inputs,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states
        readouts = {"embedding": hidden_states[0]}
        for index in range(1, len(hidden_states)):
            readouts[f"layer_{index}"] = hidden_states[index]
        readouts["mean_transformer_layers"] = torch.stack(hidden_states[1:], dim=0).mean(dim=0)
        readouts["mean_embedding_and_layers"] = torch.stack(hidden_states, dim=0).mean(dim=0)

        result = {name: [] for name in readout_names}
        for batch_index, (_, target_index) in enumerate(examples):
            word_ids = encoding.word_ids(batch_index=batch_index)
            positions = [
                position
                for position, word_id in enumerate(word_ids)
                if word_id == target_index
            ]
            if not positions:
                positions = [1]
            position_tensor = torch.tensor(positions, device=args.device)
            for name, hidden in readouts.items():
                result[name].append(hidden[batch_index, position_tensor].mean(dim=0).cpu())
        return {name: torch.stack(vectors) for name, vectors in result.items()}

    for target in targets:
        samples = []
        for period_occurrences in occurrences:
            sample = list(period_occurrences[target])
            if len(sample) > args.max_occurrences:
                sample = rng.sample(sample, args.max_occurrences)
            samples.append(sample)
        if not samples[0] or not samples[1]:
            continue
        vectors = {
            name: [[], []]
            for name in readout_names
        }
        for period, sample in enumerate(samples):
            for start in range(0, len(sample), args.batch_size):
                encoded = encode_batch(sample[start : start + args.batch_size])
                for name in readout_names:
                    vectors[name][period].append(encoded[name])
        for name in readout_names:
            before = torch.cat(vectors[name][0])
            after = torch.cat(vectors[name][1])
            rows_by_readout[name].append({
                "target": target,
                "apd": average_pairwise_distance(before, after),
                "n_d0": len(before),
                "n_d1": len(after),
            })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {}
    truth = read_truth(args.truth) if args.truth is not None else None
    for name, rows in rows_by_readout.items():
        with (args.output_dir / f"rows_{name}.csv").open(
            "w",
            encoding="utf-8",
            newline="",
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        if truth is not None:
            metrics[name] = score_rows(rows, truth)
    result = {
        "checkpoint": args.checkpoint,
        "max_occurrences": args.max_occurrences,
        "metrics": metrics,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
