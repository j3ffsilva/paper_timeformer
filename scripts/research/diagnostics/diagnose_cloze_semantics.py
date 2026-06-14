#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.research.common.cloze import occurrence_contexts  # noqa: E402
from scripts.research.common.encoders import build_model  # noqa: E402
from timeformers.real_corpus import RealTargetOccurrenceDataset, read_period_corpora  # noqa: E402
from timeformers.real_models import RealStaticMLM  # noqa: E402
from timeformers.relational import jensen_shannon_divergence_rows  # noqa: E402


def entropy(distribution: torch.Tensor) -> float:
    values = distribution.double().clamp_min(torch.finfo(torch.float64).tiny)
    return float(-(values * values.log()).sum())


def top_tokens(distribution: torch.Tensor, vocab: list[str], k: int) -> list[dict]:
    values, indices = torch.topk(distribution, k=min(k, distribution.numel()))
    return [
        {
            "rank": rank,
            "token": vocab[int(index)],
            "probability": float(value),
        }
        for rank, (value, index) in enumerate(zip(values, indices), start=1)
    ]


@torch.no_grad()
def infer_occurrences(
    model: RealStaticMLM,
    dataset: RealTargetOccurrenceDataset,
    *,
    batch_size: int,
    device: str,
) -> torch.Tensor:
    model.eval()
    model.to(device)
    rows = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        out = model(input_ids, batch["epoch_idx"].to(device))
        batch_indices = torch.arange(input_ids.size(0), device=device)
        logits = out["logits"][batch_indices, batch["mask_pos"].to(device), :]
        rows.append(torch.softmax(logits, dim=-1).cpu())
    if not rows:
        return torch.empty(0, model.vocab_size)
    return torch.cat(rows, dim=0)


def summarize_distribution(
    distribution: torch.Tensor,
    *,
    target_id: int,
    vocab: list[str],
    top_k: int,
) -> dict:
    target_probability = float(distribution[target_id])
    target_rank = int((distribution > distribution[target_id]).sum()) + 1
    return {
        "entropy": entropy(distribution),
        "normalized_entropy": entropy(distribution) / math.log(distribution.numel()),
        "top_1_mass": float(torch.topk(distribution, 1).values.sum()),
        "top_10_mass": float(torch.topk(distribution, min(10, distribution.numel())).values.sum()),
        "top_100_mass": float(torch.topk(distribution, min(100, distribution.numel())).values.sum()),
        "target_probability": target_probability,
        "target_rank": target_rank,
        "top_tokens": top_tokens(distribution, vocab, top_k),
    }


def write_occurrence_csv(rows: list[dict], path: Path) -> None:
    fields = [
        "checkpoint",
        "corpus_period",
        "occurrence",
        "mask_pos",
        "sense",
        "context",
        "target_probability",
        "target_rank",
        "entropy",
        "normalized_entropy",
        "top_predictions",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def format_top(items: list[dict]) -> str:
    return ", ".join(
        f"{item['token']} ({item['probability']:.4g})"
        for item in items
    )


def write_markdown(report: dict, path: Path) -> None:
    lines = [
        f"# Cloze semantic diagnostic: `{report['target']}`",
        "",
        "This report inspects raw MLM distributions before any prior or PMI transformation.",
        "",
        "## Aggregate checkpoint x corpus matrix",
        "",
        "| Checkpoint | Corpus | N | H(q)/log|V| | Target rank | Target probability | Top predictions |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for cell in report["cells"]:
        summary = cell["aggregate"]
        lines.append(
            f"| {cell['checkpoint']} | {cell['corpus_period']} | {cell['n_occurrences']} "
            f"| {summary['normalized_entropy']:.3f} | {summary['target_rank']} "
            f"| {summary['target_probability']:.4g} | {format_top(summary['top_tokens'])} |"
        )

    lines.extend(["", "## Pairwise JSD between aggregate raw q distributions", ""])
    for row in report["pairwise_jsd"]:
        lines.append(f"- `{row['left']}` vs `{row['right']}`: {row['jsd']:.6f}")

    lines.extend(["", "## Sense-group summaries", ""])
    for cell in report["cells"]:
        lines.append(f"### {cell['checkpoint']} on {cell['corpus_period']}")
        lines.append("")
        if not cell["sense_groups"]:
            lines.append("No labeled groups.")
            lines.append("")
            continue
        for sense, group in cell["sense_groups"].items():
            lines.append(
                f"- **{sense}** (n={group['n']}): target rank={group['target_rank']}, "
                f"target p={group['target_probability']:.4g}; "
                f"{format_top(group['top_tokens'])}"
            )
        lines.append("")

    lines.extend(["## Example occurrences", ""])
    for cell in report["cells"]:
        lines.append(f"### {cell['checkpoint']} on {cell['corpus_period']}")
        lines.append("")
        for example in cell["examples"]:
            lines.append(
                f"- **{example['sense']}**: `{example['context']}`  \n"
                f"  target rank={example['target_rank']}, p={example['target_probability']:.4g}; "
                f"top: {format_top(example['top_tokens'])}"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect raw cloze distributions across checkpoint and corpus periods."
    )
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--examples-per-cell", type=int, default=8)
    parser.add_argument("--max-occurrences", type=int, default=None)
    args = parser.parse_args()

    config = json.loads((args.experiment_dir / "config.json").read_text(encoding="utf-8"))
    vocab = json.loads((args.experiment_dir / "vocab.json").read_text(encoding="utf-8"))
    token_to_id = {token: index for index, token in enumerate(vocab)}
    if args.target not in token_to_id:
        raise ValueError(f"Target is not in vocabulary: {args.target}")

    corpora = read_period_corpora(Path(config["input_dir"]))
    checkpoint_paths = sorted((args.experiment_dir / "continual_real").glob("checkpoint_t*.pt"))
    if len(checkpoint_paths) != len(corpora):
        raise ValueError("Expected one checkpoint per corpus period")

    cells = []
    occurrence_rows = []
    aggregate_distributions: dict[str, torch.Tensor] = {}
    for checkpoint_index, checkpoint_path in enumerate(checkpoint_paths):
        model = build_model(config, len(vocab), token_to_id["[PAD]"])
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
        checkpoint_label = f"theta_{checkpoint_index}"
        for corpus_index, corpus in enumerate(corpora):
            contexts = occurrence_contexts(corpus, args.target, int(config["seq_len"]))
            dataset = RealTargetOccurrenceDataset(
                corpus,
                [args.target],
                token_to_id,
                period_idx=corpus_index,
                seq_len=int(config["seq_len"]),
                max_occurrences_per_target=args.max_occurrences,
            )
            if args.max_occurrences is not None:
                contexts = contexts[: args.max_occurrences]
            probabilities = infer_occurrences(
                model,
                dataset,
                batch_size=args.batch_size,
                device=args.device,
            )
            if probabilities.size(0) != len(contexts):
                raise RuntimeError("Occurrence contexts and model inputs are misaligned")
            if probabilities.size(0) == 0:
                continue

            key = f"{checkpoint_label}@{corpus.period}"
            aggregate = probabilities.mean(dim=0)
            aggregate_distributions[key] = aggregate
            aggregate_summary = summarize_distribution(
                aggregate,
                target_id=token_to_id[args.target],
                vocab=vocab,
                top_k=args.top_k,
            )

            grouped_indices: dict[str, list[int]] = defaultdict(list)
            examples = []
            for index, (context, distribution) in enumerate(zip(contexts, probabilities)):
                grouped_indices[context["sense"]].append(index)
                mask_pos = int(dataset[index]["mask_pos"])
                summary = summarize_distribution(
                    distribution,
                    target_id=token_to_id[args.target],
                    vocab=vocab,
                    top_k=args.top_k,
                )
                occurrence_rows.append(
                    {
                        "checkpoint": checkpoint_label,
                        "corpus_period": corpus.period,
                        "occurrence": index,
                        "mask_pos": mask_pos,
                        "sense": context["sense"],
                        "context": context["display"],
                        "target_probability": summary["target_probability"],
                        "target_rank": summary["target_rank"],
                        "entropy": summary["entropy"],
                        "normalized_entropy": summary["normalized_entropy"],
                        "top_predictions": format_top(summary["top_tokens"]),
                    }
                )
                if len(examples) < args.examples_per_cell:
                    examples.append(
                        {
                            "sense": context["sense"],
                            "context": context["display"],
                            "target_probability": summary["target_probability"],
                            "target_rank": summary["target_rank"],
                            "top_tokens": summary["top_tokens"],
                        }
                    )

            sense_groups = {}
            for sense, indices in sorted(grouped_indices.items()):
                group_distribution = probabilities[indices].mean(dim=0)
                sense_groups[sense] = {
                    "n": len(indices),
                    **summarize_distribution(
                        group_distribution,
                        target_id=token_to_id[args.target],
                        vocab=vocab,
                        top_k=args.top_k,
                    ),
                }
            cells.append(
                {
                    "checkpoint": checkpoint_label,
                    "corpus_period": corpus.period,
                    "n_occurrences": probabilities.size(0),
                    "aggregate": aggregate_summary,
                    "sense_groups": sense_groups,
                    "examples": examples,
                }
            )

    pairwise_jsd = []
    keys = list(aggregate_distributions)
    for left_index, left in enumerate(keys):
        for right in keys[left_index + 1 :]:
            divergence = jensen_shannon_divergence_rows(
                aggregate_distributions[left].unsqueeze(0),
                aggregate_distributions[right].unsqueeze(0),
            )
            pairwise_jsd.append(
                {
                    "left": left,
                    "right": right,
                    "jsd": float(divergence[0]),
                }
            )

    report = {
        "target": args.target,
        "experiment_dir": str(args.experiment_dir),
        "cells": cells,
        "pairwise_jsd": pairwise_jsd,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_occurrence_csv(occurrence_rows, args.output_dir / "occurrences.csv")
    write_markdown(report, args.output_dir / "report.md")
    print(f"Wrote cloze diagnostic to {args.output_dir}")


if __name__ == "__main__":
    main()
