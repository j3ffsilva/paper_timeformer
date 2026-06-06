#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.diagnose_cloze_semantics import occurrence_contexts  # noqa: E402
from scripts.evaluate_hidden_relational_profiles import contextual_centroids, write_csv  # noqa: E402
from scripts.evaluate_seed_community_profiles import (  # noqa: E402
    DEFAULT_FIELDS,
    community_memberships,
)
from timeformers.real_corpus import read_period_corpora  # noqa: E402


PLANE_KEYWORDS = {
    "geometry": {
        "angle",
        "axis",
        "curve",
        "degree",
        "figure",
        "geometric",
        "geometry",
        "horizontal",
        "inclined",
        "line",
        "parallel",
        "perpendicular",
        "point",
        "section",
        "surface",
        "vertical",
    },
    "aviation": {
        "air",
        "aircraft",
        "airline",
        "airport",
        "aboard",
        "bomb",
        "bomber",
        "cargo",
        "crew",
        "engine",
        "flight",
        "fly",
        "flying",
        "land",
        "landing",
        "passenger",
        "pilot",
        "runway",
        "wing",
    },
}


def heuristic_sense(tokens: list[str]) -> str:
    scores = {
        sense: len(set(tokens) & keywords)
        for sense, keywords in PLANE_KEYWORDS.items()
    }
    best = max(scores.values())
    if best == 0:
        return "unlabeled"
    winners = [sense for sense, score in scores.items() if score == best]
    return winners[0] if len(winners) == 1 else "ambiguous"


def transition_counts(rows: list[dict]) -> list[dict]:
    counts = Counter((row["theta0_label"], row["theta1_label"]) for row in rows)
    return [
        {"theta0_label": left, "theta1_label": right, "count": count}
        for (left, right), count in sorted(counts.items())
    ]


def aggregate_rows(rows: list[dict], field_names: list[str]) -> list[dict]:
    output = []
    keys = sorted({(row["corpus"], row["checkpoint"]) for row in rows})
    for corpus, checkpoint in keys:
        selected = [
            row
            for row in rows
            if row["corpus"] == corpus and row["checkpoint"] == checkpoint
        ]
        label_counts = Counter(row["model_label"] for row in selected)
        output.append(
            {
                "corpus": corpus,
                "checkpoint": checkpoint,
                "n": len(selected),
                **{
                    f"mean_{field}": float(
                        np.mean([float(row[field]) for row in selected])
                    )
                    for field in field_names
                },
                **{
                    f"fraction_{field}": label_counts[field] / len(selected)
                    for field in field_names
                },
            }
        )
    return output


def format_example(row: dict) -> str:
    return (
        f"- `{row['context']}`\n"
        f"  heuristic={row['heuristic_sense']}; "
        f"theta0 geometry={float(row['theta0_geometry']):.3f}, "
        f"transport={float(row['theta0_transport']):.3f}; "
        f"theta1 geometry={float(row['theta1_geometry']):.3f}, "
        f"transport={float(row['theta1_transport']):.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit how theta0 and theta1 read the same plane_nn occurrences."
    )
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layer", default="layer_2")
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--examples", type=int, default=8)
    args = parser.parse_args()

    config = json.loads((args.experiment_dir / "config.json").read_text(encoding="utf-8"))
    vocab = json.loads((args.experiment_dir / "vocab.json").read_text(encoding="utf-8"))
    targets = json.loads((args.experiment_dir / "targets.json").read_text(encoding="utf-8"))
    references = json.loads((args.profile_dir / "references.json").read_text(encoding="utf-8"))
    token_to_id = {token: index for index, token in enumerate(vocab)}
    reference_ids = [token_to_id[token] for token in references]
    target = "plane_nn"
    target_index = targets.index(target)
    fields = {
        name: [token for token in tokens if token in token_to_id]
        for name, tokens in DEFAULT_FIELDS.items()
    }
    field_names = list(fields)
    field_ids = [
        [token_to_id[token] for token in fields[field]]
        for field in field_names
    ]
    corpora = read_period_corpora(Path(config["input_dir"]))
    stats = {
        (checkpoint, corpus): torch.load(
            args.profile_dir / "cache" / f"theta{checkpoint}_d{corpus}.pt",
            map_location="cpu",
            weights_only=True,
        )
        for checkpoint in range(2)
        for corpus in range(2)
    }

    profiles = {}
    for checkpoint in range(2):
        for corpus_index in range(2):
            cell = stats[(checkpoint, corpus_index)]
            centroids = contextual_centroids(cell, args.layer)
            profiles[(checkpoint, corpus_index)] = community_memberships(
                cell,
                args.layer,
                target_index,
                centroids,
                reference_ids,
                field_ids,
                temperature=args.temperature,
            ).numpy()

    occurrence_rows = []
    paired_rows = []
    for corpus_index, corpus in enumerate(corpora):
        contexts = occurrence_contexts(corpus, target, int(config["seq_len"]))
        expected = profiles[(0, corpus_index)].shape[0]
        if len(contexts) != expected or profiles[(1, corpus_index)].shape[0] != expected:
            raise RuntimeError(
                f"Context/profile misalignment for {corpus.period}: "
                f"contexts={len(contexts)}, theta0={expected}, "
                f"theta1={profiles[(1, corpus_index)].shape[0]}"
            )
        for occurrence_index, context in enumerate(contexts):
            base = {
                "corpus": corpus.period,
                "occurrence_index": occurrence_index,
                "heuristic_sense": heuristic_sense(context["tokens"]),
                "context": context["display"],
            }
            paired = dict(base)
            for checkpoint in range(2):
                values = profiles[(checkpoint, corpus_index)][occurrence_index]
                model_label = field_names[int(np.argmax(values))]
                row = {
                    **base,
                    "checkpoint": checkpoint,
                    "model_label": model_label,
                }
                for field_index, field in enumerate(field_names):
                    row[field] = float(values[field_index])
                    paired[f"theta{checkpoint}_{field}"] = float(values[field_index])
                occurrence_rows.append(row)
                paired[f"theta{checkpoint}_label"] = model_label
            paired["transport_shift"] = (
                paired["theta1_transport"] - paired["theta0_transport"]
            )
            paired["geometry_shift"] = (
                paired["theta1_geometry"] - paired["theta0_geometry"]
            )
            paired_rows.append(paired)

    aggregates = aggregate_rows(occurrence_rows, field_names)
    transitions = []
    for corpus in [corpus.period for corpus in corpora]:
        rows = [row for row in paired_rows if row["corpus"] == corpus]
        for transition in transition_counts(rows):
            transitions.append({"corpus": corpus, **transition})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(occurrence_rows, args.output_dir / "occurrence_memberships.csv")
    write_csv(paired_rows, args.output_dir / "paired_occurrences.csv")
    write_csv(aggregates, args.output_dir / "aggregate_cells.csv")
    write_csv(transitions, args.output_dir / "checkpoint_transitions.csv")

    aggregate_lookup = {
        (row["checkpoint"], row["corpus"]): row for row in aggregates
    }
    report = [
        "# `plane_nn` checkpoint semantic audit",
        "",
        "The same occurrence is projected onto fixed lexical fields under both checkpoints.",
        "The lexical heuristic is used only to select auditable examples.",
        "",
        "## Aggregate field masses",
        "",
        "| Corpus | Checkpoint | N | Geometry | Transport | Geometry argmax | Transport argmax |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for corpus in [corpus.period for corpus in corpora]:
        for checkpoint in range(2):
            row = aggregate_lookup[(checkpoint, corpus)]
            report.append(
                f"| {corpus} | theta{checkpoint} | {row['n']} | "
                f"{row['mean_geometry']:.3f} | {row['mean_transport']:.3f} | "
                f"{row['fraction_geometry']:.3f} | {row['fraction_transport']:.3f} |"
            )

    report.extend(["", "## Checkpoint label transitions", ""])
    for corpus in [corpus.period for corpus in corpora]:
        report.extend(
            [
                f"### {corpus}",
                "",
                "| theta0 | theta1 | N |",
                "|---|---|---:|",
            ]
        )
        for row in transitions:
            if row["corpus"] == corpus:
                report.append(
                    f"| {row['theta0_label']} | {row['theta1_label']} | {row['count']} |"
                )
        report.append("")

    report.extend(
        [
            "## Lexical-heuristic subsets",
            "",
            "These labels are incomplete and are used only as an audit aid.",
            "",
            "| Corpus | Heuristic | N | theta0 geometry | theta0 transport | theta1 geometry | theta1 transport |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for corpus in [corpus.period for corpus in corpora]:
        corpus_rows = [row for row in paired_rows if row["corpus"] == corpus]
        for sense in ("geometry", "aviation", "unlabeled", "ambiguous"):
            selected = [
                row for row in corpus_rows if row["heuristic_sense"] == sense
            ]
            if not selected:
                continue
            report.append(
                f"| {corpus} | {sense} | {len(selected)} | "
                f"{np.mean([row['theta0_geometry'] for row in selected]):.3f} | "
                f"{np.mean([row['theta0_transport'] for row in selected]):.3f} | "
                f"{np.mean([row['theta1_geometry'] for row in selected]):.3f} | "
                f"{np.mean([row['theta1_transport'] for row in selected]):.3f} |"
            )
    report.append("")

    for corpus in [corpus.period for corpus in corpora]:
        corpus_rows = [row for row in paired_rows if row["corpus"] == corpus]
        changed = sorted(
            corpus_rows,
            key=lambda row: abs(float(row["transport_shift"])),
            reverse=True,
        )[: args.examples]
        report.extend(
            [
                f"## Largest checkpoint reinterpretations: {corpus}",
                "",
                *[format_example(row) for row in changed],
                "",
            ]
        )
        for sense in ("geometry", "aviation"):
            selected = [
                row for row in corpus_rows if row["heuristic_sense"] == sense
            ][: args.examples]
            report.extend(
                [
                    f"## Heuristic {sense} examples: {corpus}",
                    "",
                    *([format_example(row) for row in selected] or ["No examples found."]),
                    "",
                ]
            )

    (args.output_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    summary = {
        "target": target,
        "layer": args.layer,
        "temperature": args.temperature,
        "aggregates": aggregates,
        "transitions": transitions,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
