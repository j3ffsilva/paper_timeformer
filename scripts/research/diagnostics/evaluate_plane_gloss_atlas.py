#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.research.common.cloze import (  # noqa: E402
    high_confidence_sense,
    occurrence_contexts,
)
from scripts.research.common.encoders import build_model, encode_layers  # noqa: E402
from scripts.research.common.io import write_csv  # noqa: E402
from timeformers.real_corpus import read_period_corpora, tokenize  # noqa: E402


GLOSSES = {
    "geometry": [
        "plane_nn mean a flat two dimensional surface in geometry",
        "a geometric plane_nn contain point and straight line",
        "an inclined plane_nn be a flat surface",
    ],
    "aircraft": [
        "plane_nn mean aircraft a powered flying vehicle",
        "passenger board a plane_nn at the airport for a flight",
        "a pilot fly the plane_nn through the air",
    ],
    "tool": [
        "plane_nn mean a carpenter tool to smooth wood",
        "a carpenter use a plane_nn to shave a wood surface",
        "the bead plane_nn cut and smooth timber",
    ],
}


@torch.no_grad()
def encode_gloss_target(
    model,
    text: str,
    token_to_id: dict[str, int],
    *,
    layer: str,
) -> tuple[torch.Tensor, list[str]]:
    tokens = tokenize(text)
    if tokens.count("plane_nn") != 1:
        raise ValueError(f"Gloss must contain plane_nn exactly once: {text}")
    content_len = model.seq_len - 2
    if len(tokens) > content_len:
        raise ValueError(f"Gloss exceeds model sequence length: {text}")
    unk = token_to_id["[UNK]"]
    ids = [token_to_id["[CLS]"]] + [token_to_id.get(token, unk) for token in tokens]
    ids += [token_to_id["[SEP]"]]
    ids += [token_to_id["[PAD]"]] * (model.seq_len - len(ids))
    input_ids = torch.tensor([ids], dtype=torch.long)
    hidden = encode_layers(model, input_ids)[layer][0, tokens.index("plane_nn") + 1]
    unknown = [token for token in tokens if token not in token_to_id]
    return hidden.float(), unknown


def build_prototypes(
    model,
    token_to_id: dict[str, int],
    center: torch.Tensor,
    *,
    layer: str,
) -> tuple[dict[str, torch.Tensor], list[dict]]:
    prototypes = {}
    gloss_rows = []
    for sense, glosses in GLOSSES.items():
        vectors = []
        for gloss in glosses:
            vector, unknown = encode_gloss_target(
                model,
                gloss,
                token_to_id,
                layer=layer,
            )
            vectors.append(F.normalize(vector - center, dim=0))
            gloss_rows.append(
                {
                    "sense": sense,
                    "gloss": gloss,
                    "unknown_tokens": " ".join(unknown),
                }
            )
        prototypes[sense] = F.normalize(torch.stack(vectors).mean(dim=0), dim=0)
    return prototypes, gloss_rows


def classify_occurrences(
    vectors: torch.Tensor,
    prototypes: dict[str, torch.Tensor],
    center: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    senses = list(prototypes)
    points = F.normalize(vectors.float() - center, dim=1)
    prototype_matrix = torch.stack([prototypes[sense] for sense in senses])
    scores = (points @ prototype_matrix.T).numpy()
    order = np.argsort(scores, axis=1)
    predictions = np.array([senses[index] for index in order[:, -1]])
    margins = scores[np.arange(len(scores)), order[:, -1]] - scores[
        np.arange(len(scores)), order[:, -2]
    ]
    return predictions, margins, senses


def aggregate(rows: list[dict]) -> list[dict]:
    output = []
    keys = sorted(
        {
            (row["checkpoint"], row["corpus"], row["gold_sense"])
            for row in rows
            if row["gold_sense"] != "unlabeled"
        }
    )
    for checkpoint, corpus, gold_sense in keys:
        selected = [
            row
            for row in rows
            if row["checkpoint"] == checkpoint
            and row["corpus"] == corpus
            and row["gold_sense"] == gold_sense
        ]
        output.append(
            {
                "checkpoint": checkpoint,
                "corpus": corpus,
                "gold_sense": gold_sense,
                "n": len(selected),
                "accuracy": float(
                    np.mean([row["prediction"] == gold_sense for row in selected])
                ),
                "mean_margin": float(np.mean([row["margin"] for row in selected])),
                "median_margin": float(np.median([row["margin"] for row in selected])),
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test a fixed lexical gloss atlas for the senses of plane_nn."
    )
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layer", default="layer_2")
    parser.add_argument("--examples", type=int, default=8)
    args = parser.parse_args()

    config = json.loads((args.experiment_dir / "config.json").read_text(encoding="utf-8"))
    vocab = json.loads((args.experiment_dir / "vocab.json").read_text(encoding="utf-8"))
    targets = json.loads((args.experiment_dir / "targets.json").read_text(encoding="utf-8"))
    references = json.loads((args.profile_dir / "references.json").read_text(encoding="utf-8"))
    token_to_id = {token: index for index, token in enumerate(vocab)}
    reference_ids = [token_to_id[token] for token in references]
    target_index = targets.index("plane_nn")
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

    contexts = {
        corpus_index: occurrence_contexts(
            corpus,
            "plane_nn",
            int(config["seq_len"]),
        )
        for corpus_index, corpus in enumerate(corpora)
    }
    rows = []
    gloss_rows = []
    prototype_similarity_rows = []
    for checkpoint in range(2):
        model = build_model(config, len(vocab), token_to_id["[PAD]"])
        model.load_state_dict(
            torch.load(
                args.experiment_dir
                / "continual_real"
                / f"checkpoint_t{checkpoint:02d}.pt",
                map_location="cpu",
                weights_only=True,
            )
        )
        model.eval()
        pooled_sums = (
            stats[(checkpoint, 0)]["sums"][args.layer][reference_ids]
            + stats[(checkpoint, 1)]["sums"][args.layer][reference_ids]
        )
        pooled_counts = (
            stats[(checkpoint, 0)]["counts"][reference_ids]
            + stats[(checkpoint, 1)]["counts"][reference_ids]
        ).float().unsqueeze(1).clamp_min(1.0)
        center = (pooled_sums / pooled_counts).mean(dim=0)
        prototypes, checkpoint_gloss_rows = build_prototypes(
            model,
            token_to_id,
            center,
            layer=args.layer,
        )
        gloss_rows.extend(
            {"checkpoint": checkpoint, **row}
            for row in checkpoint_gloss_rows
        )
        senses = list(prototypes)
        for left in senses:
            for right in senses:
                prototype_similarity_rows.append(
                    {
                        "checkpoint": checkpoint,
                        "left": left,
                        "right": right,
                        "cosine": float(prototypes[left] @ prototypes[right]),
                    }
                )
        for corpus_index, corpus in enumerate(corpora):
            selected = stats[(checkpoint, corpus_index)]["occurrence_targets"] == target_index
            vectors = stats[(checkpoint, corpus_index)]["occurrence_vectors"][args.layer][selected]
            if len(vectors) != len(contexts[corpus_index]):
                raise RuntimeError("Occurrence vectors and contexts are misaligned")
            predictions, margins, senses = classify_occurrences(
                vectors,
                prototypes,
                center,
            )
            point_scores = (
                F.normalize(vectors.float() - center, dim=1)
                @ torch.stack([prototypes[sense] for sense in senses]).T
            ).numpy()
            for index, context in enumerate(contexts[corpus_index]):
                row = {
                    "checkpoint": checkpoint,
                    "corpus": corpus.period,
                    "occurrence_index": index,
                    "gold_sense": high_confidence_sense(context["tokens"]),
                    "prediction": predictions[index],
                    "margin": float(margins[index]),
                    "context": context["display"],
                }
                for sense_index, sense in enumerate(senses):
                    row[f"score_{sense}"] = float(point_scores[index, sense_index])
                rows.append(row)

    aggregates = aggregate(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.output_dir / "occurrence_scores.csv")
    write_csv(aggregates, args.output_dir / "metrics.csv")
    write_csv(gloss_rows, args.output_dir / "glosses.csv")
    write_csv(
        prototype_similarity_rows,
        args.output_dir / "prototype_similarities.csv",
    )

    report = [
        "# Fixed gloss atlas feasibility test for `plane_nn`",
        "",
        "Sense supervision is explicit. Labels for corpus examples are lexical heuristics",
        "used for this feasibility audit, not benchmark annotations.",
        "",
        "## Metrics",
        "",
        "| Checkpoint | Corpus | Sense | N | Accuracy | Mean margin |",
        "|---:|---|---|---:|---:|---:|",
    ]
    for row in aggregates:
        report.append(
            f"| theta{row['checkpoint']} | {row['corpus']} | {row['gold_sense']} | "
            f"{row['n']} | {row['accuracy']:.3f} | {row['mean_margin']:.3f} |"
        )
    report.extend(["", "## Prediction distributions", ""])
    for checkpoint in range(2):
        for corpus in [corpus.period for corpus in corpora]:
            selected = [
                row
                for row in rows
                if row["checkpoint"] == checkpoint and row["corpus"] == corpus
            ]
            counts = Counter(row["prediction"] for row in selected)
            report.append(
                f"- theta{checkpoint} on {corpus}: "
                + ", ".join(
                    f"{sense}={counts[sense]}/{len(selected)}"
                    for sense in GLOSSES
                )
            )
    for checkpoint in range(2):
        report.extend(["", f"## theta{checkpoint} errors and successes", ""])
        selected = [
            row
            for row in rows
            if row["checkpoint"] == checkpoint and row["gold_sense"] != "unlabeled"
        ]
        errors = sorted(
            [row for row in selected if row["prediction"] != row["gold_sense"]],
            key=lambda row: row["margin"],
            reverse=True,
        )[: args.examples]
        successes = sorted(
            [row for row in selected if row["prediction"] == row["gold_sense"]],
            key=lambda row: row["margin"],
            reverse=True,
        )[: args.examples]
        report.extend(["### Confident errors", ""])
        report.extend(
            [
                f"- expected={row['gold_sense']}, predicted={row['prediction']}, "
                f"margin={row['margin']:.3f}: `{row['context']}`"
                for row in errors
            ]
            or ["No errors."]
        )
        report.extend(["", "### Confident successes", ""])
        report.extend(
            [
                f"- expected={row['gold_sense']}, predicted={row['prediction']}, "
                f"margin={row['margin']:.3f}: `{row['context']}`"
                for row in successes
            ]
            or ["No successes."]
        )
    (args.output_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    summary = {
        "target": "plane_nn",
        "layer": args.layer,
        "glosses": GLOSSES,
        "metrics": aggregates,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
