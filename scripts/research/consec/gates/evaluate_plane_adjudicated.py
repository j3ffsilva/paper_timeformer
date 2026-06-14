#!/usr/bin/env python3
"""Evaluate frozen ConSeC on the blind-adjudicated historical plane contexts.

This entry point is intentionally compatible with Python 3.7 because the
official ConSeC checkpoint depends on its published 2021 software stack.
"""

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np


PLANE_SENSES = (
    (
        "plane%1:06:00::",
        "tool",
        "a carpenter's hand tool with an adjustable blade for smoothing or shaping wood",
    ),
    (
        "plane%1:06:01::",
        "aircraft",
        "an aircraft that has a fixed wing and is powered by propellers or jets",
    ),
    (
        "plane%1:06:02::",
        "tool",
        "a power tool for smoothing or shaping wood",
    ),
    (
        "plane%1:25:00::",
        "geometry",
        "(mathematics) an unbounded two-dimensional shape",
    ),
    (
        "plane%1:26:00::",
        "other",
        "a level of existence or development",
    ),
)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def prepare_context(context):
    tokens = context.split()
    marked = [index for index, token in enumerate(tokens) if token == "[plane]"]
    if len(marked) != 1:
        raise ValueError("Expected exactly one [plane] marker: {}".format(context))
    target_index = marked[0]
    tokens[target_index] = "plane"
    return tokens, target_index


def compute_summary(rows):
    evaluable = [row for row in rows if row["human_label"] != "unclear"]
    tools = [row for row in rows if row["human_label"] == "tool"]
    correct = sum(row["prediction"] == row["human_label"] for row in evaluable)
    tool_correct = sum(row["prediction"] == "tool" for row in tools)
    confusion = Counter(
        (row["human_label"], row["prediction"]) for row in rows
    )
    return {
        "method": "ConSeC SemCor+WNGT frozen, target-only extraction",
        "number_of_items": len(rows),
        "model_accuracy_excluding_unclear": (
            correct / len(evaluable) if evaluable else None
        ),
        "model_correct_excluding_unclear": correct,
        "model_evaluable_items": len(evaluable),
        "model_tool_accuracy": tool_correct / len(tools) if tools else None,
        "model_tool_correct": tool_correct,
        "model_tool_items": len(tools),
        "confusion": [
            {
                "human_label": human_label,
                "model_prediction": prediction,
                "count": count,
            }
            for (human_label, prediction), count in sorted(confusion.items())
        ],
        "protocol_notes": [
            "The official checkpoint and candidate definitions are frozen.",
            "No parameter or threshold was selected on these 19 items.",
            "The target-only official extraction interface is used without the feedback loop.",
            "The simple lemma plane has no botanical candidate in WordNet.",
            "Human labels come from one blind annotator who used Google Translate.",
        ],
    }


def bootstrap_accuracy(correct, n_bootstrap, seed):
    values = np.asarray(correct, dtype=np.float32)
    rng = np.random.default_rng(seed)
    samples = values[
        rng.integers(0, len(values), size=(n_bootstrap, len(values)))
    ].mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return {
        "n": len(values),
        "accuracy": float(values.mean()),
        "ci_95_low": float(low),
        "ci_95_high": float(high),
    }


def gate_summary(rows, label_field, n_bootstrap, seed):
    strata = (
        ("1810-1860", "geometry"),
        ("1810-1860", "tool"),
        ("1960-2010", "aircraft"),
    )
    metrics = []
    arrays = []
    for offset, (corpus, label) in enumerate(strata):
        stratum = [
            row
            for row in rows
            if row["corpus"] == corpus and row[label_field] == label
        ]
        correct = [row["prediction"] == label for row in stratum]
        metric = bootstrap_accuracy(correct, n_bootstrap, seed + offset)
        metric.update({"corpus": corpus, "gold_sense": label})
        metrics.append(metric)
        arrays.append(np.asarray(correct, dtype=np.float32))

    rng = np.random.default_rng(seed + 100)
    macro_samples = np.stack(
        [
            values[
                rng.integers(0, len(values), size=(n_bootstrap, len(values)))
            ].mean(axis=1)
            for values in arrays
        ],
        axis=1,
    ).mean(axis=1)
    low, high = np.quantile(macro_samples, [0.025, 0.975])
    macro = {
        "accuracy": float(np.mean([values.mean() for values in arrays])),
        "ci_95_low": float(low),
        "ci_95_high": float(high),
        "chance_baseline": 1.0 / 3.0,
    }
    by_key = {
        (metric["corpus"], metric["gold_sense"]): metric for metric in metrics
    }
    anchors = [
        row
        for row in rows
        if row["corpus"] == "1810-1860" and row["anchor"] == "True"
    ]
    checks = {
        "macro_accuracy_ci_above_one_third": macro["ci_95_low"] > (1.0 / 3.0),
        "d0_geometry_accuracy_at_least_0_75": (
            by_key[("1810-1860", "geometry")]["accuracy"] >= 0.75
        ),
        "d1_aircraft_accuracy_at_least_0_80": (
            by_key[("1960-2010", "aircraft")]["accuracy"] >= 0.80
        ),
        "d0_tool_ci_above_one_third": (
            by_key[("1810-1860", "tool")]["ci_95_low"] > (1.0 / 3.0)
        ),
        "historical_anchor_is_geometry": bool(
            anchors and all(row["prediction"] == "geometry" for row in anchors)
        ),
    }
    return {
        "label_field": label_field,
        "metrics": metrics,
        "macro_accuracy": macro,
        "checks": checks,
        "passed": all(checks.values()),
        "n_historical_anchors": len(anchors),
    }


def load_gate_rows(gate_predictions, annotations_path, manifest_path):
    source_rows = read_csv(gate_predictions)
    selected = [
        row
        for row in source_rows
        if (
            row["corpus"] == "1810-1860"
            and row["gold_sense"] in {"geometry", "tool"}
        )
        or (
            row["corpus"] == "1960-2010"
            and row["gold_sense"] == "aircraft"
        )
    ]
    annotations = {row["item_id"]: row for row in read_csv(annotations_path)}
    manifest = {row["item_id"]: row for row in read_csv(manifest_path)}
    human_by_source = {
        (manifest[item_id]["corpus"], manifest[item_id]["occurrence_index"]): row[
            "label"
        ]
        for item_id, row in annotations.items()
    }
    output = []
    for row in selected:
        key = (row["corpus"], row["occurrence_index"])
        output.append(
            {
                "sample_id": "{}-{}".format(*key),
                "corpus": row["corpus"],
                "occurrence_index": row["occurrence_index"],
                "original_gold_sense": row["gold_sense"],
                "adjudicated_gold_sense": human_by_source.get(
                    key, row["gold_sense"]
                ),
                "anchor": row["anchor"],
                "context": row["context"],
            }
        )
    if len(output) != 409:
        raise ValueError("Expected 409 Gate 1 occurrences, found {}".format(len(output)))
    return output


def evaluate(args):
    repo = args.consec_repo.resolve()
    sys.path.insert(0, str(repo))

    import hydra
    import torch

    from src.consec_dataset import ConsecDefinition, ConsecSample
    from src.disambiguation_corpora import DisambiguationInstance
    from src.pl_modules import ConsecPLModule
    from src.scripts.model.predict import predict

    if args.gate_predictions:
        evaluation_rows = load_gate_rows(
            args.gate_predictions, args.annotations, args.manifest
        )
    else:
        annotations = {row["item_id"]: row for row in read_csv(args.annotations)}
        manifest = {row["item_id"]: row for row in read_csv(args.manifest)}
        if set(annotations) != set(manifest):
            raise ValueError("Annotation and manifest item sets differ")
        evaluation_rows = [
            {
                "sample_id": item_id,
                "human_label": row["label"],
                "confidence": row["confidence"],
                "context": row["context"],
            }
            for item_id, row in sorted(annotations.items())
        ]

    checkpoint = torch.load(str(args.checkpoint), map_location="cpu")
    conf = checkpoint["hyper_parameters"]
    base_model = str(args.base_model.resolve())
    conf.model.sense_extractor.transformer_model = base_model
    conf.tokenizer.consec_tokenizer.transformer_model = base_model
    module = ConsecPLModule(conf)
    module.load_state_dict(checkpoint["state_dict"], strict=True)
    device = torch.device(
        "cuda:{}".format(args.device)
        if args.device >= 0 and torch.cuda.is_available()
        else "cpu"
    )
    module.to(device)
    module.eval()
    module.freeze()
    module.sense_extractor.evaluation_mode = True
    tokenizer = hydra.utils.instantiate(module.hparams.tokenizer.consec_tokenizer)

    definitions = [
        ConsecDefinition(definition, sense_key)
        for sense_key, _, definition in PLANE_SENSES
    ]
    samples = []
    rows_by_id = {row["sample_id"]: row for row in evaluation_rows}
    for row in evaluation_rows:
        sample_id = row["sample_id"]
        tokens, target_index = prepare_context(row["context"])
        context = [
            DisambiguationInstance(
                "historical-plane",
                sample_id,
                sample_id if index == target_index else None,
                token,
                "n" if index == target_index else None,
                "plane" if index == target_index else token,
                None,
            )
            for index, token in enumerate(tokens)
        ]
        samples.append(
            ConsecSample(
                sample_id=sample_id,
                position=target_index,
                disambiguation_context=context,
                candidate_definitions=definitions,
                gold_definitions=None,
                context_definitions=[],
                in_context_sample_id2position={sample_id: target_index},
                disambiguation_instance=context[target_index],
                kwargs={},
            )
        )

    prediction_rows = []
    coarse_by_key = {sense_key: coarse for sense_key, coarse, _ in PLANE_SENSES}
    for sample, probabilities in predict(
        module=module,
        tokenizer=tokenizer,
        samples=iter(samples),
        text_encoding_strategy="simple-with-linker",
        token_batch_size=args.token_batch_size,
        progress_bar=True,
    ):
        ranked = sorted(
            enumerate(probabilities), key=lambda item: item[1], reverse=True
        )
        best_index, best_probability = ranked[0]
        second_probability = ranked[1][1]
        sense_key = definitions[best_index].linker
        source = rows_by_id[sample.sample_id]
        result_row = dict(source)
        result_row.update(
            {
                "prediction": coarse_by_key[sense_key],
                "prediction_sensekey": sense_key,
                "probability": "{:.8f}".format(best_probability),
                "margin": "{:.8f}".format(best_probability - second_probability),
            }
        )
        prediction_rows.append(result_row)

    prediction_rows.sort(key=lambda row: row["sample_id"])
    if args.gate_predictions:
        summary = {
            "method": "ConSeC SemCor+WNGT frozen, target-only extraction",
            "number_of_items": len(prediction_rows),
            "original_gate": gate_summary(
                prediction_rows,
                "original_gold_sense",
                args.n_bootstrap,
                args.seed,
            ),
            "post_adjudication_gate": gate_summary(
                [
                    row
                    for row in prediction_rows
                    if row["adjudicated_gold_sense"] != "unclear"
                ],
                "adjudicated_gold_sense",
                args.n_bootstrap,
                args.seed,
            ),
        }
    else:
        summary = compute_summary(prediction_rows)
    summary["checkpoint"] = str(args.checkpoint)
    summary["checkpoint_sha256"] = args.checkpoint_sha256
    summary["consec_commit"] = args.consec_commit

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fieldnames = list(prediction_rows[0])
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(prediction_rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gate-predictions", type=Path)
    parser.add_argument("--consec-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument("--token-batch-size", type=int, default=1024)
    parser.add_argument("--n-bootstrap", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260613)
    parser.add_argument(
        "--checkpoint-sha256",
        default="92421ed245723964db09ce396f19a0d1e55fe4d6e10d5ecb83278d9bc70ce8af",
    )
    parser.add_argument(
        "--consec-commit",
        default="9602b5fd69f57be08a186988d1df34fe4152b63f",
    )
    return parser


if __name__ == "__main__":
    evaluate(build_parser().parse_args())
