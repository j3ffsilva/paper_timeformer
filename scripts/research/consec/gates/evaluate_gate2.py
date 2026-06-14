#!/usr/bin/env python3
"""Evaluate frozen ConSeC on the preregistered Gate 2 subsets.

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


TARGET_SENSES = {
    "graft_nn": (
        (
            "graft%1:08:00::",
            "medical",
            "(surgery) tissue or organ transplanted from a donor to a recipient",
        ),
        (
            "graft%1:04:01::",
            "corruption",
            "the practice of offering something (usually money) in order to gain an illicit advantage",
        ),
        (
            "graft%1:04:00::",
            "grafting_act",
            "the act of grafting something onto something else",
        ),
    ),
    "chairman_nn": (
        (
            "chairman%1:18:01::",
            "presiding_officer",
            "the officer who presides at the meetings of an organization",
        ),
    ),
    "tree_nn": (
        (
            "tree%1:20:00::",
            "plant",
            "a tall perennial woody plant having a main trunk and branches forming a distinct elevated crown",
        ),
        (
            "tree%1:25:00::",
            "diagram",
            "a figure that branches from a single root",
        ),
        (
            "tree%1:18:00::",
            "person",
            "English actor and theatrical producer noted for his lavish productions of Shakespeare (1853-1917)",
        ),
    ),
}

GATE_STRATA = (
    ("graft_nn", "corruption", 0.75),
    ("graft_nn", "medical", 0.75),
    ("tree_nn", "diagram", 0.75),
    ("tree_nn", "plant", 0.90),
)

EXCLUDED_HUMAN_LABELS = {"other", "unclear", "botanical"}


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def prepare_context(context, target):
    lemma = target[:-3] if target.endswith("_nn") else target
    marker = "[{}]".format(lemma)
    tokens = context.split()
    marked = [index for index, token in enumerate(tokens) if token == marker]
    if len(marked) != 1:
        raise ValueError(
            "Expected exactly one {} marker: {}".format(marker, context)
        )
    target_index = marked[0]
    tokens[target_index] = lemma
    return tokens, target_index, lemma


def apply_audit(candidates, annotations, manifest):
    annotation_by_id = {row["item_id"]: row for row in annotations}
    manifest_by_id = {row["item_id"]: row for row in manifest}
    if set(annotation_by_id) != set(manifest_by_id):
        raise ValueError("Annotation and manifest item sets differ")

    candidate_by_id = {row["sample_id"]: row for row in candidates}
    human_by_sample = {}
    disagreements = []
    for item_id in sorted(annotation_by_id):
        annotation = annotation_by_id[item_id]
        public = manifest_by_id[item_id]
        sample_id = public["sample_id"]
        if sample_id not in candidate_by_id:
            raise ValueError("Unknown audited sample: {}".format(sample_id))
        candidate = candidate_by_id[sample_id]
        for field in ("target", "context"):
            if annotation[field] != public[field] or public[field] != candidate[field]:
                raise ValueError(
                    "Audit {} differs in field {}".format(item_id, field)
                )
        label = annotation["label"].strip()
        confidence = annotation["confidence"].strip()
        if not label or not confidence:
            raise ValueError("Incomplete annotation: {}".format(item_id))
        human_by_sample[sample_id] = {
            "item_id": item_id,
            "label": label,
            "confidence": confidence,
            "notes": annotation["notes"],
        }
        if label != candidate["heuristic_sense"]:
            disagreements.append(
                {
                    "item_id": item_id,
                    "sample_id": sample_id,
                    "target": candidate["target"],
                    "heuristic_sense": candidate["heuristic_sense"],
                    "human_label": label,
                    "confidence": confidence,
                    "context": candidate["context"],
                }
            )

    output = []
    for candidate in candidates:
        row = dict(candidate)
        audit = human_by_sample.get(candidate["sample_id"])
        if audit:
            row.update(
                {
                    "audit_status": "human",
                    "audit_item_id": audit["item_id"],
                    "human_label": audit["label"],
                    "human_confidence": audit["confidence"],
                    "human_notes": audit["notes"],
                    "post_audit_gold_sense": audit["label"],
                }
            )
        else:
            row.update(
                {
                    "audit_status": "heuristic_unaudited",
                    "audit_item_id": "",
                    "human_label": "",
                    "human_confidence": "",
                    "human_notes": "",
                    "post_audit_gold_sense": candidate["heuristic_sense"],
                }
            )

        row["post_audit_evaluable"] = str(
            row["post_audit_gold_sense"] not in EXCLUDED_HUMAN_LABELS
        )
        output.append(row)

    audited = len(human_by_sample)
    return output, {
        "audited_items": audited,
        "agreements": audited - len(disagreements),
        "agreement_rate": (
            (audited - len(disagreements)) / audited if audited else None
        ),
        "disagreements": disagreements,
        "policy": (
            "Human labels replace heuristic labels only for audited items. "
            "Human other, unclear, and botanical labels are excluded from "
            "the confirmatory gate; no label is silently corrected."
        ),
    }


def bootstrap_accuracy(correct, n_bootstrap, seed):
    values = np.asarray(correct, dtype=np.float32)
    if not len(values):
        raise ValueError("Cannot bootstrap an empty stratum")
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
    metrics = []
    arrays = []
    checks = {}
    for offset, (target, label, threshold) in enumerate(GATE_STRATA):
        stratum = [
            row
            for row in rows
            if row["target"] == target and row[label_field] == label
        ]
        correct = [row["prediction"] == label for row in stratum]
        metric = bootstrap_accuracy(correct, n_bootstrap, seed + offset)
        metric.update(
            {
                "target": target,
                "gold_sense": label,
                "threshold": threshold,
                "passed": metric["accuracy"] >= threshold,
            }
        )
        metrics.append(metric)
        arrays.append(np.asarray(correct, dtype=np.float32))
        checks["{}_accuracy_at_least_{:.2f}".format(label, threshold)] = metric[
            "passed"
        ]

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
    checks["macro_accuracy_ci_above_one_third"] = macro["ci_95_low"] > (
        1.0 / 3.0
    )
    return {
        "label_field": label_field,
        "metrics": metrics,
        "macro_accuracy": macro,
        "checks": checks,
        "passed": all(checks.values()),
    }


def diagnostic_summary(rows):
    chairman = [
        row
        for row in rows
        if row["target"] == "chairman_nn"
        and row["heuristic_sense"] == "monosemous_control"
    ]
    botanical = [
        row
        for row in rows
        if row["target"] == "graft_nn"
        and row["heuristic_sense"] == "botanical_inventory_gap"
    ]
    return {
        "chairman_monosemous_control": {
            "n": len(chairman),
            "prediction_counts": dict(
                sorted(Counter(row["prediction"] for row in chairman).items())
            ),
            "note": "Coverage control only; excluded from GO/NO-GO.",
        },
        "graft_botanical_inventory_gap": {
            "n": len(botanical),
            "prediction_counts": dict(
                sorted(Counter(row["prediction"] for row in botanical).items())
            ),
            "note": (
                "Diagnostic only. The botanical object sense is absent from "
                "the simple noun graft WordNet inventory used by ConSeC."
            ),
        },
    }


def evaluate(args):
    candidates = read_csv(args.candidates)
    evaluation_rows, audit_summary = apply_audit(
        candidates,
        read_csv(args.annotations),
        read_csv(args.manifest),
    )

    repo = args.consec_repo.resolve()
    sys.path.insert(0, str(repo))

    import hydra
    import torch

    from src.consec_dataset import ConsecDefinition, ConsecSample
    from src.disambiguation_corpora import DisambiguationInstance
    from src.pl_modules import ConsecPLModule
    from src.scripts.model.predict import predict

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

    definitions_by_target = {
        target: [
            ConsecDefinition(definition, sense_key)
            for sense_key, _, definition in senses
        ]
        for target, senses in TARGET_SENSES.items()
    }
    coarse_by_target = {
        target: {
            sense_key: coarse for sense_key, coarse, _ in senses
        }
        for target, senses in TARGET_SENSES.items()
    }
    samples = []
    rows_by_id = {row["sample_id"]: row for row in evaluation_rows}
    for row in evaluation_rows:
        sample_id = row["sample_id"]
        target = row["target"]
        tokens, target_index, lemma = prepare_context(row["context"], target)
        context = [
            DisambiguationInstance(
                "historical-gate2",
                sample_id,
                sample_id if index == target_index else None,
                token,
                "n" if index == target_index else None,
                lemma if index == target_index else token,
                None,
            )
            for index, token in enumerate(tokens)
        ]
        samples.append(
            ConsecSample(
                sample_id=sample_id,
                position=target_index,
                disambiguation_context=context,
                candidate_definitions=definitions_by_target[target],
                gold_definitions=None,
                context_definitions=[],
                in_context_sample_id2position={sample_id: target_index},
                disambiguation_instance=context[target_index],
                kwargs={"target": target},
            )
        )

    prediction_rows = []
    for sample, probabilities in predict(
        module=module,
        tokenizer=tokenizer,
        samples=iter(samples),
        text_encoding_strategy="simple-with-linker",
        token_batch_size=args.token_batch_size,
        progress_bar=True,
    ):
        source = rows_by_id[sample.sample_id]
        target = source["target"]
        definitions = definitions_by_target[target]
        ranked = sorted(
            enumerate(probabilities), key=lambda item: item[1], reverse=True
        )
        best_index, best_probability = ranked[0]
        second_probability = ranked[1][1] if len(ranked) > 1 else 0.0
        sense_key = definitions[best_index].linker
        result_row = dict(source)
        result_row.update(
            {
                "prediction": coarse_by_target[target][sense_key],
                "prediction_sensekey": sense_key,
                "probability": "{:.8f}".format(best_probability),
                "margin": "{:.8f}".format(
                    best_probability - second_probability
                ),
            }
        )
        prediction_rows.append(result_row)

    prediction_rows.sort(key=lambda row: row["sample_id"])
    if len(prediction_rows) != len(evaluation_rows):
        raise ValueError(
            "Expected {} predictions, found {}".format(
                len(evaluation_rows), len(prediction_rows)
            )
        )

    post_audit_rows = [
        row
        for row in prediction_rows
        if row["post_audit_evaluable"] == "True"
    ]
    original_gate = gate_summary(
        prediction_rows,
        "heuristic_sense",
        args.n_bootstrap,
        args.seed,
    )
    post_audit_gate = gate_summary(
        post_audit_rows,
        "post_audit_gold_sense",
        args.n_bootstrap,
        args.seed,
    )
    summary = {
        "method": "ConSeC SemCor+WNGT frozen, target-only extraction",
        "number_of_items": len(prediction_rows),
        "audit": audit_summary,
        "original_preregistered_gate": original_gate,
        "post_audit_gate": post_audit_gate,
        "decision": "GO" if post_audit_gate["passed"] else "NO-GO",
        "diagnostics": diagnostic_summary(prediction_rows),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": args.checkpoint_sha256,
        "consec_commit": args.consec_commit,
        "seed": args.seed,
        "n_bootstrap": args.n_bootstrap,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0]))
        writer.writeheader()
        writer.writerows(prediction_rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
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
