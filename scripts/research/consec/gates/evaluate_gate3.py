#!/usr/bin/env python3
"""Run frozen ConSeC and evaluate preregistered temporal sense distributions.

Compatible with Python 3.7 for the official ConSeC environment.
"""

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def read_csv(path, delimiter=","):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def prepare_context(context, target):
    lemma = target[:-3]
    marker = "[{}]".format(lemma)
    tokens = context.split()
    positions = [index for index, token in enumerate(tokens) if token == marker]
    if len(positions) != 1:
        raise ValueError("Expected one {} marker".format(marker))
    tokens[positions[0]] = lemma
    return tokens, positions[0], lemma


def jensen_shannon(values_a, values_b):
    a = np.asarray(values_a, dtype=np.float64)
    b = np.asarray(values_b, dtype=np.float64)
    a = a / a.sum()
    b = b / b.sum()
    midpoint = 0.5 * (a + b)

    def kl_divergence(values, reference):
        selected = values > 0
        return float(
            np.sum(values[selected] * np.log(values[selected] / reference[selected]))
        )

    return 0.5 * (
        kl_divergence(a, midpoint) + kl_divergence(b, midpoint)
    )


def rankdata(values):
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def spearman(values_a, values_b):
    ranks_a = rankdata(values_a)
    ranks_b = rankdata(values_b)
    centered_a = ranks_a - ranks_a.mean()
    centered_b = ranks_b - ranks_b.mean()
    denominator = np.sqrt(
        np.sum(centered_a ** 2) * np.sum(centered_b ** 2)
    )
    return float(np.sum(centered_a * centered_b) / denominator)


def partial_spearman(values_a, values_b, controls):
    ranks_a = rankdata(values_a)
    ranks_b = rankdata(values_b)
    control_ranks = rankdata(controls)
    design = np.column_stack(
        [np.ones(len(control_ranks), dtype=np.float64), control_ranks]
    )
    residual_a = ranks_a - design.dot(
        np.linalg.lstsq(design, ranks_a, rcond=None)[0]
    )
    residual_b = ranks_b - design.dot(
        np.linalg.lstsq(design, ranks_b, rcond=None)[0]
    )
    denominator = np.sqrt(
        np.sum(residual_a ** 2) * np.sum(residual_b ** 2)
    )
    return float(np.sum(residual_a * residual_b) / denominator)


def roc_auc(binary, scores):
    binary = np.asarray(binary)
    ranks = rankdata(scores)
    n_positive = int(binary.sum())
    n_negative = len(binary) - n_positive
    rank_sum = float(ranks[binary == 1].sum())
    return (
        rank_sum - n_positive * (n_positive + 1) / 2.0
    ) / (n_positive * n_negative)


def average_precision(binary, scores):
    binary = np.asarray(binary)
    order = np.argsort(-np.asarray(scores), kind="mergesort")
    sorted_binary = binary[order]
    cumulative = np.cumsum(sorted_binary)
    precision = cumulative / np.arange(1, len(binary) + 1)
    return float(np.sum(precision * sorted_binary) / binary.sum())


def correlation_summary(rows, n_bootstrap, n_permutations, seed):
    scores = np.asarray([float(row["jsd"]) for row in rows])
    gold = np.asarray([float(row["graded"]) for row in rows])
    binary = np.asarray([int(row["binary"]) for row in rows])
    rho = spearman(scores, gold)
    rng = np.random.default_rng(seed)

    bootstrap = []
    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(rows), size=len(rows))
        value = spearman(scores[indices], gold[indices])
        if not math.isnan(value):
            bootstrap.append(value)
    low, high = np.quantile(bootstrap, [0.025, 0.975])

    extreme = 0
    for _ in range(n_permutations):
        permuted = rng.permutation(gold)
        value = spearman(scores, permuted)
        if abs(value) >= abs(rho):
            extreme += 1
    permutation_p = (extreme + 1) / (n_permutations + 1)
    return {
        "n_targets": len(rows),
        "spearman": rho,
        "bootstrap_ci_95_low": float(low),
        "bootstrap_ci_95_high": float(high),
        "permutation_p_two_sided": float(permutation_p),
        "roc_auc": float(roc_auc(binary, scores)),
        "average_precision": float(average_precision(binary, scores)),
        "passed": rho > 0 and permutation_p < 0.05,
    }


def inventory_control_summary(rows, n_permutations, seed):
    scores = np.asarray([float(row["jsd"]) for row in rows])
    gold = np.asarray([float(row["graded"]) for row in rows])
    n_senses = np.asarray([int(row["n_senses"]) for row in rows])
    score_inventory_rho = spearman(scores, n_senses)
    partial_rho = partial_spearman(scores, gold, n_senses)
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(n_permutations):
        permuted = rng.permutation(gold)
        value = partial_spearman(scores, permuted, n_senses)
        if abs(value) >= abs(partial_rho):
            extreme += 1
    return {
        "score_vs_n_senses_spearman": score_inventory_rho,
        "gold_partial_spearman_controlling_n_senses": partial_rho,
        "partial_permutation_p_two_sided": (
            (extreme + 1) / (n_permutations + 1)
        ),
    }


def load_cached_predictions(path, valid_sample_ids):
    if not path.exists():
        return []
    rows = read_csv(path)
    cached_ids = [row["sample_id"] for row in rows]
    if len(cached_ids) != len(set(cached_ids)):
        raise ValueError("Duplicate sample IDs in prediction cache")
    unknown = sorted(set(cached_ids) - valid_sample_ids)
    if unknown:
        raise ValueError("Prediction cache contains unknown samples: {}".format(unknown[:5]))
    return rows


def seed_prediction_cache(output_path, source_paths, valid_sample_ids):
    if output_path.exists() or not source_paths:
        return 0
    rows_by_id = {}
    for path in source_paths:
        for row in read_csv(path):
            sample_id = row["sample_id"]
            if sample_id in valid_sample_ids:
                rows_by_id.setdefault(sample_id, row)
    rows = sorted(rows_by_id.values(), key=lambda row: row["sample_id"])
    if not rows:
        return 0
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def evaluate(args):
    occurrence_rows = read_csv(args.occurrences)
    inventory_rows = read_csv(args.sense_inventory)
    inventory_by_target = defaultdict(list)
    for row in inventory_rows:
        inventory_by_target[row["target"]].append(row)

    definitions_by_target_rows = {}
    for target, rows in inventory_by_target.items():
        definitions_by_target_rows[target] = rows

    source_by_id = {row["sample_id"]: row for row in occurrence_rows}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "occurrence_predictions.csv"
    reused_predictions = seed_prediction_cache(
        predictions_path,
        args.reuse_predictions,
        set(source_by_id),
    )
    prediction_rows = (
        load_cached_predictions(predictions_path, set(source_by_id))
        if args.resume
        else []
    )
    cached_ids = {row["sample_id"] for row in prediction_rows}
    pending_rows = [
        row for row in occurrence_rows if row["sample_id"] not in cached_ids
    ]

    if pending_rows:
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
        tokenizer = hydra.utils.instantiate(
            module.hparams.tokenizer.consec_tokenizer
        )
        definitions_by_target = {
            target: [
                ConsecDefinition(row["definition"], row["sensekey"])
                for row in rows
            ]
            for target, rows in definitions_by_target_rows.items()
        }
        samples = []
        for row in pending_rows:
            target = row["target"]
            tokens, target_index, lemma = prepare_context(row["context"], target)
            context = [
                DisambiguationInstance(
                    "historical-gate3",
                    row["sample_id"],
                    row["sample_id"] if index == target_index else None,
                    token,
                    "n" if target.endswith("_nn") and index == target_index
                    else "v" if index == target_index
                    else None,
                    lemma if index == target_index else token,
                    None,
                )
                for index, token in enumerate(tokens)
            ]
            samples.append(
                ConsecSample(
                    sample_id=row["sample_id"],
                    position=target_index,
                    disambiguation_context=context,
                    candidate_definitions=definitions_by_target[target],
                    gold_definitions=None,
                    context_definitions=[],
                    in_context_sample_id2position={
                        row["sample_id"]: target_index
                    },
                    disambiguation_instance=context[target_index],
                    kwargs={"target": target},
                )
            )

        append = args.resume and predictions_path.exists()
        with predictions_path.open(
            "a" if append else "w", newline="", encoding="utf-8"
        ) as handle:
            writer = None
            for sample, probabilities in predict(
                module=module,
                tokenizer=tokenizer,
                samples=iter(samples),
                text_encoding_strategy="simple-with-linker",
                token_batch_size=args.token_batch_size,
                progress_bar=True,
            ):
                source = source_by_id[sample.sample_id]
                target = source["target"]
                definitions = definitions_by_target[target]
                probabilities = np.asarray(probabilities, dtype=np.float64)
                best = int(np.argmax(probabilities))
                output = dict(source)
                output.update(
                    {
                        "prediction_sensekey": definitions[best].linker,
                        "prediction_probability": "{:.8f}".format(
                            probabilities[best]
                        ),
                        "sense_probabilities": json.dumps(
                            {
                                definition.linker: float(probability)
                                for definition, probability in zip(
                                    definitions, probabilities
                                )
                            },
                            sort_keys=True,
                        ),
                    }
                )
                if writer is None:
                    writer = csv.DictWriter(handle, fieldnames=list(output))
                    if not append:
                        writer.writeheader()
                writer.writerow(output)
                handle.flush()

    prediction_rows = load_cached_predictions(predictions_path, set(source_by_id))
    if len(prediction_rows) != len(occurrence_rows):
        raise ValueError(
            "Prediction cache incomplete: {} of {}".format(
                len(prediction_rows), len(occurrence_rows)
            )
        )

    probability_sums = {}
    counts = defaultdict(int)
    for row in prediction_rows:
        target = row["target"]
        probabilities_by_key = json.loads(row["sense_probabilities"])
        sensekeys = [
            item["sensekey"] for item in definitions_by_target_rows[target]
        ]
        probabilities = np.asarray(
            [probabilities_by_key[sensekey] for sensekey in sensekeys],
            dtype=np.float64,
        )
        key = (target, row["period"])
        if key not in probability_sums:
            probability_sums[key] = np.zeros(len(probabilities), dtype=np.float64)
        probability_sums[key] += probabilities
        counts[key] += 1

    truth = {
        row["target"]: row for row in read_csv(args.truth, delimiter="\t")
    }
    target_rows = []
    roles = {}
    confidence = {}
    for row in occurrence_rows:
        roles[row["target"]] = row["role"]
        confidence[row["target"]] = row["coverage_confidence"]
    for target in sorted(roles):
        definitions = definitions_by_target_rows[target]
        d0 = probability_sums[(target, "1810-1860")]
        d1 = probability_sums[(target, "1960-2010")]
        d0 = d0 / d0.sum()
        d1 = d1 / d1.sum()
        target_rows.append(
            {
                "target": target,
                "role": roles[target],
                "coverage_confidence": confidence[target],
                "n_d0": counts[(target, "1810-1860")],
                "n_d1": counts[(target, "1960-2010")],
                "n_senses": len(definitions),
                "jsd": jensen_shannon(d0, d1),
                "binary": truth[target]["binary"],
                "graded": truth[target]["graded"],
                "d0_distribution": json.dumps(
                    {
                        definition["sensekey"]: float(value)
                        for definition, value in zip(definitions, d0)
                    },
                    sort_keys=True,
                ),
                "d1_distribution": json.dumps(
                    {
                        definition["sensekey"]: float(value)
                        for definition, value in zip(definitions, d1)
                    },
                    sort_keys=True,
                ),
            }
        )

    confirmatory = [
        row for row in target_rows if row["role"] == "confirmatory"
    ]
    high_confidence = [
        row
        for row in confirmatory
        if row["coverage_confidence"] == "high"
    ]
    primary = correlation_summary(
        confirmatory, args.n_bootstrap, args.n_permutations, args.seed
    )
    sensitivity = correlation_summary(
        high_confidence,
        args.n_bootstrap,
        args.n_permutations,
        args.seed + 1,
    )
    inventory_control = inventory_control_summary(
        confirmatory, args.n_permutations, args.seed + 2
    )
    summary = {
        "method": "Frozen ConSeC temporal soft sense-distribution JSD",
        "number_of_occurrences": len(prediction_rows),
        "number_of_targets": len(target_rows),
        "reused_predictions": reused_predictions,
        "primary_confirmatory": primary,
        "high_confidence_sensitivity": sensitivity,
        "inventory_size_control": inventory_control,
        "decision": "GO" if primary["passed"] else "NO-GO",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": args.checkpoint_sha256,
        "consec_commit": args.consec_commit,
        "seed": args.seed,
        "n_bootstrap": args.n_bootstrap,
        "n_permutations": args.n_permutations,
    }

    with (args.output_dir / "target_scores.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(target_rows[0]))
        writer.writeheader()
        writer.writerows(target_rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--occurrences", type=Path, required=True)
    parser.add_argument("--sense-inventory", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--consec-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument("--token-batch-size", type=int, default=1024)
    parser.add_argument("--n-bootstrap", type=int, default=20000)
    parser.add_argument("--n-permutations", type=int, default=20000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--reuse-predictions",
        action="append",
        type=Path,
        default=[],
    )
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
