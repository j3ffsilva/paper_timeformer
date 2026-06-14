#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.research.common.cloze import (  # noqa: E402
    SENSE_KEYWORDS,
    high_confidence_sense,
)
from timeformers.bert_continual import strip_pos_suffix  # noqa: E402
from timeformers.real_corpus import read_period_corpora  # noqa: E402


PLANE_SENSES = {
    "plane%1:06:00::": {
        "coarse_sense": "tool",
        "synset": "plane.n.05",
        "definition": "a carpenter's hand tool for smoothing or shaping wood",
    },
    "plane%1:06:01::": {
        "coarse_sense": "aircraft",
        "synset": "airplane.n.01",
        "definition": "an aircraft with fixed wings powered by propellers or jets",
    },
    "plane%1:06:02::": {
        "coarse_sense": "tool",
        "synset": "plane.n.04",
        "definition": "a power tool for smoothing or shaping wood",
    },
    "plane%1:25:00::": {
        "coarse_sense": "geometry",
        "synset": "plane.n.02",
        "definition": "an unbounded two-dimensional mathematical shape",
    },
    "plane%1:26:00::": {
        "coarse_sense": "other",
        "synset": "plane.n.03",
        "definition": "a level of existence or development",
    },
}

ANCHOR_PHRASES = ("plate figure represent an inclined plane",)

# Published LMMS-SP WSD profile for bert-large-cased, ordered last layer first.
LMMS_BERT_LARGE_WEIGHTS = np.asarray(
    [
        0.00090,
        0.53920,
        0.36144,
        0.05975,
        0.01473,
        0.00542,
        0.00662,
        0.00542,
        0.00049,
        0.00297,
        0.00163,
        0.00049,
        0.00033,
        0.00040,
        0.00015,
        0.00004,
        0.00001,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    dtype=np.float32,
)


@dataclass(frozen=True)
class Occurrence:
    corpus: str
    index: int
    tokens: list[str]
    target_index: int
    gold_sense: str
    display: str
    anchor: bool


def centered_window(
    document: list[str],
    target_index: int,
    content_len: int,
) -> tuple[list[str], int]:
    left_budget = content_len // 2
    start = max(0, min(target_index - left_budget, len(document) - content_len))
    end = min(len(document), start + content_len)
    return list(document[start:end]), target_index - start


def collect_occurrences(
    corpus,
    *,
    target: str = "plane_nn",
    label_content_len: int = 30,
    encoder_content_len: int = 126,
) -> list[Occurrence]:
    occurrences = []
    for document in corpus.documents:
        for target_index, token in enumerate(document):
            if token != target:
                continue
            label_tokens, _ = centered_window(
                document,
                target_index,
                label_content_len,
            )
            encoder_tokens, relative_index = centered_window(
                document,
                target_index,
                encoder_content_len,
            )
            normalized = [strip_pos_suffix(token) for token in encoder_tokens]
            display_tokens = list(normalized)
            display_tokens[relative_index] = "[plane]"
            display = " ".join(display_tokens)
            occurrences.append(
                Occurrence(
                    corpus=corpus.period,
                    index=len(occurrences),
                    tokens=normalized,
                    target_index=relative_index,
                    gold_sense=high_confidence_sense(label_tokens),
                    display=display,
                    anchor=any(phrase in " ".join(normalized) for phrase in ANCHOR_PHRASES),
                )
            )
    return occurrences


def load_plane_vectors(path: Path) -> dict[str, np.ndarray]:
    vectors = {}

    def consume(lines) -> None:
        for raw_line in lines:
            line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
            if not line.startswith("plane%1:"):
                continue
            fields = line.split()
            sensekey = fields[0]
            if sensekey in PLANE_SENSES:
                vectors[sensekey] = np.asarray(fields[1:], dtype=np.float32)

    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            members = [
                name
                for name in archive.namelist()
                if name.endswith((".txt", ".vectors"))
                and "lmms-sp-wsd" in name
                and "synsets" not in name
            ]
            if len(members) != 1:
                raise ValueError(
                    f"Expected one LMMS WSD sensekey vector file, found {members}"
                )
            with archive.open(members[0]) as handle:
                consume(handle)
    else:
        with path.open("r", encoding="utf-8") as handle:
            consume(handle)

    missing = sorted(set(PLANE_SENSES) - set(vectors))
    if missing:
        raise ValueError(f"LMMS vectors are missing plane senses: {missing}")
    dimensions = {vector.shape for vector in vectors.values()}
    if len(dimensions) != 1:
        raise ValueError(f"Inconsistent vector dimensions: {dimensions}")
    return {
        sensekey: vector / np.linalg.norm(vector)
        for sensekey, vector in vectors.items()
    }


def load_layer_weights(path: Path | None) -> np.ndarray:
    weights = (
        LMMS_BERT_LARGE_WEIGHTS.copy()
        if path is None
        else np.asarray(
            [
                float(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ],
            dtype=np.float32,
        )
    )
    if not np.isclose(weights.sum(), 1.0, atol=1e-3):
        raise ValueError(f"Layer weights must sum to one, got {weights.sum():.6f}")
    return weights


def combine_hidden_states(
    hidden_states: tuple[torch.Tensor, ...],
    weights: torch.Tensor,
) -> torch.Tensor:
    if len(hidden_states) != len(weights):
        raise ValueError(
            f"Expected {len(weights)} hidden states, got {len(hidden_states)}"
        )
    reversed_states = torch.stack(tuple(reversed(hidden_states)), dim=0)
    return torch.einsum("l,lbsd->bsd", weights, reversed_states)


def pool_target_subwords(
    embeddings: torch.Tensor,
    word_ids: list[int | None],
    target_index: int,
) -> torch.Tensor:
    positions = [
        index
        for index, word_id in enumerate(word_ids)
        if word_id == target_index
    ]
    if not positions:
        raise ValueError("Target word was truncated or could not be aligned")
    return embeddings[positions].mean(dim=0)


@torch.no_grad()
def encode_occurrences(
    occurrences: list[Occurrence],
    model,
    tokenizer,
    layer_weights: np.ndarray,
    *,
    batch_size: int,
    max_length: int,
    device: str,
) -> np.ndarray:
    model.eval()
    model.to(device)
    weights = torch.as_tensor(layer_weights, device=device)
    rows = []
    for start in range(0, len(occurrences), batch_size):
        batch = occurrences[start : start + batch_size]
        encoded = tokenizer(
            [row.tokens for row in batch],
            is_split_into_words=True,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        word_ids = [encoded.word_ids(batch_index=index) for index in range(len(batch))]
        model_inputs = {
            key: value.to(device)
            for key, value in encoded.items()
            if key in {"input_ids", "attention_mask", "token_type_ids"}
        }
        output = model(**model_inputs, output_hidden_states=True, return_dict=True)
        combined = combine_hidden_states(output.hidden_states, weights)
        for batch_index, row in enumerate(batch):
            vector = pool_target_subwords(
                combined[batch_index],
                word_ids[batch_index],
                row.target_index,
            )
            vector = torch.nn.functional.normalize(vector.float(), dim=0)
            rows.append(vector.cpu().numpy())
    return np.stack(rows)


def classify(
    occurrence_vectors: np.ndarray,
    sense_vectors: dict[str, np.ndarray],
) -> list[dict]:
    sensekeys = list(sense_vectors)
    matrix = np.stack([sense_vectors[sensekey] for sensekey in sensekeys])
    scores = occurrence_vectors @ matrix.T
    ordering = np.argsort(scores, axis=1)
    rows = []
    for index in range(len(occurrence_vectors)):
        best = int(ordering[index, -1])
        second = int(ordering[index, -2])
        sensekey = sensekeys[best]
        rows.append(
            {
                "prediction_sensekey": sensekey,
                "prediction": PLANE_SENSES[sensekey]["coarse_sense"],
                "margin": float(scores[index, best] - scores[index, second]),
                **{
                    f"score_{key}": float(scores[index, sense_index])
                    for sense_index, key in enumerate(sensekeys)
                },
            }
        )
    return rows


def bootstrap_accuracy(
    correctness: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, float]:
    if len(correctness) == 0:
        raise ValueError("Cannot bootstrap an empty subset")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(correctness), size=(n_bootstrap, len(correctness)))
    samples = correctness[indices].mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return {
        "n": int(len(correctness)),
        "accuracy": float(correctness.mean()),
        "ci_95_low": float(low),
        "ci_95_high": float(high),
    }


def summarize(
    rows: list[dict],
    *,
    n_bootstrap: int,
    seed: int,
) -> list[dict]:
    summaries = []
    keys = sorted(
        {
            (row["corpus"], row["gold_sense"])
            for row in rows
            if row["gold_sense"] != "unlabeled"
        }
    )
    for offset, (corpus, gold_sense) in enumerate(keys):
        subset = [
            row
            for row in rows
            if row["corpus"] == corpus and row["gold_sense"] == gold_sense
        ]
        correctness = np.asarray(
            [row["prediction"] == gold_sense for row in subset],
            dtype=np.float32,
        )
        summaries.append(
            {
                "corpus": corpus,
                "gold_sense": gold_sense,
                **bootstrap_accuracy(
                    correctness,
                    n_bootstrap=n_bootstrap,
                    seed=seed + offset,
                ),
            }
        )
    return summaries


def bootstrap_macro_accuracy(
    rows: list[dict],
    *,
    periods: list[str],
    n_bootstrap: int,
    seed: int,
) -> dict[str, float]:
    d0, d1 = periods
    subsets = [
        [
            row["prediction"] == "geometry"
            for row in rows
            if row["corpus"] == d0 and row["gold_sense"] == "geometry"
        ],
        [
            row["prediction"] == "tool"
            for row in rows
            if row["corpus"] == d0 and row["gold_sense"] == "tool"
        ],
        [
            row["prediction"] == "aircraft"
            for row in rows
            if row["corpus"] == d1 and row["gold_sense"] == "aircraft"
        ],
    ]
    if any(not subset for subset in subsets):
        raise ValueError("Macro accuracy requires all three predefined subsets")
    arrays = [np.asarray(subset, dtype=np.float32) for subset in subsets]
    rng = np.random.default_rng(seed)
    samples = np.stack(
        [
            values[
                rng.integers(0, len(values), size=(n_bootstrap, len(values)))
            ].mean(axis=1)
            for values in arrays
        ],
        axis=1,
    ).mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return {
        "accuracy": float(np.mean([values.mean() for values in arrays])),
        "ci_95_low": float(low),
        "ci_95_high": float(high),
        "chance_baseline": 1.0 / 3.0,
    }


def gate_decision(
    summaries: list[dict],
    rows: list[dict],
    macro_accuracy: dict[str, float],
) -> dict:
    by_key = {
        (row["corpus"], row["gold_sense"]): row
        for row in summaries
    }
    periods = sorted({row["corpus"] for row in rows})
    if len(periods) != 2:
        raise ValueError(f"Gate 1 requires exactly two periods, got {periods}")
    d0, d1 = periods
    geometry = by_key.get((d0, "geometry"))
    tool = by_key.get((d0, "tool"))
    aircraft = by_key.get((d1, "aircraft"))
    anchors = [row for row in rows if row["corpus"] == d0 and row["anchor"]]
    checks = {
        "macro_accuracy_ci_above_one_third": bool(
            macro_accuracy["ci_95_low"] > (1.0 / 3.0)
        ),
        "d0_geometry_accuracy_at_least_0_75": bool(
            geometry and geometry["accuracy"] >= 0.75
        ),
        "d1_aircraft_accuracy_at_least_0_80": bool(
            aircraft and aircraft["accuracy"] >= 0.80
        ),
        "d0_tool_ci_above_one_third": bool(
            tool and tool["ci_95_low"] > (1.0 / 3.0)
        ),
        "historical_anchor_is_geometry": bool(
            anchors and all(row["prediction"] == "geometry" for row in anchors)
        ),
    }
    return {
        "periods": {"d0": d0, "d1": d1},
        "checks": checks,
        "passed": all(checks.values()),
        "n_historical_anchors": len(anchors),
    }


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen LMMS-SP WSD on predefined plane subsets."
    )
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--vectors", type=Path, required=True)
    parser.add_argument(
        "--weights",
        type=Path,
        help="Optional override for the published LMMS BERT-large WSD weights.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="bert-large-cased")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--n-bootstrap", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260613)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    corpora = read_period_corpora(args.corpus_dir)
    occurrences = [
        occurrence
        for corpus in corpora
        for occurrence in collect_occurrences(
            corpus,
            encoder_content_len=max(2, (args.max_length - 2) // 2),
        )
    ]
    sense_vectors = load_plane_vectors(args.vectors)
    layer_weights = load_layer_weights(args.weights)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModel.from_pretrained(args.model_name)
    vectors = encode_occurrences(
        occurrences,
        model,
        tokenizer,
        layer_weights,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
    )
    predictions = classify(vectors, sense_vectors)
    rows = [
        {
            "corpus": occurrence.corpus,
            "occurrence_index": occurrence.index,
            "gold_sense": occurrence.gold_sense,
            "anchor": occurrence.anchor,
            **prediction,
            "context": occurrence.display,
        }
        for occurrence, prediction in zip(occurrences, predictions)
    ]
    summaries = summarize(
        rows,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    periods = sorted({row["corpus"] for row in rows})
    macro_accuracy = bootstrap_macro_accuracy(
        rows,
        periods=periods,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed + 100,
    )
    decision = gate_decision(summaries, rows, macro_accuracy)
    result = {
        "method": "LMMS-SP WSD with frozen bert-large-cased",
        "model_name": args.model_name,
        "sense_inventory": PLANE_SENSES,
        "n_occurrences": len(rows),
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "metrics": summaries,
        "macro_accuracy": macro_accuracy,
        "gate_1": decision,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.output_dir / "occurrence_predictions.csv")
    write_csv(summaries, args.output_dir / "metrics.csv")
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
