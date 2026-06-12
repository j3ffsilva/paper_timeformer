#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from gensim.models import Word2Vec
from scipy.linalg import orthogonal_procrustes
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


DEFAULT_AUDIT_TARGETS = [
    "plane_nn",
    "graft_nn",
    "chairman_nn",
    "tree_nn",
    "attack_nn",
    "record_nn",
    "stab_nn",
]


def read_sentences(path: Path, *, max_sentences: int | None = None) -> list[list[str]]:
    sentences = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if max_sentences is not None and index >= max_sentences:
                break
            tokens = line.strip().split()
            if tokens:
                sentences.append(tokens)
    return sentences


def read_list(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_truth(path: Path) -> dict[str, dict[str, float]]:
    rows = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            rows[row["target"]] = {
                "binary": float(row["binary"]),
                "graded": float(row["graded"]),
            }
    return rows


def train_word2vec(
    sentences: list[list[str]],
    *,
    vector_size: int,
    window: int,
    min_count: int,
    workers: int,
    epochs: int,
    seed: int,
    sg: int,
) -> Word2Vec:
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        epochs=epochs,
        seed=seed,
        sg=sg,
        negative=5,
        sample=1e-3,
    )
    model.wv.fill_norms(force=True)
    return model


def normalized_matrix(model: Word2Vec, vocab: list[str]) -> np.ndarray:
    matrix = np.vstack([model.wv[token] for token in vocab]).astype(np.float64)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-12, None)


def rank_descending(values: np.ndarray) -> np.ndarray:
    order = np.argsort(-values)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, values.size + 1)
    return ranks


def standardize(values: np.ndarray) -> np.ndarray:
    std = float(values.std())
    if std < 1e-12:
        return values * 0.0
    return (values - float(values.mean())) / std


def best_f1(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    best = {"f1": 0.0, "threshold": float("nan")}
    for threshold in sorted(set(float(value) for value in y_score)):
        predicted = (y_score >= threshold).astype(int)
        score = f1_score(y_true, predicted, zero_division=0)
        if score > best["f1"]:
            best = {"f1": float(score), "threshold": threshold}
    return best


def evaluate(truth: dict[str, dict[str, float]], scores: dict[str, float]) -> tuple[dict, list[dict]]:
    shared = sorted(set(truth) & set(scores))
    if not shared:
        raise ValueError("No overlapping targets between truth and scores")
    y_score = np.array([scores[target] for target in shared], dtype=float)
    y_binary = np.array([truth[target]["binary"] for target in shared], dtype=int)
    y_graded = np.array([truth[target]["graded"] for target in shared], dtype=float)
    rho, rho_p = spearmanr(y_graded, y_score)
    metrics = {
        "n_targets": len(shared),
        "spearman_graded": float(rho),
        "spearman_graded_p": float(rho_p),
        "roc_auc_binary": float(roc_auc_score(y_binary, y_score)),
        "average_precision_binary": float(average_precision_score(y_binary, y_score)),
        "best_f1_binary": best_f1(y_binary, y_score),
    }
    joined = [
        {
            "target": target,
            "predicted_score": scores[target],
            "binary": truth[target]["binary"],
            "graded": truth[target]["graded"],
        }
        for target in shared
    ]
    joined.sort(key=lambda row: row["predicted_score"], reverse=True)
    return metrics, joined


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def top_rows(rows: list[dict], key: str, n: int, *, reverse: bool = True) -> list[dict]:
    return sorted(rows, key=lambda row: row[key], reverse=reverse)[:n]


def format_table(rows: list[dict], value_key: str) -> list[str]:
    lines = ["| reference | value | rank D0 | rank D1 |", "|---|---:|---:|---:|"]
    for row in rows:
        lines.append(
            f"| `{row['reference']}` | {row[value_key]:.3f} | "
            f"{row['rank_d0']} | {row['rank_d1']} |"
        )
    return lines


def write_report(
    path: Path,
    *,
    rows_by_target: dict[str, list[dict]],
    audit_targets: Iterable[str],
    top_k: int,
    salience_rank: int,
    metrics: dict,
    config: dict,
) -> None:
    lines = [
        "# Hamilton 2016 word2vec baseline",
        "",
        "This baseline trains independent static word2vec models for D0 and D1,",
        "aligns D1 to D0 with orthogonal Procrustes, and applies the same",
        "temporal-neighborhood reporting protocol used for TimeFormer.",
        "",
        "## Configuration",
        "",
    ]
    for key, value in config.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## SemEval validation",
            "",
            f"- Spearman graded: `{metrics['spearman_graded']:.3f}`",
            f"- ROC-AUC binary: `{metrics['roc_auc_binary']:.3f}`",
            f"- Average precision: `{metrics['average_precision_binary']:.3f}`",
            "",
            "## Definition",
            "",
            "```text",
            "E0 = word2vec(D0)",
            "E1 = word2vec(D1) @ R, where R is orthogonal Procrustes",
            "r_t(w)[v] = cos(E_t[w], E_t[v])",
            "z_t(w)[v] = standardize_v(r_t(w)[v])",
            "delta_z(w)[v] = z_1(w)[v] - z_0(w)[v]",
            "score(w) = 1 - cos(E0[w], E1_aligned[w])",
            "```",
            "",
        ]
    )
    for target in audit_targets:
        rows = rows_by_target.get(target)
        if rows is None:
            continue
        salient = [row for row in rows if min(row["rank_d0"], row["rank_d1"]) <= salience_rank]
        lines.extend([f"## `{target}`", "", "### Nearest in D0", ""])
        lines.extend(format_table(top_rows(rows, "similarity_d0", top_k), "similarity_d0"))
        lines.extend(["", "### Nearest in D1", ""])
        lines.extend(format_table(top_rows(rows, "similarity_d1", top_k), "similarity_d1"))
        lines.extend(["", "### Largest relative gains", ""])
        lines.extend(format_table(top_rows(salient, "delta_z", top_k), "delta_z"))
        lines.extend(["", "### Largest relative losses", ""])
        lines.extend(format_table(top_rows(salient, "delta_z", top_k, reverse=False), "delta_z"))
        lines.append("")

    lines.extend(
        [
            "## Interpretation limits",
            "",
            "1. This is a static model: all occurrences of a word are collapsed into",
            "   one vector per period.",
            "2. The D1 space is post-hoc aligned to D0, unlike TimeFormer checkpoint",
            "   queries.",
            "3. Gains and losses are relational coordinates, not sense labels.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Hamilton-style aligned word2vec baseline.")
    parser.add_argument("--corpus-d0", type=Path, default=Path("data/processed/semeval2020_task1/eng_lemma/corpus/1810-1860.txt"))
    parser.add_argument("--corpus-d1", type=Path, default=Path("data/processed/semeval2020_task1/eng_lemma/corpus/1960-2010.txt"))
    parser.add_argument("--targets", type=Path, default=Path("data/processed/semeval2020_task1/eng_lemma/targets.txt"))
    parser.add_argument("--truth", type=Path, default=Path("data/processed/semeval2020_task1/eng_lemma/truth.tsv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/hamilton2016_word2vec_baseline"))
    parser.add_argument("--vector-size", type=int, default=100)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--min-count", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--sg", type=int, default=1, choices=[0, 1])
    parser.add_argument("--max-references", type=int, default=5000)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--salience-rank", type=int, default=50)
    parser.add_argument("--max-sentences", type=int, default=None)
    parser.add_argument("--audit-targets", nargs="*", default=DEFAULT_AUDIT_TARGETS)
    args = parser.parse_args()

    print("Reading corpora...")
    sentences_d0 = read_sentences(args.corpus_d0, max_sentences=args.max_sentences)
    sentences_d1 = read_sentences(args.corpus_d1, max_sentences=args.max_sentences)

    print("Training D0 word2vec...")
    model_d0 = train_word2vec(
        sentences_d0,
        vector_size=args.vector_size,
        window=args.window,
        min_count=args.min_count,
        workers=args.workers,
        epochs=args.epochs,
        seed=args.seed,
        sg=args.sg,
    )
    print("Training D1 word2vec...")
    model_d1 = train_word2vec(
        sentences_d1,
        vector_size=args.vector_size,
        window=args.window,
        min_count=args.min_count,
        workers=args.workers,
        epochs=args.epochs,
        seed=args.seed,
        sg=args.sg,
    )

    targets = read_list(args.targets)
    missing_targets = [
        target for target in targets if target not in model_d0.wv or target not in model_d1.wv
    ]
    if missing_targets:
        raise ValueError(f"Targets missing from one period: {', '.join(missing_targets)}")

    shared_vocab = sorted(set(model_d0.wv.index_to_key) & set(model_d1.wv.index_to_key))
    shared_vocab = [token for token in shared_vocab if token not in {"[PAD]", "[UNK]", "[MASK]", "[CLS]", "[SEP]"}]
    if args.max_references and len(shared_vocab) > args.max_references:
        shared_vocab.sort(key=lambda token: model_d0.wv.get_vecattr(token, "count") + model_d1.wv.get_vecattr(token, "count"), reverse=True)
        selected = set(shared_vocab[: args.max_references])
        selected.update(targets)
        shared_vocab = sorted(selected)

    print(f"Aligning {len(shared_vocab)} shared vectors...")
    x0 = normalized_matrix(model_d0, shared_vocab)
    x1 = normalized_matrix(model_d1, shared_vocab)
    rotation, _ = orthogonal_procrustes(x1, x0)
    x1_aligned = x1 @ rotation

    token_to_index = {token: index for index, token in enumerate(shared_vocab)}
    references = [token for token in shared_vocab if token not in set(targets)]
    reference_indices = np.array([token_to_index[token] for token in references], dtype=int)
    reference_d0 = x0[reference_indices]
    reference_d1 = x1_aligned[reference_indices]

    rows_by_target: dict[str, list[dict]] = {}
    all_rows: list[dict] = []
    scores: dict[str, float] = {}
    score_rows: list[dict] = []
    for target in targets:
        target_index = token_to_index[target]
        target_d0 = x0[target_index]
        target_d1 = x1_aligned[target_index]
        score = float(1.0 - np.dot(target_d0, target_d1))
        scores[target] = score
        before = reference_d0 @ target_d0
        after = reference_d1 @ target_d1
        before_z = standardize(before)
        after_z = standardize(after)
        before_rank = rank_descending(before)
        after_rank = rank_descending(after)
        target_rows = []
        for idx, reference in enumerate(references):
            row = {
                "target": target,
                "reference": reference,
                "similarity_d0": float(before[idx]),
                "similarity_d1": float(after[idx]),
                "delta_similarity": float(after[idx] - before[idx]),
                "z_d0": float(before_z[idx]),
                "z_d1": float(after_z[idx]),
                "delta_z": float(after_z[idx] - before_z[idx]),
                "rank_d0": int(before_rank[idx]),
                "rank_d1": int(after_rank[idx]),
                "rank_gain": int(before_rank[idx] - after_rank[idx]),
            }
            target_rows.append(row)
            all_rows.append(row)
        rows_by_target[target] = target_rows
        score_rows.append({"target": target, "comparison": "from_t0", "hamilton_cosine": score})

    truth = read_truth(args.truth)
    metrics, joined = evaluate(truth, scores)
    config = {
        "vector_size": args.vector_size,
        "window": args.window,
        "min_count": args.min_count,
        "epochs": args.epochs,
        "sg": "skipgram" if args.sg else "cbow",
        "shared_vocab": len(shared_vocab),
        "references": len(references),
        "max_sentences": args.max_sentences,
    }
    metrics = {**metrics, **config}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "neighborhoods.csv", all_rows)
    write_csv(args.output_dir / "scores.csv", score_rows)
    write_csv(args.output_dir / "joined_scores.csv", joined)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    write_report(
        args.output_dir / "report.md",
        rows_by_target=rows_by_target,
        audit_targets=args.audit_targets,
        top_k=args.top_k,
        salience_rank=args.salience_rank,
        metrics=metrics,
        config=config,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
