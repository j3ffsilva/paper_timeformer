#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path


PERIODS = {
    "eng": {
        "corpus1": "1810-1860",
        "corpus2": "1960-2010",
    }
}

DEFAULT_STOPWORDS = {
    "'s",
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "also",
    "am",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can",
    "could",
    "do",
    "does",
    "doing",
    "down",
    "during",
    "each",
    "for",
    "from",
    "had",
    "has",
    "have",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "itself",
    "just",
    "me",
    "may",
    "more",
    "most",
    "my",
    "n't",
    "no",
    "nor",
    "not",
    "now",
    "of",
    "only",
    "on",
    "one",
    "or",
    "our",
    "out",
    "over",
    "own",
    "s",
    "she",
    "should",
    "so",
    "some",
    "such",
    "than",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "under",
    "until",
    "up",
    "upon",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "will",
    "with",
    "would",
    "you",
    "your",
}


def read_lines(path: Path, *, max_lines: int | None = None) -> list[str]:
    lines = []
    with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
        for index, line in enumerate(f):
            if max_lines is not None and index >= max_lines:
                break
            line = line.strip()
            if line:
                lines.append(line)
    return lines


def write_period_file(source: Path, target: Path, *, max_lines: int | None = None) -> Counter[str]:
    target.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    n_lines = 0
    with gzip.open(source, "rt", encoding="utf-8", errors="ignore") as src, target.open("w", encoding="utf-8") as dst:
        for index, line in enumerate(src):
            if max_lines is not None and index >= max_lines:
                break
            line = line.strip()
            if not line:
                continue
            tokens = line.split()
            counts.update(tokens)
            dst.write(" ".join(tokens))
            dst.write("\n")
            n_lines += 1
    if n_lines == 0:
        raise ValueError(f"No lines written from {source}")
    return counts


def read_word_score_file(path: Path) -> dict[str, str]:
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        word, value = line.split(maxsplit=1)
        values[word] = value
    return values


def write_truth_table(targets: list[str], raw_dir: Path, output_dir: Path) -> None:
    binary = read_word_score_file(raw_dir / "truth" / "binary.txt")
    graded = read_word_score_file(raw_dir / "truth" / "graded.txt")
    with (output_dir / "truth.tsv").open("w", encoding="utf-8") as f:
        f.write("target\tbinary\tgraded\n")
        for target in targets:
            f.write(f"{target}\t{binary.get(target, '')}\t{graded.get(target, '')}\n")


def select_anchors(
    counts_by_period: list[Counter[str]],
    targets: set[str],
    *,
    min_count: int,
    max_anchors: int,
    exclude_stopwords: bool = False,
    min_length: int = 1,
    max_period_count: int | None = None,
    max_period_fraction: float | None = None,
    alpha_only: bool = False,
) -> list[str]:
    shared = set(counts_by_period[0])
    for counts in counts_by_period[1:]:
        shared &= set(counts)
    candidates = []
    for token in shared:
        if token in targets:
            continue
        if exclude_stopwords and token in DEFAULT_STOPWORDS:
            continue
        if alpha_only and not token.isalpha():
            continue
        if len(token) < min_length:
            continue
        period_counts = [counts[token] for counts in counts_by_period]
        if min(period_counts) < min_count:
            continue
        if max_period_count is not None and max(period_counts) > max_period_count:
            continue
        if max_period_fraction is not None:
            too_frequent = any(
                count / max(sum(period_counts_for_period.values()), 1) > max_period_fraction
                for count, period_counts_for_period in zip(period_counts, counts_by_period)
            )
            if too_frequent:
                continue
        candidates.append((sum(period_counts), min(period_counts), token))
    candidates.sort(reverse=True)
    return [token for _, _, token in candidates[:max_anchors]]


def write_list(words: list[str], path: Path) -> None:
    path.write_text("\n".join(words) + "\n", encoding="utf-8")


def prepare(args: argparse.Namespace) -> None:
    if args.language not in PERIODS:
        raise ValueError(f"Unsupported language: {args.language}")
    raw_dir = args.raw_dir
    corpus_kind = args.corpus_kind
    output_dir = args.output_dir
    corpus_output = output_dir / "corpus"
    output_dir.mkdir(parents=True, exist_ok=True)

    targets = [
        line.strip()
        for line in (raw_dir / "targets.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    period_counts = []
    period_files = {}
    for corpus_name, period_label in PERIODS[args.language].items():
        source = raw_dir / corpus_name / corpus_kind / f"ccoha{corpus_name[-1]}.txt.gz"
        target = corpus_output / f"{period_label}.txt"
        counts = write_period_file(source, target, max_lines=args.max_lines)
        period_counts.append(counts)
        period_files[period_label] = str(target)

    anchors = select_anchors(
        period_counts,
        set(targets),
        min_count=args.anchor_min_count,
        max_anchors=args.max_anchors,
        exclude_stopwords=args.exclude_stopwords,
        min_length=args.anchor_min_length,
        max_period_count=args.anchor_max_period_count,
        max_period_fraction=args.anchor_max_period_fraction,
        alpha_only=args.anchor_alpha_only,
    )
    if not anchors:
        raise ValueError("No anchors selected; lower --anchor-min-count")

    write_list(targets, output_dir / "targets.txt")
    write_list(anchors, output_dir / "anchors.txt")
    write_truth_table(targets, raw_dir, output_dir)

    metadata = {
        "dataset": "SemEval-2020 Task 1",
        "language": args.language,
        "corpus_kind": corpus_kind,
        "raw_dir": str(raw_dir),
        "output_dir": str(output_dir),
        "period_files": period_files,
        "n_targets": len(targets),
        "n_anchors": len(anchors),
        "anchor_min_count": args.anchor_min_count,
        "anchor_min_length": args.anchor_min_length,
        "anchor_max_period_count": args.anchor_max_period_count,
        "anchor_max_period_fraction": args.anchor_max_period_fraction,
        "anchor_alpha_only": args.anchor_alpha_only,
        "exclude_stopwords": args.exclude_stopwords,
        "max_anchors": args.max_anchors,
        "max_lines": args.max_lines,
        "target_coverage": {
            target: [counts.get(target, 0) for counts in period_counts]
            for target in targets
        },
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote processed SemEval data to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare SemEval-2020 Task 1 for diachronic relational experiments.")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw/semeval2020_task1/semeval2020_ulscd_eng"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/semeval2020_task1/eng_lemma"),
    )
    parser.add_argument("--language", default="eng")
    parser.add_argument("--corpus-kind", choices=["lemma", "token"], default="lemma")
    parser.add_argument("--anchor-min-count", type=int, default=500)
    parser.add_argument("--anchor-min-length", type=int, default=1)
    parser.add_argument("--anchor-max-period-count", type=int, default=None)
    parser.add_argument("--anchor-max-period-fraction", type=float, default=None)
    parser.add_argument("--anchor-alpha-only", action="store_true")
    parser.add_argument("--exclude-stopwords", action="store_true")
    parser.add_argument("--max-anchors", type=int, default=1000)
    parser.add_argument("--max-lines", type=int, default=None)
    args = parser.parse_args()
    prepare(args)


if __name__ == "__main__":
    main()
