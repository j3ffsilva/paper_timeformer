#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_rows(path: Path) -> dict[str, list[dict]]:
    rows_by_target: dict[str, list[dict]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            parsed = {
                **row,
                "similarity_d0": float(row["similarity_d0"]),
                "similarity_d1": float(row["similarity_d1"]),
                "delta_z": float(row["delta_z"]),
                "rank_d0": int(row["rank_d0"]),
                "rank_d1": int(row["rank_d1"]),
                "rank_gain": int(row["rank_gain"]),
            }
            rows_by_target.setdefault(row["target"], []).append(parsed)
    return rows_by_target


def read_truth(path: Path) -> dict[str, dict[str, float]]:
    truth = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            truth[row["target"]] = {
                "binary": float(row["binary"]),
                "graded": float(row["graded"]),
            }
    return truth


def top_refs(rows: list[dict], key: str, n: int, *, reverse: bool = True) -> list[str]:
    return [
        row["reference"]
        for row in sorted(rows, key=lambda item: item[key], reverse=reverse)[:n]
    ]


def salient_rows(rows: list[dict], salience_rank: int) -> list[dict]:
    return [row for row in rows if min(row["rank_d0"], row["rank_d1"]) <= salience_rank]


def method_summary(rows: list[dict], *, top_k: int, salience_rank: int) -> dict:
    d0 = top_refs(rows, "similarity_d0", top_k)
    d1 = top_refs(rows, "similarity_d1", top_k)
    salient = salient_rows(rows, salience_rank)
    gains = top_refs(salient, "delta_z", top_k)
    losses = top_refs(salient, "delta_z", top_k, reverse=False)
    overlap = len(set(d0) & set(d1))
    return {
        "d0": d0,
        "d1": d1,
        "gains": gains,
        "losses": losses,
        "overlap": overlap,
        "turnover": 1.0 - (overlap / top_k),
    }


def fmt_refs(refs: list[str]) -> str:
    return ", ".join(refs)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    path: Path,
    *,
    comparison_rows: list[dict],
    target_sections: list[dict],
    timeformer_metrics: dict | None,
    hamilton_metrics: dict | None,
    top_k: int,
    salience_rank: int,
) -> None:
    lines = [
        "# TimeFormer vs Hamilton temporal-neighborhood comparison",
        "",
        "This report compares temporal lexical neighborhoods, not only scalar",
        "semantic-change scores. The current SemEval corpus has only two broad",
        "periods, so this is a D0-to-D1 linguistic displacement audit rather",
        "than a full continuity test across many checkpoints.",
        "",
        "## Configuration",
        "",
        f"- top-k displayed: `{top_k}`",
        f"- gain/loss salience cutoff: top `{salience_rank}` in D0 or D1",
        "",
        "## Quantitative validation",
        "",
    ]
    if timeformer_metrics:
        lines.extend(
            [
                "TimeFormer:",
                f"- Spearman/APD or selected metric: `{timeformer_metrics}`",
                "",
            ]
        )
    if hamilton_metrics:
        lines.extend(
            [
                "Hamilton word2vec:",
                f"- Spearman graded: `{hamilton_metrics.get('spearman_graded'):.3f}`",
                f"- ROC-AUC binary: `{hamilton_metrics.get('roc_auc_binary'):.3f}`",
                f"- Average precision: `{hamilton_metrics.get('average_precision_binary'):.3f}`",
                "",
            ]
        )

    avg_tf_turnover = sum(row["timeformer_turnover"] for row in comparison_rows) / len(comparison_rows)
    avg_ham_turnover = sum(row["hamilton_turnover"] for row in comparison_rows) / len(comparison_rows)
    lines.extend(
        [
            "## Turnover summary",
            "",
            "`turnover@k = 1 - |top_k(D0) ∩ top_k(D1)| / k`.",
            "",
            f"- TimeFormer mean turnover@{top_k}: `{avg_tf_turnover:.3f}`",
            f"- Hamilton mean turnover@{top_k}: `{avg_ham_turnover:.3f}`",
            "",
            "High turnover means the visible lexical neighborhood changed more.",
            "It does not by itself decide whether the change is lexical sense,",
            "domain, register, genre, or broader lexical ecology.",
            "",
            "## Targets with largest turnover disagreement",
            "",
            "| target | gold graded | TimeFormer turnover | Hamilton turnover | difference |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    disagreement = sorted(
        comparison_rows,
        key=lambda row: abs(row["timeformer_turnover"] - row["hamilton_turnover"]),
        reverse=True,
    )[:10]
    for row in disagreement:
        lines.append(
            f"| `{row['target']}` | {row['graded']:.3f} | "
            f"{row['timeformer_turnover']:.3f} | {row['hamilton_turnover']:.3f} | "
            f"{row['timeformer_turnover'] - row['hamilton_turnover']:.3f} |"
        )

    lines.extend(["", "## Per-target neighborhoods", ""])
    for section in target_sections:
        target = section["target"]
        tf = section["timeformer"]
        ham = section["hamilton"]
        lines.extend(
            [
                f"### `{target}`",
                "",
                f"Gold graded: `{section['graded']:.3f}`; binary: `{section['binary']:.0f}`.",
                "",
                "| Method | D0 top | D1 top | gains | losses | turnover |",
                "|---|---|---|---|---|---:|",
                (
                    f"| TimeFormer | {fmt_refs(tf['d0'])} | {fmt_refs(tf['d1'])} | "
                    f"{fmt_refs(tf['gains'])} | {fmt_refs(tf['losses'])} | {tf['turnover']:.3f} |"
                ),
                (
                    f"| Hamilton | {fmt_refs(ham['d0'])} | {fmt_refs(ham['d1'])} | "
                    f"{fmt_refs(ham['gains'])} | {fmt_refs(ham['losses'])} | {ham['turnover']:.3f} |"
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Interpretation",
            "",
            "This report should be read as a qualitative audit. Hamilton is expected",
            "to be strong on two-period static changes because it directly optimizes",
            "one vector per word per period and then aligns the spaces. TimeFormer",
            "must earn its contribution through contextuality, in-domain continual",
            "training, and eventually multi-checkpoint temporal consultation.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def load_json_if_exists(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare TimeFormer and Hamilton temporal neighborhoods.")
    parser.add_argument(
        "--timeformer-neighborhoods",
        type=Path,
        default=Path("outputs/semeval2020_pmi_dynamic_mlm_12_8_d128/temporal_relational_neighborhoods/neighborhoods.csv"),
    )
    parser.add_argument(
        "--hamilton-neighborhoods",
        type=Path,
        default=Path("outputs/hamilton2016_word2vec_baseline/neighborhoods.csv"),
    )
    parser.add_argument(
        "--hamilton-metrics",
        type=Path,
        default=Path("outputs/hamilton2016_word2vec_baseline/metrics.json"),
    )
    parser.add_argument(
        "--timeformer-metrics",
        type=Path,
        default=Path("outputs/semeval2020_pmi_dynamic_mlm_12_8_d128/hidden_relational_profiles/metrics.json"),
    )
    parser.add_argument("--truth", type=Path, default=Path("data/processed/semeval2020_task1/eng_lemma/truth.tsv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/timeformer_vs_hamilton_neighborhood_comparison"))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--salience-rank", type=int, default=50)
    args = parser.parse_args()

    timeformer = read_rows(args.timeformer_neighborhoods)
    hamilton = read_rows(args.hamilton_neighborhoods)
    truth = read_truth(args.truth)
    targets = sorted(set(timeformer) & set(hamilton) & set(truth))
    if not targets:
        raise ValueError("No shared targets across both neighborhood files and truth")

    comparison_rows = []
    target_sections = []
    for target in targets:
        tf = method_summary(timeformer[target], top_k=args.top_k, salience_rank=args.salience_rank)
        ham = method_summary(hamilton[target], top_k=args.top_k, salience_rank=args.salience_rank)
        comparison_rows.append(
            {
                "target": target,
                "binary": truth[target]["binary"],
                "graded": truth[target]["graded"],
                "timeformer_overlap": tf["overlap"],
                "timeformer_turnover": tf["turnover"],
                "hamilton_overlap": ham["overlap"],
                "hamilton_turnover": ham["turnover"],
                "turnover_difference": tf["turnover"] - ham["turnover"],
            }
        )
        target_sections.append(
            {
                "target": target,
                "binary": truth[target]["binary"],
                "graded": truth[target]["graded"],
                "timeformer": tf,
                "hamilton": ham,
            }
        )

    comparison_rows.sort(key=lambda row: row["graded"], reverse=True)
    target_sections.sort(key=lambda row: row["graded"], reverse=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "turnover_comparison.csv", comparison_rows)
    hamilton_metrics = load_json_if_exists(args.hamilton_metrics)
    timeformer_metrics = load_json_if_exists(args.timeformer_metrics)
    if isinstance(timeformer_metrics, list):
        timeformer_metrics = timeformer_metrics[0] if timeformer_metrics else None
    write_report(
        args.output_dir / "report.md",
        comparison_rows=comparison_rows,
        target_sections=target_sections,
        timeformer_metrics=timeformer_metrics,
        hamilton_metrics=hamilton_metrics,
        top_k=args.top_k,
        salience_rank=args.salience_rank,
    )
    summary = {
        "n_targets": len(targets),
        "top_k": args.top_k,
        "salience_rank": args.salience_rank,
        "mean_timeformer_turnover": sum(row["timeformer_turnover"] for row in comparison_rows) / len(comparison_rows),
        "mean_hamilton_turnover": sum(row["hamilton_turnover"] for row in comparison_rows) / len(comparison_rows),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
