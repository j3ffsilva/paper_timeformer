#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.evaluate_contextual_usage_clusters import (  # noqa: E402
    aggregate_runs,
    balanced_sample,
    cluster_period_distributions,
    pooled_reference_anchors,
    relational_occurrence_vectors,
)
from scripts.evaluate_hidden_relational_profiles import (  # noqa: E402
    ContextChunkDataset,
    build_model,
    encode_layers,
    write_csv,
)
from timeformers.real_corpus import read_period_corpora  # noqa: E402


FIELD_CONTROLS = {
    "leadership": [
        "secretary",
        "director",
        "committee",
        "president",
        "commander",
        "commissioner",
        "governor",
        "editor",
        "jury",
    ],
    "geometry": ["line", "angle", "plate", "surface", "column", "axis"],
    "botanical": ["branch", "plant", "stock", "wood", "bark", "root", "garden"],
}

TARGET_FIELDS = {
    "chairman_nn": "leadership",
    "plane_nn": "geometry",
    "graft_nn": "botanical",
    "tree_nn": "botanical",
}


@torch.no_grad()
def extract_selected_occurrences(
    model,
    corpus,
    token_to_id: dict[str, int],
    selected_tokens: list[str],
    *,
    layer: str,
    seq_len: int,
    batch_size: int,
    device: str,
) -> dict[str, torch.Tensor]:
    model.eval().to(device)
    dataset = ContextChunkDataset(corpus, token_to_id, seq_len=seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    selected_ids = torch.tensor(
        [token_to_id[token] for token in selected_tokens],
        dtype=torch.long,
        device=device,
    )
    lookup = torch.full(
        (len(token_to_id),),
        -1,
        dtype=torch.long,
        device=device,
    )
    lookup[selected_ids] = torch.arange(len(selected_tokens), device=device)
    chunks = [[] for _ in selected_tokens]

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        lexical_ids = batch["lexical_ids"].to(device)
        valid = lexical_ids.ge(0)
        flat_ids = lexical_ids[valid]
        selected_indices = lookup[flat_ids]
        keep = selected_indices.ge(0)
        if not keep.any():
            continue
        hidden = encode_layers(model, input_ids)[layer][valid][keep].float().cpu()
        indices = selected_indices[keep].cpu()
        for selected_index in indices.unique().tolist():
            chunks[selected_index].append(hidden[indices == selected_index])

    return {
        token: torch.cat(token_chunks) if token_chunks else torch.empty(0, model.d_model)
        for token, token_chunks in zip(selected_tokens, chunks)
    }


def field_adjusted_scores(token_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    scores = {row["token"]: row for row in token_rows}
    field_rows = []
    for field, controls in FIELD_CONTROLS.items():
        values = np.array([scores[token]["score"] for token in controls])
        field_rows.append(
            {
                "field": field,
                "n_controls": len(controls),
                "mean_score": float(values.mean()),
                "median_score": float(np.median(values)),
                "std_score": float(values.std(ddof=1)),
                "min_score": float(values.min()),
                "max_score": float(values.max()),
            }
        )
    field_stats = {row["field"]: row for row in field_rows}
    adjusted = []
    for target, field in TARGET_FIELDS.items():
        row = scores[target]
        baseline = field_stats[field]["median_score"]
        adjusted.append(
            {
                "target": target,
                "field": field,
                "raw_score": row["score"],
                "field_median": baseline,
                "adjusted_score": row["score"] - baseline,
                "theta0_score": row["theta0_score"],
                "theta1_score": row["theta1_score"],
                "checkpoint_disagreement": row["checkpoint_disagreement"],
                "count_d0": row["count_d0"],
                "count_d1": row["count_d1"],
            }
        )
    return field_rows, adjusted


def run_level_adjustments(run_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    scores = {
        (
            row["target"],
            int(row["checkpoint"]),
            int(row["n_clusters"]),
            int(row["seed"]),
        ): float(row["jsd"])
        for row in run_rows
    }
    adjusted_rows = []
    for target, field in TARGET_FIELDS.items():
        controls = FIELD_CONTROLS[field]
        for checkpoint in range(2):
            for n_clusters in sorted({int(row["n_clusters"]) for row in run_rows}):
                for seed in sorted({int(row["seed"]) for row in run_rows}):
                    key = (target, checkpoint, n_clusters, seed)
                    control_values = [
                        scores[(control, checkpoint, n_clusters, seed)]
                        for control in controls
                    ]
                    baseline = float(np.median(control_values))
                    adjusted_rows.append(
                        {
                            "target": target,
                            "field": field,
                            "checkpoint": checkpoint,
                            "n_clusters": n_clusters,
                            "seed": seed,
                            "raw_score": scores[key],
                            "field_median": baseline,
                            "adjusted_score": scores[key] - baseline,
                        }
                    )
    robustness_rows = []
    for target in TARGET_FIELDS:
        values = np.array(
            [
                row["adjusted_score"]
                for row in adjusted_rows
                if row["target"] == target
            ]
        )
        robustness_rows.append(
            {
                "target": target,
                "median_adjusted": float(np.median(values)),
                "mean_adjusted": float(values.mean()),
                "positive_fraction": float(np.mean(values > 0)),
                "q05": float(np.quantile(values, 0.05)),
                "q95": float(np.quantile(values, 0.95)),
                "n_runs": len(values),
            }
        )
    return adjusted_rows, robustness_rows


def field_for_token(token: str) -> str:
    if token in TARGET_FIELDS:
        return TARGET_FIELDS[token]
    for field, controls in FIELD_CONTROLS.items():
        if token in controls:
            return field
    raise ValueError(f"No field configured for token: {token}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose shared temporal nuisance with paired semantic-field controls."
    )
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layer", default="layer_2")
    parser.add_argument("--pca-components", type=int, default=32)
    parser.add_argument("--clusters", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--max-per-period", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--reuse-cache", action="store_true")
    args = parser.parse_args()

    config = json.loads((args.experiment_dir / "config.json").read_text(encoding="utf-8"))
    vocab = json.loads((args.experiment_dir / "vocab.json").read_text(encoding="utf-8"))
    token_to_id = {token: index for index, token in enumerate(vocab)}
    references = json.loads((args.profile_dir / "references.json").read_text(encoding="utf-8"))
    reference_ids = [token_to_id[token] for token in references]
    selected_tokens = list(TARGET_FIELDS)
    for controls in FIELD_CONTROLS.values():
        selected_tokens.extend(controls)
    selected_tokens = list(dict.fromkeys(selected_tokens))
    missing = [token for token in selected_tokens if token not in token_to_id]
    if missing:
        raise ValueError(f"Control tokens missing from vocabulary: {missing}")

    corpora = read_period_corpora(Path(config["input_dir"]))
    full_stats = {
        (checkpoint, corpus): torch.load(
            args.profile_dir / "cache" / f"theta{checkpoint}_d{corpus}.pt",
            map_location="cpu",
            weights_only=True,
        )
        for checkpoint in range(2)
        for corpus in range(2)
    }
    cache_dir = args.output_dir / "cache"
    selected_stats = {}
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
        for corpus_index, corpus in enumerate(corpora):
            cache_path = cache_dir / f"theta{checkpoint}_d{corpus_index}.pt"
            if args.reuse_cache and cache_path.exists():
                print(f"[cache] loading {cache_path}", flush=True)
                cell = torch.load(cache_path, map_location="cpu", weights_only=True)
            else:
                started = time.perf_counter()
                print(
                    f"[extract-controls] theta={checkpoint} corpus={corpus.period}",
                    flush=True,
                )
                cell = extract_selected_occurrences(
                    model,
                    corpus,
                    token_to_id,
                    selected_tokens,
                    layer=args.layer,
                    seq_len=int(config["seq_len"]),
                    batch_size=args.batch_size,
                    device=args.device,
                )
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(cell, cache_path)
                print(
                    f"[extract-controls] finished in {time.perf_counter() - started:.1f}s",
                    flush=True,
                )
            selected_stats[(checkpoint, corpus_index)] = cell
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    run_rows = []
    for checkpoint in range(2):
        anchors, anchor_mean = pooled_reference_anchors(
            full_stats[(checkpoint, 0)],
            full_stats[(checkpoint, 1)],
            args.layer,
            reference_ids,
        )
        for token in selected_tokens:
            before_all = selected_stats[(checkpoint, 0)][token].float().numpy()
            after_all = selected_stats[(checkpoint, 1)][token].float().numpy()
            sampled_n = min(len(before_all), len(after_all), args.max_per_period)
            if sampled_n < max(args.clusters):
                print(f"[skip] {token} has only {sampled_n} paired occurrences", flush=True)
                continue
            for seed in args.seeds:
                before, after = balanced_sample(
                    before_all,
                    after_all,
                    max_per_period=args.max_per_period,
                    seed=seed,
                )
                before, after = relational_occurrence_vectors(
                    before,
                    after,
                    anchors,
                    anchor_mean=anchor_mean,
                    pca_components=args.pca_components,
                    seed=seed,
                )
                for n_clusters in args.clusters:
                    result = cluster_period_distributions(
                        before,
                        after,
                        n_clusters=n_clusters,
                        seed=seed,
                    )
                    run_rows.append(
                        {
                            "target": token,
                            "checkpoint": checkpoint,
                            "n_clusters": n_clusters,
                            "seed": seed,
                            "jsd": result["jsd"],
                            "silhouette": result["silhouette"],
                            "count_d0": len(before_all),
                            "count_d1": len(after_all),
                            "sampled_per_period": sampled_n,
                        }
                    )
            print(
                f"[paired-control] theta={checkpoint} token={token} "
                f"counts={len(before_all)}/{len(after_all)}",
                flush=True,
            )

    checkpoint_rows = aggregate_runs(run_rows)
    grouped: dict[str, list[dict]] = {}
    for row in checkpoint_rows:
        grouped.setdefault(row["target"], []).append(row)
    token_rows = []
    for token, rows in sorted(grouped.items()):
        by_checkpoint = {row["checkpoint"]: row for row in rows}
        theta0 = by_checkpoint[0]["score"]
        theta1 = by_checkpoint[1]["score"]
        token_rows.append(
            {
                "token": token,
                "field": field_for_token(token),
                "role": "target" if token in TARGET_FIELDS else "control",
                "score": 0.5 * (theta0 + theta1),
                "theta0_score": theta0,
                "theta1_score": theta1,
                "checkpoint_disagreement": abs(theta0 - theta1),
                "count_d0": by_checkpoint[0]["count_d0"],
                "count_d1": by_checkpoint[0]["count_d1"],
            }
        )

    field_rows, adjusted_rows = field_adjusted_scores(token_rows)
    run_adjusted_rows, robustness_rows = run_level_adjustments(run_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(run_rows, args.output_dir / "runs.csv")
    write_csv(token_rows, args.output_dir / "token_scores.csv")
    write_csv(field_rows, args.output_dir / "field_baselines.csv")
    write_csv(adjusted_rows, args.output_dir / "adjusted_targets.csv")
    write_csv(run_adjusted_rows, args.output_dir / "run_adjusted_scores.csv")
    write_csv(robustness_rows, args.output_dir / "adjustment_robustness.csv")
    summary = {
        "interpretation": "diagnostic of field-shared temporal nuisance, not a validated semantic score",
        "field_controls": FIELD_CONTROLS,
        "target_fields": TARGET_FIELDS,
        "field_baselines": field_rows,
        "adjusted_targets": adjusted_rows,
        "adjustment_robustness": robustness_rows,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
