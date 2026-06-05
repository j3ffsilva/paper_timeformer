#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from timeformers.real_corpus import (  # noqa: E402
    RealMLMDataset,
    RealWordProbeDataset,
    build_vocabulary,
    read_period_corpora,
    tokenize,
)
from timeformers.real_models import RealStaticMLM  # noqa: E402
from timeformers.relational import jensen_shannon_similarity_matrix  # noqa: E402
from timeformers.train import ContinualPeriodTrainer  # noqa: E402


def read_word_list(path: Path | None) -> list[str]:
    if path is None:
        return []
    words = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        words.extend(tokenize(line))
    return sorted(set(words))


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_model(args, vocab_size: int, pad_id: int) -> RealStaticMLM:
    return RealStaticMLM(
        vocab_size=vocab_size,
        seq_len=args.seq_len,
        d_model=args.d_model,
        n_layers=args.layers,
        n_heads=args.heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
        pad_id=pad_id,
    )


def checkpoint_paths(output_dir: Path, n_periods: int) -> list[Path]:
    return [output_dir / "continual_real" / f"checkpoint_t{period:02d}.pt" for period in range(n_periods)]


def maybe_limit_dataset(dataset, max_windows: int | None):
    if max_windows is None or len(dataset) <= max_windows:
        return dataset
    return Subset(dataset, range(max_windows))


@torch.no_grad()
def prediction_distributions(
    model: RealStaticMLM,
    words: list[str],
    anchor_ids: list[int],
    token_to_id: dict[str, int],
    *,
    period_idx: int,
    seq_len: int,
    batch_size: int,
    device: str,
) -> torch.Tensor:
    model.eval()
    model.to(device)
    dataset = RealWordProbeDataset(words, token_to_id, period_idx=period_idx, seq_len=seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    rows = []
    for batch in loader:
        out = model(batch["input_ids"].to(device), batch["epoch_idx"].to(device))
        probabilities = torch.softmax(out["logits"][:, 2, :], dim=-1)
        rows.append(probabilities[:, anchor_ids].cpu())
    distributions = torch.cat(rows, dim=0)
    return distributions / distributions.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(distributions.dtype).eps)


def extract_period_profiles(args, token_to_id: dict[str, int], targets: list[str], anchors: list[str]) -> list[dict]:
    anchor_ids = [token_to_id[word] for word in anchors]
    profiles = []
    for period in range(args.n_periods):
        model = build_model(args, len(token_to_id), token_to_id["[PAD]"])
        checkpoint = args.output_dir / "continual_real" / f"checkpoint_t{period:02d}.pt"
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        distributions = prediction_distributions(
            model,
            targets,
            anchor_ids,
            token_to_id,
            period_idx=period,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            device=args.device,
        )
        similarities = jensen_shannon_similarity_matrix(distributions)
        path = args.output_dir / "profiles" / "prediction_anchor_js" / f"t{period:02d}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        profile = {
            "period": period,
            "targets": targets,
            "anchors": anchors,
            "distributions": distributions,
            "similarities": similarities,
        }
        torch.save(profile, path)
        profiles.append(profile)
    return profiles


def relational_rows(profiles: list[dict]) -> list[dict]:
    rows = []
    targets = profiles[0]["targets"]
    for period in range(1, len(profiles)):
        before = profiles[period - 1]["similarities"]
        after = profiles[period]["similarities"]
        from_start = profiles[period]["similarities"] - profiles[0]["similarities"]
        consecutive = after - before
        for index, target in enumerate(targets):
            rows.append(
                {
                    "target": target,
                    "comparison": "consecutive",
                    "from_period": period - 1,
                    "to_period": period,
                    "mean_abs_delta": float(consecutive[index].abs().mean()),
                    "max_abs_delta": float(consecutive[index].abs().max()),
                }
            )
            rows.append(
                {
                    "target": target,
                    "comparison": "from_t0",
                    "from_period": 0,
                    "to_period": period,
                    "mean_abs_delta": float(from_start[index].abs().mean()),
                    "max_abs_delta": float(from_start[index].abs().max()),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a minimal relational Timeformer experiment on diachronic text.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/diachronic_relational"))
    parser.add_argument("--targets", type=Path, default=None)
    parser.add_argument("--anchors", type=Path, default=None)
    parser.add_argument("--min-count", type=int, default=10)
    parser.add_argument("--max-vocab", type=int, default=30_000)
    parser.add_argument("--max-targets", type=int, default=200)
    parser.add_argument("--max-anchors", type=int, default=500)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--max-windows-per-period", type=int, default=None)
    parser.add_argument("--base-epochs", type=int, default=2)
    parser.add_argument("--epochs-per-period", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--reuse-checkpoints", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    corpora = read_period_corpora(args.input_dir)
    args.n_periods = len(corpora)
    requested_targets = read_word_list(args.targets)
    requested_anchors = read_word_list(args.anchors)
    required = sorted(set(requested_targets + requested_anchors))
    vocab, token_to_id = build_vocabulary(
        corpora,
        min_count=args.min_count,
        max_vocab=args.max_vocab,
        required_tokens=required,
    )
    counts = {}
    for corpus in corpora:
        period_counts = {}
        for document in corpus.documents:
            for token in document:
                period_counts[token] = period_counts.get(token, 0) + 1
        counts[corpus.period] = period_counts

    candidates = [token for token in vocab if token not in {"[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"}]
    if requested_targets:
        targets = [word for word in requested_targets if word in token_to_id]
    else:
        targets = candidates[: args.max_targets]
    if requested_anchors:
        anchors = [word for word in requested_anchors if word in token_to_id]
    else:
        anchors = [word for word in candidates if word not in set(targets)][: args.max_anchors]
    if not targets or not anchors:
        raise ValueError("Need at least one target and one anchor word")

    full_datasets = [
        RealMLMDataset(
            corpus,
            token_to_id,
            period_idx=period_idx,
            seq_len=args.seq_len,
            stride=args.stride,
        )
        for period_idx, corpus in enumerate(corpora)
    ]
    datasets = [maybe_limit_dataset(dataset, args.max_windows_per_period) for dataset in full_datasets]
    if any(len(dataset) == 0 for dataset in datasets):
        raise ValueError("Every period must yield at least one training window")

    config = {
        **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "periods": [corpus.period for corpus in corpora],
        "vocab_size": len(vocab),
        "n_targets": len(targets),
        "n_anchors": len(anchors),
        "n_windows_by_period": [len(dataset) for dataset in datasets],
        "n_available_windows_by_period": [len(dataset) for dataset in full_datasets],
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (args.output_dir / "vocab.json").write_text(json.dumps(vocab, indent=2), encoding="utf-8")
    (args.output_dir / "targets.json").write_text(json.dumps(targets, indent=2), encoding="utf-8")
    (args.output_dir / "anchors.json").write_text(json.dumps(anchors, indent=2), encoding="utf-8")

    checkpoints = checkpoint_paths(args.output_dir, args.n_periods)
    if not args.reuse_checkpoints or not all(path.exists() for path in checkpoints):
        model = build_model(args, len(vocab), token_to_id["[PAD]"])
        ContinualPeriodTrainer(model, args.output_dir / "continual_real", device=args.device).train(
            datasets,
            val_period_datasets=None,
            n_epochs_first_period=args.base_epochs,
            n_epochs_per_period=args.epochs_per_period,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            restore_best_model=False,
            verbose=not args.quiet,
        )

    profiles = extract_period_profiles(args, token_to_id, targets, anchors)
    rows = relational_rows(profiles)
    write_csv(rows, args.output_dir / "diachronic_relational_changes.csv")
    print(f"Wrote {args.output_dir / 'diachronic_relational_changes.csv'}")


if __name__ == "__main__":
    main()
