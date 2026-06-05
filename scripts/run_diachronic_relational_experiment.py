#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from timeformers.real_corpus import (  # noqa: E402
    RealMLMDataset,
    RealTargetOccurrenceDataset,
    RealWordProbeDataset,
    build_vocabulary,
    read_period_corpora,
    tokenize,
)
from timeformers.real_models import RealStaticMLM  # noqa: E402
from timeformers.relational import (  # noqa: E402
    jensen_shannon_divergence_rows,
    jensen_shannon_similarity_matrix,
    log_pmi_profiles,
    pmi_cosine_displacement,
    ppmi_jsd_displacement,
)
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


def write_top_anchor_csv(profiles: list[dict], path: Path, *, top_k: int) -> None:
    rows = []
    for profile in profiles:
        distributions = profile["distributions"]
        anchors = profile["anchors"]
        targets = profile["targets"]
        k = min(top_k, len(anchors))
        values, indices = torch.topk(distributions, k=k, dim=1)
        for target_index, target in enumerate(targets):
            for rank in range(k):
                anchor_index = int(indices[target_index, rank])
                rows.append(
                    {
                        "period": profile["period"],
                        "target": target,
                        "rank": rank + 1,
                        "anchor": anchors[anchor_index],
                        "probability": float(values[target_index, rank]),
                    }
                )
    write_csv(rows, path)


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


def log(args, message: str) -> None:
    if not args.quiet:
        print(message, flush=True)


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


@torch.no_grad()
def occurrence_prediction_distributions(
    model: RealStaticMLM,
    corpus,
    words: list[str],
    anchor_ids: list[int],
    token_to_id: dict[str, int],
    *,
    period_idx: int,
    seq_len: int,
    batch_size: int,
    device: str,
    max_occurrences_per_target: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    model.to(device)
    dataset = RealTargetOccurrenceDataset(
        corpus,
        words,
        token_to_id,
        period_idx=period_idx,
        seq_len=seq_len,
        max_occurrences_per_target=max_occurrences_per_target,
    )
    sums = torch.zeros(len(words), len(anchor_ids), dtype=torch.float32)
    counts = torch.zeros(len(words), dtype=torch.long)
    if len(dataset) == 0:
        sums.fill_(1.0 / len(anchor_ids))
        return sums, counts
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for batch in loader:
        out = model(batch["input_ids"].to(device), batch["epoch_idx"].to(device))
        batch_indices = torch.arange(batch["input_ids"].size(0), device=device)
        logits = out["logits"][batch_indices, batch["mask_pos"].to(device), :]
        probabilities = torch.softmax(logits, dim=-1)[:, anchor_ids].cpu()
        word_indices = batch["word_idx"].cpu()
        sums.index_add_(0, word_indices, probabilities)
        counts.index_add_(0, word_indices, torch.ones_like(word_indices))
    missing = counts == 0
    if missing.any():
        sums[missing] = 1.0 / len(anchor_ids)
        counts[missing] = 1
    distributions = sums / counts.float().unsqueeze(-1)
    return distributions / distributions.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(distributions.dtype).eps), counts


@torch.no_grad()
def neutral_probe_distribution(
    model: RealStaticMLM,
    token_to_id: dict[str, int],
    *,
    seq_len: int,
    device: str,
) -> torch.Tensor:
    """Estimate p_t = P_{theta_t}(· | [CLS] [MASK] [SEP]).

    This is the model's unconditional prediction — what it expects at a
    masked position with no target word as context.  Used as the marginal
    baseline for log-PMI profile computation.

    Returns:
        (vocab_size,) probability distribution over the full vocabulary.
    """
    model.eval()
    model.to(device)
    pad_id = token_to_id["[PAD]"]
    ids = [token_to_id["[CLS]"], token_to_id["[MASK]"], token_to_id["[SEP]"]]
    ids += [pad_id] * (seq_len - len(ids))
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    epoch_idx = torch.zeros(1, dtype=torch.long, device=device)
    out = model(input_ids, epoch_idx)
    return torch.softmax(out["logits"][0, 1, :], dim=-1).cpu()  # position 1 = [MASK]


@torch.no_grad()
def full_vocab_occurrence_distributions(
    model: RealStaticMLM,
    corpus,
    words: list[str],
    token_to_id: dict[str, int],
    *,
    period_idx: int,
    seq_len: int,
    batch_size: int,
    device: str,
    max_occurrences_per_target: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute q_t(w) over the FULL vocabulary for each target word.

    Unlike occurrence_prediction_distributions, this does not restrict the
    output to an anchor subset — the distribution is over all vocab_size tokens.

    Returns:
        distributions: (n_words, vocab_size) — q_t(w) for each word.
        counts:        (n_words,) — number of occurrences found per word.
    """
    vocab_size = len(token_to_id)
    model.eval()
    model.to(device)
    dataset = RealTargetOccurrenceDataset(
        corpus,
        words,
        token_to_id,
        period_idx=period_idx,
        seq_len=seq_len,
        max_occurrences_per_target=max_occurrences_per_target,
    )
    sums = torch.zeros(len(words), vocab_size, dtype=torch.float32)
    counts = torch.zeros(len(words), dtype=torch.long)
    if len(dataset) == 0:
        sums.fill_(1.0 / vocab_size)
        return sums, counts
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for batch in loader:
        out = model(batch["input_ids"].to(device), batch["epoch_idx"].to(device))
        batch_indices = torch.arange(batch["input_ids"].size(0), device=device)
        logits = out["logits"][batch_indices, batch["mask_pos"].to(device), :]
        probabilities = torch.softmax(logits, dim=-1).cpu()
        word_indices = batch["word_idx"].cpu()
        sums.index_add_(0, word_indices, probabilities)
        counts.index_add_(0, word_indices, torch.ones_like(word_indices))
    missing = counts == 0
    if missing.any():
        sums[missing] = 1.0 / vocab_size
        counts[missing] = 1
    distributions = sums / counts.float().unsqueeze(-1)
    return distributions / distributions.sum(dim=-1, keepdim=True).clamp_min(
        torch.finfo(distributions.dtype).eps
    ), counts


def extract_period_profiles(args, corpora, token_to_id: dict[str, int], targets: list[str], anchors: list[str]) -> list[dict]:
    anchor_ids = [token_to_id[word] for word in anchors]
    use_pmi = args.profile_mode == "pmi"
    profiles = []
    for period in range(args.n_periods):
        started = time.perf_counter()
        log(args, f"[profiles] period={period} loading checkpoint")
        model = build_model(args, len(token_to_id), token_to_id["[PAD]"])
        checkpoint = args.output_dir / "continual_real" / f"checkpoint_t{period:02d}.pt"
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))

        if use_pmi:
            log(args, f"[profiles] period={period} computing neutral probe (p_t)")
            p_t = neutral_probe_distribution(
                model, token_to_id, seq_len=args.seq_len, device=args.device,
            )
            log(args, f"[profiles] period={period} computing full-vocab occurrence distributions (q_t)")
            q_t, occurrence_counts = full_vocab_occurrence_distributions(
                model,
                corpora[period],
                targets,
                token_to_id,
                period_idx=period,
                seq_len=args.seq_len,
                batch_size=args.batch_size,
                device=args.device,
                max_occurrences_per_target=args.max_probe_occurrences_per_target,
            )
            log(args, f"[profiles] period={period} computing log-PMI profiles")
            pmi_profiles = log_pmi_profiles(q_t, p_t)
            # Also keep anchor-restricted distributions for backward compat diagnostics
            distributions = q_t[:, anchor_ids]
            distributions = distributions / distributions.sum(dim=-1, keepdim=True).clamp_min(
                torch.finfo(distributions.dtype).eps
            )
        else:
            log(args, f"[profiles] period={period} computing target-anchor distributions")
            if args.probe_mode == "occurrence":
                distributions, occurrence_counts = occurrence_prediction_distributions(
                    model,
                    corpora[period],
                    targets,
                    anchor_ids,
                    token_to_id,
                    period_idx=period,
                    seq_len=args.seq_len,
                    batch_size=args.batch_size,
                    device=args.device,
                    max_occurrences_per_target=args.max_probe_occurrences_per_target,
                )
            else:
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
                occurrence_counts = torch.ones(len(targets), dtype=torch.long)
            pmi_profiles = None
            p_t = None
            q_t = None

        similarities = jensen_shannon_similarity_matrix(distributions)
        path = args.output_dir / "profiles" / "prediction_anchor_js" / f"t{period:02d}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        profile = {
            "period": period,
            "targets": targets,
            "anchors": anchors,
            "distributions": distributions,       # anchor-restricted q_t
            "full_distributions": q_t,            # full-vocab q_t (None if not pmi mode)
            "marginal": p_t,                      # p_t from neutral probe (None if not pmi mode)
            "pmi_profiles": pmi_profiles,         # log-PMI R_t(w) (None if not pmi mode)
            "similarities": similarities,
            "occurrence_counts": occurrence_counts,
            "probe_mode": args.probe_mode,
            "profile_mode": args.profile_mode,
        }
        torch.save(profile, path)
        profiles.append(profile)
        elapsed = time.perf_counter() - started
        log(args, f"[profiles] period={period} wrote {path} ({elapsed:.1f}s)")
    return profiles


def relational_rows(profiles: list[dict]) -> list[dict]:
    rows = []
    targets = profiles[0]["targets"]
    has_pmi = profiles[0].get("pmi_profiles") is not None
    for period in range(1, len(profiles)):
        before_sim = profiles[period - 1]["similarities"]
        after_sim = profiles[period]["similarities"]
        direct_jsd = jensen_shannon_divergence_rows(
            profiles[period - 1]["distributions"], profiles[period]["distributions"]
        )
        direct_jsd_from_start = jensen_shannon_divergence_rows(
            profiles[0]["distributions"], profiles[period]["distributions"]
        )
        from_start_sim = after_sim - profiles[0]["similarities"]
        consecutive_sim = after_sim - before_sim

        if has_pmi:
            pmi_cos = pmi_cosine_displacement(
                profiles[period - 1]["pmi_profiles"], profiles[period]["pmi_profiles"]
            )
            pmi_cos_from_start = pmi_cosine_displacement(
                profiles[0]["pmi_profiles"], profiles[period]["pmi_profiles"]
            )
            ppmi_jsd = ppmi_jsd_displacement(
                profiles[period - 1]["pmi_profiles"], profiles[period]["pmi_profiles"]
            )
            ppmi_jsd_from_start = ppmi_jsd_displacement(
                profiles[0]["pmi_profiles"], profiles[period]["pmi_profiles"]
            )

        for index, target in enumerate(targets):
            base_consec = {
                "target": target,
                "comparison": "consecutive",
                "from_period": period - 1,
                "to_period": period,
                "mean_abs_delta": float(consecutive_sim[index].abs().mean()),
                "max_abs_delta": float(consecutive_sim[index].abs().max()),
                "direct_jsd": float(direct_jsd[index]),
            }
            base_from_start = {
                "target": target,
                "comparison": "from_t0",
                "from_period": 0,
                "to_period": period,
                "mean_abs_delta": float(from_start_sim[index].abs().mean()),
                "max_abs_delta": float(from_start_sim[index].abs().max()),
                "direct_jsd": float(direct_jsd_from_start[index]),
            }
            if has_pmi:
                base_consec["pmi_cosine"] = float(pmi_cos[index])
                base_consec["ppmi_jsd"] = float(ppmi_jsd[index])
                base_from_start["pmi_cosine"] = float(pmi_cos_from_start[index])
                base_from_start["ppmi_jsd"] = float(ppmi_jsd_from_start[index])
            rows.append(base_consec)
            rows.append(base_from_start)
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
    parser.add_argument("--probe-mode", choices=["occurrence", "template"], default="occurrence")
    parser.add_argument(
        "--profile-mode",
        choices=["anchor", "pmi"],
        default="anchor",
        help=(
            "anchor: compare distributions restricted to the anchor list (legacy). "
            "pmi: compute log-PMI profiles over the full vocabulary using a neutral probe."
        ),
    )
    parser.add_argument("--max-probe-occurrences-per-target", type=int, default=None)
    parser.add_argument("--top-anchors-k", type=int, default=10)
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

    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log(args, f"[setup] reading corpora from {args.input_dir}")
    corpora = read_period_corpora(args.input_dir)
    args.n_periods = len(corpora)
    log(args, f"[setup] periods={', '.join(corpus.period for corpus in corpora)}")
    requested_targets = read_word_list(args.targets)
    requested_anchors = read_word_list(args.anchors)
    log(args, f"[setup] requested targets={len(requested_targets)} anchors={len(requested_anchors)}")
    required = sorted(set(requested_targets + requested_anchors))
    log(args, "[setup] building vocabulary")
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
    log(args, f"[setup] vocab={len(vocab)} targets={len(targets)} anchors={len(anchors)}")

    log(args, "[setup] building MLM windows")
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
    for corpus, full_dataset, dataset in zip(corpora, full_datasets, datasets):
        log(
            args,
            f"[setup] period={corpus.period} windows={len(dataset)} available={len(full_dataset)}",
        )

    config = {
        **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "periods": [corpus.period for corpus in corpora],
        "vocab_size": len(vocab),
        "n_targets": len(targets),
        "n_anchors": len(anchors),
        "n_windows_by_period": [len(dataset) for dataset in datasets],
        "n_available_windows_by_period": [len(dataset) for dataset in full_datasets],
        "probe_mode": args.probe_mode,
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (args.output_dir / "vocab.json").write_text(json.dumps(vocab, indent=2), encoding="utf-8")
    (args.output_dir / "targets.json").write_text(json.dumps(targets, indent=2), encoding="utf-8")
    (args.output_dir / "anchors.json").write_text(json.dumps(anchors, indent=2), encoding="utf-8")
    log(args, f"[setup] wrote config and vocabulary to {args.output_dir}")

    checkpoints = checkpoint_paths(args.output_dir, args.n_periods)
    checkpoint_ready = all(path.exists() for path in checkpoints)
    if args.reuse_checkpoints and checkpoint_ready:
        log(args, "[train] reusing existing checkpoints")
    else:
        log(args, "[train] training chronological checkpoints")
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
        log(args, "[train] finished chronological training")

    profiles = extract_period_profiles(args, corpora, token_to_id, targets, anchors)
    rows = relational_rows(profiles)
    write_csv(rows, args.output_dir / "diachronic_relational_changes.csv")
    write_top_anchor_csv(profiles, args.output_dir / "top_anchors.csv", top_k=args.top_anchors_k)
    elapsed = time.perf_counter() - started
    log(args, f"[done] elapsed={elapsed:.1f}s")
    print(f"Wrote {args.output_dir / 'diachronic_relational_changes.csv'}")


if __name__ == "__main__":
    main()
