#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from timeformers.bert_continual import (  # noqa: E402
    CheckpointDiagnostic,
    DynamicWordPieceMLMDataset,
    anchor_hidden_cosine,
    assert_weight_tying,
    embedding_pairwise_cosine,
    encode_windows,
    evaluate_mlm_loss,
    parameter_relative_l2,
    random_pseudo_periods,
    read_tokenized_documents,
    relative_l2_sp_penalty,
    select_checkpoint,
    snapshot_named_parameters,
    snapshot_parameters,
    split_documents,
    write_diagnostics,
)


def limit_windows(windows: list[list[int]], limit: int | None) -> list[list[int]]:
    return windows if limit is None else windows[:limit]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Option D: continual temporal MLM with the intact bert-tiny model/tokenizer/head."
    )
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--period-files", nargs="+", default=["1810-1860.txt", "1960-2010.txt"])
    parser.add_argument("--model-name", default="prajjwal1/bert-tiny")
    parser.add_argument("--period-mode", choices=["chronological", "random"], default="chronological")
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--max-train-windows-per-period", type=int, default=None)
    parser.add_argument("--max-validation-windows-per-period", type=int, default=5000)
    parser.add_argument("--epochs-first-period", type=float, default=3.0)
    parser.add_argument("--epochs-per-period", type=float, default=2.0)
    parser.add_argument("--checkpoint-fractions", nargs="*", type=float, default=[0.25, 0.5, 1.0])
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument(
        "--lower-lr-scale",
        type=float,
        default=1.0,
        help="Multiplier applied to embeddings and encoder layer 1 learning rates.",
    )
    parser.add_argument(
        "--layer2-l2sp-lambda",
        type=float,
        default=0.0,
        help="Weight for relative L2-SP regularization of encoder layer 2.",
    )
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--freeze-lower", action="store_true")
    parser.add_argument("--anchor-batch-size", type=int, default=64)
    args = parser.parse_args()

    from transformers import AutoModelForMaskedLM, AutoTokenizer, get_linear_schedule_with_warmup

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForMaskedLM.from_pretrained(args.model_name)
    assert_weight_tying(model)
    if args.freeze_lower:
        for parameter in model.bert.embeddings.parameters():
            parameter.requires_grad = False
        for parameter in model.bert.encoder.layer[0].parameters():
            parameter.requires_grad = False
    model.to(args.device)
    device = torch.device(args.device)

    period_documents = [
        read_tokenized_documents(args.corpus_dir / filename)
        for filename in args.period_files
    ]
    if args.period_mode == "random":
        period_documents = random_pseudo_periods(period_documents, seed=args.seed)

    train_datasets = []
    validation_datasets = []
    split_summary = []
    for period, documents in enumerate(period_documents):
        train_documents, validation_documents = split_documents(
            documents,
            validation_fraction=args.validation_fraction,
            seed=args.seed + period,
        )
        train_windows = limit_windows(
            encode_windows(train_documents, tokenizer, seq_len=args.seq_len, stride=args.stride),
            args.max_train_windows_per_period,
        )
        validation_windows = limit_windows(
            encode_windows(validation_documents, tokenizer, seq_len=args.seq_len, stride=args.stride),
            args.max_validation_windows_per_period,
        )
        train_datasets.append(
            DynamicWordPieceMLMDataset(
                train_windows,
                tokenizer,
                seq_len=args.seq_len,
                seed=args.seed + 10_000 * period,
            )
        )
        validation_datasets.append(
            DynamicWordPieceMLMDataset(
                validation_windows,
                tokenizer,
                seq_len=args.seq_len,
                seed=args.seed + 20_000 * period,
            )
        )
        split_summary.append({
            "period": period,
            "documents": len(documents),
            "train_documents": len(train_documents),
            "validation_documents": len(validation_documents),
            "train_windows": len(train_windows),
            "validation_windows": len(validation_windows),
        })

    anchor_windows = []
    for dataset in validation_datasets:
        anchor_windows.extend(dataset.windows[: args.anchor_batch_size])
    anchor_dataset = DynamicWordPieceMLMDataset(
        anchor_windows[: args.anchor_batch_size],
        tokenizer,
        seq_len=args.seq_len,
        mask_probability=0.15,
        seed=args.seed + 99_000,
    )
    anchor_loader = DataLoader(anchor_dataset, batch_size=len(anchor_dataset), shuffle=False)
    anchor_batch = next(iter(anchor_loader))
    with torch.no_grad():
        initial_outputs = model.bert(
            input_ids=anchor_batch["input_ids"].to(device),
            attention_mask=anchor_batch["attention_mask"].to(device),
            return_dict=True,
        )
        mask = anchor_batch["attention_mask"].to(device).bool().unsqueeze(-1)
        initial_hidden = (
            (initial_outputs.last_hidden_state * mask).sum(dim=1)
            / mask.sum(dim=1).clamp_min(1)
        ).cpu()
    initial_parameters = snapshot_parameters(model)
    layer2_reference = snapshot_named_parameters(
        model,
        prefix="bert.encoder.layer.1.",
        device=device,
    )
    generator = torch.Generator().manual_seed(args.seed)
    non_special_ids = [
        token_id for token_id in range(tokenizer.vocab_size)
        if token_id not in set(tokenizer.all_special_ids)
    ]
    sampled_indices = torch.randperm(len(non_special_ids), generator=generator)[:256]
    diagnostic_token_ids = torch.tensor(
        [non_special_ids[index] for index in sampled_indices.tolist()],
        dtype=torch.long,
        device=device,
    )

    validation_loaders = [
        DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        for dataset in validation_datasets
    ]
    period_epochs = [args.epochs_first_period] + [args.epochs_per_period] * (len(train_datasets) - 1)
    steps_per_epoch = [
        math.ceil(len(dataset) / args.batch_size)
        for dataset in train_datasets
    ]
    total_steps = sum(math.ceil(epochs * steps) for epochs, steps in zip(period_epochs, steps_per_epoch))
    lower_parameters = []
    upper_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if (
            name.startswith("bert.embeddings.")
            or name.startswith("bert.encoder.layer.0.")
        ):
            lower_parameters.append(parameter)
        else:
            upper_parameters.append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {"params": lower_parameters, "lr": args.lr * args.lower_lr_scale},
            {"params": upper_parameters, "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=round(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )

    config = {
        **vars(args),
        "corpus_dir": str(args.corpus_dir),
        "output_dir": str(args.output_dir),
        "split_summary": split_summary,
        "total_steps": total_steps,
        "warmup_steps": round(total_steps * args.warmup_ratio),
        "weight_tying_verified": True,
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    tokenizer.save_pretrained(args.output_dir / "tokenizer")

    diagnostics: list[CheckpointDiagnostic] = []
    global_step = 0

    def save_checkpoint(
        name: str,
        period: int,
        epoch_fraction: float,
        train_loss: float | None,
        train_mlm_loss: float | None = None,
        train_l2sp_penalty: float | None = None,
    ) -> None:
        checkpoint_dir = args.output_dir / "checkpoints" / name
        model.save_pretrained(checkpoint_dir, safe_serialization=True)
        tokenizer.save_pretrained(checkpoint_dir)
        validation_losses = [
            evaluate_mlm_loss(model, loader, device)
            for loader in validation_loaders
        ]
        diagnostic = CheckpointDiagnostic(
            name=name,
            period=period,
            epoch_fraction=epoch_fraction,
            global_step=global_step,
            train_loss=train_loss,
            train_mlm_loss=train_mlm_loss,
            train_l2sp_penalty=train_l2sp_penalty,
            validation_losses=validation_losses,
            mean_validation_loss=sum(validation_losses) / len(validation_losses),
            selection_loss=(
                sum(validation_losses[: period + 1]) / (period + 1)
                if period >= 0
                else sum(validation_losses) / len(validation_losses)
            ),
            parameter_relative_l2=parameter_relative_l2(model, initial_parameters),
            layer2_relative_l2=parameter_relative_l2(
                model,
                initial_parameters,
                prefix="bert.encoder.layer.1.",
            ),
            anchor_hidden_cosine=anchor_hidden_cosine(
                model,
                anchor_batch,
                initial_hidden,
                device=device,
            ),
            embedding_pairwise_cosine=embedding_pairwise_cosine(model, diagnostic_token_ids),
        )
        diagnostics.append(diagnostic)
        print(json.dumps(diagnostic.__dict__), flush=True)

    save_checkpoint("init", period=-1, epoch_fraction=0.0, train_loss=None)
    for period, dataset in enumerate(train_datasets):
        loader_generator = torch.Generator().manual_seed(args.seed + period)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            generator=loader_generator,
        )
        n_steps = steps_per_epoch[period]
        total_period_steps = math.ceil(period_epochs[period] * n_steps)
        checkpoint_steps = {
            max(1, round(fraction * n_steps)): fraction
            for fraction in args.checkpoint_fractions
            if fraction <= period_epochs[period]
        }
        for epoch in range(math.ceil(period_epochs[period])):
            dataset.set_epoch(epoch)
            model.train()
            running_loss = 0.0
            running_mlm_loss = 0.0
            running_l2sp_penalty = 0.0
            running_batches = 0
            for batch_index, batch in enumerate(loader):
                period_step = epoch * n_steps + batch_index + 1
                if period_step > total_period_steps:
                    break
                optimizer.zero_grad()
                outputs = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    labels=batch["labels"].to(device),
                )
                l2sp_penalty = relative_l2_sp_penalty(model, layer2_reference)
                loss = outputs.loss + args.layer2_l2sp_lambda * l2sp_penalty
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                global_step += 1
                running_loss += float(loss.detach())
                running_mlm_loss += float(outputs.loss.detach())
                running_l2sp_penalty += float(l2sp_penalty.detach())
                running_batches += 1
                if period_step in checkpoint_steps:
                    fraction = checkpoint_steps[period_step]
                    save_checkpoint(
                        f"period{period}_epoch{fraction:g}",
                        period=period,
                        epoch_fraction=fraction,
                        train_loss=running_loss / running_batches,
                        train_mlm_loss=running_mlm_loss / running_batches,
                        train_l2sp_penalty=running_l2sp_penalty / running_batches,
                    )
            completed_fraction = min(float(epoch + 1), period_epochs[period])
            if completed_fraction not in checkpoint_steps.values():
                save_checkpoint(
                    f"period{period}_epoch{completed_fraction:g}",
                    period=period,
                    epoch_fraction=completed_fraction,
                    train_loss=running_loss / max(running_batches, 1),
                    train_mlm_loss=running_mlm_loss / max(running_batches, 1),
                    train_l2sp_penalty=running_l2sp_penalty / max(running_batches, 1),
                )
            if completed_fraction >= period_epochs[period]:
                break

    selected_by_period = {
        period: select_checkpoint([
            diagnostic
            for diagnostic in diagnostics
            if diagnostic.period == period
        ])
        for period in range(len(train_datasets))
    }
    selected = selected_by_period[len(train_datasets) - 1]
    write_diagnostics(
        diagnostics,
        selected,
        args.output_dir,
        selected_by_period=selected_by_period,
    )
    (args.output_dir / "selected_checkpoint.txt").write_text(
        str(args.output_dir / "checkpoints" / selected.name) + "\n",
        encoding="utf-8",
    )
    print(
        "Selected checkpoints without gold: "
        + ", ".join(
            f"period {period}={diagnostic.name}"
            for period, diagnostic in selected_by_period.items()
        )
    )


if __name__ == "__main__":
    main()
