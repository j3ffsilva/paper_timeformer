#!/usr/bin/env python3
"""Extract `token@time` profile caches from a fixed encoder over N periods.

Implements Fase A item 1 of `docs/39-token_time_analysis_framework.md`:
"extrair perfis D0/D1 no `bert-tiny` integral".

Design notes (see `history/conceitos/08-desenhos_temporais_e_reguas.md` and
`history/09-bert_tiny_option_d_e_l2sp.md`):

- "Encoder fixo": a single checkpoint is applied to every period's corpus, so
  that representations across periods live in the same coordinate system
  (`theta1_d0` vs. `theta1_d1`, not the "diagonal" `theta0_d0` vs `theta1_d1`).
- Targets that tokenize into a single WordPiece accumulate naturally into
  `sums`/`counts` for that vocabulary id during the general pass. Targets
  that split into multiple WordPieces (e.g. "graft" -> "graf", "##t") get a
  "virtual" vocabulary slot whose statistics are the mean-pooled hidden
  state over their subtoken positions, per occurrence (capítulo 09, Option D:
  "as palavras-alvo passam a ser representadas pela média de seus subtokens
  WordPiece").

Generic over corpus, target list and checkpoint (Fase A.6 smoke test): no
SemEval-specific paths are hardcoded, all are CLI arguments.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from timeformers.bert_continual import read_tokenized_documents, strip_pos_suffix  # noqa: E402
from timeformers.token_time_occurrences import OccurrenceCache  # noqa: E402
from timeformers.token_time_statistics import PeriodStatistics  # noqa: E402


def read_targets(path: Path) -> list[str]:
    words = []
    for line in path.read_text(encoding="utf-8").splitlines():
        word = strip_pos_suffix(line.strip())
        if word:
            words.append(word)
    return words


def build_target_ids(
    tokenizer,
    targets: list[str],
    *,
    vocab_size: int,
) -> tuple[dict[str, int], dict[int, list[int]], list[str]]:
    """Map each target to a vocabulary id.

    Single-WordPiece targets reuse their native id. Multi-WordPiece targets
    get a new "virtual" id (appended after `vocab_size`), and are returned in
    `multi_subtoken_targets` as `{virtual_id: [subtoken_id, ...]}` so the
    extraction loop knows to mean-pool over their occurrences.
    """
    target_ids: dict[str, int] = {}
    multi_subtoken_targets: dict[int, list[int]] = {}
    extra_vocab: list[str] = []
    for word in targets:
        subtoken_ids = tokenizer(word, add_special_tokens=False)["input_ids"]
        if len(subtoken_ids) == 1:
            target_ids[word] = subtoken_ids[0]
        else:
            virtual_id = vocab_size + len(extra_vocab)
            extra_vocab.append(word)
            multi_subtoken_targets[virtual_id] = subtoken_ids
            target_ids[word] = virtual_id
    return target_ids, multi_subtoken_targets, extra_vocab


def document_standalone_counts(
    documents: list[list[str]],
    tokenizer,
    continuation_mask: Tensor,
    total_vocab: int,
) -> Tensor:
    """`(total_vocab,)`: how many times each vocabulary item occurs as a
    standalone word (see `PeriodStatistics`/`standalone_counts`), counted
    from each document's full WordPiece tokenization.

    This must run on whole, unwindowed documents: a word whose subtokens
    straddle an `encode_windows` chunk boundary would otherwise have its
    first piece misclassified as standalone (its continuation piece falls
    in the next window, out of view).
    """
    standalone_counts = torch.zeros(total_vocab, dtype=torch.long)
    for tokens in documents:
        encoded = tokenizer(
            tokens, is_split_into_words=True, add_special_tokens=False, truncation=False
        )["input_ids"]
        if not encoded:
            continue
        ids = torch.tensor(encoded, dtype=torch.long)
        is_continuation = continuation_mask[ids]
        next_is_continuation = torch.zeros_like(is_continuation)
        next_is_continuation[:-1] = is_continuation[1:]
        standalone = ~is_continuation & ~next_is_continuation
        standalone_counts += torch.bincount(ids[standalone], minlength=total_vocab)
    return standalone_counts


def encode_windows_with_doc_index(
    documents: list[list[str]],
    tokenizer,
    *,
    seq_len: int,
    stride: int,
) -> tuple[list[list[int]], list[int]]:
    """Same chunking as `timeformers.bert_continual.encode_windows`, but also
    returns each window's source document index (its position in
    `documents`), for `OccurrenceCache.doc_index`."""
    if seq_len < 4:
        raise ValueError("seq_len must be at least 4")
    content_len = seq_len - 2
    windows: list[list[int]] = []
    doc_indices: list[int] = []
    for doc_index, tokens in enumerate(documents):
        encoded = tokenizer(
            tokens, is_split_into_words=True, add_special_tokens=False, truncation=False
        )["input_ids"]
        if not encoded:
            continue
        if len(encoded) <= content_len:
            chunks = [encoded]
        else:
            starts = list(range(0, len(encoded) - content_len + 1, stride))
            final_start = len(encoded) - content_len
            if starts[-1] != final_start:
                starts.append(final_start)
            chunks = [encoded[start : start + content_len] for start in starts]
        windows.extend(chunks)
        doc_indices.extend([doc_index] * len(chunks))
    return windows, doc_indices


@torch.no_grad()
def extract_context_statistics(
    model,
    tokenizer,
    windows: list[list[int]],
    doc_indices: list[int],
    *,
    total_vocab: int,
    target_ids: dict[str, int],
    multi_subtoken_targets: dict[int, list[int]],
    layers: tuple[int, ...],
    batch_size: int,
    device: str,
) -> tuple[PeriodStatistics, OccurrenceCache]:
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    pad_id = tokenizer.pad_token_id
    hidden_size = model.config.hidden_size

    sums = {
        f"layer_{layer}": torch.zeros(total_vocab, hidden_size, dtype=torch.float32)
        for layer in layers
    }
    sum_sq = {
        f"layer_{layer}": torch.zeros(total_vocab, dtype=torch.float32)
        for layer in layers
    }
    counts = torch.zeros(total_vocab, dtype=torch.long)
    standalone_counts = torch.zeros(total_vocab, dtype=torch.long)

    occurrence_ids = set(target_ids.values())
    real_occurrence_ids = occurrence_ids - set(multi_subtoken_targets.keys())
    occurrence_layers: dict[int, dict[str, list[Tensor]]] = {
        target_id: {f"layer_{layer}": [] for layer in layers} for target_id in occurrence_ids
    }
    occurrence_doc_index: dict[int, list[Tensor]] = {target_id: [] for target_id in occurrence_ids}

    for start in range(0, len(windows), batch_size):
        batch = windows[start : start + batch_size]
        longest = max(len(window) for window in batch)
        input_ids = []
        attention_mask = []
        content_lengths = []
        for window in batch:
            ids = [cls_id, *window, sep_id]
            padding = (longest + 2) - len(ids)
            input_ids.append(ids + [pad_id] * padding)
            attention_mask.append([1] * len(ids) + [0] * padding)
            content_lengths.append(len(window))

        input_ids_t = torch.tensor(input_ids, device=device)
        attention_mask_t = torch.tensor(attention_mask, device=device)
        outputs = model.bert(
            input_ids=input_ids_t,
            attention_mask=attention_mask_t,
            output_hidden_states=True,
            return_dict=True,
        )

        valid = (
            attention_mask_t.bool()
            & input_ids_t.ne(cls_id)
            & input_ids_t.ne(sep_id)
        )
        flat_ids = input_ids_t[valid].cpu()
        valid_hidden_by_layer: dict[str, Tensor] = {}
        for layer in layers:
            hidden = outputs.hidden_states[layer]
            valid_hidden = hidden[valid].float().cpu()
            valid_hidden_by_layer[f"layer_{layer}"] = valid_hidden
            sums[f"layer_{layer}"].index_add_(0, flat_ids, valid_hidden)
            sum_sq[f"layer_{layer}"].index_add_(0, flat_ids, valid_hidden.pow(2).sum(dim=-1))
        counts += torch.bincount(flat_ids, minlength=total_vocab)

        if real_occurrence_ids:
            batch_doc_index = torch.tensor(doc_indices[start : start + len(batch)], dtype=torch.long)
            doc_index_grid = batch_doc_index.unsqueeze(1).expand(-1, valid.shape[1])
            flat_doc_index = doc_index_grid[valid.cpu()]
            for target_id in real_occurrence_ids:
                mask = flat_ids == target_id
                if mask.any():
                    occurrence_doc_index[target_id].append(flat_doc_index[mask])
                    for layer in layers:
                        occurrence_layers[target_id][f"layer_{layer}"].append(
                            valid_hidden_by_layer[f"layer_{layer}"][mask]
                        )

        if multi_subtoken_targets:
            for batch_index, ids in enumerate(input_ids):
                content_length = content_lengths[batch_index]
                for virtual_id, subtoken_ids in multi_subtoken_targets.items():
                    n = len(subtoken_ids)
                    for position in range(1, content_length - n + 2):
                        if ids[position : position + n] == subtoken_ids:
                            for layer in layers:
                                hidden = outputs.hidden_states[layer][
                                    batch_index, position : position + n
                                ].mean(dim=0)
                                hidden_f = hidden.float().cpu()
                                sums[f"layer_{layer}"][virtual_id] += hidden_f
                                sum_sq[f"layer_{layer}"][virtual_id] += hidden_f.pow(2).sum()
                                occurrence_layers[virtual_id][f"layer_{layer}"].append(hidden_f.unsqueeze(0))
                            occurrence_doc_index[virtual_id].append(
                                torch.tensor([doc_indices[start + batch_index]], dtype=torch.long)
                            )
                            counts[virtual_id] += 1
                            standalone_counts[virtual_id] += 1

    occurrence_targets: dict[int, dict[str, Tensor]] = {}
    for target_id in occurrence_ids:
        tensors: dict[str, Tensor] = {
            "doc_index": (
                torch.cat(occurrence_doc_index[target_id])
                if occurrence_doc_index[target_id]
                else torch.empty(0, dtype=torch.long)
            )
        }
        for layer in layers:
            layer_chunks = occurrence_layers[target_id][f"layer_{layer}"]
            tensors[f"layer_{layer}"] = (
                torch.cat(layer_chunks, dim=0) if layer_chunks else torch.empty(0, hidden_size, dtype=torch.float32)
            )
        occurrence_targets[target_id] = tensors

    stats = PeriodStatistics(counts=counts, sums=sums, sum_sq=sum_sq, standalone_counts=standalone_counts)
    occurrences = OccurrenceCache(targets=occurrence_targets)
    return stats, occurrences


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--period-files", nargs="+", required=True)
    parser.add_argument("--targets-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--stride", type=int, default=None, help="default: --seq-len (no overlap)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--layers", nargs="*", type=int, default=[1, 2])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--reuse-cache", action="store_true")
    args = parser.parse_args()
    stride = args.stride or args.seq_len

    from transformers import AutoModelForMaskedLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForMaskedLM.from_pretrained(args.checkpoint)
    model.eval().to(args.device)

    vocab_size = model.config.vocab_size
    targets = read_targets(args.targets_file)
    target_ids, multi_subtoken_targets, extra_vocab = build_target_ids(
        tokenizer, targets, vocab_size=vocab_size
    )
    total_vocab = vocab_size + len(extra_vocab)
    vocab = [tokenizer.convert_ids_to_tokens(token_id) for token_id in range(vocab_size)] + extra_vocab
    continuation_mask = torch.zeros(total_vocab, dtype=torch.bool)
    for token_id in range(vocab_size):
        if vocab[token_id].startswith("##"):
            continuation_mask[token_id] = True

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "vocab.json").write_text(
        json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "targets.json").write_text(
        json.dumps(targets, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "target_ids.json").write_text(
        json.dumps(target_ids, indent=2), encoding="utf-8"
    )
    (args.output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "corpus_dir": str(args.corpus_dir),
                "period_files": [f"d{index}" for index in range(len(args.period_files))],
                "period_corpus_files": args.period_files,
                "seq_len": args.seq_len,
                "stride": stride,
                "layers": args.layers,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    cache_dir = args.output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    for period_index, filename in enumerate(args.period_files):
        cache_path = cache_dir / f"theta_d{period_index}.pt"
        if args.reuse_cache and cache_path.exists():
            print(json.dumps({"period": period_index, "cache": "reused"}), flush=True)
            continue
        documents = read_tokenized_documents(args.corpus_dir / filename)
        windows, doc_indices = encode_windows_with_doc_index(
            documents, tokenizer, seq_len=args.seq_len, stride=stride
        )
        stats, occurrences = extract_context_statistics(
            model,
            tokenizer,
            windows,
            doc_indices,
            total_vocab=total_vocab,
            target_ids=target_ids,
            multi_subtoken_targets=multi_subtoken_targets,
            layers=tuple(args.layers),
            batch_size=args.batch_size,
            device=args.device,
        )
        stats.standalone_counts = stats.standalone_counts + document_standalone_counts(
            documents, tokenizer, continuation_mask, total_vocab
        )
        stats.save(cache_path)
        occurrences.save(cache_dir / f"occurrences_d{period_index}.pt")
        print(
            json.dumps(
                {
                    "period": period_index,
                    "file": filename,
                    "n_windows": len(windows),
                    "n_occurrences": int(stats.counts.sum()),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
