from __future__ import annotations

import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset


POS_SUFFIX = re.compile(r"_(nn|vb)$")


def strip_pos_suffix(token: str) -> str:
    return POS_SUFFIX.sub("", token)


def read_tokenized_documents(path: Path) -> list[list[str]]:
    documents = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            tokens = [strip_pos_suffix(token) for token in line.split()]
            if tokens:
                documents.append(tokens)
    return documents


def split_documents(
    documents: list[list[str]],
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[list[list[str]], list[list[str]]]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")
    indices = list(range(len(documents)))
    random.Random(seed).shuffle(indices)
    n_validation = max(1, round(len(indices) * validation_fraction))
    validation_indices = set(indices[:n_validation])
    train = [document for index, document in enumerate(documents) if index not in validation_indices]
    validation = [document for index, document in enumerate(documents) if index in validation_indices]
    if not train:
        raise ValueError("validation split left no training documents")
    return train, validation


def random_pseudo_periods(
    period_documents: list[list[list[str]]],
    *,
    seed: int,
) -> list[list[list[str]]]:
    sizes = [len(documents) for documents in period_documents]
    pooled = [document for documents in period_documents for document in documents]
    random.Random(seed).shuffle(pooled)
    result = []
    start = 0
    for size in sizes:
        result.append(pooled[start : start + size])
        start += size
    return result


def encode_windows(
    documents: list[list[str]],
    tokenizer,
    *,
    seq_len: int,
    stride: int,
) -> list[list[int]]:
    if seq_len < 4:
        raise ValueError("seq_len must be at least 4")
    content_len = seq_len - 2
    windows = []
    for tokens in documents:
        encoded = tokenizer(
            tokens,
            is_split_into_words=True,
            add_special_tokens=False,
            truncation=False,
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
    return windows


class DynamicWordPieceMLMDataset(Dataset):
    def __init__(
        self,
        windows: list[list[int]],
        tokenizer,
        *,
        seq_len: int,
        mask_probability: float = 0.15,
        mask_replace_probability: float = 0.8,
        random_replace_probability: float = 0.1,
        seed: int = 0,
    ) -> None:
        if not 0.0 < mask_probability <= 1.0:
            raise ValueError("mask_probability must be in (0, 1]")
        if mask_replace_probability + random_replace_probability > 1.0:
            raise ValueError("replacement probabilities must sum to at most 1")
        self.windows = windows
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.mask_probability = mask_probability
        self.mask_replace_probability = mask_replace_probability
        self.random_replace_probability = random_replace_probability
        self.seed = seed
        self.epoch = 0
        self.pad_id = tokenizer.pad_token_id
        self.mask_id = tokenizer.mask_token_id
        self.special_ids = set(tokenizer.all_special_ids)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _generator(self, index: int) -> torch.Generator:
        generator = torch.Generator()
        generator.manual_seed(self.seed + 10_000_019 * self.epoch + 100_000_007 * index)
        return generator

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        ids = self.tokenizer.build_inputs_with_special_tokens(self.windows[index])
        ids = ids[: self.seq_len]
        attention_mask = [1] * len(ids)
        ids += [self.pad_id] * (self.seq_len - len(ids))
        attention_mask += [0] * (self.seq_len - len(attention_mask))
        labels = [-100] * self.seq_len
        candidates = [
            position
            for position, token_id in enumerate(ids)
            if attention_mask[position] and token_id not in self.special_ids
        ]
        generator = self._generator(index)
        selected = [
            position
            for position in candidates
            if float(torch.rand((), generator=generator)) < self.mask_probability
        ]
        if not selected:
            selected = [candidates[int(torch.randint(len(candidates), (1,), generator=generator))]]
        for position in selected:
            labels[position] = ids[position]
            draw = float(torch.rand((), generator=generator))
            if draw < self.mask_replace_probability:
                ids[position] = self.mask_id
            elif draw < self.mask_replace_probability + self.random_replace_probability:
                while True:
                    replacement = int(
                        torch.randint(self.tokenizer.vocab_size, (1,), generator=generator)
                    )
                    if replacement not in self.special_ids:
                        ids[position] = replacement
                        break
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


@dataclass(frozen=True)
class CheckpointDiagnostic:
    name: str
    period: int
    epoch_fraction: float
    global_step: int
    train_loss: float | None
    train_mlm_loss: float | None
    train_l2sp_penalty: float | None
    validation_losses: list[float]
    mean_validation_loss: float
    selection_loss: float
    parameter_relative_l2: float
    layer2_relative_l2: float
    anchor_hidden_cosine: float
    embedding_pairwise_cosine: float


def assert_weight_tying(model: nn.Module) -> None:
    input_weight = model.get_input_embeddings().weight
    output_weight = model.get_output_embeddings().weight
    if not model.config.tie_word_embeddings:
        raise ValueError("Model config has tie_word_embeddings=False")
    if input_weight.data_ptr() != output_weight.data_ptr():
        raise ValueError("Input embeddings and MLM decoder are not tied")


def snapshot_parameters(model: nn.Module) -> dict[str, Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
    }


def snapshot_named_parameters(
    model: nn.Module,
    *,
    prefix: str,
    device: torch.device | None = None,
) -> dict[str, Tensor]:
    return {
        name: parameter.detach().clone().to(device=device or parameter.device)
        for name, parameter in model.named_parameters()
        if name.startswith(prefix)
    }


def relative_l2_sp_penalty(
    model: nn.Module,
    reference: dict[str, Tensor],
) -> Tensor:
    if not reference:
        raise ValueError("L2-SP reference is empty")
    squared_delta = None
    squared_reference = None
    parameters = dict(model.named_parameters())
    for name, initial in reference.items():
        parameter = parameters[name]
        initial = initial.to(device=parameter.device, dtype=parameter.dtype)
        delta_term = (parameter - initial).pow(2).sum()
        reference_term = initial.pow(2).sum()
        squared_delta = delta_term if squared_delta is None else squared_delta + delta_term
        squared_reference = (
            reference_term
            if squared_reference is None
            else squared_reference + reference_term
        )
    return squared_delta / squared_reference.clamp_min(torch.finfo(squared_reference.dtype).eps)


@torch.no_grad()
def parameter_relative_l2(
    model: nn.Module,
    initial: dict[str, Tensor],
    *,
    prefix: str | None = None,
) -> float:
    squared_delta = 0.0
    squared_initial = 0.0
    for name, parameter in model.named_parameters():
        if prefix is not None and not name.startswith(prefix):
            continue
        reference = initial[name].to(device=parameter.device, dtype=parameter.dtype)
        squared_delta += float((parameter - reference).pow(2).sum())
        squared_initial += float(reference.pow(2).sum())
    return math.sqrt(squared_delta / max(squared_initial, torch.finfo(torch.float32).eps))


@torch.no_grad()
def anchor_hidden_cosine(
    model: nn.Module,
    anchor_batch: dict[str, Tensor],
    initial_hidden: Tensor,
    *,
    device: torch.device,
) -> float:
    model.eval()
    outputs = model.bert(
        input_ids=anchor_batch["input_ids"].to(device),
        attention_mask=anchor_batch["attention_mask"].to(device),
        return_dict=True,
    )
    hidden = outputs.last_hidden_state
    mask = anchor_batch["attention_mask"].to(device).bool().unsqueeze(-1)
    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
    return float(F.cosine_similarity(pooled, initial_hidden.to(device), dim=1).mean())


@torch.no_grad()
def embedding_pairwise_cosine(model: nn.Module, token_ids: Tensor) -> float:
    vectors = F.normalize(model.get_input_embeddings().weight[token_ids], dim=1)
    similarities = vectors @ vectors.T
    upper = torch.triu_indices(len(token_ids), len(token_ids), offset=1, device=vectors.device)
    return float(similarities[upper[0], upper[1]].mean())


@torch.no_grad()
def evaluate_mlm_loss(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for batch in loader:
        outputs = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            labels=batch["labels"].to(device),
        )
        total_loss += float(outputs.loss)
        n_batches += 1
    return total_loss / max(n_batches, 1)


def select_checkpoint(
    diagnostics: list[CheckpointDiagnostic],
    *,
    loss_tolerance: float = 0.01,
) -> CheckpointDiagnostic:
    if not diagnostics:
        raise ValueError("No checkpoint diagnostics available")
    best_loss = min(diagnostic.selection_loss for diagnostic in diagnostics)
    eligible = [
        diagnostic
        for diagnostic in diagnostics
        if diagnostic.selection_loss <= best_loss * (1.0 + loss_tolerance)
    ]
    return max(
        eligible,
        key=lambda diagnostic: (
            diagnostic.anchor_hidden_cosine,
            -diagnostic.parameter_relative_l2,
        ),
    )


def write_diagnostics(
    diagnostics: list[CheckpointDiagnostic],
    selected: CheckpointDiagnostic,
    output_dir: Path,
    *,
    selected_by_period: dict[int, CheckpointDiagnostic] | None = None,
) -> None:
    payload = {
        "selection_rule": (
            "Among checkpoints within 1% of the best mean validation loss, "
            "maximize anchor hidden cosine; break ties by minimum parameter relative L2."
        ),
        "selected_checkpoint": selected.name,
        "selected_by_period": {
            str(period): diagnostic.name
            for period, diagnostic in (selected_by_period or {}).items()
        },
        "checkpoints": [diagnostic.__dict__ for diagnostic in diagnostics],
    }
    (output_dir / "diagnostics.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
