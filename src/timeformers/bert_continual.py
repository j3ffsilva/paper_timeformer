"""Continual fine-tuning of a pretrained BERT-tiny encoder on real-text
period corpora, plus the diagnostics used to pick a checkpoint after
fine-tuning.

Unlike `real_corpus.py`/`real_models.py` (which build a custom small
Transformer and vocabulary from scratch), this module works with a
pretrained Hugging Face BERT model and its own WordPiece tokenizer. The goal
of fine-tuning here is "continual learning": adapt the encoder slightly to a
new period's text via masked-language-modeling, while keeping it close
enough to its starting point that token representations remain comparable
across periods (the `token@time` framework, `relational.py`, then measures
exactly that kind of representational change).

Two groups of functionality:

1. Data preparation: `read_tokenized_documents`, `split_documents`,
   `random_pseudo_periods`, `encode_windows`, `DynamicWordPieceMLMDataset`.
2. Fine-tuning diagnostics and checkpoint selection: `CheckpointDiagnostic`,
   `assert_weight_tying`, `snapshot_parameters`,
   `snapshot_named_parameters`, `relative_l2_sp_penalty`,
   `parameter_relative_l2`, `anchor_hidden_cosine`,
   `embedding_pairwise_cosine`, `evaluate_mlm_loss`, `select_checkpoint`,
   `write_diagnostics`.
"""

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


# Some corpora tag tokens with a part-of-speech suffix, e.g. "bank_nn" (noun)
# or "bank_vb" (verb). `strip_pos_suffix` removes that tag so the token can
# be fed to a standard WordPiece tokenizer as plain text.
POS_SUFFIX = re.compile(r"_(nn|vb)$")


def strip_pos_suffix(token: str) -> str:
    """Remove a trailing `"_nn"` or `"_vb"` part-of-speech tag, if present.

    Example: `strip_pos_suffix("bank_nn")` -> `"bank"`;
    `strip_pos_suffix("running")` -> `"running"` (unchanged).
    """
    return POS_SUFFIX.sub("", token)


def read_tokenized_documents(path: Path) -> list[list[str]]:
    """Read a corpus file where each non-empty line is one document of
    whitespace-separated, POS-tagged tokens.

    Each token has its POS suffix stripped via `strip_pos_suffix`. Empty
    lines (after stripping) produce no document.
    """
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
    """Randomly split `documents` into `(train, validation)` sets.

    `validation_fraction` (in `(0, 1)`, exclusive) controls the size of the
    validation set: `round(len(documents) * validation_fraction)`, but at
    least 1 document. The split is shuffled using `seed`, so it is
    reproducible. Raises `ValueError` if the validation set would consume
    every document, leaving none for training.
    """
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
    """Build "pseudo-periods" with the same sizes as `period_documents`, but
    by pooling all documents from every period together and reshuffling them
    with `seed`.

    Used as a random control: if a period's documents are replaced by a
    random sample of the same size from the pooled corpus, any
    representational change measured between "periods" should be due to
    chance alone, not to genuine temporal signal. The returned list has the
    same lengths as `period_documents` (`len(result[i]) ==
    len(period_documents[i])`), but document *contents* are shuffled across
    periods.
    """
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
    """Tokenize each document with `tokenizer` (WordPiece, no special
    tokens added yet) and split it into overlapping windows of
    `seq_len - 2` subword-token ids (the `-2` reserves room for
    `[CLS]`/`[SEP]`, added later by the dataset/model).

    Documents that tokenize to nothing are skipped. If a document fits in
    one window, it is returned as-is. Otherwise, windows start at
    `0, stride, 2*stride, ...`, plus a final window aligned to the end of
    the document, so the tail is never dropped (same strategy as
    `real_corpus.make_windows`).
    """
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
    """Standard BERT-style MLM examples over pre-tokenized WordPiece windows
    (as produced by `encode_windows`), using `tokenizer`'s special tokens.

    The masking recipe is the same as `real_corpus.RealMLMDataset` (each
    non-special token is masked independently with `mask_probability`, with
    at least one masked position guaranteed; masked positions are replaced
    with `[MASK]` / a random token / left unchanged with probabilities
    `mask_replace_probability` / `random_replace_probability` / the
    remainder). The only difference is that special-token ids and vocabulary
    size come from `tokenizer` instead of a custom vocabulary, and masking
    is "dynamic": `set_epoch` lets the same window be re-masked differently
    each epoch.
    """

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
        """Change the epoch used to seed per-example masking (see
        `_generator`), so masking varies across training epochs while
        remaining reproducible for a given `(seed, epoch, index)`."""
        self.epoch = epoch

    def _generator(self, index: int) -> torch.Generator:
        # Combine the dataset seed, current epoch and example index into a
        # single seed using large odd multipliers, so masking is
        # deterministic per (seed, epoch, index) without per-example state.
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
    """Diagnostics recorded for one fine-tuning checkpoint, used by
    `select_checkpoint` to pick the checkpoint that best balances "learned
    the new period's text" against "stayed close to the starting point".

    - `name`: checkpoint identifier (e.g. `"period1_epoch2"`).
    - `period`, `epoch_fraction`, `global_step`: where in training this
      checkpoint was taken.
    - `train_loss`, `train_mlm_loss`, `train_l2sp_penalty`: training-time
      loss components (`train_loss = train_mlm_loss + lambda *
      train_l2sp_penalty`, if L2-SP regularization is used).
    - `validation_losses`, `mean_validation_loss`: held-out MLM loss, e.g.
      one value per period evaluated.
    - `selection_loss`: the loss value `select_checkpoint` ranks checkpoints
      by (typically `mean_validation_loss`).
    - `parameter_relative_l2`: see `parameter_relative_l2` -- how far the
      model's parameters have moved from their initial values, relative to
      the initial values' own magnitude.
    - `layer2_relative_l2`: same, restricted to layer-2 parameters.
    - `anchor_hidden_cosine`: see `anchor_hidden_cosine` -- how similar this
      checkpoint's representations of a fixed anchor batch are to the
      initial checkpoint's.
    - `embedding_pairwise_cosine`: see `embedding_pairwise_cosine` -- average
      pairwise similarity among a fixed set of token embeddings (a measure
      of embedding-space "collapse").
    """

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
    """Raise `ValueError` unless `model`'s input embeddings and MLM output
    projection share the same underlying storage ("weight tying").

    Many BERT-like models are configured to tie these two weight matrices
    (`config.tie_word_embeddings = True`); this is a sanity check that the
    tying survived model construction/loading, by comparing the tensors'
    memory addresses (`data_ptr()`).
    """
    input_weight = model.get_input_embeddings().weight
    output_weight = model.get_output_embeddings().weight
    if not model.config.tie_word_embeddings:
        raise ValueError("Model config has tie_word_embeddings=False")
    if input_weight.data_ptr() != output_weight.data_ptr():
        raise ValueError("Input embeddings and MLM decoder are not tied")


def snapshot_parameters(model: nn.Module) -> dict[str, Tensor]:
    """`{parameter_name: detached CPU copy}` for every parameter in `model`.

    Used to capture the model's state before fine-tuning, so later
    checkpoints can be compared back to this snapshot (see
    `parameter_relative_l2`, `relative_l2_sp_penalty`)."""
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
    """Like `snapshot_parameters`, but only for parameters whose name starts
    with `prefix` (e.g. `"bert.encoder.layer.1."` for just layer 2's
    parameters), and kept on `device` (or each parameter's own device, by
    default) instead of being moved to CPU."""
    return {
        name: parameter.detach().clone().to(device=device or parameter.device)
        for name, parameter in model.named_parameters()
        if name.startswith(prefix)
    }


def relative_l2_sp_penalty(
    model: nn.Module,
    reference: dict[str, Tensor],
) -> Tensor:
    """L2-SP ("L2 starting point") regularization penalty: how far `model`'s
    current parameters have drifted from `reference` (typically a snapshot
    of the pretrained model before fine-tuning), relative to the overall
    magnitude of `reference`.

        penalty = sum_p ||p_now - p_ref||^2  /  sum_p ||p_ref||^2

    Adding `lambda * penalty` to the training loss encourages the model to
    stay close to its pretrained initialization while still fitting the new
    period's MLM objective. Returns a scalar tensor (with gradient, so it
    can be backpropagated); raises `ValueError` if `reference` is empty.
    """
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
    """Same ratio as `relative_l2_sp_penalty`, but as a plain float (no
    gradient) and under the square root:

        sqrt( sum_p ||p_now - p_init||^2  /  sum_p ||p_init||^2 )

    If `prefix` is given, only parameters whose name starts with `prefix`
    are included (e.g. to measure drift of a single layer). This is the
    diagnostic reported as `parameter_relative_l2` /
    `layer2_relative_l2` in `CheckpointDiagnostic`: "what fraction of the
    initial parameter norm has the model moved by, overall".
    """
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
    """How similar `model`'s representation of a fixed `anchor_batch` is to
    `initial_hidden` (the same batch's representation from the model
    *before* fine-tuning).

    `anchor_batch` is encoded with `model.bert(...)`, and each sequence's
    hidden states are mean-pooled over non-padding positions (using
    `anchor_batch["attention_mask"]`) to get one vector per sequence. The
    result is the average cosine similarity, across the batch, between
    these pooled vectors and `initial_hidden`.

    `1.0` means fine-tuning hasn't changed how the model represents these
    anchor sentences at all; lower values mean the representation space has
    shifted. This is the `anchor_hidden_cosine` diagnostic in
    `CheckpointDiagnostic`, used by `select_checkpoint` to avoid checkpoints
    that drifted too far from the pretrained representation space.
    """
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
    """Average pairwise cosine similarity among the input-embedding vectors
    of `token_ids`.

    Takes the rows of `model`'s input embedding matrix at `token_ids`,
    normalizes each to unit length, computes all pairwise cosine
    similarities, and averages the entries above the diagonal (each
    unordered pair counted once, excluding self-similarity).

    A value close to `1` means these tokens' embeddings have become nearly
    indistinguishable from each other ("representation collapse"); this is
    the `embedding_pairwise_cosine` diagnostic in `CheckpointDiagnostic`,
    used as an early-warning signal during checkpoint selection.
    """
    vectors = F.normalize(model.get_input_embeddings().weight[token_ids], dim=1)
    similarities = vectors @ vectors.T
    upper = torch.triu_indices(len(token_ids), len(token_ids), offset=1, device=vectors.device)
    return float(similarities[upper[0], upper[1]].mean())


@torch.no_grad()
def evaluate_mlm_loss(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Mean MLM loss of `model` over every batch in `loader`, with the model
    in `eval()` mode and no gradient tracking. Returns `0.0` if `loader` is
    empty."""
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
    """Pick the checkpoint that best balances fitting the new period's text
    against staying close to the pretrained starting point.

    Selection rule (also recorded in `write_diagnostics`'s output):

    1. Find the best (lowest) `selection_loss` among `diagnostics`.
    2. Keep only checkpoints within `loss_tolerance` (relative, default 1%)
       of that best loss -- i.e. checkpoints that fit about as well as the
       single best one.
    3. Among those, pick the one with the highest `anchor_hidden_cosine`
       (closest to the pretrained representation space), breaking ties by
       the lowest `parameter_relative_l2` (smallest parameter drift).

    Raises `ValueError` if `diagnostics` is empty.
    """
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
    """Write `diagnostics.json` to `output_dir`, recording the selection
    rule used by `select_checkpoint`, the overall `selected` checkpoint, an
    optional per-period selection (`selected_by_period`), and the full list
    of `diagnostics` (one dict per `CheckpointDiagnostic`)."""
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
