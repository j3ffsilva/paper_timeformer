"""Reading real-text period corpora and turning them into PyTorch datasets
for masked-language-model (MLM) training and word-level probing.

A "period corpus" is one period's worth of text (e.g. all documents from a
given decade), represented as `RealPeriodCorpus(period, documents)` where
`documents` is a list of token lists. `read_period_corpora` reads a whole
directory of periods at once; `build_vocabulary` then builds a shared
vocabulary across all periods so that the same token always maps to the same
vocabulary index regardless of which period it appears in.

The dataset classes turn token-id sequences into model inputs:

- `RealMLMDataset`: standard BERT-style MLM training examples (mask some
  tokens, predict them).
- `RealWordProbeDataset`: minimal `[CLS] word [MASK] [SEP]` probes, one per
  word, used to read off a model's prediction for a word in isolation.
- `RealTargetOccurrenceDataset`: one example per occurrence of a target word
  in the corpus, with that occurrence masked -- used to collect
  contextualized representations of target words "in the wild".
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset


SPECIAL_TOKENS = ("[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]")
# Lowercase words, optionally with a single internal underscore (e.g.
# "north_america") or a trailing apostrophe-suffix (e.g. "don't"). Anything
# else (digits, punctuation, ...) is dropped by `tokenize`.
TOKEN_RE = re.compile(r"[a-z]+(?:_[a-z]+)?(?:'[a-z]+)?")


@dataclass(frozen=True)
class RealPeriodCorpus:
    """One period's worth of text: `period` is a label (e.g. `"1950"`),
    `documents` is a list of documents, each a list of word tokens."""

    period: str
    documents: list[list[str]]


def tokenize(text: str) -> list[str]:
    """Lowercase `text` and extract all tokens matching `TOKEN_RE`.

    Example: `tokenize("The Queen's Speech (1952)")` ->
    `["the", "queen's", "speech"]` -- punctuation and digits are dropped.
    """
    return TOKEN_RE.findall(text.lower())


def read_period_corpora(input_dir: Path) -> list[RealPeriodCorpus]:
    """Read period corpora from files or period directories.

    Supported layouts (entries are processed in sorted order):

    - `input_dir/1950.txt`: one period per `.txt` file, one document per
      line, period label = file stem (`"1950"`).
    - `input_dir/1950/*.txt`: one period per subdirectory, one document per
      file (the whole file is one document), period label = directory name.

    Raises `FileNotFoundError` if `input_dir` contains neither.
    """
    corpora = []
    for path in sorted(input_dir.iterdir()):
        if path.is_file() and path.suffix.lower() == ".txt":
            documents = [
                tokenize(line)
                for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()
                if line.strip()
            ]
            corpora.append(RealPeriodCorpus(path.stem, [doc for doc in documents if doc]))
        elif path.is_dir():
            documents = [
                tokenize(file.read_text(encoding="utf-8", errors="ignore"))
                for file in sorted(path.glob("*.txt"))
            ]
            corpora.append(RealPeriodCorpus(path.name, [doc for doc in documents if doc]))
    if not corpora:
        raise FileNotFoundError(f"No .txt period files or period directories found under {input_dir}")
    return corpora


def build_vocabulary(
    corpora: list[RealPeriodCorpus],
    *,
    min_count: int = 5,
    max_vocab: int = 30_000,
    required_tokens: list[str] | None = None,
) -> tuple[list[str], dict[str, int]]:
    """Build a single vocabulary shared across all `corpora`.

    Tokens are counted across every document of every period. A token is
    kept if it occurs at least `min_count` times overall, or if it is in
    `required_tokens` (e.g. the target words being analyzed -- these must be
    in the vocabulary even if they are rare). The kept tokens are truncated
    to `max_vocab - len(SPECIAL_TOKENS)` (most frequent first), then any
    `required_tokens` still missing are appended.

    Returns `(vocab, token_to_id)` where `vocab[0:5] == list(SPECIAL_TOKENS)`
    (`["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"]`) and `token_to_id` is
    the inverse mapping.
    """
    required = set(required_tokens or [])
    counts = Counter(token for corpus in corpora for document in corpus.documents for token in document)
    kept = [
        token
        for token, count in counts.most_common()
        if count >= min_count or token in required
    ]
    kept = kept[: max(0, max_vocab - len(SPECIAL_TOKENS))]
    for token in sorted(required):
        if token not in kept:
            kept.append(token)
    vocab = list(SPECIAL_TOKENS) + [token for token in kept if token not in SPECIAL_TOKENS]
    token_to_id = {token: index for index, token in enumerate(vocab)}
    return vocab, token_to_id


def encode_document(tokens: list[str], token_to_id: dict[str, int]) -> list[int]:
    """Map each token to its vocabulary id, using `[UNK]` for any token not
    in `token_to_id`."""
    unk = token_to_id["[UNK]"]
    return [token_to_id.get(token, unk) for token in tokens]


def make_windows(encoded: list[int], seq_len: int, *, stride: int) -> list[list[int]]:
    """Split a token-id sequence into overlapping windows of content length
    `seq_len - 2` (the `-2` reserves room for `[CLS]`/`[SEP]`, added later).

    If the whole document already fits in one window, it is returned as a
    single chunk. Otherwise, windows start at `0, stride, 2*stride, ...`,
    plus one final window aligned to the end of the document (so the last
    few tokens are never left out even if `stride` doesn't divide evenly).

    Example: `encoded` has length 10, `seq_len=6` (so `content_len=4`),
    `stride=2`. Window starts would naturally be `0, 2, 4`, covering tokens
    `[0:4], [2:6], [4:8]` -- but that misses tokens `8` and `9`. The final
    start `10 - 4 = 6` is appended, giving a fourth window `[6:10]` so the
    document's tail is covered too.
    """
    if seq_len < 4:
        raise ValueError("seq_len must be at least 4")
    content_len = seq_len - 2
    if len(encoded) <= content_len:
        return [encoded]
    starts = list(range(0, len(encoded) - content_len + 1, stride))
    final_start = len(encoded) - content_len
    if starts[-1] != final_start:
        starts.append(final_start)
    return [encoded[start : start + content_len] for start in starts]


class RealMLMDataset(Dataset):
    """Standard BERT-style masked-language-model examples from one period's
    corpus.

    Each document is split into `[CLS] ... [SEP]`-wrapped windows via
    `make_windows`. Windows made up entirely of special tokens (which can
    only happen for an empty document) are dropped.

    Masking follows the usual BERT recipe, applied independently per
    example and per epoch (see `_generator`):

    - each non-special token is selected for masking independently with
      probability `mask_probability` (default 15%);
    - if *no* token happens to be selected, one candidate position is chosen
      at random, so every example always has at least one masked position;
    - for each selected position, the *original* token id becomes the
      training label (all other positions get label `-100`, i.e. "ignore in
      the loss"), and the input token is replaced:
      - with `[MASK]` with probability `mask_replace_probability` (default
        80%),
      - with a random vocabulary token with probability
        `random_replace_probability` (default 10%),
      - left unchanged otherwise (the remaining ~10%).
    """

    def __init__(
        self,
        corpus: RealPeriodCorpus,
        token_to_id: dict[str, int],
        *,
        period_idx: int,
        seq_len: int = 32,
        stride: int = 16,
        mask_probability: float = 0.15,
        mask_replace_probability: float = 0.8,
        random_replace_probability: float = 0.1,
        seed: int = 0,
    ) -> None:
        if not 0.0 < mask_probability <= 1.0:
            raise ValueError("mask_probability must be in (0, 1]")
        if mask_replace_probability < 0.0 or random_replace_probability < 0.0:
            raise ValueError("replacement probabilities must be non-negative")
        if mask_replace_probability + random_replace_probability > 1.0:
            raise ValueError("replacement probabilities must sum to at most 1")
        self.token_to_id = token_to_id
        self.period_idx = period_idx
        self.seq_len = seq_len
        self.mask_probability = mask_probability
        self.mask_replace_probability = mask_replace_probability
        self.random_replace_probability = random_replace_probability
        self.seed = seed
        self.epoch = 0
        self.pad_id = token_to_id["[PAD]"]
        self.cls_id = token_to_id["[CLS]"]
        self.sep_id = token_to_id["[SEP]"]
        self.mask_id = token_to_id["[MASK]"]
        self.special_ids = {
            token_to_id[token]
            for token in SPECIAL_TOKENS
            if token in token_to_id
        }
        self.replacement_ids = [
            token_id
            for token, token_id in token_to_id.items()
            if token not in SPECIAL_TOKENS
        ]
        if not self.replacement_ids:
            raise ValueError("vocabulary must contain at least one lexical token")
        self.windows = []
        for document in corpus.documents:
            encoded = encode_document(document, token_to_id)
            for window in make_windows(encoded, seq_len, stride=stride):
                if any(token_id not in self.special_ids for token_id in window):
                    self.windows.append(window)

    def set_epoch(self, epoch: int) -> None:
        """Change the epoch used to seed per-example masking (see
        `_generator`), so masking varies across training epochs while
        remaining reproducible for a given `(seed, epoch, index)`."""
        self.epoch = epoch

    def _generator(self, index: int) -> torch.Generator:
        # Combine the dataset seed, current epoch, period index and example
        # index into a single seed, using large odd multipliers to avoid
        # collisions between different (epoch, index) pairs. This makes
        # masking deterministic for a given (seed, epoch, period, index)
        # without needing to store per-example RNG state.
        generator = torch.Generator()
        item_seed = (
            self.seed
            + 1_000_003 * self.period_idx
            + 10_000_019 * self.epoch
            + 100_000_007 * index
        )
        generator.manual_seed(item_seed)
        return generator

    def _make_item(self, window: list[int], index: int) -> dict[str, Tensor]:
        ids = [self.cls_id] + window[: self.seq_len - 2] + [self.sep_id]
        ids += [self.pad_id] * (self.seq_len - len(ids))
        labels = [-100] * self.seq_len
        candidate_positions = [
            index
            for index, token_id in enumerate(ids)
            if token_id not in self.special_ids
        ]
        generator = self._generator(index)
        selected = [
            position
            for position in candidate_positions
            if float(torch.rand((), generator=generator)) < self.mask_probability
        ]
        if not selected:
            choice = int(torch.randint(len(candidate_positions), (1,), generator=generator))
            selected = [candidate_positions[choice]]
        for position in selected:
            labels[position] = ids[position]
            replacement_draw = float(torch.rand((), generator=generator))
            if replacement_draw < self.mask_replace_probability:
                ids[position] = self.mask_id
            elif replacement_draw < self.mask_replace_probability + self.random_replace_probability:
                replacement_index = int(
                    torch.randint(len(self.replacement_ids), (1,), generator=generator)
                )
                ids[position] = self.replacement_ids[replacement_index]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            # `epoch_idx` here is overloaded to mean "period index": the
            # model architectures this feeds into use it to pick a
            # per-period component (when `needs_time` is set).
            "epoch_idx": torch.tensor(self.period_idx, dtype=torch.long),
            "true_context": torch.tensor(0, dtype=torch.long),
            "p_n1": torch.tensor(0.0, dtype=torch.float32),
            "subject_idx": torch.tensor(0, dtype=torch.long),
        }

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        return self._make_item(self.windows[index], index)


class RealWordProbeDataset(Dataset):
    """One minimal `[CLS] word [MASK] [SEP]` example per word in `words`.

    Used to read off a model's prediction "for `word`" in a fixed, minimal
    context: the model sees `word` followed by `[MASK]`, and its prediction
    for the masked position can be inspected directly. Words not in the
    vocabulary are encoded as `[UNK]`.
    """

    def __init__(
        self,
        words: list[str],
        token_to_id: dict[str, int],
        *,
        period_idx: int,
        seq_len: int = 32,
    ) -> None:
        self.words = words
        self.token_to_id = token_to_id
        self.period_idx = period_idx
        self.seq_len = seq_len
        self.pad_id = token_to_id["[PAD]"]
        self.cls_id = token_to_id["[CLS]"]
        self.sep_id = token_to_id["[SEP]"]
        self.mask_id = token_to_id["[MASK]"]
        self.unk_id = token_to_id["[UNK]"]

    def __len__(self) -> int:
        return len(self.words)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        word_id = self.token_to_id.get(self.words[index], self.unk_id)
        ids = [self.cls_id, word_id, self.mask_id, self.sep_id]
        ids += [self.pad_id] * (self.seq_len - len(ids))
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "epoch_idx": torch.tensor(self.period_idx, dtype=torch.long),
            "word_idx": torch.tensor(index, dtype=torch.long),
        }


class RealTargetOccurrenceDataset(Dataset):
    """One example per occurrence of a target word in `corpus`, with that
    occurrence's position replaced by `[MASK]`.

    For each occurrence of a target word, a window of `seq_len - 2` tokens
    is taken from the surrounding document, centered on the target where
    possible (the target is placed `(seq_len - 2) // 2` tokens from the
    window's start, clamped so the window stays within the document), then
    wrapped in `[CLS] ... [SEP]` with the target position masked. This gives
    a batch of "the model's view of this specific occurrence of the target
    word", which can be encoded to get a contextualized representation of
    that occurrence.

    If `max_occurrences_per_target` is set, only the first that many
    occurrences of each target word are kept (in document order) -- useful
    to bound dataset size for very frequent targets.
    """

    def __init__(
        self,
        corpus: RealPeriodCorpus,
        target_words: list[str],
        token_to_id: dict[str, int],
        *,
        period_idx: int,
        seq_len: int = 32,
        max_occurrences_per_target: int | None = None,
    ) -> None:
        self.target_words = target_words
        self.target_to_index = {word: index for index, word in enumerate(target_words)}
        self.token_to_id = token_to_id
        self.period_idx = period_idx
        self.seq_len = seq_len
        self.pad_id = token_to_id["[PAD]"]
        self.cls_id = token_to_id["[CLS]"]
        self.sep_id = token_to_id["[SEP]"]
        self.mask_id = token_to_id["[MASK]"]
        self.items = []
        occurrence_counts = {word: 0 for word in target_words}
        target_set = set(target_words)
        for document in corpus.documents:
            for token_index, token in enumerate(document):
                if token not in target_set:
                    continue
                if (
                    max_occurrences_per_target is not None
                    and occurrence_counts[token] >= max_occurrences_per_target
                ):
                    continue
                self.items.append(self._make_item(document, token_index, self.target_to_index[token]))
                occurrence_counts[token] += 1

    def _make_item(self, document: list[str], token_index: int, word_index: int) -> dict[str, Tensor]:
        content_len = self.seq_len - 2
        left_budget = content_len // 2
        # Try to place the target `left_budget` tokens from the window
        # start (i.e. roughly centered), but clamp so the window doesn't run
        # past either end of the document.
        start = max(0, min(token_index - left_budget, len(document) - content_len))
        end = min(len(document), start + content_len)
        window = list(document[start:end])
        mask_pos_in_window = token_index - start
        encoded = encode_document(window, self.token_to_id)
        ids = [self.cls_id] + encoded + [self.sep_id]
        mask_pos = mask_pos_in_window + 1  # +1 for the leading [CLS]
        ids[mask_pos] = self.mask_id
        ids += [self.pad_id] * (self.seq_len - len(ids))
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "epoch_idx": torch.tensor(self.period_idx, dtype=torch.long),
            "word_idx": torch.tensor(word_index, dtype=torch.long),
            "mask_pos": torch.tensor(mask_pos, dtype=torch.long),
        }

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        return self.items[index]
