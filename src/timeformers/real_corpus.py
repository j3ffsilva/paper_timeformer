from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset


SPECIAL_TOKENS = ("[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]")
TOKEN_RE = re.compile(r"[a-z]+(?:_[a-z]+)?(?:'[a-z]+)?")


@dataclass(frozen=True)
class RealPeriodCorpus:
    period: str
    documents: list[list[str]]


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def read_period_corpora(input_dir: Path) -> list[RealPeriodCorpus]:
    """Read period corpora from files or period directories.

    Supported layouts:
    - input_dir/1950.txt
    - input_dir/1950/*.txt
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
    unk = token_to_id["[UNK]"]
    return [token_to_id.get(token, unk) for token in tokens]


def make_windows(encoded: list[int], seq_len: int, *, stride: int) -> list[list[int]]:
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
        self.epoch = epoch

    def _generator(self, index: int) -> torch.Generator:
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
        start = max(0, min(token_index - left_budget, len(document) - content_len))
        end = min(len(document), start + content_len)
        window = list(document[start:end])
        mask_pos_in_window = token_index - start
        encoded = encode_document(window, self.token_to_id)
        ids = [self.cls_id] + encoded + [self.sep_id]
        mask_pos = mask_pos_in_window + 1
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
