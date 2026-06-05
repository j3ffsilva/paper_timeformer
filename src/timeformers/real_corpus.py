from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset


SPECIAL_TOKENS = ("[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]")
TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")


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
            documents = [tokenize(path.read_text(encoding="utf-8", errors="ignore"))]
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
    return [encoded[start : start + content_len] for start in range(0, len(encoded) - content_len + 1, stride)]


class RealMLMDataset(Dataset):
    def __init__(
        self,
        corpus: RealPeriodCorpus,
        token_to_id: dict[str, int],
        *,
        period_idx: int,
        seq_len: int = 32,
        stride: int = 16,
    ) -> None:
        self.token_to_id = token_to_id
        self.period_idx = period_idx
        self.seq_len = seq_len
        self.pad_id = token_to_id["[PAD]"]
        self.cls_id = token_to_id["[CLS]"]
        self.sep_id = token_to_id["[SEP]"]
        self.mask_id = token_to_id["[MASK]"]
        self.items = []
        for document in corpus.documents:
            encoded = encode_document(document, token_to_id)
            for window in make_windows(encoded, seq_len, stride=stride):
                if window:
                    self.items.append(self._make_item(window))

    def _make_item(self, window: list[int]) -> dict[str, Tensor]:
        ids = [self.cls_id] + window[: self.seq_len - 2] + [self.sep_id]
        ids += [self.pad_id] * (self.seq_len - len(ids))
        labels = [-100] * self.seq_len
        candidate_positions = [
            index
            for index, token_id in enumerate(ids)
            if index > 0 and token_id not in {self.pad_id, self.sep_id}
        ]
        mask_pos = candidate_positions[len(candidate_positions) // 2]
        labels[mask_pos] = ids[mask_pos]
        ids[mask_pos] = self.mask_id
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "epoch_idx": torch.tensor(self.period_idx, dtype=torch.long),
            "true_context": torch.tensor(0, dtype=torch.long),
            "p_n1": torch.tensor(0.0, dtype=torch.float32),
            "subject_idx": torch.tensor(0, dtype=torch.long),
        }

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        return self.items[index]


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
