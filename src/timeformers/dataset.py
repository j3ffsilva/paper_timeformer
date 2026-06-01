from __future__ import annotations

import random

import torch
from torch import Tensor
from torch.utils.data import Dataset

from .corpus import OBJS_N1, OBJS_N2, SUBJECTS, SUBJECT_CLASSES, VERBS_N1, VERBS_N2, Example


SPECIAL_TOKENS = ["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"]
VOCAB_TOKENS = SPECIAL_TOKENS + SUBJECTS + VERBS_N1 + VERBS_N2 + OBJS_N1 + OBJS_N2
TOKEN2ID = {token: i for i, token in enumerate(VOCAB_TOKENS)}
ID2TOKEN = {i: token for token, i in TOKEN2ID.items()}

PAD_ID = TOKEN2ID["[PAD]"]
CLS_ID = TOKEN2ID["[CLS]"]
SEP_ID = TOKEN2ID["[SEP]"]
MASK_ID = TOKEN2ID["[MASK]"]
POS_SUBJECT = 1
POS_VERB = 2
POS_OBJECT = 3
SEQ_LEN = 5
VOCAB_SIZE = len(VOCAB_TOKENS)
CLASS2ID = {name: idx for idx, name in enumerate(SUBJECT_CLASSES)}
ID2CLASS = {idx: name for name, idx in CLASS2ID.items()}


class MLMDataset(Dataset):
    def __init__(self, rows: list[Example], split: str | None = None, seed: int = 42) -> None:
        self.rows = [r for r in rows if split is None or r.split == split]
        rng = random.Random(seed)
        self.items = [self._make_item(row, rng) for row in self.rows]

    def _make_item(self, row: Example, rng: random.Random) -> dict[str, Tensor]:
        ids = [
            CLS_ID,
            TOKEN2ID[row.subject],
            TOKEN2ID[row.verb],
            TOKEN2ID[row.obj],
            SEP_ID,
        ]
        context_ids = [ids[POS_VERB], ids[POS_OBJECT]]
        labels = [-100] * SEQ_LEN
        mask_pos = rng.choice([POS_VERB, POS_OBJECT])
        labels[mask_pos] = ids[mask_pos]
        ids[mask_pos] = MASK_ID
        subject_idx = int(row.subject[1:]) - 1
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "context_ids": torch.tensor(context_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "epoch_idx": torch.tensor(row.epoch, dtype=torch.long),
            "subject_idx": torch.tensor(subject_idx, dtype=torch.long),
            "true_context": torch.tensor(row.true_context, dtype=torch.long),
            "p_n1": torch.tensor(row.p_n1, dtype=torch.float32),
            "class_id": torch.tensor(CLASS2ID[row.subject_class], dtype=torch.long),
        }

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        return self.items[idx]
