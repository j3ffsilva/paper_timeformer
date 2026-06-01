from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable

import torch
from torch import Tensor
from torch.utils.data import DataLoader


REP_KEYS = ("h", "context", "subject_idx", "epoch_idx", "true_context", "p_n1", "class_id")


@torch.no_grad()
def extract_occurrence_representations(
    model: torch.nn.Module,
    dataset,
    batch_size: int = 256,
    device: str = "cpu",
) -> dict[str, Tensor]:
    """Extract h_s^i(t) for every occurrence in a dataset."""
    device_t = torch.device(device)
    model.eval()
    model.to(device_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    out: dict[str, list[Tensor]] = defaultdict(list)

    for batch in loader:
        pred = model(batch["input_ids"].to(device_t), batch["epoch_idx"].to(device_t))
        out["h"].append(pred["h_subj"].detach().cpu())
        context_ids = batch["context_ids"].to(device_t)
        context = model.token_emb(context_ids).mean(dim=1)
        out["context"].append(context.detach().cpu())
        for key in REP_KEYS[2:]:
            out[key].append(batch[key].detach().cpu())

    return {key: torch.cat(out[key], dim=0) for key in REP_KEYS}


def save_representations(reps: dict[str, Tensor], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({key: reps[key].cpu() for key in REP_KEYS}, path)


def load_representations(path: Path) -> dict[str, Tensor]:
    loaded = torch.load(path, map_location="cpu")
    missing = [key for key in REP_KEYS if key not in loaded]
    if missing:
        raise ValueError(f"Representation file is missing keys: {missing}")
    return {key: loaded[key] for key in REP_KEYS}


def iter_subject_epoch_groups(reps: dict[str, Tensor]) -> Iterable[tuple[tuple[int, int], Tensor]]:
    subject_idx = reps["subject_idx"]
    epoch_idx = reps["epoch_idx"]
    for subject in torch.unique(subject_idx, sorted=True):
        for epoch in torch.unique(epoch_idx[subject_idx == subject], sorted=True):
            mask = (subject_idx == subject) & (epoch_idx == epoch)
            yield (int(subject), int(epoch)), torch.nonzero(mask, as_tuple=False).flatten()


def group_representations_by_subject_epoch(reps: dict[str, Tensor]) -> dict[tuple[int, int], dict[str, Tensor]]:
    grouped = {}
    for key, indices in iter_subject_epoch_groups(reps):
        grouped[key] = {name: value[indices] for name, value in reps.items()}
    return grouped
