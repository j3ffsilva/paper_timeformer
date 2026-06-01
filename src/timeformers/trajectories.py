from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.utils.data import Dataset


@dataclass(frozen=True)
class TrajectorySequences:
    values: Tensor
    valid_mask: Tensor
    observed_mask: Tensor
    subject_idx: Tensor
    start_epoch: Tensor
    epoch_idx: Tensor
    p_n1: Tensor
    class_id: Tensor


def _interp(values: dict[int, Tensor], epoch: int, left: int, right: int) -> Tensor:
    alpha = (epoch - left) / max(right - left, 1)
    return (1.0 - alpha) * values[left] + alpha * values[right]


def build_trajectory_sequences(aggregated: dict[str, Tensor], min_periods: int = 3) -> TrajectorySequences:
    """Build padded Seq_s tensors with internal linear interpolation only."""
    subjects = torch.unique(aggregated["subject_idx"], sorted=True)
    sequences = []
    valid_masks = []
    observed_masks = []
    subject_ids = []
    starts = []
    epoch_rows = []
    p_rows = []
    class_ids = []

    max_len = 0
    per_subject = []
    for subject in subjects:
        mask = aggregated["subject_idx"] == subject
        epochs = [int(x) for x in aggregated["epoch_idx"][mask].tolist()]
        if len(epochs) < min_periods:
            continue
        order = torch.argsort(aggregated["epoch_idx"][mask])
        rows = {int(aggregated["epoch_idx"][mask][i]): i for i in order.tolist()}
        start, end = min(epochs), max(epochs)
        full_epochs = list(range(start, end + 1))
        max_len = max(max_len, len(full_epochs))
        per_subject.append((int(subject), mask, rows, epochs, full_epochs))

    if not per_subject:
        raise ValueError("No subject has enough periods to build trajectories")

    d_model = aggregated["R"].size(-1)
    for subject, mask, rows, epochs, full_epochs in per_subject:
        local_R = {epoch: aggregated["R"][mask][idx] for epoch, idx in rows.items()}
        local_p = {epoch: aggregated["p_n1"][mask][idx] for epoch, idx in rows.items()}
        class_id = torch.mode(aggregated["class_id"][mask]).values

        seq = torch.zeros(max_len, d_model, dtype=aggregated["R"].dtype)
        valid = torch.zeros(max_len, dtype=torch.bool)
        observed = torch.zeros(max_len, dtype=torch.bool)
        epoch_row = torch.full((max_len,), -1, dtype=torch.long)
        p_row = torch.zeros(max_len, dtype=aggregated["p_n1"].dtype)

        for pos, epoch in enumerate(full_epochs):
            valid[pos] = True
            epoch_row[pos] = epoch
            if epoch in local_R:
                seq[pos] = local_R[epoch]
                p_row[pos] = local_p[epoch]
                observed[pos] = True
                continue

            left_candidates = [e for e in epochs if e < epoch]
            right_candidates = [e for e in epochs if e > epoch]
            if not left_candidates or not right_candidates:
                continue
            left = max(left_candidates)
            right = min(right_candidates)
            seq[pos] = _interp(local_R, epoch, left, right)
            p_row[pos] = _interp(local_p, epoch, left, right)

        sequences.append(seq)
        valid_masks.append(valid)
        observed_masks.append(observed)
        subject_ids.append(torch.tensor(subject, dtype=torch.long))
        starts.append(torch.tensor(full_epochs[0], dtype=torch.long))
        epoch_rows.append(epoch_row)
        p_rows.append(p_row)
        class_ids.append(class_id)

    return TrajectorySequences(
        values=torch.stack(sequences, dim=0),
        valid_mask=torch.stack(valid_masks, dim=0),
        observed_mask=torch.stack(observed_masks, dim=0),
        subject_idx=torch.stack(subject_ids, dim=0),
        start_epoch=torch.stack(starts, dim=0),
        epoch_idx=torch.stack(epoch_rows, dim=0),
        p_n1=torch.stack(p_rows, dim=0),
        class_id=torch.stack(class_ids, dim=0),
    )


class TrajectoryDataset(Dataset):
    def __init__(self, sequences: TrajectorySequences) -> None:
        self.sequences = sequences

    def __len__(self) -> int:
        return self.sequences.values.size(0)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        return {
            "values": self.sequences.values[idx],
            "valid_mask": self.sequences.valid_mask[idx],
            "observed_mask": self.sequences.observed_mask[idx],
            "subject_idx": self.sequences.subject_idx[idx],
            "start_epoch": self.sequences.start_epoch[idx],
            "epoch_idx": self.sequences.epoch_idx[idx],
            "p_n1": self.sequences.p_n1[idx],
            "class_id": self.sequences.class_id[idx],
        }
