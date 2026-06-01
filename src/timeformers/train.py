from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .losses import mlm_loss, trajectory_axis_loss, trajectory_ranking_loss


def _forward_loss(
    model: nn.Module,
    batch: dict,
    device: torch.device,
    lambda_traj: float,
    traj_prototypes: tuple[torch.Tensor, torch.Tensor] | None = None,
    traj_loss: str = "axis",
    traj_margin: float = 0.05,
) -> tuple[dict, torch.Tensor, dict[str, float]]:
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    epoch_idx = batch["epoch_idx"].to(device)
    true_context = batch["true_context"].to(device)
    p_n1 = batch["p_n1"].to(device)
    subject_idx = batch["subject_idx"].to(device)

    out = model(input_ids, epoch_idx)
    loss_mlm = mlm_loss(out["logits"], labels)
    if not lambda_traj:
        loss_traj = loss_mlm.new_zeros(())
    elif traj_loss == "axis":
        loss_traj = trajectory_axis_loss(out["h_subj"], true_context, p_n1, traj_prototypes)
    elif traj_loss == "rank":
        loss_traj = trajectory_ranking_loss(
            out["h_subj"],
            true_context,
            p_n1,
            subject_idx,
            prototypes=traj_prototypes,
            margin=traj_margin,
        )
    else:
        raise ValueError(f"Unknown trajectory loss: {traj_loss!r}")
    loss = loss_mlm + lambda_traj * loss_traj
    parts = {"loss": float(loss.detach()), "mlm": float(loss_mlm.detach()), "traj": float(loss_traj.detach())}
    return out, loss, parts


class Trainer:
    def __init__(self, model: nn.Module, output_dir: Path, device: str = "cpu") -> None:
        self.model = model
        self.output_dir = output_dir
        self.device = torch.device(device)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model.to(self.device)

    def train(
        self,
        train_ds,
        val_ds=None,
        n_epochs: int = 12,
        batch_size: int = 64,
        lr: float = 1e-3,
        seed: int = 42,
        lambda_traj: float = 0.0,
        traj_prototypes: str = "batch",
        traj_loss: str = "axis",
        traj_margin: float = 0.05,
        verbose: bool = True,
    ) -> list[dict]:
        torch.manual_seed(seed)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        proto_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False) if val_ds is not None else None
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
        history = []
        best = float("inf")

        for epoch in range(n_epochs):
            prototypes = None
            if lambda_traj and traj_prototypes == "global":
                prototypes = self.compute_context_prototypes(proto_loader)
            self.model.train()
            totals = {"loss": 0.0, "mlm": 0.0, "traj": 0.0}
            n_batches = 0
            for batch in train_loader:
                opt.zero_grad()
                _, loss, parts = _forward_loss(
                    self.model,
                    batch,
                    self.device,
                    lambda_traj,
                    prototypes,
                    traj_loss=traj_loss,
                    traj_margin=traj_margin,
                )
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                for key, value in parts.items():
                    totals[key] += value
                n_batches += 1
            scheduler.step()
            record = {f"train_{k}": v / max(n_batches, 1) for k, v in totals.items()}
            record["epoch"] = epoch
            record["traj_prototypes"] = traj_prototypes if lambda_traj else "none"
            record["traj_loss"] = traj_loss if lambda_traj else "none"
            if val_loader is not None:
                record["val_loss"] = self.evaluate_loss(
                    val_loader,
                    lambda_traj,
                    prototypes,
                    traj_loss=traj_loss,
                    traj_margin=traj_margin,
                )
            history.append(record)
            monitor = record.get("val_loss", record["train_loss"])
            if monitor < best:
                best = monitor
                torch.save(self.model.state_dict(), self.output_dir / "best.pt")
            if verbose:
                print(
                    f"  epoch {epoch:02d} loss={record['train_loss']:.4f} "
                    f"mlm={record['train_mlm']:.4f} traj={record['train_traj']:.4f}"
                )

        torch.save(self.model.state_dict(), self.output_dir / "final.pt")
        (self.output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        return history

    @torch.no_grad()
    def evaluate_loss(
        self,
        loader: DataLoader,
        lambda_traj: float,
        traj_prototypes: tuple[torch.Tensor, torch.Tensor] | None = None,
        traj_loss: str = "axis",
        traj_margin: float = 0.05,
    ) -> float:
        self.model.eval()
        total = 0.0
        n_batches = 0
        for batch in loader:
            _, loss, _ = _forward_loss(
                self.model,
                batch,
                self.device,
                lambda_traj,
                traj_prototypes,
                traj_loss=traj_loss,
                traj_margin=traj_margin,
            )
            total += float(loss.detach())
            n_batches += 1
        return total / max(n_batches, 1)

    @torch.no_grad()
    def compute_context_prototypes(self, loader: DataLoader) -> tuple[torch.Tensor, torch.Tensor] | None:
        self.model.eval()
        sums = {
            0: torch.zeros(self.model.d_model, device=self.device),
            1: torch.zeros(self.model.d_model, device=self.device),
        }
        counts = {0: 0, 1: 0}
        for batch in loader:
            input_ids = batch["input_ids"].to(self.device)
            epoch_idx = batch["epoch_idx"].to(self.device)
            true_context = batch["true_context"].to(self.device)
            out = self.model(input_ids, epoch_idx)
            h_subj = out["h_subj"]
            for context_id in (0, 1):
                mask = true_context == context_id
                if mask.any():
                    sums[context_id] += h_subj[mask].sum(dim=0)
                    counts[context_id] += int(mask.sum())
        if counts[0] == 0 or counts[1] == 0:
            return None
        return (sums[0] / counts[0]).detach(), (sums[1] / counts[1]).detach()
