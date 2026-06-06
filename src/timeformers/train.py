from __future__ import annotations

import copy
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .losses import mlm_loss, trajectory_axis_loss, trajectory_ranking_loss


def _should_log(epoch: int, n_epochs: int, every: int = 10) -> bool:
    return epoch == 0 or epoch == n_epochs - 1 or (epoch + 1) % every == 0


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


class MLMTrainer:
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
            if verbose and _should_log(epoch, n_epochs):
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


def _set_dataset_epoch(dataset, epoch: int) -> None:
    if hasattr(dataset, "set_epoch"):
        dataset.set_epoch(epoch)
    elif hasattr(dataset, "dataset"):
        _set_dataset_epoch(dataset.dataset, epoch)
    elif hasattr(dataset, "datasets"):
        for child in dataset.datasets:
            _set_dataset_epoch(child, epoch)


class ContinualPeriodTrainer:
    """Train one model over chronologically ordered period datasets."""

    def __init__(self, model: nn.Module, output_dir: Path, device: str = "cpu") -> None:
        self.model = model
        self.output_dir = output_dir
        self.device = torch.device(device)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model.to(self.device)

    def train(
        self,
        period_datasets: list,
        val_period_datasets: list | None = None,
        n_epochs_per_period: int = 5,
        n_epochs_first_period: int | None = None,
        batch_size: int = 64,
        lr: float = 1e-4,
        seed: int = 42,
        early_stopping_patience: int | None = None,
        early_stopping_min_delta: float = 1e-4,
        restore_best_model: bool = True,
        verbose: bool = True,
    ) -> list[dict]:
        torch.manual_seed(seed)
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-2)
        history = []
        period_summaries = []
        cumulative_gradient_steps = 0

        for period, dataset in enumerate(period_datasets):
            period_start_steps = cumulative_gradient_steps
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            val_loader = (
                DataLoader(val_period_datasets[period], batch_size=batch_size, shuffle=False)
                if val_period_datasets is not None
                else None
            )
            period_epochs = n_epochs_first_period if period == 0 and n_epochs_first_period is not None else n_epochs_per_period
            best_val = float("inf")
            best_model_state = None
            best_opt_state = None
            best_epoch = None
            best_gradient_step = None
            stale_epochs = 0
            for epoch in range(period_epochs):
                _set_dataset_epoch(dataset, epoch)
                self.model.train()
                total_loss = 0.0
                n_batches = 0
                for batch in loader:
                    opt.zero_grad()
                    _, loss, _ = _forward_loss(self.model, batch, self.device, lambda_traj=0.0)
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    opt.step()
                    cumulative_gradient_steps += 1
                    total_loss += float(loss.detach())
                    n_batches += 1

                record = {
                    "period": period,
                    "epoch": epoch,
                    "train_loss": total_loss / max(n_batches, 1),
                    "n_examples": len(dataset),
                    "gradient_steps_this_epoch": n_batches,
                    "cumulative_gradient_steps": cumulative_gradient_steps,
                }
                if val_loader is not None:
                    record["val_loss"] = self.evaluate_loss(val_loader)
                    if record["val_loss"] < best_val - early_stopping_min_delta:
                        best_val = record["val_loss"]
                        best_model_state = {key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()}
                        best_opt_state = copy.deepcopy(opt.state_dict())
                        best_epoch = epoch
                        best_gradient_step = cumulative_gradient_steps
                        stale_epochs = 0
                    else:
                        stale_epochs += 1
                history.append(record)
                if verbose and _should_log(epoch, period_epochs):
                    print(
                        f"  period {period:02d} epoch {epoch:02d} "
                        f"loss={record['train_loss']:.4f}"
                    )
                if early_stopping_patience is not None and stale_epochs >= early_stopping_patience:
                    break

            if restore_best_model and best_model_state is not None:
                self.model.load_state_dict(best_model_state)
                opt.load_state_dict(best_opt_state)
            else:
                best_epoch = epoch
                best_gradient_step = cumulative_gradient_steps
            torch.save(self.model.state_dict(), self.output_dir / f"checkpoint_t{period:02d}.pt")
            torch.save(
                {
                    "model": self.model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "period": period,
                    "selected_epoch": best_epoch,
                    "selected_cumulative_gradient_steps": best_gradient_step,
                },
                self.output_dir / f"training_state_t{period:02d}.pt",
            )
            period_summaries.append(
                {
                    "period": period,
                    "epochs_run": epoch + 1,
                    "gradient_steps_computed": cumulative_gradient_steps - period_start_steps,
                    "cumulative_gradient_steps_computed": cumulative_gradient_steps,
                    "selected_epoch": best_epoch,
                    "selected_cumulative_gradient_steps": best_gradient_step,
                    "best_val_loss": best_val if val_loader is not None else None,
                }
            )

        (self.output_dir / "continual_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        (self.output_dir / "period_summaries.json").write_text(
            json.dumps(period_summaries, indent=2),
            encoding="utf-8",
        )
        return history

    @torch.no_grad()
    def evaluate_loss(self, loader: DataLoader) -> float:
        self.model.eval()
        total = 0.0
        n_batches = 0
        for batch in loader:
            _, loss, _ = _forward_loss(self.model, batch, self.device, lambda_traj=0.0)
            total += float(loss.detach())
            n_batches += 1
        return total / max(n_batches, 1)


Trainer = MLMTrainer  # backward-compat alias
