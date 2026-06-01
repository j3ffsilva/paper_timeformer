from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .trajectory_losses import anti_identity_loss, masked_mse


class TrajectoryTeacherTrainer:
    def __init__(self, model: nn.Module, output_dir: Path, device: str = "cpu") -> None:
        self.model = model
        self.output_dir = output_dir
        self.device = torch.device(device)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model.to(self.device)

    def train(
        self,
        dataset,
        n_epochs: int = 100,
        batch_size: int = 32,
        lr: float = 1e-3,
        beta: float = 1.0,
        tau_cka: float = 0.7,
        verbose: bool = True,
    ) -> list[dict[str, float]]:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-2)
        history = []

        for epoch in range(n_epochs):
            self.model.train()
            totals = {"loss": 0.0, "recon": 0.0, "anti_id": 0.0, "cka": 0.0, "variance": 0.0}
            n_batches = 0
            for batch in loader:
                values = batch["values"].to(self.device)
                valid_mask = batch["valid_mask"].to(self.device)
                opt.zero_grad()
                out = self.model(values, valid_mask)
                recon = masked_mse(out["recon"], values, valid_mask)
                anti_id, anti_parts = anti_identity_loss(out["M"], values, valid_mask, tau_cka=tau_cka)
                loss = recon + beta * anti_id
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()

                totals["loss"] += float(loss.detach())
                totals["recon"] += float(recon.detach())
                totals["anti_id"] += float(anti_id.detach())
                totals["cka"] += anti_parts["cka"]
                totals["variance"] += anti_parts["variance"]
                n_batches += 1

            record = {"epoch": epoch, **{key: value / max(n_batches, 1) for key, value in totals.items()}}
            history.append(record)
            if verbose and (epoch == 0 or epoch == n_epochs - 1 or (epoch + 1) % 10 == 0):
                print(
                    f"  teacher epoch {epoch:03d} loss={record['loss']:.4f} "
                    f"recon={record['recon']:.4f} cka={record['cka']:.4f}"
                )

        torch.save(self.model.state_dict(), self.output_dir / "teacher.pt")
        (self.output_dir / "teacher_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        return history

    @torch.no_grad()
    def encode(self, dataset, batch_size: int = 64) -> dict[str, torch.Tensor]:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        self.model.eval()
        out = {"M": [], "R": [], "valid_mask": [], "p_n1": [], "class_id": []}
        for batch in loader:
            values = batch["values"].to(self.device)
            valid_mask = batch["valid_mask"].to(self.device)
            pred = self.model(values, valid_mask)
            out["M"].append(pred["M"].cpu())
            out["R"].append(batch["values"].cpu())
            out["valid_mask"].append(batch["valid_mask"].cpu())
            out["p_n1"].append(batch["p_n1"].cpu())
            out["class_id"].append(batch["class_id"].cpu())
        return {key: torch.cat(value, dim=0) for key, value in out.items()}


def sample_mask_positions(valid_mask: torch.Tensor) -> torch.Tensor:
    mask_positions = torch.zeros_like(valid_mask)
    for row in range(valid_mask.size(0)):
        candidates = torch.nonzero(valid_mask[row], as_tuple=False).flatten()
        choice = candidates[torch.randint(len(candidates), (1,), device=valid_mask.device)]
        mask_positions[row, choice] = True
    return mask_positions


class TrajectoryStudentTrainer:
    def __init__(self, student: nn.Module, teacher: nn.Module, output_dir: Path, device: str = "cpu") -> None:
        self.student = student
        self.teacher = teacher
        self.output_dir = output_dir
        self.device = torch.device(device)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.student.to(self.device)
        self.teacher.to(self.device)
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad_(False)

    def train(
        self,
        dataset,
        n_epochs: int = 100,
        batch_size: int = 32,
        lr: float = 1e-3,
        verbose: bool = True,
    ) -> list[dict[str, float]]:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        opt = torch.optim.AdamW(self.student.parameters(), lr=lr, weight_decay=1e-2)
        history = []

        for epoch in range(n_epochs):
            self.student.train()
            total = 0.0
            n_batches = 0
            for batch in loader:
                values = batch["values"].to(self.device)
                valid_mask = batch["valid_mask"].to(self.device)
                mask_positions = sample_mask_positions(valid_mask)
                with torch.no_grad():
                    target = self.teacher(values, valid_mask)["M"]
                pred = self.student(values, mask_positions, valid_mask)["M"]
                loss = masked_mse(pred, target, mask_positions)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
                opt.step()
                total += float(loss.detach())
                n_batches += 1

            record = {"epoch": epoch, "loss": total / max(n_batches, 1)}
            history.append(record)
            if verbose and (epoch == 0 or epoch == n_epochs - 1 or (epoch + 1) % 10 == 0):
                print(f"  student epoch {epoch:03d} loss={record['loss']:.4f}")

        torch.save(self.student.state_dict(), self.output_dir / "student.pt")
        (self.output_dir / "student_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        return history

    @torch.no_grad()
    def evaluate_masked_reconstruction(self, dataset, batch_size: int = 64) -> dict[str, torch.Tensor]:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        self.student.eval()
        losses = []
        class_ids = []
        for batch in loader:
            values = batch["values"].to(self.device)
            valid_mask = batch["valid_mask"].to(self.device)
            mask_positions = sample_mask_positions(valid_mask)
            target = self.teacher(values, valid_mask)["M"]
            pred = self.student(values, mask_positions, valid_mask)["M"]
            per_token = (pred - target).pow(2).mean(dim=-1)
            losses.append(per_token[mask_positions].cpu())
            class_ids.append(batch["class_id"].cpu())
        return {"loss": torch.cat(losses, dim=0), "class_id": torch.cat(class_ids, dim=0)}

    @torch.no_grad()
    def evaluate_all_masked_reconstruction(self, dataset, batch_size: int = 64) -> dict[str, torch.Tensor]:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        self.student.eval()
        losses = []
        class_ids = []
        epoch_positions = []
        for batch in loader:
            values = batch["values"].to(self.device)
            valid_mask = batch["valid_mask"].to(self.device)
            target = self.teacher(values, valid_mask)["M"]
            batch_losses = []
            batch_positions = []
            for pos in range(values.size(1)):
                mask_positions = torch.zeros_like(valid_mask)
                mask_positions[:, pos] = valid_mask[:, pos]
                if not mask_positions.any():
                    continue
                pred = self.student(values, mask_positions, valid_mask)["M"]
                per_token = (pred - target).pow(2).mean(dim=-1)
                batch_losses.append(per_token[mask_positions].cpu())
                batch_positions.append(batch["epoch_idx"][:, pos][mask_positions[:, pos].cpu()].cpu())
            if batch_losses:
                losses.append(torch.cat(batch_losses, dim=0))
                epoch_positions.append(torch.cat(batch_positions, dim=0))
                repeated_class = batch["class_id"].unsqueeze(1).expand_as(batch["valid_mask"])
                class_ids.append(repeated_class[batch["valid_mask"]].cpu())
        return {
            "loss": torch.cat(losses, dim=0),
            "class_id": torch.cat(class_ids, dim=0),
            "epoch_idx": torch.cat(epoch_positions, dim=0),
        }
