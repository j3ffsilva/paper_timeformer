from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from torch.utils.data import DataLoader


CLASS_NAMES = {0: "stable", 1: "drift", 2: "bifurcating", 3: "abrupt"}


@torch.no_grad()
def extract_representations(model, dataset, batch_size: int = 256, device: str = "cpu") -> dict[str, np.ndarray]:
    device_t = torch.device(device)
    model.eval()
    model.to(device_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    out = defaultdict(list)
    for batch in loader:
        pred = model(batch["input_ids"].to(device_t), batch["epoch_idx"].to(device_t))
        out["h_subj"].append(pred["h_subj"].cpu().numpy())
        for key in ["subject_idx", "epoch_idx", "true_context", "p_n1", "class_id"]:
            out[key].append(batch[key].cpu().numpy())
    return {key: np.concatenate(value, axis=0) for key, value in out.items()}


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return float("nan")
    return float(np.dot(a, b) / denom)


def subject_epoch_centroids(reps: dict[str, np.ndarray]) -> dict[tuple[int, int], np.ndarray]:
    h = reps["h_subj"]
    result = {}
    for subject in np.unique(reps["subject_idx"]):
        for epoch in np.unique(reps["epoch_idx"]):
            mask = (reps["subject_idx"] == subject) & (reps["epoch_idx"] == epoch)
            if mask.any():
                result[(int(subject), int(epoch))] = h[mask].mean(axis=0)
    return result


def trajectory_metrics(reps: dict[str, np.ndarray]) -> dict[str, float]:
    h = reps["h_subj"]
    context = reps["true_context"]
    centroids = subject_epoch_centroids(reps)
    proto_n1 = h[context == 0].mean(axis=0)
    proto_n2 = h[context == 1].mean(axis=0)
    axis = proto_n2 - proto_n1
    axis_norm = np.linalg.norm(axis)
    if axis_norm:
        axis = axis / axis_norm

    movement = defaultdict(list)
    directed = defaultdict(list)
    spearman = defaultdict(list)

    for subject in sorted(np.unique(reps["subject_idx"])):
        points = [centroids.get((int(subject), epoch)) for epoch in range(10)]
        if any(p is None for p in points):
            continue
        subject_mask = reps["subject_idx"] == subject
        class_id = int(stats.mode(reps["class_id"][subject_mask], keepdims=False).mode)
        cls = CLASS_NAMES[class_id]
        path = 0.0
        drift = 0.0
        for i in range(len(points) - 1):
            path += 1.0 - _cosine(points[i], points[i + 1])
            drift += float(np.dot(points[i + 1] - points[i], axis))
        movement[cls].append(path)
        directed[cls].append(drift)

        observed = [_cosine(p, proto_n1) - _cosine(p, proto_n2) for p in points]
        target = []
        for epoch in range(10):
            mask = subject_mask & (reps["epoch_idx"] == epoch)
            target.append(float(np.mean(reps["p_n1"][mask])))
        if len(set(np.round(target, 6))) > 1:
            rho, _ = stats.spearmanr(observed, target)
            if not math.isnan(float(rho)):
                spearman[cls].append(float(rho))

    metrics = {}
    for cls in CLASS_NAMES.values():
        metrics[f"path_{cls}"] = float(np.mean(movement[cls])) if movement[cls] else float("nan")
        metrics[f"directed_{cls}"] = float(np.mean(directed[cls])) if directed[cls] else float("nan")
        metrics[f"spearman_{cls}"] = float(np.mean(spearman[cls])) if spearman[cls] else float("nan")
    metrics["path_contrast_drift_minus_stable"] = metrics["path_drift"] - metrics["path_stable"]
    metrics["directed_contrast_drift_minus_stable"] = metrics["directed_drift"] - metrics["directed_stable"]
    return metrics


@torch.no_grad()
def mlm_accuracy(model, dataset, batch_size: int = 256, device: str = "cpu") -> float:
    device_t = torch.device(device)
    model.eval()
    model.to(device_t)
    correct = 0
    total = 0
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        logits = model(batch["input_ids"].to(device_t), batch["epoch_idx"].to(device_t))["logits"]
        labels = batch["labels"].to(device_t)
        mask = labels != -100
        pred = logits.argmax(dim=-1)
        correct += int((pred[mask] == labels[mask]).sum())
        total += int(mask.sum())
    return correct / max(total, 1)
