from __future__ import annotations

import torch
import torch.nn.functional as F


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    logits = z1 @ z2.T / temperature
    labels = torch.arange(z1.shape[0], device=z1.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def masked_reconstruction_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    diff = F.smooth_l1_loss(pred, target, reduction="none", beta=1.0)
    masked = diff * mask
    return masked.sum() / mask.sum().clamp_min(1.0)


def relational_kd_loss(student: torch.Tensor, teacher: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    s_rel = (F.normalize(student, dim=-1) @ F.normalize(student, dim=-1).T) / temperature
    t_rel = (F.normalize(teacher, dim=-1) @ F.normalize(teacher, dim=-1).T) / temperature
    return F.mse_loss(s_rel, t_rel)


def baseline_relative_margin_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    subject_ids: list[str],
    quality: torch.Tensor,
    margin: float = 0.2,
    return_stats: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
    if logits.shape[0] < 2:
        loss = logits.new_tensor(0.0)
        stats = {
            "available_subjects": 0.0,
            "active_subjects": 0.0,
            "active_subject_fraction": 0.0,
            "active_samples": 0.0,
            "active_sample_fraction": 0.0,
        }
        return (loss, stats) if return_stats else loss

    scores = (logits[:, 1] - logits[:, 0]).float()
    sample_weights = 0.35 + 0.65 * quality.squeeze(1).clamp(0.0, 1.0)

    subject_to_indices: dict[str, list[int]] = {}
    for idx, subject_id in enumerate(subject_ids):
        subject_to_indices.setdefault(str(subject_id), []).append(idx)

    losses: list[torch.Tensor] = []
    weights: list[torch.Tensor] = []
    active_samples = 0
    for indices in subject_to_indices.values():
        baseline_idx = [idx for idx in indices if int(labels[idx].item()) == 0]
        stress_idx = [idx for idx in indices if int(labels[idx].item()) == 1]
        if not baseline_idx or not stress_idx:
            continue
        active_samples += len(baseline_idx) + len(stress_idx)

        baseline_tensor = torch.tensor(baseline_idx, device=logits.device, dtype=torch.long)
        stress_tensor = torch.tensor(stress_idx, device=logits.device, dtype=torch.long)

        baseline_scores = scores.index_select(0, baseline_tensor)
        stress_scores = scores.index_select(0, stress_tensor)
        baseline_weights = sample_weights.index_select(0, baseline_tensor)
        stress_weights = sample_weights.index_select(0, stress_tensor)

        baseline_mean = (baseline_scores * baseline_weights).sum() / baseline_weights.sum().clamp(min=1e-6)
        stress_mean = (stress_scores * stress_weights).sum() / stress_weights.sum().clamp(min=1e-6)
        subject_loss = F.relu(margin - (stress_mean - baseline_mean))
        subject_weight = 0.5 * (baseline_weights.mean() + stress_weights.mean())

        losses.append(subject_loss)
        weights.append(subject_weight)

    if not losses:
        loss = logits.new_tensor(0.0)
        stats = {
            "available_subjects": float(len(subject_to_indices)),
            "active_subjects": 0.0,
            "active_subject_fraction": 0.0,
            "active_samples": 0.0,
            "active_sample_fraction": 0.0,
        }
        return (loss, stats) if return_stats else loss

    loss_tensor = torch.stack(losses)
    weight_tensor = torch.stack(weights)
    loss = (loss_tensor * weight_tensor).sum() / weight_tensor.sum().clamp(min=1e-6)
    stats = {
        "available_subjects": float(len(subject_to_indices)),
        "active_subjects": float(len(losses)),
        "active_subject_fraction": float(len(losses) / max(len(subject_to_indices), 1)),
        "active_samples": float(active_samples),
        "active_sample_fraction": float(active_samples / max(logits.shape[0], 1)),
    }
    return (loss, stats) if return_stats else loss
