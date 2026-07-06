from __future__ import annotations

import torch
import torch.nn.functional as F


def _flat_quality(quality: torch.Tensor) -> torch.Tensor:
    if quality.ndim > 1:
        quality = quality.squeeze(-1)
    return quality.float().clamp(0.0, 1.0)


def true_class_confidence(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    return probs.gather(1, labels.view(-1, 1)).squeeze(1).clamp(0.0, 1.0)


def cross_calibrated_trust(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    quality: torch.Tensor,
) -> torch.Tensor:
    teacher_reliability = true_class_confidence(teacher_logits.detach(), labels)
    student_readiness = true_class_confidence(student_logits.detach(), labels)
    trust = _flat_quality(quality) * torch.minimum(teacher_reliability, student_readiness)
    return trust.detach().clamp(0.0, 1.0)


def trust_weighted_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    trust: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
    teacher_probs = F.softmax(teacher_logits.detach() / temperature, dim=1)
    per_sample = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=1)
    weights = trust.detach().float().clamp(0.0, 1.0)
    return (per_sample * weights).sum() / weights.sum().clamp(min=1e-6) * (temperature ** 2)

