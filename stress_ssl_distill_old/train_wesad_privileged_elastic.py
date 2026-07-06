from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .catsa_dataset import CATSAPrivilegedWindowDataset
from .early_stopping import EarlyStopping
from .losses import baseline_relative_margin_loss
from .reliability import cross_calibrated_trust, reliability_bce_loss, true_class_confidence, trust_weighted_kl_loss
from .samplers import SubjectAwareBatchSampler
from .train_galaxy_watch import (
    aggregate_predictions,
    build_loader,
    evaluate_with_threshold,
    maybe_parse_sessions,
    quality_aware_focal_loss,
    select_threshold,
    set_random_seed,
    supervised_contrastive_loss,
    update_ema_model,
)
from .wesad_dataset import WESADPrivilegedWindowDataset
from .wesad_models_safe_sgpc import WESADPrivilegedTeacherNet


DEFAULT_CALM_SESSIONS = ["baseline"]
DEFAULT_STRESS_SESSIONS = ["stress"]


PRIVILEGED_DATASETS = {
    "wesad": WESADPrivilegedWindowDataset,
    "catsa": CATSAPrivilegedWindowDataset,
}


def focal_loss_values(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_weights: torch.Tensor,
    gamma: float,
    label_smoothing: float,
) -> torch.Tensor:
    ce = F.cross_entropy(
        logits,
        labels,
        weight=class_weights,
        reduction="none",
        label_smoothing=label_smoothing,
    )
    probs = torch.softmax(logits, dim=1)
    pt = probs.gather(1, labels.unsqueeze(1)).squeeze(1).clamp(min=1e-6, max=1.0)
    return (1.0 - pt).pow(gamma) * ce


def quality_weighted_mean(values: torch.Tensor, quality: torch.Tensor) -> torch.Tensor:
    weights = 0.35 + 0.65 * quality.squeeze(1).clamp(0.0, 1.0)
    return (values * weights).sum() / weights.sum().clamp(min=1e-6)


def apply_min_trust_floor(trust: torch.Tensor, min_weight: float) -> torch.Tensor:
    trust = trust.detach().float().clamp(0.0, 1.0)
    if min_weight <= 0.0:
        return trust
    floor = max(0.0, min(float(min_weight), 1.0))
    return floor + (1.0 - floor) * trust


def true_class_kd_gate(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    quality: torch.Tensor,
    mode: str,
    min_weight: float,
) -> torch.Tensor | None:
    if mode == "none":
        return None
    if mode == "teacher_true_confidence":
        confidence = true_class_confidence(teacher_logits.detach(), labels)
    elif mode == "student_true_confidence":
        confidence = true_class_confidence(student_logits.detach(), labels)
    else:
        raise ValueError(f"Unsupported KD gate mode: {mode}")
    if quality.ndim > 1:
        quality = quality.squeeze(-1)
    trust = quality.float().clamp(0.0, 1.0) * confidence.detach().float().clamp(0.0, 1.0)
    return apply_min_trust_floor(trust, min_weight=min_weight)


def nondegradation_loss(
    candidate_logits: torch.Tensor,
    base_logits: torch.Tensor,
    labels: torch.Tensor,
    quality: torch.Tensor,
    class_weights: torch.Tensor,
    gamma: float,
    label_smoothing: float,
    margin: float,
) -> torch.Tensor:
    candidate_values = focal_loss_values(
        candidate_logits,
        labels,
        class_weights=class_weights,
        gamma=gamma,
        label_smoothing=label_smoothing,
    )
    base_values = focal_loss_values(
        base_logits.detach(),
        labels,
        class_weights=class_weights,
        gamma=gamma,
        label_smoothing=label_smoothing,
    )
    penalty = F.relu(candidate_values - base_values + margin)
    return quality_weighted_mean(penalty, quality)


def helpful_correction_alignment_loss(
    deploy_delta: torch.Tensor,
    privileged_delta: torch.Tensor,
    privileged_logits: torch.Tensor,
    base_logits: torch.Tensor,
    labels: torch.Tensor,
    quality: torch.Tensor,
    class_weights: torch.Tensor,
    gamma: float,
    label_smoothing: float,
) -> torch.Tensor:
    privileged_values = focal_loss_values(
        privileged_logits.detach(),
        labels,
        class_weights=class_weights,
        gamma=gamma,
        label_smoothing=label_smoothing,
    )
    base_values = focal_loss_values(
        base_logits.detach(),
        labels,
        class_weights=class_weights,
        gamma=gamma,
        label_smoothing=label_smoothing,
    )
    helpful = (privileged_values < base_values).float()
    weights = helpful * (0.35 + 0.65 * quality.squeeze(1).clamp(0.0, 1.0))
    if weights.sum() <= 1e-6:
        return deploy_delta.new_tensor(0.0)
    per_sample = F.smooth_l1_loss(deploy_delta, privileged_delta.detach(), reduction="none").mean(dim=1)
    return (per_sample * weights).sum() / weights.sum().clamp(min=1e-6)


def alpha_helpfulness_loss(
    alpha: torch.Tensor,
    privileged_logits: torch.Tensor,
    base_logits: torch.Tensor,
    labels: torch.Tensor,
    quality: torch.Tensor,
    class_weights: torch.Tensor,
    gamma: float,
    label_smoothing: float,
    margin: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    privileged_values = focal_loss_values(
        privileged_logits.detach(),
        labels,
        class_weights=class_weights,
        gamma=gamma,
        label_smoothing=label_smoothing,
    )
    base_values = focal_loss_values(
        base_logits.detach(),
        labels,
        class_weights=class_weights,
        gamma=gamma,
        label_smoothing=label_smoothing,
    )
    helpful = (privileged_values + float(margin) < base_values).float()
    alpha_values = alpha.squeeze(1).clamp(min=1e-6, max=1.0 - 1e-6)
    loss_values = F.binary_cross_entropy(alpha_values, helpful, reduction="none")
    return quality_weighted_mean(loss_values, quality), helpful.mean().detach()


def binary_margin(logits: torch.Tensor) -> torch.Tensor:
    if logits.shape[1] != 2:
        raise ValueError("Elastic residual correction currently expects binary logits.")
    return logits[:, 1] - logits[:, 0]


def elastic_privileged_residual_loss(
    base_logits: torch.Tensor,
    corrected_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    quality: torch.Tensor,
    class_weights: torch.Tensor,
    gamma: float,
    label_smoothing: float,
    reliability_temp: float,
    label_margin: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    base_values = focal_loss_values(
        base_logits.detach(),
        labels,
        class_weights=class_weights,
        gamma=gamma,
        label_smoothing=label_smoothing,
    )
    teacher_values = focal_loss_values(
        teacher_logits.detach(),
        labels,
        class_weights=class_weights,
        gamma=gamma,
        label_smoothing=label_smoothing,
    )
    reliability = torch.sigmoid((base_values - teacher_values) / max(float(reliability_temp), 1e-6))

    base_margin = binary_margin(base_logits)
    corrected_margin = binary_margin(corrected_logits)
    teacher_margin = binary_margin(teacher_logits).detach()
    label_margin_target = (labels.float() * 2.0 - 1.0) * float(label_margin)

    teacher_delta = teacher_margin - base_margin.detach()
    label_delta = label_margin_target - base_margin.detach()
    target_delta = reliability * teacher_delta + (1.0 - reliability) * label_delta
    predicted_delta = corrected_margin - base_margin

    loss_values = F.smooth_l1_loss(predicted_delta, target_delta.detach(), reduction="none")
    return quality_weighted_mean(loss_values, quality), reliability.mean().detach(), target_delta.detach().abs().mean()


def elastic_alpha_target_loss(
    alpha: torch.Tensor,
    base_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    quality: torch.Tensor,
    class_weights: torch.Tensor,
    gamma: float,
    label_smoothing: float,
    reliability_temp: float,
    uncertainty_temp: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    base_values = focal_loss_values(
        base_logits.detach(),
        labels,
        class_weights=class_weights,
        gamma=gamma,
        label_smoothing=label_smoothing,
    )
    teacher_values = focal_loss_values(
        teacher_logits.detach(),
        labels,
        class_weights=class_weights,
        gamma=gamma,
        label_smoothing=label_smoothing,
    )
    reliability = torch.sigmoid((base_values - teacher_values) / max(float(reliability_temp), 1e-6))
    base_uncertainty = torch.exp(-binary_margin(base_logits.detach()).abs() / max(float(uncertainty_temp), 1e-6))
    alpha_target = base_uncertainty * (0.25 + 0.75 * reliability)
    alpha_values = alpha.squeeze(1).clamp(min=1e-6, max=1.0 - 1e-6)
    loss_values = F.binary_cross_entropy(alpha_values, alpha_target.detach(), reduction="none")
    return quality_weighted_mean(loss_values, quality), alpha_target.mean().detach()


def distillation_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    quality: torch.Tensor,
    temperature: float,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
    teacher_probs = F.softmax(teacher_logits.detach() / temperature, dim=1)
    kl = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=1)
    weights = 0.35 + 0.65 * quality.squeeze(1).clamp(0.0, 1.0)
    if sample_weights is not None:
        weights = weights * sample_weights
    return (kl * weights).sum() / weights.sum().clamp(min=1e-6) * (temperature ** 2)


def embedding_alignment_loss(
    watch_align: torch.Tensor,
    teacher_align: torch.Tensor,
    quality: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    cosine = F.cosine_similarity(watch_align, teacher_align.detach(), dim=1)
    weights = 0.35 + 0.65 * quality.squeeze(1).clamp(0.0, 1.0)
    if sample_weights is not None:
        weights = weights * sample_weights
    return ((1.0 - cosine) * weights).sum() / weights.sum().clamp(min=1e-6)


def margin_matching_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    quality: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    student_margin = (student_logits[:, 1] - student_logits[:, 0]).float()
    teacher_margin = (teacher_logits[:, 1] - teacher_logits[:, 0]).float().detach()
    weights = 0.35 + 0.65 * quality.squeeze(1).clamp(0.0, 1.0)
    if sample_weights is not None:
        weights = weights * sample_weights
    loss = F.smooth_l1_loss(student_margin, teacher_margin, reduction="none")
    return (loss * weights).sum() / weights.sum().clamp(min=1e-6)


def _weighted_zscore(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    weights = weights / weights.sum().clamp(min=1e-6)
    mean = (values * weights).sum()
    var = ((values - mean).pow(2) * weights).sum()
    return (values - mean) / torch.sqrt(var.clamp(min=1e-4))


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (values * weights).sum() / weights.sum().clamp(min=1e-6)


def normalized_margin_alignment_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    quality: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if student_logits.shape[0] < 2:
        return student_logits.new_tensor(0.0)

    student_margin = (student_logits[:, 1] - student_logits[:, 0]).float()
    teacher_margin = (teacher_logits[:, 1] - teacher_logits[:, 0]).float().detach()
    weights = 0.35 + 0.65 * quality.squeeze(1).clamp(0.0, 1.0)
    if sample_weights is not None:
        weights = weights * sample_weights

    student_z = _weighted_zscore(student_margin, weights)
    teacher_z = _weighted_zscore(teacher_margin, weights)
    loss = F.smooth_l1_loss(student_z, teacher_z, reduction="none")
    return (loss * weights).sum() / weights.sum().clamp(min=1e-6)


def subject_center_stability_loss(
    student_logits: torch.Tensor,
    labels: torch.Tensor,
    subject_ids: list[str],
    quality: torch.Tensor,
) -> torch.Tensor:
    if student_logits.shape[0] < 2:
        return student_logits.new_tensor(0.0)

    margins = (student_logits[:, 1] - student_logits[:, 0]).float()
    quality_weights = 0.35 + 0.65 * quality.squeeze(1).clamp(0.0, 1.0)

    class_centers: dict[int, list[torch.Tensor]] = {0: [], 1: []}
    for label in (0, 1):
        label_mask = labels == label
        if label_mask.sum() < 2:
            continue
        global_center = (margins[label_mask] * quality_weights[label_mask]).sum() / quality_weights[label_mask].sum().clamp(min=1e-6)
        for subject_id in set(subject_ids):
            subject_mask = torch.tensor(
                [str(item) == str(subject_id) for item in subject_ids],
                device=student_logits.device,
                dtype=torch.bool,
            )
            group_mask = label_mask & subject_mask
            if group_mask.sum() < 2:
                continue
            group_center = (margins[group_mask] * quality_weights[group_mask]).sum() / quality_weights[group_mask].sum().clamp(min=1e-6)
            class_centers[int(label)].append(F.smooth_l1_loss(group_center, global_center.detach(), reduction="mean"))

    losses = class_centers[0] + class_centers[1]
    if not losses:
        return student_logits.new_tensor(0.0)
    return torch.stack(losses).mean()


def validation_threshold_stability_loss(
    student_logits: torch.Tensor,
    labels: torch.Tensor,
    subject_ids: list[str],
    sessions: list[str],
    quality: torch.Tensor,
) -> torch.Tensor:
    if student_logits.shape[0] < 2:
        return student_logits.new_tensor(0.0)

    margins = (student_logits[:, 1] - student_logits[:, 0]).float()
    probs = torch.softmax(student_logits, dim=1)[:, 1]
    quality_weights = 0.35 + 0.65 * quality.squeeze(1).clamp(0.0, 1.0)
    label_masks = {label: labels == label for label in (0, 1)}
    if not label_masks[0].any() or not label_masks[1].any():
        return student_logits.new_tensor(0.0)

    global_centers = {
        label: _weighted_mean(margins[label_masks[label]], quality_weights[label_masks[label]])
        for label in (0, 1)
    }
    global_threshold = (0.5 * (global_centers[0] + global_centers[1])).detach()
    global_centers = {label: center.detach() for label, center in global_centers.items()}

    losses: list[torch.Tensor] = []
    unique_subjects = sorted(set(subject_ids))
    for subject_id in unique_subjects:
        subject_mask = torch.tensor(
            [str(item) == str(subject_id) for item in subject_ids],
            device=student_logits.device,
            dtype=torch.bool,
        )
        subject_label_masks = {label: subject_mask & label_masks[label] for label in (0, 1)}

        if subject_label_masks[0].any() and subject_label_masks[1].any():
            neg_center = _weighted_mean(margins[subject_label_masks[0]], quality_weights[subject_label_masks[0]])
            pos_center = _weighted_mean(margins[subject_label_masks[1]], quality_weights[subject_label_masks[1]])
            subject_threshold = 0.5 * (neg_center + pos_center)
            losses.append(F.smooth_l1_loss(subject_threshold, global_threshold, reduction="mean"))

        for label in (0, 1):
            if subject_label_masks[label].sum() < 2:
                continue
            class_center = _weighted_mean(margins[subject_label_masks[label]], quality_weights[subject_label_masks[label]])
            losses.append(0.5 * F.smooth_l1_loss(class_center, global_centers[label], reduction="mean"))

    group_to_indices: dict[tuple[str, str], list[int]] = {}
    for idx, (subject_id, session) in enumerate(zip(subject_ids, sessions)):
        group_to_indices.setdefault((str(subject_id), str(session)), []).append(idx)

    for indices in group_to_indices.values():
        if len(indices) < 2:
            continue
        index_tensor = torch.tensor(indices, device=student_logits.device, dtype=torch.long)
        group_probs = probs.index_select(0, index_tensor)
        group_labels = labels.index_select(0, index_tensor).float()
        group_weights = quality_weights.index_select(0, index_tensor)
        group_prob_rate = _weighted_mean(group_probs, group_weights)
        group_label_rate = _weighted_mean(group_labels, group_weights).detach()
        losses.append(0.25 * F.smooth_l1_loss(group_prob_rate, group_label_rate, reduction="mean"))

    if not losses:
        return student_logits.new_tensor(0.0)
    return torch.stack(losses).mean()


def ranking_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    quality: torch.Tensor,
    distill_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if student_logits.shape[0] < 2:
        return student_logits.new_tensor(0.0)

    student_scores = (student_logits[:, 1] - student_logits[:, 0]).float()
    teacher_scores = (teacher_logits[:, 1] - teacher_logits[:, 0]).float().detach()

    teacher_diff = teacher_scores.unsqueeze(1) - teacher_scores.unsqueeze(0)
    student_diff = student_scores.unsqueeze(1) - student_scores.unsqueeze(0)

    upper_mask = torch.triu(torch.ones_like(teacher_diff, dtype=torch.bool), diagonal=1)
    informative_mask = upper_mask & (teacher_diff.abs() > 1e-6)
    if not informative_mask.any():
        return student_logits.new_tensor(0.0)

    pair_sign = torch.sign(teacher_diff[informative_mask])
    pair_student_diff = student_diff[informative_mask]
    pair_teacher_strength = teacher_diff[informative_mask].abs()

    sample_weights = 0.35 + 0.65 * quality.squeeze(1).clamp(0.0, 1.0)
    if distill_weights is not None:
        sample_weights = sample_weights * distill_weights
    pair_weights = ((sample_weights.unsqueeze(1) + sample_weights.unsqueeze(0)) * 0.5)[informative_mask]
    total_weights = pair_weights * pair_teacher_strength
    total_weights = total_weights / total_weights.mean().clamp(min=1e-6)

    loss = F.softplus(-pair_sign * pair_student_diff)
    return (loss * total_weights).mean()


def distill_sample_weights(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mode: str,
    disagreement_weight: float,
    teacher_confidence_threshold: float,
    teacher_confidence_temperature: float,
    min_distill_weight: float,
) -> torch.Tensor | None:
    if mode == "none":
        return None

    teacher_margin = (teacher_logits[:, 1] - teacher_logits[:, 0]).detach()
    if mode == "score_agreement":
        student_margin = (student_logits[:, 1] - student_logits[:, 0]).detach()
        agrees = torch.sign(student_margin) == torch.sign(teacher_margin)
        low_weight = float(disagreement_weight)
        return torch.where(
            agrees,
            torch.ones_like(student_margin),
            torch.full_like(student_margin, low_weight),
        )

    if mode == "teacher_confidence":
        min_weight = min(max(float(min_distill_weight), 0.0), 1.0)
        temperature = max(float(teacher_confidence_temperature), 1e-6)
        confidence = teacher_margin.abs()
        soft_gate = torch.sigmoid((confidence - float(teacher_confidence_threshold)) / temperature)
        return min_weight + (1.0 - min_weight) * soft_gate

    raise ValueError(f"Unsupported distill gating mode: {mode}")


def distribution_regularization_loss(
    student_logits: torch.Tensor,
    labels: torch.Tensor,
    teacher_logits: torch.Tensor | None = None,
) -> torch.Tensor:
    student_probs = torch.softmax(student_logits, dim=1)[:, 1]
    label_rate = labels.float().mean()
    prior_loss = F.smooth_l1_loss(student_probs.mean(), label_rate)
    if teacher_logits is None:
        return prior_loss

    teacher_probs = torch.softmax(teacher_logits.detach(), dim=1)[:, 1]
    teacher_rate_loss = F.smooth_l1_loss(student_probs.mean(), teacher_probs.mean())
    return 0.5 * (prior_loss + teacher_rate_loss)


def confidence_weighted_distribution_regularization_loss(
    student_logits: torch.Tensor,
    labels: torch.Tensor,
    teacher_logits: torch.Tensor | None,
    trust: torch.Tensor,
) -> torch.Tensor:
    weights = trust.detach().float().clamp(0.0, 1.0)
    if weights.sum() <= 1e-6:
        return student_logits.new_tensor(0.0)
    weights = weights / weights.sum().clamp(min=1e-6)
    student_probs = torch.softmax(student_logits, dim=1)[:, 1]
    student_rate = (student_probs * weights).sum()
    label_rate = (labels.float() * weights).sum()
    prior_loss = F.smooth_l1_loss(student_rate, label_rate)
    if teacher_logits is None:
        return prior_loss

    teacher_probs = torch.softmax(teacher_logits.detach(), dim=1)[:, 1]
    teacher_rate = (teacher_probs * weights).sum()
    teacher_rate_loss = F.smooth_l1_loss(student_rate, teacher_rate)
    return 0.5 * (prior_loss + teacher_rate_loss)


def session_consistency_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    subject_ids: list[str],
    sessions: list[str],
    quality: torch.Tensor,
) -> torch.Tensor:
    if student_logits.shape[0] < 2:
        return student_logits.new_tensor(0.0)

    student_scores = (student_logits[:, 1] - student_logits[:, 0]).float()
    teacher_scores = (teacher_logits[:, 1] - teacher_logits[:, 0]).float().detach()
    quality_weights = 0.35 + 0.65 * quality.squeeze(1).clamp(0.0, 1.0)

    group_to_indices: dict[tuple[str, str], list[int]] = {}
    for idx, (subject_id, session) in enumerate(zip(subject_ids, sessions)):
        key = (str(subject_id), str(session))
        group_to_indices.setdefault(key, []).append(idx)

    group_losses: list[torch.Tensor] = []
    for indices in group_to_indices.values():
        if len(indices) < 2:
            continue
        index_tensor = torch.tensor(indices, device=student_logits.device, dtype=torch.long)
        student_group = student_scores.index_select(0, index_tensor)
        teacher_group = teacher_scores.index_select(0, index_tensor)
        group_weight = quality_weights.index_select(0, index_tensor).mean()

        mean_loss = F.smooth_l1_loss(student_group.mean(), teacher_group.mean(), reduction="mean")
        centered_loss = F.smooth_l1_loss(
            student_group - student_group.mean(),
            teacher_group - teacher_group.mean(),
            reduction="mean",
        )
        group_losses.append(group_weight * (mean_loss + 0.5 * centered_loss))

    if not group_losses:
        return student_logits.new_tensor(0.0)
    return torch.stack(group_losses).mean()


def collect_outputs(
    model: WESADPrivilegedTeacherNet,
    loader: DataLoader,
    device: str,
    pin_memory: bool,
    mode: str = "watch",
    aggregation: str = "window",
    baseline_reference: bool = False,
) -> tuple[list[int], list[float]]:
    model.eval()
    y_true: list[int] = []
    y_prob: list[float] = []
    subject_ids: list[str] = []
    sessions: list[str] = []

    with torch.no_grad():
        for batch in loader:
            watch_signal = batch["watch_signal"].to(device, non_blocking=pin_memory)
            wavelet = batch["wavelet_features"].to(device, non_blocking=pin_memory)
            quality = batch["watch_quality"].to(device, non_blocking=pin_memory)
            labels = batch["label"].to(device, non_blocking=pin_memory).long()
            baseline_kwargs = {}
            if baseline_reference:
                baseline_kwargs = {
                    "baseline_watch_signal": batch["baseline_watch_signal"].to(device, non_blocking=pin_memory),
                    "baseline_wavelet_features": batch["baseline_wavelet_features"].to(device, non_blocking=pin_memory),
                    "baseline_quality": batch["baseline_watch_quality"].to(device, non_blocking=pin_memory),
                }

            if mode == "watch":
                logits = model.forward_watch(
                    watch_signal,
                    wavelet,
                    quality,
                    baseline_signal=baseline_kwargs.get("baseline_watch_signal"),
                    baseline_wavelet_features=baseline_kwargs.get("baseline_wavelet_features"),
                    baseline_quality=baseline_kwargs.get("baseline_quality"),
                )["logits"]
            elif mode == "teacher":
                privileged_signal = batch["privileged_signal"].to(device, non_blocking=pin_memory)
                logits = model(
                    watch_signal,
                    wavelet,
                    quality,
                    privileged_signal=privileged_signal,
                    **baseline_kwargs,
                )["teacher_logits"]
            else:
                raise ValueError(f"Unsupported collect_outputs mode: {mode}")

            probs = torch.softmax(logits, dim=1)[:, 1]
            y_true.extend(labels.detach().cpu().tolist())
            y_prob.extend(probs.detach().cpu().tolist())
            subject_ids.extend(str(item) for item in batch["subject_id"])
            sessions.extend(str(item) for item in batch["session"])

    return aggregate_predictions(y_true, y_prob, subject_ids, sessions, aggregation)


def load_watch_checkpoint_into_privileged(model: WESADPrivilegedTeacherNet, checkpoint_path: Path) -> tuple[int, int]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported watch checkpoint format: {checkpoint_path}")

    current_state = model.state_dict()
    loaded = 0
    skipped = 0
    updated_state = dict(current_state)
    for source_key, value in checkpoint.items():
        if not torch.is_tensor(value):
            skipped += 1
            continue
        if source_key.startswith("classifier."):
            target_key = "watch_classifier." + source_key[len("classifier.") :]
        elif source_key.startswith("contrastive_head."):
            target_key = "watch_contrastive_head." + source_key[len("contrastive_head.") :]
        elif source_key.startswith("wavelet_predictor."):
            target_key = "wavelet_predictor." + source_key[len("wavelet_predictor.") :]
        else:
            target_key = "watch_encoder." + source_key

        target_value = current_state.get(target_key)
        if target_value is None or tuple(target_value.shape) != tuple(value.shape):
            skipped += 1
            continue
        updated_state[target_key] = value
        loaded += 1

    model.load_state_dict(updated_state, strict=True)
    return loaded, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a privileged WESAD teacher with deployable watch inference.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", "--wesad-root", dest="dataset_root", type=Path, required=True)
    parser.add_argument("--dataset-kind", type=str, default="wesad", choices=sorted(PRIVILEGED_DATASETS))
    parser.add_argument("--save-path", type=Path, required=True)
    parser.add_argument("--init-watch-checkpoint", type=Path, default=None)
    parser.add_argument("--include-initial-eval", action="store_true")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--focal-gamma", type=float, default=1.5)
    parser.add_argument("--watch-cls-weight", type=float, default=1.0)
    parser.add_argument("--watch-contrastive-weight", type=float, default=0.0)
    parser.add_argument("--contrastive-temperature", type=float, default=0.1)
    parser.add_argument("--privileged-cls-weight", type=float, default=0.05)
    parser.add_argument("--teacher-cls-weight", type=float, default=0.80)
    parser.add_argument("--distill-weight", type=float, default=0.10)
    parser.add_argument("--embedding-align-weight", type=float, default=0.05)
    parser.add_argument("--margin-match-weight", type=float, default=0.0)
    parser.add_argument("--normalized-margin-align-weight", type=float, default=0.0)
    parser.add_argument("--ranking-distill-weight", type=float, default=0.10)
    parser.add_argument("--distribution-weight", type=float, default=0.05)
    parser.add_argument("--subject-center-stability-weight", type=float, default=0.0)
    parser.add_argument("--validation-threshold-stability-weight", type=float, default=0.0)
    parser.add_argument("--session-consistency-weight", type=float, default=0.0)
    parser.add_argument("--distill-gating", type=str, default="none", choices=["none", "score_agreement", "teacher_confidence"])
    parser.add_argument("--distill-disagreement-weight", type=float, default=0.25)
    parser.add_argument("--teacher-confidence-threshold", type=float, default=1.0)
    parser.add_argument("--teacher-confidence-temperature", type=float, default=0.5)
    parser.add_argument("--min-distill-weight", type=float, default=0.2)
    parser.add_argument("--cross-confidence-distill", action="store_true")
    parser.add_argument(
        "--cross-confidence-targets",
        nargs="*",
        default=["kd", "ranking", "distribution"],
        choices=["kd", "ranking", "distribution"],
        help="Transfer terms gated by cross-confidence when --cross-confidence-distill is enabled.",
    )
    parser.add_argument("--cross-confidence-min-weight", type=float, default=0.0)
    parser.add_argument(
        "--kd-gate-mode",
        type=str,
        default="none",
        choices=["none", "teacher_true_confidence", "student_true_confidence"],
        help="Alternative true-class-confidence KD gate for teacher/student gated KD ablations.",
    )
    parser.add_argument("--kd-gate-min-weight", type=float, default=0.0)
    parser.add_argument("--distill-temp", type=float, default=2.0)
    parser.add_argument("--reliability-distill-weight", type=float, default=0.0)
    parser.add_argument("--clean-reliability-objective", action="store_true")
    parser.add_argument("--student-gated-correction", action="store_true")
    parser.add_argument("--correction-cls-weight", type=float, default=0.0)
    parser.add_argument("--correction-nondegradation-weight", type=float, default=0.0)
    parser.add_argument("--correction-align-weight", type=float, default=0.0)
    parser.add_argument("--correction-base-anchor-weight", type=float, default=0.0)
    parser.add_argument("--correction-margin", type=float, default=0.0)
    parser.add_argument("--correction-scale-init", type=float, default=0.05)
    parser.add_argument("--correction-alpha-init-bias", type=float, default=-2.0)
    parser.add_argument("--correction-alpha-max", type=float, default=0.35)
    parser.add_argument("--correction-mode", type=str, default="logit_mix", choices=["logit_mix", "margin_residual"])
    parser.add_argument("--alpha-helpfulness-weight", type=float, default=0.0)
    parser.add_argument("--alpha-help-margin", type=float, default=0.0)
    parser.add_argument("--alpha-sparsity-weight", type=float, default=0.0)
    parser.add_argument("--elastic-residual-weight", type=float, default=0.0)
    parser.add_argument("--elastic-alpha-target-weight", type=float, default=0.0)
    parser.add_argument("--elastic-reliability-temp", type=float, default=0.25)
    parser.add_argument("--elastic-uncertainty-temp", type=float, default=1.0)
    parser.add_argument("--elastic-label-margin", type=float, default=2.0)
    parser.add_argument("--wavelet-weight", type=float, default=0.05)
    parser.add_argument("--baseline-relative-weight", type=float, default=0.0)
    parser.add_argument("--baseline-relative-margin", type=float, default=0.2)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--disable-ema-eval", action="store_true")
    parser.add_argument("--monitor", type=str, default="balanced_acc", choices=["acc", "balanced_acc", "f1", "auroc"])
    parser.add_argument("--threshold-metric", type=str, default="monitor", choices=["monitor", "acc", "balanced_acc", "f1", "auroc"])
    parser.add_argument("--selection-target", type=str, default="watch", choices=["watch", "teacher"])
    parser.add_argument("--selection-mode", type=str, default="early_stop", choices=["fixed_epoch", "early_stop"])
    parser.add_argument("--selection-epoch", type=int, default=20)
    parser.add_argument("--early-stop-patience", type=int, default=12)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--metrics-path", type=Path, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--calm-sessions", nargs="*", default=None)
    parser.add_argument("--stress-sessions", nargs="*", default=None)
    parser.add_argument("--wavelet", type=str, default="db4")
    parser.add_argument("--wavelet-level", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-aggregation", type=str, default="window", choices=["window", "session_mean"])
    parser.add_argument("--threshold-mode", type=str, default="val_search", choices=["val_search", "fixed"])
    parser.add_argument("--fixed-threshold", type=float, default=0.5)
    parser.add_argument("--baseline-reference", action="store_true")
    parser.add_argument("--subject-aware-batching", action="store_true")
    parser.add_argument("--cache-subjects", type=int, default=2)
    parser.add_argument("--catsa-privileged-modalities", nargs="*", default=["EDA", "TEMP"])
    parser.add_argument("--watch-model-dim", type=int, default=192)
    parser.add_argument("--watch-transformer-layers", type=int, default=2)
    parser.add_argument("--watch-transformer-heads", type=int, default=4)
    parser.add_argument("--watch-fusion-hidden-dim", type=int, default=256)
    parser.add_argument("--watch-embed-dim", type=int, default=160)
    parser.add_argument("--watch-enhancement", type=str, default="none", choices=["none", "motion_disentangled", "acc_concat"])
    parser.add_argument("--watch-motion-mode", type=str, default="strong", choices=["strong", "residual", "scaled"])
    parser.add_argument("--align-proj-dim", type=int, default=128)
    parser.add_argument(
        "--watch-backbone",
        type=str,
        default="wavelet_guided",
        choices=["wavelet_guided", "resnet18_1d", "resnet34_1d", "resnet50_1d"],
    )
    args = parser.parse_args()
    if args.watch_backbone != "wavelet_guided" and args.watch_enhancement != "none":
        raise ValueError("watch-enhancement is currently only supported for the wavelet_guided backbone.")

    set_random_seed(args.seed)
    calm_sessions = maybe_parse_sessions(args.calm_sessions, DEFAULT_CALM_SESSIONS)
    stress_sessions = maybe_parse_sessions(args.stress_sessions, DEFAULT_STRESS_SESSIONS)
    include_sessions = calm_sessions + stress_sessions
    dataset_cls = PRIVILEGED_DATASETS[args.dataset_kind]
    dataset_kwargs = (
        {"privileged_modalities": [str(item) for item in args.catsa_privileged_modalities]}
        if args.dataset_kind == "catsa"
        else {}
    )

    train_ds = dataset_cls(
        manifest_csv=args.manifest,
        split="train",
        wesad_root=args.dataset_root,
        include_sessions=include_sessions,
        cache_subjects=args.cache_subjects,
        wavelet=args.wavelet,
        wavelet_level=args.wavelet_level,
        baseline_reference=args.baseline_reference,
        **dataset_kwargs,
    )
    val_ds = dataset_cls(
        manifest_csv=args.manifest,
        split="val",
        wesad_root=args.dataset_root,
        include_sessions=include_sessions,
        cache_subjects=args.cache_subjects,
        wavelet=args.wavelet,
        wavelet_level=args.wavelet_level,
        baseline_reference=args.baseline_reference,
        **dataset_kwargs,
    )
    test_ds = dataset_cls(
        manifest_csv=args.manifest,
        split="test",
        wesad_root=args.dataset_root,
        include_sessions=include_sessions,
        cache_subjects=args.cache_subjects,
        wavelet=args.wavelet,
        wavelet_level=args.wavelet_level,
        baseline_reference=args.baseline_reference,
        **dataset_kwargs,
    )
    if len(train_ds) == 0 or len(test_ds) == 0:
        raise ValueError("Train or test split is empty after session filtering.")

    train_batch_sampler = None
    if args.subject_aware_batching:
        train_batch_sampler = SubjectAwareBatchSampler(
            train_ds.manifest,
            batch_size=args.batch_size,
            seed=args.seed,
        )

    train_loader = build_loader(
        train_ds,
        args.batch_size,
        shuffle=train_batch_sampler is None,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        batch_sampler=train_batch_sampler,
    )
    test_loader = build_loader(test_ds, args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)
    has_val_split = len(val_ds) > 0
    val_loader = build_loader(val_ds, args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory) if has_val_split else test_loader

    sample = train_ds[0]
    wavelet_dim = int(sample["wavelet_features"].shape[0])
    privileged_channels = int(sample["privileged_signal"].shape[0])
    model = WESADPrivilegedTeacherNet(
        wavelet_dim=wavelet_dim,
        privileged_channels=privileged_channels,
        watch_backbone=args.watch_backbone,
        embed_dim=args.watch_embed_dim,
        align_dim=args.align_proj_dim,
        model_dim=args.watch_model_dim,
        transformer_layers=args.watch_transformer_layers,
        transformer_heads=args.watch_transformer_heads,
        fusion_hidden_dim=args.watch_fusion_hidden_dim,
        watch_enhancement=args.watch_enhancement,
        watch_motion_mode=args.watch_motion_mode,
        use_student_gated_correction=args.student_gated_correction,
        correction_scale_init=args.correction_scale_init,
        correction_alpha_init_bias=args.correction_alpha_init_bias,
        correction_alpha_max=args.correction_alpha_max,
        correction_mode=args.correction_mode,
    ).to(args.device)
    if args.init_watch_checkpoint is not None:
        loaded, skipped = load_watch_checkpoint_into_privileged(model, args.init_watch_checkpoint)
        print(f"init_watch_checkpoint={args.init_watch_checkpoint} loaded_tensors={loaded} skipped_tensors={skipped}")
    use_ema_eval = not args.disable_ema_eval
    ema_model = copy.deepcopy(model).to(args.device) if use_ema_eval else None
    if ema_model is not None:
        ema_model.eval()
        for param in ema_model.parameters():
            param.requires_grad_(False)

    total_params = sum(p.numel() for p in model.parameters())
    watch_only_params = sum(
        p.numel()
        for name, p in model.named_parameters()
        if name.startswith("watch_encoder")
        or name.startswith("watch_classifier")
        or name.startswith("reliability_head")
        or name.startswith("deploy_correction")
        or name.startswith("correction_norm")
        or name == "correction_scale"
    )
    print(f"teacher_params={total_params}")
    print(f"watch_inference_params={watch_only_params}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    class_counts = train_ds.manifest["label"].value_counts().sort_index()
    neg = float(class_counts.get(0, 1.0))
    pos = float(class_counts.get(1, 1.0))
    class_weights = torch.tensor([1.0, neg / max(pos, 1.0)], device=args.device, dtype=torch.float32)

    stopper = EarlyStopping(patience=args.early_stop_patience, mode="max", min_delta=args.min_delta)
    best_state = None
    best_watch_metrics = None
    best_teacher_metrics = None
    best_watch_threshold = 0.5
    best_teacher_threshold = 0.5
    history_rows: list[dict[str, float | int]] = []

    reliability_enabled = args.reliability_distill_weight > 0.0
    cross_confidence_targets = set(args.cross_confidence_targets or [])
    cross_confidence_enabled = args.cross_confidence_distill and bool(cross_confidence_targets)
    if cross_confidence_enabled and args.kd_gate_mode != "none":
        raise ValueError("--kd-gate-mode is mutually exclusive with --cross-confidence-distill.")
    if args.distill_gating != "none" and args.kd_gate_mode != "none":
        raise ValueError("--kd-gate-mode is mutually exclusive with the legacy --distill-gating modes.")
    clean_reliability_objective = reliability_enabled and args.clean_reliability_objective
    effective_distill_weight = 0.0 if clean_reliability_objective else args.distill_weight
    effective_embedding_align_weight = 0.0 if clean_reliability_objective else args.embedding_align_weight
    effective_margin_match_weight = 0.0 if clean_reliability_objective else args.margin_match_weight
    effective_normalized_margin_align_weight = 0.0 if clean_reliability_objective else args.normalized_margin_align_weight
    effective_ranking_distill_weight = 0.0 if clean_reliability_objective else args.ranking_distill_weight
    effective_distribution_weight = 0.0 if clean_reliability_objective else args.distribution_weight
    effective_subject_center_stability_weight = 0.0 if clean_reliability_objective else args.subject_center_stability_weight
    effective_validation_threshold_stability_weight = 0.0 if clean_reliability_objective else args.validation_threshold_stability_weight
    effective_session_consistency_weight = 0.0 if clean_reliability_objective else args.session_consistency_weight

    print(f"train windows={len(train_ds)} val windows={len(val_ds)} test windows={len(test_ds)}")
    if not has_val_split:
        print("selection_warning=val split missing; using test split for threshold/model selection")
    print(f"dataset={args.dataset_kind}")
    print(f"wavelet_dim={wavelet_dim} privileged_channels={privileged_channels}")
    print(f"watch_backbone={args.watch_backbone}")
    if args.watch_backbone == "wavelet_guided":
        print(f"watch_enhancement={args.watch_enhancement}")
        print(
            "watch_arch="
            f"dim:{args.watch_model_dim} "
            f"layers:{args.watch_transformer_layers} "
            f"heads:{args.watch_transformer_heads} "
            f"fusion_hidden:{args.watch_fusion_hidden_dim} "
            f"embed:{args.watch_embed_dim}"
        )
    print(f"calm_sessions={calm_sessions}")
    print(f"stress_sessions={stress_sessions}")
    print(f"cache_subjects={args.cache_subjects}")
    print(
        f"loss_weights=watch_cls:{args.watch_cls_weight:.2f} "
        f"watch_contrastive:{args.watch_contrastive_weight:.2f} "
        f"privileged_cls:{args.privileged_cls_weight:.2f} "
        f"teacher_cls:{args.teacher_cls_weight:.2f} "
        f"distill:{effective_distill_weight:.2f} "
        f"emb_align:{effective_embedding_align_weight:.2f} "
        f"margin:{effective_margin_match_weight:.2f} "
        f"norm_margin:{effective_normalized_margin_align_weight:.2f} "
        f"rank:{effective_ranking_distill_weight:.2f} "
        f"distr:{effective_distribution_weight:.2f} "
        f"subj_center:{effective_subject_center_stability_weight:.2f} "
        f"val_thr:{effective_validation_threshold_stability_weight:.2f} "
        f"session:{effective_session_consistency_weight:.2f} "
        f"rel:{args.reliability_distill_weight:.2f} "
        f"corr_cls:{args.correction_cls_weight:.2f} "
        f"corr_nd:{args.correction_nondegradation_weight:.2f} "
        f"corr_align:{args.correction_align_weight:.2f} "
        f"corr_anchor:{args.correction_base_anchor_weight:.2f} "
        f"alpha_help:{args.alpha_helpfulness_weight:.2f} "
        f"alpha_sparse:{args.alpha_sparsity_weight:.2f} "
        f"elastic_resid:{args.elastic_residual_weight:.2f} "
        f"elastic_alpha:{args.elastic_alpha_target_weight:.2f} "
        f"wavelet:{args.wavelet_weight:.2f} "
        f"baseline_relative:{args.baseline_relative_weight:.2f}"
    )
    print(
        f"student_gated_correction={'on' if args.student_gated_correction else 'off'} "
        f"correction_margin={args.correction_margin:.4f} "
        f"correction_scale_init={args.correction_scale_init:.4f} "
        f"correction_alpha_init_bias={args.correction_alpha_init_bias:.4f} "
        f"correction_alpha_max={args.correction_alpha_max:.4f} "
        f"correction_mode={args.correction_mode} "
        f"alpha_help_margin={args.alpha_help_margin:.4f} "
        f"elastic_tau={args.elastic_reliability_temp:.3f} "
        f"elastic_uncertainty_tau={args.elastic_uncertainty_temp:.3f} "
        f"elastic_label_margin={args.elastic_label_margin:.3f}"
    )
    print(
        f"distill_gating={args.distill_gating} "
        f"distill_disagreement_weight={args.distill_disagreement_weight:.2f} "
        f"teacher_conf_threshold={args.teacher_confidence_threshold:.2f} "
        f"teacher_conf_temp={args.teacher_confidence_temperature:.2f} "
        f"min_distill_weight={args.min_distill_weight:.2f}"
    )
    print(
        f"cross_confidence_distill={'on' if cross_confidence_enabled else 'off'} "
        f"targets={sorted(cross_confidence_targets) if cross_confidence_enabled else []} "
        f"min_weight={args.cross_confidence_min_weight:.2f}"
    )
    print(f"kd_gate_mode={args.kd_gate_mode} kd_gate_min_weight={args.kd_gate_min_weight:.2f}")
    print(
        f"reliability_distill={'on' if reliability_enabled else 'off'} "
        f"clean_reliability_objective={'on' if clean_reliability_objective else 'off'}"
    )
    print(
        f"selection_mode={args.selection_mode} "
        f"selection_target={args.selection_target} "
        f"monitor={args.monitor} "
        f"early_stop_patience={args.early_stop_patience}"
    )
    print(f"baseline_reference={'on' if args.baseline_reference else 'off'}")
    print(f"subject_aware_batching={'on' if args.subject_aware_batching else 'off'}")
    print(f"eval_aggregation={args.eval_aggregation}")
    print(f"threshold_mode={args.threshold_mode} fixed_threshold={args.fixed_threshold:.4f}")
    threshold_metric = args.monitor if args.threshold_metric == "monitor" else args.threshold_metric
    print(f"threshold_metric={threshold_metric}")
    print(f"ema_eval={'on' if use_ema_eval else 'off'} ema_decay={args.ema_decay:.4f} seed={args.seed}")
    print("priv_mode=chest_teacher_distill")

    if args.include_initial_eval and args.selection_mode == "early_stop":
        eval_model = ema_model if ema_model is not None else model
        val_watch_true, val_watch_prob = collect_outputs(
            eval_model,
            val_loader,
            args.device,
            args.pin_memory,
            mode="watch",
            aggregation=args.eval_aggregation,
            baseline_reference=args.baseline_reference,
        )
        if args.threshold_mode == "fixed":
            watch_threshold = args.fixed_threshold
            val_watch_metrics = evaluate_with_threshold(val_watch_true, val_watch_prob, threshold=watch_threshold)
        else:
            watch_threshold, val_watch_metrics = select_threshold(val_watch_true, val_watch_prob, metric=threshold_metric)
        test_watch_true, test_watch_prob = collect_outputs(
            eval_model,
            test_loader,
            args.device,
            args.pin_memory,
            mode="watch",
            aggregation=args.eval_aggregation,
            baseline_reference=args.baseline_reference,
        )
        test_watch_metrics = evaluate_with_threshold(test_watch_true, test_watch_prob, threshold=watch_threshold)

        val_teacher_true, val_teacher_prob = collect_outputs(
            eval_model,
            val_loader,
            args.device,
            args.pin_memory,
            mode="teacher",
            aggregation=args.eval_aggregation,
            baseline_reference=args.baseline_reference,
        )
        if args.threshold_mode == "fixed":
            teacher_threshold = args.fixed_threshold
            val_teacher_metrics = evaluate_with_threshold(val_teacher_true, val_teacher_prob, threshold=teacher_threshold)
        else:
            teacher_threshold, val_teacher_metrics = select_threshold(val_teacher_true, val_teacher_prob, metric=threshold_metric)
        test_teacher_true, test_teacher_prob = collect_outputs(
            eval_model,
            test_loader,
            args.device,
            args.pin_memory,
            mode="teacher",
            aggregation=args.eval_aggregation,
            baseline_reference=args.baseline_reference,
        )
        test_teacher_metrics = evaluate_with_threshold(test_teacher_true, test_teacher_prob, threshold=teacher_threshold)

        selection_metrics = val_watch_metrics if args.selection_target == "watch" else val_teacher_metrics
        score = selection_metrics[args.monitor]
        if stopper.step(score, 0):
            best_state = {key: value.detach().cpu() for key, value in eval_model.state_dict().items()}
            best_watch_threshold = watch_threshold
            best_teacher_threshold = teacher_threshold
            best_watch_metrics = test_watch_metrics
            best_teacher_metrics = test_teacher_metrics
            print(
                f"initial candidate {args.selection_target}_{args.monitor}={score:.4f} "
                f"| watch_threshold={best_watch_threshold:.4f} "
                f"teacher_threshold={best_teacher_threshold:.4f} "
                f"| watch_test_balanced_acc={best_watch_metrics['balanced_acc']:.4f} "
                f"teacher_test_balanced_acc={best_teacher_metrics['balanced_acc']:.4f}"
            )

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        epoch_brel_subject_fracs: list[float] = []
        epoch_brel_sample_fracs: list[float] = []
        epoch_brel_active_batches = 0
        epoch_distill_gate_means: list[float] = []
        epoch_cross_confidence_trust_means: list[float] = []
        epoch_cross_confidence_trust_high_fracs: list[float] = []
        epoch_kd_gate_trust_means: list[float] = []
        epoch_kd_gate_trust_high_fracs: list[float] = []
        epoch_reliability_trust_means: list[float] = []
        epoch_reliability_trust_high_fracs: list[float] = []
        epoch_reliability_kd_losses: list[float] = []
        epoch_reliability_cal_losses: list[float] = []
        epoch_alpha_means: list[float] = []
        epoch_alpha_helpful_fracs: list[float] = []
        epoch_alpha_sparsity_losses: list[float] = []
        epoch_elastic_residual_losses: list[float] = []
        epoch_elastic_reliability_means: list[float] = []
        epoch_elastic_target_delta_means: list[float] = []
        epoch_elastic_alpha_losses: list[float] = []
        epoch_elastic_alpha_target_means: list[float] = []
        progress = tqdm(train_loader, desc=f"wesad-priv epoch {epoch + 1}/{args.epochs}", leave=True)

        for batch in progress:
            watch_signal = batch["watch_signal"].to(args.device, non_blocking=args.pin_memory)
            privileged_signal = batch["privileged_signal"].to(args.device, non_blocking=args.pin_memory)
            wavelet = batch["wavelet_features"].to(args.device, non_blocking=args.pin_memory)
            quality = batch["watch_quality"].to(args.device, non_blocking=args.pin_memory)
            labels = batch["label"].to(args.device, non_blocking=args.pin_memory).long()
            baseline_kwargs = {}
            if args.baseline_reference:
                baseline_kwargs = {
                    "baseline_watch_signal": batch["baseline_watch_signal"].to(args.device, non_blocking=args.pin_memory),
                    "baseline_wavelet_features": batch["baseline_wavelet_features"].to(args.device, non_blocking=args.pin_memory),
                    "baseline_quality": batch["baseline_watch_quality"].to(args.device, non_blocking=args.pin_memory),
                }

            out = model(
                watch_signal,
                wavelet,
                quality,
                privileged_signal=privileged_signal,
                **baseline_kwargs,
            )
            zero_loss = out["logits"].new_tensor(0.0)
            watch_cls_loss = quality_aware_focal_loss(
                out["logits"],
                labels,
                quality,
                class_weights=class_weights,
                gamma=args.focal_gamma,
                label_smoothing=args.label_smoothing,
            )
            watch_contrastive_loss = supervised_contrastive_loss(
                out["contrastive"],
                labels,
                quality,
                temperature=args.contrastive_temperature,
            )
            privileged_cls_loss = quality_aware_focal_loss(
                out["privileged_logits"],
                labels,
                quality,
                class_weights=class_weights,
                gamma=args.focal_gamma,
                label_smoothing=args.label_smoothing,
            )
            teacher_cls_loss = quality_aware_focal_loss(
                out["teacher_logits"],
                labels,
                quality,
                class_weights=class_weights,
                gamma=args.focal_gamma,
                label_smoothing=args.label_smoothing,
            )
            base_anchor_loss = zero_loss
            if args.correction_base_anchor_weight > 0 and "base_logits" in out:
                base_anchor_loss = quality_aware_focal_loss(
                    out["base_logits"],
                    labels,
                    quality,
                    class_weights=class_weights,
                    gamma=args.focal_gamma,
                    label_smoothing=args.label_smoothing,
                )
            correction_cls_loss = zero_loss
            correction_nondegradation_loss = zero_loss
            correction_align_loss = zero_loss
            alpha_loss = zero_loss
            alpha_sparsity_loss = zero_loss
            alpha_mean = zero_loss
            alpha_helpful_frac = zero_loss
            elastic_residual_loss = zero_loss
            elastic_alpha_loss = zero_loss
            elastic_reliability_mean = zero_loss
            elastic_target_delta_mean = zero_loss
            elastic_alpha_target_mean = zero_loss
            if args.student_gated_correction and "privileged_correction_logits" in out:
                correction_cls_loss = quality_aware_focal_loss(
                    out["privileged_correction_logits"],
                    labels,
                    quality,
                    class_weights=class_weights,
                    gamma=args.focal_gamma,
                    label_smoothing=args.label_smoothing,
                )
                if args.correction_nondegradation_weight > 0 and "base_logits" in out:
                    privileged_nd = nondegradation_loss(
                        out["privileged_correction_logits"],
                        out["base_logits"],
                        labels,
                        quality,
                        class_weights=class_weights,
                        gamma=args.focal_gamma,
                        label_smoothing=args.label_smoothing,
                        margin=args.correction_margin,
                    )
                    deploy_nd = nondegradation_loss(
                        out["logits"],
                        out["base_logits"],
                        labels,
                        quality,
                        class_weights=class_weights,
                        gamma=args.focal_gamma,
                        label_smoothing=args.label_smoothing,
                        margin=args.correction_margin,
                    )
                    correction_nondegradation_loss = 0.5 * (privileged_nd + deploy_nd)
                if (
                    args.correction_align_weight > 0
                    and "deploy_correction_delta" in out
                    and "privileged_correction_delta" in out
                    and "base_logits" in out
                ):
                    correction_align_loss = helpful_correction_alignment_loss(
                        out["deploy_correction_delta"],
                        out["privileged_correction_delta"],
                        out["privileged_correction_logits"],
                        out["base_logits"],
                        labels,
                        quality,
                        class_weights=class_weights,
                        gamma=args.focal_gamma,
                        label_smoothing=args.label_smoothing,
                    )
                if "deploy_correction_alpha" in out:
                    alpha_mean = out["deploy_correction_alpha"].detach().mean()
                    epoch_alpha_means.append(float(alpha_mean.detach().cpu().item()))
                if args.alpha_helpfulness_weight > 0 and "deploy_correction_alpha" in out and "base_logits" in out:
                    alpha_for_helpfulness = out.get("deploy_correction_alpha_unit", out["deploy_correction_alpha"])
                    alpha_loss, alpha_helpful_frac = alpha_helpfulness_loss(
                        alpha_for_helpfulness,
                        out["privileged_correction_logits"],
                        out["base_logits"],
                        labels,
                        quality,
                        class_weights=class_weights,
                        gamma=args.focal_gamma,
                        label_smoothing=args.label_smoothing,
                        margin=args.alpha_help_margin,
                    )
                    epoch_alpha_helpful_fracs.append(float(alpha_helpful_frac.detach().cpu().item()))
                if args.alpha_sparsity_weight > 0 and "deploy_correction_alpha" in out:
                    alpha_sparsity_loss = quality_weighted_mean(
                        out["deploy_correction_alpha"].squeeze(1),
                        quality,
                    )
                    epoch_alpha_sparsity_losses.append(float(alpha_sparsity_loss.detach().cpu().item()))
                if (
                    args.elastic_residual_weight > 0
                    and "deploy_corrected_logits" in out
                    and "base_logits" in out
                    and "teacher_logits" in out
                ):
                    elastic_residual_loss, elastic_reliability_mean, elastic_target_delta_mean = elastic_privileged_residual_loss(
                        out["base_logits"],
                        out["deploy_corrected_logits"],
                        out["teacher_logits"],
                        labels,
                        quality,
                        class_weights=class_weights,
                        gamma=args.focal_gamma,
                        label_smoothing=args.label_smoothing,
                        reliability_temp=args.elastic_reliability_temp,
                        label_margin=args.elastic_label_margin,
                    )
                    epoch_elastic_residual_losses.append(float(elastic_residual_loss.detach().cpu().item()))
                    epoch_elastic_reliability_means.append(float(elastic_reliability_mean.detach().cpu().item()))
                    epoch_elastic_target_delta_means.append(float(elastic_target_delta_mean.detach().cpu().item()))
                if (
                    args.elastic_alpha_target_weight > 0
                    and "deploy_correction_alpha_unit" in out
                    and "base_logits" in out
                    and "teacher_logits" in out
                ):
                    elastic_alpha_loss, elastic_alpha_target_mean = elastic_alpha_target_loss(
                        out["deploy_correction_alpha_unit"],
                        out["base_logits"],
                        out["teacher_logits"],
                        labels,
                        quality,
                        class_weights=class_weights,
                        gamma=args.focal_gamma,
                        label_smoothing=args.label_smoothing,
                        reliability_temp=args.elastic_reliability_temp,
                        uncertainty_temp=args.elastic_uncertainty_temp,
                    )
                    epoch_elastic_alpha_losses.append(float(elastic_alpha_loss.detach().cpu().item()))
                    epoch_elastic_alpha_target_means.append(float(elastic_alpha_target_mean.detach().cpu().item()))
            distill_weights = distill_sample_weights(
                out["logits"],
                out["teacher_logits"],
                mode=args.distill_gating,
                disagreement_weight=args.distill_disagreement_weight,
                teacher_confidence_threshold=args.teacher_confidence_threshold,
                teacher_confidence_temperature=args.teacher_confidence_temperature,
                min_distill_weight=args.min_distill_weight,
            )
            distill_gate_mean = (
                distill_weights.mean().detach()
                if distill_weights is not None
                else out["logits"].new_tensor(1.0)
            )
            epoch_distill_gate_means.append(float(distill_gate_mean.detach().cpu().item()))
            trust_signal = None
            cross_confidence_trust = None
            cross_confidence_trust_mean = zero_loss
            kd_gate_trust = None
            kd_gate_trust_mean = zero_loss
            if cross_confidence_enabled or reliability_enabled:
                trust_signal = cross_calibrated_trust(
                    out["logits"],
                    out["teacher_logits"],
                    labels,
                    quality,
                )
            if cross_confidence_enabled and trust_signal is not None:
                cross_confidence_trust = apply_min_trust_floor(
                    trust_signal,
                    min_weight=args.cross_confidence_min_weight,
                )
                cross_confidence_trust_mean = cross_confidence_trust.mean()
                epoch_cross_confidence_trust_means.append(float(cross_confidence_trust_mean.detach().cpu().item()))
                epoch_cross_confidence_trust_high_fracs.append(
                    float((cross_confidence_trust >= 0.5).float().mean().detach().cpu().item())
                )
            if args.kd_gate_mode != "none":
                kd_gate_trust = true_class_kd_gate(
                    out["logits"],
                    out["teacher_logits"],
                    labels,
                    quality,
                    mode=args.kd_gate_mode,
                    min_weight=args.kd_gate_min_weight,
                )
                kd_gate_trust_mean = kd_gate_trust.mean()
                epoch_kd_gate_trust_means.append(float(kd_gate_trust_mean.detach().cpu().item()))
                epoch_kd_gate_trust_high_fracs.append(
                    float((kd_gate_trust >= 0.5).float().mean().detach().cpu().item())
                )
            if cross_confidence_enabled and "kd" in cross_confidence_targets and cross_confidence_trust is not None:
                distill_loss = trust_weighted_kl_loss(
                    out["logits"],
                    out["teacher_logits"],
                    cross_confidence_trust,
                    temperature=args.distill_temp,
                )
            elif kd_gate_trust is not None:
                distill_loss = trust_weighted_kl_loss(
                    out["logits"],
                    out["teacher_logits"],
                    kd_gate_trust,
                    temperature=args.distill_temp,
                )
            else:
                distill_loss = distillation_kl_loss(
                    out["logits"],
                    out["teacher_logits"],
                    quality=quality,
                    temperature=args.distill_temp,
                    sample_weights=distill_weights,
                )
            reliability_loss = zero_loss
            reliability_trust_mean = zero_loss
            if reliability_enabled and "reliability_logit" in out and trust_signal is not None:
                reliability_trust = trust_signal
                reliability_kd_loss = trust_weighted_kl_loss(
                    out["logits"],
                    out["teacher_logits"],
                    reliability_trust,
                    temperature=args.distill_temp,
                )
                reliability_cal_loss = reliability_bce_loss(out["reliability_logit"], reliability_trust)
                reliability_loss = reliability_kd_loss + reliability_cal_loss
                reliability_trust_mean = reliability_trust.mean()
                epoch_reliability_trust_means.append(float(reliability_trust_mean.detach().cpu().item()))
                epoch_reliability_trust_high_fracs.append(float((reliability_trust >= 0.5).float().mean().detach().cpu().item()))
                epoch_reliability_kd_losses.append(float(reliability_kd_loss.detach().cpu().item()))
                epoch_reliability_cal_losses.append(float(reliability_cal_loss.detach().cpu().item()))
            align_loss = embedding_alignment_loss(
                out["watch_align"],
                out["teacher_align"],
                quality=quality,
                sample_weights=distill_weights,
            )
            margin_loss = margin_matching_loss(
                out["logits"],
                out["teacher_logits"],
                quality=quality,
                sample_weights=distill_weights,
            )
            normalized_margin_loss = normalized_margin_alignment_loss(
                out["logits"],
                out["teacher_logits"],
                quality=quality,
                sample_weights=distill_weights,
            )
            ranking_weights = distill_weights
            if cross_confidence_enabled and "ranking" in cross_confidence_targets and cross_confidence_trust is not None:
                ranking_weights = cross_confidence_trust
            ranking_loss = ranking_distillation_loss(
                out["logits"],
                out["teacher_logits"],
                quality=quality,
                distill_weights=ranking_weights,
            )
            if cross_confidence_enabled and "distribution" in cross_confidence_targets and cross_confidence_trust is not None:
                distribution_loss = confidence_weighted_distribution_regularization_loss(
                    out["logits"],
                    labels,
                    teacher_logits=out["teacher_logits"],
                    trust=cross_confidence_trust,
                )
            else:
                distribution_loss = distribution_regularization_loss(
                    out["logits"],
                    labels,
                    teacher_logits=out["teacher_logits"],
                )
            subject_center_loss = subject_center_stability_loss(
                out["logits"],
                labels,
                [str(item) for item in batch["subject_id"]],
                quality=quality,
            )
            validation_threshold_loss = validation_threshold_stability_loss(
                out["logits"],
                labels,
                [str(item) for item in batch["subject_id"]],
                [str(item) for item in batch["session"]],
                quality=quality,
            )
            session_consistency_loss = session_consistency_distillation_loss(
                out["logits"],
                out["teacher_logits"],
                [str(item) for item in batch["subject_id"]],
                [str(item) for item in batch["session"]],
                quality=quality,
            )
            wavelet_loss = F.smooth_l1_loss(out["wavelet_pred"], wavelet)
            br_loss, br_stats = baseline_relative_margin_loss(
                out["logits"],
                labels,
                [str(item) for item in batch["subject_id"]],
                quality,
                margin=args.baseline_relative_margin,
                return_stats=True,
            )
            loss = (
                args.watch_cls_weight * watch_cls_loss
                + args.watch_contrastive_weight * watch_contrastive_loss
                + args.privileged_cls_weight * privileged_cls_loss
                + args.teacher_cls_weight * teacher_cls_loss
                + effective_distill_weight * distill_loss
                + effective_embedding_align_weight * align_loss
                + effective_margin_match_weight * margin_loss
                + effective_normalized_margin_align_weight * normalized_margin_loss
                + effective_ranking_distill_weight * ranking_loss
                + effective_distribution_weight * distribution_loss
                + effective_subject_center_stability_weight * subject_center_loss
                + effective_validation_threshold_stability_weight * validation_threshold_loss
                + effective_session_consistency_weight * session_consistency_loss
                + args.reliability_distill_weight * reliability_loss
                + args.correction_base_anchor_weight * base_anchor_loss
                + args.correction_cls_weight * correction_cls_loss
                + args.correction_nondegradation_weight * correction_nondegradation_loss
                + args.correction_align_weight * correction_align_loss
                + args.alpha_helpfulness_weight * alpha_loss
                + args.alpha_sparsity_weight * alpha_sparsity_loss
                + args.elastic_residual_weight * elastic_residual_loss
                + args.elastic_alpha_target_weight * elastic_alpha_loss
                + args.wavelet_weight * wavelet_loss
                + args.baseline_relative_weight * br_loss
            )

            epoch_brel_subject_fracs.append(float(br_stats["active_subject_fraction"]))
            epoch_brel_sample_fracs.append(float(br_stats["active_sample_fraction"]))
            if br_stats["active_subjects"] > 0:
                epoch_brel_active_batches += 1

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if ema_model is not None:
                update_ema_model(ema_model, model, args.ema_decay)
            total_loss += float(loss.item())
            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                watch=f"{watch_cls_loss.item():.4f}",
                con=f"{watch_contrastive_loss.item():.4f}",
                teacher=f"{teacher_cls_loss.item():.4f}",
                gate=f"{distill_gate_mean.item():.3f}",
                distill=f"{distill_loss.item():.4f}",
                rel=f"{reliability_loss.item():.4f}",
                trust=f"{reliability_trust_mean.item():.3f}",
                xconf=f"{cross_confidence_trust_mean.item():.3f}",
                kdgate=f"{kd_gate_trust_mean.item():.3f}",
                align=f"{align_loss.item():.4f}",
                margin=f"{margin_loss.item():.4f}",
                nmargin=f"{normalized_margin_loss.item():.4f}",
                rank=f"{ranking_loss.item():.4f}",
                distr=f"{distribution_loss.item():.4f}",
                center=f"{subject_center_loss.item():.4f}",
                vthr=f"{validation_threshold_loss.item():.4f}",
                sess=f"{session_consistency_loss.item():.4f}",
                cbase=f"{base_anchor_loss.item():.4f}",
                cpriv=f"{correction_cls_loss.item():.4f}",
                cnd=f"{correction_nondegradation_loss.item():.4f}",
                calign=f"{correction_align_loss.item():.4f}",
                alpha=f"{alpha_mean.item():.3f}",
                ahelp=f"{alpha_helpful_frac.item():.3f}",
                asparse=f"{alpha_sparsity_loss.item():.4f}",
                eres=f"{elastic_residual_loss.item():.4f}",
                erel=f"{elastic_reliability_mean.item():.3f}",
                ealpha=f"{elastic_alpha_loss.item():.4f}",
                wav=f"{wavelet_loss.item():.4f}",
            )

        scheduler.step()
        train_loss = total_loss / max(len(train_loader), 1)
        eval_model = ema_model if ema_model is not None else model

        val_watch_true, val_watch_prob = collect_outputs(
            eval_model,
            val_loader,
            args.device,
            args.pin_memory,
            mode="watch",
            aggregation=args.eval_aggregation,
            baseline_reference=args.baseline_reference,
        )
        if args.threshold_mode == "fixed":
            watch_threshold = args.fixed_threshold
            val_watch_metrics = evaluate_with_threshold(val_watch_true, val_watch_prob, threshold=watch_threshold)
        else:
            watch_threshold, val_watch_metrics = select_threshold(val_watch_true, val_watch_prob, metric=threshold_metric)
        test_watch_true, test_watch_prob = collect_outputs(
            eval_model,
            test_loader,
            args.device,
            args.pin_memory,
            mode="watch",
            aggregation=args.eval_aggregation,
            baseline_reference=args.baseline_reference,
        )
        test_watch_metrics = evaluate_with_threshold(test_watch_true, test_watch_prob, threshold=watch_threshold)

        val_teacher_true, val_teacher_prob = collect_outputs(
            eval_model,
            val_loader,
            args.device,
            args.pin_memory,
            mode="teacher",
            aggregation=args.eval_aggregation,
            baseline_reference=args.baseline_reference,
        )
        if args.threshold_mode == "fixed":
            teacher_threshold = args.fixed_threshold
            val_teacher_metrics = evaluate_with_threshold(val_teacher_true, val_teacher_prob, threshold=teacher_threshold)
        else:
            teacher_threshold, val_teacher_metrics = select_threshold(val_teacher_true, val_teacher_prob, metric=threshold_metric)
        test_teacher_true, test_teacher_prob = collect_outputs(
            eval_model,
            test_loader,
            args.device,
            args.pin_memory,
            mode="teacher",
            aggregation=args.eval_aggregation,
            baseline_reference=args.baseline_reference,
        )
        test_teacher_metrics = evaluate_with_threshold(test_teacher_true, test_teacher_prob, threshold=teacher_threshold)

        print(
            f"epoch={epoch + 1} "
            f"train_loss={train_loss:.4f} "
            f"val_watch_balanced_acc={val_watch_metrics['balanced_acc']:.4f} "
            f"val_watch_auroc={val_watch_metrics['auroc']:.4f} "
            f"val_teacher_balanced_acc={val_teacher_metrics['balanced_acc']:.4f} "
            f"val_teacher_auroc={val_teacher_metrics['auroc']:.4f} "
            f"watch_threshold={watch_threshold:.4f} "
            f"teacher_threshold={teacher_threshold:.4f}"
        )

        history_rows.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_watch_acc": val_watch_metrics["acc"],
                "val_watch_balanced_acc": val_watch_metrics["balanced_acc"],
                "val_watch_f1": val_watch_metrics["f1"],
                "val_watch_auroc": val_watch_metrics["auroc"],
                "val_watch_threshold": val_watch_metrics["threshold"],
                "val_watch_positive_rate": val_watch_metrics["positive_rate"],
                "val_teacher_acc": val_teacher_metrics["acc"],
                "val_teacher_balanced_acc": val_teacher_metrics["balanced_acc"],
                "val_teacher_f1": val_teacher_metrics["f1"],
                "val_teacher_auroc": val_teacher_metrics["auroc"],
                "val_teacher_threshold": val_teacher_metrics["threshold"],
                "val_teacher_positive_rate": val_teacher_metrics["positive_rate"],
                "threshold_mode": args.threshold_mode,
                "fixed_threshold": args.fixed_threshold,
                "threshold_metric": threshold_metric,
                "baseline_relative_active_subject_fraction": float(np.mean(epoch_brel_subject_fracs)) if epoch_brel_subject_fracs else 0.0,
                "baseline_relative_active_sample_fraction": float(np.mean(epoch_brel_sample_fracs)) if epoch_brel_sample_fracs else 0.0,
                "baseline_relative_active_batch_fraction": float(epoch_brel_active_batches / max(len(train_loader), 1)),
                "watch_contrastive_weight": args.watch_contrastive_weight,
                "contrastive_temperature": args.contrastive_temperature,
                "embedding_align_weight": effective_embedding_align_weight,
                "margin_match_weight": effective_margin_match_weight,
                "normalized_margin_align_weight": effective_normalized_margin_align_weight,
                "ranking_distill_weight": effective_ranking_distill_weight,
                "distribution_weight": effective_distribution_weight,
                "subject_center_stability_weight": effective_subject_center_stability_weight,
                "validation_threshold_stability_weight": effective_validation_threshold_stability_weight,
                "session_consistency_weight": effective_session_consistency_weight,
                "reliability_distill_weight": args.reliability_distill_weight,
                "clean_reliability_objective": int(clean_reliability_objective),
                "cross_confidence_distill": int(cross_confidence_enabled),
                "cross_confidence_min_weight": args.cross_confidence_min_weight,
                "cross_confidence_targets": ",".join(sorted(cross_confidence_targets)) if cross_confidence_enabled else "",
                "cross_confidence_trust_mean": float(np.mean(epoch_cross_confidence_trust_means)) if epoch_cross_confidence_trust_means else 0.0,
                "cross_confidence_trust_high_frac": float(np.mean(epoch_cross_confidence_trust_high_fracs)) if epoch_cross_confidence_trust_high_fracs else 0.0,
                "kd_gate_mode": args.kd_gate_mode,
                "kd_gate_min_weight": args.kd_gate_min_weight,
                "kd_gate_trust_mean": float(np.mean(epoch_kd_gate_trust_means)) if epoch_kd_gate_trust_means else 0.0,
                "kd_gate_trust_high_frac": float(np.mean(epoch_kd_gate_trust_high_fracs)) if epoch_kd_gate_trust_high_fracs else 0.0,
                "student_gated_correction": int(args.student_gated_correction),
                "correction_cls_weight": args.correction_cls_weight,
                "correction_nondegradation_weight": args.correction_nondegradation_weight,
                "correction_align_weight": args.correction_align_weight,
                "correction_base_anchor_weight": args.correction_base_anchor_weight,
                "correction_margin": args.correction_margin,
                "correction_scale_init": args.correction_scale_init,
                "correction_alpha_init_bias": args.correction_alpha_init_bias,
                "correction_alpha_max": args.correction_alpha_max,
                "alpha_helpfulness_weight": args.alpha_helpfulness_weight,
                "alpha_help_margin": args.alpha_help_margin,
                "alpha_sparsity_weight": args.alpha_sparsity_weight,
                "alpha_sparsity_loss": float(np.mean(epoch_alpha_sparsity_losses)) if epoch_alpha_sparsity_losses else 0.0,
                "alpha_mean": float(np.mean(epoch_alpha_means)) if epoch_alpha_means else 0.0,
                "alpha_helpful_frac": float(np.mean(epoch_alpha_helpful_fracs)) if epoch_alpha_helpful_fracs else 0.0,
                "elastic_residual_weight": args.elastic_residual_weight,
                "elastic_alpha_target_weight": args.elastic_alpha_target_weight,
                "elastic_reliability_temp": args.elastic_reliability_temp,
                "elastic_uncertainty_temp": args.elastic_uncertainty_temp,
                "elastic_label_margin": args.elastic_label_margin,
                "elastic_residual_loss": float(np.mean(epoch_elastic_residual_losses)) if epoch_elastic_residual_losses else 0.0,
                "elastic_reliability_mean": float(np.mean(epoch_elastic_reliability_means)) if epoch_elastic_reliability_means else 0.0,
                "elastic_target_delta_mean": float(np.mean(epoch_elastic_target_delta_means)) if epoch_elastic_target_delta_means else 0.0,
                "elastic_alpha_loss": float(np.mean(epoch_elastic_alpha_losses)) if epoch_elastic_alpha_losses else 0.0,
                "elastic_alpha_target_mean": float(np.mean(epoch_elastic_alpha_target_means)) if epoch_elastic_alpha_target_means else 0.0,
                "reliability_trust_mean": float(np.mean(epoch_reliability_trust_means)) if epoch_reliability_trust_means else 0.0,
                "reliability_trust_high_frac": float(np.mean(epoch_reliability_trust_high_fracs)) if epoch_reliability_trust_high_fracs else 0.0,
                "reliability_kd_loss": float(np.mean(epoch_reliability_kd_losses)) if epoch_reliability_kd_losses else 0.0,
                "reliability_cal_loss": float(np.mean(epoch_reliability_cal_losses)) if epoch_reliability_cal_losses else 0.0,
                "distill_gating": args.distill_gating,
                "distill_disagreement_weight": args.distill_disagreement_weight,
                "teacher_confidence_threshold": args.teacher_confidence_threshold,
                "teacher_confidence_temperature": args.teacher_confidence_temperature,
                "min_distill_weight": args.min_distill_weight,
                "distill_gate_mean": float(np.mean(epoch_distill_gate_means)) if epoch_distill_gate_means else 1.0,
                "test_watch_acc": test_watch_metrics["acc"],
                "test_watch_balanced_acc": test_watch_metrics["balanced_acc"],
                "test_watch_f1": test_watch_metrics["f1"],
                "test_watch_auroc": test_watch_metrics["auroc"],
                "test_watch_positive_rate": test_watch_metrics["positive_rate"],
                "test_teacher_acc": test_teacher_metrics["acc"],
                "test_teacher_balanced_acc": test_teacher_metrics["balanced_acc"],
                "test_teacher_f1": test_teacher_metrics["f1"],
                "test_teacher_auroc": test_teacher_metrics["auroc"],
                "test_teacher_positive_rate": test_teacher_metrics["positive_rate"],
            }
        )

        selection_metrics = val_watch_metrics if args.selection_target == "watch" else val_teacher_metrics
        score = selection_metrics[args.monitor]

        if args.selection_mode == "fixed_epoch" and epoch + 1 == args.selection_epoch:
            source_model = eval_model
            best_state = {key: value.detach().cpu() for key, value in source_model.state_dict().items()}
            best_watch_threshold = watch_threshold
            best_teacher_threshold = teacher_threshold
            best_watch_metrics = test_watch_metrics
            best_teacher_metrics = test_teacher_metrics
            print(
                f"selected fixed epoch {epoch + 1} "
                f"| watch_test_balanced_acc={best_watch_metrics['balanced_acc']:.4f} "
                f"teacher_test_balanced_acc={best_teacher_metrics['balanced_acc']:.4f}"
            )

        if args.selection_mode == "early_stop":
            improved = stopper.step(score, epoch + 1)
            if improved:
                source_model = eval_model
                best_state = {key: value.detach().cpu() for key, value in source_model.state_dict().items()}
                best_watch_threshold = watch_threshold
                best_teacher_threshold = teacher_threshold
                best_watch_metrics = test_watch_metrics
                best_teacher_metrics = test_teacher_metrics
                print(
                    f"new best {args.selection_target}_{args.monitor}={score:.4f} at epoch {epoch + 1} "
                    f"| watch_threshold={best_watch_threshold:.4f} "
                    f"teacher_threshold={best_teacher_threshold:.4f} "
                    f"| watch_test_balanced_acc={best_watch_metrics['balanced_acc']:.4f} "
                    f"teacher_test_balanced_acc={best_teacher_metrics['balanced_acc']:.4f}"
                )

            if stopper.should_stop():
                print(
                    f"early stopping triggered: no improvement in {args.early_stop_patience} epochs; "
                    f"best_{args.selection_target}_{args.monitor}={stopper.best_score:.4f} at epoch {stopper.best_epoch}"
                )
                break

    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state if best_state is not None else model.state_dict(), args.save_path)
    if args.metrics_path is not None:
        args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(history_rows).to_csv(args.metrics_path, index=False)
        print(f"Saved epoch metrics to {args.metrics_path}")
    if args.selection_mode == "early_stop":
        print(f"best_{args.selection_target}_{args.monitor}={stopper.best_score:.4f} at epoch {stopper.best_epoch}")
    else:
        print(f"selected_epoch={args.selection_epoch}")
    if best_watch_metrics is not None:
        print(
            f"best_watch_threshold={best_watch_threshold:.4f} "
            f"best_watch_test_acc={best_watch_metrics['acc']:.4f} "
            f"best_watch_test_balanced_acc={best_watch_metrics['balanced_acc']:.4f} "
            f"best_watch_test_f1={best_watch_metrics['f1']:.4f} "
            f"best_watch_test_auroc={best_watch_metrics['auroc']:.4f} "
            f"best_watch_test_positive_rate={best_watch_metrics['positive_rate']:.4f}"
        )
    if best_teacher_metrics is not None:
        print(
            f"best_teacher_threshold={best_teacher_threshold:.4f} "
            f"best_teacher_test_acc={best_teacher_metrics['acc']:.4f} "
            f"best_teacher_test_balanced_acc={best_teacher_metrics['balanced_acc']:.4f} "
            f"best_teacher_test_f1={best_teacher_metrics['f1']:.4f} "
            f"best_teacher_test_auroc={best_teacher_metrics['auroc']:.4f} "
            f"best_teacher_test_positive_rate={best_teacher_metrics['positive_rate']:.4f}"
        )
    print(f"Saved WESAD privileged model checkpoint to {args.save_path}")


if __name__ == "__main__":
    main()
