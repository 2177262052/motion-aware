from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .early_stopping import EarlyStopping
from .galaxy_dataset import GalaxyPrivilegedWindowDataset
from .galaxy_models_adaptive_correction import AdaptiveCorrectionGalaxyTeacherNet
from .losses import baseline_relative_margin_loss
from .metrics import classification_metrics
from .reliability import cross_calibrated_trust, reliability_bce_loss, true_class_confidence, trust_weighted_kl_loss
from .samplers import SubjectAwareBatchSampler


DEFAULT_CALM_SESSIONS = [
    "baseline",
    "meditation-1",
    "meditation-2",
    "rest-1",
    "rest-2",
    "rest-3",
    "rest-4",
    "rest-5",
]

DEFAULT_STRESS_SESSIONS = ["tsst-prep"]
PHENOTYPE_CLASS_NAMES = ["canonical", "flat", "atypical"]


def build_loader(
    dataset: GalaxyPrivilegedWindowDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    batch_sampler: SubjectAwareBatchSampler | None = None,
) -> DataLoader:
    kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if batch_sampler is not None:
        kwargs["batch_sampler"] = batch_sampler
    else:
        kwargs["batch_size"] = batch_size
        kwargs["shuffle"] = shuffle
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **kwargs)


def maybe_parse_sessions(values: Sequence[str] | None, fallback: Iterable[str]) -> list[str]:
    if values is None or len(values) == 0:
        return list(fallback)
    return [str(item) for item in values]


def set_random_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters_with_prefixes(model: torch.nn.Module, prefixes: Sequence[str]) -> int:
    return sum(
        param.numel()
        for name, param in model.named_parameters()
        if any(name.startswith(prefix) for prefix in prefixes)
    )


def collapse_phenotype_label(value: object) -> str | None:
    if pd.isna(value):
        return None
    normalized = str(value).strip().lower()
    if normalized in {"canonical", "canonical_responder"}:
        return "canonical"
    if normalized in {"flat", "flat_responder"}:
        return "flat"
    if normalized in {"mixed", "mixed_responder", "inverse", "inverse_responder", "atypical", "atypical_responder"}:
        return "atypical"
    return None


def load_subject_phenotypes(phenotype_csv: Path, phenotype_column: str) -> tuple[dict[str, int], list[str]]:
    df = pd.read_csv(phenotype_csv)
    required_cols = {"subject_id", phenotype_column}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Phenotype CSV missing required columns: {sorted(missing)}")

    subject_to_index: dict[str, int] = {}
    label_to_index = {name: idx for idx, name in enumerate(PHENOTYPE_CLASS_NAMES)}
    for row in df.itertuples(index=False):
        subject_id = str(getattr(row, "subject_id"))
        phenotype_value = getattr(row, phenotype_column)
        collapsed = collapse_phenotype_label(phenotype_value)
        if collapsed is None:
            continue
        subject_to_index[subject_id] = label_to_index[collapsed]
    return subject_to_index, PHENOTYPE_CLASS_NAMES.copy()


def phenotype_supervision_loss(
    phenotype_logits: torch.Tensor,
    subject_ids: Sequence[str],
    phenotype_lookup: dict[str, int],
    quality: torch.Tensor,
) -> torch.Tensor:
    valid_rows: list[int] = []
    targets: list[int] = []
    for batch_idx, subject_id in enumerate(subject_ids):
        target = phenotype_lookup.get(str(subject_id))
        if target is None:
            continue
        valid_rows.append(batch_idx)
        targets.append(target)

    if not valid_rows:
        return phenotype_logits.new_tensor(0.0)

    index_tensor = torch.tensor(valid_rows, device=phenotype_logits.device, dtype=torch.long)
    target_tensor = torch.tensor(targets, device=phenotype_logits.device, dtype=torch.long)
    selected_logits = phenotype_logits.index_select(0, index_tensor)
    ce = F.cross_entropy(selected_logits, target_tensor, reduction="none")
    selected_quality = quality.index_select(0, index_tensor).squeeze(1).clamp(0.0, 1.0)
    weights = 0.35 + 0.65 * selected_quality
    return (ce * weights).sum() / weights.sum().clamp(min=1e-6)


def quality_aware_focal_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    quality: torch.Tensor,
    class_weights: torch.Tensor,
    gamma: float,
    label_smoothing: float,
) -> torch.Tensor:
    focal = focal_loss_values(
        logits,
        labels,
        class_weights=class_weights,
        gamma=gamma,
        label_smoothing=label_smoothing,
    )
    return quality_weighted_mean(focal, quality)


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


def prototype_alignment_loss(
    watch_proj: torch.Tensor,
    e4_proj: torch.Tensor,
    labels: torch.Tensor,
    quality: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    watch_proj = F.normalize(watch_proj, dim=1)
    e4_proj = F.normalize(e4_proj, dim=1)
    sample_weights = (0.35 + 0.65 * quality.squeeze(1).clamp(0.0, 1.0)).detach()

    classes = torch.unique(labels)
    watch_prototypes = []
    e4_prototypes = []
    prototype_labels = []

    for cls in classes:
        mask = labels == cls
        if not torch.any(mask):
            continue
        cls_weights = sample_weights[mask]
        watch_proto = (watch_proj[mask] * cls_weights.unsqueeze(1)).sum(dim=0) / cls_weights.sum().clamp(min=1e-6)
        e4_proto = (e4_proj[mask] * cls_weights.unsqueeze(1)).sum(dim=0) / cls_weights.sum().clamp(min=1e-6)
        watch_prototypes.append(F.normalize(watch_proto.unsqueeze(0), dim=1))
        e4_prototypes.append(F.normalize(e4_proto.unsqueeze(0), dim=1))
        prototype_labels.append(int(cls.item()))

    if not watch_prototypes:
        return watch_proj.new_tensor(0.0)

    watch_proto_bank = torch.cat(watch_prototypes, dim=0)
    e4_proto_bank = torch.cat(e4_prototypes, dim=0)

    proto_match = 1.0 - F.cosine_similarity(watch_proto_bank, e4_proto_bank, dim=1)
    proto_match_loss = proto_match.mean()

    if len(prototype_labels) == 1:
        return proto_match_loss

    label_to_index = {label: idx for idx, label in enumerate(prototype_labels)}
    target_indices = torch.tensor([label_to_index[int(label.item())] for label in labels], device=labels.device, dtype=torch.long)

    watch_logits = torch.matmul(watch_proj, e4_proto_bank.T) / temperature
    e4_logits = torch.matmul(e4_proj, watch_proto_bank.T) / temperature
    watch_ce = F.cross_entropy(watch_logits, target_indices, reduction="none")
    e4_ce = F.cross_entropy(e4_logits, target_indices, reduction="none")
    proto_ce_loss = ((watch_ce + e4_ce) * 0.5 * sample_weights).sum() / sample_weights.sum().clamp(min=1e-6)
    return proto_match_loss + proto_ce_loss


def distillation_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    quality: torch.Tensor,
    temperature: float,
    detach_teacher: bool = False,
) -> torch.Tensor:
    student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
    teacher_source = teacher_logits.detach() if detach_teacher else teacher_logits
    teacher_probs = F.softmax(teacher_source / temperature, dim=1)
    kl = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=1)
    weights = 0.35 + 0.65 * quality.squeeze(1).clamp(0.0, 1.0)
    return (kl * weights).sum() / weights.sum().clamp(min=1e-6) * (temperature ** 2)


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


def ranking_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    quality: torch.Tensor,
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
    pair_weights = ((sample_weights.unsqueeze(1) + sample_weights.unsqueeze(0)) * 0.5)[informative_mask]
    total_weights = pair_weights * pair_teacher_strength
    total_weights = total_weights / total_weights.mean().clamp(min=1e-6)

    loss = F.softplus(-pair_sign * pair_student_diff)
    return (loss * total_weights).mean()


def confidence_weighted_ranking_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    trust: torch.Tensor,
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
    sample_weights = trust.detach().float().clamp(0.0, 1.0)
    pair_weights = torch.minimum(sample_weights.unsqueeze(1), sample_weights.unsqueeze(0))[informative_mask]
    total_weights = pair_weights * pair_teacher_strength
    if total_weights.sum() <= 1e-6:
        return student_logits.new_tensor(0.0)

    loss = F.softplus(-pair_sign * pair_student_diff)
    return (loss * total_weights).sum() / total_weights.sum().clamp(min=1e-6)


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
    subject_ids: Sequence[str],
    sessions: Sequence[str],
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


def masked_rhythm_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    scales = pred.new_tensor([200.0, 1500.0, 300.0]).unsqueeze(0)
    pred_scaled = pred
    target_scaled = target / scales
    loss = F.smooth_l1_loss(pred_scaled, target_scaled, reduction="none")
    weighted = loss * mask
    denom = mask.sum().clamp(min=1.0)
    return weighted.sum() / denom


def scheduled_weight(base_weight: float, epoch: int, total_epochs: int, mode: str, floor: float) -> float:
    if base_weight <= 0:
        return 0.0
    if mode == "constant":
        return base_weight
    progress = (epoch - 1) / max(total_epochs - 1, 1)
    if mode == "linear":
        scale = 1.0 - progress
    elif mode == "cosine":
        scale = 0.5 * (1.0 + math.cos(math.pi * progress))
    else:
        raise ValueError(f"Unsupported schedule mode: {mode}")
    scale = max(scale, floor)
    return base_weight * scale


def apply_modality_dropout(signal: torch.Tensor, dropout_prob: float) -> tuple[torch.Tensor, float]:
    if dropout_prob <= 0.0:
        return signal, 0.0
    if dropout_prob >= 1.0:
        return torch.zeros_like(signal), 1.0

    batch_size = signal.size(0)
    keep_mask = torch.rand(batch_size, 1, 1, device=signal.device) >= dropout_prob
    dropped_signal = signal * keep_mask.to(signal.dtype)
    drop_rate = 1.0 - float(keep_mask.float().mean().item())
    return dropped_signal, drop_rate


@torch.no_grad()
def update_ema_model(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    ema_params = dict(ema_model.named_parameters())
    model_params = dict(model.named_parameters())
    for name, ema_param in ema_params.items():
        ema_param.mul_(decay).add_(model_params[name], alpha=1.0 - decay)

    ema_buffers = dict(ema_model.named_buffers())
    model_buffers = dict(model.named_buffers())
    for name, ema_buffer in ema_buffers.items():
        ema_buffer.copy_(model_buffers[name])


def collect_outputs(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    pin_memory: bool,
    mode: str = "watch",
    aggregation: str = "window",
    baseline_reference: bool = False,
) -> tuple[list[int], list[float]]:
    model.eval()
    y_true = []
    y_prob = []
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
                logits = model.forward_watch(watch_signal, wavelet, quality, **{
                    "baseline_signal": baseline_kwargs.get("baseline_watch_signal"),
                    "baseline_wavelet_features": baseline_kwargs.get("baseline_wavelet_features"),
                    "baseline_quality": baseline_kwargs.get("baseline_quality"),
                    "return_aux": False,
                })["logits"]
            elif mode == "teacher":
                e4_signal = batch["e4_signal"].to(device, non_blocking=pin_memory)
                logits = model(
                    watch_signal,
                    wavelet,
                    quality,
                    e4_signal=e4_signal,
                    return_aux=False,
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


def aggregate_predictions(
    y_true: list[int],
    y_prob: list[float],
    subject_ids: list[str],
    sessions: list[str],
    aggregation: str,
) -> tuple[list[int], list[float]]:
    if aggregation == "window":
        return y_true, y_prob
    if aggregation != "session_mean":
        raise ValueError(f"Unsupported eval aggregation: {aggregation}")

    frame = pd.DataFrame(
        {
            "subject_id": subject_ids,
            "session": sessions,
            "label": y_true,
            "prob": y_prob,
        }
    )
    grouped = (
        frame.groupby(["subject_id", "session"], as_index=False)
        .agg(label=("label", "first"), prob=("prob", "mean"))
        .reset_index(drop=True)
    )
    return grouped["label"].astype(int).tolist(), grouped["prob"].astype(float).tolist()


def evaluate_with_threshold(y_true: list[int], y_prob: list[float], threshold: float) -> dict[str, float]:
    y_pred = [1 if prob >= threshold else 0 for prob in y_prob]
    metrics = classification_metrics(y_true, y_pred, y_prob)
    metrics["threshold"] = threshold
    metrics["positive_rate"] = float(np.mean(y_pred)) if len(y_pred) else 0.0
    return metrics


def select_threshold(y_true: list[int], y_prob: list[float], metric: str = "balanced_acc") -> tuple[float, dict[str, float]]:
    if len(set(y_true)) < 2:
        return 0.5, evaluate_with_threshold(y_true, y_prob, threshold=0.5)

    candidates = sorted(set([0.0, 1.0] + [round(prob, 6) for prob in y_prob]))
    best_threshold = 0.5
    best_metrics = evaluate_with_threshold(y_true, y_prob, threshold=0.5)
    best_score = best_metrics[metric]

    for threshold in candidates:
        metrics = evaluate_with_threshold(y_true, y_prob, threshold=threshold)
        score = metrics[metric]
        if score > best_score + 1e-12:
            best_score = score
            best_threshold = threshold
            best_metrics = metrics
    return best_threshold, best_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a privileged GalaxyPPG teacher with watch-only inference.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--save-path", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--focal-gamma", type=float, default=1.5)
    parser.add_argument("--watch-cls-weight", type=float, default=1.0)
    parser.add_argument("--align-weight", type=float, default=0.0)
    parser.add_argument("--rhythm-weight", type=float, default=0.15)
    parser.add_argument("--e4-cls-weight", type=float, default=0.05)
    parser.add_argument("--teacher-cls-weight", type=float, default=0.80)
    parser.add_argument("--teacher-fused-cls-weight", type=float, default=0.00)
    parser.add_argument("--distill-weight", type=float, default=0.10)
    parser.add_argument("--ranking-distill-weight", type=float, default=0.10)
    parser.add_argument("--distribution-weight", type=float, default=0.05)
    parser.add_argument("--session-consistency-weight", type=float, default=0.05)
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
    parser.add_argument(
        "--detach-standard-kd-teacher",
        action="store_true",
        help="Detach teacher soft targets for ungated standard KD. Gated KD losses already detach teacher logits.",
    )
    parser.add_argument("--reliability-distill-weight", type=float, default=0.0)
    parser.add_argument("--clean-reliability-objective", action="store_true")
    parser.add_argument("--wavelet-weight", type=float, default=0.05)
    parser.add_argument("--baseline-relative-weight", type=float, default=0.0)
    parser.add_argument("--baseline-relative-margin", type=float, default=0.2)
    parser.add_argument("--student-gated-correction", action="store_true")
    parser.add_argument("--correction-cls-weight", type=float, default=0.0)
    parser.add_argument("--correction-nondegradation-weight", type=float, default=0.0)
    parser.add_argument("--correction-align-weight", type=float, default=0.0)
    parser.add_argument("--correction-base-anchor-weight", type=float, default=0.0)
    parser.add_argument("--correction-margin", type=float, default=0.0)
    parser.add_argument("--correction-scale-init", type=float, default=0.05)
    parser.add_argument("--correction-alpha-init-bias", type=float, default=-3.0)
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
    parser.add_argument("--phenotype-csv", type=Path, default=None)
    parser.add_argument("--phenotype-column", type=str, default="phenotype")
    parser.add_argument("--phenotype-weight", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--disable-ema-eval", action="store_true")
    parser.add_argument("--e4-modality-dropout", type=float, default=0.0)
    parser.add_argument("--priv-schedule", type=str, default="cosine", choices=["constant", "linear", "cosine"])
    parser.add_argument("--priv-floor", type=float, default=0.2)
    parser.add_argument("--teacher-warmup-epochs", type=int, default=0)
    parser.add_argument("--teacher-warmup-watch-weight", type=float, default=0.40)
    parser.add_argument("--teacher-warmup-distill-weight", type=float, default=0.0)
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
    parser.add_argument("--baseline-reference", action="store_true")
    parser.add_argument("--subject-aware-batching", action="store_true")
    parser.add_argument(
        "--watch-backbone",
        type=str,
        default="wavelet_guided",
        choices=["wavelet_guided", "resnet18_1d", "resnet34_1d", "resnet50_1d"],
    )
    parser.add_argument(
        "--watch-enhancement",
        type=str,
        default="none",
        choices=["none", "motion_disentangled", "acc_concat"],
    )
    parser.add_argument(
        "--watch-motion-mode",
        type=str,
        default="strong",
        choices=["strong", "residual", "scaled"],
    )
    args = parser.parse_args()

    set_random_seed(args.seed)
    calm_sessions = maybe_parse_sessions(args.calm_sessions, DEFAULT_CALM_SESSIONS)
    stress_sessions = maybe_parse_sessions(args.stress_sessions, DEFAULT_STRESS_SESSIONS)
    include_sessions = calm_sessions + stress_sessions

    train_ds = GalaxyPrivilegedWindowDataset(
        manifest_csv=args.manifest,
        split="train",
        dataset_root=args.dataset_root,
        include_sessions=include_sessions,
        cache_tables=True,
        wavelet=args.wavelet,
        wavelet_level=args.wavelet_level,
        baseline_reference=args.baseline_reference,
    )
    val_ds = GalaxyPrivilegedWindowDataset(
        manifest_csv=args.manifest,
        split="val",
        dataset_root=args.dataset_root,
        include_sessions=include_sessions,
        cache_tables=True,
        wavelet=args.wavelet,
        wavelet_level=args.wavelet_level,
        baseline_reference=args.baseline_reference,
    )
    test_ds = GalaxyPrivilegedWindowDataset(
        manifest_csv=args.manifest,
        split="test",
        dataset_root=args.dataset_root,
        include_sessions=include_sessions,
        cache_tables=True,
        wavelet=args.wavelet,
        wavelet_level=args.wavelet_level,
        baseline_reference=args.baseline_reference,
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

    phenotype_lookup: dict[str, int] = {}
    phenotype_names: list[str] = []
    if args.phenotype_csv is not None:
        phenotype_lookup, phenotype_names = load_subject_phenotypes(args.phenotype_csv, args.phenotype_column)
    if args.phenotype_weight > 0 and not phenotype_lookup:
        raise ValueError("Phenotype-aware training requested but no usable subject phenotypes were loaded.")
    num_phenotypes = len(phenotype_names) if args.phenotype_weight > 0 and phenotype_names else 0
    reliability_enabled = args.reliability_distill_weight > 0.0
    model = AdaptiveCorrectionGalaxyTeacherNet(
        num_phenotypes=num_phenotypes,
        watch_backbone=args.watch_backbone,
        use_reliability_head=reliability_enabled,
        use_projection_heads=args.align_weight > 0.0,
        use_e4_classifier=args.e4_cls_weight > 0.0,
        use_rhythm_heads=args.rhythm_weight > 0.0,
        use_wavelet_head=args.wavelet_weight > 0.0,
        use_teacher_fused_classifier=args.teacher_fused_cls_weight > 0.0,
        use_student_gated_correction=args.student_gated_correction,
        correction_scale_init=args.correction_scale_init,
        correction_alpha_init_bias=args.correction_alpha_init_bias,
        correction_alpha_max=args.correction_alpha_max,
        correction_mode=args.correction_mode,
        watch_enhancement=args.watch_enhancement,
        watch_motion_mode=args.watch_motion_mode,
    ).to(args.device)
    use_ema_eval = not args.disable_ema_eval
    ema_model = copy.deepcopy(model).to(args.device) if use_ema_eval else None
    if ema_model is not None:
        ema_model.eval()
        for param in ema_model.parameters():
            param.requires_grad_(False)
    total_params = sum(p.numel() for p in model.parameters())
    watch_only_params = count_parameters_with_prefixes(model, ("watch_encoder", "watch_classifier"))
    watch_reliability_params = count_parameters_with_prefixes(
        model,
        ("watch_encoder", "watch_classifier", "reliability_head"),
    )
    watch_aux_params = count_parameters_with_prefixes(
        model,
        ("reliability_head", "watch_projector", "rhythm_head", "wavelet_predictor"),
    )
    correction_params = count_parameters_with_prefixes(
        model,
        (
            "deploy_correction",
            "deploy_correction_gate",
            "privileged_correction",
            "privileged_correction_gate",
            "correction_norm",
            "correction_scale",
        ),
    )
    print(f"teacher_params={total_params}")
    print(f"training_params={total_params}")
    print(f"watch_inference_params={watch_only_params}")
    if reliability_enabled:
        print(f"watch_reliability_params={watch_reliability_params}")
    print(f"watch_aux_head_params={watch_aux_params}")
    if args.student_gated_correction:
        print(f"student_gated_correction_params={correction_params}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    class_counts = train_ds.manifest["label"].value_counts().sort_index()
    neg = float(class_counts.get(0, 1.0))
    pos = float(class_counts.get(1, 1.0))
    class_weights = torch.tensor([1.0, neg / max(pos, 1.0)], device=args.device, dtype=torch.float32)

    stopper = EarlyStopping(patience=args.early_stop_patience, mode="max", min_delta=args.min_delta)
    best_state = None
    best_test_metrics = None
    best_teacher_test_metrics = None
    history_rows = []
    best_watch_epoch = None
    best_watch_score = float("-inf")
    best_teacher_epoch = None
    best_teacher_score = float("-inf")
    clean_reliability_objective = reliability_enabled and args.clean_reliability_objective
    cross_confidence_targets = set(args.cross_confidence_targets)
    cross_confidence_enabled = args.cross_confidence_distill and bool(cross_confidence_targets)
    if cross_confidence_enabled and args.kd_gate_mode != "none":
        raise ValueError("--kd-gate-mode is mutually exclusive with --cross-confidence-distill.")

    print(f"train windows={len(train_ds)} val windows={len(val_ds)} test windows={len(test_ds)}")
    if not has_val_split:
        print("selection_warning=val split missing; using test split for threshold/model selection")
    print(f"watch_backbone={args.watch_backbone}")
    print(f"calm_sessions={calm_sessions}")
    print(f"stress_sessions={stress_sessions}")
    print(
        "loss_weights="
        f"watch_cls:{args.watch_cls_weight:.2f} teacher_cls:{args.teacher_cls_weight:.2f} distill:{args.distill_weight:.2f} "
        f"rank:{args.ranking_distill_weight:.2f} distr:{args.distribution_weight:.2f} "
        f"session:{args.session_consistency_weight:.2f} teacher_fused:{args.teacher_fused_cls_weight:.2f} "
        f"rel:{args.reliability_distill_weight:.2f} "
        f"align:{args.align_weight:.2f} "
        f"baseline_relative:{args.baseline_relative_weight:.2f} "
        f"corr_cls:{args.correction_cls_weight:.2f} "
        f"corr_nd:{args.correction_nondegradation_weight:.2f} "
        f"corr_align:{args.correction_align_weight:.2f} "
        f"corr_anchor:{args.correction_base_anchor_weight:.2f} "
        f"alpha_help:{args.alpha_helpfulness_weight:.2f} "
        f"alpha_sparse:{args.alpha_sparsity_weight:.2f} "
        f"elastic_resid:{args.elastic_residual_weight:.2f} "
        f"elastic_alpha:{args.elastic_alpha_target_weight:.2f} "
        f"phenotype:{args.phenotype_weight:.2f} "
        f"e4_cls:{args.e4_cls_weight:.2f} rhythm:{args.rhythm_weight:.2f} "
        f"wavelet:{args.wavelet_weight:.2f}"
    )
    print(
        f"phenotype_aware={'on' if num_phenotypes > 0 else 'off'} "
        f"phenotype_classes={phenotype_names if phenotype_names else []}"
    )
    print(f"e4_modality_dropout={args.e4_modality_dropout:.2f}")
    print(
        f"reliability_distill={'on' if reliability_enabled else 'off'} "
        f"clean_reliability_objective={'on' if clean_reliability_objective else 'off'}"
    )
    print(
        f"cross_confidence_distill={'on' if cross_confidence_enabled else 'off'} "
        f"targets={sorted(cross_confidence_targets) if cross_confidence_enabled else []} "
        f"min_weight={args.cross_confidence_min_weight:.2f}"
    )
    print(
        f"kd_gate_mode={args.kd_gate_mode} "
        f"kd_gate_min_weight={args.kd_gate_min_weight:.2f} "
        f"detach_standard_kd_teacher={'on' if args.detach_standard_kd_teacher else 'off'}"
    )
    print(
        f"student_gated_correction={'on' if args.student_gated_correction else 'off'} "
        f"watch_enhancement={args.watch_enhancement} "
        f"scale_init={args.correction_scale_init:.3f} "
        f"margin={args.correction_margin:.3f} "
        f"alpha_init_bias={args.correction_alpha_init_bias:.3f} "
        f"alpha_max={args.correction_alpha_max:.3f} "
        f"correction_mode={args.correction_mode} "
        f"alpha_help_margin={args.alpha_help_margin:.3f} "
        f"elastic_tau={args.elastic_reliability_temp:.3f} "
        f"elastic_uncertainty_tau={args.elastic_uncertainty_temp:.3f} "
        f"elastic_label_margin={args.elastic_label_margin:.3f}"
    )
    print(
        f"teacher_warmup_epochs={args.teacher_warmup_epochs} "
        f"warmup_watch_weight={args.teacher_warmup_watch_weight:.2f} "
        f"warmup_distill_weight={args.teacher_warmup_distill_weight:.2f}"
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
    threshold_metric = args.monitor if args.threshold_metric == "monitor" else args.threshold_metric
    print(f"threshold_metric={threshold_metric}")
    print(f"ema_eval={'on' if use_ema_eval else 'off'} ema_decay={args.ema_decay:.4f} seed={args.seed}")
    print(f"priv_schedule={args.priv_schedule} priv_floor={args.priv_floor:.2f}")
    print("priv_mode=fused_teacher_distill")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        epoch_brel_subject_fracs: list[float] = []
        epoch_brel_sample_fracs: list[float] = []
        epoch_brel_active_batches = 0
        epoch_reliability_trust_means: list[float] = []
        epoch_reliability_trust_high_fracs: list[float] = []
        epoch_reliability_kd_losses: list[float] = []
        epoch_reliability_cal_losses: list[float] = []
        epoch_cross_confidence_trust_means: list[float] = []
        epoch_cross_confidence_trust_high_fracs: list[float] = []
        epoch_kd_gate_trust_means: list[float] = []
        epoch_kd_gate_trust_high_fracs: list[float] = []
        epoch_alpha_means: list[float] = []
        epoch_alpha_helpful_fracs: list[float] = []
        epoch_alpha_sparsity_losses: list[float] = []
        epoch_elastic_residual_losses: list[float] = []
        epoch_elastic_reliability_means: list[float] = []
        epoch_elastic_target_delta_means: list[float] = []
        epoch_elastic_alpha_losses: list[float] = []
        epoch_elastic_alpha_target_means: list[float] = []
        e4_cls_weight = scheduled_weight(args.e4_cls_weight, epoch + 1, args.epochs, args.priv_schedule, args.priv_floor)
        align_weight = scheduled_weight(args.align_weight, epoch + 1, args.epochs, args.priv_schedule, args.priv_floor)
        rhythm_weight = scheduled_weight(args.rhythm_weight, epoch + 1, args.epochs, args.priv_schedule, args.priv_floor)
        teacher_cls_weight = scheduled_weight(args.teacher_cls_weight, epoch + 1, args.epochs, args.priv_schedule, args.priv_floor)
        teacher_fused_cls_weight = scheduled_weight(args.teacher_fused_cls_weight, epoch + 1, args.epochs, args.priv_schedule, args.priv_floor)
        phenotype_weight = scheduled_weight(args.phenotype_weight, epoch + 1, args.epochs, args.priv_schedule, args.priv_floor)
        distill_weight = scheduled_weight(args.distill_weight, epoch + 1, args.epochs, args.priv_schedule, args.priv_floor)
        ranking_distill_weight = scheduled_weight(args.ranking_distill_weight, epoch + 1, args.epochs, args.priv_schedule, args.priv_floor)
        distribution_weight = scheduled_weight(args.distribution_weight, epoch + 1, args.epochs, args.priv_schedule, args.priv_floor)
        session_consistency_weight = scheduled_weight(args.session_consistency_weight, epoch + 1, args.epochs, args.priv_schedule, args.priv_floor)
        wavelet_weight = scheduled_weight(args.wavelet_weight, epoch + 1, args.epochs, args.priv_schedule, args.priv_floor)
        baseline_relative_weight = scheduled_weight(args.baseline_relative_weight, epoch + 1, args.epochs, args.priv_schedule, args.priv_floor)
        correction_cls_weight = scheduled_weight(args.correction_cls_weight, epoch + 1, args.epochs, args.priv_schedule, args.priv_floor)
        correction_nondegradation_weight = scheduled_weight(
            args.correction_nondegradation_weight,
            epoch + 1,
            args.epochs,
            args.priv_schedule,
            args.priv_floor,
        )
        correction_align_weight = scheduled_weight(args.correction_align_weight, epoch + 1, args.epochs, args.priv_schedule, args.priv_floor)
        correction_base_anchor_weight = scheduled_weight(
            args.correction_base_anchor_weight,
            epoch + 1,
            args.epochs,
            args.priv_schedule,
            args.priv_floor,
        )
        watch_cls_weight = args.watch_cls_weight
        if epoch + 1 <= args.teacher_warmup_epochs:
            watch_cls_weight = args.teacher_warmup_watch_weight
            distill_weight = args.teacher_warmup_distill_weight
        if clean_reliability_objective:
            distill_weight = 0.0
            ranking_distill_weight = 0.0
            distribution_weight = 0.0
            session_consistency_weight = 0.0
        progress = tqdm(train_loader, desc=f"galaxy-priv epoch {epoch + 1}/{args.epochs}", leave=True)
        epoch_e4_drop_rates = []
        for batch in progress:
            watch_signal = batch["watch_signal"].to(args.device, non_blocking=args.pin_memory)
            e4_signal = batch["e4_signal"].to(args.device, non_blocking=args.pin_memory)
            wavelet = batch["wavelet_features"].to(args.device, non_blocking=args.pin_memory)
            quality = batch["watch_quality"].to(args.device, non_blocking=args.pin_memory)
            labels = batch["label"].to(args.device, non_blocking=args.pin_memory).long()
            polar_targets = batch["polar_targets"].to(args.device, non_blocking=args.pin_memory)
            polar_mask = batch["polar_target_mask"].to(args.device, non_blocking=args.pin_memory)
            baseline_kwargs = {}
            if args.baseline_reference:
                baseline_kwargs = {
                    "baseline_watch_signal": batch["baseline_watch_signal"].to(args.device, non_blocking=args.pin_memory),
                    "baseline_wavelet_features": batch["baseline_wavelet_features"].to(args.device, non_blocking=args.pin_memory),
                    "baseline_quality": batch["baseline_watch_quality"].to(args.device, non_blocking=args.pin_memory),
                }

            e4_signal, e4_drop_rate = apply_modality_dropout(e4_signal, args.e4_modality_dropout)
            epoch_e4_drop_rates.append(e4_drop_rate)

            out = model(watch_signal, wavelet, quality, e4_signal=e4_signal, **baseline_kwargs)
            cls_loss = quality_aware_focal_loss(
                out["logits"],
                labels,
                quality,
                class_weights=class_weights,
                gamma=args.focal_gamma,
                label_smoothing=args.label_smoothing,
            )
            zero_loss = out["logits"].new_tensor(0.0)
            e4_cls_loss = zero_loss
            if e4_cls_weight > 0 and "e4_logits" in out:
                e4_quality = torch.ones_like(quality)
                e4_cls_loss = quality_aware_focal_loss(
                    out["e4_logits"],
                    labels,
                    e4_quality,
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
            teacher_fused_cls_loss = zero_loss
            if teacher_fused_cls_weight > 0 and "teacher_fused_logits" in out:
                teacher_fused_cls_loss = quality_aware_focal_loss(
                    out["teacher_fused_logits"],
                    labels,
                    quality,
                    class_weights=class_weights,
                    gamma=args.focal_gamma,
                    label_smoothing=args.label_smoothing,
                )
            phenotype_loss = zero_loss
            if phenotype_weight > 0 and "phenotype_logits" in out:
                phenotype_loss = phenotype_supervision_loss(
                    out["phenotype_logits"],
                    batch["subject_id"],
                    phenotype_lookup,
                    quality,
                )
            base_anchor_loss = zero_loss
            if correction_base_anchor_weight > 0 and "base_logits" in out:
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
            elastic_residual_loss = zero_loss
            elastic_alpha_loss = zero_loss
            elastic_reliability_mean = zero_loss
            elastic_target_delta_mean = zero_loss
            elastic_alpha_target_mean = zero_loss
            alpha_mean = zero_loss
            alpha_helpful_frac = zero_loss
            if args.student_gated_correction and "privileged_correction_logits" in out:
                correction_cls_loss = quality_aware_focal_loss(
                    out["privileged_correction_logits"],
                    labels,
                    quality,
                    class_weights=class_weights,
                    gamma=args.focal_gamma,
                    label_smoothing=args.label_smoothing,
                )
                if correction_nondegradation_weight > 0 and "base_logits" in out:
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
                    correction_align_weight > 0
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
                    detach_teacher=args.detach_standard_kd_teacher,
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
            if cross_confidence_enabled and "ranking" in cross_confidence_targets and cross_confidence_trust is not None:
                ranking_loss = confidence_weighted_ranking_distillation_loss(
                    out["logits"],
                    out["teacher_logits"],
                    cross_confidence_trust,
                )
            else:
                ranking_loss = ranking_distillation_loss(
                    out["logits"],
                    out["teacher_logits"],
                    quality=quality,
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
            session_consistency_loss = session_consistency_distillation_loss(
                out["logits"],
                out["teacher_logits"],
                batch["subject_id"],
                batch["session"],
                quality=quality,
            )
            watch_baseline_relative_loss, br_stats = baseline_relative_margin_loss(
                out["logits"],
                labels,
                [str(item) for item in batch["subject_id"]],
                quality,
                margin=args.baseline_relative_margin,
                return_stats=True,
            )
            teacher_baseline_relative_loss = baseline_relative_margin_loss(
                out["teacher_logits"],
                labels,
                [str(item) for item in batch["subject_id"]],
                quality,
                margin=args.baseline_relative_margin,
            )
            baseline_relative_loss = 0.5 * (watch_baseline_relative_loss + teacher_baseline_relative_loss)
            epoch_brel_subject_fracs.append(float(br_stats["active_subject_fraction"]))
            epoch_brel_sample_fracs.append(float(br_stats["active_sample_fraction"]))
            if br_stats["active_subjects"] > 0:
                epoch_brel_active_batches += 1
            align_loss = zero_loss
            if align_weight > 0 and "watch_proj" in out and "e4_proj" in out:
                align_loss = prototype_alignment_loss(
                    out["watch_proj"],
                    out["e4_proj"],
                    labels,
                    quality,
                    temperature=args.temperature,
                )
            rhythm_loss = zero_loss
            if rhythm_weight > 0 and "rhythm_pred" in out and "teacher_rhythm_pred" in out:
                watch_rhythm_loss = masked_rhythm_loss(out["rhythm_pred"], polar_targets, polar_mask)
                teacher_rhythm_loss = masked_rhythm_loss(out["teacher_rhythm_pred"], polar_targets, polar_mask)
                rhythm_loss = 0.5 * (watch_rhythm_loss + teacher_rhythm_loss)
            wavelet_loss = zero_loss
            if wavelet_weight > 0 and "wavelet_pred" in out:
                wavelet_loss = F.smooth_l1_loss(out["wavelet_pred"], wavelet)
            loss = (
                watch_cls_weight * cls_loss
                + teacher_cls_weight * teacher_cls_loss
                + teacher_fused_cls_weight * teacher_fused_cls_loss
                + phenotype_weight * phenotype_loss
                + distill_weight * distill_loss
                + args.reliability_distill_weight * reliability_loss
                + ranking_distill_weight * ranking_loss
                + distribution_weight * distribution_loss
                + session_consistency_weight * session_consistency_loss
                + baseline_relative_weight * baseline_relative_loss
                + correction_base_anchor_weight * base_anchor_loss
                + correction_cls_weight * correction_cls_loss
                + correction_nondegradation_weight * correction_nondegradation_loss
                + correction_align_weight * correction_align_loss
                + args.alpha_helpfulness_weight * alpha_loss
                + args.alpha_sparsity_weight * alpha_sparsity_loss
                + args.elastic_residual_weight * elastic_residual_loss
                + args.elastic_alpha_target_weight * elastic_alpha_loss
                + e4_cls_weight * e4_cls_loss
                + align_weight * align_loss
                + rhythm_weight * rhythm_loss
                + wavelet_weight * wavelet_loss
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if ema_model is not None:
                update_ema_model(ema_model, model, args.ema_decay)
            total_loss += float(loss.item())
            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                cls=f"{cls_loss.item():.4f}",
                teacher=f"{teacher_cls_loss.item():.4f}",
                tfused=f"{teacher_fused_cls_loss.item():.4f}",
                ptype=f"{phenotype_loss.item():.4f}",
                kd=f"{distill_loss.item():.4f}",
                rel=f"{reliability_loss.item():.4f}",
                trust=f"{reliability_trust_mean.item():.3f}",
                xconf=f"{cross_confidence_trust_mean.item():.3f}",
                kdgate=f"{kd_gate_trust_mean.item():.3f}",
                rank=f"{ranking_loss.item():.4f}",
                distr=f"{distribution_loss.item():.4f}",
                sess=f"{session_consistency_loss.item():.4f}",
                brel=f"{baseline_relative_loss.item():.4f}",
                bsub=f"{int(br_stats['active_subjects'])}/{int(br_stats['available_subjects'])}",
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
                e4=f"{e4_cls_loss.item():.4f}",
                align=f"{align_loss.item():.4f}",
                rhythm=f"{rhythm_loss.item():.4f}",
                watch_w=f"{watch_cls_weight:.3f}",
                kd_w=f"{distill_weight:.3f}",
                e4_drop=f"{e4_drop_rate:.2f}",
            )

        scheduler.step()
        train_loss = total_loss / max(len(train_loader), 1)
        eval_model = ema_model if ema_model is not None else model
        val_true, val_prob = collect_outputs(
            eval_model,
            val_loader,
            args.device,
            args.pin_memory,
            mode="watch",
            aggregation=args.eval_aggregation,
            baseline_reference=args.baseline_reference,
        )
        watch_threshold, val_metrics = select_threshold(val_true, val_prob, metric=threshold_metric)
        val_teacher_true, val_teacher_prob = collect_outputs(
            eval_model,
            val_loader,
            args.device,
            args.pin_memory,
            mode="teacher",
            aggregation=args.eval_aggregation,
            baseline_reference=args.baseline_reference,
        )
        teacher_threshold, val_teacher_metrics = select_threshold(val_teacher_true, val_teacher_prob, metric=threshold_metric)
        test_true, test_prob = collect_outputs(
            eval_model,
            test_loader,
            args.device,
            args.pin_memory,
            mode="watch",
            aggregation=args.eval_aggregation,
            baseline_reference=args.baseline_reference,
        )
        test_metrics = evaluate_with_threshold(test_true, test_prob, threshold=watch_threshold)
        teacher_test_true, teacher_test_prob = collect_outputs(
            eval_model,
            test_loader,
            args.device,
            args.pin_memory,
            mode="teacher",
            aggregation=args.eval_aggregation,
            baseline_reference=args.baseline_reference,
        )
        teacher_test_metrics = evaluate_with_threshold(teacher_test_true, teacher_test_prob, threshold=teacher_threshold)
        watch_score = val_metrics[args.monitor]
        teacher_score = val_teacher_metrics[args.monitor]
        if watch_score > best_watch_score + 1e-12:
            best_watch_score = watch_score
            best_watch_epoch = epoch + 1
        if teacher_score > best_teacher_score + 1e-12:
            best_teacher_score = teacher_score
            best_teacher_epoch = epoch + 1
        print(
            f"epoch={epoch + 1} "
            f"train_loss={train_loss:.4f} "
            f"val_watch_balanced_acc={val_metrics['balanced_acc']:.4f} "
            f"val_watch_auroc={val_metrics['auroc']:.4f} "
            f"val_teacher_balanced_acc={val_teacher_metrics['balanced_acc']:.4f} "
            f"val_teacher_auroc={val_teacher_metrics['auroc']:.4f} "
            f"val_watch_threshold={watch_threshold:.4f} "
            f"val_teacher_threshold={teacher_threshold:.4f} "
            f"watch_test_balanced_acc={test_metrics['balanced_acc']:.4f} "
            f"watch_test_auroc={test_metrics['auroc']:.4f} "
            f"teacher_test_balanced_acc={teacher_test_metrics['balanced_acc']:.4f} "
            f"teacher_test_auroc={teacher_test_metrics['auroc']:.4f}"
        )

        history_rows.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "watch_cls_weight": watch_cls_weight,
                "teacher_cls_weight": teacher_cls_weight,
                "teacher_fused_cls_weight": teacher_fused_cls_weight,
                "phenotype_weight": phenotype_weight,
                "distill_weight": distill_weight,
                "reliability_distill_weight": args.reliability_distill_weight,
                "clean_reliability_objective": int(clean_reliability_objective),
                "cross_confidence_distill": int(cross_confidence_enabled),
                "cross_confidence_min_weight": args.cross_confidence_min_weight,
                "cross_confidence_targets": ",".join(sorted(cross_confidence_targets)) if cross_confidence_enabled else "",
                "cross_confidence_trust_mean": float(np.mean(epoch_cross_confidence_trust_means)) if epoch_cross_confidence_trust_means else 0.0,
                "cross_confidence_trust_high_frac": float(np.mean(epoch_cross_confidence_trust_high_fracs)) if epoch_cross_confidence_trust_high_fracs else 0.0,
                "kd_gate_mode": args.kd_gate_mode,
                "kd_gate_min_weight": args.kd_gate_min_weight,
                "detach_standard_kd_teacher": int(args.detach_standard_kd_teacher),
                "kd_gate_trust_mean": float(np.mean(epoch_kd_gate_trust_means)) if epoch_kd_gate_trust_means else 0.0,
                "kd_gate_trust_high_frac": float(np.mean(epoch_kd_gate_trust_high_fracs)) if epoch_kd_gate_trust_high_fracs else 0.0,
                "reliability_trust_mean": float(np.mean(epoch_reliability_trust_means)) if epoch_reliability_trust_means else 0.0,
                "reliability_trust_high_frac": float(np.mean(epoch_reliability_trust_high_fracs)) if epoch_reliability_trust_high_fracs else 0.0,
                "reliability_kd_loss": float(np.mean(epoch_reliability_kd_losses)) if epoch_reliability_kd_losses else 0.0,
                "reliability_cal_loss": float(np.mean(epoch_reliability_cal_losses)) if epoch_reliability_cal_losses else 0.0,
                "ranking_distill_weight": ranking_distill_weight,
                "distribution_weight": distribution_weight,
                "session_consistency_weight": session_consistency_weight,
                "baseline_relative_weight": baseline_relative_weight,
                "student_gated_correction": int(args.student_gated_correction),
                "correction_cls_weight": correction_cls_weight,
                "correction_nondegradation_weight": correction_nondegradation_weight,
                "correction_align_weight": correction_align_weight,
                "correction_base_anchor_weight": correction_base_anchor_weight,
                "correction_margin": args.correction_margin,
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
                "baseline_relative_active_subject_fraction": float(np.mean(epoch_brel_subject_fracs)) if epoch_brel_subject_fracs else 0.0,
                "baseline_relative_active_sample_fraction": float(np.mean(epoch_brel_sample_fracs)) if epoch_brel_sample_fracs else 0.0,
                "baseline_relative_active_batch_fraction": float(epoch_brel_active_batches / max(len(train_loader), 1)),
                "e4_modality_dropout": float(np.mean(epoch_e4_drop_rates)) if epoch_e4_drop_rates else 0.0,
                "e4_cls_weight": e4_cls_weight,
                "align_weight": align_weight,
                "rhythm_weight": rhythm_weight,
                "wavelet_weight": wavelet_weight,
                "val_watch_acc": val_metrics["acc"],
                "val_watch_balanced_acc": val_metrics["balanced_acc"],
                "val_watch_f1": val_metrics["f1"],
                "val_watch_auroc": val_metrics["auroc"],
                "val_teacher_acc": val_teacher_metrics["acc"],
                "val_teacher_balanced_acc": val_teacher_metrics["balanced_acc"],
                "val_teacher_f1": val_teacher_metrics["f1"],
                "val_teacher_auroc": val_teacher_metrics["auroc"],
                "val_watch_threshold": watch_threshold,
                "val_teacher_threshold": teacher_threshold,
                "val_watch_positive_rate": val_metrics["positive_rate"],
                "val_teacher_positive_rate": val_teacher_metrics["positive_rate"],
                "threshold_metric": threshold_metric,
                "watch_test_acc": test_metrics["acc"],
                "watch_test_balanced_acc": test_metrics["balanced_acc"],
                "watch_test_f1": test_metrics["f1"],
                "watch_test_auroc": test_metrics["auroc"],
                "watch_test_positive_rate": test_metrics["positive_rate"],
                "teacher_test_acc": teacher_test_metrics["acc"],
                "teacher_test_balanced_acc": teacher_test_metrics["balanced_acc"],
                "teacher_test_f1": teacher_test_metrics["f1"],
                "teacher_test_auroc": teacher_test_metrics["auroc"],
                "teacher_test_positive_rate": teacher_test_metrics["positive_rate"],
            }
        )

        if args.selection_mode == "fixed_epoch" and epoch + 1 == args.selection_epoch:
            source_model = ema_model if ema_model is not None else model
            best_state = {key: value.detach().cpu() for key, value in source_model.state_dict().items()}
            best_watch_threshold = watch_threshold
            best_teacher_threshold = teacher_threshold
            best_test_metrics = test_metrics
            best_teacher_test_metrics = teacher_test_metrics
            print(
                f"selected fixed epoch {epoch + 1} "
                f"| watch_test_balanced_acc={best_test_metrics['balanced_acc']:.4f} "
                f"watch_test_auroc={best_test_metrics['auroc']:.4f} "
                f"| teacher_test_balanced_acc={best_teacher_test_metrics['balanced_acc']:.4f} "
                f"teacher_test_auroc={best_teacher_test_metrics['auroc']:.4f}"
            )

        if args.selection_mode == "early_stop":
            score = watch_score if args.selection_target == "watch" else teacher_score
            improved = stopper.step(score, epoch + 1)
            if improved:
                source_model = ema_model if ema_model is not None else model
                best_state = {key: value.detach().cpu() for key, value in source_model.state_dict().items()}
                best_watch_threshold = watch_threshold
                best_teacher_threshold = teacher_threshold
                best_test_metrics = test_metrics
                best_teacher_test_metrics = teacher_test_metrics
                print(
                    f"new best {args.selection_target}_{args.monitor}={score:.4f} at epoch {epoch + 1} "
                    f"| watch_threshold={best_watch_threshold:.4f} "
                    f"teacher_threshold={best_teacher_threshold:.4f} "
                    f"| watch_test_balanced_acc={best_test_metrics['balanced_acc']:.4f} "
                    f"watch_test_auroc={best_test_metrics['auroc']:.4f} "
                    f"| teacher_test_balanced_acc={best_teacher_test_metrics['balanced_acc']:.4f} "
                    f"teacher_test_auroc={best_teacher_test_metrics['auroc']:.4f}"
                )

            if stopper.should_stop():
                print(
                    f"early stopping triggered: no improvement in {args.early_stop_patience} epochs; "
                    f"best_{args.selection_target}_{args.monitor}={stopper.best_score:.4f} at epoch {stopper.best_epoch}"
                )
                break

    if best_state is None:
        source_model = ema_model if ema_model is not None else model
        best_state = {key: value.detach().cpu() for key, value in source_model.state_dict().items()}
        if history_rows:
            final = history_rows[-1]
            best_watch_threshold = float(final["val_watch_threshold"])
            best_teacher_threshold = float(final["val_teacher_threshold"])
            best_test_metrics = {
                "acc": float(final["watch_test_acc"]),
                "balanced_acc": float(final["watch_test_balanced_acc"]),
                "f1": float(final["watch_test_f1"]),
                "auroc": float(final["watch_test_auroc"]),
                "positive_rate": float(final["watch_test_positive_rate"]),
            }
            best_teacher_test_metrics = {
                "acc": float(final["teacher_test_acc"]),
                "balanced_acc": float(final["teacher_test_balanced_acc"]),
                "f1": float(final["teacher_test_f1"]),
                "auroc": float(final["teacher_test_auroc"]),
                "positive_rate": float(final["teacher_test_positive_rate"]),
            }

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
    if best_watch_epoch is not None:
        print(f"best_watch_{args.monitor}={best_watch_score:.4f} at epoch {best_watch_epoch}")
    if best_teacher_epoch is not None:
        print(f"best_teacher_{args.monitor}={best_teacher_score:.4f} at epoch {best_teacher_epoch}")
    if best_test_metrics is not None:
        print(
            f"best_watch_threshold={best_watch_threshold:.4f} "
            f"best_watch_test_acc={best_test_metrics['acc']:.4f} "
            f"best_watch_test_balanced_acc={best_test_metrics['balanced_acc']:.4f} "
            f"best_watch_test_f1={best_test_metrics['f1']:.4f} "
            f"best_watch_test_auroc={best_test_metrics['auroc']:.4f} "
            f"best_watch_test_positive_rate={best_test_metrics['positive_rate']:.4f}"
        )
    if best_teacher_test_metrics is not None:
        print(
            f"best_teacher_threshold={best_teacher_threshold:.4f} "
            f"best_teacher_test_acc={best_teacher_test_metrics['acc']:.4f} "
            f"best_teacher_test_balanced_acc={best_teacher_test_metrics['balanced_acc']:.4f} "
            f"best_teacher_test_f1={best_teacher_test_metrics['f1']:.4f} "
            f"best_teacher_test_auroc={best_teacher_test_metrics['auroc']:.4f} "
            f"best_teacher_test_positive_rate={best_teacher_test_metrics['positive_rate']:.4f}"
        )
    print(f"Saved privileged Galaxy teacher checkpoint to {args.save_path}")


if __name__ == "__main__":
    main()
