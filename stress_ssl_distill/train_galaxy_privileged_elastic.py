from __future__ import annotations

import argparse
import copy
import math
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

if __package__ in {None, ""}:
    # Allow IDEs to run this file directly while keeping package-relative imports
    # for normal `python -m stress_ssl_distill_new.train_galaxy_privileged_elastic`.
    package_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(package_dir.parent))
    __package__ = package_dir.name

from .early_stopping import EarlyStopping
from .galaxy_dataset import GalaxyPrivilegedWindowDataset
from .galaxy_models import PrivilegedGalaxyTeacherNet
from .metrics import classification_metrics
from .reliability import cross_calibrated_trust, true_class_confidence, trust_weighted_kl_loss
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


def build_loader(
    dataset: GalaxyPrivilegedWindowDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    batch_sampler: SubjectAwareBatchSampler | None = None,
) -> DataLoader:
    kwargs = {"num_workers": num_workers, "pin_memory": pin_memory}
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


def quality_aware_focal_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    quality: torch.Tensor,
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
    focal = (1.0 - pt).pow(gamma)
    weights = 0.35 + 0.65 * quality.squeeze(1).clamp(0.0, 1.0)
    return (focal * ce * weights).sum() / weights.sum().clamp(min=1e-6)


def distillation_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    quality: torch.Tensor,
    temperature: float,
    detach_teacher: bool,
) -> torch.Tensor:
    student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
    teacher_source = teacher_logits.detach() if detach_teacher else teacher_logits
    teacher_probs = F.softmax(teacher_source / temperature, dim=1)
    kl = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=1)
    weights = 0.35 + 0.65 * quality.squeeze(1).clamp(0.0, 1.0)
    return (kl * weights).sum() / weights.sum().clamp(min=1e-6) * (temperature**2)


def apply_min_trust_floor(trust: torch.Tensor, min_weight: float) -> torch.Tensor:
    floor = min(max(float(min_weight), 0.0), 1.0)
    if floor <= 0.0:
        return trust
    return floor + (1.0 - floor) * trust


def true_class_kd_gate(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    quality: torch.Tensor,
    mode: str,
    min_weight: float,
) -> torch.Tensor:
    if mode == "teacher_true_confidence":
        trust = true_class_confidence(teacher_logits, labels)
    elif mode == "student_true_confidence":
        trust = true_class_confidence(student_logits, labels)
    else:
        raise ValueError(f"Unsupported KD gate mode: {mode}")
    trust = trust * quality.squeeze(1).clamp(0.0, 1.0)
    return apply_min_trust_floor(trust, min_weight=min_weight)


def masked_rhythm_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.sum() <= 1e-6:
        return pred.new_tensor(0.0)
    loss = F.smooth_l1_loss(pred, target, reduction="none")
    return (loss * mask).sum() / mask.sum().clamp(min=1e-6)


def scheduled_weight(base_weight: float, epoch: int, total_epochs: int, mode: str, floor: float) -> float:
    if base_weight == 0.0 or mode == "constant":
        return float(base_weight)
    progress = min(max(epoch / max(total_epochs, 1), 0.0), 1.0)
    floor = min(max(float(floor), 0.0), 1.0)
    if mode == "linear":
        scale = floor + (1.0 - floor) * progress
    elif mode == "cosine":
        scale = floor + (1.0 - floor) * 0.5 * (1.0 - math.cos(math.pi * progress))
    else:
        raise ValueError(f"Unsupported schedule mode: {mode}")
    return float(base_weight) * scale


def update_ema_model(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    with torch.no_grad():
        ema_state = ema_model.state_dict()
        model_state = model.state_dict()
        for key, ema_value in ema_state.items():
            model_value = model_state[key].detach()
            if not torch.is_floating_point(ema_value):
                ema_value.copy_(model_value)
            else:
                ema_value.mul_(decay).add_(model_value, alpha=1.0 - decay)


def _build_debug_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Debug one Galaxy privileged training step.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--debug-random-batch", action="store_true")
    mode.add_argument("--debug-real-batch", action="store_true")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--calm-sessions", nargs="*", default=["baseline"])
    parser.add_argument("--stress-sessions", nargs="*", default=["tsst-prep"])
    parser.add_argument("--watch-enhancement", type=str, default="motion_disentangled")
    parser.add_argument("--watch-motion-mode", type=str, default="strong", choices=["strong", "residual", "scaled"])
    parser.add_argument("--no-scaled-motion-compat", dest="scaled_motion_compat", action="store_false")
    parser.set_defaults(scaled_motion_compat=True)
    parser.add_argument("--watch-cls-weight", type=float, default=1.0)
    parser.add_argument("--teacher-cls-weight", type=float, default=0.80)
    parser.add_argument("--distill-weight", type=float, default=0.08)
    parser.add_argument("--e4-cls-weight", type=float, default=0.05)
    parser.add_argument("--rhythm-weight", type=float, default=0.15)
    parser.add_argument("--wavelet-weight", type=float, default=0.05)
    parser.add_argument("--distill-temp", type=float, default=4.0)
    parser.add_argument("--cross-confidence-distill", dest="cross_confidence_distill", action="store_true")
    parser.add_argument("--no-cross-confidence-distill", dest="cross_confidence_distill", action="store_false")
    parser.set_defaults(cross_confidence_distill=True)
    parser.add_argument("--cross-confidence-min-weight", type=float, default=0.0)
    parser.add_argument("--kd-gate-mode", type=str, default="none", choices=["none", "teacher_true_confidence", "student_true_confidence"])
    parser.add_argument("--kd-gate-min-weight", type=float, default=0.0)
    parser.add_argument("--detach-standard-kd-teacher", action="store_true")
    parser.add_argument("--focal-gamma", type=float, default=1.5)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    return parser


def _make_debug_model(args: argparse.Namespace) -> PrivilegedGalaxyTeacherNet:
    if args.scaled_motion_compat:
        from .run_galaxy_loso_eval_elastic_scaled_motion import install_scaled_motion_film

        install_scaled_motion_film()
        print("debug_scaled_motion_compat=on")
    return PrivilegedGalaxyTeacherNet(
        watch_backbone="wavelet_guided",
        use_e4_classifier=args.e4_cls_weight > 0.0,
        use_rhythm_heads=args.rhythm_weight > 0.0,
        use_wavelet_head=args.wavelet_weight > 0.0,
        watch_enhancement=args.watch_enhancement,
        watch_motion_mode=args.watch_motion_mode,
    ).to(args.device)


def _make_random_debug_batch(batch_size: int, device: str) -> dict[str, torch.Tensor]:
    labels = torch.randint(0, 2, (batch_size,), device=device)
    if batch_size >= 2:
        labels[0] = 0
        labels[1] = 1
    return {
        "watch_signal": torch.randn(batch_size, 5, 500, device=device),
        "e4_signal": torch.randn(batch_size, 5, 640, device=device),
        "wavelet_features": torch.rand(batch_size, 4, device=device),
        "watch_quality": torch.rand(batch_size, 1, device=device),
        "label": labels.long(),
        "polar_targets": torch.randn(batch_size, 3, device=device),
        "polar_target_mask": (torch.rand(batch_size, 3, device=device) > 0.2).float(),
    }


def _move_debug_batch_to_device(batch: dict[str, torch.Tensor], device: str, pin_memory: bool) -> dict[str, torch.Tensor]:
    keys = (
        "watch_signal",
        "e4_signal",
        "wavelet_features",
        "watch_quality",
        "label",
        "polar_targets",
        "polar_target_mask",
    )
    out: dict[str, torch.Tensor] = {}
    for key in keys:
        value = batch[key]
        dtype = torch.long if key == "label" else torch.float32
        out[key] = value.to(device, dtype=dtype, non_blocking=pin_memory)
    return out


def _run_debug_train_step(
    model: PrivilegedGalaxyTeacherNet,
    batch: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> dict[str, float]:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    labels = batch["label"]
    neg = float((labels == 0).sum().detach().cpu().item())
    pos = float((labels == 1).sum().detach().cpu().item())
    class_weights = torch.tensor([1.0, neg / max(pos, 1.0)], device=args.device, dtype=torch.float32)

    out = model(
        batch["watch_signal"],
        batch["wavelet_features"],
        batch["watch_quality"],
        e4_signal=batch["e4_signal"],
    )
    zero_loss = out["logits"].new_tensor(0.0)
    cls_loss = quality_aware_focal_loss(
        out["logits"],
        labels,
        batch["watch_quality"],
        class_weights=class_weights,
        gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
    )
    teacher_cls_loss = quality_aware_focal_loss(
        out["teacher_logits"],
        labels,
        batch["watch_quality"],
        class_weights=class_weights,
        gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
    )
    e4_cls_loss = zero_loss
    if args.e4_cls_weight > 0 and "e4_logits" in out:
        e4_cls_loss = quality_aware_focal_loss(
            out["e4_logits"],
            labels,
            torch.ones_like(batch["watch_quality"]),
            class_weights=class_weights,
            gamma=args.focal_gamma,
            label_smoothing=args.label_smoothing,
        )

    trust = None
    if args.cross_confidence_distill:
        trust = cross_calibrated_trust(out["logits"], out["teacher_logits"], labels, batch["watch_quality"])
        trust = apply_min_trust_floor(trust, min_weight=args.cross_confidence_min_weight)
    elif args.kd_gate_mode != "none":
        trust = true_class_kd_gate(
            out["logits"],
            out["teacher_logits"],
            labels,
            batch["watch_quality"],
            mode=args.kd_gate_mode,
            min_weight=args.kd_gate_min_weight,
        )

    if trust is not None:
        distill_loss = trust_weighted_kl_loss(
            out["logits"],
            out["teacher_logits"],
            trust,
            temperature=args.distill_temp,
        )
    else:
        distill_loss = distillation_kl_loss(
            out["logits"],
            out["teacher_logits"],
            quality=batch["watch_quality"],
            temperature=args.distill_temp,
            detach_teacher=args.detach_standard_kd_teacher,
        )

    rhythm_loss = zero_loss
    if args.rhythm_weight > 0 and "rhythm_pred" in out and "teacher_rhythm_pred" in out:
        watch_rhythm_loss = masked_rhythm_loss(out["rhythm_pred"], batch["polar_targets"], batch["polar_target_mask"])
        teacher_rhythm_loss = masked_rhythm_loss(out["teacher_rhythm_pred"], batch["polar_targets"], batch["polar_target_mask"])
        rhythm_loss = 0.5 * (watch_rhythm_loss + teacher_rhythm_loss)

    wavelet_loss = zero_loss
    if args.wavelet_weight > 0 and "wavelet_pred" in out:
        wavelet_loss = F.smooth_l1_loss(out["wavelet_pred"], batch["wavelet_features"])

    loss = (
        args.watch_cls_weight * cls_loss
        + args.teacher_cls_weight * teacher_cls_loss
        + args.distill_weight * distill_loss
        + args.e4_cls_weight * e4_cls_loss
        + args.rhythm_weight * rhythm_loss
        + args.wavelet_weight * wavelet_loss
    )

    # Debug breakpoint sweet spot: inspect batch/out/losses here before backward.
    optimizer.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    output_shapes = {key: tuple(value.shape) for key, value in out.items() if torch.is_tensor(value)}
    print(f"debug_output_shapes={output_shapes}")
    return {
        "loss": float(loss.detach().cpu().item()),
        "cls_loss": float(cls_loss.detach().cpu().item()),
        "teacher_cls_loss": float(teacher_cls_loss.detach().cpu().item()),
        "distill_loss": float(distill_loss.detach().cpu().item()),
        "e4_cls_loss": float(e4_cls_loss.detach().cpu().item()),
        "rhythm_loss": float(rhythm_loss.detach().cpu().item()),
        "wavelet_loss": float(wavelet_loss.detach().cpu().item()),
        "trust_mean": float(trust.mean().detach().cpu().item()) if trust is not None else 1.0,
        "grad_norm": float(grad_norm.detach().cpu().item() if torch.is_tensor(grad_norm) else grad_norm),
    }


def debug_main(argv: Sequence[str] | None = None) -> None:
    parser = _build_debug_parser()
    args = parser.parse_args(argv)
    set_random_seed(args.seed)
    model = _make_debug_model(args)
    total_params = sum(param.numel() for param in model.parameters())
    watch_params = count_parameters_with_prefixes(model, ("watch_encoder", "watch_classifier"))
    print(f"debug_model_params total={total_params} watch_inference={watch_params}")

    if args.debug_random_batch:
        batch = _make_random_debug_batch(args.batch_size, args.device)
        print(
            "debug_batch=random "
            f"watch_signal={tuple(batch['watch_signal'].shape)} "
            f"e4_signal={tuple(batch['e4_signal'].shape)}"
        )
    else:
        if args.manifest is None or args.dataset_root is None:
            raise ValueError("--debug-real-batch requires --manifest and --dataset-root.")
        dataset = GalaxyPrivilegedWindowDataset(
            manifest_csv=args.manifest,
            split=args.split,
            dataset_root=args.dataset_root,
            include_sessions=list(args.calm_sessions) + list(args.stress_sessions),
            cache_tables=True,
        )
        loader = build_loader(
            dataset,
            args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
        )
        raw_batch = next(iter(loader))
        print(f"debug_batch=real split={args.split} windows={len(dataset)}")
        batch = _move_debug_batch_to_device(raw_batch, args.device, args.pin_memory)
        print(
            f"watch_signal={tuple(batch['watch_signal'].shape)} "
            f"e4_signal={tuple(batch['e4_signal'].shape)} "
            f"wavelet={tuple(batch['wavelet_features'].shape)}"
        )

    stats = _run_debug_train_step(model, batch, args)
    print("debug_train_step=OK " + " ".join(f"{key}={value:.6f}" for key, value in stats.items()))


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
        raise ValueError(f"Unsupported aggregation: {aggregation}")
    grouped: dict[tuple[str, str], list[tuple[int, float]]] = {}
    for label, prob, subject_id, session in zip(y_true, y_prob, subject_ids, sessions):
        grouped.setdefault((subject_id, session), []).append((label, prob))
    agg_true: list[int] = []
    agg_prob: list[float] = []
    for values in grouped.values():
        labels = [item[0] for item in values]
        probs = [item[1] for item in values]
        agg_true.append(int(round(float(np.mean(labels)))))
        agg_prob.append(float(np.mean(probs)))
    return agg_true, agg_prob


def evaluate_with_threshold(y_true: list[int], y_prob: list[float], threshold: float) -> dict[str, float]:
    metrics = classification_metrics(np.array(y_true), np.array(y_prob), threshold=threshold)
    metrics["threshold"] = float(threshold)
    return metrics


def select_threshold(y_true: list[int], y_prob: list[float], metric: str = "balanced_acc") -> tuple[float, dict[str, float]]:
    unique_probs = sorted(set(float(p) for p in y_prob))
    candidates = [0.5]
    candidates.extend(unique_probs)
    candidates.extend([min(max(p + 1e-6, 0.0), 1.0) for p in unique_probs])
    best_threshold = 0.5
    best_metrics = evaluate_with_threshold(y_true, y_prob, threshold=0.5)
    best_score = best_metrics[metric]
    for threshold in candidates:
        metrics = evaluate_with_threshold(y_true, y_prob, threshold=threshold)
        score = metrics[metric]
        if score > best_score + 1e-12:
            best_threshold = float(threshold)
            best_metrics = metrics
            best_score = score
    return best_threshold, best_metrics


def collect_outputs(
    model: PrivilegedGalaxyTeacherNet,
    loader: DataLoader,
    device: str,
    pin_memory: bool,
    mode: str = "watch",
    aggregation: str = "window",
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
            if mode == "watch":
                logits = model.forward_watch(watch_signal, wavelet, quality)["logits"]
            elif mode == "teacher":
                e4_signal = batch["e4_signal"].to(device, non_blocking=pin_memory)
                logits = model(watch_signal, wavelet, quality, e4_signal=e4_signal)["teacher_logits"]
            else:
                raise ValueError(f"Unsupported collect_outputs mode: {mode}")

            probs = torch.softmax(logits, dim=1)[:, 1]
            y_true.extend(labels.detach().cpu().tolist())
            y_prob.extend(probs.detach().cpu().tolist())
            subject_ids.extend(str(item) for item in batch["subject_id"])
            sessions.extend(str(item) for item in batch["session"])

    return aggregate_predictions(y_true, y_prob, subject_ids, sessions, aggregation)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the final Galaxy privileged-to-deployable KD model."
    )
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
    parser.add_argument("--rhythm-weight", type=float, default=0.15)
    parser.add_argument("--e4-cls-weight", type=float, default=0.05)
    parser.add_argument("--teacher-cls-weight", type=float, default=0.80)
    parser.add_argument("--distill-weight", type=float, default=0.08)
    parser.add_argument("--distill-temp", type=float, default=4.0)
    parser.add_argument("--detach-standard-kd-teacher", action="store_true")
    parser.add_argument("--wavelet-weight", type=float, default=0.05)
    parser.add_argument("--cross-confidence-distill", action="store_true")
    parser.add_argument("--cross-confidence-min-weight", type=float, default=0.0)
    parser.add_argument(
        "--kd-gate-mode",
        type=str,
        default="none",
        choices=["none", "teacher_true_confidence", "student_true_confidence"],
    )
    parser.add_argument("--kd-gate-min-weight", type=float, default=0.0)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--disable-ema-eval", action="store_true")
    parser.add_argument("--priv-schedule", type=str, default="cosine", choices=["constant", "linear", "cosine"])
    parser.add_argument("--priv-floor", type=float, default=0.2)
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
    )
    val_ds = GalaxyPrivilegedWindowDataset(
        manifest_csv=args.manifest,
        split="val",
        dataset_root=args.dataset_root,
        include_sessions=include_sessions,
        cache_tables=True,
        wavelet=args.wavelet,
        wavelet_level=args.wavelet_level,
    )
    test_ds = GalaxyPrivilegedWindowDataset(
        manifest_csv=args.manifest,
        split="test",
        dataset_root=args.dataset_root,
        include_sessions=include_sessions,
        cache_tables=True,
        wavelet=args.wavelet,
        wavelet_level=args.wavelet_level,
    )
    if len(train_ds) == 0 or len(test_ds) == 0:
        raise ValueError("Train or test split is empty after session filtering.")
    print(f"train windows={len(train_ds)} val windows={len(val_ds)} test windows={len(test_ds)}")
    print(f"watch_backbone={args.watch_backbone}")
    print(f"calm_sessions={calm_sessions}")
    print(f"stress_sessions={stress_sessions}")
    print(
        f"selection_mode={args.selection_mode} selection_target={args.selection_target} "
        f"monitor={args.monitor} early_stop_patience={args.early_stop_patience}"
    )
    print(f"subject_aware_batching={'on' if args.subject_aware_batching else 'off'}")
    print(f"eval_aggregation={args.eval_aggregation}")

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
    val_loader = (
        build_loader(val_ds, args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)
        if len(val_ds) > 0
        else test_loader
    )

    model = PrivilegedGalaxyTeacherNet(
        watch_backbone=args.watch_backbone,
        use_e4_classifier=args.e4_cls_weight > 0.0,
        use_rhythm_heads=args.rhythm_weight > 0.0,
        use_wavelet_head=args.wavelet_weight > 0.0,
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
    watch_params = count_parameters_with_prefixes(model, ("watch_encoder", "watch_classifier"))
    watch_aux_params = count_parameters_with_prefixes(model, ("rhythm_head", "wavelet_predictor"))
    e4_aux_params = count_parameters_with_prefixes(model, ("e4_classifier",))
    print(f"teacher_params={total_params}")
    print(f"training_params={total_params}")
    print(f"watch_inference_params={watch_params}")
    print(f"watch_aux_head_params={watch_aux_params}")
    print(f"e4_aux_head_params={e4_aux_params}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    class_counts = train_ds.manifest["label"].value_counts().sort_index()
    neg = float(class_counts.get(0, 1.0))
    pos = float(class_counts.get(1, 1.0))
    class_weights = torch.tensor([1.0, neg / max(pos, 1.0)], device=args.device, dtype=torch.float32)

    stopper = EarlyStopping(
        mode="max",
        patience=args.early_stop_patience,
        min_delta=args.min_delta,
    )
    history_rows: list[dict[str, float | int | str]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_watch_threshold = 0.5
    best_teacher_threshold = 0.5
    best_test_metrics: dict[str, float] | None = None
    best_teacher_test_metrics: dict[str, float] | None = None
    best_watch_score = -float("inf")
    best_teacher_score = -float("inf")
    best_watch_epoch = 0
    best_teacher_epoch = 0

    threshold_metric = args.monitor if args.threshold_metric == "monitor" else args.threshold_metric
    cross_confidence_enabled = bool(args.cross_confidence_distill)
    print(f"threshold_metric={threshold_metric}")
    print(f"ema_eval={'on' if use_ema_eval else 'off'} ema_decay={args.ema_decay:.4f} seed={args.seed}")
    print(f"priv_schedule={args.priv_schedule} priv_floor={args.priv_floor:.2f}")
    print(
        "loss_weights="
        f"watch:{args.watch_cls_weight:.2f} "
        f"teacher:{args.teacher_cls_weight:.2f} "
        f"distill:{args.distill_weight:.2f} "
        f"e4:{args.e4_cls_weight:.2f} "
        f"rhythm:{args.rhythm_weight:.2f} "
        f"wavelet:{args.wavelet_weight:.2f}"
    )
    print(
        "kd_mode="
        f"{'cross_confidence' if cross_confidence_enabled else args.kd_gate_mode} "
        f"cross_min={args.cross_confidence_min_weight:.3f} "
        f"kd_gate_min={args.kd_gate_min_weight:.3f} "
        f"detach_standard_kd_teacher={int(args.detach_standard_kd_teacher)}"
    )

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        epoch_cross_trust: list[float] = []
        epoch_cross_high: list[float] = []
        epoch_kd_gate_trust: list[float] = []
        epoch_kd_gate_high: list[float] = []

        e4_cls_weight = scheduled_weight(args.e4_cls_weight, epoch + 1, args.epochs, args.priv_schedule, args.priv_floor)
        rhythm_weight = scheduled_weight(args.rhythm_weight, epoch + 1, args.epochs, args.priv_schedule, args.priv_floor)
        teacher_cls_weight = scheduled_weight(args.teacher_cls_weight, epoch + 1, args.epochs, args.priv_schedule, args.priv_floor)
        distill_weight = scheduled_weight(args.distill_weight, epoch + 1, args.epochs, args.priv_schedule, args.priv_floor)
        wavelet_weight = scheduled_weight(args.wavelet_weight, epoch + 1, args.epochs, args.priv_schedule, args.priv_floor)
        watch_cls_weight = args.watch_cls_weight

        progress = tqdm(train_loader, desc=f"galaxy-priv epoch {epoch + 1}/{args.epochs}", leave=True)
        for batch in progress:
            watch_signal = batch["watch_signal"].to(args.device, non_blocking=args.pin_memory)
            e4_signal = batch["e4_signal"].to(args.device, non_blocking=args.pin_memory)
            wavelet = batch["wavelet_features"].to(args.device, non_blocking=args.pin_memory)
            quality = batch["watch_quality"].to(args.device, non_blocking=args.pin_memory)
            labels = batch["label"].to(args.device, non_blocking=args.pin_memory).long()
            polar_targets = batch["polar_targets"].to(args.device, non_blocking=args.pin_memory)
            polar_mask = batch["polar_target_mask"].to(args.device, non_blocking=args.pin_memory)

            out = model(watch_signal, wavelet, quality, e4_signal=e4_signal)
            zero_loss = out["logits"].new_tensor(0.0)
            cls_loss = quality_aware_focal_loss(
                out["logits"],
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
            e4_cls_loss = zero_loss
            if e4_cls_weight > 0 and "e4_logits" in out:
                e4_cls_loss = quality_aware_focal_loss(
                    out["e4_logits"],
                    labels,
                    torch.ones_like(quality),
                    class_weights=class_weights,
                    gamma=args.focal_gamma,
                    label_smoothing=args.label_smoothing,
                )

            trust = None
            cross_trust_mean = zero_loss
            kd_gate_mean = zero_loss
            if cross_confidence_enabled:
                trust = cross_calibrated_trust(out["logits"], out["teacher_logits"], labels, quality)
                trust = apply_min_trust_floor(trust, min_weight=args.cross_confidence_min_weight)
                cross_trust_mean = trust.mean()
                epoch_cross_trust.append(float(cross_trust_mean.detach().cpu().item()))
                epoch_cross_high.append(float((trust >= 0.5).float().mean().detach().cpu().item()))
            elif args.kd_gate_mode != "none":
                trust = true_class_kd_gate(
                    out["logits"],
                    out["teacher_logits"],
                    labels,
                    quality,
                    mode=args.kd_gate_mode,
                    min_weight=args.kd_gate_min_weight,
                )
                kd_gate_mean = trust.mean()
                epoch_kd_gate_trust.append(float(kd_gate_mean.detach().cpu().item()))
                epoch_kd_gate_high.append(float((trust >= 0.5).float().mean().detach().cpu().item()))

            if trust is not None:
                distill_loss = trust_weighted_kl_loss(
                    out["logits"],
                    out["teacher_logits"],
                    trust,
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
                + distill_weight * distill_loss
                + e4_cls_weight * e4_cls_loss
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
                kd=f"{distill_loss.item():.4f}",
                xconf=f"{cross_trust_mean.item():.3f}",
                kdgate=f"{kd_gate_mean.item():.3f}",
                e4=f"{e4_cls_loss.item():.4f}",
                rhythm=f"{rhythm_loss.item():.4f}",
                wav=f"{wavelet_loss.item():.4f}",
                kd_w=f"{distill_weight:.3f}",
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
        )
        watch_threshold, val_metrics = select_threshold(val_true, val_prob, metric=threshold_metric)
        val_teacher_true, val_teacher_prob = collect_outputs(
            eval_model,
            val_loader,
            args.device,
            args.pin_memory,
            mode="teacher",
            aggregation=args.eval_aggregation,
        )
        teacher_threshold, val_teacher_metrics = select_threshold(
            val_teacher_true,
            val_teacher_prob,
            metric=threshold_metric,
        )
        test_true, test_prob = collect_outputs(
            eval_model,
            test_loader,
            args.device,
            args.pin_memory,
            mode="watch",
            aggregation=args.eval_aggregation,
        )
        test_metrics = evaluate_with_threshold(test_true, test_prob, threshold=watch_threshold)
        teacher_test_true, teacher_test_prob = collect_outputs(
            eval_model,
            test_loader,
            args.device,
            args.pin_memory,
            mode="teacher",
            aggregation=args.eval_aggregation,
        )
        teacher_test_metrics = evaluate_with_threshold(
            teacher_test_true,
            teacher_test_prob,
            threshold=teacher_threshold,
        )

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
                "distill_weight": distill_weight,
                "e4_cls_weight": e4_cls_weight,
                "rhythm_weight": rhythm_weight,
                "wavelet_weight": wavelet_weight,
                "cross_confidence_distill": int(cross_confidence_enabled),
                "cross_confidence_min_weight": args.cross_confidence_min_weight,
                "cross_confidence_trust_mean": float(np.mean(epoch_cross_trust)) if epoch_cross_trust else 0.0,
                "cross_confidence_trust_high_frac": float(np.mean(epoch_cross_high)) if epoch_cross_high else 0.0,
                "kd_gate_mode": args.kd_gate_mode,
                "kd_gate_min_weight": args.kd_gate_min_weight,
                "kd_gate_trust_mean": float(np.mean(epoch_kd_gate_trust)) if epoch_kd_gate_trust else 0.0,
                "kd_gate_trust_high_frac": float(np.mean(epoch_kd_gate_high)) if epoch_kd_gate_high else 0.0,
                "detach_standard_kd_teacher": int(args.detach_standard_kd_teacher),
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
    torch.save(best_state, args.save_path)
    if args.metrics_path is not None:
        args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(history_rows).to_csv(args.metrics_path, index=False)
        print(f"Saved epoch metrics to {args.metrics_path}")
    if args.selection_mode == "early_stop":
        print(f"best_{args.selection_target}_{args.monitor}={stopper.best_score:.4f} at epoch {stopper.best_epoch}")
    else:
        print(f"selected_epoch={args.selection_epoch}")
    print(f"best_watch_val_{args.monitor}={best_watch_score:.4f} at epoch {best_watch_epoch}")
    print(f"best_teacher_val_{args.monitor}={best_teacher_score:.4f} at epoch {best_teacher_epoch}")
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
    print(f"Saved Galaxy privileged model checkpoint to {args.save_path}")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        debug_main(["--debug-random-batch"])
    elif "--debug-random-batch" in sys.argv or "--debug-real-batch" in sys.argv:
        debug_main(sys.argv[1:])
    else:
        main()
