from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .early_stopping import EarlyStopping
from .metrics import classification_metrics
from .reliability import cross_calibrated_trust, true_class_confidence, trust_weighted_kl_loss
from .samplers import SubjectAwareBatchSampler
from .wesad_dataset import WESADPrivilegedWindowDataset
from .wesad_models_safe_sgpc import WESADPrivilegedTeacherNet


DEFAULT_CALM_SESSIONS = ["baseline"]
DEFAULT_STRESS_SESSIONS = ["stress"]


def build_loader(
    dataset: WESADPrivilegedWindowDataset,
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
    model: WESADPrivilegedTeacherNet,
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
                privileged_signal = batch["privileged_signal"].to(device, non_blocking=pin_memory)
                logits = model(
                    watch_signal,
                    wavelet,
                    quality,
                    privileged_signal=privileged_signal,
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
    parser = argparse.ArgumentParser(
        description="Train the final WESAD privileged-to-deployable KD model."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", "--wesad-root", dest="dataset_root", type=Path, required=True)
    parser.add_argument("--save-path", type=Path, required=True)
    parser.add_argument("--init-watch-checkpoint", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--focal-gamma", type=float, default=1.5)
    parser.add_argument("--watch-cls-weight", type=float, default=1.0)
    parser.add_argument("--privileged-cls-weight", type=float, default=0.05)
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
    parser.add_argument("--cache-subjects", type=int, default=2)
    parser.add_argument(
        "--watch-backbone",
        type=str,
        default="wavelet_guided",
        choices=["wavelet_guided", "resnet18_1d", "resnet34_1d", "resnet50_1d"],
    )
    parser.add_argument("--watch-model-dim", type=int, default=192)
    parser.add_argument("--watch-transformer-layers", type=int, default=2)
    parser.add_argument("--watch-transformer-heads", type=int, default=4)
    parser.add_argument("--watch-fusion-hidden-dim", type=int, default=256)
    parser.add_argument("--watch-embed-dim", type=int, default=160)
    parser.add_argument("--watch-enhancement", type=str, default="none", choices=["none", "motion_disentangled"])
    parser.add_argument("--watch-motion-mode", type=str, default="strong", choices=["strong", "residual", "scaled"])
    args = parser.parse_args()

    set_random_seed(args.seed)
    calm_sessions = maybe_parse_sessions(args.calm_sessions, DEFAULT_CALM_SESSIONS)
    stress_sessions = maybe_parse_sessions(args.stress_sessions, DEFAULT_STRESS_SESSIONS)
    include_sessions = calm_sessions + stress_sessions

    dataset_kwargs = {
        "manifest_csv": args.manifest,
        "wesad_root": args.dataset_root,
        "include_sessions": include_sessions,
        "cache_subjects": args.cache_subjects,
        "wavelet": args.wavelet,
        "wavelet_level": args.wavelet_level,
    }
    train_ds = WESADPrivilegedWindowDataset(split="train", **dataset_kwargs)
    val_ds = WESADPrivilegedWindowDataset(split="val", **dataset_kwargs)
    test_ds = WESADPrivilegedWindowDataset(split="test", **dataset_kwargs)
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
    val_loader = (
        build_loader(val_ds, args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)
        if len(val_ds) > 0
        else test_loader
    )

    model = WESADPrivilegedTeacherNet(
        watch_backbone=args.watch_backbone,
        model_dim=args.watch_model_dim,
        transformer_layers=args.watch_transformer_layers,
        transformer_heads=args.watch_transformer_heads,
        fusion_hidden_dim=args.watch_fusion_hidden_dim,
        embed_dim=args.watch_embed_dim,
        watch_enhancement=args.watch_enhancement,
        watch_motion_mode=args.watch_motion_mode,
    ).to(args.device)
    if args.init_watch_checkpoint is not None:
        loaded, skipped = load_watch_checkpoint_into_privileged(model, args.init_watch_checkpoint)
        print(f"initialized_watch_checkpoint={args.init_watch_checkpoint} loaded={loaded} skipped={skipped}")

    use_ema_eval = not args.disable_ema_eval
    ema_model = copy.deepcopy(model).to(args.device) if use_ema_eval else None
    if ema_model is not None:
        ema_model.eval()
        for param in ema_model.parameters():
            param.requires_grad_(False)

    total_params = sum(p.numel() for p in model.parameters())
    watch_params = count_parameters_with_prefixes(model, ("watch_encoder", "watch_classifier"))
    aux_params = count_parameters_with_prefixes(model, ("wavelet_predictor",))
    print(f"training_params={total_params}")
    print(f"watch_inference_params={watch_params}")
    print(f"aux_head_params={aux_params}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    class_counts = train_ds.manifest["label"].value_counts().sort_index()
    neg = float(class_counts.get(0, 1.0))
    pos = float(class_counts.get(1, 1.0))
    class_weights = torch.tensor([1.0, neg / max(pos, 1.0)], device=args.device, dtype=torch.float32)

    stopper = EarlyStopping(mode="max", patience=args.early_stop_patience, min_delta=args.min_delta)
    history_rows: list[dict[str, float | int | str]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_watch_threshold = 0.5
    best_teacher_threshold = 0.5
    best_watch_metrics: dict[str, float] | None = None
    best_teacher_metrics: dict[str, float] | None = None

    threshold_metric = args.monitor if args.threshold_metric == "monitor" else args.threshold_metric
    cross_confidence_enabled = bool(args.cross_confidence_distill)
    print(f"threshold_metric={threshold_metric}")
    print(f"ema_eval={'on' if use_ema_eval else 'off'} ema_decay={args.ema_decay:.4f} seed={args.seed}")
    print(
        "loss_weights="
        f"watch:{args.watch_cls_weight:.2f} "
        f"privileged:{args.privileged_cls_weight:.2f} "
        f"teacher:{args.teacher_cls_weight:.2f} "
        f"distill:{args.distill_weight:.2f} "
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
        progress = tqdm(train_loader, desc=f"wesad-priv epoch {epoch + 1}/{args.epochs}", leave=True)

        for batch in progress:
            watch_signal = batch["watch_signal"].to(args.device, non_blocking=args.pin_memory)
            privileged_signal = batch["privileged_signal"].to(args.device, non_blocking=args.pin_memory)
            wavelet = batch["wavelet_features"].to(args.device, non_blocking=args.pin_memory)
            quality = batch["watch_quality"].to(args.device, non_blocking=args.pin_memory)
            labels = batch["label"].to(args.device, non_blocking=args.pin_memory).long()

            out = model(
                watch_signal,
                wavelet,
                quality,
                privileged_signal=privileged_signal,
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
            wavelet_loss = zero_loss
            if args.wavelet_weight > 0 and "wavelet_pred" in out:
                wavelet_loss = F.smooth_l1_loss(out["wavelet_pred"], wavelet)

            loss = (
                args.watch_cls_weight * watch_cls_loss
                + args.privileged_cls_weight * privileged_cls_loss
                + args.teacher_cls_weight * teacher_cls_loss
                + args.distill_weight * distill_loss
                + args.wavelet_weight * wavelet_loss
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
                watch=f"{watch_cls_loss.item():.4f}",
                priv=f"{privileged_cls_loss.item():.4f}",
                teacher=f"{teacher_cls_loss.item():.4f}",
                kd=f"{distill_loss.item():.4f}",
                xconf=f"{cross_trust_mean.item():.3f}",
                kdgate=f"{kd_gate_mean.item():.3f}",
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
        )
        watch_threshold, val_watch_metrics = select_threshold(
            val_watch_true,
            val_watch_prob,
            metric=threshold_metric,
        )
        test_watch_true, test_watch_prob = collect_outputs(
            eval_model,
            test_loader,
            args.device,
            args.pin_memory,
            mode="watch",
            aggregation=args.eval_aggregation,
        )
        test_watch_metrics = evaluate_with_threshold(test_watch_true, test_watch_prob, threshold=watch_threshold)

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
        test_teacher_true, test_teacher_prob = collect_outputs(
            eval_model,
            test_loader,
            args.device,
            args.pin_memory,
            mode="teacher",
            aggregation=args.eval_aggregation,
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
                "watch_cls_weight": args.watch_cls_weight,
                "privileged_cls_weight": args.privileged_cls_weight,
                "teacher_cls_weight": args.teacher_cls_weight,
                "distill_weight": args.distill_weight,
                "wavelet_weight": args.wavelet_weight,
                "cross_confidence_distill": int(cross_confidence_enabled),
                "cross_confidence_min_weight": args.cross_confidence_min_weight,
                "cross_confidence_trust_mean": float(np.mean(epoch_cross_trust)) if epoch_cross_trust else 0.0,
                "cross_confidence_trust_high_frac": float(np.mean(epoch_cross_high)) if epoch_cross_high else 0.0,
                "kd_gate_mode": args.kd_gate_mode,
                "kd_gate_min_weight": args.kd_gate_min_weight,
                "kd_gate_trust_mean": float(np.mean(epoch_kd_gate_trust)) if epoch_kd_gate_trust else 0.0,
                "kd_gate_trust_high_frac": float(np.mean(epoch_kd_gate_high)) if epoch_kd_gate_high else 0.0,
                "detach_standard_kd_teacher": int(args.detach_standard_kd_teacher),
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
                "threshold_metric": threshold_metric,
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

    if best_state is None:
        source_model = ema_model if ema_model is not None else model
        best_state = {key: value.detach().cpu() for key, value in source_model.state_dict().items()}
        if history_rows:
            final = history_rows[-1]
            best_watch_threshold = float(final["val_watch_threshold"])
            best_teacher_threshold = float(final["val_teacher_threshold"])
            best_watch_metrics = {
                "acc": float(final["test_watch_acc"]),
                "balanced_acc": float(final["test_watch_balanced_acc"]),
                "f1": float(final["test_watch_f1"]),
                "auroc": float(final["test_watch_auroc"]),
                "positive_rate": float(final["test_watch_positive_rate"]),
            }
            best_teacher_metrics = {
                "acc": float(final["test_teacher_acc"]),
                "balanced_acc": float(final["test_teacher_balanced_acc"]),
                "f1": float(final["test_teacher_f1"]),
                "auroc": float(final["test_teacher_auroc"]),
                "positive_rate": float(final["test_teacher_positive_rate"]),
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
