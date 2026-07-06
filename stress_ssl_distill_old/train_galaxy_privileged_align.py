from __future__ import annotations

import argparse
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
from .galaxy_models import PrivilegedGalaxyTeacherNet
from .metrics import classification_metrics


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


def build_loader(dataset: GalaxyPrivilegedWindowDataset, batch_size: int, shuffle: bool, num_workers: int, pin_memory: bool) -> DataLoader:
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **kwargs)


def maybe_parse_sessions(values: Sequence[str] | None, fallback: Iterable[str]) -> list[str]:
    if values is None or len(values) == 0:
        return list(fallback)
    return [str(item) for item in values]


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
    focal = (1.0 - pt).pow(gamma) * ce
    weights = 0.35 + 0.65 * quality.squeeze(1).clamp(0.0, 1.0)
    return (focal * weights).sum() / weights.sum().clamp(min=1e-6)


def cross_modal_contrastive_loss(
    watch_proj: torch.Tensor,
    e4_proj: torch.Tensor,
    quality: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    logits = torch.matmul(watch_proj, e4_proj.T) / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    weights = (0.35 + 0.65 * quality.squeeze(1).clamp(0.0, 1.0)).detach()

    loss_w = F.cross_entropy(logits, labels, reduction="none")
    loss_e = F.cross_entropy(logits.T, labels, reduction="none")
    loss = ((loss_w + loss_e) * 0.5 * weights).sum() / weights.sum().clamp(min=1e-6)
    return loss


def masked_rhythm_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    scales = pred.new_tensor([200.0, 1500.0, 300.0]).unsqueeze(0)
    target_scaled = target / scales
    loss = F.smooth_l1_loss(pred, target_scaled, reduction="none")
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


def collect_outputs(model: PrivilegedGalaxyTeacherNet, loader: DataLoader, device: str, pin_memory: bool) -> tuple[list[int], list[float]]:
    model.eval()
    y_true: list[int] = []
    y_prob: list[float] = []
    with torch.no_grad():
        for batch in loader:
            watch_signal = batch["watch_signal"].to(device, non_blocking=pin_memory)
            wavelet = batch["wavelet_features"].to(device, non_blocking=pin_memory)
            quality = batch["watch_quality"].to(device, non_blocking=pin_memory)
            labels = batch["label"].to(device, non_blocking=pin_memory).long()

            logits = model.forward_watch(watch_signal, wavelet, quality)["logits"]
            probs = torch.softmax(logits, dim=1)[:, 1]
            y_true.extend(labels.detach().cpu().tolist())
            y_prob.extend(probs.detach().cpu().tolist())
    return y_true, y_prob


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
    parser = argparse.ArgumentParser(description="Train privileged GalaxyPPG with direct watch-E4 alignment and scheduled auxiliary losses.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--save-path", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--focal-gamma", type=float, default=1.5)
    parser.add_argument("--align-weight", type=float, default=0.10)
    parser.add_argument("--rhythm-weight", type=float, default=0.10)
    parser.add_argument("--e4-cls-weight", type=float, default=0.05)
    parser.add_argument("--wavelet-weight", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--priv-schedule", type=str, default="cosine", choices=["constant", "linear", "cosine"])
    parser.add_argument("--priv-floor", type=float, default=0.2)
    parser.add_argument("--monitor", type=str, default="balanced_acc", choices=["acc", "balanced_acc", "f1", "auroc"])
    parser.add_argument("--selection-mode", type=str, default="fixed_epoch", choices=["fixed_epoch", "early_stop"])
    parser.add_argument("--selection-epoch", type=int, default=2)
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
    args = parser.parse_args()

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

    train_loader = build_loader(train_ds, args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=args.pin_memory)
    test_loader = build_loader(test_ds, args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)
    has_val_split = len(val_ds) > 0
    val_loader = build_loader(val_ds, args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory) if has_val_split else test_loader

    model = PrivilegedGalaxyTeacherNet().to(args.device)
    total_params = sum(p.numel() for p in model.parameters())
    watch_only_params = sum(p.numel() for n, p in model.named_parameters() if n.startswith("watch_encoder") or n.startswith("watch_classifier"))
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
    best_test_metrics = None
    best_threshold = 0.5
    history_rows: list[dict[str, float]] = []

    print(f"train windows={len(train_ds)} val windows={len(val_ds)} test windows={len(test_ds)}")
    if not has_val_split:
        print("selection_warning=val split missing; using test split for threshold/model selection")
    print(f"calm_sessions={calm_sessions}")
    print(f"stress_sessions={stress_sessions}")
    print(
        "loss_weights="
        f"cls:1.00 e4_cls:{args.e4_cls_weight:.2f} "
        f"align:{args.align_weight:.2f} rhythm:{args.rhythm_weight:.2f} "
        f"wavelet:{args.wavelet_weight:.2f}"
    )
    print(f"priv_schedule={args.priv_schedule} priv_floor={args.priv_floor:.2f}")
    print("priv_mode=direct_align_schedule")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        e4_cls_weight = scheduled_weight(args.e4_cls_weight, epoch + 1, args.epochs, args.priv_schedule, args.priv_floor)
        align_weight = scheduled_weight(args.align_weight, epoch + 1, args.epochs, args.priv_schedule, args.priv_floor)
        rhythm_weight = scheduled_weight(args.rhythm_weight, epoch + 1, args.epochs, args.priv_schedule, args.priv_floor)
        wavelet_weight = scheduled_weight(args.wavelet_weight, epoch + 1, args.epochs, args.priv_schedule, args.priv_floor)
        progress = tqdm(train_loader, desc=f"galaxy-priv-align epoch {epoch + 1}/{args.epochs}", leave=True)
        for batch in progress:
            watch_signal = batch["watch_signal"].to(args.device, non_blocking=args.pin_memory)
            e4_signal = batch["e4_signal"].to(args.device, non_blocking=args.pin_memory)
            wavelet = batch["wavelet_features"].to(args.device, non_blocking=args.pin_memory)
            quality = batch["watch_quality"].to(args.device, non_blocking=args.pin_memory)
            labels = batch["label"].to(args.device, non_blocking=args.pin_memory).long()
            polar_targets = batch["polar_targets"].to(args.device, non_blocking=args.pin_memory)
            polar_mask = batch["polar_target_mask"].to(args.device, non_blocking=args.pin_memory)

            out = model(watch_signal, wavelet, quality, e4_signal=e4_signal)
            cls_loss = quality_aware_focal_loss(
                out["logits"],
                labels,
                quality,
                class_weights=class_weights,
                gamma=args.focal_gamma,
                label_smoothing=args.label_smoothing,
            )
            e4_cls_loss = quality_aware_focal_loss(
                out["e4_logits"],
                labels,
                torch.ones_like(quality),
                class_weights=class_weights,
                gamma=args.focal_gamma,
                label_smoothing=args.label_smoothing,
            )
            align_loss = cross_modal_contrastive_loss(
                out["watch_proj"],
                out["e4_proj"],
                quality=quality,
                temperature=args.temperature,
            )
            rhythm_loss = masked_rhythm_loss(out["rhythm_pred"], polar_targets, polar_mask)
            wavelet_loss = F.smooth_l1_loss(out["wavelet_pred"], wavelet)
            loss = (
                cls_loss
                + e4_cls_weight * e4_cls_loss
                + align_weight * align_loss
                + rhythm_weight * rhythm_loss
                + wavelet_weight * wavelet_loss
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item())
            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                cls=f"{cls_loss.item():.4f}",
                e4=f"{e4_cls_loss.item():.4f}",
                align=f"{align_loss.item():.4f}",
                rhythm=f"{rhythm_loss.item():.4f}",
                align_w=f"{align_weight:.3f}",
            )

        scheduler.step()
        train_loss = total_loss / max(len(train_loader), 1)
        val_true, val_prob = collect_outputs(model, val_loader, args.device, args.pin_memory)
        threshold, val_metrics = select_threshold(val_true, val_prob, metric=args.monitor)
        test_true, test_prob = collect_outputs(model, test_loader, args.device, args.pin_memory)
        test_metrics = evaluate_with_threshold(test_true, test_prob, threshold=threshold)
        print(
            f"epoch={epoch + 1} "
            f"train_loss={train_loss:.4f} "
            f"val_acc={val_metrics['acc']:.4f} "
            f"val_balanced_acc={val_metrics['balanced_acc']:.4f} "
            f"val_f1={val_metrics['f1']:.4f} "
            f"val_auroc={val_metrics['auroc']:.4f} "
            f"val_threshold={val_metrics['threshold']:.4f} "
            f"val_positive_rate={val_metrics['positive_rate']:.4f} "
            f"test_balanced_acc={test_metrics['balanced_acc']:.4f} "
            f"test_auroc={test_metrics['auroc']:.4f}"
        )

        history_rows.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "e4_cls_weight": e4_cls_weight,
                "align_weight": align_weight,
                "rhythm_weight": rhythm_weight,
                "wavelet_weight": wavelet_weight,
                "val_acc": val_metrics["acc"],
                "val_balanced_acc": val_metrics["balanced_acc"],
                "val_f1": val_metrics["f1"],
                "val_auroc": val_metrics["auroc"],
                "val_threshold": val_metrics["threshold"],
                "val_positive_rate": val_metrics["positive_rate"],
                "test_acc": test_metrics["acc"],
                "test_balanced_acc": test_metrics["balanced_acc"],
                "test_f1": test_metrics["f1"],
                "test_auroc": test_metrics["auroc"],
                "test_positive_rate": test_metrics["positive_rate"],
            }
        )

        if args.selection_mode == "fixed_epoch" and epoch + 1 == args.selection_epoch:
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            best_threshold = threshold
            best_test_metrics = test_metrics
            print(
                f"selected fixed epoch {epoch + 1} "
                f"| test_balanced_acc={best_test_metrics['balanced_acc']:.4f} "
                f"test_auroc={best_test_metrics['auroc']:.4f}"
            )

        if args.selection_mode == "early_stop":
            score = val_metrics[args.monitor]
            improved = stopper.step(score, epoch + 1)
            if improved:
                best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
                best_threshold = threshold
                best_test_metrics = test_metrics
                print(
                    f"new best {args.monitor}={score:.4f} at epoch {epoch + 1} "
                    f"| val_threshold={best_threshold:.4f} "
                    f"| test_balanced_acc={best_test_metrics['balanced_acc']:.4f} "
                    f"test_auroc={best_test_metrics['auroc']:.4f}"
                )

            if stopper.should_stop():
                print(
                    f"early stopping triggered: no improvement in {args.early_stop_patience} epochs; "
                    f"best_{args.monitor}={stopper.best_score:.4f} at epoch {stopper.best_epoch}"
                )
                break

    if best_state is None:
        best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        if history_rows:
            final = history_rows[-1]
            best_threshold = float(final["val_threshold"])
            best_test_metrics = {
                "acc": float(final["test_acc"]),
                "balanced_acc": float(final["test_balanced_acc"]),
                "f1": float(final["test_f1"]),
                "auroc": float(final["test_auroc"]),
                "positive_rate": float(final["test_positive_rate"]),
            }

    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state if best_state is not None else model.state_dict(), args.save_path)
    if args.metrics_path is not None:
        args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(history_rows).to_csv(args.metrics_path, index=False)
        print(f"Saved epoch metrics to {args.metrics_path}")
    if args.selection_mode == "early_stop":
        print(f"best_{args.monitor}={stopper.best_score:.4f} at epoch {stopper.best_epoch}")
    else:
        print(f"selected_epoch={args.selection_epoch}")
    if best_test_metrics is not None:
        print(
            f"best_threshold={best_threshold:.4f} "
            f"best_test_acc={best_test_metrics['acc']:.4f} "
            f"best_test_balanced_acc={best_test_metrics['balanced_acc']:.4f} "
            f"best_test_f1={best_test_metrics['f1']:.4f} "
            f"best_test_auroc={best_test_metrics['auroc']:.4f} "
            f"best_test_positive_rate={best_test_metrics['positive_rate']:.4f}"
        )
    print(f"Saved privileged alignment checkpoint to {args.save_path}")


if __name__ == "__main__":
    main()
