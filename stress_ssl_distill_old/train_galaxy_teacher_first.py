from __future__ import annotations

import argparse
import copy
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .early_stopping import EarlyStopping
from .galaxy_dataset import GalaxyPrivilegedWindowDataset
from .galaxy_models import PrivilegedGalaxyTeacherNet
from .train_galaxy_privileged import (
    DEFAULT_CALM_SESSIONS,
    DEFAULT_STRESS_SESSIONS,
    build_loader,
    collect_outputs,
    evaluate_with_threshold,
    masked_rhythm_loss,
    maybe_parse_sessions,
    quality_aware_focal_loss,
    scheduled_weight,
    select_threshold,
    update_ema_model,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a multimodal GalaxyPPG teacher first, without distillation.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--save-path", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--focal-gamma", type=float, default=1.5)
    parser.add_argument("--rhythm-weight", type=float, default=0.15)
    parser.add_argument("--e4-cls-weight", type=float, default=0.05)
    parser.add_argument("--teacher-cls-weight", type=float, default=1.00)
    parser.add_argument("--wavelet-weight", type=float, default=0.05)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--disable-ema-eval", action="store_true")
    parser.add_argument("--priv-schedule", type=str, default="cosine", choices=["constant", "linear", "cosine"])
    parser.add_argument("--priv-floor", type=float, default=0.2)
    parser.add_argument("--monitor", type=str, default="balanced_acc", choices=["acc", "balanced_acc", "f1", "auroc"])
    parser.add_argument("--selection-target", type=str, default="teacher", choices=["watch", "teacher"])
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
    use_ema_eval = not args.disable_ema_eval
    ema_model = copy.deepcopy(model).to(args.device) if use_ema_eval else None
    if ema_model is not None:
        ema_model.eval()
        for param in ema_model.parameters():
            param.requires_grad_(False)

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
    best_watch_test_metrics = None
    best_teacher_test_metrics = None
    best_watch_threshold = 0.5
    best_teacher_threshold = 0.5
    history_rows = []

    print(f"train windows={len(train_ds)} val windows={len(val_ds)} test windows={len(test_ds)}")
    if not has_val_split:
        print("selection_warning=val split missing; using test split for threshold/model selection")
    print(f"calm_sessions={calm_sessions}")
    print(f"stress_sessions={stress_sessions}")
    print(
        "loss_weights="
        f"teacher_cls:{args.teacher_cls_weight:.2f} "
        f"e4_cls:{args.e4_cls_weight:.2f} rhythm:{args.rhythm_weight:.2f} "
        f"wavelet:{args.wavelet_weight:.2f}"
    )
    print(
        f"selection_mode={args.selection_mode} "
        f"selection_target={args.selection_target} "
        f"monitor={args.monitor} "
        f"early_stop_patience={args.early_stop_patience}"
    )
    print(f"ema_eval={'on' if use_ema_eval else 'off'} ema_decay={args.ema_decay:.4f}")
    print(f"priv_schedule={args.priv_schedule} priv_floor={args.priv_floor:.2f}")
    print("priv_mode=teacher_first")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        e4_cls_weight = scheduled_weight(args.e4_cls_weight, epoch + 1, args.epochs, args.priv_schedule, args.priv_floor)
        rhythm_weight = scheduled_weight(args.rhythm_weight, epoch + 1, args.epochs, args.priv_schedule, args.priv_floor)
        teacher_cls_weight = scheduled_weight(args.teacher_cls_weight, epoch + 1, args.epochs, args.priv_schedule, args.priv_floor)
        wavelet_weight = scheduled_weight(args.wavelet_weight, epoch + 1, args.epochs, args.priv_schedule, args.priv_floor)

        progress = tqdm(train_loader, desc=f"teacher-first epoch {epoch + 1}/{args.epochs}", leave=True)
        for batch in progress:
            watch_signal = batch["watch_signal"].to(args.device, non_blocking=args.pin_memory)
            e4_signal = batch["e4_signal"].to(args.device, non_blocking=args.pin_memory)
            wavelet = batch["wavelet_features"].to(args.device, non_blocking=args.pin_memory)
            quality = batch["watch_quality"].to(args.device, non_blocking=args.pin_memory)
            labels = batch["label"].to(args.device, non_blocking=args.pin_memory).long()
            polar_targets = batch["polar_targets"].to(args.device, non_blocking=args.pin_memory)
            polar_mask = batch["polar_target_mask"].to(args.device, non_blocking=args.pin_memory)

            out = model(watch_signal, wavelet, quality, e4_signal=e4_signal)
            e4_quality = torch.ones_like(quality)
            teacher_cls_loss = quality_aware_focal_loss(
                out["teacher_logits"],
                labels,
                quality,
                class_weights=class_weights,
                gamma=args.focal_gamma,
                label_smoothing=args.label_smoothing,
            )
            e4_cls_loss = quality_aware_focal_loss(
                out["e4_logits"],
                labels,
                e4_quality,
                class_weights=class_weights,
                gamma=args.focal_gamma,
                label_smoothing=args.label_smoothing,
            )
            teacher_rhythm_loss = masked_rhythm_loss(out["teacher_rhythm_pred"], polar_targets, polar_mask)
            wavelet_loss = F.smooth_l1_loss(out["wavelet_pred"], wavelet)
            loss = (
                teacher_cls_weight * teacher_cls_loss
                + e4_cls_weight * e4_cls_loss
                + rhythm_weight * teacher_rhythm_loss
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
                teacher=f"{teacher_cls_loss.item():.4f}",
                e4=f"{e4_cls_loss.item():.4f}",
                rhythm=f"{teacher_rhythm_loss.item():.4f}",
            )

        scheduler.step()
        train_loss = total_loss / max(len(train_loader), 1)
        eval_model = ema_model if ema_model is not None else model
        val_watch_true, val_watch_prob = collect_outputs(eval_model, val_loader, args.device, args.pin_memory, mode="watch")
        watch_threshold, val_watch_metrics = select_threshold(val_watch_true, val_watch_prob, metric=args.monitor)
        val_teacher_true, val_teacher_prob = collect_outputs(eval_model, val_loader, args.device, args.pin_memory, mode="teacher")
        teacher_threshold, val_teacher_metrics = select_threshold(val_teacher_true, val_teacher_prob, metric=args.monitor)

        test_watch_true, test_watch_prob = collect_outputs(eval_model, test_loader, args.device, args.pin_memory, mode="watch")
        watch_test_metrics = evaluate_with_threshold(test_watch_true, test_watch_prob, threshold=watch_threshold)
        test_teacher_true, test_teacher_prob = collect_outputs(eval_model, test_loader, args.device, args.pin_memory, mode="teacher")
        teacher_test_metrics = evaluate_with_threshold(test_teacher_true, test_teacher_prob, threshold=teacher_threshold)

        watch_score = val_watch_metrics[args.monitor]
        teacher_score = val_teacher_metrics[args.monitor]
        score = watch_score if args.selection_target == "watch" else teacher_score

        print(
            f"epoch={epoch + 1} "
            f"train_loss={train_loss:.4f} "
            f"val_watch_balanced_acc={val_watch_metrics['balanced_acc']:.4f} "
            f"val_watch_auroc={val_watch_metrics['auroc']:.4f} "
            f"val_teacher_balanced_acc={val_teacher_metrics['balanced_acc']:.4f} "
            f"val_teacher_auroc={val_teacher_metrics['auroc']:.4f} "
            f"val_watch_threshold={watch_threshold:.4f} "
            f"val_teacher_threshold={teacher_threshold:.4f} "
            f"watch_test_balanced_acc={watch_test_metrics['balanced_acc']:.4f} "
            f"watch_test_auroc={watch_test_metrics['auroc']:.4f} "
            f"teacher_test_balanced_acc={teacher_test_metrics['balanced_acc']:.4f} "
            f"teacher_test_auroc={teacher_test_metrics['auroc']:.4f}"
        )

        history_rows.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "teacher_cls_weight": teacher_cls_weight,
                "e4_cls_weight": e4_cls_weight,
                "rhythm_weight": rhythm_weight,
                "wavelet_weight": wavelet_weight,
                "val_watch_acc": val_watch_metrics["acc"],
                "val_watch_balanced_acc": val_watch_metrics["balanced_acc"],
                "val_watch_f1": val_watch_metrics["f1"],
                "val_watch_auroc": val_watch_metrics["auroc"],
                "val_teacher_acc": val_teacher_metrics["acc"],
                "val_teacher_balanced_acc": val_teacher_metrics["balanced_acc"],
                "val_teacher_f1": val_teacher_metrics["f1"],
                "val_teacher_auroc": val_teacher_metrics["auroc"],
                "val_watch_threshold": watch_threshold,
                "val_teacher_threshold": teacher_threshold,
                "watch_test_acc": watch_test_metrics["acc"],
                "watch_test_balanced_acc": watch_test_metrics["balanced_acc"],
                "watch_test_f1": watch_test_metrics["f1"],
                "watch_test_auroc": watch_test_metrics["auroc"],
                "watch_test_positive_rate": watch_test_metrics["positive_rate"],
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
            best_watch_test_metrics = watch_test_metrics
            best_teacher_test_metrics = teacher_test_metrics
            print(
                f"selected fixed epoch {epoch + 1} "
                f"| watch_test_balanced_acc={best_watch_test_metrics['balanced_acc']:.4f} "
                f"watch_test_auroc={best_watch_test_metrics['auroc']:.4f} "
                f"| teacher_test_balanced_acc={best_teacher_test_metrics['balanced_acc']:.4f} "
                f"teacher_test_auroc={best_teacher_test_metrics['auroc']:.4f}"
            )

        if args.selection_mode == "early_stop":
            improved = stopper.step(score, epoch + 1)
            if improved:
                source_model = ema_model if ema_model is not None else model
                best_state = {key: value.detach().cpu() for key, value in source_model.state_dict().items()}
                best_watch_threshold = watch_threshold
                best_teacher_threshold = teacher_threshold
                best_watch_test_metrics = watch_test_metrics
                best_teacher_test_metrics = teacher_test_metrics
                print(
                    f"new best {args.selection_target}_{args.monitor}={score:.4f} at epoch {epoch + 1} "
                    f"| watch_threshold={best_watch_threshold:.4f} "
                    f"teacher_threshold={best_teacher_threshold:.4f} "
                    f"| watch_test_balanced_acc={best_watch_test_metrics['balanced_acc']:.4f} "
                    f"watch_test_auroc={best_watch_test_metrics['auroc']:.4f} "
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
            best_watch_test_metrics = {
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

    if best_watch_test_metrics is not None:
        print(
            f"best_watch_threshold={best_watch_threshold:.4f} "
            f"best_watch_test_acc={best_watch_test_metrics['acc']:.4f} "
            f"best_watch_test_balanced_acc={best_watch_test_metrics['balanced_acc']:.4f} "
            f"best_watch_test_f1={best_watch_test_metrics['f1']:.4f} "
            f"best_watch_test_auroc={best_watch_test_metrics['auroc']:.4f} "
            f"best_watch_test_positive_rate={best_watch_test_metrics['positive_rate']:.4f}"
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
    print(f"Saved teacher-first Galaxy checkpoint to {args.save_path}")


if __name__ == "__main__":
    main()
