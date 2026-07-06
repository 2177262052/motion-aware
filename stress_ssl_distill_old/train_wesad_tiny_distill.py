from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .early_stopping import EarlyStopping
from .galaxy_models import ResNet18WatchNet, ResNet34WatchNet, ResNet50WatchNet, TinyWaveletDistillNet, WaveletGuidedWatchNet
from .train_galaxy_tiny_distill import (
    collect_outputs,
    distillation_kl_loss,
    evaluate_with_threshold,
    feature_alignment_loss,
    ranking_distillation_loss,
    select_threshold,
)
from .train_galaxy_watch import build_loader, maybe_parse_sessions, quality_aware_focal_loss, set_random_seed
from .wesad_dataset import WESADWatchWindowDataset
from .wesad_models import WESADPrivilegedTeacherNet


DEFAULT_CALM_SESSIONS = ["baseline"]
DEFAULT_STRESS_SESSIONS = ["stress"]


def load_state_dict_flexible(model: torch.nn.Module, checkpoint_path: Path, device: str) -> None:
    state = torch.load(checkpoint_path, map_location=device)
    result = model.load_state_dict(state, strict=False)
    ignored_missing = [
        key
        for key in result.missing_keys
        if key.startswith("watch_contrastive_head.") or key.startswith("reliability_head.")
    ]
    remaining_missing = [key for key in result.missing_keys if key not in ignored_missing]
    if remaining_missing:
        print(f"teacher_load_missing_keys={remaining_missing}")
    if ignored_missing:
        print(f"teacher_load_ignored_missing_keys={ignored_missing}")
    if result.unexpected_keys:
        print(f"teacher_load_unexpected_keys={result.unexpected_keys}")


def build_watch_only_teacher(model_type: str, device: str) -> torch.nn.Module:
    if model_type == "wavelet_guided":
        return WaveletGuidedWatchNet().to(device)
    if model_type == "resnet18_1d":
        return ResNet18WatchNet().to(device)
    if model_type == "resnet34_1d":
        return ResNet34WatchNet().to(device)
    if model_type == "resnet50_1d":
        return ResNet50WatchNet().to(device)
    raise ValueError(f"Unsupported watch-only teacher model type: {model_type}")


def build_teacher(
    teacher_kind: str,
    teacher_path: Path,
    device: str,
    watch_only_model_type: str,
    watch_model_dim: int,
    watch_transformer_layers: int,
    watch_transformer_heads: int,
    watch_fusion_hidden_dim: int,
    watch_embed_dim: int,
    align_proj_dim: int,
    watch_backbone: str,
) -> torch.nn.Module:
    if teacher_kind == "deploy_watch":
        teacher = WESADPrivilegedTeacherNet(
            watch_backbone=watch_backbone,
            embed_dim=watch_embed_dim,
            align_dim=align_proj_dim,
            model_dim=watch_model_dim,
            transformer_layers=watch_transformer_layers,
            transformer_heads=watch_transformer_heads,
            fusion_hidden_dim=watch_fusion_hidden_dim,
        ).to(device)
    elif teacher_kind == "watch_only":
        teacher = build_watch_only_teacher(watch_only_model_type, device)
    else:
        raise ValueError(f"Unsupported teacher kind: {teacher_kind}")

    load_state_dict_flexible(teacher, teacher_path, device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    return teacher


@torch.no_grad()
def forward_teacher(
    teacher: torch.nn.Module,
    teacher_kind: str,
    signal: torch.Tensor,
    wavelet: torch.Tensor,
    quality: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if teacher_kind == "deploy_watch":
        out = teacher.forward_watch(signal, wavelet, quality)
        return out["logits"], out["watch_embedding"]
    out = teacher(signal, wavelet, quality)
    return out["logits"], out["embedding"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Distill a lightweight WESAD watch-only student from a stronger watch teacher.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", "--wesad-root", dest="dataset_root", type=Path, required=True)
    parser.add_argument("--teacher-path", type=Path, required=True)
    parser.add_argument("--teacher-kind", type=str, default="deploy_watch", choices=["deploy_watch", "watch_only"])
    parser.add_argument("--save-path", type=Path, required=True)
    parser.add_argument("--metrics-path", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--focal-gamma", type=float, default=1.5)
    parser.add_argument("--hard-weight", type=float, default=1.0)
    parser.add_argument("--kd-weight", type=float, default=1.0)
    parser.add_argument("--feat-weight", type=float, default=0.25)
    parser.add_argument("--ranking-distill-weight", type=float, default=0.0)
    parser.add_argument("--distill-temp", type=float, default=2.0)
    parser.add_argument("--monitor", type=str, default="balanced_acc", choices=["acc", "balanced_acc", "f1", "auroc"])
    parser.add_argument("--selection-mode", type=str, default="early_stop", choices=["fixed_epoch", "early_stop"])
    parser.add_argument("--selection-epoch", type=int, default=20)
    parser.add_argument("--early-stop-patience", type=int, default=12)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--eval-aggregation", type=str, default="window", choices=["window", "session_mean"])
    parser.add_argument("--threshold-mode", type=str, default="val_search", choices=["val_search", "fixed"])
    parser.add_argument("--fixed-threshold", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--calm-sessions", nargs="*", default=None)
    parser.add_argument("--stress-sessions", nargs="*", default=None)
    parser.add_argument("--wavelet", type=str, default="db4")
    parser.add_argument("--wavelet-level", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-subjects", type=int, default=15)
    parser.add_argument("--teacher-watch-only-model-type", type=str, default="wavelet_guided", choices=["wavelet_guided", "resnet18_1d", "resnet34_1d", "resnet50_1d"])
    parser.add_argument("--teacher-watch-backbone", type=str, default="wavelet_guided", choices=["wavelet_guided", "resnet18_1d", "resnet34_1d", "resnet50_1d"])
    parser.add_argument("--watch-model-dim", type=int, default=192)
    parser.add_argument("--watch-transformer-layers", type=int, default=2)
    parser.add_argument("--watch-transformer-heads", type=int, default=4)
    parser.add_argument("--watch-fusion-hidden-dim", type=int, default=256)
    parser.add_argument("--watch-embed-dim", type=int, default=160)
    parser.add_argument("--align-proj-dim", type=int, default=128)
    args = parser.parse_args()

    set_random_seed(args.seed)
    calm_sessions = maybe_parse_sessions(args.calm_sessions, DEFAULT_CALM_SESSIONS)
    stress_sessions = maybe_parse_sessions(args.stress_sessions, DEFAULT_STRESS_SESSIONS)
    include_sessions = calm_sessions + stress_sessions

    train_ds = WESADWatchWindowDataset(
        manifest_csv=args.manifest,
        split="train",
        wesad_root=args.dataset_root,
        include_sessions=include_sessions,
        cache_subjects=args.cache_subjects,
        wavelet=args.wavelet,
        wavelet_level=args.wavelet_level,
    )
    val_ds = WESADWatchWindowDataset(
        manifest_csv=args.manifest,
        split="val",
        wesad_root=args.dataset_root,
        include_sessions=include_sessions,
        cache_subjects=args.cache_subjects,
        wavelet=args.wavelet,
        wavelet_level=args.wavelet_level,
    )
    test_ds = WESADWatchWindowDataset(
        manifest_csv=args.manifest,
        split="test",
        wesad_root=args.dataset_root,
        include_sessions=include_sessions,
        cache_subjects=args.cache_subjects,
        wavelet=args.wavelet,
        wavelet_level=args.wavelet_level,
    )
    if len(train_ds) == 0 or len(test_ds) == 0:
        raise ValueError("Train or test split is empty after session filtering.")

    train_loader = build_loader(train_ds, args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=args.pin_memory)
    test_loader = build_loader(test_ds, args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)
    has_val_split = len(val_ds) > 0
    val_loader = build_loader(val_ds, args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory) if has_val_split else test_loader

    teacher = build_teacher(
        teacher_kind=args.teacher_kind,
        teacher_path=args.teacher_path,
        device=args.device,
        watch_only_model_type=args.teacher_watch_only_model_type,
        watch_model_dim=args.watch_model_dim,
        watch_transformer_layers=args.watch_transformer_layers,
        watch_transformer_heads=args.watch_transformer_heads,
        watch_fusion_hidden_dim=args.watch_fusion_hidden_dim,
        watch_embed_dim=args.watch_embed_dim,
        align_proj_dim=args.align_proj_dim,
        watch_backbone=args.teacher_watch_backbone,
    )
    sample = train_ds[0]
    signal_channels = int(sample["signal"].shape[0])
    wavelet_dim = int(sample["wavelet_features"].shape[0])
    student = TinyWaveletDistillNet(in_channels=signal_channels, wavelet_dim=wavelet_dim).to(args.device)
    tiny_params = sum(p.numel() for p in student.parameters())
    teacher_params = sum(p.numel() for p in teacher.parameters())
    print(f"tiny_student_params={tiny_params}")
    print(f"teacher_params={teacher_params}")
    print(f"tiny_input_channels={signal_channels} tiny_wavelet_dim={wavelet_dim}")

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    class_counts = train_ds.manifest["label"].value_counts().sort_index()
    neg = float(class_counts.get(0, 1.0))
    pos = float(class_counts.get(1, 1.0))
    class_weights = torch.tensor([1.0, neg / max(pos, 1.0)], device=args.device, dtype=torch.float32)

    stopper = EarlyStopping(patience=args.early_stop_patience, mode="max", min_delta=args.min_delta)
    best_state = None
    best_test_metrics = None
    best_threshold = 0.5
    history_rows: list[dict[str, float | int | str]] = []

    print(f"train windows={len(train_ds)} val windows={len(val_ds)} test windows={len(test_ds)}")
    if not has_val_split:
        print("selection_warning=val split missing; using test split for threshold/model selection")
    print("dataset=WESAD")
    print(f"teacher_kind={args.teacher_kind}")
    print(f"teacher_path={args.teacher_path}")
    print(f"calm_sessions={calm_sessions}")
    print(f"stress_sessions={stress_sessions}")
    print(f"cache_subjects={args.cache_subjects}")
    print(
        "loss_weights="
        f"hard:{args.hard_weight:.2f} "
        f"kd:{args.kd_weight:.2f} "
        f"feat:{args.feat_weight:.2f} "
        f"rank:{args.ranking_distill_weight:.2f}"
    )
    print(
        f"selection_mode={args.selection_mode} "
        f"monitor={args.monitor} "
        f"early_stop_patience={args.early_stop_patience}"
    )
    print(f"eval_aggregation={args.eval_aggregation}")
    print(f"threshold_mode={args.threshold_mode} fixed_threshold={args.fixed_threshold:.4f} seed={args.seed}")

    for epoch in range(args.epochs):
        student.train()
        total_loss = 0.0
        progress = tqdm(train_loader, desc=f"wesad-tiny epoch {epoch + 1}/{args.epochs}", leave=True)
        for batch in progress:
            signal = batch["signal"].to(args.device, non_blocking=args.pin_memory)
            wavelet = batch["wavelet_features"].to(args.device, non_blocking=args.pin_memory)
            quality = batch["watch_quality"].to(args.device, non_blocking=args.pin_memory)
            labels = batch["label"].to(args.device, non_blocking=args.pin_memory).long()

            with torch.no_grad():
                teacher_logits, teacher_features = forward_teacher(teacher, args.teacher_kind, signal, wavelet, quality)

            student_out = student(signal, wavelet, quality)
            hard_loss = quality_aware_focal_loss(
                student_out["logits"],
                labels,
                quality,
                class_weights=class_weights,
                gamma=args.focal_gamma,
                label_smoothing=args.label_smoothing,
            )
            kd_loss = distillation_kl_loss(
                student_out["logits"],
                teacher_logits,
                quality=quality,
                temperature=args.distill_temp,
            )
            ranking_loss = ranking_distillation_loss(
                student_out["logits"],
                teacher_logits,
                quality=quality,
            )
            feat_loss = feature_alignment_loss(
                student_out["distill_features"],
                teacher_features.detach(),
                quality=quality,
            )
            loss = (
                args.hard_weight * hard_loss
                + args.kd_weight * kd_loss
                + args.feat_weight * feat_loss
                + args.ranking_distill_weight * ranking_loss
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item())
            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                hard=f"{hard_loss.item():.4f}",
                kd=f"{kd_loss.item():.4f}",
                rank=f"{ranking_loss.item():.4f}",
                feat=f"{feat_loss.item():.4f}",
            )

        scheduler.step()
        train_loss = total_loss / max(len(train_loader), 1)
        val_true, val_prob = collect_outputs(student, val_loader, args.device, args.pin_memory, aggregation=args.eval_aggregation)
        if args.threshold_mode == "fixed":
            threshold = args.fixed_threshold
            val_metrics = evaluate_with_threshold(val_true, val_prob, threshold=threshold)
        else:
            threshold, val_metrics = select_threshold(val_true, val_prob, metric=args.monitor)
        test_true, test_prob = collect_outputs(student, test_loader, args.device, args.pin_memory, aggregation=args.eval_aggregation)
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
                "ranking_distill_weight": args.ranking_distill_weight,
                "threshold_mode": args.threshold_mode,
                "fixed_threshold": args.fixed_threshold,
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
            best_state = {key: value.detach().cpu() for key, value in student.state_dict().items()}
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
                best_state = {key: value.detach().cpu() for key, value in student.state_dict().items()}
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

    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state if best_state is not None else student.state_dict(), args.save_path)
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
    print(f"Saved distilled WESAD tiny checkpoint to {args.save_path}")


if __name__ == "__main__":
    main()
