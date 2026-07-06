from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .early_stopping import EarlyStopping
from .galaxy_dataset import GalaxyWatchWindowDataset
from .galaxy_models import PrivilegedGalaxyTeacherNet, TinyWaveletDistillNet, WaveletGuidedWatchNet
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


def build_loader(
    dataset: GalaxyWatchWindowDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
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


def set_random_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def distillation_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    quality: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
    kl = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=1)
    weights = 0.35 + 0.65 * quality.squeeze(1).clamp(0.0, 1.0)
    return (kl * weights).sum() / weights.sum().clamp(min=1e-6) * (temperature ** 2)


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


def feature_alignment_loss(
    student_features: torch.Tensor,
    teacher_features: torch.Tensor,
    quality: torch.Tensor,
) -> torch.Tensor:
    if student_features.shape != teacher_features.shape:
        raise ValueError(
            "Student/teacher feature shapes do not match for distillation: "
            f"{tuple(student_features.shape)} vs {tuple(teacher_features.shape)}"
        )
    per_sample = F.smooth_l1_loss(student_features, teacher_features, reduction="none").mean(dim=1)
    weights = 0.35 + 0.65 * quality.squeeze(1).clamp(0.0, 1.0)
    return (per_sample * weights).sum() / weights.sum().clamp(min=1e-6)


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


def collect_outputs(
    model: TinyWaveletDistillNet,
    loader: DataLoader,
    device: str,
    pin_memory: bool,
    aggregation: str = "window",
) -> tuple[list[int], list[float]]:
    model.eval()
    y_true = []
    y_prob = []
    subject_ids: list[str] = []
    sessions: list[str] = []
    with torch.no_grad():
        for batch in loader:
            signal = batch["signal"].to(device, non_blocking=pin_memory)
            wavelet = batch["wavelet_features"].to(device, non_blocking=pin_memory)
            quality = batch["watch_quality"].to(device, non_blocking=pin_memory)
            labels = batch["label"].to(device, non_blocking=pin_memory).long()

            logits = model(signal, wavelet, quality)["logits"]
            probs = torch.softmax(logits, dim=1)[:, 1]

            y_true.extend(labels.detach().cpu().tolist())
            y_prob.extend(probs.detach().cpu().tolist())
            subject_ids.extend(str(item) for item in batch["subject_id"])
            sessions.extend(str(item) for item in batch["session"])
    return aggregate_predictions(y_true, y_prob, subject_ids, sessions, aggregation)


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


def load_state_dict_flexible(model: torch.nn.Module, checkpoint_path: Path, device: str) -> None:
    state = torch.load(checkpoint_path, map_location=device)
    result = model.load_state_dict(state, strict=False)
    optional_prefixes = (
        "reliability_head.",
        "watch_projector.",
        "e4_projector.",
        "e4_classifier.",
        "rhythm_head.",
        "teacher_rhythm_head.",
        "wavelet_predictor.",
        "teacher_fused_classifier.",
        "deploy_correction.",
        "deploy_correction_gate.",
        "privileged_correction.",
        "privileged_correction_gate.",
        "correction_norm.",
        "correction_scale",
        "watch_encoder.ppg_enhancer.",
        "ppg_enhancer.",
    )
    ignored_missing = [key for key in result.missing_keys if key.startswith(optional_prefixes)]
    remaining_missing = [key for key in result.missing_keys if key not in ignored_missing]
    if remaining_missing:
        print(f"teacher_load_missing_keys={remaining_missing}")
    if ignored_missing:
        print(f"teacher_load_ignored_missing_keys={ignored_missing}")
    ignored_unexpected = [key for key in result.unexpected_keys if key.startswith(optional_prefixes)]
    remaining_unexpected = [key for key in result.unexpected_keys if key not in ignored_unexpected]
    if ignored_unexpected:
        print(f"teacher_load_ignored_unexpected_keys={ignored_unexpected}")
    if remaining_unexpected:
        print(f"teacher_load_unexpected_keys={remaining_unexpected}")


def build_teacher(
    teacher_kind: str,
    teacher_path: Path,
    device: str,
    teacher_watch_enhancement: str = "none",
) -> torch.nn.Module:
    if teacher_kind == "deploy_watch":
        teacher = PrivilegedGalaxyTeacherNet(watch_enhancement=teacher_watch_enhancement).to(device)
    elif teacher_kind == "watch_only":
        teacher = WaveletGuidedWatchNet(watch_enhancement=teacher_watch_enhancement).to(device)
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
        out = teacher.forward_watch(signal, wavelet, quality, return_aux=False)
        return out["logits"], out["watch_embedding"]
    out = teacher(signal, wavelet, quality)
    return out["logits"], out["embedding"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Distill a lightweight Galaxy watch-only student from a stronger watch teacher.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--teacher-path", type=Path, required=True)
    parser.add_argument("--teacher-kind", type=str, default="deploy_watch", choices=["deploy_watch", "watch_only"])
    parser.add_argument(
        "--teacher-watch-enhancement",
        type=str,
        default="none",
        choices=["none", "motion_disentangled"],
    )
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
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--calm-sessions", nargs="*", default=None)
    parser.add_argument("--stress-sessions", nargs="*", default=None)
    parser.add_argument("--wavelet", type=str, default="db4")
    parser.add_argument("--wavelet-level", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_random_seed(args.seed)
    calm_sessions = maybe_parse_sessions(args.calm_sessions, DEFAULT_CALM_SESSIONS)
    stress_sessions = maybe_parse_sessions(args.stress_sessions, DEFAULT_STRESS_SESSIONS)
    include_sessions = calm_sessions + stress_sessions

    train_ds = GalaxyWatchWindowDataset(
        manifest_csv=args.manifest,
        split="train",
        dataset_root=args.dataset_root,
        include_sessions=include_sessions,
        cache_tables=True,
        wavelet=args.wavelet,
        wavelet_level=args.wavelet_level,
    )
    val_ds = GalaxyWatchWindowDataset(
        manifest_csv=args.manifest,
        split="val",
        dataset_root=args.dataset_root,
        include_sessions=include_sessions,
        cache_tables=True,
        wavelet=args.wavelet,
        wavelet_level=args.wavelet_level,
    )
    test_ds = GalaxyWatchWindowDataset(
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

    teacher = build_teacher(
        args.teacher_kind,
        args.teacher_path,
        args.device,
        teacher_watch_enhancement=args.teacher_watch_enhancement,
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
    print(f"teacher_kind={args.teacher_kind}")
    print(f"calm_sessions={calm_sessions}")
    print(f"stress_sessions={stress_sessions}")
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
    print(f"eval_aggregation={args.eval_aggregation} seed={args.seed}")

    for epoch in range(args.epochs):
        student.train()
        total_loss = 0.0
        progress = tqdm(train_loader, desc=f"galaxy-tiny epoch {epoch + 1}/{args.epochs}", leave=True)
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
    print(f"Saved distilled Galaxy tiny checkpoint to {args.save_path}")


if __name__ == "__main__":
    main()
