from __future__ import annotations

import argparse
import copy
import statistics
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .early_stopping import EarlyStopping
from .galaxy_dataset import GalaxyPrivilegedWindowDataset
from .galaxy_models import E4ReferenceEncoder
from .train_galaxy_privileged_elastic import masked_rhythm_loss
from .train_galaxy_watch import (
    aggregate_predictions,
    build_loader,
    evaluate_with_threshold,
    maybe_parse_sessions,
    quality_aware_focal_loss,
    select_threshold,
    set_random_seed,
    update_ema_model,
)
from .wesad_dataset import WESADPrivilegedWindowDataset
from .wesad_models_safe_sgpc import WESADChestEncoder


GALAXY_DEFAULT_CALM_SESSIONS = ["baseline"]
GALAXY_DEFAULT_STRESS_SESSIONS = ["tsst-prep"]
WESAD_DEFAULT_CALM_SESSIONS = ["baseline"]
WESAD_DEFAULT_STRESS_SESSIONS = ["stress"]


class GalaxyTeacherOnlyNet(nn.Module):
    """Privileged-only Galaxy teacher: E4 BVP/ACC with optional Polar rhythm supervision."""

    def __init__(self, embed_dim: int = 160, rhythm_dim: int = 3, use_rhythm_head: bool = True) -> None:
        super().__init__()
        self.encoder = E4ReferenceEncoder(out_dim=embed_dim)
        self.classifier = nn.Linear(embed_dim, 2)
        self.rhythm_head = (
            nn.Sequential(
                nn.Linear(embed_dim, 128),
                nn.GELU(),
                nn.Linear(128, 64),
                nn.GELU(),
                nn.Linear(64, rhythm_dim),
            )
            if use_rhythm_head
            else None
        )

    def forward(self, e4_signal: torch.Tensor) -> dict[str, torch.Tensor]:
        embedding = self.encoder(e4_signal)
        out = {
            "embedding": embedding,
            "logits": self.classifier(embedding),
        }
        if self.rhythm_head is not None:
            out["rhythm_pred"] = self.rhythm_head(embedding)
        return out


class WESADTeacherOnlyNet(nn.Module):
    """Privileged-only WESAD teacher: chest multimodal physiological streams only."""

    def __init__(self, privileged_channels: int, embed_dim: int = 160) -> None:
        super().__init__()
        self.encoder = WESADChestEncoder(in_channels=privileged_channels, out_dim=embed_dim)
        self.classifier = nn.Linear(embed_dim, 2)

    def forward(self, privileged_signal: torch.Tensor) -> dict[str, torch.Tensor]:
        embedding = self.encoder(privileged_signal)
        return {
            "embedding": embedding,
            "logits": self.classifier(embedding),
        }


def discover_manifests(manifests_dir: Path, subjects: list[str] | None, dataset_kind: str) -> list[tuple[str, Path]]:
    requested = {subject.strip() for subject in subjects or [] if subject.strip()}
    prefix = "galaxy" if dataset_kind == "galaxy" else "wesad"
    manifests: dict[str, Path] = {}
    for pattern in (f"{prefix}_*_loso_val.csv", "*_loso_val.csv"):
        for path in sorted(manifests_dir.glob(pattern)):
            subject = path.stem
            if subject.startswith(f"{prefix}_"):
                subject = subject[len(prefix) + 1 :]
            if subject.endswith("_loso_val"):
                subject = subject[: -len("_loso_val")]
            if requested and subject not in requested:
                continue
            manifests.setdefault(subject, path)
    return sorted(manifests.items())


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def collapse_flag(positive_rate: float) -> bool:
    return positive_rate <= 0.05 or positive_rate >= 0.95


def build_datasets(
    dataset_kind: str,
    manifest: Path,
    dataset_root: Path,
    include_sessions: list[str],
    cache_subjects: int,
) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, torch.utils.data.Dataset]:
    if dataset_kind == "galaxy":
        kwargs = {
            "manifest_csv": manifest,
            "dataset_root": dataset_root,
            "include_sessions": include_sessions,
        }
        return (
            GalaxyPrivilegedWindowDataset(split="train", **kwargs),
            GalaxyPrivilegedWindowDataset(split="val", **kwargs),
            GalaxyPrivilegedWindowDataset(split="test", **kwargs),
        )
    if dataset_kind == "wesad":
        kwargs = {
            "manifest_csv": manifest,
            "wesad_root": dataset_root,
            "include_sessions": include_sessions,
            "cache_subjects": cache_subjects,
        }
        return (
            WESADPrivilegedWindowDataset(split="train", **kwargs),
            WESADPrivilegedWindowDataset(split="val", **kwargs),
            WESADPrivilegedWindowDataset(split="test", **kwargs),
        )
    raise ValueError(f"Unsupported dataset kind: {dataset_kind}")


def teacher_logits_from_batch(
    model: nn.Module,
    batch: dict,
    dataset_kind: str,
    device: str,
    pin_memory: bool,
) -> dict[str, torch.Tensor]:
    if dataset_kind == "galaxy":
        e4_signal = batch["e4_signal"].to(device, non_blocking=pin_memory)
        return model(e4_signal)
    privileged_signal = batch["privileged_signal"].to(device, non_blocking=pin_memory)
    return model(privileged_signal)


def collect_teacher_outputs(
    model: nn.Module,
    loader: DataLoader,
    dataset_kind: str,
    device: str,
    pin_memory: bool,
    aggregation: str,
) -> tuple[list[int], list[float]]:
    model.eval()
    y_true: list[int] = []
    y_prob: list[float] = []
    subject_ids: list[str] = []
    sessions: list[str] = []
    with torch.no_grad():
        for batch in loader:
            labels = batch["label"].to(device, non_blocking=pin_memory).long()
            out = teacher_logits_from_batch(model, batch, dataset_kind, device, pin_memory)
            probs = torch.softmax(out["logits"], dim=1)[:, 1]
            y_true.extend(labels.detach().cpu().tolist())
            y_prob.extend(probs.detach().cpu().tolist())
            subject_ids.extend(str(item) for item in batch["subject_id"])
            sessions.extend(str(item) for item in batch["session"])
    return aggregate_predictions(y_true, y_prob, subject_ids, sessions, aggregation)


def evaluate_teacher(
    model: nn.Module,
    val_loader: DataLoader,
    test_loader: DataLoader,
    dataset_kind: str,
    device: str,
    pin_memory: bool,
    threshold_metric: str,
    threshold_mode: str,
    fixed_threshold: float,
    aggregation: str,
) -> tuple[float, dict[str, float], dict[str, float]]:
    val_true, val_prob = collect_teacher_outputs(
        model,
        val_loader,
        dataset_kind=dataset_kind,
        device=device,
        pin_memory=pin_memory,
        aggregation=aggregation,
    )
    if threshold_mode == "fixed":
        threshold = fixed_threshold
        val_metrics = evaluate_with_threshold(val_true, val_prob, threshold=threshold)
    else:
        threshold, val_metrics = select_threshold(val_true, val_prob, metric=threshold_metric)

    test_true, test_prob = collect_teacher_outputs(
        model,
        test_loader,
        dataset_kind=dataset_kind,
        device=device,
        pin_memory=pin_memory,
        aggregation=aggregation,
    )
    test_metrics = evaluate_with_threshold(test_true, test_prob, threshold=threshold)
    return threshold, val_metrics, test_metrics


def train_one_fold(
    args: argparse.Namespace,
    subject: str,
    manifest: Path,
    output_dir: Path,
    include_sessions: list[str],
) -> dict[str, object]:
    train_ds, val_ds, test_ds = build_datasets(
        args.dataset_kind,
        manifest=manifest,
        dataset_root=args.dataset_root,
        include_sessions=include_sessions,
        cache_subjects=args.cache_subjects,
    )
    if len(train_ds) == 0 or len(test_ds) == 0:
        raise ValueError(f"Empty train/test split for {subject}: train={len(train_ds)} test={len(test_ds)}")

    train_loader = build_loader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    val_loader = build_loader(
        val_ds if len(val_ds) > 0 else test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    test_loader = build_loader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    if args.dataset_kind == "galaxy":
        model: nn.Module = GalaxyTeacherOnlyNet(embed_dim=args.embed_dim, use_rhythm_head=args.rhythm_weight > 0.0)
        teacher_signal = "E4_BVP_ACC"
        auxiliary_signal = "Polar_rhythm_targets" if args.rhythm_weight > 0.0 else "none"
    else:
        sample = train_ds[0]
        privileged_channels = int(sample["privileged_signal"].shape[0])
        model = WESADTeacherOnlyNet(privileged_channels=privileged_channels, embed_dim=args.embed_dim)
        teacher_signal = "WESAD_chest_multimodal"
        auxiliary_signal = "none"

    model = model.to(args.device)
    ema_model = copy.deepcopy(model).to(args.device) if not args.disable_ema_eval else None
    if ema_model is not None:
        ema_model.eval()
        for param in ema_model.parameters():
            param.requires_grad_(False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    class_counts = train_ds.manifest["label"].value_counts().sort_index()
    neg = float(class_counts.get(0, 1.0))
    pos = float(class_counts.get(1, 1.0))
    class_weights = torch.tensor([1.0, neg / max(pos, 1.0)], device=args.device, dtype=torch.float32)

    stopper = EarlyStopping(patience=args.early_stop_patience, mode="max", min_delta=args.min_delta)
    best_state: dict[str, torch.Tensor] | None = None
    best_threshold = args.fixed_threshold
    best_test_metrics: dict[str, float] | None = None
    history_rows: list[dict[str, float | int | str]] = []
    threshold_metric = args.monitor if args.threshold_metric == "monitor" else args.threshold_metric

    metrics_dir = output_dir / "logs"
    ckpt_dir = output_dir / "checkpoints"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[{subject}] teacher_only dataset={args.dataset_kind} "
        f"teacher_signal={teacher_signal} auxiliary_signal={auxiliary_signal} "
        f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}"
    )

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        total_cls = 0.0
        total_rhythm = 0.0
        progress = tqdm(train_loader, desc=f"{args.dataset_kind}-teacher-only {subject} epoch {epoch + 1}/{args.epochs}", leave=False)
        for batch in progress:
            labels = batch["label"].to(args.device, non_blocking=args.pin_memory).long()
            quality = torch.ones((labels.shape[0], 1), device=args.device, dtype=torch.float32)
            out = teacher_logits_from_batch(model, batch, args.dataset_kind, args.device, args.pin_memory)
            cls_loss = quality_aware_focal_loss(
                out["logits"],
                labels,
                quality=quality,
                class_weights=class_weights,
                gamma=args.focal_gamma,
                label_smoothing=args.label_smoothing,
            )
            rhythm_loss = out["logits"].new_tensor(0.0)
            if args.dataset_kind == "galaxy" and args.rhythm_weight > 0.0 and "rhythm_pred" in out:
                polar_targets = batch["polar_targets"].to(args.device, non_blocking=args.pin_memory)
                polar_mask = batch["polar_target_mask"].to(args.device, non_blocking=args.pin_memory)
                rhythm_loss = masked_rhythm_loss(out["rhythm_pred"], polar_targets, polar_mask)

            loss = args.teacher_cls_weight * cls_loss + args.rhythm_weight * rhythm_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            if ema_model is not None:
                update_ema_model(ema_model, model, args.ema_decay)
            total_loss += float(loss.item())
            total_cls += float(cls_loss.item())
            total_rhythm += float(rhythm_loss.item())
            progress.set_postfix(loss=f"{loss.item():.4f}", cls=f"{cls_loss.item():.4f}", rhythm=f"{rhythm_loss.item():.4f}")

        scheduler.step()
        eval_model = ema_model if ema_model is not None else model
        threshold, val_metrics, test_metrics = evaluate_teacher(
            eval_model,
            val_loader,
            test_loader,
            dataset_kind=args.dataset_kind,
            device=args.device,
            pin_memory=args.pin_memory,
            threshold_metric=threshold_metric,
            threshold_mode=args.threshold_mode,
            fixed_threshold=args.fixed_threshold,
            aggregation=args.eval_aggregation,
        )
        train_loss = total_loss / max(len(train_loader), 1)
        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_cls_loss": total_cls / max(len(train_loader), 1),
            "train_rhythm_loss": total_rhythm / max(len(train_loader), 1),
            "val_acc": val_metrics["acc"],
            "val_balanced_acc": val_metrics["balanced_acc"],
            "val_f1": val_metrics["f1"],
            "val_auroc": val_metrics["auroc"],
            "val_threshold": threshold,
            "test_acc": test_metrics["acc"],
            "test_balanced_acc": test_metrics["balanced_acc"],
            "test_f1": test_metrics["f1"],
            "test_auroc": test_metrics["auroc"],
            "test_positive_rate": test_metrics["positive_rate"],
            "teacher_signal": teacher_signal,
            "auxiliary_signal": auxiliary_signal,
        }
        history_rows.append(row)
        score = val_metrics[args.monitor]
        if stopper.step(score, epoch + 1):
            best_state = {key: value.detach().cpu() for key, value in eval_model.state_dict().items()}
            best_threshold = threshold
            best_test_metrics = test_metrics
            print(
                f"[{subject}] new best teacher_{args.monitor}={score:.4f} epoch={epoch + 1} "
                f"threshold={threshold:.4f} test_ba={test_metrics['balanced_acc']:.4f} test_auroc={test_metrics['auroc']:.4f}"
            )
        if stopper.should_stop():
            print(f"[{subject}] early stopping at epoch={epoch + 1} best_epoch={stopper.best_epoch}")
            break

    pd.DataFrame(history_rows).to_csv(metrics_dir / f"{subject}_teacher_only_metrics.csv", index=False)
    if best_state is None:
        best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        if best_test_metrics is None:
            _, _, best_test_metrics = evaluate_teacher(
                ema_model if ema_model is not None else model,
                val_loader,
                test_loader,
                dataset_kind=args.dataset_kind,
                device=args.device,
                pin_memory=args.pin_memory,
                threshold_metric=threshold_metric,
                threshold_mode=args.threshold_mode,
                fixed_threshold=args.fixed_threshold,
                aggregation=args.eval_aggregation,
            )
    torch.save(best_state, ckpt_dir / f"{subject}_teacher_only.pt")

    assert best_test_metrics is not None
    test_prior = float(test_ds.manifest["label"].astype(float).mean())
    result = {
        "subject": subject,
        "model": "teacher_only",
        "dataset": args.dataset_kind,
        "teacher_signal": teacher_signal,
        "auxiliary_signal": auxiliary_signal,
        "best_epoch": stopper.best_epoch,
        "best_val_score": stopper.best_score,
        "best_threshold": best_threshold,
        "acc": best_test_metrics["acc"],
        "balanced_acc": best_test_metrics["balanced_acc"],
        "f1": best_test_metrics["f1"],
        "auroc": best_test_metrics["auroc"],
        "positive_rate": best_test_metrics["positive_rate"],
        "collapse": float(collapse_flag(best_test_metrics["positive_rate"])),
        "positive_rate_error": abs(best_test_metrics["positive_rate"] - test_prior),
        "test_positive_prior": test_prior,
    }
    return result


def write_summary(output_dir: Path, dataset_kind: str, rows: list[dict[str, object]]) -> None:
    metrics = ["balanced_acc", "auroc", "f1", "positive_rate_error"]
    lines = [
        f"# {dataset_kind.upper()} Teacher-Only Diagnosis",
        "",
        "This diagnostic trains and selects the privileged teacher without deployable student losses, KD, or adaptive correction.",
        "",
    ]
    for metric in metrics:
        values = [float(row[metric]) for row in rows]
        mean, std = mean_std(values)
        lines.append(f"teacher_only {metric}_mean={mean:.4f} {metric}_std={std:.4f}")
    collapse = float(np.mean([float(row["collapse"]) for row in rows])) if rows else float("nan")
    lines.append(f"teacher_only collapse_rate={collapse:.4f}")
    lines.append("")
    lines.append("```")
    lines.append(pd.DataFrame(rows).to_string(index=False))
    lines.append("```")
    (output_dir / f"{dataset_kind}_teacher_only_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run privileged teacher-only LOSO diagnosis without student training.")
    parser.add_argument("--dataset-kind", type=str, required=True, choices=["galaxy", "wesad"])
    parser.add_argument("--manifests-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", "--wesad-root", dest="dataset_root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--focal-gamma", type=float, default=1.5)
    parser.add_argument("--teacher-cls-weight", type=float, default=1.0)
    parser.add_argument("--rhythm-weight", type=float, default=0.15)
    parser.add_argument("--embed-dim", type=int, default=160)
    parser.add_argument("--monitor", type=str, default="auroc", choices=["acc", "balanced_acc", "f1", "auroc"])
    parser.add_argument("--threshold-metric", type=str, default="balanced_acc", choices=["monitor", "acc", "balanced_acc", "f1", "auroc"])
    parser.add_argument("--threshold-mode", type=str, default="val_search", choices=["val_search", "fixed"])
    parser.add_argument("--fixed-threshold", type=float, default=0.5)
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--eval-aggregation", type=str, default="window", choices=["window", "session_mean"])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--cache-subjects", type=int, default=4)
    parser.add_argument("--disable-ema-eval", action="store_true")
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calm-sessions", nargs="*", default=None)
    parser.add_argument("--stress-sessions", nargs="*", default=None)
    args = parser.parse_args()

    set_random_seed(args.seed)
    if args.dataset_kind == "galaxy":
        calm_sessions = maybe_parse_sessions(args.calm_sessions, GALAXY_DEFAULT_CALM_SESSIONS)
        stress_sessions = maybe_parse_sessions(args.stress_sessions, GALAXY_DEFAULT_STRESS_SESSIONS)
    else:
        calm_sessions = maybe_parse_sessions(args.calm_sessions, WESAD_DEFAULT_CALM_SESSIONS)
        stress_sessions = maybe_parse_sessions(args.stress_sessions, WESAD_DEFAULT_STRESS_SESSIONS)
    include_sessions = calm_sessions + stress_sessions
    threshold_metric = args.monitor if args.threshold_metric == "monitor" else args.threshold_metric
    args.threshold_metric = threshold_metric

    manifests = discover_manifests(args.manifests_dir, args.subjects, args.dataset_kind)
    if not manifests:
        raise ValueError(f"No manifests found in {args.manifests_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for subject, manifest in manifests:
        result = train_one_fold(args, subject, manifest, args.output_dir, include_sessions)
        rows.append(result)
        pd.DataFrame(rows).to_csv(args.output_dir / f"{args.dataset_kind}_teacher_only_results.csv", index=False)

    write_summary(args.output_dir, args.dataset_kind, rows)
    print(f"Saved teacher-only results to {args.output_dir / f'{args.dataset_kind}_teacher_only_results.csv'}")
    print(f"Saved teacher-only summary to {args.output_dir / f'{args.dataset_kind}_teacher_only_summary.md'}")


if __name__ == "__main__":
    main()
