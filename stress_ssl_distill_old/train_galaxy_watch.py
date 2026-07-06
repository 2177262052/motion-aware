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
from .galaxy_models import ResNet18WatchNet, ResNet34WatchNet, ResNet50WatchNet, WaveletGuidedWatchNet
from .losses import baseline_relative_margin_loss
from .metrics import classification_metrics
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
    dataset: GalaxyWatchWindowDataset,
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


def supervised_contrastive_loss(features: torch.Tensor, labels: torch.Tensor, quality: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    if features.shape[0] < 2:
        return features.new_tensor(0.0)

    logits = torch.matmul(features, features.T) / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    labels = labels.contiguous().view(-1, 1)
    positive_mask = torch.eq(labels, labels.T).to(features.device)
    self_mask = torch.eye(features.shape[0], device=features.device, dtype=torch.bool)
    positive_mask = positive_mask & (~self_mask)

    exp_logits = torch.exp(logits) * (~self_mask)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)
    pos_counts = positive_mask.sum(dim=1)
    valid = pos_counts > 0
    if not valid.any():
        return features.new_tensor(0.0)

    mean_log_prob_pos = (positive_mask.float() * log_prob).sum(dim=1) / pos_counts.clamp(min=1)
    anchor_weights = (0.35 + 0.65 * quality.squeeze(1).clamp(0.0, 1.0))[valid]
    loss = -(mean_log_prob_pos[valid] * anchor_weights).sum() / anchor_weights.sum().clamp(min=1e-6)
    return loss


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
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    pin_memory: bool,
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
            signal = batch["signal"].to(device, non_blocking=pin_memory)
            wavelet = batch["wavelet_features"].to(device, non_blocking=pin_memory)
            quality = batch["watch_quality"].to(device, non_blocking=pin_memory)
            labels = batch["label"].to(device, non_blocking=pin_memory).long()
            baseline_kwargs = {}
            if baseline_reference:
                baseline_kwargs = {
                    "baseline_signal": batch["baseline_signal"].to(device, non_blocking=pin_memory),
                    "baseline_wavelet_features": batch["baseline_wavelet_features"].to(device, non_blocking=pin_memory),
                    "baseline_quality": batch["baseline_watch_quality"].to(device, non_blocking=pin_memory),
                }

            logits = model(signal, wavelet, quality, **baseline_kwargs)["logits"]
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


def maybe_parse_sessions(values: Sequence[str] | None, fallback: Iterable[str]) -> list[str]:
    if values is None or len(values) == 0:
        return list(fallback)
    return [str(item) for item in values]


def set_random_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Galaxy Watch stress model.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--save-path", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--focal-gamma", type=float, default=1.5)
    parser.add_argument("--contrastive-weight", type=float, default=0.15)
    parser.add_argument("--wavelet-weight", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--baseline-relative-weight", type=float, default=0.0)
    parser.add_argument("--baseline-relative-margin", type=float, default=0.2)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--disable-ema-eval", action="store_true")
    parser.add_argument("--monitor", type=str, default="balanced_acc", choices=["acc", "balanced_acc", "f1", "auroc"])
    parser.add_argument("--threshold-metric", type=str, default="monitor", choices=["monitor", "acc", "balanced_acc", "f1", "auroc"])
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
    parser.add_argument("--watch-model-dim", type=int, default=192)
    parser.add_argument("--watch-transformer-layers", type=int, default=2)
    parser.add_argument("--watch-transformer-heads", type=int, default=4)
    parser.add_argument("--watch-fusion-hidden-dim", type=int, default=256)
    parser.add_argument("--watch-embed-dim", type=int, default=160)
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
    parser.add_argument(
        "--model-type",
        type=str,
        default="wavelet_guided",
        choices=["wavelet_guided", "resnet18_1d", "resnet34_1d", "resnet50_1d"],
    )
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
        baseline_reference=args.baseline_reference,
    )
    val_ds = GalaxyWatchWindowDataset(
        manifest_csv=args.manifest,
        split="val",
        dataset_root=args.dataset_root,
        include_sessions=include_sessions,
        cache_tables=True,
        wavelet=args.wavelet,
        wavelet_level=args.wavelet_level,
        baseline_reference=args.baseline_reference,
    )
    test_ds = GalaxyWatchWindowDataset(
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

    wavelet_guided_kwargs = {
        "model_dim": args.watch_model_dim,
        "transformer_layers": args.watch_transformer_layers,
        "transformer_heads": args.watch_transformer_heads,
        "fusion_hidden_dim": args.watch_fusion_hidden_dim,
        "embed_dim": args.watch_embed_dim,
        "watch_enhancement": args.watch_enhancement,
        "watch_motion_mode": args.watch_motion_mode,
    }

    if args.model_type == "wavelet_guided":
        model = WaveletGuidedWatchNet(**wavelet_guided_kwargs).to(args.device)
    elif args.model_type == "resnet18_1d":
        model = ResNet18WatchNet().to(args.device)
    elif args.model_type == "resnet34_1d":
        model = ResNet34WatchNet().to(args.device)
    elif args.model_type == "resnet50_1d":
        model = ResNet50WatchNet().to(args.device)
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")
    use_ema_eval = not args.disable_ema_eval
    ema_model = model
    if use_ema_eval:
        if args.model_type == "wavelet_guided":
            ema_model = WaveletGuidedWatchNet(**wavelet_guided_kwargs).to(args.device)
        elif args.model_type == "resnet18_1d":
            ema_model = ResNet18WatchNet().to(args.device)
        elif args.model_type == "resnet34_1d":
            ema_model = ResNet34WatchNet().to(args.device)
        else:
            ema_model = ResNet50WatchNet().to(args.device)
        ema_model.load_state_dict(model.state_dict())
        ema_model.eval()
        for param in ema_model.parameters():
            param.requires_grad_(False)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"model_params={total_params}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    class_counts = train_ds.manifest["label"].value_counts().sort_index()
    neg = float(class_counts.get(0, 1.0))
    pos = float(class_counts.get(1, 1.0))
    class_weights = torch.tensor([1.0, neg / max(pos, 1.0)], device=args.device, dtype=torch.float32)

    stopper = EarlyStopping(patience=args.early_stop_patience, mode="max", min_delta=args.min_delta)
    best_state = None
    best_test_metrics = None
    history_rows = []

    print(f"train windows={len(train_ds)} val windows={len(val_ds)} test windows={len(test_ds)}")
    if not has_val_split:
        print("selection_warning=val split missing; using test split for threshold/model selection")
    print(f"model_type={args.model_type}")
    if args.model_type == "wavelet_guided":
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
    print(
        f"loss_weights=contrastive:{args.contrastive_weight:.2f} "
        f"wavelet:{args.wavelet_weight:.2f} "
        f"baseline_relative:{args.baseline_relative_weight:.2f}"
    )
    print(
        f"selection_mode={args.selection_mode} "
        f"monitor={args.monitor} "
        f"early_stop_patience={args.early_stop_patience}"
    )
    print(f"baseline_reference={'on' if args.baseline_reference else 'off'}")
    print(f"subject_aware_batching={'on' if args.subject_aware_batching else 'off'}")
    print(f"eval_aggregation={args.eval_aggregation}")
    threshold_metric = args.monitor if args.threshold_metric == "monitor" else args.threshold_metric
    print(f"threshold_metric={threshold_metric}")
    print(f"ema_eval={'on' if use_ema_eval else 'off'} ema_decay={args.ema_decay:.4f} seed={args.seed}")
    best_threshold = 0.5

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        epoch_brel_subject_fracs: list[float] = []
        epoch_brel_sample_fracs: list[float] = []
        epoch_brel_active_batches = 0
        progress = tqdm(train_loader, desc=f"galaxy-watch epoch {epoch + 1}/{args.epochs}", leave=True)
        for batch in progress:
            signal = batch["signal"].to(args.device, non_blocking=args.pin_memory)
            wavelet = batch["wavelet_features"].to(args.device, non_blocking=args.pin_memory)
            quality = batch["watch_quality"].to(args.device, non_blocking=args.pin_memory)
            labels = batch["label"].to(args.device, non_blocking=args.pin_memory).long()
            baseline_kwargs = {}
            if args.baseline_reference:
                baseline_kwargs = {
                    "baseline_signal": batch["baseline_signal"].to(args.device, non_blocking=args.pin_memory),
                    "baseline_wavelet_features": batch["baseline_wavelet_features"].to(args.device, non_blocking=args.pin_memory),
                    "baseline_quality": batch["baseline_watch_quality"].to(args.device, non_blocking=args.pin_memory),
                }

            out = model(signal, wavelet, quality, **baseline_kwargs)
            cls_loss = quality_aware_focal_loss(
                out["logits"],
                labels,
                quality,
                class_weights=class_weights,
                gamma=args.focal_gamma,
                label_smoothing=args.label_smoothing,
            )
            con_loss = supervised_contrastive_loss(out["contrastive"], labels, quality, temperature=args.temperature)
            wav_loss = F.smooth_l1_loss(out["wavelet_pred"], wavelet)
            br_loss, br_stats = baseline_relative_margin_loss(
                out["logits"],
                labels,
                [str(item) for item in batch["subject_id"]],
                quality,
                margin=args.baseline_relative_margin,
                return_stats=True,
            )
            loss = (
                cls_loss
                + args.contrastive_weight * con_loss
                + args.wavelet_weight * wav_loss
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
            if use_ema_eval:
                update_ema_model(ema_model, model, args.ema_decay)
            total_loss += float(loss.item())
            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                cls=f"{cls_loss.item():.4f}",
                con=f"{con_loss.item():.4f}",
                wav=f"{wav_loss.item():.4f}",
                brel=f"{br_loss.item():.4f}",
                bsub=f"{int(br_stats['active_subjects'])}/{int(br_stats['available_subjects'])}",
            )

        scheduler.step()
        train_loss = total_loss / max(len(train_loader), 1)
        eval_model = ema_model if use_ema_eval else model
        val_true, val_prob = collect_outputs(
            eval_model,
            val_loader,
            args.device,
            args.pin_memory,
            aggregation=args.eval_aggregation,
            baseline_reference=args.baseline_reference,
        )
        threshold, val_metrics = select_threshold(val_true, val_prob, metric=threshold_metric)
        test_true, test_prob = collect_outputs(
            eval_model,
            test_loader,
            args.device,
            args.pin_memory,
            aggregation=args.eval_aggregation,
            baseline_reference=args.baseline_reference,
        )
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
                "val_acc": val_metrics["acc"],
                "val_balanced_acc": val_metrics["balanced_acc"],
                "val_f1": val_metrics["f1"],
                "val_auroc": val_metrics["auroc"],
                "val_threshold": val_metrics["threshold"],
                "val_positive_rate": val_metrics["positive_rate"],
                "threshold_metric": threshold_metric,
                "baseline_relative_weight": args.baseline_relative_weight,
                "baseline_relative_active_subject_fraction": float(np.mean(epoch_brel_subject_fracs)) if epoch_brel_subject_fracs else 0.0,
                "baseline_relative_active_sample_fraction": float(np.mean(epoch_brel_sample_fracs)) if epoch_brel_sample_fracs else 0.0,
                "baseline_relative_active_batch_fraction": float(epoch_brel_active_batches / max(len(train_loader), 1)),
                "test_acc": test_metrics["acc"],
                "test_balanced_acc": test_metrics["balanced_acc"],
                "test_f1": test_metrics["f1"],
                "test_auroc": test_metrics["auroc"],
                "test_positive_rate": test_metrics["positive_rate"],
            }
        )

        if args.selection_mode == "fixed_epoch" and epoch + 1 == args.selection_epoch:
            source_model = ema_model if use_ema_eval else model
            best_state = {key: value.detach().cpu() for key, value in source_model.state_dict().items()}
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
                source_model = ema_model if use_ema_eval else model
                best_state = {key: value.detach().cpu() for key, value in source_model.state_dict().items()}
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
    print(f"Saved Galaxy Watch model checkpoint to {args.save_path}")


if __name__ == "__main__":
    main()
