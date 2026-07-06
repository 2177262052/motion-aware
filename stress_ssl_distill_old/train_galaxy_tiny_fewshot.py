from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from .early_stopping import EarlyStopping
from .galaxy_dataset import GalaxyWatchWindowDataset
from .galaxy_models import TinyWaveletDistillNet
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
    dataset,
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


def sample_support_query_indices(
    dataset: GalaxyWatchWindowDataset,
    shots_per_class: int,
    seed: int,
) -> tuple[list[int], list[int]]:
    by_class: dict[int, list[int]] = {0: [], 1: []}
    for idx in range(len(dataset)):
        label = int(dataset.manifest.iloc[idx]["label"])
        by_class[label].append(idx)

    rng = np.random.default_rng(seed)
    support_indices: list[int] = []
    for label, indices in by_class.items():
        if len(indices) < shots_per_class + 1:
            raise ValueError(
                f"Not enough windows for class {label}: need at least {shots_per_class + 1}, got {len(indices)}"
            )
        picked = rng.choice(indices, size=shots_per_class, replace=False).tolist()
        support_indices.extend(int(item) for item in picked)

    support_set = set(support_indices)
    query_indices = [idx for idx in range(len(dataset)) if idx not in support_set]
    if not query_indices:
        raise ValueError("Few-shot query set is empty after sampling support windows.")
    return sorted(support_indices), query_indices


def load_student(student_path: Path, dataset: GalaxyWatchWindowDataset, device: str) -> TinyWaveletDistillNet:
    sample = dataset[0]
    signal_channels = int(sample["signal"].shape[0])
    wavelet_dim = int(sample["wavelet_features"].shape[0])
    model = TinyWaveletDistillNet(in_channels=signal_channels, wavelet_dim=wavelet_dim).to(device)
    state = torch.load(student_path, map_location=device)
    result = model.load_state_dict(state, strict=False)
    if result.missing_keys:
        print(f"student_load_missing_keys={result.missing_keys}")
    if result.unexpected_keys:
        print(f"student_load_unexpected_keys={result.unexpected_keys}")
    return model


def freeze_for_head_only(model: TinyWaveletDistillNet) -> None:
    for name, param in model.named_parameters():
        param.requires_grad_(name.startswith("classifier"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Few-shot personalization for Galaxy tiny student on the held-out subject.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--student-path", type=Path, required=True)
    parser.add_argument("--save-path", type=Path, default=None)
    parser.add_argument("--metrics-path", type=Path, default=None)
    parser.add_argument("--shots-per-class", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--selection-mode", type=str, default="early_stop", choices=["fixed_epoch", "early_stop"])
    parser.add_argument("--selection-epoch", type=int, default=10)
    parser.add_argument("--monitor", type=str, default="balanced_acc", choices=["acc", "balanced_acc", "f1", "auroc"])
    parser.add_argument("--early-stop-patience", type=int, default=8)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--eval-aggregation", type=str, default="window", choices=["window", "session_mean"])
    parser.add_argument("--threshold-source", type=str, default="support", choices=["support", "fixed_0.5"])
    parser.add_argument("--full-finetune", action="store_true")
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

    test_ds = GalaxyWatchWindowDataset(
        manifest_csv=args.manifest,
        split="test",
        dataset_root=args.dataset_root,
        include_sessions=include_sessions,
        cache_tables=True,
        wavelet=args.wavelet,
        wavelet_level=args.wavelet_level,
    )
    if len(test_ds) == 0:
        raise ValueError("Test split is empty after session filtering.")

    model = load_student(args.student_path, test_ds, args.device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"tiny_student_params={total_params}")

    if args.full_finetune:
        print("fewshot_mode=full_finetune")
    else:
        freeze_for_head_only(model)
        print("fewshot_mode=head_only")

    support_indices, query_indices = sample_support_query_indices(test_ds, args.shots_per_class, args.seed)
    support_subset = Subset(test_ds, support_indices)
    query_subset = Subset(test_ds, query_indices)
    support_loader = build_loader(support_subset, args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=args.pin_memory)
    support_eval_loader = build_loader(support_subset, args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)
    query_loader = build_loader(query_subset, args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters left for few-shot adaptation.")
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    stopper = EarlyStopping(patience=args.early_stop_patience, mode="max", min_delta=args.min_delta)

    print(f"support_size={len(support_subset)} query_size={len(query_subset)} shots_per_class={args.shots_per_class}")
    print(f"threshold_source={args.threshold_source} eval_aggregation={args.eval_aggregation} seed={args.seed}")

    best_state = None
    best_support_metrics = None
    best_query_metrics = None
    best_threshold = 0.5
    history_rows: list[dict[str, float | int | str]] = []

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        progress = tqdm(support_loader, desc=f"galaxy-tiny-fewshot epoch {epoch + 1}/{args.epochs}", leave=True)
        for batch in progress:
            signal = batch["signal"].to(args.device, non_blocking=args.pin_memory)
            wavelet = batch["wavelet_features"].to(args.device, non_blocking=args.pin_memory)
            quality = batch["watch_quality"].to(args.device, non_blocking=args.pin_memory)
            labels = batch["label"].to(args.device, non_blocking=args.pin_memory).long()

            logits = model(signal, wavelet, quality)["logits"]
            loss = F.cross_entropy(logits, labels, label_smoothing=args.label_smoothing)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            total_loss += float(loss.item())
            progress.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = total_loss / max(len(support_loader), 1)
        support_true, support_prob = collect_outputs(
            model,
            support_eval_loader,
            args.device,
            args.pin_memory,
            aggregation=args.eval_aggregation,
        )
        if args.threshold_source == "support":
            threshold, support_metrics = select_threshold(support_true, support_prob, metric=args.monitor)
        else:
            threshold = 0.5
            support_metrics = evaluate_with_threshold(support_true, support_prob, threshold=threshold)

        query_true, query_prob = collect_outputs(
            model,
            query_loader,
            args.device,
            args.pin_memory,
            aggregation=args.eval_aggregation,
        )
        query_metrics = evaluate_with_threshold(query_true, query_prob, threshold=threshold)

        print(
            f"epoch={epoch + 1} "
            f"fewshot_loss={train_loss:.4f} "
            f"support_balanced_acc={support_metrics['balanced_acc']:.4f} "
            f"support_auroc={support_metrics['auroc']:.4f} "
            f"support_threshold={threshold:.4f} "
            f"query_balanced_acc={query_metrics['balanced_acc']:.4f} "
            f"query_auroc={query_metrics['auroc']:.4f}"
        )

        history_rows.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "support_acc": support_metrics["acc"],
                "support_balanced_acc": support_metrics["balanced_acc"],
                "support_f1": support_metrics["f1"],
                "support_auroc": support_metrics["auroc"],
                "support_threshold": threshold,
                "support_positive_rate": support_metrics["positive_rate"],
                "query_acc": query_metrics["acc"],
                "query_balanced_acc": query_metrics["balanced_acc"],
                "query_f1": query_metrics["f1"],
                "query_auroc": query_metrics["auroc"],
                "query_positive_rate": query_metrics["positive_rate"],
            }
        )

        if args.selection_mode == "fixed_epoch" and epoch + 1 == args.selection_epoch:
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            best_support_metrics = support_metrics
            best_query_metrics = query_metrics
            best_threshold = threshold
            print(
                f"selected fixed epoch {epoch + 1} "
                f"| query_balanced_acc={best_query_metrics['balanced_acc']:.4f} "
                f"query_auroc={best_query_metrics['auroc']:.4f}"
            )

        if args.selection_mode == "early_stop":
            score = support_metrics[args.monitor]
            improved = stopper.step(score, epoch + 1)
            if improved:
                best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
                best_support_metrics = support_metrics
                best_query_metrics = query_metrics
                best_threshold = threshold
                print(
                    f"new best support_{args.monitor}={score:.4f} at epoch {epoch + 1} "
                    f"| query_balanced_acc={best_query_metrics['balanced_acc']:.4f} "
                    f"query_auroc={best_query_metrics['auroc']:.4f}"
                )

            if stopper.should_stop():
                print(
                    f"early stopping triggered: no improvement in {args.early_stop_patience} epochs; "
                    f"best_support_{args.monitor}={stopper.best_score:.4f} at epoch {stopper.best_epoch}"
                )
                break

    if args.save_path is not None:
        args.save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_state if best_state is not None else model.state_dict(), args.save_path)
        print(f"Saved few-shot checkpoint to {args.save_path}")

    if args.metrics_path is not None:
        args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(history_rows).to_csv(args.metrics_path, index=False)
        print(f"Saved epoch metrics to {args.metrics_path}")

    if args.selection_mode == "early_stop":
        print(f"best_support_{args.monitor}={stopper.best_score:.4f} at epoch {stopper.best_epoch}")
    else:
        print(f"selected_epoch={args.selection_epoch}")

    if best_query_metrics is not None:
        print(
            f"best_threshold={best_threshold:.4f} "
            f"best_query_acc={best_query_metrics['acc']:.4f} "
            f"best_query_balanced_acc={best_query_metrics['balanced_acc']:.4f} "
            f"best_query_f1={best_query_metrics['f1']:.4f} "
            f"best_query_auroc={best_query_metrics['auroc']:.4f} "
            f"best_query_positive_rate={best_query_metrics['positive_rate']:.4f}"
        )


if __name__ == "__main__":
    main()
