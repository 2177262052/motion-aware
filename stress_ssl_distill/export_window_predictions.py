from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .analyze_motion_aware_mechanism import (
    DEFAULT_GALAXY_CALM_SESSIONS,
    DEFAULT_GALAXY_STRESS_SESSIONS,
    DEFAULT_WESAD_CALM_SESSIONS,
    DEFAULT_WESAD_STRESS_SESSIONS,
    build_dataset,
    build_model_from_state,
    load_state,
)


def maybe_parse_sessions(values: list[str] | None, fallback: list[str]) -> list[str]:
    if values is None or len(values) == 0:
        return list(fallback)
    return [str(item) for item in values]


def evaluate_with_threshold(labels: list[int], probs: list[float], threshold: float) -> dict[str, float]:
    preds = [1 if prob >= threshold else 0 for prob in probs]
    return {
        "acc": float(accuracy_score(labels, preds)),
        "balanced_acc": float(balanced_accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "auroc": float(roc_auc_score(labels, probs)),
    }


def select_threshold_local(labels: list[int], probs: list[float], metric: str) -> tuple[float, dict[str, float]]:
    if len(set(labels)) < 2:
        return 0.5, evaluate_with_threshold(labels, probs, 0.5)
    candidates = sorted(set([0.0, 1.0] + [round(float(prob), 6) for prob in probs]))
    best_threshold = 0.5
    best_metrics = evaluate_with_threshold(labels, probs, 0.5)
    best_score = best_metrics[metric]
    for threshold in candidates:
        metrics = evaluate_with_threshold(labels, probs, threshold)
        score = metrics[metric]
        if score > best_score + 1e-12:
            best_threshold = float(threshold)
            best_metrics = metrics
            best_score = score
    return best_threshold, best_metrics


def discover_manifests(manifests_dir: Path, dataset_kind: str, subjects: list[str] | None) -> list[tuple[str, Path]]:
    requested = {str(subject).strip() for subject in subjects or [] if str(subject).strip()}
    prefix = "galaxy" if dataset_kind == "galaxy" else "wesad"
    found: dict[str, Path] = {}
    for pattern in (f"{prefix}_*_loso_val.csv", "*_loso_val.csv"):
        for path in sorted(manifests_dir.glob(pattern)):
            subject = path.stem
            if subject.startswith(f"{prefix}_"):
                subject = subject[len(prefix) + 1 :]
            if subject.endswith("_loso_val"):
                subject = subject[: -len("_loso_val")]
            if dataset_kind == "galaxy" and not subject.upper().startswith("P"):
                continue
            if dataset_kind == "wesad" and not subject.upper().startswith("S"):
                continue
            if requested and subject not in requested:
                continue
            found.setdefault(subject, path)
    if not found:
        raise ValueError(f"No {dataset_kind} LOSO manifests found in {manifests_dir}")
    return sorted(found.items())


def find_checkpoint(checkpoint_dir: Path, subject: str) -> Path:
    roots = [checkpoint_dir]
    if (checkpoint_dir / "checkpoints").exists():
        roots.insert(0, checkpoint_dir / "checkpoints")
    candidates: list[Path] = []
    for root in roots:
        candidates.extend(
            [
                root / f"{subject}.pt",
                root / f"{subject.upper()}.pt",
                root / f"{subject.lower()}.pt",
                root / f"{subject}_watch_only.pt",
                root / f"{subject}_deploy_watch.pt",
                root / f"galaxy_{subject}.pt",
                root / f"wesad_{subject}.pt",
                root / f"galaxy_{subject}_watch_only.pt",
                root / f"wesad_{subject}_watch_only.pt",
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    for root in roots:
        matches = sorted(root.glob(f"*{subject}*.pt"))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Could not find checkpoint for {subject} under {checkpoint_dir}")


def build_loader(dataset: Any, batch_size: int, num_workers: int, pin_memory: bool) -> DataLoader:
    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **kwargs)


def batch_values(batch: dict[str, Any], key: str, n: int, default: Any = "") -> list[Any]:
    if key not in batch:
        return [default for _ in range(n)]
    value = batch[key]
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        values = list(value)
    else:
        values = [value]
    if len(values) == n:
        return values
    if len(values) == 1:
        return values * n
    return (values + [default for _ in range(n)])[:n]


def threshold_from_results(results_csv: Path, subject: str, result_model: str | None) -> float:
    df = pd.read_csv(results_csv)
    subject_col = next((col for col in ("subject", "subject_id", "heldout_subject", "test_subject") if col in df.columns), None)
    if subject_col is None:
        raise ValueError(f"{results_csv} has no subject column.")
    sub = df[df[subject_col].astype(str) == str(subject)].copy()
    if sub.empty:
        raise ValueError(f"{results_csv} has no threshold row for subject {subject}.")

    if result_model is not None and "model" in sub.columns:
        filtered = sub[sub["model"].astype(str) == str(result_model)].copy()
        if filtered.empty:
            raise ValueError(f"{results_csv} has subject {subject}, but no model={result_model}.")
        sub = filtered

    candidates: list[str] = []
    if result_model:
        candidates.append(f"{result_model}_threshold")
    candidates.extend(["threshold", "deploy_watch_threshold", "watch_only_threshold", "val_threshold"])
    threshold_col = next((col for col in candidates if col in sub.columns), None)
    if threshold_col is None:
        raise ValueError(f"{results_csv} has no threshold column among {candidates}.")

    values = pd.to_numeric(sub[threshold_col], errors="coerce").dropna().unique()
    if len(values) != 1:
        raise ValueError(f"{results_csv} has conflicting thresholds for subject {subject}: {values}")
    value = float(values[0])
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"Threshold outside [0,1] for subject {subject}: {value}")
    return value


def collect_probs(model: torch.nn.Module, loader: DataLoader, args: argparse.Namespace) -> tuple[list[int], list[float]]:
    labels_all: list[int] = []
    probs_all: list[float] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            signal = batch["signal"].to(args.device, non_blocking=args.pin_memory)
            wavelet = batch["wavelet_features"].to(args.device, non_blocking=args.pin_memory)
            quality = batch["watch_quality"].to(args.device, non_blocking=args.pin_memory)
            logits = model(signal, wavelet, quality)["logits"]
            probs = torch.softmax(logits, dim=1)[:, 1]
            labels_all.extend(batch["label"].detach().cpu().long().tolist())
            probs_all.extend(probs.detach().cpu().float().tolist())
    return labels_all, probs_all


def export_fold(
    subject: str,
    manifest_path: Path,
    checkpoint_path: Path,
    include_sessions: list[str],
    threshold: float | None,
    args: argparse.Namespace,
) -> pd.DataFrame:
    state = load_state(checkpoint_path)
    model = build_model_from_state(
        state,
        device=args.device,
        model_dim=args.watch_model_dim,
        transformer_layers=args.watch_transformer_layers,
        transformer_heads=args.watch_transformer_heads,
        fusion_hidden_dim=args.watch_fusion_hidden_dim,
        embed_dim=args.watch_embed_dim,
        input_ablation=args.input_ablation,
    )

    if threshold is None:
        val_ds = build_dataset(
            args.dataset_kind,
            manifest_path,
            "val",
            args.dataset_root,
            include_sessions,
            args.cache_subjects,
        )
        val_loader = build_loader(val_ds, args.batch_size, args.num_workers, args.pin_memory)
        val_labels, val_probs = collect_probs(model, val_loader, args)
        if len(set(val_labels)) < 2:
            threshold = 0.5
        else:
            threshold, _ = select_threshold_local(val_labels, val_probs, metric=args.threshold_metric)

    test_ds = build_dataset(
        args.dataset_kind,
        manifest_path,
        "test",
        args.dataset_root,
        include_sessions,
        args.cache_subjects,
    )
    test_loader = build_loader(test_ds, args.batch_size, args.num_workers, args.pin_memory)

    rows: list[dict[str, Any]] = []
    row_order = 0
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"export:{args.dataset_kind}:{args.method}:{subject}", leave=False):
            signal = batch["signal"].to(args.device, non_blocking=args.pin_memory)
            wavelet = batch["wavelet_features"].to(args.device, non_blocking=args.pin_memory)
            quality = batch["watch_quality"].to(args.device, non_blocking=args.pin_memory)
            logits = model(signal, wavelet, quality)["logits"]
            probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            labels = batch["label"].detach().cpu().long().numpy()
            n = int(len(labels))
            subject_ids = [str(x) for x in batch_values(batch, "subject_id", n, subject)]
            sessions = [str(x) for x in batch_values(batch, "session", n, "")]
            starts = batch_values(batch, "window_start_ms", n, -1)
            ends = batch_values(batch, "window_end_ms", n, -1)
            qualities = batch["watch_quality"].detach().cpu().float().reshape(n, -1)[:, 0].numpy()
            for idx in range(n):
                start = int(starts[idx])
                end = int(ends[idx])
                subject_id = subject_ids[idx]
                session = sessions[idx]
                window_id = f"{subject_id}|{session}|{start}|{end}"
                prob = float(probs[idx])
                pred = int(prob >= float(threshold))
                rows.append(
                    {
                        "dataset": args.dataset_kind,
                        "group": args.group,
                        "method": args.method,
                        "fold": subject,
                        "subject_id": subject_id,
                        "session": session,
                        "window_start_ms": start,
                        "window_end_ms": end,
                        "window_id": window_id,
                        "row_order": row_order,
                        "label": int(labels[idx]),
                        "prob": prob,
                        "threshold": float(threshold),
                        "pred": pred,
                        "watch_quality": float(qualities[idx]),
                        "checkpoint": str(checkpoint_path),
                        "manifest": str(manifest_path),
                    }
                )
                row_order += 1
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export window-level predictions from existing LOSO checkpoints.")
    parser.add_argument("--dataset-kind", choices=["galaxy", "wesad"], required=True)
    parser.add_argument("--manifests-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--group", type=str, default="main")
    parser.add_argument("--results-csv", type=Path, default=None)
    parser.add_argument("--result-model", type=str, default=None, help="e.g. watch_only or deploy_watch for long result CSVs.")
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--threshold-metric", choices=["acc", "balanced_acc", "f1", "auroc"], default="balanced_acc")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cache-subjects", type=int, default=4)
    parser.add_argument("--calm-sessions", nargs="*", default=None)
    parser.add_argument("--stress-sessions", nargs="*", default=None)
    parser.add_argument(
        "--input-ablation",
        choices=[
            "none",
            "acc_only",
            "ppg_only",
            "simple_concat",
            "ppg_only_refine",
            "simple_concat_refine",
            "gated_fusion",
            "gated_fusion_refine",
        ],
        default="none",
    )
    parser.add_argument("--watch-model-dim", type=int, default=192)
    parser.add_argument("--watch-transformer-layers", type=int, default=2)
    parser.add_argument("--watch-transformer-heads", type=int, default=4)
    parser.add_argument("--watch-fusion-hidden-dim", type=int, default=256)
    parser.add_argument("--watch-embed-dim", type=int, default=160)
    args = parser.parse_args()

    if args.dataset_kind == "galaxy":
        calm = maybe_parse_sessions(args.calm_sessions, DEFAULT_GALAXY_CALM_SESSIONS)
        stress = maybe_parse_sessions(args.stress_sessions, DEFAULT_GALAXY_STRESS_SESSIONS)
    else:
        calm = maybe_parse_sessions(args.calm_sessions, DEFAULT_WESAD_CALM_SESSIONS)
        stress = maybe_parse_sessions(args.stress_sessions, DEFAULT_WESAD_STRESS_SESSIONS)
    include_sessions = calm + stress

    frames: list[pd.DataFrame] = []
    for subject, manifest_path in discover_manifests(args.manifests_dir, args.dataset_kind, args.subjects):
        checkpoint_path = find_checkpoint(args.checkpoint_dir, subject)
        threshold = (
            threshold_from_results(args.results_csv, subject, args.result_model)
            if args.results_csv is not None
            else None
        )
        frames.append(
            export_fold(
                subject=subject,
                manifest_path=manifest_path,
                checkpoint_path=checkpoint_path,
                include_sessions=include_sessions,
                threshold=threshold,
                args=args,
            )
        )

    out = pd.concat(frames, ignore_index=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    print(f"saved_window_predictions={args.output_csv}")
    print(f"rows={len(out)} subjects={out['subject_id'].nunique()} method={args.method}")


if __name__ == "__main__":
    main()
