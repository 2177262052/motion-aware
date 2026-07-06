from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .analyze_motion_aware_mechanism import (
    DEFAULT_GALAXY_CALM_SESSIONS,
    DEFAULT_GALAXY_STRESS_SESSIONS,
    DEFAULT_WESAD_CALM_SESSIONS,
    DEFAULT_WESAD_STRESS_SESSIONS,
    build_dataset,
    build_loader,
    build_model_from_state,
    evaluate_with_threshold_local,
    infer_watch_enhancement,
    load_state,
    select_threshold_local,
    uses_privileged_prefix,
    uses_scaled_motion,
)
from .run_galaxy_loso_eval_elastic import collapse_flag


def maybe_parse_sessions(values: list[str] | None, fallback: list[str]) -> list[str]:
    if values is None or len(values) == 0:
        return list(fallback)
    return [str(item) for item in values]


def batch_values(batch: dict[str, Any], key: str, n: int, default: Any = "") -> list[Any]:
    if key not in batch:
        return [default for _ in range(n)]
    value = batch[key]
    if torch.is_tensor(value):
        values = value.detach().cpu().tolist()
    elif isinstance(value, np.ndarray):
        values = value.tolist()
    elif isinstance(value, (list, tuple)):
        values = list(value)
    else:
        values = [value]
    if len(values) == n:
        return values
    if len(values) == 1:
        return values * n
    return (values + [default for _ in range(n)])[:n]


def collect_predictions(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    *,
    device: str,
    pin_memory: bool,
    split: str,
    fold: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    model.eval()
    row_order = 0
    with torch.no_grad():
        for batch in loader:
            signal = batch["signal"].to(device, non_blocking=pin_memory)
            wavelet = batch["wavelet_features"].to(device, non_blocking=pin_memory)
            quality = batch["watch_quality"].to(device, non_blocking=pin_memory)
            logits = model(signal, wavelet, quality)["logits"]
            probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().float().numpy()
            labels = batch["label"].detach().cpu().long().numpy()
            n = int(len(labels))
            subject_ids = [str(item) for item in batch_values(batch, "subject_id", n, fold)]
            sessions = [str(item) for item in batch_values(batch, "session", n, "")]
            starts = batch_values(batch, "window_start_ms", n, -1)
            ends = batch_values(batch, "window_end_ms", n, -1)
            qualities = batch["watch_quality"].detach().cpu().float().reshape(n, -1)[:, 0].numpy()
            for idx in range(n):
                start = int(starts[idx])
                end = int(ends[idx])
                rows.append(
                    {
                        "split": split,
                        "fold": fold,
                        "subject_id": subject_ids[idx],
                        "session": sessions[idx],
                        "window_start_ms": start,
                        "window_end_ms": end,
                        "window_id": f"{subject_ids[idx]}|{sessions[idx]}|{start}|{end}",
                        "row_order": row_order,
                        "label": int(labels[idx]),
                        "prob": float(probs[idx]),
                        "watch_quality": float(qualities[idx]),
                    }
                )
                row_order += 1
    return pd.DataFrame(rows)


def print_metrics(name: str, metrics: dict[str, float], positive_prior: float) -> None:
    positive_rate = float(metrics.get("positive_rate", 0.0))
    positive_rate_error = abs(positive_rate - positive_prior)
    print(
        f"{name} "
        f"threshold={metrics['threshold']:.6f} "
        f"acc={metrics['acc']:.4f} "
        f"balanced_acc={metrics['balanced_acc']:.4f} "
        f"f1={metrics['f1']:.4f} "
        f"auroc={metrics['auroc']:.4f} "
        f"positive_rate={positive_rate:.4f} "
        f"collapse={collapse_flag(positive_rate)} "
        f"positive_rate_error={positive_rate_error:.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deploy-watch inference for one checkpoint and one LOSO manifest.")
    parser.add_argument("--dataset-kind", choices=["galaxy", "wesad"], default="galaxy")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/single_checkpoint_inference"))
    parser.add_argument("--fold", type=str, default=None)
    parser.add_argument("--threshold-metric", choices=["acc", "balanced_acc", "f1", "auroc"], default="balanced_acc")
    parser.add_argument("--fixed-threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cache-subjects", type=int, default=4)
    parser.add_argument("--calm-sessions", nargs="*", default=None)
    parser.add_argument("--stress-sessions", nargs="*", default=None)
    parser.add_argument("--input-ablation", type=str, default="none")
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

    fold = args.fold
    if fold is None:
        stem = args.manifest.stem
        prefix = "galaxy_" if args.dataset_kind == "galaxy" else "wesad_"
        if stem.startswith(prefix):
            stem = stem[len(prefix) :]
        fold = stem.replace("_loso_val", "")

    state = load_state(args.checkpoint)
    print(f"checkpoint={args.checkpoint}")
    print(f"state_tensors={len(state)} state_numel={sum(v.numel() for v in state.values())}")
    print(f"uses_privileged_prefix={int(uses_privileged_prefix(state))}")
    print(f"uses_scaled_motion={int(uses_scaled_motion(state))}")
    print(f"watch_enhancement={infer_watch_enhancement(state)}")

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
    print(f"loaded_model={model.__class__.__name__}")
    print(f"deploy_params={sum(p.numel() for p in model.parameters())}")

    val_ds = build_dataset(args.dataset_kind, args.manifest, "val", args.dataset_root, include_sessions, args.cache_subjects)
    test_ds = build_dataset(args.dataset_kind, args.manifest, "test", args.dataset_root, include_sessions, args.cache_subjects)
    print(f"val_windows={len(val_ds)} test_windows={len(test_ds)} sessions={include_sessions}")
    val_loader = build_loader(val_ds, args.batch_size, args.num_workers, args.pin_memory)
    test_loader = build_loader(test_ds, args.batch_size, args.num_workers, args.pin_memory)

    val_frame = collect_predictions(model, val_loader, device=args.device, pin_memory=args.pin_memory, split="val", fold=fold)
    test_frame = collect_predictions(model, test_loader, device=args.device, pin_memory=args.pin_memory, split="test", fold=fold)

    val_labels = val_frame["label"].astype(int).tolist()
    val_probs = val_frame["prob"].astype(float).tolist()
    test_labels = test_frame["label"].astype(int).tolist()
    test_probs = test_frame["prob"].astype(float).tolist()
    positive_prior = float(np.mean(test_labels)) if test_labels else 0.0

    val_threshold, val_metrics = select_threshold_local(val_labels, val_probs, metric=args.threshold_metric)
    fixed_metrics = evaluate_with_threshold_local(test_labels, test_probs, args.fixed_threshold)
    selected_metrics = evaluate_with_threshold_local(test_labels, test_probs, val_threshold)

    print_metrics("val_selected", val_metrics, positive_prior=float(np.mean(val_labels)) if val_labels else 0.0)
    print_metrics("test_fixed", fixed_metrics, positive_prior=positive_prior)
    print_metrics("test_val_threshold", selected_metrics, positive_prior=positive_prior)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_frame = pd.concat([val_frame, test_frame], ignore_index=True)
    pred_frame["val_selected_threshold"] = float(val_threshold)
    pred_frame["fixed_threshold"] = float(args.fixed_threshold)
    pred_frame["pred_val_threshold"] = (pred_frame["prob"] >= float(val_threshold)).astype(int)
    pred_frame["pred_fixed_threshold"] = (pred_frame["prob"] >= float(args.fixed_threshold)).astype(int)
    pred_path = output_dir / f"{args.dataset_kind}_{fold}_single_checkpoint_predictions.csv"
    pred_frame.to_csv(pred_path, index=False)

    summary_path = output_dir / f"{args.dataset_kind}_{fold}_single_checkpoint_summary.csv"
    pd.DataFrame(
        [
            {"split": "val", "threshold_source": "selected_on_val", **val_metrics},
            {"split": "test", "threshold_source": "fixed", **fixed_metrics},
            {"split": "test", "threshold_source": "selected_on_val", **selected_metrics},
        ]
    ).to_csv(summary_path, index=False)
    print(f"saved_predictions={pred_path}")
    print(f"saved_summary={summary_path}")


if __name__ == "__main__":
    main()
