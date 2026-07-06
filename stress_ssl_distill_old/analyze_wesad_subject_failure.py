from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .analyze_wesad_teacher_student_gap import (
    collect_prediction_frame,
    evaluate_with_threshold,
)
from .train_galaxy_watch import build_loader, maybe_parse_sessions, select_threshold
from .wesad_dataset import WESADPrivilegedWindowDataset
from .wesad_models import WESADPrivilegedTeacherNet


DEFAULT_CALM_SESSIONS = ["baseline"]
DEFAULT_STRESS_SESSIONS = ["stress"]


def resolve_manifest(args: argparse.Namespace) -> Path:
    if args.manifest is not None:
        return args.manifest
    if args.subject is None or args.manifests_dir is None:
        raise ValueError("Provide either --manifest or both --subject and --manifests-dir.")
    return args.manifests_dir / f"wesad_{args.subject}_loso_val.csv"


def resolve_checkpoint(args: argparse.Namespace) -> Path:
    if args.checkpoint is not None:
        return args.checkpoint
    if args.subject is None or args.run_dir is None:
        raise ValueError("Provide either --checkpoint or both --subject and --run-dir.")

    candidates = [
        args.run_dir / "checkpoints" / f"{args.subject}_deploy_watch.pt",
        args.run_dir / "checkpoints" / f"{args.subject}.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not find checkpoint. Tried: " + ", ".join(str(item) for item in candidates))


def metric_row(
    frame: pd.DataFrame,
    prob_col: str,
    threshold: float,
    prefix: str,
    threshold_name: str,
) -> dict[str, float | str]:
    metrics = evaluate_with_threshold(
        frame["label"].astype(int).tolist(),
        frame[prob_col].astype(float).tolist(),
        threshold,
    )
    return {
        "model": prefix,
        "threshold_name": threshold_name,
        "threshold": float(threshold),
        "acc": metrics["acc"],
        "balanced_acc": metrics["balanced_acc"],
        "f1": metrics["f1"],
        "auroc": metrics["auroc"],
        "positive_rate": metrics["positive_rate"],
        "positive_rate_error": abs(metrics["positive_rate"] - float(frame["label"].mean())),
        "ece": metrics["ece"],
    }


def oracle_threshold(frame: pd.DataFrame, prob_col: str, metric: str) -> tuple[float, dict[str, float]]:
    return select_threshold(
        frame["label"].astype(int).tolist(),
        frame[prob_col].astype(float).tolist(),
        metric=metric,
    )


def summarize_distribution(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (split, label, session), group in frame.groupby(["split", "label", "session"], sort=True):
        row: dict[str, object] = {
            "split": split,
            "label": int(label),
            "session": session,
            "n": int(len(group)),
            "true_positive_rate": float(group["label"].mean()),
        }
        for col in ("watch_prob", "teacher_prob", "watch_score", "teacher_score"):
            values = group[col].astype(float)
            row[f"{col}_mean"] = float(values.mean())
            row[f"{col}_std"] = float(values.std(ddof=0))
            row[f"{col}_min"] = float(values.min())
            row[f"{col}_q10"] = float(values.quantile(0.10))
            row[f"{col}_q25"] = float(values.quantile(0.25))
            row[f"{col}_median"] = float(values.median())
            row[f"{col}_q75"] = float(values.quantile(0.75))
            row[f"{col}_q90"] = float(values.quantile(0.90))
            row[f"{col}_max"] = float(values.max())
        rows.append(row)
    return pd.DataFrame(rows)


def label_separation(frame: pd.DataFrame, split: str) -> dict[str, float]:
    split_frame = frame[frame["split"] == split]
    baseline = split_frame[split_frame["label"] == 0]
    stress = split_frame[split_frame["label"] == 1]
    result: dict[str, float] = {"true_positive_prior": float(split_frame["label"].mean())}
    for col in ("watch_prob", "teacher_prob", "watch_score", "teacher_score"):
        if baseline.empty or stress.empty:
            result[f"{col}_stress_minus_baseline"] = float("nan")
            result[f"{col}_cohen_d"] = float("nan")
            continue
        baseline_values = baseline[col].astype(float).to_numpy()
        stress_values = stress[col].astype(float).to_numpy()
        pooled_std = np.sqrt(0.5 * (np.var(baseline_values) + np.var(stress_values)))
        result[f"{col}_stress_minus_baseline"] = float(np.mean(stress_values) - np.mean(baseline_values))
        result[f"{col}_cohen_d"] = float((np.mean(stress_values) - np.mean(baseline_values)) / max(pooled_std, 1e-6))
    return result


def threshold_curve(frame: pd.DataFrame, thresholds: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model_name, prob_col in (("watch", "watch_prob"), ("teacher", "teacher_prob")):
        for threshold in thresholds:
            rows.append(metric_row(frame, prob_col, float(threshold), model_name, "sweep"))
    return pd.DataFrame(rows)


def add_prediction_columns(frame: pd.DataFrame, watch_threshold: float, teacher_threshold: float) -> pd.DataFrame:
    frame = frame.copy()
    frame["watch_pred"] = (frame["watch_prob"] >= watch_threshold).astype(int)
    frame["teacher_pred"] = (frame["teacher_prob"] >= teacher_threshold).astype(int)
    frame["watch_correct"] = (frame["watch_pred"] == frame["label"]).astype(int)
    frame["teacher_correct"] = (frame["teacher_pred"] == frame["label"]).astype(int)
    frame["teacher_only_correct"] = ((frame["teacher_correct"] == 1) & (frame["watch_correct"] == 0)).astype(int)
    frame["watch_only_correct"] = ((frame["watch_correct"] == 1) & (frame["teacher_correct"] == 0)).astype(int)
    frame["prediction_disagree"] = (frame["watch_pred"] != frame["teacher_pred"]).astype(int)
    return frame


def diagnose_failure(summary: dict[str, float]) -> list[str]:
    notes: list[str] = []
    if summary["watch_test_auroc"] < 0.5:
        notes.append("watch ranking is reversed or strongly subject-misaligned (AUROC < 0.5).")
    if summary["watch_oracle_gap_balanced_acc"] > 0.10 and summary["watch_test_auroc"] >= 0.75:
        notes.append("watch ranking is usable, but validation threshold transfers poorly to the held-out subject.")
    if summary["watch_positive_rate_error"] > 0.20:
        notes.append("watch positive rate is far from the held-out prior; calibration/score scale is unstable.")
    if summary["teacher_only_correct_rate"] > 0.20:
        notes.append("teacher sees useful information that the deployable watch branch is not absorbing.")
    if summary["watch_teacher_prob_corr"] < 0.30:
        notes.append("student and teacher probabilities are weakly aligned on this held-out subject.")
    if not notes:
        notes.append("no single severe failure mode dominates; inspect distribution_summary.csv for smaller shifts.")
    return notes


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose WESAD deploy-watch subject-level failure modes.")
    parser.add_argument("--subject", type=str, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--manifests-dir", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--dataset-root", "--wesad-root", dest="dataset_root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--cache-subjects", type=int, default=15)
    parser.add_argument("--wavelet", type=str, default="db4")
    parser.add_argument("--wavelet-level", type=int, default=4)
    parser.add_argument("--baseline-reference", action="store_true")
    parser.add_argument("--calm-sessions", nargs="*", default=None)
    parser.add_argument("--stress-sessions", nargs="*", default=None)
    parser.add_argument("--threshold-metric", type=str, default="balanced_acc", choices=["acc", "balanced_acc", "f1", "auroc"])
    parser.add_argument("--watch-backbone", type=str, default="wavelet_guided", choices=["wavelet_guided", "resnet18_1d", "resnet34_1d", "resnet50_1d"])
    parser.add_argument("--watch-model-dim", type=int, default=192)
    parser.add_argument("--watch-transformer-layers", type=int, default=2)
    parser.add_argument("--watch-transformer-heads", type=int, default=4)
    parser.add_argument("--watch-fusion-hidden-dim", type=int, default=256)
    parser.add_argument("--watch-embed-dim", type=int, default=160)
    parser.add_argument("--align-proj-dim", type=int, default=128)
    args = parser.parse_args()

    manifest = resolve_manifest(args)
    checkpoint = resolve_checkpoint(args)
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest}")
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    calm_sessions = maybe_parse_sessions(args.calm_sessions, DEFAULT_CALM_SESSIONS)
    stress_sessions = maybe_parse_sessions(args.stress_sessions, DEFAULT_STRESS_SESSIONS)
    include_sessions = calm_sessions + stress_sessions

    val_ds = WESADPrivilegedWindowDataset(
        manifest_csv=manifest,
        split="val",
        wesad_root=args.dataset_root,
        include_sessions=include_sessions,
        cache_subjects=args.cache_subjects,
        wavelet=args.wavelet,
        wavelet_level=args.wavelet_level,
        baseline_reference=args.baseline_reference,
    )
    test_ds = WESADPrivilegedWindowDataset(
        manifest_csv=manifest,
        split="test",
        wesad_root=args.dataset_root,
        include_sessions=include_sessions,
        cache_subjects=args.cache_subjects,
        wavelet=args.wavelet,
        wavelet_level=args.wavelet_level,
        baseline_reference=args.baseline_reference,
    )
    if len(val_ds) == 0 or len(test_ds) == 0:
        raise ValueError("Validation or test split is empty after session filtering.")

    val_loader = build_loader(val_ds, args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)
    test_loader = build_loader(test_ds, args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)

    model = WESADPrivilegedTeacherNet(
        watch_backbone=args.watch_backbone,
        embed_dim=args.watch_embed_dim,
        align_dim=args.align_proj_dim,
        model_dim=args.watch_model_dim,
        transformer_layers=args.watch_transformer_layers,
        transformer_heads=args.watch_transformer_heads,
        fusion_hidden_dim=args.watch_fusion_hidden_dim,
    ).to(args.device)
    state_dict = torch.load(checkpoint, map_location=args.device)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    ignored_missing = [
        key
        for key in missing_keys
        if key.startswith("watch_contrastive_head.") or key.startswith("reliability_head.")
    ]
    remaining_missing = [key for key in missing_keys if key not in ignored_missing]
    if remaining_missing or unexpected_keys:
        raise RuntimeError(
            "Checkpoint does not match WESAD model. "
            f"missing={remaining_missing} unexpected={unexpected_keys}"
        )
    model.eval()

    val_frame = collect_prediction_frame(model, val_loader, args.device, args.pin_memory, baseline_reference=args.baseline_reference)
    test_frame = collect_prediction_frame(model, test_loader, args.device, args.pin_memory, baseline_reference=args.baseline_reference)
    val_frame["split"] = "val"
    test_frame["split"] = "test"

    watch_threshold, _ = select_threshold(
        val_frame["label"].astype(int).tolist(),
        val_frame["watch_prob"].astype(float).tolist(),
        metric=args.threshold_metric,
    )
    teacher_threshold, _ = select_threshold(
        val_frame["label"].astype(int).tolist(),
        val_frame["teacher_prob"].astype(float).tolist(),
        metric=args.threshold_metric,
    )
    watch_oracle_threshold, watch_oracle_metrics = oracle_threshold(test_frame, "watch_prob", metric=args.threshold_metric)
    teacher_oracle_threshold, teacher_oracle_metrics = oracle_threshold(test_frame, "teacher_prob", metric=args.threshold_metric)

    test_frame = add_prediction_columns(test_frame, watch_threshold, teacher_threshold)
    combined_frame = pd.concat([val_frame, test_frame], ignore_index=True)

    metric_rows = [
        metric_row(test_frame, "watch_prob", watch_threshold, "watch", "val_selected"),
        metric_row(test_frame, "teacher_prob", teacher_threshold, "teacher", "val_selected"),
        metric_row(test_frame, "watch_prob", 0.5, "watch", "fixed_0.5"),
        metric_row(test_frame, "teacher_prob", 0.5, "teacher", "fixed_0.5"),
        metric_row(test_frame, "watch_prob", watch_oracle_threshold, "watch", "test_oracle"),
        metric_row(test_frame, "teacher_prob", teacher_oracle_threshold, "teacher", "test_oracle"),
    ]
    metrics_df = pd.DataFrame(metric_rows)
    distribution_df = summarize_distribution(combined_frame)
    curve_df = threshold_curve(test_frame, thresholds=np.linspace(0.0, 1.0, 101))

    watch_test_metrics = metric_rows[0]
    teacher_test_metrics = metric_rows[1]
    val_sep = label_separation(combined_frame, "val")
    test_sep = label_separation(combined_frame, "test")

    summary = {
        "watch_threshold": float(watch_threshold),
        "teacher_threshold": float(teacher_threshold),
        "watch_oracle_threshold": float(watch_oracle_threshold),
        "teacher_oracle_threshold": float(teacher_oracle_threshold),
        "watch_test_balanced_acc": float(watch_test_metrics["balanced_acc"]),
        "watch_test_auroc": float(watch_test_metrics["auroc"]),
        "watch_test_f1": float(watch_test_metrics["f1"]),
        "watch_positive_rate": float(watch_test_metrics["positive_rate"]),
        "watch_positive_rate_error": float(watch_test_metrics["positive_rate_error"]),
        "teacher_test_balanced_acc": float(teacher_test_metrics["balanced_acc"]),
        "teacher_test_auroc": float(teacher_test_metrics["auroc"]),
        "teacher_test_f1": float(teacher_test_metrics["f1"]),
        "teacher_positive_rate": float(teacher_test_metrics["positive_rate"]),
        "teacher_positive_rate_error": float(teacher_test_metrics["positive_rate_error"]),
        "watch_oracle_balanced_acc": float(watch_oracle_metrics["balanced_acc"]),
        "teacher_oracle_balanced_acc": float(teacher_oracle_metrics["balanced_acc"]),
        "watch_oracle_gap_balanced_acc": float(watch_oracle_metrics["balanced_acc"] - watch_test_metrics["balanced_acc"]),
        "teacher_oracle_gap_balanced_acc": float(teacher_oracle_metrics["balanced_acc"] - teacher_test_metrics["balanced_acc"]),
        "watch_teacher_prob_corr": float(test_frame["watch_prob"].corr(test_frame["teacher_prob"])),
        "teacher_only_correct_rate": float(test_frame["teacher_only_correct"].mean()),
        "watch_only_correct_rate": float(test_frame["watch_only_correct"].mean()),
        "prediction_disagree_rate": float(test_frame["prediction_disagree"].mean()),
        "watch_teacher_cosine_mean": float(test_frame["watch_teacher_cosine"].mean()),
        "watch_teacher_align_cosine_mean": float(test_frame["watch_teacher_align_cosine"].mean()),
        "watch_teacher_l2_mean": float(test_frame["watch_teacher_l2"].mean()),
        "val_watch_prob_stress_minus_baseline": val_sep["watch_prob_stress_minus_baseline"],
        "test_watch_prob_stress_minus_baseline": test_sep["watch_prob_stress_minus_baseline"],
        "val_teacher_prob_stress_minus_baseline": val_sep["teacher_prob_stress_minus_baseline"],
        "test_teacher_prob_stress_minus_baseline": test_sep["teacher_prob_stress_minus_baseline"],
        "val_positive_prior": val_sep["true_positive_prior"],
        "test_positive_prior": test_sep["true_positive_prior"],
    }
    notes = diagnose_failure(summary)

    output_dir = args.output_dir
    if args.subject is not None:
        output_dir = output_dir / args.subject
    output_dir.mkdir(parents=True, exist_ok=True)

    val_frame.to_csv(output_dir / "val_predictions.csv", index=False)
    test_frame.to_csv(output_dir / "test_predictions.csv", index=False)
    metrics_df.to_csv(output_dir / "threshold_metrics.csv", index=False)
    distribution_df.to_csv(output_dir / "distribution_summary.csv", index=False)
    curve_df.to_csv(output_dir / "threshold_curve.csv", index=False)
    pd.DataFrame([summary]).to_csv(output_dir / "failure_summary.csv", index=False)

    summary_lines = [
        f"subject={args.subject or 'from_manifest'}",
        f"manifest={manifest}",
        f"checkpoint={checkpoint}",
        "",
        f"watch_threshold={watch_threshold:.4f} watch_oracle_threshold={watch_oracle_threshold:.4f}",
        f"teacher_threshold={teacher_threshold:.4f} teacher_oracle_threshold={teacher_oracle_threshold:.4f}",
        "",
        (
            "watch "
            f"balanced_acc={summary['watch_test_balanced_acc']:.4f} "
            f"auroc={summary['watch_test_auroc']:.4f} "
            f"f1={summary['watch_test_f1']:.4f} "
            f"positive_rate={summary['watch_positive_rate']:.4f} "
            f"positive_rate_error={summary['watch_positive_rate_error']:.4f}"
        ),
        (
            "teacher "
            f"balanced_acc={summary['teacher_test_balanced_acc']:.4f} "
            f"auroc={summary['teacher_test_auroc']:.4f} "
            f"f1={summary['teacher_test_f1']:.4f} "
            f"positive_rate={summary['teacher_positive_rate']:.4f} "
            f"positive_rate_error={summary['teacher_positive_rate_error']:.4f}"
        ),
        "",
        f"watch_oracle_balanced_acc={summary['watch_oracle_balanced_acc']:.4f}",
        f"watch_oracle_gap_balanced_acc={summary['watch_oracle_gap_balanced_acc']:.4f}",
        f"teacher_oracle_balanced_acc={summary['teacher_oracle_balanced_acc']:.4f}",
        f"teacher_oracle_gap_balanced_acc={summary['teacher_oracle_gap_balanced_acc']:.4f}",
        "",
        f"val_watch_prob_stress_minus_baseline={summary['val_watch_prob_stress_minus_baseline']:.4f}",
        f"test_watch_prob_stress_minus_baseline={summary['test_watch_prob_stress_minus_baseline']:.4f}",
        f"val_teacher_prob_stress_minus_baseline={summary['val_teacher_prob_stress_minus_baseline']:.4f}",
        f"test_teacher_prob_stress_minus_baseline={summary['test_teacher_prob_stress_minus_baseline']:.4f}",
        "",
        f"watch_teacher_prob_corr={summary['watch_teacher_prob_corr']:.4f}",
        f"teacher_only_correct_rate={summary['teacher_only_correct_rate']:.4f}",
        f"watch_only_correct_rate={summary['watch_only_correct_rate']:.4f}",
        f"prediction_disagree_rate={summary['prediction_disagree_rate']:.4f}",
        "",
        "diagnosis:",
        *[f"- {note}" for note in notes],
        "",
        f"saved_dir={output_dir}",
    ]
    summary_text = "\n".join(summary_lines) + "\n"
    (output_dir / "failure_report.txt").write_text(summary_text, encoding="utf-8")
    print(summary_text, end="")


if __name__ == "__main__":
    main()
