from __future__ import annotations

import argparse
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd


WATCH_ONLY_PATTERN = re.compile(
    r"best_threshold=(?P<threshold>[-+]?\d*\.?\d+)\s+"
    r"best_test_acc=(?P<acc>[-+]?\d*\.?\d+)\s+"
    r"best_test_balanced_acc=(?P<balanced_acc>[-+]?\d*\.?\d+)\s+"
    r"best_test_f1=(?P<f1>[-+]?\d*\.?\d+)\s+"
    r"best_test_auroc=(?P<auroc>[-+]?\d*\.?\d+)\s+"
    r"best_test_positive_rate=(?P<positive_rate>[-+]?\d*\.?\d+)"
)

PRIV_WATCH_PATTERN = re.compile(
    r"best_watch_threshold=(?P<threshold>[-+]?\d*\.?\d+)\s+"
    r"best_watch_test_acc=(?P<acc>[-+]?\d*\.?\d+)\s+"
    r"best_watch_test_balanced_acc=(?P<balanced_acc>[-+]?\d*\.?\d+)\s+"
    r"best_watch_test_f1=(?P<f1>[-+]?\d*\.?\d+)\s+"
    r"best_watch_test_auroc=(?P<auroc>[-+]?\d*\.?\d+)\s+"
    r"best_watch_test_positive_rate=(?P<positive_rate>[-+]?\d*\.?\d+)"
)

TEACHER_PATTERN = re.compile(
    r"best_teacher_threshold=(?P<threshold>[-+]?\d*\.?\d+)\s+"
    r"best_teacher_test_acc=(?P<acc>[-+]?\d*\.?\d+)\s+"
    r"best_teacher_test_balanced_acc=(?P<balanced_acc>[-+]?\d*\.?\d+)\s+"
    r"best_teacher_test_f1=(?P<f1>[-+]?\d*\.?\d+)\s+"
    r"best_teacher_test_auroc=(?P<auroc>[-+]?\d*\.?\d+)\s+"
    r"best_teacher_test_positive_rate=(?P<positive_rate>[-+]?\d*\.?\d+)"
)


def discover_manifests(manifests_dir: Path, subjects: Iterable[str] | None, dataset_kind: str) -> list[tuple[str, Path]]:
    requested = {subject.strip() for subject in subjects or [] if subject.strip()}
    manifests: list[tuple[str, Path]] = []
    prefix = "catsa" if dataset_kind == "catsa" else "wesad"
    for path in sorted(manifests_dir.glob(f"{prefix}_*_loso_val.csv")):
        subject = path.stem.replace(f"{prefix}_", "").replace("_loso_val", "")
        if requested and subject not in requested:
            continue
        manifests.append((subject, path))
    return manifests


def run_and_capture(command: list[str], cwd: Path, log_path: Path) -> str:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    chunks: list[str] = []
    assert process.stdout is not None
    with log_path.open("w", encoding="utf-8") as log_file:
        while True:
            chunk = process.stdout.read(1)
            if chunk == "":
                break
            sys.stdout.write(chunk)
            sys.stdout.flush()
            log_file.write(chunk)
            log_file.flush()
            chunks.append(chunk)
    return_code = process.wait()
    output = "".join(chunks)
    if return_code != 0:
        raise RuntimeError(f"Command failed with exit code {return_code}: {' '.join(command)}")
    return output


def parse_last_metrics(output: str, pattern: re.Pattern[str], label: str) -> dict[str, float]:
    groups = None
    for item in pattern.finditer(output):
        groups = item.groupdict()
    if groups is None:
        raise ValueError(f"Could not parse {label} metrics from command output.")
    return {key: float(value) for key, value in groups.items()}


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def collapse_flag(positive_rate: float, low: float = 0.05, high: float = 0.95) -> int:
    return int(positive_rate <= low or positive_rate >= high)


def load_test_positive_prior(
    manifest_path: Path,
    calm_sessions: list[str],
    stress_sessions: list[str],
) -> float:
    df = pd.read_csv(manifest_path)
    if "split" in df.columns:
        df = df[df["split"] == "test"]
    if "session" in df.columns:
        session_set = set(calm_sessions) | set(stress_sessions)
        df = df[df["session"].isin(session_set)]
    if df.empty or "label" not in df.columns:
        return 0.5
    labels = pd.to_numeric(df["label"], errors="coerce").dropna()
    if labels.empty:
        return 0.5
    return float(labels.mean())


def append_summary_block(lines: list[str], title: str, rows: list[dict[str, float]]) -> None:
    balanced = [row["balanced_acc"] for row in rows]
    auroc = [row["auroc"] for row in rows]
    f1 = [row["f1"] for row in rows]
    collapse = [row["collapse"] for row in rows]
    positive_rate_error = [row["positive_rate_error"] for row in rows]
    ba_mean, ba_std = mean_std(balanced)
    auroc_mean, auroc_std = mean_std(auroc)
    f1_mean, f1_std = mean_std(f1)
    collapse_mean, _ = mean_std(collapse)
    pre_mean, pre_std = mean_std(positive_rate_error)
    lines.append(f"{title} balanced_acc_mean={ba_mean:.4f} balanced_acc_std={ba_std:.4f}")
    lines.append(f"{title} auroc_mean={auroc_mean:.4f} auroc_std={auroc_std:.4f}")
    lines.append(f"{title} f1_mean={f1_mean:.4f} f1_std={f1_std:.4f}")
    lines.append(f"{title} collapse_rate={collapse_mean:.4f}")
    lines.append(f"{title} positive_rate_error_mean={pre_mean:.4f} positive_rate_error_std={pre_std:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run formal WESAD LOSO comparison for watch-only baseline and teacher-guided deploy watch."
    )
    parser.add_argument("--manifests-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", "--wesad-root", dest="dataset_root", type=Path, required=True)
    parser.add_argument("--dataset-kind", type=str, default="wesad", choices=["wesad", "catsa"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--skip-watch-only", action="store_true")
    parser.add_argument(
        "--init-deploy-from-watch-only",
        action="store_true",
        help="Initialize each deploy-watch model from its matching watch-only checkpoint.",
    )
    parser.add_argument(
        "--watch-checkpoints-dir",
        type=Path,
        default=None,
        help="Directory containing watch-only checkpoints when --skip-watch-only is used.",
    )
    parser.add_argument(
        "--include-initial-eval",
        action="store_true",
        help="Treat the deploy model before privileged fine-tuning as an early-stopping candidate.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--monitor", type=str, default="balanced_acc")
    parser.add_argument("--threshold-metric", type=str, default="monitor", choices=["monitor", "acc", "balanced_acc", "f1", "auroc"])
    parser.add_argument("--early-stop-patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--watch-only-model-type",
        type=str,
        default="wavelet_guided",
        choices=["wavelet_guided", "resnet18_1d", "resnet34_1d", "resnet50_1d"],
    )
    parser.add_argument(
        "--deploy-watch-backbone",
        type=str,
        default="wavelet_guided",
        choices=["wavelet_guided", "resnet18_1d", "resnet34_1d", "resnet50_1d"],
    )
    parser.add_argument("--watch-only-enhancement", type=str, default="none", choices=["none", "motion_disentangled"])
    parser.add_argument("--watch-only-motion-mode", type=str, default="strong", choices=["strong", "residual"])
    parser.add_argument("--deploy-watch-enhancement", type=str, default="none", choices=["none", "motion_disentangled"])
    parser.add_argument("--deploy-watch-motion-mode", type=str, default="strong", choices=["strong", "residual"])
    parser.add_argument("--watch-batch-size", type=int, default=32)
    parser.add_argument("--deploy-batch-size", type=int, default=24)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--focal-gamma", type=float, default=1.5)
    parser.add_argument("--contrastive-weight", type=float, default=0.15)
    parser.add_argument("--deploy-contrastive-weight", type=float, default=0.0)
    parser.add_argument("--watch-wavelet-weight", type=float, default=0.2)
    parser.add_argument("--priv-wavelet-weight", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--distill-weight", type=float, default=0.10)
    parser.add_argument("--embedding-align-weight", type=float, default=0.05)
    parser.add_argument("--margin-match-weight", type=float, default=0.0)
    parser.add_argument("--normalized-margin-align-weight", type=float, default=0.0)
    parser.add_argument("--ranking-distill-weight", type=float, default=0.10)
    parser.add_argument("--distribution-weight", type=float, default=0.05)
    parser.add_argument("--subject-center-stability-weight", type=float, default=0.0)
    parser.add_argument("--validation-threshold-stability-weight", type=float, default=0.0)
    parser.add_argument("--session-consistency-weight", type=float, default=0.0)
    parser.add_argument("--distill-gating", type=str, default="none", choices=["none", "score_agreement", "teacher_confidence"])
    parser.add_argument("--distill-disagreement-weight", type=float, default=0.25)
    parser.add_argument("--teacher-confidence-threshold", type=float, default=1.0)
    parser.add_argument("--teacher-confidence-temperature", type=float, default=0.5)
    parser.add_argument("--min-distill-weight", type=float, default=0.2)
    parser.add_argument("--distill-temp", type=float, default=2.0)
    parser.add_argument("--reliability-distill-weight", type=float, default=0.0)
    parser.add_argument("--clean-reliability-objective", action="store_true")
    parser.add_argument("--student-gated-correction", action="store_true")
    parser.add_argument("--correction-cls-weight", type=float, default=0.0)
    parser.add_argument("--correction-nondegradation-weight", type=float, default=0.0)
    parser.add_argument("--correction-align-weight", type=float, default=0.0)
    parser.add_argument("--correction-base-anchor-weight", type=float, default=0.0)
    parser.add_argument("--correction-margin", type=float, default=0.0)
    parser.add_argument("--correction-scale-init", type=float, default=0.05)
    parser.add_argument("--correction-alpha-init-bias", type=float, default=-2.0)
    parser.add_argument("--correction-alpha-max", type=float, default=0.35)
    parser.add_argument("--alpha-helpfulness-weight", type=float, default=0.0)
    parser.add_argument("--alpha-help-margin", type=float, default=0.0)
    parser.add_argument("--alpha-sparsity-weight", type=float, default=0.0)
    parser.add_argument("--teacher-cls-weight", type=float, default=0.80)
    parser.add_argument("--privileged-cls-weight", type=float, default=0.05)
    parser.add_argument("--baseline-relative-weight", type=float, default=0.0)
    parser.add_argument("--baseline-relative-margin", type=float, default=0.2)
    parser.add_argument("--catsa-privileged-modalities", nargs="*", default=["EDA", "TEMP"])
    parser.add_argument("--eval-aggregation", type=str, default="window", choices=["window", "session_mean"])
    parser.add_argument("--threshold-mode", type=str, default="val_search", choices=["val_search", "fixed"])
    parser.add_argument("--fixed-threshold", type=float, default=0.5)
    parser.add_argument("--watch-model-dim", type=int, default=192)
    parser.add_argument("--watch-transformer-layers", type=int, default=2)
    parser.add_argument("--watch-transformer-heads", type=int, default=4)
    parser.add_argument("--watch-fusion-hidden-dim", type=int, default=256)
    parser.add_argument("--watch-embed-dim", type=int, default=160)
    parser.add_argument("--align-proj-dim", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cache-subjects", type=int, default=15)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--baseline-reference", action="store_true")
    parser.add_argument("--subject-aware-batching", action="store_true")
    parser.add_argument("--calm-sessions", nargs="*", default=["baseline"])
    parser.add_argument("--stress-sessions", nargs="*", default=["stress"])
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    manifests = discover_manifests(args.manifests_dir, args.subjects, args.dataset_kind)
    if not manifests:
        raise ValueError(f"No manifests found in {args.manifests_dir}")

    output_dir = args.output_dir
    logs_dir = output_dir / "logs"
    ckpt_dir = output_dir / "checkpoints"
    output_prefix = "catsa" if args.dataset_kind == "catsa" else "wesad"
    csv_path = output_dir / f"{output_prefix}_loso_formal_results.csv"
    summary_path = output_dir / f"{output_prefix}_loso_formal_summary.txt"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    summary_lines: list[str] = []
    csv_rows: list[dict[str, object]] = []
    baseline_rows: list[dict[str, float]] = []
    deploy_rows: list[dict[str, float]] = []
    teacher_rows: list[dict[str, float]] = []

    for subject, manifest_path in manifests:
        summary_lines.append(f"[{subject}]")
        test_positive_prior = load_test_positive_prior(
            manifest_path,
            calm_sessions=list(args.calm_sessions),
            stress_sessions=list(args.stress_sessions),
        )
        summary_lines.append(f"test_positive_prior={test_positive_prior:.4f}")

        watch_metrics: dict[str, float] | None = None
        if not args.skip_watch_only:
            watch_log = logs_dir / f"{subject}_watch_only.log"
            watch_metrics_csv = logs_dir / f"{subject}_watch_only_metrics.csv"
            watch_ckpt = ckpt_dir / f"{subject}_watch_only.pt"
            watch_command = [
                sys.executable,
                "-m",
                "stress_ssl_distill.train_wesad_watch",
                "--manifest",
                str(manifest_path),
                "--dataset-root",
                str(args.dataset_root),
                "--dataset-kind",
                args.dataset_kind,
                "--save-path",
                str(watch_ckpt),
                "--metrics-path",
                str(watch_metrics_csv),
                "--device",
                args.device,
                "--epochs",
                str(args.epochs),
                "--selection-mode",
                "early_stop",
                "--monitor",
                args.monitor,
                "--threshold-metric",
                args.threshold_metric,
                "--early-stop-patience",
                str(args.early_stop_patience),
                "--batch-size",
                str(args.watch_batch_size),
                "--lr",
                str(args.lr),
                "--weight-decay",
                str(args.weight_decay),
                "--label-smoothing",
                str(args.label_smoothing),
                "--focal-gamma",
                str(args.focal_gamma),
                "--contrastive-weight",
                str(args.contrastive_weight),
                "--wavelet-weight",
                str(args.watch_wavelet_weight),
                "--temperature",
                str(args.temperature),
                "--baseline-relative-weight",
                str(args.baseline_relative_weight),
                "--baseline-relative-margin",
                str(args.baseline_relative_margin),
                "--eval-aggregation",
                args.eval_aggregation,
                "--threshold-mode",
                args.threshold_mode,
                "--fixed-threshold",
                str(args.fixed_threshold),
                "--num-workers",
                str(args.num_workers),
                "--cache-subjects",
                str(args.cache_subjects),
                "--seed",
                str(args.seed),
                "--model-type",
                args.watch_only_model_type,
                "--watch-model-dim",
                str(args.watch_model_dim),
                "--watch-transformer-layers",
                str(args.watch_transformer_layers),
                "--watch-transformer-heads",
                str(args.watch_transformer_heads),
                "--watch-fusion-hidden-dim",
                str(args.watch_fusion_hidden_dim),
                "--watch-embed-dim",
                str(args.watch_embed_dim),
                "--watch-enhancement",
                args.watch_only_enhancement,
                "--watch-motion-mode",
                args.watch_only_motion_mode,
                "--calm-sessions",
                *args.calm_sessions,
                "--stress-sessions",
                *args.stress_sessions,
            ]
            if args.pin_memory:
                watch_command.append("--pin-memory")
            if args.baseline_reference:
                watch_command.append("--baseline-reference")
            if args.subject_aware_batching:
                watch_command.append("--subject-aware-batching")
            watch_output = run_and_capture(watch_command, cwd=repo_root, log_path=watch_log)
            watch_metrics = parse_last_metrics(watch_output, WATCH_ONLY_PATTERN, "watch-only")
            watch_metrics["collapse"] = float(collapse_flag(watch_metrics["positive_rate"]))
            watch_metrics["positive_rate_error"] = abs(watch_metrics["positive_rate"] - test_positive_prior)
            baseline_rows.append(watch_metrics)
            summary_lines.append("watch_only " + " ".join(f"{key}={value:.4f}" for key, value in watch_metrics.items() if key != "collapse"))
        else:
            summary_lines.append("watch_only skipped=true")

        deploy_log = logs_dir / f"{subject}_deploy_watch.log"
        deploy_metrics_csv = logs_dir / f"{subject}_deploy_watch_metrics.csv"
        deploy_ckpt = ckpt_dir / f"{subject}_deploy_watch.pt"
        init_watch_ckpt: Path | None = None
        if args.init_deploy_from_watch_only:
            if watch_metrics is not None:
                init_watch_ckpt = watch_ckpt
            elif args.watch_checkpoints_dir is not None:
                candidates = [
                    args.watch_checkpoints_dir / "checkpoints" / f"{subject}.pt",
                    args.watch_checkpoints_dir / "checkpoints" / f"{subject}_watch_only.pt",
                    args.watch_checkpoints_dir / f"{subject}.pt",
                    args.watch_checkpoints_dir / f"{subject}_watch_only.pt",
                ]
                init_watch_ckpt = next((path for path in candidates if path.exists()), None)
            if init_watch_ckpt is None:
                raise FileNotFoundError(
                    f"Could not find watch-only checkpoint for {subject}; "
                    "run without --skip-watch-only or set --watch-checkpoints-dir."
                )
        deploy_command = [
            sys.executable,
            "-m",
            "stress_ssl_distill.train_wesad_privileged_safe_sgpc",
            "--manifest",
            str(manifest_path),
            "--dataset-root",
            str(args.dataset_root),
            "--dataset-kind",
            args.dataset_kind,
            "--save-path",
            str(deploy_ckpt),
            "--metrics-path",
            str(deploy_metrics_csv),
            "--device",
            args.device,
            "--epochs",
            str(args.epochs),
            "--selection-mode",
            "early_stop",
            "--selection-target",
            "watch",
            "--monitor",
            args.monitor,
            "--threshold-metric",
            args.threshold_metric,
            "--early-stop-patience",
            str(args.early_stop_patience),
            "--batch-size",
            str(args.deploy_batch_size),
            "--lr",
            str(args.lr),
            "--weight-decay",
            str(args.weight_decay),
            "--label-smoothing",
            str(args.label_smoothing),
            "--focal-gamma",
            str(args.focal_gamma),
            "--watch-contrastive-weight",
            str(args.deploy_contrastive_weight),
            "--contrastive-temperature",
            str(args.temperature),
            "--teacher-cls-weight",
            str(args.teacher_cls_weight),
            "--privileged-cls-weight",
            str(args.privileged_cls_weight),
            "--distill-weight",
            str(args.distill_weight),
            "--embedding-align-weight",
            str(args.embedding_align_weight),
            "--margin-match-weight",
            str(args.margin_match_weight),
            "--normalized-margin-align-weight",
            str(args.normalized_margin_align_weight),
            "--ranking-distill-weight",
            str(args.ranking_distill_weight),
            "--distribution-weight",
            str(args.distribution_weight),
            "--subject-center-stability-weight",
            str(args.subject_center_stability_weight),
            "--validation-threshold-stability-weight",
            str(args.validation_threshold_stability_weight),
            "--session-consistency-weight",
            str(args.session_consistency_weight),
            "--distill-gating",
            args.distill_gating,
            "--distill-disagreement-weight",
            str(args.distill_disagreement_weight),
            "--teacher-confidence-threshold",
            str(args.teacher_confidence_threshold),
            "--teacher-confidence-temperature",
            str(args.teacher_confidence_temperature),
            "--min-distill-weight",
            str(args.min_distill_weight),
            "--distill-temp",
            str(args.distill_temp),
            "--reliability-distill-weight",
            str(args.reliability_distill_weight),
            "--correction-cls-weight",
            str(args.correction_cls_weight),
            "--correction-nondegradation-weight",
            str(args.correction_nondegradation_weight),
            "--correction-align-weight",
            str(args.correction_align_weight),
            "--correction-base-anchor-weight",
            str(args.correction_base_anchor_weight),
            "--correction-margin",
            str(args.correction_margin),
            "--correction-scale-init",
            str(args.correction_scale_init),
            "--correction-alpha-init-bias",
            str(args.correction_alpha_init_bias),
            "--correction-alpha-max",
            str(args.correction_alpha_max),
            "--alpha-helpfulness-weight",
            str(args.alpha_helpfulness_weight),
            "--alpha-help-margin",
            str(args.alpha_help_margin),
            "--alpha-sparsity-weight",
            str(args.alpha_sparsity_weight),
            "--wavelet-weight",
            str(args.priv_wavelet_weight),
            "--baseline-relative-weight",
            str(args.baseline_relative_weight),
            "--baseline-relative-margin",
            str(args.baseline_relative_margin),
            "--catsa-privileged-modalities",
            *args.catsa_privileged_modalities,
            "--eval-aggregation",
            args.eval_aggregation,
            "--threshold-mode",
            args.threshold_mode,
            "--fixed-threshold",
            str(args.fixed_threshold),
            "--num-workers",
            str(args.num_workers),
            "--cache-subjects",
            str(args.cache_subjects),
            "--seed",
            str(args.seed),
            "--watch-backbone",
            args.deploy_watch_backbone,
            "--watch-model-dim",
            str(args.watch_model_dim),
            "--watch-transformer-layers",
            str(args.watch_transformer_layers),
            "--watch-transformer-heads",
            str(args.watch_transformer_heads),
            "--watch-fusion-hidden-dim",
            str(args.watch_fusion_hidden_dim),
            "--watch-embed-dim",
            str(args.watch_embed_dim),
            "--watch-enhancement",
            args.deploy_watch_enhancement,
            "--watch-motion-mode",
            args.deploy_watch_motion_mode,
            "--align-proj-dim",
            str(args.align_proj_dim),
            "--calm-sessions",
            *args.calm_sessions,
            "--stress-sessions",
            *args.stress_sessions,
        ]
        if init_watch_ckpt is not None:
            deploy_command.extend(["--init-watch-checkpoint", str(init_watch_ckpt)])
        if args.include_initial_eval:
            deploy_command.append("--include-initial-eval")
        if args.pin_memory:
            deploy_command.append("--pin-memory")
        if args.clean_reliability_objective:
            deploy_command.append("--clean-reliability-objective")
        if args.student_gated_correction:
            deploy_command.append("--student-gated-correction")
        if args.baseline_reference:
            deploy_command.append("--baseline-reference")
        if args.subject_aware_batching:
            deploy_command.append("--subject-aware-batching")
        deploy_output = run_and_capture(deploy_command, cwd=repo_root, log_path=deploy_log)
        deploy_metrics = parse_last_metrics(deploy_output, PRIV_WATCH_PATTERN, "deploy-watch")
        teacher_metrics = parse_last_metrics(deploy_output, TEACHER_PATTERN, "teacher")
        for row in (deploy_metrics, teacher_metrics):
            row["collapse"] = float(collapse_flag(row["positive_rate"]))
            row["positive_rate_error"] = abs(row["positive_rate"] - test_positive_prior)
        deploy_rows.append(deploy_metrics)
        teacher_rows.append(teacher_metrics)
        summary_lines.append("deploy_watch " + " ".join(f"{key}={value:.4f}" for key, value in deploy_metrics.items() if key != "collapse"))
        summary_lines.append("teacher " + " ".join(f"{key}={value:.4f}" for key, value in teacher_metrics.items() if key != "collapse"))
        summary_lines.append("")

        csv_rows.append(
            {
                "subject": subject,
                "manifest": str(manifest_path),
                "test_positive_prior": test_positive_prior,
                "watch_only_threshold": watch_metrics["threshold"] if watch_metrics is not None else None,
                "watch_only_acc": watch_metrics["acc"] if watch_metrics is not None else None,
                "watch_only_balanced_acc": watch_metrics["balanced_acc"] if watch_metrics is not None else None,
                "watch_only_f1": watch_metrics["f1"] if watch_metrics is not None else None,
                "watch_only_auroc": watch_metrics["auroc"] if watch_metrics is not None else None,
                "watch_only_positive_rate": watch_metrics["positive_rate"] if watch_metrics is not None else None,
                "watch_only_collapse": watch_metrics["collapse"] if watch_metrics is not None else None,
                "watch_only_positive_rate_error": watch_metrics["positive_rate_error"] if watch_metrics is not None else None,
                "deploy_watch_threshold": deploy_metrics["threshold"],
                "deploy_watch_acc": deploy_metrics["acc"],
                "deploy_watch_balanced_acc": deploy_metrics["balanced_acc"],
                "deploy_watch_f1": deploy_metrics["f1"],
                "deploy_watch_auroc": deploy_metrics["auroc"],
                "deploy_watch_positive_rate": deploy_metrics["positive_rate"],
                "deploy_watch_collapse": deploy_metrics["collapse"],
                "deploy_watch_positive_rate_error": deploy_metrics["positive_rate_error"],
                "teacher_threshold": teacher_metrics["threshold"],
                "teacher_acc": teacher_metrics["acc"],
                "teacher_balanced_acc": teacher_metrics["balanced_acc"],
                "teacher_f1": teacher_metrics["f1"],
                "teacher_auroc": teacher_metrics["auroc"],
                "teacher_positive_rate": teacher_metrics["positive_rate"],
                "teacher_collapse": teacher_metrics["collapse"],
                "teacher_positive_rate_error": teacher_metrics["positive_rate_error"],
            }
        )

    summary_lines.append("[summary]")
    if baseline_rows:
        append_summary_block(summary_lines, "watch_only", baseline_rows)
    append_summary_block(summary_lines, "deploy_watch", deploy_rows)
    append_summary_block(summary_lines, "teacher", teacher_rows)
    summary_text = "\n".join(summary_lines) + "\n"

    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    summary_path.write_text(summary_text, encoding="utf-8")

    print(summary_text, end="")
    print(f"Saved WESAD formal LOSO CSV to {csv_path}")
    print(f"Saved WESAD formal LOSO summary to {summary_path}")


if __name__ == "__main__":
    main()
