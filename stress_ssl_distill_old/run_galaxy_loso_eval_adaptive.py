from __future__ import annotations

import argparse
import csv
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


def discover_manifests(manifests_dir: Path, subjects: Iterable[str] | None) -> list[tuple[str, Path]]:
    requested = {subject.strip() for subject in subjects or [] if subject.strip()}
    manifests: list[tuple[str, Path]] = []
    for path in sorted(manifests_dir.glob("galaxy_*_loso_val.csv")):
        subject = path.stem.replace("galaxy_", "").replace("_loso_val", "")
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
    )
    lines: list[str] = []
    assert process.stdout is not None
    with log_path.open("w", encoding="utf-8") as log_file:
        for line in process.stdout:
            sys.stdout.write(line)
            log_file.write(line)
            lines.append(line)
    return_code = process.wait()
    output = "".join(lines)
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
    lines.append(
        f"{title} balanced_acc_mean={ba_mean:.4f} balanced_acc_std={ba_std:.4f}"
    )
    lines.append(
        f"{title} auroc_mean={auroc_mean:.4f} auroc_std={auroc_std:.4f}"
    )
    lines.append(
        f"{title} f1_mean={f1_mean:.4f} f1_std={f1_std:.4f}"
    )
    lines.append(
        f"{title} collapse_rate={collapse_mean:.4f}"
    )
    lines.append(
        f"{title} positive_rate_error_mean={pre_mean:.4f} positive_rate_error_std={pre_std:.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run formal LOSO comparison for watch-only baseline and teacher-guided deploy watch."
    )
    parser.add_argument("--manifests-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--skip-watch-only", action="store_true")
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
    parser.add_argument(
        "--watch-only-enhancement",
        type=str,
        default="none",
        choices=["none", "motion_disentangled"],
    )
    parser.add_argument(
        "--watch-only-motion-mode",
        type=str,
        default="strong",
        choices=["strong", "residual"],
    )
    parser.add_argument(
        "--deploy-watch-enhancement",
        type=str,
        default="none",
        choices=["none", "motion_disentangled"],
    )
    parser.add_argument(
        "--deploy-watch-motion-mode",
        type=str,
        default="strong",
        choices=["strong", "residual"],
    )
    parser.add_argument("--watch-batch-size", type=int, default=32)
    parser.add_argument("--deploy-batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--calm-sessions", nargs="*", default=["baseline"])
    parser.add_argument("--stress-sessions", nargs="*", default=["tsst-prep"])
    parser.add_argument("--priv-schedule", type=str, default="cosine")
    parser.add_argument("--priv-floor", type=float, default=0.2)
    parser.add_argument("--watch-cls-weight", type=float, default=1.0)
    parser.add_argument("--teacher-cls-weight", type=float, default=0.80)
    parser.add_argument("--teacher-fused-cls-weight", type=float, default=0.0)
    parser.add_argument("--e4-cls-weight", type=float, default=0.05)
    parser.add_argument("--rhythm-weight", type=float, default=0.15)
    parser.add_argument("--wavelet-weight", type=float, default=0.05)
    parser.add_argument("--align-weight", type=float, default=0.0)
    parser.add_argument("--distill-weight", type=float, default=0.10)
    parser.add_argument("--distill-temp", type=float, default=2.0)
    parser.add_argument("--ranking-distill-weight", type=float, default=0.10)
    parser.add_argument("--distribution-weight", type=float, default=0.05)
    parser.add_argument("--session-consistency-weight", type=float, default=0.00)
    parser.add_argument("--cross-confidence-distill", action="store_true")
    parser.add_argument(
        "--cross-confidence-targets",
        nargs="*",
        default=["kd", "ranking", "distribution"],
        choices=["kd", "ranking", "distribution"],
    )
    parser.add_argument("--cross-confidence-min-weight", type=float, default=0.0)
    parser.add_argument("--reliability-distill-weight", type=float, default=0.0)
    parser.add_argument("--clean-reliability-objective", action="store_true")
    parser.add_argument("--student-gated-correction", action="store_true")
    parser.add_argument("--correction-cls-weight", type=float, default=0.0)
    parser.add_argument("--correction-nondegradation-weight", type=float, default=0.0)
    parser.add_argument("--correction-align-weight", type=float, default=0.0)
    parser.add_argument("--correction-base-anchor-weight", type=float, default=0.0)
    parser.add_argument("--correction-margin", type=float, default=0.0)
    parser.add_argument("--correction-scale-init", type=float, default=0.05)
    parser.add_argument("--correction-alpha-init-bias", type=float, default=-3.0)
    parser.add_argument("--correction-alpha-max", type=float, default=0.35)
    parser.add_argument("--alpha-helpfulness-weight", type=float, default=0.0)
    parser.add_argument("--alpha-help-margin", type=float, default=0.0)
    parser.add_argument("--alpha-sparsity-weight", type=float, default=0.0)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    manifests = discover_manifests(args.manifests_dir, args.subjects)
    if not manifests:
        raise ValueError(f"No manifests found in {args.manifests_dir}")

    output_dir = args.output_dir
    logs_dir = output_dir / "logs"
    ckpt_dir = output_dir / "checkpoints"
    csv_path = output_dir / "galaxy_loso_formal_results.csv"
    summary_path = output_dir / "galaxy_loso_formal_summary.txt"
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
                "stress_ssl_distill.train_galaxy_watch",
                "--manifest",
                str(manifest_path),
                "--dataset-root",
                str(args.dataset_root),
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
                "--early-stop-patience",
                str(args.early_stop_patience),
                "--batch-size",
                str(args.watch_batch_size),
                "--lr",
                str(args.lr),
                "--num-workers",
                str(args.num_workers),
                "--seed",
                str(args.seed),
                "--model-type",
                args.watch_only_model_type,
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
        deploy_command = [
            sys.executable,
            "-m",
            "stress_ssl_distill.train_galaxy_privileged_adaptive",
            "--manifest",
            str(manifest_path),
            "--dataset-root",
            str(args.dataset_root),
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
            "--num-workers",
            str(args.num_workers),
            "--seed",
            str(args.seed),
            "--watch-backbone",
            args.deploy_watch_backbone,
            "--watch-enhancement",
            args.deploy_watch_enhancement,
            "--watch-motion-mode",
            args.deploy_watch_motion_mode,
            "--priv-schedule",
            args.priv_schedule,
            "--priv-floor",
            str(args.priv_floor),
            "--watch-cls-weight",
            str(args.watch_cls_weight),
            "--teacher-cls-weight",
            str(args.teacher_cls_weight),
            "--teacher-fused-cls-weight",
            str(args.teacher_fused_cls_weight),
            "--e4-cls-weight",
            str(args.e4_cls_weight),
            "--rhythm-weight",
            str(args.rhythm_weight),
            "--wavelet-weight",
            str(args.wavelet_weight),
            "--align-weight",
            str(args.align_weight),
            "--distill-weight",
            str(args.distill_weight),
            "--distill-temp",
            str(args.distill_temp),
            "--ranking-distill-weight",
            str(args.ranking_distill_weight),
            "--distribution-weight",
            str(args.distribution_weight),
            "--session-consistency-weight",
            str(args.session_consistency_weight),
            "--cross-confidence-min-weight",
            str(args.cross_confidence_min_weight),
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
            "--calm-sessions",
            *args.calm_sessions,
            "--stress-sessions",
            *args.stress_sessions,
        ]
        if args.pin_memory:
            deploy_command.append("--pin-memory")
        if args.cross_confidence_distill:
            deploy_command.append("--cross-confidence-distill")
            deploy_command.append("--cross-confidence-targets")
            deploy_command.extend(args.cross_confidence_targets)
        if args.clean_reliability_objective:
            deploy_command.append("--clean-reliability-objective")
        if args.student_gated_correction:
            deploy_command.append("--student-gated-correction")
        deploy_output = run_and_capture(deploy_command, cwd=repo_root, log_path=deploy_log)
        deploy_metrics = parse_last_metrics(deploy_output, PRIV_WATCH_PATTERN, "deploy-watch")
        teacher_metrics = parse_last_metrics(deploy_output, TEACHER_PATTERN, "teacher")
        deploy_metrics["collapse"] = float(collapse_flag(deploy_metrics["positive_rate"]))
        teacher_metrics["collapse"] = float(collapse_flag(teacher_metrics["positive_rate"]))
        deploy_metrics["positive_rate_error"] = abs(deploy_metrics["positive_rate"] - test_positive_prior)
        teacher_metrics["positive_rate_error"] = abs(teacher_metrics["positive_rate"] - test_positive_prior)
        deploy_rows.append(deploy_metrics)
        teacher_rows.append(teacher_metrics)

        summary_lines.append("deploy_watch " + " ".join(f"{key}={value:.4f}" for key, value in deploy_metrics.items() if key != "collapse"))
        summary_lines.append("teacher " + " ".join(f"{key}={value:.4f}" for key, value in teacher_metrics.items() if key != "collapse"))
        summary_lines.append("")

        csv_rows.append(
            {
                "subject": subject,
                "watch_only_threshold": watch_metrics["threshold"] if watch_metrics is not None else None,
                "watch_only_acc": watch_metrics["acc"] if watch_metrics is not None else None,
                "watch_only_balanced_acc": watch_metrics["balanced_acc"] if watch_metrics is not None else None,
                "watch_only_f1": watch_metrics["f1"] if watch_metrics is not None else None,
                "watch_only_auroc": watch_metrics["auroc"] if watch_metrics is not None else None,
                "watch_only_positive_rate": watch_metrics["positive_rate"] if watch_metrics is not None else None,
                "watch_only_positive_rate_error": watch_metrics["positive_rate_error"] if watch_metrics is not None else None,
                "watch_only_collapse": int(watch_metrics["collapse"]) if watch_metrics is not None else None,
                "deploy_watch_threshold": deploy_metrics["threshold"],
                "deploy_watch_acc": deploy_metrics["acc"],
                "deploy_watch_balanced_acc": deploy_metrics["balanced_acc"],
                "deploy_watch_f1": deploy_metrics["f1"],
                "deploy_watch_auroc": deploy_metrics["auroc"],
                "deploy_watch_positive_rate": deploy_metrics["positive_rate"],
                "deploy_watch_positive_rate_error": deploy_metrics["positive_rate_error"],
                "deploy_watch_collapse": int(deploy_metrics["collapse"]),
                "teacher_threshold": teacher_metrics["threshold"],
                "teacher_acc": teacher_metrics["acc"],
                "teacher_balanced_acc": teacher_metrics["balanced_acc"],
                "teacher_f1": teacher_metrics["f1"],
                "teacher_auroc": teacher_metrics["auroc"],
                "teacher_positive_rate": teacher_metrics["positive_rate"],
                "teacher_positive_rate_error": teacher_metrics["positive_rate_error"],
                "teacher_collapse": int(teacher_metrics["collapse"]),
                "test_positive_prior": test_positive_prior,
                "delta_deploy_vs_watch_balanced_acc": deploy_metrics["balanced_acc"] - watch_metrics["balanced_acc"] if watch_metrics is not None else None,
                "delta_deploy_vs_watch_auroc": deploy_metrics["auroc"] - watch_metrics["auroc"] if watch_metrics is not None else None,
                "delta_deploy_vs_watch_positive_rate_error": deploy_metrics["positive_rate_error"] - watch_metrics["positive_rate_error"] if watch_metrics is not None else None,
                "delta_teacher_vs_watch_balanced_acc": teacher_metrics["balanced_acc"] - watch_metrics["balanced_acc"] if watch_metrics is not None else None,
                "delta_teacher_vs_watch_auroc": teacher_metrics["auroc"] - watch_metrics["auroc"] if watch_metrics is not None else None,
                "delta_teacher_vs_watch_positive_rate_error": teacher_metrics["positive_rate_error"] - watch_metrics["positive_rate_error"] if watch_metrics is not None else None,
            }
        )

    summary_lines.append("[summary]")
    if baseline_rows:
        append_summary_block(summary_lines, "watch_only", baseline_rows)
    append_summary_block(summary_lines, "deploy_watch", deploy_rows)
    append_summary_block(summary_lines, "teacher", teacher_rows)

    if baseline_rows:
        watch_vs_deploy_ba = [row["balanced_acc"] for row in deploy_rows]
        watch_only_ba = [row["balanced_acc"] for row in baseline_rows]
        watch_vs_deploy_auroc = [row["auroc"] for row in deploy_rows]
        watch_only_auroc = [row["auroc"] for row in baseline_rows]
        watch_vs_deploy_pre = [row["positive_rate_error"] for row in deploy_rows]
        watch_only_pre = [row["positive_rate_error"] for row in baseline_rows]
        deploy_wins_ba = sum(1 for deploy, base in zip(watch_vs_deploy_ba, watch_only_ba) if deploy > base + 1e-12)
        deploy_losses_ba = sum(1 for deploy, base in zip(watch_vs_deploy_ba, watch_only_ba) if deploy < base - 1e-12)
        deploy_ties_ba = len(watch_vs_deploy_ba) - deploy_wins_ba - deploy_losses_ba
        deploy_wins_auroc = sum(1 for deploy, base in zip(watch_vs_deploy_auroc, watch_only_auroc) if deploy > base + 1e-12)
        deploy_losses_auroc = sum(1 for deploy, base in zip(watch_vs_deploy_auroc, watch_only_auroc) if deploy < base - 1e-12)
        deploy_ties_auroc = len(watch_vs_deploy_auroc) - deploy_wins_auroc - deploy_losses_auroc
        deploy_better_pre = sum(1 for deploy, base in zip(watch_vs_deploy_pre, watch_only_pre) if deploy < base - 1e-12)
        deploy_worse_pre = sum(1 for deploy, base in zip(watch_vs_deploy_pre, watch_only_pre) if deploy > base + 1e-12)
        deploy_ties_pre = len(watch_vs_deploy_pre) - deploy_better_pre - deploy_worse_pre
        summary_lines.append(
            f"deploy_vs_watch balanced_acc_wins={deploy_wins_ba} balanced_acc_losses={deploy_losses_ba} balanced_acc_ties={deploy_ties_ba}"
        )
        summary_lines.append(
            f"deploy_vs_watch auroc_wins={deploy_wins_auroc} auroc_losses={deploy_losses_auroc} auroc_ties={deploy_ties_auroc}"
        )
        summary_lines.append(
            f"deploy_vs_watch positive_rate_error_wins={deploy_better_pre} positive_rate_error_losses={deploy_worse_pre} positive_rate_error_ties={deploy_ties_pre}"
        )

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"Saved formal LOSO CSV to {csv_path}")
    print(f"Saved formal LOSO summary to {summary_path}")


if __name__ == "__main__":
    main()
