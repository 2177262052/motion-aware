from __future__ import annotations

import argparse
import os
import statistics
import subprocess
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

from .run_galaxy_loso_eval_elastic import (
    PRIV_WATCH_PATTERN,
    TEACHER_PATTERN,
    WATCH_ONLY_PATTERN,
    append_summary_block,
    collapse_flag,
    load_test_positive_prior,
    parse_last_metrics,
)


class ScaledMotionFiLM(nn.Module):
    """Historical learnable-strength motion FiLM used by scale_logit checkpoints."""

    def __init__(
        self,
        cond_dim: int,
        target_dim: int,
        *args: object,
        scale_logit_init: float = -2.0,
        **kwargs: object,
    ) -> None:
        super().__init__()
        self.to_gamma = nn.Sequential(
            nn.Linear(cond_dim, target_dim),
            nn.GELU(),
            nn.Linear(target_dim, target_dim),
        )
        self.to_beta = nn.Sequential(
            nn.Linear(cond_dim, target_dim),
            nn.GELU(),
            nn.Linear(target_dim, target_dim),
        )
        self.scale_logit = nn.Parameter(torch.tensor(float(scale_logit_init)))

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        gamma = self.to_gamma(condition).unsqueeze(-1)
        beta = self.to_beta(condition).unsqueeze(-1)
        scale = torch.sigmoid(self.scale_logit)
        return x * (1.0 + scale * torch.tanh(gamma)) + scale * beta


def install_scaled_motion_film() -> None:
    from . import galaxy_models

    galaxy_models.MotionFiLM = ScaledMotionFiLM


def train_one_fold() -> None:
    install_scaled_motion_film()
    sys.argv = [arg for arg in sys.argv if arg != "--_train-one"]
    print("scaled_motion_compat=on scale_logit_init=-2.0")
    print("motion_formula=x*(1+sigmoid(scale_logit)*tanh(gamma))+sigmoid(scale_logit)*beta")
    from .train_galaxy_privileged_elastic import main as train_main

    train_main()


def discover_manifests(manifests_dir: Path, subjects: list[str] | None) -> list[tuple[str, Path]]:
    requested = {subject.strip() for subject in subjects or [] if subject.strip()}
    manifests: dict[str, Path] = {}
    for pattern in ("galaxy_*_loso_val.csv", "*_loso_val.csv"):
        for path in sorted(manifests_dir.glob(pattern)):
            subject = path.stem
            if subject.startswith("galaxy_"):
                subject = subject[len("galaxy_") :]
            if subject.endswith("_loso_val"):
                subject = subject[: -len("_loso_val")]
            if requested and subject not in requested:
                continue
            manifests.setdefault(subject, path)
    return sorted(manifests.items())


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


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def main() -> None:
    if "--_train-one" in sys.argv:
        train_one_fold()
        return

    parser = argparse.ArgumentParser(
        description="Run Galaxy LOSO elastic privileged training with historical scale_logit motion FiLM."
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
    parser.add_argument("--selection-target", type=str, default="watch", choices=["watch", "teacher"])
    parser.add_argument("--early-stop-patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--watch-only-model-type", type=str, default="wavelet_guided", choices=["wavelet_guided", "resnet18_1d", "resnet34_1d", "resnet50_1d"])
    parser.add_argument("--deploy-watch-backbone", type=str, default="wavelet_guided", choices=["wavelet_guided", "resnet18_1d", "resnet34_1d", "resnet50_1d"])
    parser.add_argument("--watch-only-enhancement", type=str, default="none", choices=["none", "motion_disentangled", "acc_concat"])
    parser.add_argument("--deploy-watch-enhancement", type=str, default="none", choices=["none", "motion_disentangled", "acc_concat"])
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
    parser.add_argument(
        "--kd-gate-mode",
        type=str,
        default="none",
        choices=["none", "teacher_true_confidence", "student_true_confidence"],
    )
    parser.add_argument("--kd-gate-min-weight", type=float, default=0.0)
    parser.add_argument(
        "--detach-standard-kd-teacher",
        action="store_true",
        help="Detach teacher soft targets for ungated standard KD. Gated KD losses already detach teacher logits.",
    )
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
    parser.add_argument("--correction-mode", type=str, default="logit_mix", choices=["logit_mix", "margin_residual"])
    parser.add_argument("--alpha-helpfulness-weight", type=float, default=0.0)
    parser.add_argument("--alpha-help-margin", type=float, default=0.0)
    parser.add_argument("--alpha-sparsity-weight", type=float, default=0.0)
    parser.add_argument("--elastic-residual-weight", type=float, default=0.0)
    parser.add_argument("--elastic-alpha-target-weight", type=float, default=0.0)
    parser.add_argument("--elastic-reliability-temp", type=float, default=0.25)
    parser.add_argument("--elastic-uncertainty-temp", type=float, default=1.0)
    parser.add_argument("--elastic-label-margin", type=float, default=2.0)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    manifests = discover_manifests(args.manifests_dir, args.subjects)
    if not manifests:
        examples = []
        if args.manifests_dir.exists():
            examples = [path.name for path in sorted(args.manifests_dir.glob("*.csv"))[:10]]
        raise ValueError(
            f"No LOSO manifests found in {args.manifests_dir}. "
            "Expected files like galaxy_P10_loso_val.csv or P10_loso_val.csv. "
            f"Directory exists={args.manifests_dir.exists()} csv_examples={examples}"
        )

    output_dir = args.output_dir
    logs_dir = output_dir / "logs"
    ckpt_dir = output_dir / "checkpoints"
    csv_path = output_dir / "galaxy_loso_elastic_scaled_motion_results.csv"
    summary_path = output_dir / "galaxy_loso_elastic_scaled_motion_summary.txt"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    summary_lines: list[str] = [
        "scaled_motion_compat=on",
        "motion_formula=x*(1+sigmoid(scale_logit)*tanh(gamma))+sigmoid(scale_logit)*beta",
        "scale_logit_init=-2.0",
        "",
    ]
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
                "stress_ssl_distill.train_galaxy_watch_scaled_motion",
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
                "--threshold-metric",
                args.threshold_metric,
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
            summary_lines.append(
                "watch_only "
                + " ".join(f"{key}={value:.4f}" for key, value in watch_metrics.items() if key != "collapse")
            )
        else:
            summary_lines.append("watch_only skipped=true")

        deploy_log = logs_dir / f"{subject}_deploy_watch.log"
        deploy_metrics_csv = logs_dir / f"{subject}_deploy_watch_metrics.csv"
        deploy_ckpt = ckpt_dir / f"{subject}_deploy_watch.pt"
        deploy_command = [
            sys.executable,
            "-m",
            "stress_ssl_distill.run_galaxy_loso_eval_elastic_scaled_motion",
            "--_train-one",
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
            args.selection_target,
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
            "--kd-gate-mode",
            args.kd_gate_mode,
            "--kd-gate-min-weight",
            str(args.kd_gate_min_weight),
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
            "--correction-mode",
            args.correction_mode,
            "--alpha-helpfulness-weight",
            str(args.alpha_helpfulness_weight),
            "--alpha-help-margin",
            str(args.alpha_help_margin),
            "--alpha-sparsity-weight",
            str(args.alpha_sparsity_weight),
            "--elastic-residual-weight",
            str(args.elastic_residual_weight),
            "--elastic-alpha-target-weight",
            str(args.elastic_alpha_target_weight),
            "--elastic-reliability-temp",
            str(args.elastic_reliability_temp),
            "--elastic-uncertainty-temp",
            str(args.elastic_uncertainty_temp),
            "--elastic-label-margin",
            str(args.elastic_label_margin),
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
        if args.detach_standard_kd_teacher:
            deploy_command.append("--detach-standard-kd-teacher")
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

        summary_lines.append(
            "deploy_watch "
            + " ".join(f"{key}={value:.4f}" for key, value in deploy_metrics.items() if key != "collapse")
        )
        summary_lines.append(
            "teacher "
            + " ".join(f"{key}={value:.4f}" for key, value in teacher_metrics.items() if key != "collapse")
        )
        summary_lines.append("")

        for model_name, metrics in (
            ("watch_only", watch_metrics),
            ("deploy_watch", deploy_metrics),
            ("teacher", teacher_metrics),
        ):
            if metrics is None:
                continue
            csv_rows.append(
                {
                    "subject": subject,
                    "manifest": str(manifest_path),
                    "model": model_name,
                    "test_positive_prior": test_positive_prior,
                    "threshold": metrics["threshold"],
                    "acc": metrics["acc"],
                    "balanced_acc": metrics["balanced_acc"],
                    "f1": metrics["f1"],
                    "auroc": metrics["auroc"],
                    "positive_rate": metrics["positive_rate"],
                    "collapse": metrics["collapse"],
                    "positive_rate_error": metrics["positive_rate_error"],
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
    print(f"Saved Galaxy elastic scaled-motion LOSO CSV to {csv_path}")
    print(f"Saved Galaxy elastic scaled-motion LOSO summary to {summary_path}")

    for label, rows in (("deploy_watch", deploy_rows), ("teacher", teacher_rows)):
        if rows:
            ba_mean, ba_std = mean_std([row["balanced_acc"] for row in rows])
            auroc_mean, auroc_std = mean_std([row["auroc"] for row in rows])
            print(f"quick_check {label}_balanced_acc_mean={ba_mean:.4f} {label}_balanced_acc_std={ba_std:.4f}")
            print(f"quick_check {label}_auroc_mean={auroc_mean:.4f} {label}_auroc_std={auroc_std:.4f}")


if __name__ == "__main__":
    main()
