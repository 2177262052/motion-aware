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

from .run_wesad_loso_eval_elastic import (
    PRIV_WATCH_PATTERN,
    TEACHER_PATTERN,
    WATCH_ONLY_PATTERN,
    append_summary_block,
    collapse_flag,
    discover_manifests,
    load_test_positive_prior,
    parse_last_metrics,
)


class ScaledMotionFiLM(nn.Module):
    """Learnable-strength MotionFiLM used by the final scaled-motion model."""

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
    from .train_wesad_privileged_elastic import main as train_main

    train_main()


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
        description="Run WESAD LOSO privileged KD training with scaled motion-aware watch encoding."
    )
    parser.add_argument("--manifests-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", "--wesad-root", dest="dataset_root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--skip-watch-only", action="store_true")
    parser.add_argument("--init-deploy-from-watch-only", action="store_true")
    parser.add_argument("--watch-checkpoints-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--monitor", type=str, default="balanced_acc")
    parser.add_argument("--threshold-metric", type=str, default="monitor", choices=["monitor", "acc", "balanced_acc", "f1", "auroc"])
    parser.add_argument("--selection-target", type=str, default="watch", choices=["watch", "teacher"])
    parser.add_argument("--early-stop-patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--watch-only-model-type", type=str, default="wavelet_guided", choices=["wavelet_guided", "resnet18_1d", "resnet34_1d", "resnet50_1d"])
    parser.add_argument("--deploy-watch-backbone", type=str, default="wavelet_guided", choices=["wavelet_guided", "resnet18_1d", "resnet34_1d", "resnet50_1d"])
    parser.add_argument("--watch-only-enhancement", type=str, default="none", choices=["none", "motion_disentangled"])
    parser.add_argument("--deploy-watch-enhancement", type=str, default="none", choices=["none", "motion_disentangled"])
    parser.add_argument("--watch-batch-size", type=int, default=32)
    parser.add_argument("--deploy-batch-size", type=int, default=24)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--focal-gamma", type=float, default=1.5)
    parser.add_argument("--watch-wavelet-weight", type=float, default=0.05)
    parser.add_argument("--priv-wavelet-weight", type=float, default=0.05)
    parser.add_argument("--distill-weight", type=float, default=0.08)
    parser.add_argument("--distill-temp", type=float, default=4.0)
    parser.add_argument("--teacher-cls-weight", type=float, default=0.80)
    parser.add_argument("--privileged-cls-weight", type=float, default=0.05)
    parser.add_argument("--cross-confidence-distill", action="store_true")
    parser.add_argument("--cross-confidence-min-weight", type=float, default=0.0)
    parser.add_argument(
        "--kd-gate-mode",
        type=str,
        default="none",
        choices=["none", "teacher_true_confidence", "student_true_confidence"],
    )
    parser.add_argument("--kd-gate-min-weight", type=float, default=0.0)
    parser.add_argument("--detach-standard-kd-teacher", action="store_true")
    parser.add_argument("--eval-aggregation", type=str, default="window", choices=["window", "session_mean"])
    parser.add_argument("--threshold-mode", type=str, default="val_search", choices=["val_search", "fixed"])
    parser.add_argument("--fixed-threshold", type=float, default=0.5)
    parser.add_argument("--watch-model-dim", type=int, default=192)
    parser.add_argument("--watch-transformer-layers", type=int, default=2)
    parser.add_argument("--watch-transformer-heads", type=int, default=4)
    parser.add_argument("--watch-fusion-hidden-dim", type=int, default=256)
    parser.add_argument("--watch-embed-dim", type=int, default=160)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cache-subjects", type=int, default=2)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--subject-aware-batching", action="store_true")
    parser.add_argument("--calm-sessions", nargs="*", default=["baseline"])
    parser.add_argument("--stress-sessions", nargs="*", default=["stress"])
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    manifests = discover_manifests(args.manifests_dir, args.subjects)
    if not manifests:
        raise ValueError(f"No manifests found in {args.manifests_dir}")

    output_dir = args.output_dir
    logs_dir = output_dir / "logs"
    ckpt_dir = output_dir / "checkpoints"
    csv_path = output_dir / "wesad_loso_privileged_scaled_motion_results.csv"
    summary_path = output_dir / "wesad_loso_privileged_scaled_motion_summary.txt"
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
        watch_ckpt: Path | None = None
        if not args.skip_watch_only:
            watch_log = logs_dir / f"{subject}_watch_only.log"
            watch_metrics_csv = logs_dir / f"{subject}_watch_only_metrics.csv"
            watch_ckpt = ckpt_dir / f"{subject}_watch_only.pt"
            watch_command = [
                sys.executable,
                "-m",
                "stress_ssl_distill.train_wesad_watch_scaled_motion",
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
                "--wavelet-weight",
                str(args.watch_wavelet_weight),
                "--selection-mode",
                "early_stop",
                "--monitor",
                args.monitor,
                "--threshold-metric",
                args.threshold_metric,
                "--early-stop-patience",
                str(args.early_stop_patience),
                "--seed",
                str(args.seed),
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
                "--calm-sessions",
                *args.calm_sessions,
                "--stress-sessions",
                *args.stress_sessions,
            ]
            if args.pin_memory:
                watch_command.append("--pin-memory")
            if args.subject_aware_batching:
                watch_command.append("--subject-aware-batching")
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

        init_watch_ckpt: Path | None = None
        if args.init_deploy_from_watch_only:
            if watch_ckpt is not None and watch_ckpt.exists():
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

        deploy_log = logs_dir / f"{subject}_deploy_watch.log"
        deploy_metrics_csv = logs_dir / f"{subject}_deploy_watch_metrics.csv"
        deploy_ckpt = ckpt_dir / f"{subject}_deploy_watch.pt"
        deploy_command = [
            sys.executable,
            "-m",
            "stress_ssl_distill.run_wesad_loso_eval_elastic_scaled_motion",
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
            "--weight-decay",
            str(args.weight_decay),
            "--label-smoothing",
            str(args.label_smoothing),
            "--focal-gamma",
            str(args.focal_gamma),
            "--teacher-cls-weight",
            str(args.teacher_cls_weight),
            "--privileged-cls-weight",
            str(args.privileged_cls_weight),
            "--distill-weight",
            str(args.distill_weight),
            "--distill-temp",
            str(args.distill_temp),
            "--wavelet-weight",
            str(args.priv_wavelet_weight),
            "--cross-confidence-min-weight",
            str(args.cross_confidence_min_weight),
            "--kd-gate-mode",
            args.kd_gate_mode,
            "--kd-gate-min-weight",
            str(args.kd_gate_min_weight),
            "--eval-aggregation",
            args.eval_aggregation,
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
            "scaled",
            "--calm-sessions",
            *args.calm_sessions,
            "--stress-sessions",
            *args.stress_sessions,
        ]
        if init_watch_ckpt is not None:
            deploy_command.extend(["--init-watch-checkpoint", str(init_watch_ckpt)])
        if args.pin_memory:
            deploy_command.append("--pin-memory")
        if args.cross_confidence_distill:
            deploy_command.append("--cross-confidence-distill")
        if args.detach_standard_kd_teacher:
            deploy_command.append("--detach-standard-kd-teacher")
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
                    "model": model_name,
                    "threshold": metrics["threshold"],
                    "acc": metrics["acc"],
                    "balanced_acc": metrics["balanced_acc"],
                    "f1": metrics["f1"],
                    "auroc": metrics["auroc"],
                    "positive_rate": metrics["positive_rate"],
                    "collapse": metrics["collapse"],
                    "positive_rate_error": metrics["positive_rate_error"],
                    "test_positive_prior": test_positive_prior,
                }
            )

    if baseline_rows:
        append_summary_block(summary_lines, "watch_only", baseline_rows)
    append_summary_block(summary_lines, "deploy_watch", deploy_rows)
    append_summary_block(summary_lines, "teacher", teacher_rows)

    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"Saved WESAD privileged scaled-motion LOSO CSV to {csv_path}")
    print(f"Saved WESAD privileged scaled-motion LOSO summary to {summary_path}")


if __name__ == "__main__":
    main()
