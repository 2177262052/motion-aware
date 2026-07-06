from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _extend_sessions(command: list[str], flag: str, values: list[str]) -> None:
    command.append(flag)
    command.extend(values)


def _maybe_append_flag(command: list[str], enabled: bool, flag: str) -> None:
    if enabled:
        command.append(flag)


def _run(command: list[str], dry_run: bool) -> None:
    print(" ".join(command), flush=True)
    if dry_run:
        return
    subprocess.run(command, check=True)


def build_galaxy_command(args: argparse.Namespace) -> list[str]:
    calm_sessions = args.calm_sessions or ["baseline"]
    stress_sessions = args.stress_sessions or ["tsst-prep"]
    deploy_batch_size = args.deploy_batch_size if args.deploy_batch_size is not None else 16

    command = [
        sys.executable,
        "-m",
        "stress_ssl_distill.run_galaxy_loso_eval_elastic_scaled_motion",
        "--manifests-dir",
        str(args.manifests_dir),
        "--dataset-root",
        str(args.dataset_root),
        "--output-dir",
        str(args.output_dir),
        "--device",
        args.device,
        "--epochs",
        str(args.epochs),
        "--monitor",
        args.monitor,
        "--threshold-metric",
        args.threshold_metric,
        "--selection-target",
        "watch",
        "--early-stop-patience",
        str(args.early_stop_patience),
        "--seed",
        str(args.seed),
        "--watch-batch-size",
        str(args.watch_batch_size),
        "--deploy-batch-size",
        str(deploy_batch_size),
        "--lr",
        str(args.lr),
        "--num-workers",
        str(args.num_workers),
        "--deploy-watch-backbone",
        "wavelet_guided",
        "--deploy-watch-enhancement",
        "motion_disentangled",
        "--watch-cls-weight",
        "1.0",
        "--teacher-cls-weight",
        str(args.teacher_cls_weight),
        "--teacher-fused-cls-weight",
        "0.0",
        "--e4-cls-weight",
        str(args.e4_cls_weight),
        "--rhythm-weight",
        str(args.rhythm_weight),
        "--wavelet-weight",
        str(args.wavelet_weight),
        "--align-weight",
        "0.0",
        "--distill-weight",
        "0.0",
        "--distill-temp",
        str(args.distill_temp),
        "--ranking-distill-weight",
        "0.0",
        "--distribution-weight",
        "0.0",
        "--session-consistency-weight",
        "0.0",
        "--cross-confidence-min-weight",
        "0.0",
        "--kd-gate-mode",
        "none",
        "--kd-gate-min-weight",
        "0.0",
        "--reliability-distill-weight",
        "0.0",
        "--correction-cls-weight",
        "0.0",
        "--correction-nondegradation-weight",
        "0.0",
        "--correction-align-weight",
        "0.0",
        "--correction-base-anchor-weight",
        "0.0",
        "--correction-margin",
        "0.0",
        "--correction-scale-init",
        "0.05",
        "--correction-alpha-init-bias",
        "-3.0",
        "--correction-alpha-max",
        "0.35",
        "--correction-mode",
        "logit_mix",
        "--alpha-helpfulness-weight",
        "0.0",
        "--alpha-help-margin",
        "0.0",
        "--alpha-sparsity-weight",
        "0.0",
        "--elastic-residual-weight",
        "0.0",
        "--elastic-alpha-target-weight",
        "0.0",
        "--elastic-reliability-temp",
        "0.25",
        "--elastic-uncertainty-temp",
        "1.0",
        "--elastic-label-margin",
        "2.0",
    ]
    _extend_sessions(command, "--calm-sessions", calm_sessions)
    _extend_sessions(command, "--stress-sessions", stress_sessions)
    if args.subjects:
        command.append("--subjects")
        command.extend(args.subjects)
    if not args.include_watch_only:
        command.append("--skip-watch-only")
    _maybe_append_flag(command, args.pin_memory, "--pin-memory")
    return command


def build_wesad_command(args: argparse.Namespace) -> list[str]:
    calm_sessions = args.calm_sessions or ["baseline"]
    stress_sessions = args.stress_sessions or ["stress"]
    deploy_batch_size = args.deploy_batch_size if args.deploy_batch_size is not None else 24

    command = [
        sys.executable,
        "-m",
        "stress_ssl_distill.run_wesad_loso_eval_elastic_scaled_motion",
        "--manifests-dir",
        str(args.manifests_dir),
        "--dataset-root",
        str(args.dataset_root),
        "--dataset-kind",
        "wesad",
        "--output-dir",
        str(args.output_dir),
        "--device",
        args.device,
        "--epochs",
        str(args.epochs),
        "--monitor",
        args.monitor,
        "--threshold-metric",
        args.threshold_metric,
        "--selection-target",
        "watch",
        "--early-stop-patience",
        str(args.early_stop_patience),
        "--seed",
        str(args.seed),
        "--watch-batch-size",
        str(args.watch_batch_size),
        "--deploy-batch-size",
        str(deploy_batch_size),
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--label-smoothing",
        str(args.label_smoothing),
        "--focal-gamma",
        str(args.focal_gamma),
        "--deploy-contrastive-weight",
        "0.0",
        "--watch-wavelet-weight",
        str(args.wavelet_weight),
        "--priv-wavelet-weight",
        str(args.wavelet_weight),
        "--temperature",
        "0.1",
        "--deploy-watch-backbone",
        "wavelet_guided",
        "--deploy-watch-enhancement",
        "motion_disentangled",
        "--teacher-cls-weight",
        str(args.teacher_cls_weight),
        "--privileged-cls-weight",
        str(args.privileged_cls_weight),
        "--distill-weight",
        "0.0",
        "--embedding-align-weight",
        "0.0",
        "--margin-match-weight",
        "0.0",
        "--normalized-margin-align-weight",
        "0.0",
        "--ranking-distill-weight",
        "0.0",
        "--distribution-weight",
        "0.0",
        "--subject-center-stability-weight",
        "0.0",
        "--validation-threshold-stability-weight",
        "0.0",
        "--session-consistency-weight",
        "0.0",
        "--distill-gating",
        "none",
        "--distill-disagreement-weight",
        "0.25",
        "--teacher-confidence-threshold",
        "1.0",
        "--teacher-confidence-temperature",
        "0.5",
        "--min-distill-weight",
        "0.2",
        "--cross-confidence-min-weight",
        "0.0",
        "--kd-gate-mode",
        "none",
        "--kd-gate-min-weight",
        "0.0",
        "--distill-temp",
        str(args.distill_temp),
        "--reliability-distill-weight",
        "0.0",
        "--correction-cls-weight",
        "0.0",
        "--correction-nondegradation-weight",
        "0.0",
        "--correction-align-weight",
        "0.0",
        "--correction-base-anchor-weight",
        "0.0",
        "--correction-margin",
        "0.0",
        "--correction-scale-init",
        "0.05",
        "--correction-alpha-init-bias",
        "-2.0",
        "--correction-alpha-max",
        "0.35",
        "--correction-mode",
        "logit_mix",
        "--alpha-helpfulness-weight",
        "0.0",
        "--alpha-help-margin",
        "0.0",
        "--alpha-sparsity-weight",
        "0.0",
        "--elastic-residual-weight",
        "0.0",
        "--elastic-alpha-target-weight",
        "0.0",
        "--elastic-reliability-temp",
        "0.25",
        "--elastic-uncertainty-temp",
        "1.0",
        "--elastic-label-margin",
        "2.0",
        "--baseline-relative-weight",
        "0.0",
        "--baseline-relative-margin",
        "0.2",
        "--eval-aggregation",
        "window",
        "--threshold-mode",
        "val_search",
        "--fixed-threshold",
        "0.5",
        "--num-workers",
        str(args.num_workers),
        "--cache-subjects",
        str(args.cache_subjects),
    ]
    _extend_sessions(command, "--calm-sessions", calm_sessions)
    _extend_sessions(command, "--stress-sessions", stress_sessions)
    if args.subjects:
        command.append("--subjects")
        command.extend(args.subjects)
    if not args.include_watch_only:
        command.append("--skip-watch-only")
    _maybe_append_flag(command, args.pin_memory, "--pin-memory")
    return command


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the fairest no-KD privileged-auxiliary control: privileged teacher "
            "and auxiliary objectives are trained, but all teacher-to-student KD, "
            "gating, correction, and elastic transfer terms are disabled."
        )
    )
    parser.add_argument("--dataset-kind", choices=["galaxy", "wesad"], required=True)
    parser.add_argument("--manifests-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--include-watch-only", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--monitor", type=str, default="auroc")
    parser.add_argument("--threshold-metric", type=str, default="balanced_acc")
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--watch-batch-size", type=int, default=32)
    parser.add_argument("--deploy-batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--focal-gamma", type=float, default=1.5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--cache-subjects", type=int, default=15)
    parser.add_argument("--calm-sessions", nargs="*", default=None)
    parser.add_argument("--stress-sessions", nargs="*", default=None)
    parser.add_argument("--teacher-cls-weight", type=float, default=0.80)
    parser.add_argument("--privileged-cls-weight", type=float, default=0.05)
    parser.add_argument("--e4-cls-weight", type=float, default=0.05)
    parser.add_argument("--rhythm-weight", type=float, default=0.15)
    parser.add_argument("--wavelet-weight", type=float, default=0.05)
    parser.add_argument("--distill-temp", type=float, default=4.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    command = build_galaxy_command(args) if args.dataset_kind == "galaxy" else build_wesad_command(args)
    _run(command, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
