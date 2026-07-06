from __future__ import annotations

import argparse
import os
import statistics
import subprocess
import sys
from pathlib import Path

import pandas as pd

from .run_galaxy_watch_loso_eval import (
    append_summary_block,
    collapse_flag,
    load_test_positive_prior,
    parse_last_metrics,
)


def discover_manifests(manifests_dir: Path, subjects: list[str] | None) -> list[tuple[str, Path]]:
    requested = {subject.strip() for subject in subjects or [] if subject.strip()}
    manifests: dict[str, Path] = {}
    patterns = ("galaxy_*_loso_val.csv", "*_loso_val.csv")
    for pattern in patterns:
        for path in sorted(manifests_dir.glob(pattern)):
            stem = path.stem
            subject = stem
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
    parser = argparse.ArgumentParser(
        description="Run Galaxy PPG watch-only LOSO using the historical scale_logit motion FiLM."
    )
    parser.add_argument("--manifests-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument(
        "--model-type",
        type=str,
        default="wavelet_guided",
        choices=["wavelet_guided", "resnet18_1d", "resnet34_1d", "resnet50_1d"],
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--monitor", type=str, default="balanced_acc")
    parser.add_argument("--threshold-metric", type=str, default="monitor", choices=["monitor", "acc", "balanced_acc", "f1", "auroc"])
    parser.add_argument("--early-stop-patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--focal-gamma", type=float, default=1.5)
    parser.add_argument("--contrastive-weight", type=float, default=0.15)
    parser.add_argument("--wavelet-weight", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--eval-aggregation", type=str, default="window", choices=["window", "session_mean"])
    parser.add_argument("--watch-model-dim", type=int, default=192)
    parser.add_argument("--watch-transformer-layers", type=int, default=2)
    parser.add_argument("--watch-transformer-heads", type=int, default=4)
    parser.add_argument("--watch-fusion-hidden-dim", type=int, default=256)
    parser.add_argument("--watch-embed-dim", type=int, default=160)
    parser.add_argument("--watch-enhancement", type=str, default="motion_disentangled", choices=["none", "motion_disentangled"])
    parser.add_argument("--watch-motion-mode", type=str, default="strong", choices=["strong", "residual"])
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--calm-sessions", nargs="*", default=["baseline"])
    parser.add_argument("--stress-sessions", nargs="*", default=["tsst-prep"])
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
    csv_path = output_dir / "galaxy_watch_scaled_motion_loso_results.csv"
    summary_path = output_dir / "galaxy_watch_scaled_motion_loso_summary.txt"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    title = f"watch_only_{args.model_type}_{args.watch_enhancement}_scaled_motion"
    summary_lines: list[str] = [
        "scaled_motion_compat=on",
        "motion_formula=x*(1+sigmoid(scale_logit)*tanh(gamma))+sigmoid(scale_logit)*beta",
        "scale_logit_init=-2.0",
        "",
    ]
    csv_rows: list[dict[str, object]] = []
    watch_rows: list[dict[str, float]] = []

    for subject, manifest_path in manifests:
        summary_lines.append(f"[{subject}]")
        test_positive_prior = load_test_positive_prior(
            manifest_path,
            calm_sessions=list(args.calm_sessions),
            stress_sessions=list(args.stress_sessions),
        )
        summary_lines.append(f"test_positive_prior={test_positive_prior:.4f}")

        log_path = logs_dir / f"{subject}.log"
        metrics_csv = logs_dir / f"{subject}_metrics.csv"
        ckpt_path = ckpt_dir / f"{subject}.pt"
        command = [
            sys.executable,
            "-m",
            "stress_ssl_distill.train_galaxy_watch_scaled_motion",
            "--manifest",
            str(manifest_path),
            "--dataset-root",
            str(args.dataset_root),
            "--save-path",
            str(ckpt_path),
            "--metrics-path",
            str(metrics_csv),
            "--device",
            args.device,
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
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
            str(args.wavelet_weight),
            "--temperature",
            str(args.temperature),
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
            "--num-workers",
            str(args.num_workers),
            "--model-type",
            args.model_type,
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
            args.watch_enhancement,
            "--calm-sessions",
            *args.calm_sessions,
            "--stress-sessions",
            *args.stress_sessions,
        ]
        if args.pin_memory:
            command.append("--pin-memory")

        output = run_and_capture(command, cwd=repo_root, log_path=log_path)
        metrics = parse_last_metrics(output)
        metrics["collapse"] = float(collapse_flag(metrics["positive_rate"]))
        metrics["positive_rate_error"] = abs(metrics["positive_rate"] - test_positive_prior)
        watch_rows.append(metrics)

        summary_lines.append(title + " " + " ".join(f"{key}={value:.4f}" for key, value in metrics.items() if key != "collapse"))
        summary_lines.append("")

        csv_rows.append(
            {
                "subject": subject,
                "manifest": str(manifest_path),
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
    append_summary_block(summary_lines, title, watch_rows)
    summary_text = "\n".join(summary_lines) + "\n"

    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    summary_path.write_text(summary_text, encoding="utf-8")

    print(summary_text, end="")
    print(f"Saved Galaxy scaled-motion watch LOSO CSV to {csv_path}")
    print(f"Saved Galaxy scaled-motion watch LOSO summary to {summary_path}")

    ba_values = [row["balanced_acc"] for row in watch_rows]
    auroc_values = [row["auroc"] for row in watch_rows]
    if ba_values:
        ba_mean, ba_std = mean_std(ba_values)
        auroc_mean, auroc_std = mean_std(auroc_values)
        print(f"quick_check balanced_acc_mean={ba_mean:.4f} balanced_acc_std={ba_std:.4f}")
        print(f"quick_check auroc_mean={auroc_mean:.4f} auroc_std={auroc_std:.4f}")


if __name__ == "__main__":
    main()
