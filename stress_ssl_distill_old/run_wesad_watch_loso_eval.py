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


WATCH_PATTERN = re.compile(
    r"best_threshold=(?P<threshold>[-+]?\d*\.?\d+)\s+"
    r"best_test_acc=(?P<acc>[-+]?\d*\.?\d+)\s+"
    r"best_test_balanced_acc=(?P<balanced_acc>[-+]?\d*\.?\d+)\s+"
    r"best_test_f1=(?P<f1>[-+]?\d*\.?\d+)\s+"
    r"best_test_auroc=(?P<auroc>[-+]?\d*\.?\d+)\s+"
    r"best_test_positive_rate=(?P<positive_rate>[-+]?\d*\.?\d+)"
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


def parse_last_metrics(output: str) -> dict[str, float]:
    groups = None
    for item in WATCH_PATTERN.finditer(output):
        groups = item.groupdict()
    if groups is None:
        raise ValueError("Could not parse WESAD watch-only metrics from command output.")
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
    parser = argparse.ArgumentParser(description="Run watch-only LOSO evaluation for WESAD.")
    parser.add_argument("--manifests-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", "--wesad-root", dest="dataset_root", type=Path, required=True)
    parser.add_argument("--dataset-kind", type=str, default="wesad", choices=["wesad", "catsa"])
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
    parser.add_argument("--baseline-relative-weight", type=float, default=0.0)
    parser.add_argument("--baseline-relative-margin", type=float, default=0.2)
    parser.add_argument("--eval-aggregation", type=str, default="window", choices=["window", "session_mean"])
    parser.add_argument("--threshold-mode", type=str, default="val_search", choices=["val_search", "fixed"])
    parser.add_argument("--fixed-threshold", type=float, default=0.5)
    parser.add_argument("--watch-model-dim", type=int, default=192)
    parser.add_argument("--watch-transformer-layers", type=int, default=2)
    parser.add_argument("--watch-transformer-heads", type=int, default=4)
    parser.add_argument("--watch-fusion-hidden-dim", type=int, default=256)
    parser.add_argument("--watch-embed-dim", type=int, default=160)
    parser.add_argument("--watch-enhancement", type=str, default="none", choices=["none", "motion_disentangled", "acc_concat"])
    parser.add_argument("--watch-motion-mode", type=str, default="strong", choices=["strong", "residual", "scaled"])
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
    csv_path = output_dir / f"{output_prefix}_watch_loso_results.csv"
    summary_path = output_dir / f"{output_prefix}_watch_loso_summary.txt"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    title = f"{args.dataset_kind}_watch_only_{args.model_type}_{args.watch_enhancement}"
    summary_lines: list[str] = []
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
            "stress_ssl_distill.train_wesad_watch",
            "--manifest",
            str(manifest_path),
            "--dataset-root",
            str(args.dataset_root),
            "--dataset-kind",
            args.dataset_kind,
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
            "--baseline-relative-weight",
            str(args.baseline_relative_weight),
            "--baseline-relative-margin",
            str(args.baseline_relative_margin),
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
            "--watch-motion-mode",
            args.watch_motion_mode,
            "--calm-sessions",
            *args.calm_sessions,
            "--stress-sessions",
            *args.stress_sessions,
        ]
        if args.pin_memory:
            command.append("--pin-memory")
        if args.baseline_reference:
            command.append("--baseline-reference")
        if args.subject_aware_batching:
            command.append("--subject-aware-batching")

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
    print(f"Saved WESAD watch LOSO CSV to {csv_path}")
    print(f"Saved WESAD watch LOSO summary to {summary_path}")


if __name__ == "__main__":
    main()
