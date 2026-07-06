from __future__ import annotations

import argparse
import csv
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd


TEACHER_PATTERN = re.compile(
    r"best_teacher_threshold=(?P<threshold>[-+]?\d*\.?\d+)\s+"
    r"best_teacher_test_acc=(?P<acc>[-+]?\d*\.?\d+)\s+"
    r"best_teacher_test_balanced_acc=(?P<balanced_acc>[-+]?\d*\.?\d+)\s+"
    r"best_teacher_test_f1=(?P<f1>[-+]?\d*\.?\d+)\s+"
    r"best_teacher_test_auroc=(?P<auroc>[-+]?\d*\.?\d+)\s+"
    r"best_teacher_test_positive_rate=(?P<positive_rate>[-+]?\d*\.?\d+)"
)

WATCH_PATTERN = re.compile(
    r"best_watch_threshold=(?P<threshold>[-+]?\d*\.?\d+)\s+"
    r"best_watch_test_acc=(?P<acc>[-+]?\d*\.?\d+)\s+"
    r"best_watch_test_balanced_acc=(?P<balanced_acc>[-+]?\d*\.?\d+)\s+"
    r"best_watch_test_f1=(?P<f1>[-+]?\d*\.?\d+)\s+"
    r"best_watch_test_auroc=(?P<auroc>[-+]?\d*\.?\d+)\s+"
    r"best_watch_test_positive_rate=(?P<positive_rate>[-+]?\d*\.?\d+)"
)


VARIANTS: dict[str, dict[str, float]] = {
    # Approximate current Galaxy teacher recipe, but with transfer losses off so the
    # diagnostic isolates the teacher rather than student distillation.
    "current_teacher": {
        "watch_cls_weight": 1.0,
        "teacher_cls_weight": 0.80,
        "teacher_fused_cls_weight": 0.0,
        "e4_cls_weight": 0.05,
        "rhythm_weight": 0.15,
        "wavelet_weight": 0.05,
    },
    # Main sanity check: is the supervised teacher itself good once auxiliaries are removed?
    "teacher_cls_only": {
        "watch_cls_weight": 0.0,
        "teacher_cls_weight": 1.0,
        "teacher_fused_cls_weight": 0.0,
        "e4_cls_weight": 0.0,
        "rhythm_weight": 0.0,
        "wavelet_weight": 0.0,
    },
    # One-auxiliary-at-a-time checks. These tell us whether an auxiliary is helping
    # the teacher or silently pulling it away from the stress label.
    "teacher_e4_aux_only": {
        "watch_cls_weight": 0.0,
        "teacher_cls_weight": 1.0,
        "teacher_fused_cls_weight": 0.0,
        "e4_cls_weight": 0.05,
        "rhythm_weight": 0.0,
        "wavelet_weight": 0.0,
    },
    "teacher_rhythm_aux_only": {
        "watch_cls_weight": 0.0,
        "teacher_cls_weight": 1.0,
        "teacher_fused_cls_weight": 0.0,
        "e4_cls_weight": 0.0,
        "rhythm_weight": 0.15,
        "wavelet_weight": 0.0,
    },
    "teacher_wavelet_aux_only": {
        "watch_cls_weight": 0.0,
        "teacher_cls_weight": 1.0,
        "teacher_fused_cls_weight": 0.0,
        "e4_cls_weight": 0.0,
        "rhythm_weight": 0.0,
        "wavelet_weight": 0.05,
    },
    # Current auxiliaries minus one component at a time.
    "teacher_no_rhythm": {
        "watch_cls_weight": 0.0,
        "teacher_cls_weight": 1.0,
        "teacher_fused_cls_weight": 0.0,
        "e4_cls_weight": 0.05,
        "rhythm_weight": 0.0,
        "wavelet_weight": 0.05,
    },
    "teacher_no_wavelet": {
        "watch_cls_weight": 0.0,
        "teacher_cls_weight": 1.0,
        "teacher_fused_cls_weight": 0.0,
        "e4_cls_weight": 0.05,
        "rhythm_weight": 0.15,
        "wavelet_weight": 0.0,
    },
    "teacher_no_e4_aux": {
        "watch_cls_weight": 0.0,
        "teacher_cls_weight": 1.0,
        "teacher_fused_cls_weight": 0.0,
        "e4_cls_weight": 0.0,
        "rhythm_weight": 0.15,
        "wavelet_weight": 0.05,
    },
    # Tests whether directly supervising the fused privileged path stabilizes teacher features.
    "teacher_fused_aux": {
        "watch_cls_weight": 0.0,
        "teacher_cls_weight": 1.0,
        "teacher_fused_cls_weight": 0.20,
        "e4_cls_weight": 0.05,
        "rhythm_weight": 0.15,
        "wavelet_weight": 0.05,
    },
}


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
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    chunks: list[str] = []
    assert process.stdout is not None
    with log_path.open("w", encoding="utf-8") as log_file:
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_file.write(line)
            log_file.flush()
            chunks.append(line)
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
        df = df[df["session"].isin(set(calm_sessions) | set(stress_sessions))]
    if df.empty or "label" not in df.columns:
        return 0.5
    labels = pd.to_numeric(df["label"], errors="coerce").dropna()
    if labels.empty:
        return 0.5
    return float(labels.mean())


def metric_prefix(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for variant in sorted({str(row["variant"]) for row in rows}):
        subset = [row for row in rows if row["variant"] == variant]
        summary: dict[str, object] = {"variant": variant, "n": len(subset)}
        for prefix in ("teacher", "watch"):
            for metric in ("balanced_acc", "auroc", "f1", "collapse", "positive_rate_error"):
                values = [float(row[f"{prefix}_{metric}"]) for row in subset]
                mean, std = mean_std(values)
                name = "collapse_rate" if metric == "collapse" else metric
                summary[f"{prefix}_{name}_mean"] = mean
                summary[f"{prefix}_{name}_std"] = std
        out.append(summary)
    return sorted(out, key=lambda row: float(row["teacher_auroc_mean"]), reverse=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Galaxy teacher ablations without touching the main model code.")
    parser.add_argument("--manifests-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--variants", nargs="*", default=["current_teacher", "teacher_cls_only", "teacher_no_rhythm", "teacher_no_wavelet"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--monitor", type=str, default="auroc", choices=["acc", "balanced_acc", "f1", "auroc"])
    parser.add_argument("--threshold-metric", type=str, default="balanced_acc", choices=["monitor", "acc", "balanced_acc", "f1", "auroc"])
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--focal-gamma", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--watch-enhancement", type=str, default="motion_disentangled", choices=["none", "motion_disentangled"])
    parser.add_argument("--priv-schedule", type=str, default="constant", choices=["constant", "linear", "cosine"])
    parser.add_argument("--priv-floor", type=float, default=1.0)
    parser.add_argument("--calm-sessions", nargs="*", default=["baseline"])
    parser.add_argument("--stress-sessions", nargs="*", default=["tsst-prep"])
    args = parser.parse_args()

    unknown_variants = sorted(set(args.variants) - set(VARIANTS))
    if unknown_variants:
        raise ValueError(f"Unknown variants: {unknown_variants}. Available: {sorted(VARIANTS)}")

    repo_root = Path(__file__).resolve().parent.parent
    manifests = discover_manifests(args.manifests_dir, args.subjects)
    if not manifests:
        raise ValueError(f"No Galaxy LOSO manifests found in {args.manifests_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = args.output_dir / "logs"
    ckpt_dir = args.output_dir / "checkpoints"
    rows: list[dict[str, object]] = []

    for variant in args.variants:
        weights = VARIANTS[variant]
        for subject, manifest_path in manifests:
            print(f"\n[{variant}] {subject}", flush=True)
            variant_dir = ckpt_dir / variant
            log_path = logs_dir / variant / f"{subject}.log"
            metrics_path = logs_dir / variant / f"{subject}_metrics.csv"
            ckpt_path = variant_dir / f"{subject}.pt"
            variant_dir.mkdir(parents=True, exist_ok=True)

            command = [
                sys.executable,
                "-m",
                "stress_ssl_distill.train_galaxy_privileged",
                "--manifest",
                str(manifest_path),
                "--dataset-root",
                str(args.dataset_root),
                "--save-path",
                str(ckpt_path),
                "--metrics-path",
                str(metrics_path),
                "--device",
                args.device,
                "--epochs",
                str(args.epochs),
                "--selection-mode",
                "early_stop",
                "--selection-target",
                "teacher",
                "--monitor",
                args.monitor,
                "--threshold-metric",
                args.threshold_metric,
                "--early-stop-patience",
                str(args.early_stop_patience),
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
                "--num-workers",
                str(args.num_workers),
                "--seed",
                str(args.seed),
                "--watch-backbone",
                "wavelet_guided",
                "--watch-enhancement",
                args.watch_enhancement,
                "--priv-schedule",
                args.priv_schedule,
                "--priv-floor",
                str(args.priv_floor),
                "--watch-cls-weight",
                str(weights["watch_cls_weight"]),
                "--teacher-cls-weight",
                str(weights["teacher_cls_weight"]),
                "--teacher-fused-cls-weight",
                str(weights["teacher_fused_cls_weight"]),
                "--e4-cls-weight",
                str(weights["e4_cls_weight"]),
                "--rhythm-weight",
                str(weights["rhythm_weight"]),
                "--wavelet-weight",
                str(weights["wavelet_weight"]),
                "--align-weight",
                "0.0",
                "--distill-weight",
                "0.0",
                "--ranking-distill-weight",
                "0.0",
                "--distribution-weight",
                "0.0",
                "--session-consistency-weight",
                "0.0",
                "--correction-cls-weight",
                "0.0",
                "--correction-nondegradation-weight",
                "0.0",
                "--correction-align-weight",
                "0.0",
                "--correction-base-anchor-weight",
                "0.0",
                "--calm-sessions",
                *args.calm_sessions,
                "--stress-sessions",
                *args.stress_sessions,
            ]
            if args.pin_memory:
                command.append("--pin-memory")

            output = run_and_capture(command, cwd=repo_root, log_path=log_path)
            teacher_metrics = parse_last_metrics(output, TEACHER_PATTERN, "teacher")
            watch_metrics = parse_last_metrics(output, WATCH_PATTERN, "watch")
            prior = load_test_positive_prior(manifest_path, args.calm_sessions, args.stress_sessions)
            for metrics in (teacher_metrics, watch_metrics):
                metrics["collapse"] = float(collapse_flag(metrics["positive_rate"]))
                metrics["positive_rate_error"] = abs(metrics["positive_rate"] - prior)

            row: dict[str, object] = {
                "variant": variant,
                "subject": subject,
                "manifest": str(manifest_path),
                "test_positive_prior": prior,
                **metric_prefix("teacher", teacher_metrics),
                **metric_prefix("watch", watch_metrics),
                **{key: value for key, value in weights.items()},
            }
            rows.append(row)

            pd.DataFrame(rows).to_csv(args.output_dir / "galaxy_teacher_diagnostic_per_subject.csv", index=False)
            pd.DataFrame(summarize(rows)).to_csv(args.output_dir / "galaxy_teacher_diagnostic_summary.csv", index=False)

    summary_rows = summarize(rows)
    summary_path = args.output_dir / "galaxy_teacher_diagnostic_summary.csv"
    per_subject_path = args.output_dir / "galaxy_teacher_diagnostic_per_subject.csv"
    pd.DataFrame(rows).to_csv(per_subject_path, index=False)
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    text_path = args.output_dir / "galaxy_teacher_diagnostic_summary.txt"
    lines = ["[Galaxy teacher diagnostic summary]"]
    for row in summary_rows:
        lines.append(
            " ".join(
                [
                    str(row["variant"]),
                    f"n={int(row['n'])}",
                    f"teacher_ba={float(row['teacher_balanced_acc_mean']):.4f}+/-{float(row['teacher_balanced_acc_std']):.4f}",
                    f"teacher_auroc={float(row['teacher_auroc_mean']):.4f}+/-{float(row['teacher_auroc_std']):.4f}",
                    f"teacher_f1={float(row['teacher_f1_mean']):.4f}+/-{float(row['teacher_f1_std']):.4f}",
                    f"teacher_collapse={float(row['teacher_collapse_rate_mean']):.4f}",
                    f"teacher_pre={float(row['teacher_positive_rate_error_mean']):.4f}+/-{float(row['teacher_positive_rate_error_std']):.4f}",
                ]
            )
        )
    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"Saved per-subject diagnostics to {per_subject_path}")
    print(f"Saved summary diagnostics to {summary_path}")
    print(f"Saved text summary to {text_path}")


if __name__ == "__main__":
    main()
