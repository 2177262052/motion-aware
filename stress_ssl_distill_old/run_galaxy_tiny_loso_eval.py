from __future__ import annotations

import argparse
import re
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd


TINY_PATTERN = re.compile(
    r"best_threshold=(?P<threshold>[-+]?\d*\.?\d+)\s+"
    r"best_test_acc=(?P<acc>[-+]?\d*\.?\d+)\s+"
    r"best_test_balanced_acc=(?P<balanced_acc>[-+]?\d*\.?\d+)\s+"
    r"best_test_f1=(?P<f1>[-+]?\d*\.?\d+)\s+"
    r"best_test_auroc=(?P<auroc>[-+]?\d*\.?\d+)\s+"
    r"best_test_positive_rate=(?P<positive_rate>[-+]?\d*\.?\d+)"
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


def parse_last_metrics(output: str, pattern: re.Pattern[str]) -> dict[str, float]:
    groups = None
    for item in pattern.finditer(output):
        groups = item.groupdict()
    if groups is None:
        raise ValueError("Could not parse tiny distillation metrics from command output.")
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
    parser = argparse.ArgumentParser(description="Run LOSO tiny-student distillation from per-subject watch teachers.")
    parser.add_argument("--manifests-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--teacher-kind", type=str, default="deploy_watch", choices=["deploy_watch", "watch_only"])
    parser.add_argument("--teacher-checkpoint-suffix", type=str, default="_deploy_watch.pt")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--monitor", type=str, default="balanced_acc")
    parser.add_argument("--early-stop-patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--focal-gamma", type=float, default=1.5)
    parser.add_argument("--hard-weight", type=float, default=1.0)
    parser.add_argument("--kd-weight", type=float, default=1.0)
    parser.add_argument("--feat-weight", type=float, default=0.25)
    parser.add_argument("--ranking-distill-weight", type=float, default=0.0)
    parser.add_argument("--distill-temp", type=float, default=2.0)
    parser.add_argument("--eval-aggregation", type=str, default="window", choices=["window", "session_mean"])
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--calm-sessions", nargs="*", default=["baseline"])
    parser.add_argument("--stress-sessions", nargs="*", default=["tsst-prep"])
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    manifests = discover_manifests(args.manifests_dir, args.subjects)
    if not manifests:
        raise ValueError(f"No manifests found in {args.manifests_dir}")

    output_dir = args.output_dir
    logs_dir = output_dir / "logs"
    ckpt_dir = output_dir / "checkpoints"
    csv_path = output_dir / "galaxy_tiny_loso_results.csv"
    summary_path = output_dir / "galaxy_tiny_loso_summary.txt"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    title = f"tiny_from_{args.teacher_kind}"
    summary_lines: list[str] = []
    csv_rows: list[dict[str, object]] = []
    tiny_rows: list[dict[str, float]] = []

    for subject, manifest_path in manifests:
        teacher_path = args.teacher_checkpoint_dir / f"{subject}{args.teacher_checkpoint_suffix}"
        if not teacher_path.exists():
            raise FileNotFoundError(f"Teacher checkpoint not found for {subject}: {teacher_path}")

        summary_lines.append(f"[{subject}]")
        summary_lines.append(f"teacher_path={teacher_path}")
        test_positive_prior = load_test_positive_prior(
            manifest_path,
            calm_sessions=list(args.calm_sessions),
            stress_sessions=list(args.stress_sessions),
        )
        summary_lines.append(f"test_positive_prior={test_positive_prior:.4f}")

        tiny_log = logs_dir / f"{subject}_tiny.log"
        tiny_metrics_csv = logs_dir / f"{subject}_tiny_metrics.csv"
        tiny_ckpt = ckpt_dir / f"{subject}_tiny.pt"
        tiny_command = [
            sys.executable,
            "-m",
            "stress_ssl_distill.train_galaxy_tiny_distill",
            "--manifest",
            str(manifest_path),
            "--dataset-root",
            str(args.dataset_root),
            "--teacher-path",
            str(teacher_path),
            "--teacher-kind",
            args.teacher_kind,
            "--save-path",
            str(tiny_ckpt),
            "--metrics-path",
            str(tiny_metrics_csv),
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
            "--hard-weight",
            str(args.hard_weight),
            "--kd-weight",
            str(args.kd_weight),
            "--feat-weight",
            str(args.feat_weight),
            "--ranking-distill-weight",
            str(args.ranking_distill_weight),
            "--distill-temp",
            str(args.distill_temp),
            "--selection-mode",
            "early_stop",
            "--monitor",
            args.monitor,
            "--early-stop-patience",
            str(args.early_stop_patience),
            "--eval-aggregation",
            args.eval_aggregation,
            "--num-workers",
            str(args.num_workers),
            "--seed",
            str(args.seed),
            "--calm-sessions",
            *args.calm_sessions,
            "--stress-sessions",
            *args.stress_sessions,
        ]
        if args.pin_memory:
            tiny_command.append("--pin-memory")
        tiny_output = run_and_capture(tiny_command, cwd=repo_root, log_path=tiny_log)
        tiny_metrics = parse_last_metrics(tiny_output, TINY_PATTERN)
        tiny_metrics["collapse"] = float(collapse_flag(tiny_metrics["positive_rate"]))
        tiny_metrics["positive_rate_error"] = abs(tiny_metrics["positive_rate"] - test_positive_prior)
        tiny_rows.append(tiny_metrics)

        summary_lines.append(title + " " + " ".join(f"{key}={value:.4f}" for key, value in tiny_metrics.items() if key != "collapse"))
        summary_lines.append("")

        csv_rows.append(
            {
                "subject": subject,
                "manifest": str(manifest_path),
                "teacher_path": str(teacher_path),
                "test_positive_prior": test_positive_prior,
                "tiny_threshold": tiny_metrics["threshold"],
                "tiny_acc": tiny_metrics["acc"],
                "tiny_balanced_acc": tiny_metrics["balanced_acc"],
                "tiny_f1": tiny_metrics["f1"],
                "tiny_auroc": tiny_metrics["auroc"],
                "tiny_positive_rate": tiny_metrics["positive_rate"],
                "tiny_collapse": tiny_metrics["collapse"],
                "tiny_positive_rate_error": tiny_metrics["positive_rate_error"],
            }
        )

    summary_lines.append("[summary]")
    append_summary_block(summary_lines, title, tiny_rows)
    summary_text = "\n".join(summary_lines) + "\n"

    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = pd.DataFrame(csv_rows)
        writer.to_csv(csv_file, index=False)
    summary_path.write_text(summary_text, encoding="utf-8")

    print(summary_text, end="")
    print(f"Saved tiny LOSO CSV to {csv_path}")
    print(f"Saved tiny LOSO summary to {summary_path}")


if __name__ == "__main__":
    main()
