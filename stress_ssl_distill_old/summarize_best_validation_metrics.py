from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = ["acc", "balanced_acc", "f1", "auroc"]
KNOWN_SUFFIXES = [
    "_teacher_only_metrics",
    "_deploy_watch_metrics",
    "_watch_only_metrics",
    "_metrics",
]


def subject_from_metrics_path(path: Path) -> str:
    name = path.stem
    for suffix in KNOWN_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def best_validation_row(path: Path, monitor: str, dataset_kind: str, method: str) -> dict[str, object]:
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Empty metrics file: {path}")

    monitor_col = f"val_{monitor}"
    if monitor_col not in frame.columns:
        raise ValueError(f"{path} does not contain required column {monitor_col}")

    scores = pd.to_numeric(frame[monitor_col], errors="coerce")
    best_index = int(scores.idxmax()) if scores.notna().any() else int(frame.index[-1])
    row = frame.loc[best_index]

    output: dict[str, object] = {
        "dataset": dataset_kind,
        "method": method,
        "subject": subject_from_metrics_path(path),
        "source_file": str(path),
        "selection_monitor": monitor,
        "best_epoch": int(row["epoch"]) if "epoch" in frame.columns and pd.notna(row["epoch"]) else best_index + 1,
        "best_val_score": float(row[monitor_col]) if pd.notna(row[monitor_col]) else float("nan"),
        "val_threshold": float(row["val_threshold"]) if "val_threshold" in frame.columns and pd.notna(row["val_threshold"]) else float("nan"),
    }
    for metric in METRICS:
        val_col = f"val_{metric}"
        test_col = f"test_{metric}"
        output[val_col] = float(row[val_col]) if val_col in frame.columns and pd.notna(row[val_col]) else float("nan")
        # Keep test columns only for sanity checks; do not use them for regime selection.
        output[test_col] = float(row[test_col]) if test_col in frame.columns and pd.notna(row[test_col]) else float("nan")
    return output


def mean_std(series: pd.Series) -> tuple[float, float]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if len(values) == 0:
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values.iloc[0]), 0.0
    return float(values.mean()), float(values.std(ddof=1))


def write_summary(frame: pd.DataFrame, output_path: Path, dataset_kind: str, method: str, monitor: str) -> None:
    lines = [
        f"# {dataset_kind.upper()} {method} Best-Validation Summary",
        "",
        "This summary is selected by validation metrics only. Test columns, if present in the CSV, are included only for sanity checking and must not be used for regime selection.",
        "",
        f"- Selection monitor: val_{monitor}",
        f"- Folds: {len(frame)}",
        "",
    ]
    for metric in METRICS:
        mean, std = mean_std(frame[f"val_{metric}"])
        lines.append(f"- val_{metric}_mean={mean:.4f} val_{metric}_std={std:.4f}")
    lines.extend(["", "```", frame.to_string(index=False), "```"])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize per-fold metrics logs by best validation epoch.")
    parser.add_argument("--logs-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-kind", type=str, required=True)
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--monitor", choices=METRICS, default="auroc")
    parser.add_argument("--glob", type=str, default="*_metrics.csv")
    parser.add_argument("--subjects", nargs="*", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requested = {str(subject).strip() for subject in args.subjects or [] if str(subject).strip()}
    paths = sorted(args.logs_dir.glob(args.glob))
    if requested:
        paths = [path for path in paths if subject_from_metrics_path(path) in requested]
    if not paths:
        raise ValueError(f"No metrics files found under {args.logs_dir} with glob {args.glob}")

    rows = [best_validation_row(path, args.monitor, args.dataset_kind, args.method) for path in paths]
    frame = pd.DataFrame(rows).sort_values("subject").reset_index(drop=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    safe_method = args.method.replace("/", "_").replace(" ", "_")
    csv_path = args.output_dir / f"{args.dataset_kind}_{safe_method}_best_val_summary.csv"
    md_path = args.output_dir / f"{args.dataset_kind}_{safe_method}_best_val_summary.md"
    frame.to_csv(csv_path, index=False)
    write_summary(frame, md_path, args.dataset_kind, args.method, args.monitor)
    print(f"saved_validation_csv={csv_path}")
    print(f"saved_validation_summary={md_path}")


if __name__ == "__main__":
    main()
