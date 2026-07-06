from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = ["acc", "balanced_acc", "f1", "auroc"]


def subject_from_metrics_path(path: Path) -> str:
    name = path.stem
    suffix = "_teacher_only_metrics"
    if name.endswith(suffix):
        return name[: -len(suffix)]
    return name


def numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series([np.nan] * len(frame), index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def best_validation_row(path: Path, monitor: str) -> dict[str, object]:
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Empty metrics file: {path}")

    monitor_col = f"val_{monitor}"
    if monitor_col not in frame.columns:
        raise ValueError(f"{path} does not contain required column {monitor_col}")

    scores = numeric_column(frame, monitor_col)
    if scores.notna().any():
        best_index = int(scores.idxmax())
    else:
        best_index = int(frame.index[-1])
    row = frame.loc[best_index]

    output: dict[str, object] = {
        "subject": subject_from_metrics_path(path),
        "source_file": str(path),
        "selection_monitor": monitor,
        "best_epoch": int(row["epoch"]) if "epoch" in frame.columns and pd.notna(row["epoch"]) else best_index + 1,
        "best_val_score": float(row[monitor_col]) if pd.notna(row[monitor_col]) else float("nan"),
        "val_threshold": float(row["val_threshold"]) if "val_threshold" in frame.columns and pd.notna(row["val_threshold"]) else float("nan"),
        "teacher_signal": str(row["teacher_signal"]) if "teacher_signal" in frame.columns else "",
        "auxiliary_signal": str(row["auxiliary_signal"]) if "auxiliary_signal" in frame.columns else "",
    }
    for metric in METRICS:
        col = f"val_{metric}"
        output[col] = float(row[col]) if col in frame.columns and pd.notna(row[col]) else float("nan")

    return output


def mean_std(values: pd.Series) -> tuple[float, float]:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if len(values) == 0:
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values.iloc[0]), 0.0
    return float(values.mean()), float(values.std(ddof=1))


def write_summary_markdown(frame: pd.DataFrame, output_path: Path, dataset_kind: str, monitor: str) -> None:
    lines: list[str] = [
        f"# {dataset_kind.upper()} Teacher-Only Validation Summary",
        "",
        "This file is for KD-regime diagnosis. It summarizes teacher-only performance on subject-disjoint validation splits.",
        "Do not use held-out test metrics to choose KD or gated KD.",
        "",
        f"- Selection monitor: val_{monitor}",
        f"- Folds: {len(frame)}",
        "",
    ]

    for metric in METRICS:
        mean, std = mean_std(frame[f"val_{metric}"])
        lines.append(f"- val_{metric}_mean={mean:.4f} val_{metric}_std={std:.4f}")

    if "val_auroc" in frame.columns:
        auroc = pd.to_numeric(frame["val_auroc"], errors="coerce")
        lines.extend(
            [
                f"- val_auroc_gt_0.50={int((auroc > 0.50).sum())}/{int(auroc.notna().sum())}",
                f"- val_auroc_gt_0.60={int((auroc > 0.60).sum())}/{int(auroc.notna().sum())}",
                f"- val_auroc_gt_0.70={int((auroc > 0.70).sum())}/{int(auroc.notna().sum())}",
            ]
        )

    lines.extend(["", "```", frame.to_string(index=False), "```"])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract best-validation teacher-only diagnostics from per-fold teacher-only metric logs."
    )
    parser.add_argument("--logs-dir", type=Path, required=True, help="Directory containing *_teacher_only_metrics.csv files.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-kind", type=str, default="dataset")
    parser.add_argument("--monitor", choices=METRICS, default="auroc", help="Select best epoch by val_<monitor>.")
    parser.add_argument("--glob", type=str, default="*_teacher_only_metrics.csv")
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

    rows = [best_validation_row(path, args.monitor) for path in paths]
    frame = pd.DataFrame(rows).sort_values("subject").reset_index(drop=True)
    frame.insert(0, "dataset", args.dataset_kind)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / f"{args.dataset_kind}_teacher_only_best_val_summary.csv"
    md_path = args.output_dir / f"{args.dataset_kind}_teacher_only_best_val_summary.md"
    frame.to_csv(csv_path, index=False)
    write_summary_markdown(frame, md_path, args.dataset_kind, args.monitor)

    print(f"saved_validation_csv={csv_path}")
    print(f"saved_validation_summary={md_path}")


if __name__ == "__main__":
    main()
