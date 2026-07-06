from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score

from .analyze_paired_bootstrap import MethodSpec, _canonicalize_prediction_frame, _parse_method, _read_registry


def _parse_offsets(values: list[str] | None) -> list[float]:
    if not values:
        return [-0.10, -0.05, 0.0, 0.05, 0.10]
    offsets: list[float] = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if part:
                offsets.append(float(part))
    return offsets


def _subject_threshold_rows(windows: pd.DataFrame, offsets: list[float]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for key, group in windows.groupby(["dataset", "group", "method", "subject_id"], sort=True):
        dataset, group_name, method, subject_id = key
        thresholds = group["threshold"].unique()
        if len(thresholds) != 1:
            raise ValueError(f"Conflicting thresholds for {key}: {thresholds}")
        base_threshold = float(thresholds[0])
        labels = group["label"].to_numpy(dtype=int)
        probs = group["prob"].to_numpy(dtype=float)
        for offset in offsets:
            applied = float(np.clip(base_threshold + float(offset), 0.0, 1.0))
            preds = (probs >= applied).astype(int)
            rows.append(
                {
                    "dataset": dataset,
                    "group": group_name,
                    "method": method,
                    "subject_id": subject_id,
                    "n_windows": int(len(group)),
                    "base_threshold": base_threshold,
                    "offset": float(offset),
                    "applied_threshold": applied,
                    "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
                    "f1": float(f1_score(labels, preds, zero_division=0)),
                    "positive_rate": float(np.mean(preds)) if len(preds) else float("nan"),
                    "single_class": int(len(np.unique(preds)) <= 1),
                }
            )
    return pd.DataFrame(rows)


def _summary(subject_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for key, group in subject_rows.groupby(["dataset", "group", "method", "offset"], sort=True):
        dataset, group_name, method, offset = key
        total = int(len(group))
        single_count = int(group["single_class"].sum())
        rows.append(
            {
                "dataset": dataset,
                "group": group_name,
                "method": method,
                "offset": float(offset),
                "n_subjects": total,
                "balanced_accuracy_mean": float(group["balanced_accuracy"].mean()),
                "balanced_accuracy_std": float(group["balanced_accuracy"].std(ddof=1)) if total > 1 else 0.0,
                "f1_mean": float(group["f1"].mean()),
                "f1_std": float(group["f1"].std(ddof=1)) if total > 1 else 0.0,
                "positive_rate_mean": float(group["positive_rate"].mean()),
                "positive_rate_std": float(group["positive_rate"].std(ddof=1)) if total > 1 else 0.0,
                "single_class_count": single_count,
                "single_class_total": total,
                "single_class_percent": float(single_count / total * 100.0) if total else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Threshold perturbation robustness from window-level predictions.")
    parser.add_argument("--registry", type=Path, default=None, help="CSV with dataset,group,method,path columns.")
    parser.add_argument(
        "--method",
        action="append",
        default=[],
        help="DATASET:GROUP:METHOD=window_predictions.csv. Can be repeated.",
    )
    parser.add_argument("--offset", action="append", default=None, help="Offset value or comma-separated values.")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    specs: list[MethodSpec] = []
    if args.registry is not None:
        specs.extend(_read_registry(args.registry))
    specs.extend(_parse_method(item) for item in args.method)
    if not specs:
        raise ValueError("Provide --registry or at least one --method.")

    windows = pd.concat([_canonicalize_prediction_frame(spec) for spec in specs], ignore_index=True)
    offsets = _parse_offsets(args.offset)
    subject_rows = _subject_threshold_rows(windows, offsets=offsets)
    summary_rows = _summary(subject_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    subject_rows.to_csv(args.output_dir / "threshold_robustness_subject_level.csv", index=False)
    summary_rows.to_csv(args.output_dir / "threshold_robustness_summary.csv", index=False)
    print(f"saved_subject_level={args.output_dir / 'threshold_robustness_subject_level.csv'}")
    print(f"saved_summary={args.output_dir / 'threshold_robustness_summary.csv'}")


if __name__ == "__main__":
    main()
