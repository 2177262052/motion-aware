from __future__ import annotations

import argparse
import re
import statistics
from pathlib import Path
from typing import Iterable

import pandas as pd


WATCH_ONLY_PATTERN = re.compile(
    r"best_threshold=(?P<threshold>[-+]?\d*\.?\d+)\s+"
    r"best_test_acc=(?P<acc>[-+]?\d*\.?\d+)\s+"
    r"best_test_balanced_acc=(?P<balanced_acc>[-+]?\d*\.?\d+)\s+"
    r"best_test_f1=(?P<f1>[-+]?\d*\.?\d+)\s+"
    r"best_test_auroc=(?P<auroc>[-+]?\d*\.?\d+)\s+"
    r"best_test_positive_rate=(?P<positive_rate>[-+]?\d*\.?\d+)"
)

PRIV_WATCH_PATTERN = re.compile(
    r"best_watch_threshold=(?P<threshold>[-+]?\d*\.?\d+)\s+"
    r"best_watch_test_acc=(?P<acc>[-+]?\d*\.?\d+)\s+"
    r"best_watch_test_balanced_acc=(?P<balanced_acc>[-+]?\d*\.?\d+)\s+"
    r"best_watch_test_f1=(?P<f1>[-+]?\d*\.?\d+)\s+"
    r"best_watch_test_auroc=(?P<auroc>[-+]?\d*\.?\d+)\s+"
    r"best_watch_test_positive_rate=(?P<positive_rate>[-+]?\d*\.?\d+)"
)

TEACHER_PATTERN = re.compile(
    r"best_teacher_threshold=(?P<threshold>[-+]?\d*\.?\d+)\s+"
    r"best_teacher_test_acc=(?P<acc>[-+]?\d*\.?\d+)\s+"
    r"best_teacher_test_balanced_acc=(?P<balanced_acc>[-+]?\d*\.?\d+)\s+"
    r"best_teacher_test_f1=(?P<f1>[-+]?\d*\.?\d+)\s+"
    r"best_teacher_test_auroc=(?P<auroc>[-+]?\d*\.?\d+)\s+"
    r"best_teacher_test_positive_rate=(?P<positive_rate>[-+]?\d*\.?\d+)"
)


def discover_manifests(manifests_dir: Path, subjects: Iterable[str] | None) -> list[tuple[str, Path]]:
    requested = {subject.strip() for subject in subjects or [] if subject.strip()}
    manifests: dict[str, Path] = {}
    for pattern in ("wesad_*_loso_val.csv", "*_loso_val.csv"):
        for path in sorted(manifests_dir.glob(pattern)):
            subject = path.stem
            if subject.startswith("wesad_"):
                subject = subject[len("wesad_") :]
            if subject.endswith("_loso_val"):
                subject = subject[: -len("_loso_val")]
            if requested and subject not in requested:
                continue
            manifests.setdefault(subject, path)
    return sorted(manifests.items())


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
    parser = argparse.ArgumentParser(
        description=(
            "Shared WESAD LOSO parsing utilities. Use "
            "`python -m stress_ssl_distill.run_wesad_loso_eval_elastic_scaled_motion` "
            "for the final paper wrapper."
        )
    )
    parser.parse_args()
    raise SystemExit(
        "This module is kept for shared LOSO parsing helpers. "
        "Run `python -m stress_ssl_distill.run_wesad_loso_eval_elastic_scaled_motion` instead."
    )


if __name__ == "__main__":
    main()
