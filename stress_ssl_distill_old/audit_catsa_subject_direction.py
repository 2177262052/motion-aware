from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


DEFAULT_CALM_SESSIONS = ("Baseline",)
DEFAULT_STRESS_SESSIONS = ("Stroop", "Logic", "Sudoku")
DEFAULT_FEATURES = (
    "proc_bvp_mean",
    "raw_acc_mag_range",
    "proc_temp_std",
    "proc_temp_diff_rms",
    "raw_eda_mean",
    "raw_eda_median",
    "proc_bvp_range",
    "raw_acc_mag_diff_rms",
)

METADATA_COLUMNS = {
    "subject_index",
    "label",
    "window_start_s",
    "window_end_s",
    "window_start_ms",
    "window_end_ms",
}


def numeric_feature_columns(frame: pd.DataFrame) -> list[str]:
    columns: list[str] = []
    for column in frame.columns:
        if column in METADATA_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            columns.append(column)
    return columns


def safe_auc(labels: pd.Series, values: pd.Series) -> float:
    y = pd.to_numeric(labels, errors="coerce")
    x = pd.to_numeric(values, errors="coerce")
    valid = y.notna() & x.notna()
    y = y[valid].astype(int)
    x = x[valid].astype(float)
    if len(y) < 3 or y.nunique() < 2 or x.nunique() < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y, x))
    except ValueError:
        return float("nan")


def signed_direction(value: float, eps: float) -> int:
    if not np.isfinite(value) or abs(value) <= eps:
        return 0
    return 1 if value > 0 else -1


def build_subject_feature_audit(
    windows: pd.DataFrame,
    features: Sequence[str],
    calm_sessions: Sequence[str],
    stress_sessions: Sequence[str],
    min_windows_per_class: int,
    direction_eps: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    calm_set = {str(item) for item in calm_sessions}

    for task in stress_sessions:
        task_set = {str(task)}
        subset = windows[windows["session"].astype(str).isin(calm_set | task_set)].copy()
        subset["task_label"] = subset["session"].astype(str).isin(task_set).astype(int)
        for subject_id, subject_frame in subset.groupby("subject_id", sort=True):
            baseline = subject_frame[subject_frame["task_label"] == 0]
            stress = subject_frame[subject_frame["task_label"] == 1]
            if len(baseline) < min_windows_per_class or len(stress) < min_windows_per_class:
                continue
            for feature in features:
                if feature not in subject_frame.columns:
                    continue
                baseline_values = pd.to_numeric(baseline[feature], errors="coerce").dropna().astype(float)
                stress_values = pd.to_numeric(stress[feature], errors="coerce").dropna().astype(float)
                if len(baseline_values) < min_windows_per_class or len(stress_values) < min_windows_per_class:
                    continue
                baseline_mean = float(baseline_values.mean())
                stress_mean = float(stress_values.mean())
                delta = stress_mean - baseline_mean
                pooled_std = float(np.sqrt(0.5 * (baseline_values.var(ddof=0) + stress_values.var(ddof=0))))
                smd = float(delta / pooled_std) if pooled_std > 1e-12 else float("nan")
                auc = safe_auc(subject_frame["task_label"], subject_frame[feature])
                rows.append(
                    {
                        "task": str(task),
                        "subject_id": str(subject_id),
                        "feature": feature,
                        "baseline_n": int(len(baseline_values)),
                        "stress_n": int(len(stress_values)),
                        "baseline_mean": baseline_mean,
                        "stress_mean": stress_mean,
                        "stress_minus_baseline": delta,
                        "direction_sign": signed_direction(delta, direction_eps),
                        "smd": smd,
                        "auc": auc,
                        "abs_auc": max(auc, 1.0 - auc) if np.isfinite(auc) else float("nan"),
                        "auc_reversed": int(np.isfinite(auc) and auc < 0.5),
                    }
                )
    return pd.DataFrame(rows)


def summarize_reversals(subject_rows: pd.DataFrame) -> pd.DataFrame:
    if subject_rows.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for (task, feature), group in subject_rows.groupby(["task", "feature"], sort=True):
        valid_auc = group[np.isfinite(group["auc"])]
        valid_sign = group[group["direction_sign"] != 0]
        positive = int((valid_sign["direction_sign"] > 0).sum())
        negative = int((valid_sign["direction_sign"] < 0).sum())
        majority_sign = 0
        if positive > negative:
            majority_sign = 1
        elif negative > positive:
            majority_sign = -1
        minority = min(positive, negative)
        rows.append(
            {
                "task": task,
                "feature": feature,
                "subjects": int(group["subject_id"].nunique()),
                "positive_direction_subjects": positive,
                "negative_direction_subjects": negative,
                "tie_direction_subjects": int((group["direction_sign"] == 0).sum()),
                "minority_direction_frac": float(minority / max(positive + negative, 1)),
                "majority_direction": (
                    "higher_in_stress" if majority_sign > 0 else "lower_in_stress" if majority_sign < 0 else "tied"
                ),
                "mean_delta": float(group["stress_minus_baseline"].mean()),
                "median_delta": float(group["stress_minus_baseline"].median()),
                "mean_abs_smd": float(group["smd"].abs().mean()),
                "mean_auc": float(valid_auc["auc"].mean()) if len(valid_auc) else float("nan"),
                "median_auc": float(valid_auc["auc"].median()) if len(valid_auc) else float("nan"),
                "auc_below_0p5_frac": float(valid_auc["auc_reversed"].mean()) if len(valid_auc) else float("nan"),
                "auc_below_0p4_subjects": int((valid_auc["auc"] < 0.4).sum()) if len(valid_auc) else 0,
                "auc_above_0p6_subjects": int((valid_auc["auc"] > 0.6).sum()) if len(valid_auc) else 0,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["minority_direction_frac", "auc_below_0p5_frac", "mean_abs_smd"],
        ascending=[False, False, False],
    )


def write_report(
    output_path: Path,
    summary: pd.DataFrame,
    subject_rows: pd.DataFrame,
    top_k: int,
) -> None:
    lines: list[str] = []
    lines.append("# CATSA Subject-Level Direction Audit")
    lines.append("")
    lines.append(
        "This audit compares each subject's Baseline windows against each stress task separately. "
        "High minority-direction or AUC-below-0.5 rates indicate that the pooled dataset-level cue is not stable under LOSO evaluation."
    )
    lines.append("")

    if summary.empty:
        lines.append("No valid subject-level comparisons were available.")
    else:
        lines.append("## Most Direction-Unstable Features")
        lines.append("")
        lines.append(
            "| Task | Feature | Subjects | +Dir | -Dir | Minority % | Mean AUC | AUC<0.5 % | AUC<0.4 subjects |"
        )
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
        for row in summary.head(top_k).itertuples(index=False):
            lines.append(
                f"| {row.task} | {row.feature} | {int(row.subjects)} | "
                f"{int(row.positive_direction_subjects)} | {int(row.negative_direction_subjects)} | "
                f"{100 * float(row.minority_direction_frac):.1f} | {float(row.mean_auc):.3f} | "
                f"{100 * float(row.auc_below_0p5_frac):.1f} | {int(row.auc_below_0p4_subjects)} |"
            )
        lines.append("")

        lines.append("## Worst Subject-Feature Reversals")
        lines.append("")
        worst = subject_rows[np.isfinite(subject_rows["auc"])].sort_values("auc").head(top_k)
        lines.append("| Task | Subject | Feature | AUC | Delta | SMD | Baseline Mean | Stress Mean |")
        lines.append("|---|---|---|---:|---:|---:|---:|---:|")
        for row in worst.itertuples(index=False):
            lines.append(
                f"| {row.task} | {row.subject_id} | {row.feature} | {float(row.auc):.3f} | "
                f"{float(row.stress_minus_baseline):.4f} | {float(row.smd):.3f} | "
                f"{float(row.baseline_mean):.4f} | {float(row.stress_mean):.4f} |"
            )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit CATSA subject-level feature direction stability.")
    parser.add_argument("--quality-windows-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--calm-sessions", nargs="*", default=list(DEFAULT_CALM_SESSIONS))
    parser.add_argument("--stress-sessions", nargs="*", default=list(DEFAULT_STRESS_SESSIONS))
    parser.add_argument("--features", nargs="*", default=list(DEFAULT_FEATURES))
    parser.add_argument("--all-features", action="store_true")
    parser.add_argument("--min-windows-per-class", type=int, default=3)
    parser.add_argument("--direction-eps", type=float, default=1e-8)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    windows = pd.read_csv(args.quality_windows_csv)
    if "subject_id" not in windows.columns or "session" not in windows.columns:
        raise ValueError("quality windows CSV must contain subject_id and session columns.")

    features = numeric_feature_columns(windows) if args.all_features else list(args.features)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    subject_rows = build_subject_feature_audit(
        windows,
        features=features,
        calm_sessions=args.calm_sessions,
        stress_sessions=args.stress_sessions,
        min_windows_per_class=args.min_windows_per_class,
        direction_eps=args.direction_eps,
    )
    summary = summarize_reversals(subject_rows)

    subject_path = args.output_dir / "catsa_subject_feature_directions.csv"
    summary_path = args.output_dir / "catsa_subject_direction_summary.csv"
    report_path = args.output_dir / "catsa_subject_direction_audit.md"
    subject_rows.to_csv(subject_path, index=False)
    summary.to_csv(summary_path, index=False)
    write_report(report_path, summary=summary, subject_rows=subject_rows, top_k=args.top_k)

    print(f"comparisons={len(subject_rows)} features={len(features)}")
    if not summary.empty:
        cols = [
            "task",
            "feature",
            "subjects",
            "positive_direction_subjects",
            "negative_direction_subjects",
            "minority_direction_frac",
            "mean_auc",
            "auc_below_0p5_frac",
            "auc_below_0p4_subjects",
        ]
        print(summary.head(args.top_k)[cols].to_string(index=False))
    print(f"Saved subject-level directions to {subject_path}")
    print(f"Saved direction summary to {summary_path}")
    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()
