from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .catsa_dataset import (
    CATSA_SAMPLE_RATES,
    _is_time_like_column,
    _numeric_column_names,
)


DEFAULT_SESSIONS = ("Baseline", "Stroop", "Logic", "Sudoku")
DEFAULT_MODALITIES = ("BVP", "ACC", "EDA", "TEMP", "HR")

CATSA_PLAUSIBLE_RANGES = {
    "BVP": (-5000.0, 5000.0),
    "ACC": (-512.0, 512.0),
    "EDA": (0.0, 100.0),
    "TEMP": (15.0, 45.0),
    "HR": (25.0, 240.0),
}


def subject_sort_key(path: Path) -> tuple[int, str]:
    suffix = path.name.replace("Sub", "")
    try:
        return int(suffix), path.name
    except ValueError:
        return 10_000, path.name


def select_signal_columns(frame: pd.DataFrame, modality: str) -> list[object]:
    normalized = {str(name).strip().lower(): name for name in frame.columns}
    modality_key = modality.strip().lower()
    if modality_key == "acc":
        columns = [normalized[name] for name in ("acc_x", "acc_y", "acc_z") if name in normalized]
        if len(columns) != 3:
            columns = [normalized[name] for name in ("x", "y", "z") if name in normalized]
        if len(columns) == 3:
            return columns
        numeric_columns = _numeric_column_names(frame)
        signal_columns = [name for name in numeric_columns if not _is_time_like_column(name)]
        if len(signal_columns) >= 3:
            return signal_columns[:3]
        return numeric_columns[-3:]

    if modality_key in normalized:
        return [normalized[modality_key]]

    named_candidates = [
        name
        for name in frame.columns
        if modality_key in str(name).strip().lower() and not _is_time_like_column(name)
    ]
    if named_candidates:
        return [named_candidates[0]]

    numeric_columns = _numeric_column_names(frame)
    signal_columns = [name for name in numeric_columns if not _is_time_like_column(name)]
    if signal_columns:
        return [signal_columns[-1]]
    return [frame.columns[-1]]


def longest_true_run(mask: np.ndarray) -> tuple[int, int | None]:
    best_len = 0
    best_start: int | None = None
    current_len = 0
    current_start = 0
    for idx, flag in enumerate(mask.astype(bool)):
        if flag:
            if current_len == 0:
                current_start = idx
            current_len += 1
            if current_len > best_len:
                best_len = current_len
                best_start = current_start
        else:
            current_len = 0
    return best_len, best_start


def safe_quantile(values: np.ndarray, q: float) -> float:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return float("nan")
    return float(np.quantile(finite, q))


def outlier_severity(values: np.ndarray, low: float | None, high: float | None) -> np.ndarray:
    severity = np.zeros_like(values, dtype=np.float64)
    finite = np.isfinite(values)
    if low is not None:
        severity = np.maximum(severity, np.where(finite & (values < low), low - values, 0.0))
    if high is not None:
        severity = np.maximum(severity, np.where(finite & (values > high), values - high, 0.0))
    return severity


def summarize_channel(
    values: np.ndarray,
    subject_id: str,
    session: str,
    modality: str,
    channel: str,
    rel_path: str,
    max_examples: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    low, high = CATSA_PLAUSIBLE_RANGES.get(modality.upper(), (None, None))
    finite = np.isfinite(values)
    nan_mask = ~finite
    low_mask = finite & (values < low) if low is not None else np.zeros_like(finite, dtype=bool)
    high_mask = finite & (values > high) if high is not None else np.zeros_like(finite, dtype=bool)
    bad_mask = nan_mask | low_mask | high_mask

    longest_run, longest_run_start = longest_true_run(bad_mask)
    bad_indices = np.flatnonzero(bad_mask)
    first_bad_index = int(bad_indices[0]) if len(bad_indices) else None
    sample_rate = CATSA_SAMPLE_RATES.get(modality.upper(), float("nan"))
    first_bad_time_s = (
        float(first_bad_index / sample_rate)
        if first_bad_index is not None and np.isfinite(sample_rate) and sample_rate > 0
        else float("nan")
    )
    first_bad_value = float(values[first_bad_index]) if first_bad_index is not None else float("nan")
    longest_run_start_s = (
        float(longest_run_start / sample_rate)
        if longest_run_start is not None and np.isfinite(sample_rate) and sample_rate > 0
        else float("nan")
    )

    finite_values = values[finite]
    summary = {
        "subject_id": subject_id,
        "session": session,
        "modality": modality,
        "channel": channel,
        "relative_path": rel_path,
        "low_bound": low,
        "high_bound": high,
        "sample_rate_hz": sample_rate,
        "n": int(len(values)),
        "finite_count": int(finite.sum()),
        "nan_count": int(nan_mask.sum()),
        "low_count": int(low_mask.sum()),
        "high_count": int(high_mask.sum()),
        "bad_count": int(bad_mask.sum()),
        "bad_frac": float(bad_mask.mean()) if len(values) else float("nan"),
        "longest_bad_run": int(longest_run),
        "longest_bad_run_start_index": longest_run_start,
        "longest_bad_run_start_s": longest_run_start_s,
        "first_bad_index": first_bad_index,
        "first_bad_time_s": first_bad_time_s,
        "first_bad_value": first_bad_value,
        "min": float(np.min(finite_values)) if len(finite_values) else float("nan"),
        "p01": safe_quantile(values, 0.01),
        "median": safe_quantile(values, 0.50),
        "p99": safe_quantile(values, 0.99),
        "max": float(np.max(finite_values)) if len(finite_values) else float("nan"),
        "mean": float(np.mean(finite_values)) if len(finite_values) else float("nan"),
        "std": float(np.std(finite_values)) if len(finite_values) else float("nan"),
    }

    examples: list[dict[str, object]] = []
    finite_bad = np.flatnonzero(low_mask | high_mask)
    severity = outlier_severity(values, low, high)
    if len(finite_bad):
        order = finite_bad[np.argsort(severity[finite_bad])[::-1]][:max_examples]
        for idx in order:
            examples.append(
                {
                    "subject_id": subject_id,
                    "session": session,
                    "modality": modality,
                    "channel": channel,
                    "relative_path": rel_path,
                    "index": int(idx),
                    "time_s": float(idx / sample_rate) if np.isfinite(sample_rate) and sample_rate > 0 else float("nan"),
                    "value": float(values[idx]),
                    "low_bound": low,
                    "high_bound": high,
                    "severity": float(severity[idx]),
                    "reason": "below_range" if low is not None and values[idx] < low else "above_range",
                }
            )
    nan_bad = np.flatnonzero(nan_mask)[:max_examples]
    for idx in nan_bad:
        examples.append(
            {
                "subject_id": subject_id,
                "session": session,
                "modality": modality,
                "channel": channel,
                "relative_path": rel_path,
                "index": int(idx),
                "time_s": float(idx / sample_rate) if np.isfinite(sample_rate) and sample_rate > 0 else float("nan"),
                "value": float("nan"),
                "low_bound": low,
                "high_bound": high,
                "severity": float("nan"),
                "reason": "non_finite",
            }
        )
    return summary, examples


def audit_file(
    path: Path,
    catsa_root: Path,
    subject_id: str,
    session: str,
    modality: str,
    max_examples: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    frame = pd.read_csv(path)
    columns = select_signal_columns(frame, modality)
    rel_path = str(path.relative_to(catsa_root))
    summaries: list[dict[str, object]] = []
    examples: list[dict[str, object]] = []
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
        summary, channel_examples = summarize_channel(
            values,
            subject_id=subject_id,
            session=session,
            modality=modality,
            channel=str(column),
            rel_path=rel_path,
            max_examples=max_examples,
        )
        summaries.append(summary)
        examples.extend(channel_examples)
    return summaries, examples


def build_file_summary(channel_df: pd.DataFrame) -> pd.DataFrame:
    if channel_df.empty:
        return pd.DataFrame()
    grouped = channel_df.groupby(["subject_id", "session", "modality", "relative_path"], as_index=False)
    out = grouped.agg(
        channels=("channel", "nunique"),
        n=("n", "sum"),
        finite_count=("finite_count", "sum"),
        nan_count=("nan_count", "sum"),
        low_count=("low_count", "sum"),
        high_count=("high_count", "sum"),
        bad_count=("bad_count", "sum"),
        longest_bad_run=("longest_bad_run", "max"),
        min=("min", "min"),
        max=("max", "max"),
    )
    out["bad_frac"] = out["bad_count"] / out["n"].replace(0, np.nan)
    return out.sort_values(["bad_count", "bad_frac"], ascending=[False, False]).reset_index(drop=True)


def build_session_summary(channel_df: pd.DataFrame) -> pd.DataFrame:
    if channel_df.empty:
        return pd.DataFrame()
    grouped = channel_df.groupby(["session", "modality"], as_index=False)
    out = grouped.agg(
        channels=("channel", "count"),
        subjects=("subject_id", "nunique"),
        n=("n", "sum"),
        bad_count=("bad_count", "sum"),
        files_with_bad=("bad_count", lambda s: int((s > 0).sum())),
        max_bad_frac=("bad_frac", "max"),
        median_bad_frac=("bad_frac", "median"),
    )
    out["bad_frac"] = out["bad_count"] / out["n"].replace(0, np.nan)
    return out.sort_values(["bad_frac", "bad_count"], ascending=[False, False]).reset_index(drop=True)


def format_float(value: object, digits: int = 4) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "nan"
    if not np.isfinite(x):
        return "nan"
    return f"{x:.{digits}f}"


def write_markdown_report(
    output_path: Path,
    channel_df: pd.DataFrame,
    file_df: pd.DataFrame,
    session_df: pd.DataFrame,
    sessions: Sequence[str],
    modalities: Sequence[str],
) -> None:
    lines: list[str] = []
    total_values = int(channel_df["n"].sum()) if not channel_df.empty else 0
    total_bad = int(channel_df["bad_count"].sum()) if not channel_df.empty else 0
    total_frac = total_bad / total_values if total_values else float("nan")

    lines.append("# CATSA Signal Outlier Audit")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append(f"- Sessions: {', '.join(sessions)}")
    lines.append(f"- Modalities: {', '.join(modalities)}")
    lines.append(f"- Total raw values inspected: {total_values}")
    lines.append(f"- Implausible or non-finite values: {total_bad} ({format_float(total_frac * 100, 3)}%)")
    lines.append("")
    lines.append("## Plausible Ranges")
    lines.append("")
    lines.append("| Modality | Lower bound | Upper bound |")
    lines.append("|---|---:|---:|")
    for modality in modalities:
        low, high = CATSA_PLAUSIBLE_RANGES.get(modality.upper(), (None, None))
        lines.append(f"| {modality} | {low} | {high} |")
    lines.append("")
    lines.append("## Cleaning Rule Used Before Preprocessing")
    lines.append("")
    lines.append(
        "For each raw stream, values outside the modality-specific plausible range and non-finite values are treated as missing. "
        "Missing samples are replaced by linear interpolation in both directions; any remaining missing values are filled with the channel median. "
        "If an entire channel is invalid, it is replaced by zeros. CATSA accelerometer counts are divided by 64 before normalization. "
        "After range cleaning, each channel is normalized with a robust z-score; when MAD is too small, the scaler falls back to the channel standard deviation, then to a unit scale for constant channels."
    )
    lines.append("")
    lines.append("## Session-Level Summary")
    lines.append("")
    if session_df.empty:
        lines.append("No files were audited.")
    else:
        lines.append("| Session | Modality | Values | Bad values | Bad % | Files/channels with bad values | Max bad % |")
        lines.append("|---|---|---:|---:|---:|---:|---:|")
        for row in session_df.itertuples(index=False):
            lines.append(
                f"| {row.session} | {row.modality} | {int(row.n)} | {int(row.bad_count)} | "
                f"{format_float(row.bad_frac * 100, 3)} | {int(row.files_with_bad)} | {format_float(row.max_bad_frac * 100, 3)} |"
            )
    lines.append("")
    lines.append("## Worst Files")
    lines.append("")
    bad_file_df = file_df[file_df["bad_count"] > 0].copy() if not file_df.empty else file_df
    if bad_file_df.empty:
        lines.append("No files were audited.")
    else:
        lines.append("| Subject | Session | Modality | Bad values | Bad % | Longest bad run | Min | Max | File |")
        lines.append("|---|---|---|---:|---:|---:|---:|---:|---|")
        for row in bad_file_df.head(20).itertuples(index=False):
            lines.append(
                f"| {row.subject_id} | {row.session} | {row.modality} | {int(row.bad_count)} | "
                f"{format_float(row.bad_frac * 100, 3)} | {int(row.longest_bad_run)} | "
                f"{format_float(row.min, 3)} | {format_float(row.max, 3)} | `{row.relative_path}` |"
            )
    lines.append("")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit CATSA raw CSVs for implausible sensor values and sentinel artifacts.")
    parser.add_argument("--catsa-root", "--dataset-root", dest="catsa_root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sessions", nargs="*", default=list(DEFAULT_SESSIONS))
    parser.add_argument("--modalities", nargs="*", default=list(DEFAULT_MODALITIES))
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--max-examples-per-channel", type=int, default=5)
    args = parser.parse_args()

    catsa_root = args.catsa_root
    requested_subjects = {str(subject) for subject in args.subjects or []}
    subjects = sorted([path for path in catsa_root.glob("Sub*") if path.is_dir()], key=subject_sort_key)
    if requested_subjects:
        subjects = [path for path in subjects if path.name in requested_subjects]
    if not subjects:
        raise FileNotFoundError(f"No CATSA subject directories found under: {catsa_root}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    channel_rows: list[dict[str, object]] = []
    example_rows: list[dict[str, object]] = []

    tasks: list[tuple[Path, str, str, str]] = []
    for subject_dir in subjects:
        for session in args.sessions:
            session_dir = subject_dir / session
            if not session_dir.exists():
                continue
            for modality in args.modalities:
                path = session_dir / f"{modality}.csv"
                if path.exists():
                    tasks.append((path, subject_dir.name, session, modality))

    for path, subject_id, session, modality in tqdm(tasks, desc="catsa outlier audit", leave=True):
        summaries, examples = audit_file(
            path,
            catsa_root=catsa_root,
            subject_id=subject_id,
            session=session,
            modality=modality,
            max_examples=args.max_examples_per_channel,
        )
        channel_rows.extend(summaries)
        example_rows.extend(examples)

    channel_df = pd.DataFrame(channel_rows)
    file_df = build_file_summary(channel_df)
    session_df = build_session_summary(channel_df)
    examples_df = pd.DataFrame(example_rows)
    if not examples_df.empty:
        examples_df = examples_df.sort_values(
            ["severity", "subject_id", "session", "modality"],
            ascending=[False, True, True, True],
            na_position="last",
        )

    channel_path = args.output_dir / "catsa_outlier_audit_per_channel.csv"
    file_path = args.output_dir / "catsa_outlier_audit_per_file.csv"
    session_path = args.output_dir / "catsa_outlier_audit_by_session.csv"
    examples_path = args.output_dir / "catsa_outlier_examples.csv"
    report_path = args.output_dir / "catsa_signal_outlier_audit.md"

    channel_df.to_csv(channel_path, index=False)
    file_df.to_csv(file_path, index=False)
    session_df.to_csv(session_path, index=False)
    examples_df.to_csv(examples_path, index=False)
    write_markdown_report(
        report_path,
        channel_df=channel_df,
        file_df=file_df,
        session_df=session_df,
        sessions=args.sessions,
        modalities=args.modalities,
    )

    total_values = int(channel_df["n"].sum()) if not channel_df.empty else 0
    total_bad = int(channel_df["bad_count"].sum()) if not channel_df.empty else 0
    bad_frac = total_bad / total_values if total_values else float("nan")
    print(f"subjects={len(subjects)} files={len(tasks)} values={total_values} bad={total_bad} bad_frac={bad_frac:.6f}")
    if not session_df.empty:
        print()
        print("Worst session/modality groups:")
        print(session_df.head(12).to_string(index=False))
    if not file_df.empty:
        print()
        print("Worst files:")
        cols = ["subject_id", "session", "modality", "bad_count", "bad_frac", "longest_bad_run", "min", "max", "relative_path"]
        print(file_df[file_df["bad_count"] > 0].head(12)[cols].to_string(index=False))
    print()
    print(f"Saved per-channel audit to {channel_path}")
    print(f"Saved per-file audit to {file_path}")
    print(f"Saved session summary to {session_path}")
    print(f"Saved outlier examples to {examples_path}")
    print(f"Saved appendix-ready report to {report_path}")


if __name__ == "__main__":
    main()
