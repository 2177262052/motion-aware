from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .galaxy_dataset import (
    POLAR_TIME_OFFSET_MS,
    compute_wavelet_band_ratios,
    preprocess_e4_bvp,
    preprocess_gw_ppg,
)
from .galaxy_protocols import pair_event_intervals


TARGET_SESSIONS = ("baseline", "tsst-prep")
WAVELET_BANDS = ("A4", "D4", "D2", "D1")


def _safe_float(value: object) -> float:
    if pd.isna(value) or value == "-":
        return float("nan")
    return float(value)


def _window_count(duration_ms: int, window_s: float, stride_s: float) -> int:
    window_ms = int(round(window_s * 1000))
    stride_ms = int(round(stride_s * 1000))
    if duration_ms < window_ms:
        return 0
    return 1 + max((duration_ms - window_ms) // stride_ms, 0)


def _read_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def _slice_by_timestamp(
    table: pd.DataFrame | None,
    time_col: str,
    start_ms: int,
    end_ms: int,
    *,
    time_divisor: float = 1.0,
    time_offset_ms: int = 0,
) -> pd.DataFrame:
    if table is None or table.empty:
        return pd.DataFrame()
    ts_ms = table[time_col].to_numpy(dtype=np.float64) / time_divisor - time_offset_ms
    mask = (ts_ms >= start_ms) & (ts_ms < end_ms)
    return table.loc[mask].reset_index(drop=True)


def _coverage_ratio(num_samples: int, duration_s: float, sampling_rate: float) -> float:
    expected = max(int(round(duration_s * sampling_rate)), 1)
    return float(min(num_samples / expected, 1.0))


def _acc_magnitude_stats(acc_df: pd.DataFrame) -> tuple[float, float]:
    if acc_df.empty:
        return float("nan"), float("nan")
    values = acc_df[["x", "y", "z"]].to_numpy(dtype=np.float32)
    mag = np.linalg.norm(values, axis=1)
    return float(np.mean(mag)), float(np.std(mag))


def _session_audit(
    session_name: str,
    intervals: list,
    gw_ppg_df: pd.DataFrame | None,
    gw_acc_df: pd.DataFrame | None,
    e4_bvp_df: pd.DataFrame | None,
    polar_ecg_df: pd.DataFrame | None,
) -> dict[str, float]:
    session_intervals = [item for item in intervals if item.session == session_name]
    duration_ms = int(sum(item.duration_ms for item in session_intervals))
    duration_s = duration_ms / 1000.0

    result: dict[str, float] = {
        "interval_count": float(len(session_intervals)),
        "duration_s": float(duration_s),
        "windows_10s": float(_window_count(duration_ms, 10.0, 10.0)),
        "windows_20s": float(_window_count(duration_ms, 20.0, 20.0)),
        "windows_30s": float(_window_count(duration_ms, 30.0, 30.0)),
        "gw_quality": float("nan"),
        "gw_ppg_coverage": float("nan"),
        "e4_bvp_coverage": float("nan"),
        "polar_ecg_coverage": float("nan"),
        "acc_mag_mean": float("nan"),
        "acc_mag_std": float("nan"),
    }
    for band in WAVELET_BANDS:
        result[f"wavelet_{band.lower()}"] = float("nan")

    if not session_intervals or duration_s <= 0:
        return result

    ppg_segments: list[np.ndarray] = []
    acc_segments: list[pd.DataFrame] = []
    e4_segments: list[np.ndarray] = []
    polar_segments: list[np.ndarray] = []
    gw_quality_parts: list[float] = []

    for interval in session_intervals:
        start_ms = interval.start_ms
        end_ms = interval.end_ms
        interval_s = interval.duration_ms / 1000.0

        ppg_window = _slice_by_timestamp(gw_ppg_df, "timestamp", start_ms, end_ms)
        acc_window = _slice_by_timestamp(gw_acc_df, "timestamp", start_ms, end_ms)
        e4_window = _slice_by_timestamp(e4_bvp_df, "timestamp", start_ms, end_ms, time_divisor=1000.0)
        polar_window = _slice_by_timestamp(
            polar_ecg_df,
            "phoneTimestamp",
            start_ms,
            end_ms,
            time_offset_ms=POLAR_TIME_OFFSET_MS,
        )

        if not ppg_window.empty:
            ppg_segments.append(ppg_window["ppg"].to_numpy(dtype=np.float32))
            if "status" in ppg_window.columns:
                status = ppg_window["status"].to_numpy(dtype=np.int32)
                gw_quality_parts.append(float(np.mean((status == 0) | (status == 500))))
            result["gw_ppg_coverage"] = (
                _coverage_ratio(len(ppg_window), interval_s, 25.0)
                if np.isnan(result["gw_ppg_coverage"])
                else result["gw_ppg_coverage"] + _coverage_ratio(len(ppg_window), interval_s, 25.0)
            )
        if not acc_window.empty:
            acc_segments.append(acc_window)
        if not e4_window.empty:
            e4_segments.append(e4_window["value"].to_numpy(dtype=np.float32))
            result["e4_bvp_coverage"] = (
                _coverage_ratio(len(e4_window), interval_s, 64.0)
                if np.isnan(result["e4_bvp_coverage"])
                else result["e4_bvp_coverage"] + _coverage_ratio(len(e4_window), interval_s, 64.0)
            )
        if not polar_window.empty:
            polar_segments.append(polar_window["ecg"].to_numpy(dtype=np.float32))
            result["polar_ecg_coverage"] = (
                _coverage_ratio(len(polar_window), interval_s, 130.0)
                if np.isnan(result["polar_ecg_coverage"])
                else result["polar_ecg_coverage"] + _coverage_ratio(len(polar_window), interval_s, 130.0)
            )

    num_intervals = max(len(session_intervals), 1)
    if not np.isnan(result["gw_ppg_coverage"]):
        result["gw_ppg_coverage"] /= num_intervals
    if not np.isnan(result["e4_bvp_coverage"]):
        result["e4_bvp_coverage"] /= num_intervals
    if not np.isnan(result["polar_ecg_coverage"]):
        result["polar_ecg_coverage"] /= num_intervals
    if gw_quality_parts:
        result["gw_quality"] = float(np.mean(gw_quality_parts))

    if acc_segments:
        acc_df = pd.concat(acc_segments, ignore_index=True)
        acc_mean, acc_std = _acc_magnitude_stats(acc_df)
        result["acc_mag_mean"] = acc_mean
        result["acc_mag_std"] = acc_std

    if ppg_segments:
        ppg_values = np.concatenate(ppg_segments)
        if len(ppg_values) >= 32:
            processed_ppg = preprocess_gw_ppg(ppg_values, sampling_rate=25.0)
            wavelet = compute_wavelet_band_ratios(
                processed_ppg,
                wavelet="db4",
                level=4,
                selected_bands=WAVELET_BANDS,
            )
            for band, value in zip(WAVELET_BANDS, wavelet):
                result[f"wavelet_{band.lower()}"] = float(value)

    return result


def build_subject_audit(dataset_root: Path) -> pd.DataFrame:
    meta = pd.read_csv(dataset_root / "Meta.csv").set_index("UID")
    subject_dirs = sorted(path for path in dataset_root.iterdir() if path.is_dir() and path.name.startswith("P"))
    rows: list[dict[str, float | str]] = []

    for subject_dir in subject_dirs:
        subject_id = subject_dir.name
        row: dict[str, float | str] = {
            "subject_id": subject_id,
            "age": _safe_float(meta.loc[subject_id, "AGE"]) if subject_id in meta.index else float("nan"),
            "gender": str(meta.loc[subject_id, "GENDER"]) if subject_id in meta.index else "",
            "tsst_score": _safe_float(meta.loc[subject_id, "TSST"]) if subject_id in meta.index else float("nan"),
            "ssst_score": _safe_float(meta.loc[subject_id, "SSST"]) if subject_id in meta.index else float("nan"),
            "watch_side": str(meta.loc[subject_id, "GalaxyWatch"]) if subject_id in meta.index else "",
        }

        event_df = _read_if_exists(subject_dir / "Event.csv")
        if event_df is None or event_df.empty:
            rows.append(row)
            continue

        intervals = pair_event_intervals(event_df.to_dict("records"))

        gw_ppg_df = _read_if_exists(subject_dir / "GalaxyWatch" / "PPG.csv")
        gw_acc_df = _read_if_exists(subject_dir / "GalaxyWatch" / "ACC.csv")
        e4_bvp_df = _read_if_exists(subject_dir / "E4" / "BVP.csv")
        polar_ecg_df = _read_if_exists(subject_dir / "PolarH10" / "ECG.csv")

        baseline_stats = _session_audit("baseline", intervals, gw_ppg_df, gw_acc_df, e4_bvp_df, polar_ecg_df)
        prep_stats = _session_audit("tsst-prep", intervals, gw_ppg_df, gw_acc_df, e4_bvp_df, polar_ecg_df)

        for key, value in baseline_stats.items():
            row[f"baseline_{key}"] = value
        for key, value in prep_stats.items():
            row[f"tsst_prep_{key}"] = value

        row["valid_10s_task"] = int(
            baseline_stats["windows_10s"] > 0 and prep_stats["windows_10s"] > 0
        )
        row["valid_20s_task"] = int(
            baseline_stats["windows_20s"] > 0 and prep_stats["windows_20s"] > 0
        )
        row["valid_30s_task"] = int(
            baseline_stats["windows_30s"] > 0 and prep_stats["windows_30s"] > 0
        )

        row["delta_duration_s"] = prep_stats["duration_s"] - baseline_stats["duration_s"]
        row["delta_gw_quality"] = prep_stats["gw_quality"] - baseline_stats["gw_quality"]
        row["delta_acc_mag_mean"] = prep_stats["acc_mag_mean"] - baseline_stats["acc_mag_mean"]
        row["delta_e4_bvp_coverage"] = prep_stats["e4_bvp_coverage"] - baseline_stats["e4_bvp_coverage"]
        row["delta_polar_ecg_coverage"] = prep_stats["polar_ecg_coverage"] - baseline_stats["polar_ecg_coverage"]
        for band in WAVELET_BANDS:
            key = f"wavelet_{band.lower()}"
            row[f"delta_{key}"] = prep_stats[key] - baseline_stats[key]

        rows.append(row)

    return pd.DataFrame(rows).sort_values("subject_id").reset_index(drop=True)


def format_mean(series: pd.Series) -> str:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return "nan"
    return f"{clean.mean():.4f}"


def format_intlike(value: object) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return "nan"
    return str(int(numeric))


def format_floatlike(value: object, digits: int = 3) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return "nan"
    return f"{float(numeric):.{digits}f}"


def build_report(df: pd.DataFrame) -> str:
    lines: list[str] = []
    lines.append("Galaxy Subject Audit")
    lines.append("")
    lines.append(f"num_subjects={len(df)}")
    lines.append(f"valid_10s_task={int(df['valid_10s_task'].sum())}")
    lines.append(f"valid_20s_task={int(df['valid_20s_task'].sum())}")
    lines.append(f"valid_30s_task={int(df['valid_30s_task'].sum())}")
    lines.append("")
    lines.append(f"baseline_duration_s_mean={format_mean(df['baseline_duration_s'])}")
    lines.append(f"tsst_prep_duration_s_mean={format_mean(df['tsst_prep_duration_s'])}")
    lines.append(f"baseline_gw_quality_mean={format_mean(df['baseline_gw_quality'])}")
    lines.append(f"tsst_prep_gw_quality_mean={format_mean(df['tsst_prep_gw_quality'])}")
    lines.append(f"baseline_e4_bvp_coverage_mean={format_mean(df['baseline_e4_bvp_coverage'])}")
    lines.append(f"tsst_prep_e4_bvp_coverage_mean={format_mean(df['tsst_prep_e4_bvp_coverage'])}")
    lines.append(f"baseline_polar_ecg_coverage_mean={format_mean(df['baseline_polar_ecg_coverage'])}")
    lines.append(f"tsst_prep_polar_ecg_coverage_mean={format_mean(df['tsst_prep_polar_ecg_coverage'])}")
    lines.append("")

    lines.append("subjects_invalid_for_30s")
    invalid_30 = df[df["valid_30s_task"] == 0]["subject_id"].tolist()
    lines.append(", ".join(invalid_30) if invalid_30 else "none")
    lines.append("")

    hard_rows = df[
        [
            "subject_id",
            "baseline_windows_20s",
            "tsst_prep_windows_20s",
            "baseline_gw_quality",
            "tsst_prep_gw_quality",
            "delta_acc_mag_mean",
            "delta_wavelet_a4",
            "delta_wavelet_d4",
            "delta_wavelet_d2",
            "delta_wavelet_d1",
        ]
    ]
    lines.append("per_subject_snapshot")
    for row in hard_rows.itertuples(index=False):
        lines.append(
            f"{row.subject_id}: "
            f"b20={format_intlike(row.baseline_windows_20s)} "
            f"p20={format_intlike(row.tsst_prep_windows_20s)} "
            f"bq={format_floatlike(row.baseline_gw_quality)} "
            f"pq={format_floatlike(row.tsst_prep_gw_quality)} "
            f"d_acc={format_floatlike(row.delta_acc_mag_mean)} "
            f"dA4={format_floatlike(row.delta_wavelet_a4)} "
            f"dD4={format_floatlike(row.delta_wavelet_d4)} "
            f"dD2={format_floatlike(row.delta_wavelet_d2)} "
            f"dD1={format_floatlike(row.delta_wavelet_d1)}"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit all GalaxyPPG subjects for baseline vs tsst-prep task readiness.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    df = build_subject_audit(args.dataset_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = args.output_dir / "galaxy_subject_audit.csv"
    txt_path = args.output_dir / "galaxy_subject_audit.txt"
    df.to_csv(csv_path, index=False)
    txt_path.write_text(build_report(df), encoding="utf-8")

    print(f"Saved subject audit CSV to {csv_path}")
    print(f"Saved subject audit TXT to {txt_path}")


if __name__ == "__main__":
    main()
