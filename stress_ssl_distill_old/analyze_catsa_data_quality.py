from __future__ import annotations

import argparse
import math
from pathlib import Path, PureWindowsPath
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

from .catsa_dataset import (
    CATSA_SAMPLE_RATES,
    _preprocess_catsa_acc,
    _preprocess_catsa_bvp,
    _preprocess_catsa_scalar,
    _read_acc_csv,
    _read_scalar_csv,
)
from .galaxy_dataset import DEFAULT_WAVELET_BANDS, compute_wavelet_band_ratios, resample_array


DEFAULT_CALM_SESSIONS = ["Baseline"]
DEFAULT_STRESS_SESSIONS = ["Stroop", "Logic", "Sudoku"]


def maybe_parse_sessions(values: Sequence[str] | None, fallback: Iterable[str]) -> list[str]:
    if values is None or len(values) == 0:
        return list(fallback)
    return [str(item) for item in values]


def resolve_session_dir(catsa_root: Path, rel_path: str) -> Path:
    rel = PureWindowsPath(str(rel_path))
    parts = list(rel.parts)
    if parts and parts[0].lower() == catsa_root.name.lower():
        parts = parts[1:]
    return catsa_root.joinpath(*parts)


def safe_std(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(x) == 0:
        return float("nan")
    return float(np.std(x))


def safe_mean(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(x) == 0:
        return float("nan")
    return float(np.mean(x))


def safe_median(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(x) == 0:
        return float("nan")
    return float(np.median(x))


def safe_range(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(x) == 0:
        return float("nan")
    return float(np.max(x) - np.min(x))


def rms(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(x) == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(x))))


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    n = min(len(x), len(y))
    if n < 3:
        return float("nan")
    x = x[:n]
    y = y[:n]
    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def linear_slope(values: np.ndarray, sampling_rate: float) -> float:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(x) < 3:
        return float("nan")
    t = np.arange(len(x), dtype=np.float64) / float(sampling_rate)
    t = t - np.mean(t)
    denom = float(np.sum(np.square(t)))
    if denom <= 1e-12:
        return float("nan")
    return float(np.sum(t * (x - np.mean(x))) / denom)


def spectral_features(values: np.ndarray, sampling_rate: float, prefix: str) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(x) < 16:
        return {
            f"{prefix}_power_0p5_1p5": float("nan"),
            f"{prefix}_power_1p5_4": float("nan"),
            f"{prefix}_power_4_8": float("nan"),
            f"{prefix}_noise_ratio_4_8": float("nan"),
            f"{prefix}_spectral_entropy": float("nan"),
        }
    x = x - np.mean(x)
    freqs = np.fft.rfftfreq(len(x), d=1.0 / sampling_rate)
    power = np.square(np.abs(np.fft.rfft(x))).astype(np.float64)
    total = float(np.sum(power[(freqs >= 0.5) & (freqs <= 8.0)])) + 1e-12

    def band_ratio(low: float, high: float) -> float:
        mask = (freqs >= low) & (freqs < high)
        return float(np.sum(power[mask]) / total)

    valid = power[(freqs >= 0.5) & (freqs <= 8.0)]
    probs = valid / (float(np.sum(valid)) + 1e-12)
    entropy = -float(np.sum(probs * np.log(probs + 1e-12)) / math.log(max(len(probs), 2)))
    high = band_ratio(4.0, 8.0)
    mid = band_ratio(1.5, 4.0)
    return {
        f"{prefix}_power_0p5_1p5": band_ratio(0.5, 1.5),
        f"{prefix}_power_1p5_4": mid,
        f"{prefix}_power_4_8": high,
        f"{prefix}_noise_ratio_4_8": high / max(mid + high, 1e-12),
        f"{prefix}_spectral_entropy": entropy,
    }


def signal_summary(values: np.ndarray, sampling_rate: float, prefix: str) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    dx = np.diff(x)
    out = {
        f"{prefix}_mean": safe_mean(x),
        f"{prefix}_median": safe_median(x),
        f"{prefix}_std": safe_std(x),
        f"{prefix}_range": safe_range(x),
        f"{prefix}_slope": linear_slope(x, sampling_rate),
        f"{prefix}_diff_rms": rms(dx),
        f"{prefix}_abs_diff_mean": safe_mean(np.abs(dx)) if len(dx) else float("nan"),
    }
    return out


def slice_stream(stream: np.ndarray, sampling_rate: float, start_s: float, end_s: float) -> np.ndarray:
    start_idx = max(int(round(start_s * sampling_rate)), 0)
    end_idx = min(int(round(end_s * sampling_rate)), len(stream))
    return np.asarray(stream[start_idx:end_idx], dtype=np.float32)


def load_session_streams(session_dir: Path, target_watch_sr: int) -> dict[str, np.ndarray]:
    raw_bvp = _read_scalar_csv(session_dir / "BVP.csv", "BVP")
    raw_acc = _read_acc_csv(session_dir / "ACC.csv")
    raw_streams: dict[str, np.ndarray] = {
        "raw_bvp": np.nan_to_num(raw_bvp, nan=0.0, posinf=0.0, neginf=0.0),
        "raw_acc": np.nan_to_num(raw_acc, nan=0.0, posinf=0.0, neginf=0.0),
    }
    for modality in ("EDA", "TEMP", "HR"):
        path = session_dir / f"{modality}.csv"
        if path.exists():
            raw_streams[f"raw_{modality.lower()}"] = np.nan_to_num(
                _read_scalar_csv(path, modality),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

    proc_bvp = _preprocess_catsa_bvp(raw_streams["raw_bvp"], sampling_rate=CATSA_SAMPLE_RATES["BVP"])
    proc_bvp = resample_array(proc_bvp, orig_sr=CATSA_SAMPLE_RATES["BVP"], target_sr=target_watch_sr)
    proc_acc = _preprocess_catsa_acc(raw_streams["raw_acc"])
    proc_acc = resample_array(proc_acc, orig_sr=CATSA_SAMPLE_RATES["ACC"], target_sr=target_watch_sr)
    min_len = min(len(proc_bvp), len(proc_acc))
    raw_streams["proc_bvp"] = proc_bvp[:min_len]
    raw_streams["proc_acc"] = proc_acc[:min_len]

    for modality in ("EDA", "TEMP", "HR"):
        key = f"raw_{modality.lower()}"
        if key not in raw_streams:
            continue
        z = _preprocess_catsa_scalar(raw_streams[key])
        raw_streams[f"proc_{modality.lower()}"] = resample_array(
            z,
            orig_sr=CATSA_SAMPLE_RATES[modality],
            target_sr=target_watch_sr,
        )
    return raw_streams


def extract_window_features(
    streams: dict[str, np.ndarray],
    start_s: float,
    end_s: float,
    wavelet: str,
    wavelet_level: int,
    wavelet_bands: Sequence[str],
    target_watch_sr: int,
) -> dict[str, float]:
    row: dict[str, float] = {}

    raw_bvp = slice_stream(streams["raw_bvp"], CATSA_SAMPLE_RATES["BVP"], start_s, end_s)
    proc_bvp = slice_stream(streams["proc_bvp"], target_watch_sr, start_s, end_s)
    raw_acc = slice_stream(streams["raw_acc"], CATSA_SAMPLE_RATES["ACC"], start_s, end_s)
    proc_acc = slice_stream(streams["proc_acc"], target_watch_sr, start_s, end_s)

    raw_acc_mag = np.linalg.norm(raw_acc[:, :3], axis=1) if raw_acc.ndim == 2 and raw_acc.shape[1] >= 3 else raw_acc.reshape(-1)
    proc_acc_mag = proc_acc[:, -1] if proc_acc.ndim == 2 and proc_acc.shape[1] >= 4 else np.linalg.norm(proc_acc[:, :3], axis=1)

    row.update(signal_summary(raw_bvp, CATSA_SAMPLE_RATES["BVP"], "raw_bvp"))
    row.update(signal_summary(proc_bvp, target_watch_sr, "proc_bvp"))
    row.update(spectral_features(proc_bvp, target_watch_sr, "proc_bvp"))
    row.update(signal_summary(raw_acc_mag, CATSA_SAMPLE_RATES["ACC"], "raw_acc_mag"))
    row.update(signal_summary(proc_acc_mag, target_watch_sr, "proc_acc_mag"))
    row["proc_bvp_acc_corr"] = safe_corr(np.abs(proc_bvp), proc_acc_mag)
    row["proc_bvp_slope_acc_corr"] = safe_corr(np.abs(np.diff(proc_bvp)), np.abs(np.diff(proc_acc_mag)))

    try:
        wavelet_values = compute_wavelet_band_ratios(
            proc_bvp,
            wavelet=wavelet,
            level=wavelet_level,
            selected_bands=wavelet_bands,
        )
    except Exception:
        wavelet_values = np.full((len(wavelet_bands),), np.nan, dtype=np.float32)
    for idx, value in enumerate(wavelet_values):
        name = wavelet_bands[idx] if idx < len(wavelet_bands) else f"band{idx}"
        row[f"watch_wavelet_{name}"] = float(value)

    for modality in ("eda", "temp", "hr"):
        raw_key = f"raw_{modality}"
        proc_key = f"proc_{modality}"
        if raw_key in streams:
            raw_slice = slice_stream(streams[raw_key], CATSA_SAMPLE_RATES[modality.upper()], start_s, end_s)
            row.update(signal_summary(raw_slice, CATSA_SAMPLE_RATES[modality.upper()], f"raw_{modality}"))
        if proc_key in streams:
            proc_slice = slice_stream(streams[proc_key], target_watch_sr, start_s, end_s)
            row.update(signal_summary(proc_slice, target_watch_sr, f"proc_{modality}"))

    return row


def select_rows(frame: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = frame.copy()
    if "split" in out.columns and args.splits:
        out = out[out["split"].astype(str).isin({str(item) for item in args.splits})]
    sessions = set(maybe_parse_sessions(args.calm_sessions, DEFAULT_CALM_SESSIONS)) | set(
        maybe_parse_sessions(args.stress_sessions, DEFAULT_STRESS_SESSIONS)
    )
    out = out[out["session"].astype(str).isin(sessions)].reset_index(drop=True)

    if args.max_per_subject_session is not None and args.max_per_subject_session > 0:
        sampled = []
        for _, group in out.groupby(["subject_id", "session"], sort=False):
            n = min(int(args.max_per_subject_session), len(group))
            sampled.append(group.sample(n=n, random_state=args.random_state, replace=False))
        out = pd.concat(sampled, axis=0, ignore_index=True) if sampled else out.iloc[0:0]

    if args.max_windows is not None and args.max_windows > 0 and len(out) > args.max_windows:
        out = out.sample(n=int(args.max_windows), random_state=args.random_state, replace=False)

    return out.sort_values(["subject_id", "session", "window_start_s"]).reset_index(drop=True)


def build_window_frame(frame: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    cache: dict[Path, dict[str, np.ndarray]] = {}
    catsa_root = args.catsa_root
    for _, item in tqdm(frame.iterrows(), total=len(frame), desc="catsa quality windows", leave=True):
        session_dir = resolve_session_dir(catsa_root, str(item["subject_session_path"]))
        streams = cache.get(session_dir)
        if streams is None:
            streams = load_session_streams(session_dir, target_watch_sr=args.target_watch_sr)
            cache[session_dir] = streams

        features = extract_window_features(
            streams,
            start_s=float(item["window_start_s"]),
            end_s=float(item["window_end_s"]),
            wavelet=args.wavelet,
            wavelet_level=args.wavelet_level,
            wavelet_bands=DEFAULT_WAVELET_BANDS,
            target_watch_sr=args.target_watch_sr,
        )
        rows.append(
            {
                "subject_id": str(item["subject_id"]),
                "subject_index": int(item["subject_index"]),
                "session": str(item["session"]),
                "group_name": str(item.get("group_name", "")),
                "label": int(item["label"]),
                "window_start_s": float(item["window_start_s"]),
                "window_end_s": float(item["window_end_s"]),
                "window_start_ms": int(item["window_start_ms"]),
                "window_end_ms": int(item["window_end_ms"]),
                "subject_session_path": str(item["subject_session_path"]),
                **features,
            }
        )
    return pd.DataFrame(rows)


def numeric_feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {
        "subject_index",
        "label",
        "window_start_s",
        "window_end_s",
        "window_start_ms",
        "window_end_ms",
    }
    cols: list[str] = []
    for column in frame.columns:
        if column in excluded:
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            cols.append(column)
    return cols


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


def feature_source(name: str) -> str:
    if name.startswith("raw_eda") or name.startswith("raw_temp") or name.startswith("raw_hr"):
        return "privileged_raw"
    if name.startswith("proc_eda") or name.startswith("proc_temp") or name.startswith("proc_hr"):
        return "privileged_zscore"
    if name.startswith("raw_bvp") or name.startswith("proc_bvp") or name.startswith("raw_acc") or name.startswith("proc_acc"):
        return "watch"
    if name.startswith("watch_wavelet"):
        return "watch_wavelet"
    return "other"


def summarize_feature_auc(windows: pd.DataFrame, calm_sessions: Sequence[str], stress_sessions: Sequence[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    features = numeric_feature_columns(windows)
    baseline = windows[windows["session"].isin(calm_sessions)].copy()
    tasks = [("pooled", list(stress_sessions))]
    tasks.extend((session, [session]) for session in stress_sessions)
    for task_name, task_sessions in tasks:
        task = windows[windows["session"].isin(task_sessions)].copy()
        subset = pd.concat([baseline, task], axis=0, ignore_index=True)
        subset["task_label"] = subset["session"].isin(task_sessions).astype(int)
        for feature in features:
            auc = safe_auc(subset["task_label"], subset[feature])
            rows.append(
                {
                    "task": task_name,
                    "feature": feature,
                    "source": feature_source(feature),
                    "auc": auc,
                    "abs_auc": max(auc, 1.0 - auc) if np.isfinite(auc) else float("nan"),
                    "direction": "higher_in_stress" if np.isfinite(auc) and auc >= 0.5 else "lower_in_stress",
                    "n": int(subset[[feature, "task_label"]].dropna().shape[0]),
                    "baseline_mean": float(pd.to_numeric(baseline[feature], errors="coerce").mean()),
                    "stress_mean": float(pd.to_numeric(task[feature], errors="coerce").mean()),
                }
            )
    out = pd.DataFrame(rows)
    return out.sort_values(["task", "abs_auc"], ascending=[True, False]).reset_index(drop=True)


def summarize_subject_session(windows: pd.DataFrame) -> pd.DataFrame:
    features = numeric_feature_columns(windows)
    agg: dict[str, tuple[str, str] | str] = {
        "label": "first",
        "group_name": "first",
        "window_start_s": "min",
        "window_end_s": "max",
    }
    for feature in features:
        agg[feature] = "mean"
    return windows.groupby(["subject_id", "session"], as_index=False).agg(agg).reset_index(drop=True)


def standardized_mean_diff(a: pd.Series, b: pd.Series) -> float:
    x = pd.to_numeric(a, errors="coerce").dropna().astype(float)
    y = pd.to_numeric(b, errors="coerce").dropna().astype(float)
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    pooled = math.sqrt((float(np.var(x)) + float(np.var(y))) * 0.5)
    if pooled <= 1e-12:
        return float("nan")
    return float((float(np.mean(y)) - float(np.mean(x))) / pooled)


def summarize_task_shifts(windows: pd.DataFrame, stress_sessions: Sequence[str]) -> pd.DataFrame:
    features = numeric_feature_columns(windows)
    rows: list[dict[str, object]] = []
    sessions = [session for session in stress_sessions if session in set(windows["session"])]
    for idx, left in enumerate(sessions):
        for right in sessions[idx + 1 :]:
            left_frame = windows[windows["session"] == left]
            right_frame = windows[windows["session"] == right]
            for feature in features:
                smd = standardized_mean_diff(left_frame[feature], right_frame[feature])
                rows.append(
                    {
                        "session_a": left,
                        "session_b": right,
                        "feature": feature,
                        "source": feature_source(feature),
                        "smd_b_minus_a": smd,
                        "abs_smd": abs(smd) if np.isfinite(smd) else float("nan"),
                        "session_a_mean": float(pd.to_numeric(left_frame[feature], errors="coerce").mean()),
                        "session_b_mean": float(pd.to_numeric(right_frame[feature], errors="coerce").mean()),
                    }
                )
    out = pd.DataFrame(rows)
    return out.sort_values(["session_a", "session_b", "abs_smd"], ascending=[True, True, False]).reset_index(drop=True)


def print_top_auc(auc_frame: pd.DataFrame, top_k: int) -> None:
    print()
    print("Top task separability features:")
    for task, group in auc_frame.groupby("task", sort=False):
        print(f"[{task}]")
        cols = ["feature", "source", "auc", "abs_auc", "direction", "baseline_mean", "stress_mean"]
        print(group.head(top_k)[cols].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze CATSA watch/privileged signal separability and task heterogeneity.")
    parser.add_argument("--windows-csv", "--manifest", dest="windows_csv", type=Path, required=True)
    parser.add_argument("--catsa-root", "--dataset-root", dest="catsa_root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="*", default=None)
    parser.add_argument("--calm-sessions", nargs="*", default=DEFAULT_CALM_SESSIONS)
    parser.add_argument("--stress-sessions", nargs="*", default=DEFAULT_STRESS_SESSIONS)
    parser.add_argument("--target-watch-sr", type=int, default=32)
    parser.add_argument("--wavelet", type=str, default="db4")
    parser.add_argument("--wavelet-level", type=int, default=4)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--max-per-subject-session", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=12)
    args = parser.parse_args()

    calm_sessions = maybe_parse_sessions(args.calm_sessions, DEFAULT_CALM_SESSIONS)
    stress_sessions = maybe_parse_sessions(args.stress_sessions, DEFAULT_STRESS_SESSIONS)
    source = pd.read_csv(args.windows_csv)
    selected = select_rows(source, args)
    if selected.empty:
        raise ValueError("No CATSA windows remain after filtering.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    windows = build_window_frame(selected, args)
    subject_session = summarize_subject_session(windows)
    auc_frame = summarize_feature_auc(windows, calm_sessions=calm_sessions, stress_sessions=stress_sessions)
    task_shifts = summarize_task_shifts(windows, stress_sessions=stress_sessions)
    session_summary = windows.groupby(["session", "label"], as_index=False).agg(
        n=("label", "size"),
        subjects=("subject_id", "nunique"),
        proc_bvp_std_mean=("proc_bvp_std", "mean"),
        proc_acc_jerk_mean=("proc_acc_mag_diff_rms", "mean"),
        raw_eda_mean=("raw_eda_mean", "mean") if "raw_eda_mean" in windows.columns else ("label", "size"),
        raw_temp_mean=("raw_temp_mean", "mean") if "raw_temp_mean" in windows.columns else ("label", "size"),
        raw_hr_mean=("raw_hr_mean", "mean") if "raw_hr_mean" in windows.columns else ("label", "size"),
    )

    windows_path = args.output_dir / "catsa_quality_windows.csv"
    subject_session_path = args.output_dir / "catsa_quality_subject_session.csv"
    auc_path = args.output_dir / "catsa_quality_feature_auc.csv"
    task_shift_path = args.output_dir / "catsa_quality_task_shifts.csv"
    session_summary_path = args.output_dir / "catsa_quality_session_summary.csv"

    windows.to_csv(windows_path, index=False)
    subject_session.to_csv(subject_session_path, index=False)
    auc_frame.to_csv(auc_path, index=False)
    task_shifts.to_csv(task_shift_path, index=False)
    session_summary.to_csv(session_summary_path, index=False)

    print(f"windows={len(windows)} subjects={windows['subject_id'].nunique()}")
    print(windows.groupby(["session", "label"]).size().to_string())
    print_top_auc(auc_frame, top_k=args.top_k)
    print()
    print(f"Saved window features to {windows_path}")
    print(f"Saved subject/session summary to {subject_session_path}")
    print(f"Saved feature AUCs to {auc_path}")
    print(f"Saved task shifts to {task_shift_path}")
    print(f"Saved session summary to {session_summary_path}")


if __name__ == "__main__":
    main()
