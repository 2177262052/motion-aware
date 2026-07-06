from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

from .galaxy_dataset import DEFAULT_WAVELET_BANDS, GalaxyPrivilegedWindowDataset


DEFAULT_CALM_SESSIONS = [
    "baseline",
    "meditation-1",
    "meditation-2",
    "rest-1",
    "rest-2",
    "rest-3",
    "rest-4",
    "rest-5",
]
DEFAULT_STRESS_SESSIONS = ["tsst-prep"]


def discover_manifests(manifests_dir: Path, subjects: Iterable[str] | None) -> list[Path]:
    requested = {subject.strip() for subject in subjects or [] if subject.strip()}
    manifests: list[Path] = []
    for path in sorted(manifests_dir.glob("galaxy_*_loso_val.csv")):
        subject = path.stem.replace("galaxy_", "").replace("_loso_val", "")
        if requested and subject not in requested:
            continue
        manifests.append(path)
    return manifests


def resolve_manifest_paths(args: argparse.Namespace) -> list[Path]:
    if args.manifest is not None and args.manifests_dir is not None:
        raise ValueError("Use only one of --manifest or --manifests-dir.")
    if args.manifest is not None:
        return [args.manifest]
    if args.manifests_dir is None:
        raise ValueError("Provide either --manifest or --manifests-dir.")
    manifests = discover_manifests(args.manifests_dir, args.subjects)
    if not manifests:
        raise ValueError(f"No Galaxy LOSO manifests found in {args.manifests_dir}.")
    if args.use_all_manifests:
        return manifests
    return [manifests[0]]


def maybe_parse_sessions(values: Sequence[str] | None, fallback: Iterable[str]) -> list[str]:
    if values is None or len(values) == 0:
        return list(fallback)
    return [str(item) for item in values]


def tensor_to_numpy(value: object) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def scalar_float(value: object) -> float:
    arr = tensor_to_numpy(value).astype(np.float64).reshape(-1)
    if len(arr) == 0:
        return float("nan")
    return float(arr[0])


def safe_std(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return float("nan")
    return float(np.std(values))


def rms(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(values))))


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    n = min(len(a), len(b))
    if n < 3:
        return float("nan")
    a = a[:n]
    b = b[:n]
    if np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spectral_features(values: np.ndarray, sampling_rate: float) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(x) < 8:
        return {
            "ppg_power_0p5_1p5": float("nan"),
            "ppg_power_1p5_4": float("nan"),
            "ppg_power_4_8": float("nan"),
            "ppg_noise_ratio_4_8": float("nan"),
            "ppg_spectral_entropy": float("nan"),
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
        "ppg_power_0p5_1p5": band_ratio(0.5, 1.5),
        "ppg_power_1p5_4": mid,
        "ppg_power_4_8": high,
        "ppg_noise_ratio_4_8": high / max(mid + high, 1e-12),
        "ppg_spectral_entropy": entropy,
    }


def extract_window_features(
    watch_signal: object,
    wavelet_features: object,
    watch_quality: object,
    e4_signal: object | None = None,
    polar_targets: object | None = None,
    polar_target_mask: object | None = None,
    polar_coverage: object | None = None,
    wavelet_bands: Sequence[str] = DEFAULT_WAVELET_BANDS,
    sampling_rate: float = 25.0,
) -> dict[str, float]:
    watch = tensor_to_numpy(watch_signal).astype(np.float64)
    if watch.ndim != 2:
        raise ValueError(f"Expected watch_signal shape [channels, time], got {watch.shape}")
    ppg = watch[0]
    acc = watch[1:] if watch.shape[0] > 1 else np.zeros((1, len(ppg)), dtype=np.float64)
    acc_mag = acc[-1] if acc.shape[0] >= 4 else np.linalg.norm(acc[:3], axis=0)
    ppg_diff = np.diff(ppg)
    acc_diff = np.diff(acc_mag)

    features: dict[str, float] = {
        "watch_quality": scalar_float(watch_quality),
        "ppg_mean": float(np.mean(ppg)),
        "ppg_std": safe_std(ppg),
        "ppg_abs_mean": float(np.mean(np.abs(ppg))),
        "ppg_range": float(np.max(ppg) - np.min(ppg)),
        "ppg_slope_rms": rms(ppg_diff),
        "ppg_abs_slope_mean": float(np.mean(np.abs(ppg_diff))) if len(ppg_diff) else float("nan"),
        "acc_mag_mean": float(np.mean(acc_mag)),
        "acc_mag_std": safe_std(acc_mag),
        "acc_mag_rms": rms(acc_mag),
        "acc_jerk_rms": rms(acc_diff),
        "ppg_acc_corr": safe_corr(np.abs(ppg), acc_mag),
        "ppg_slope_acc_corr": safe_corr(np.abs(ppg_diff), np.abs(acc_diff)),
    }
    features.update(spectral_features(ppg, sampling_rate=sampling_rate))

    wavelet = tensor_to_numpy(wavelet_features).astype(np.float64).reshape(-1)
    for idx, value in enumerate(wavelet):
        name = wavelet_bands[idx] if idx < len(wavelet_bands) else f"band{idx}"
        features[f"watch_wavelet_{name}"] = float(value)

    if e4_signal is not None:
        e4 = tensor_to_numpy(e4_signal).astype(np.float64)
        if e4.ndim == 2 and e4.shape[0] >= 1:
            e4_bvp = e4[0]
            e4_acc = e4[1:] if e4.shape[0] > 1 else np.zeros((1, len(e4_bvp)), dtype=np.float64)
            e4_acc_mag = e4_acc[-1] if e4_acc.shape[0] >= 4 else np.linalg.norm(e4_acc[:3], axis=0)
            features.update(
                {
                    "e4_bvp_std": safe_std(e4_bvp),
                    "e4_bvp_range": float(np.max(e4_bvp) - np.min(e4_bvp)),
                    "e4_bvp_slope_rms": rms(np.diff(e4_bvp)),
                    "e4_acc_mag_rms": rms(e4_acc_mag),
                    "watch_e4_ppg_corr": safe_corr(ppg, e4_bvp),
                }
            )

    if polar_coverage is not None:
        features["polar_coverage"] = scalar_float(polar_coverage)
    if polar_targets is not None:
        targets = tensor_to_numpy(polar_targets).astype(np.float64).reshape(-1)
        mask = (
            tensor_to_numpy(polar_target_mask).astype(np.float64).reshape(-1)
            if polar_target_mask is not None
            else np.ones_like(targets)
        )
        names = ("polar_hr_bpm", "polar_ibi_mean_ms", "polar_ibi_std_ms")
        for idx, value in enumerate(targets):
            name = names[idx] if idx < len(names) else f"polar_target_{idx}"
            features[name] = float(value) if idx < len(mask) and mask[idx] > 0 else float("nan")

    return features


def select_indices(
    dataset: GalaxyPrivilegedWindowDataset,
    max_windows: int | None,
    max_per_subject_session: int | None,
    random_state: int,
) -> list[int]:
    frame = dataset.manifest.reset_index().rename(columns={"index": "dataset_index"})
    if max_per_subject_session is not None and max_per_subject_session > 0:
        sampled = []
        for _, group in frame.groupby(["subject_id", "session"], sort=False):
            n = min(len(group), max_per_subject_session)
            sampled.append(group.sample(n=n, random_state=random_state, replace=False))
        frame = pd.concat(sampled, axis=0, ignore_index=True) if sampled else frame.iloc[0:0]
    if max_windows is not None and max_windows > 0 and len(frame) > max_windows:
        frame = frame.sample(n=max_windows, random_state=random_state, replace=False)
    frame = frame.sort_values(["subject_id", "session", "window_start_ms"]).reset_index(drop=True)
    return frame["dataset_index"].astype(int).tolist()


def build_window_frame(
    manifest_paths: list[Path],
    dataset_root: Path,
    splits: Sequence[str],
    include_sessions: Sequence[str],
    args: argparse.Namespace,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    seen: set[tuple[str, str, int, int]] = set()
    for manifest_path in manifest_paths:
        for split in splits:
            dataset = GalaxyPrivilegedWindowDataset(
                manifest_csv=manifest_path,
                split=split,
                dataset_root=dataset_root,
                include_sessions=include_sessions,
                cache_tables=True,
                wavelet=args.wavelet,
                wavelet_level=args.wavelet_level,
                wavelet_bands=DEFAULT_WAVELET_BANDS,
            )
            if len(dataset) == 0:
                continue
            indices = select_indices(
                dataset,
                max_windows=args.max_windows_per_split,
                max_per_subject_session=args.max_per_subject_session,
                random_state=args.random_state,
            )
            for index in tqdm(indices, desc=f"{manifest_path.stem}:{split}", leave=True):
                sample = dataset[index]
                key = (
                    str(sample["subject_id"]),
                    str(sample["session"]),
                    int(sample["window_start_ms"]),
                    int(sample["window_end_ms"]),
                )
                if key in seen:
                    continue
                seen.add(key)
                features = extract_window_features(
                    sample["watch_signal"],
                    sample["wavelet_features"],
                    sample["watch_quality"],
                    e4_signal=sample["e4_signal"],
                    polar_targets=sample["polar_targets"],
                    polar_target_mask=sample["polar_target_mask"],
                    polar_coverage=sample["polar_coverage"],
                    wavelet_bands=DEFAULT_WAVELET_BANDS,
                )
                rows.append(
                    {
                        "source_manifest": str(manifest_path),
                        "source_split": split,
                        "subject_id": str(sample["subject_id"]),
                        "subject_index": int(sample["subject_index"]),
                        "session": str(sample["session"]),
                        "group_name": str(sample["group_name"]),
                        "label": int(sample["label"]),
                        "window_start_ms": int(sample["window_start_ms"]),
                        "window_end_ms": int(sample["window_end_ms"]),
                        **features,
                    }
                )
    if not rows:
        raise ValueError("No windows collected. Check manifests, splits, and session filters.")
    return pd.DataFrame(rows)


def numeric_feature_columns(frame: pd.DataFrame) -> list[str]:
    exclude = {
        "label",
        "subject_index",
        "window_start_ms",
        "window_end_ms",
    }
    cols = []
    for col in frame.columns:
        if col in exclude:
            continue
        if pd.api.types.is_numeric_dtype(frame[col]):
            cols.append(col)
    return cols


def cohens_d(x0: np.ndarray, x1: np.ndarray) -> float:
    x0 = x0[np.isfinite(x0)]
    x1 = x1[np.isfinite(x1)]
    if len(x0) < 2 or len(x1) < 2:
        return float("nan")
    var0 = np.var(x0, ddof=1)
    var1 = np.var(x1, ddof=1)
    pooled = ((len(x0) - 1) * var0 + (len(x1) - 1) * var1) / max(len(x0) + len(x1) - 2, 1)
    if pooled <= 0:
        return float("nan")
    return float((np.mean(x1) - np.mean(x0)) / np.sqrt(pooled))


def safe_auc(y: np.ndarray, values: np.ndarray) -> float:
    mask = np.isfinite(values)
    y = y[mask]
    values = values[mask]
    if len(np.unique(y)) < 2 or len(np.unique(values)) < 2:
        return float("nan")
    return float(roc_auc_score(y, values))


def normalize_within_subject(frame: pd.DataFrame, feature_cols: Sequence[str]) -> pd.DataFrame:
    out = frame.copy()
    for _, idx in out.groupby("subject_id").groups.items():
        block = out.loc[idx, feature_cols].astype(float)
        means = block.mean(axis=0)
        stds = block.std(axis=0, ddof=0).replace(0.0, np.nan)
        out.loc[idx, feature_cols] = (block - means) / stds
    out[list(feature_cols)] = out[list(feature_cols)].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def summarize_univariate_auc(frame: pd.DataFrame, feature_cols: Sequence[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    y = frame["label"].to_numpy(dtype=int)
    normalized = normalize_within_subject(frame, feature_cols)
    for col in feature_cols:
        raw_values = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=np.float64)
        norm_values = pd.to_numeric(normalized[col], errors="coerce").to_numpy(dtype=np.float64)
        auc = safe_auc(y, raw_values)
        norm_auc = safe_auc(y, norm_values)
        x0 = raw_values[y == 0]
        x1 = raw_values[y == 1]
        rows.append(
            {
                "feature": col,
                "auc": auc,
                "separability_auc": max(auc, 1.0 - auc) if np.isfinite(auc) else float("nan"),
                "subject_norm_auc": norm_auc,
                "subject_norm_separability_auc": max(norm_auc, 1.0 - norm_auc) if np.isfinite(norm_auc) else float("nan"),
                "calm_mean": float(np.nanmean(x0)) if len(x0) else float("nan"),
                "stress_mean": float(np.nanmean(x1)) if len(x1) else float("nan"),
                "delta_stress_minus_calm": float(np.nanmean(x1) - np.nanmean(x0)) if len(x0) and len(x1) else float("nan"),
                "cohens_d": cohens_d(x0, x1),
                "finite_fraction": float(np.isfinite(raw_values).mean()) if len(raw_values) else 0.0,
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(
        ["subject_norm_separability_auc", "separability_auc"],
        ascending=[False, False],
    ).reset_index(drop=True)


def add_quantile_bin(frame: pd.DataFrame, column: str, output_column: str) -> pd.DataFrame:
    out = frame.copy()
    labels = ["low", "mid", "high"]
    values = pd.to_numeric(out[column], errors="coerce")
    try:
        out[output_column] = pd.qcut(values, q=3, labels=labels, duplicates="drop")
    except ValueError:
        out[output_column] = "all"
    out[output_column] = out[output_column].astype(str).replace("nan", "missing")
    return out


def summarize_bins(frame: pd.DataFrame) -> pd.DataFrame:
    out = add_quantile_bin(frame, "watch_quality", "watch_quality_bin")
    out = add_quantile_bin(out, "acc_jerk_rms", "motion_bin")
    rows: list[pd.DataFrame] = []
    for keys in (["watch_quality_bin"], ["motion_bin"], ["watch_quality_bin", "motion_bin"]):
        grouped = (
            out.groupby(keys, dropna=False)
            .agg(
                n=("label", "size"),
                positive_prior=("label", "mean"),
                watch_quality_mean=("watch_quality", "mean"),
                acc_jerk_rms_mean=("acc_jerk_rms", "mean"),
                acc_mag_rms_mean=("acc_mag_rms", "mean"),
                ppg_noise_ratio_4_8_mean=("ppg_noise_ratio_4_8", "mean"),
                polar_hr_bpm_mean=("polar_hr_bpm", "mean"),
            )
            .reset_index()
        )
        grouped.insert(0, "grouping", "+".join(keys))
        rows.append(grouped)
    return pd.concat(rows, axis=0, ignore_index=True)


def summarize_subject_session(frame: pd.DataFrame, feature_cols: Sequence[str]) -> pd.DataFrame:
    keep_cols = [
        "label",
        "watch_quality",
        "acc_jerk_rms",
        "acc_mag_rms",
        "ppg_noise_ratio_4_8",
        "ppg_spectral_entropy",
        "polar_hr_bpm",
        "polar_ibi_std_ms",
    ]
    keep_cols = [col for col in keep_cols if col in feature_cols or col == "label"]
    return (
        frame.groupby(["subject_id", "session"], as_index=False)
        .agg(
            n=("label", "size"),
            positive_prior=("label", "mean"),
            **{f"{col}_mean": (col, "mean") for col in keep_cols if col != "label"},
        )
        .sort_values(["subject_id", "session"])
        .reset_index(drop=True)
    )


def summarize_correlations(frame: pd.DataFrame, feature_cols: Sequence[str]) -> pd.DataFrame:
    anchors = [
        "label",
        "watch_quality",
        "acc_jerk_rms",
        "acc_mag_rms",
        "ppg_noise_ratio_4_8",
        "polar_hr_bpm",
    ]
    rows = []
    numeric = frame[list(feature_cols) + ["label"]].apply(pd.to_numeric, errors="coerce")
    for anchor in anchors:
        if anchor not in numeric.columns:
            continue
        for col in feature_cols:
            corr = numeric[anchor].corr(numeric[col])
            rows.append({"anchor": anchor, "feature": col, "pearson_corr": float(corr) if pd.notna(corr) else float("nan")})
    return pd.DataFrame(rows).sort_values("pearson_corr", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose GalaxyPPG data quality, motion, wavelet, and rhythm features before changing the model."
    )
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--manifests-dir", type=Path, default=None)
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument(
        "--use-all-manifests",
        action="store_true",
        help="Use every LOSO manifest. By default only the first manifest is used to avoid duplicate windows.",
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="*", default=["train", "val", "test"], choices=["train", "val", "test"])
    parser.add_argument("--calm-sessions", nargs="*", default=DEFAULT_CALM_SESSIONS)
    parser.add_argument("--stress-sessions", nargs="*", default=DEFAULT_STRESS_SESSIONS)
    parser.add_argument("--wavelet", type=str, default="db4")
    parser.add_argument("--wavelet-level", type=int, default=4)
    parser.add_argument("--max-windows-per-split", type=int, default=None)
    parser.add_argument("--max-per-subject-session", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=7)
    args = parser.parse_args()

    manifest_paths = resolve_manifest_paths(args)
    include_sessions = maybe_parse_sessions(args.calm_sessions, DEFAULT_CALM_SESSIONS) + maybe_parse_sessions(
        args.stress_sessions,
        DEFAULT_STRESS_SESSIONS,
    )
    frame = build_window_frame(
        manifest_paths=manifest_paths,
        dataset_root=args.dataset_root,
        splits=list(args.splits),
        include_sessions=include_sessions,
        args=args,
    )
    feature_cols = numeric_feature_columns(frame)
    auc_summary = summarize_univariate_auc(frame, feature_cols)
    bin_summary = summarize_bins(frame)
    subject_session = summarize_subject_session(frame, feature_cols)
    corr_summary = summarize_correlations(frame, feature_cols)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    windows_path = args.output_dir / "galaxy_quality_windows.csv"
    auc_path = args.output_dir / "galaxy_quality_feature_auc.csv"
    bins_path = args.output_dir / "galaxy_quality_bins.csv"
    subject_session_path = args.output_dir / "galaxy_quality_subject_session.csv"
    corr_path = args.output_dir / "galaxy_quality_feature_correlations.csv"

    frame.to_csv(windows_path, index=False)
    auc_summary.to_csv(auc_path, index=False)
    bin_summary.to_csv(bins_path, index=False)
    subject_session.to_csv(subject_session_path, index=False)
    corr_summary.to_csv(corr_path, index=False)

    print(f"collected_windows={len(frame)}")
    print("Top subject-normalized single-feature separators:")
    print(auc_summary.head(12).to_string(index=False))
    print()
    print("Quality/motion bins:")
    print(bin_summary.head(18).to_string(index=False))
    print()
    print(f"Saved windows to {windows_path}")
    print(f"Saved feature AUROC summary to {auc_path}")
    print(f"Saved quality/motion bins to {bins_path}")
    print(f"Saved subject-session summary to {subject_session_path}")
    print(f"Saved feature correlations to {corr_path}")


if __name__ == "__main__":
    main()
