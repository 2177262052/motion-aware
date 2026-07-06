from __future__ import annotations

from collections import OrderedDict
from pathlib import Path, PureWindowsPath
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from scipy import signal
from torch.utils.data import Dataset

from .galaxy_dataset import (
    DEFAULT_WAVELET_BANDS,
    compute_wavelet_band_ratios,
    ensure_length,
    resample_array,
)


CATSA_SAMPLE_RATES = {
    "BVP": 64.0,
    "ACC": 32.0,
    "EDA": 4.0,
    "TEMP": 4.0,
    "HR": 1.0,
}

DEFAULT_CATSA_TARGET_RATES = {
    "watch": 32,
    "privileged": 32,
}

DEFAULT_PRIVILEGED_MODALITIES = ("EDA", "TEMP")

CATSA_PLAUSIBLE_RANGES = {
    "BVP": (-5000.0, 5000.0),
    "ACC": (-512.0, 512.0),
    "EDA": (0.0, 100.0),
    "TEMP": (15.0, 45.0),
    "HR": (25.0, 240.0),
}


def _clean_catsa_channel(values: np.ndarray, low: float | None, high: float | None) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32).reshape(-1)
    if len(x) == 0:
        return x
    valid = np.isfinite(x)
    if low is not None:
        valid &= x >= low
    if high is not None:
        valid &= x <= high
    if valid.any():
        median = float(np.median(x[valid]))
    else:
        return np.zeros_like(x, dtype=np.float32)
    series = pd.Series(x)
    series.loc[~valid] = np.nan
    series = series.interpolate(limit_direction="both").fillna(median)
    return series.to_numpy(dtype=np.float32)


def _clean_catsa_values(values: np.ndarray, modality: str) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32)
    low, high = CATSA_PLAUSIBLE_RANGES.get(str(modality).upper(), (None, None))
    if x.ndim == 1:
        return _clean_catsa_channel(x, low, high)
    channels = [_clean_catsa_channel(x[:, idx], low, high) for idx in range(x.shape[1])]
    return np.stack(channels, axis=1).astype(np.float32)


def _safe_robust_zscore(values: np.ndarray, min_scale: float = 1e-3) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    median = np.median(values, axis=0, keepdims=True)
    mad = np.median(np.abs(values - median), axis=0, keepdims=True)
    robust_scale = 1.4826 * mad
    std_scale = np.std(values, axis=0, keepdims=True)
    scale = np.where(robust_scale >= min_scale, robust_scale, std_scale)
    scale = np.where(scale >= min_scale, scale, 1.0)
    return ((values - median) / scale).astype(np.float32)


def _safe_bandpass_zscore(values: np.ndarray, sampling_rate: float, low_hz: float = 0.5, high_hz: float = 8.0) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if len(values) < max(int(sampling_rate), 16):
        return _safe_robust_zscore(values)

    nyquist = sampling_rate / 2.0
    low = max(low_hz / nyquist, 1e-4)
    high = min(high_hz / nyquist, 0.99)
    if low >= high:
        return _safe_robust_zscore(values)

    b, a = signal.butter(3, [low, high], btype="bandpass")
    filtered = signal.filtfilt(b, a, values).astype(np.float32)
    return _safe_robust_zscore(filtered)


def _preprocess_catsa_bvp(values: np.ndarray, sampling_rate: float) -> np.ndarray:
    return _safe_bandpass_zscore(values, sampling_rate=sampling_rate)


def _preprocess_catsa_acc(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 1:
        values = values[:, None]
    values = values / 64.0
    magnitude = np.linalg.norm(values, axis=1, keepdims=True)
    merged = np.concatenate([values, magnitude], axis=1)
    return _safe_robust_zscore(merged)


def _preprocess_catsa_scalar(values: np.ndarray) -> np.ndarray:
    return _safe_robust_zscore(np.asarray(values, dtype=np.float32).reshape(-1))


def _is_time_like_column(column: object) -> bool:
    name = str(column).strip().lower()
    return (
        name in {"time", "timestamp", "timestamps", "datetime", "date", "unix", "ms", "seconds", "second"}
        or name.startswith("time")
        or "timestamp" in name
    )


def _numeric_column_names(frame: pd.DataFrame) -> list[object]:
    columns: list[object] = []
    for column in frame.columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.notna().any():
            columns.append(column)
    return columns


def _read_scalar_csv(path: Path, column: str) -> np.ndarray:
    frame = pd.read_csv(path)
    normalized = {str(name).strip().lower(): name for name in frame.columns}
    wanted = str(column).strip().lower()
    if wanted in normalized:
        series = frame[normalized[wanted]]
    else:
        named_candidates = [
            name
            for name in frame.columns
            if wanted in str(name).strip().lower() and not _is_time_like_column(name)
        ]
        if named_candidates:
            series = frame[named_candidates[0]]
        else:
            numeric_columns = _numeric_column_names(frame)
            signal_columns = [name for name in numeric_columns if not _is_time_like_column(name)]
            if signal_columns:
                series = frame[signal_columns[-1]]
            elif numeric_columns:
                series = frame[numeric_columns[-1]]
            else:
                series = frame.iloc[:, -1]
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float32)
    return _clean_catsa_values(values, column)


def _read_acc_csv(path: Path) -> np.ndarray:
    frame = pd.read_csv(path)
    normalized = {str(name).strip().lower(): name for name in frame.columns}
    columns = [
        normalized[name]
        for name in ("acc_x", "acc_y", "acc_z")
        if name in normalized
    ]
    if len(columns) != 3:
        columns = [
            normalized[name]
            for name in ("x", "y", "z")
            if name in normalized
        ]
    if len(columns) == 3:
        values = frame[columns]
    else:
        numeric_columns = _numeric_column_names(frame)
        signal_columns = [name for name in numeric_columns if not _is_time_like_column(name)]
        columns = signal_columns[:3] if len(signal_columns) >= 3 else numeric_columns[-3:]
        values = frame[columns]
    array = values.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    return _clean_catsa_values(array, "ACC")


class CATSABaseWindowDataset(Dataset):
    def __init__(
        self,
        manifest_csv: Path,
        split: str,
        catsa_root: Path | None = None,
        wesad_root: Path | None = None,
        include_sessions: Optional[Sequence[str]] = None,
        target_rates: Optional[Dict[str, int]] = None,
        cache_subjects: int = 8,
        wavelet: str = "db4",
        wavelet_level: int = 4,
        wavelet_bands: Sequence[str] = DEFAULT_WAVELET_BANDS,
        baseline_reference: bool = False,
        privileged_modalities: Sequence[str] = DEFAULT_PRIVILEGED_MODALITIES,
    ) -> None:
        self.manifest_csv = Path(manifest_csv)
        self.catsa_root = Path(catsa_root if catsa_root is not None else wesad_root)
        self.manifest = pd.read_csv(self.manifest_csv)
        self.manifest = self.manifest[self.manifest["split"] == split].reset_index(drop=True)
        if include_sessions is not None:
            include_sessions = {str(name) for name in include_sessions}
            self.manifest = self.manifest[self.manifest["session"].isin(include_sessions)].reset_index(drop=True)

        self.target_rates = {**DEFAULT_CATSA_TARGET_RATES, **(target_rates or {})}
        self.cache_subjects = max(int(cache_subjects), 0)
        self.wavelet = wavelet
        self.wavelet_level = wavelet_level
        self.wavelet_bands = tuple(wavelet_bands)
        self.baseline_reference = baseline_reference
        self.privileged_modalities = tuple(str(item) for item in privileged_modalities)

        self._watch_cache: OrderedDict[Path, np.ndarray] = OrderedDict()
        self._privileged_cache: OrderedDict[Path, np.ndarray] = OrderedDict()
        self._baseline_watch_reference: Dict[str, torch.Tensor] = {}
        self._baseline_wavelet_reference: Dict[str, torch.Tensor] = {}
        self._baseline_quality_reference: Dict[str, float] = {}
        if self.baseline_reference:
            self._build_baseline_reference_cache()

    def __len__(self) -> int:
        return len(self.manifest)

    def _resolve_session_dir(self, rel_path: str) -> Path:
        rel = PureWindowsPath(str(rel_path))
        parts = list(rel.parts)
        if parts and parts[0].lower() == self.catsa_root.name.lower():
            parts = parts[1:]
        return self.catsa_root.joinpath(*parts)

    @staticmethod
    def _touch_cache(cache: OrderedDict[Path, np.ndarray], key: Path, value: np.ndarray, max_items: int) -> None:
        if max_items <= 0:
            return
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > max_items:
            cache.popitem(last=False)

    @staticmethod
    def _slice_processed_stream(stream: np.ndarray, sampling_rate: float, start_s: float, end_s: float) -> np.ndarray:
        start_idx = max(int(round(start_s * sampling_rate)), 0)
        end_idx = min(int(round(end_s * sampling_rate)), len(stream))
        return np.asarray(stream[start_idx:end_idx], dtype=np.float32)

    def _get_processed_watch_stream(self, rel_path: str) -> np.ndarray:
        session_dir = self._resolve_session_dir(rel_path)
        cached = self._watch_cache.get(session_dir)
        if cached is not None:
            self._watch_cache.move_to_end(session_dir)
            return cached

        target_sr = self.target_rates["watch"]
        bvp = _read_scalar_csv(session_dir / "BVP.csv", "BVP")
        acc = _read_acc_csv(session_dir / "ACC.csv")

        bvp = _preprocess_catsa_bvp(bvp, sampling_rate=CATSA_SAMPLE_RATES["BVP"])
        acc = _preprocess_catsa_acc(acc)

        bvp = resample_array(bvp, orig_sr=CATSA_SAMPLE_RATES["BVP"], target_sr=target_sr)
        acc = resample_array(acc, orig_sr=CATSA_SAMPLE_RATES["ACC"], target_sr=target_sr)

        min_len = min(len(bvp), len(acc))
        stream = np.concatenate([bvp[:min_len, None], acc[:min_len]], axis=1).astype(np.float32)
        stream = np.nan_to_num(stream, nan=0.0, posinf=0.0, neginf=0.0)
        self._touch_cache(self._watch_cache, session_dir, stream, self.cache_subjects)
        return stream

    def _get_processed_privileged_stream(self, rel_path: str) -> np.ndarray:
        session_dir = self._resolve_session_dir(rel_path)
        cached = self._privileged_cache.get(session_dir)
        if cached is not None:
            self._privileged_cache.move_to_end(session_dir)
            return cached

        target_sr = self.target_rates["privileged"]
        pieces: list[np.ndarray] = []
        for modality in self.privileged_modalities:
            if modality not in CATSA_SAMPLE_RATES:
                raise ValueError(f"Unsupported CATSA privileged modality: {modality}")
            stream = _read_scalar_csv(session_dir / f"{modality}.csv", modality)
            stream = _preprocess_catsa_scalar(stream)
            stream = resample_array(stream, orig_sr=CATSA_SAMPLE_RATES[modality], target_sr=target_sr)
            stream = np.nan_to_num(stream, nan=0.0, posinf=0.0, neginf=0.0)
            pieces.append(stream[:, None])

        if not pieces:
            raise ValueError("At least one privileged modality is required for CATSA privileged training.")
        min_len = min(len(piece) for piece in pieces)
        stream = np.concatenate([piece[:min_len] for piece in pieces], axis=1).astype(np.float32)
        self._touch_cache(self._privileged_cache, session_dir, stream, self.cache_subjects)
        return stream

    def _load_watch(self, row: pd.Series, start_s: float, end_s: float, duration_s: float) -> tuple[torch.Tensor, float]:
        target_sr = self.target_rates["watch"]
        target_len = int(round(duration_s * target_sr))
        watch_stream = self._get_processed_watch_stream(row["subject_session_path"])
        watch = self._slice_processed_stream(watch_stream, target_sr, start_s, end_s)
        if len(watch) == 0:
            raise ValueError(
                f"Empty CATSA watch window for {row['subject_id']} {row['session']} {start_s:.3f}-{end_s:.3f}s"
            )
        watch = ensure_length(watch, target_len)
        watch_quality = 1.0
        return torch.from_numpy(watch.T.copy()), watch_quality

    def _load_privileged(self, row: pd.Series, start_s: float, end_s: float, duration_s: float) -> torch.Tensor:
        target_sr = self.target_rates["privileged"]
        target_len = int(round(duration_s * target_sr))
        privileged_stream = self._get_processed_privileged_stream(row["subject_session_path"])
        privileged = self._slice_processed_stream(privileged_stream, target_sr, start_s, end_s)
        if len(privileged) == 0:
            raise ValueError(
                f"Empty CATSA privileged window for {row['subject_id']} {row['session']} {start_s:.3f}-{end_s:.3f}s"
            )
        privileged = ensure_length(privileged, target_len)
        return torch.from_numpy(privileged.T.copy())

    def _build_baseline_reference_cache(self) -> None:
        baseline_rows = self.manifest[self.manifest["session"] == "Baseline"]
        if baseline_rows.empty:
            return

        for subject_id, group in baseline_rows.groupby("subject_id", sort=False):
            watch_list: list[torch.Tensor] = []
            wavelet_list: list[np.ndarray] = []
            quality_list: list[float] = []
            for _, row in group.iterrows():
                start_s = float(row["window_start_s"])
                end_s = float(row["window_end_s"])
                duration_s = float(row["window_duration_s"])
                watch, watch_quality = self._load_watch(row, start_s, end_s, duration_s)
                wavelet_features = compute_wavelet_band_ratios(
                    watch[0].detach().cpu().numpy(),
                    wavelet=self.wavelet,
                    level=self.wavelet_level,
                    selected_bands=self.wavelet_bands,
                )
                watch_list.append(watch.float())
                wavelet_list.append(wavelet_features.astype(np.float32))
                quality_list.append(float(watch_quality))

            if watch_list:
                self._baseline_watch_reference[str(subject_id)] = torch.stack(watch_list, dim=0).mean(dim=0)
                self._baseline_wavelet_reference[str(subject_id)] = torch.from_numpy(
                    np.stack(wavelet_list, axis=0).mean(axis=0).astype(np.float32)
                )
                self._baseline_quality_reference[str(subject_id)] = float(np.mean(quality_list))

    def _get_baseline_reference(
        self,
        subject_id: str,
        current_watch: torch.Tensor,
        current_wavelet: np.ndarray,
        current_quality: float,
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        subject_key = str(subject_id)
        baseline_watch = self._baseline_watch_reference.get(subject_key, current_watch.float())
        baseline_wavelet = self._baseline_wavelet_reference.get(
            subject_key,
            torch.from_numpy(current_wavelet.astype(np.float32)),
        )
        baseline_quality = self._baseline_quality_reference.get(subject_key, float(current_quality))
        return baseline_watch.clone(), baseline_wavelet.clone(), float(baseline_quality)


class CATSAWatchWindowDataset(CATSABaseWindowDataset):
    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.manifest.iloc[index]
        start_s = float(row["window_start_s"])
        end_s = float(row["window_end_s"])
        duration_s = float(row["window_duration_s"])

        watch, watch_quality = self._load_watch(row, start_s, end_s, duration_s)
        wavelet_features = compute_wavelet_band_ratios(
            watch[0].detach().cpu().numpy(),
            wavelet=self.wavelet,
            level=self.wavelet_level,
            selected_bands=self.wavelet_bands,
        )
        baseline_pack = None
        if self.baseline_reference:
            baseline_pack = self._get_baseline_reference(row["subject_id"], watch, wavelet_features, watch_quality)

        return {
            "signal": watch,
            "wavelet_features": torch.from_numpy(wavelet_features.copy()),
            "watch_quality": torch.tensor([watch_quality], dtype=torch.float32),
            "label": int(row["label"]),
            "subject_id": row["subject_id"],
            "subject_index": int(row["subject_index"]),
            "session": row["session"],
            "group_name": row["group_name"],
            "window_start_ms": int(row["window_start_ms"]),
            "window_end_ms": int(row["window_end_ms"]),
            "subject_session_path": row["subject_session_path"],
            **(
                {
                    "baseline_signal": baseline_pack[0],
                    "baseline_wavelet_features": baseline_pack[1],
                    "baseline_watch_quality": torch.tensor([baseline_pack[2]], dtype=torch.float32),
                }
                if self.baseline_reference
                else {}
            ),
        }


class CATSAPrivilegedWindowDataset(CATSABaseWindowDataset):
    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.manifest.iloc[index]
        start_s = float(row["window_start_s"])
        end_s = float(row["window_end_s"])
        duration_s = float(row["window_duration_s"])

        watch, watch_quality = self._load_watch(row, start_s, end_s, duration_s)
        privileged = self._load_privileged(row, start_s, end_s, duration_s)
        wavelet_features = compute_wavelet_band_ratios(
            watch[0].detach().cpu().numpy(),
            wavelet=self.wavelet,
            level=self.wavelet_level,
            selected_bands=self.wavelet_bands,
        )
        baseline_pack = None
        if self.baseline_reference:
            baseline_pack = self._get_baseline_reference(row["subject_id"], watch, wavelet_features, watch_quality)

        return {
            "watch_signal": watch,
            "privileged_signal": privileged,
            "wavelet_features": torch.from_numpy(wavelet_features.copy()),
            "watch_quality": torch.tensor([watch_quality], dtype=torch.float32),
            "label": int(row["label"]),
            "subject_id": row["subject_id"],
            "subject_index": int(row["subject_index"]),
            "session": row["session"],
            "group_name": row["group_name"],
            "window_start_ms": int(row["window_start_ms"]),
            "window_end_ms": int(row["window_end_ms"]),
            "subject_session_path": row["subject_session_path"],
            **(
                {
                    "baseline_watch_signal": baseline_pack[0],
                    "baseline_wavelet_features": baseline_pack[1],
                    "baseline_watch_quality": torch.tensor([baseline_pack[2]], dtype=torch.float32),
                }
                if self.baseline_reference
                else {}
            ),
        }
