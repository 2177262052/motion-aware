from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
import pywt
import torch
from scipy import signal
from torch.utils.data import Dataset

from .dataset import preprocess_acc, robust_zscore


POLAR_TIME_OFFSET_MS = 9 * 3600 * 1000

DEFAULT_TARGET_RATES = {
    "gw_ppg": 25,
    "gw_acc": 25,
    "e4_bvp": 64,
    "e4_acc": 32,
    "polar_ecg": 130,
}

DEFAULT_WAVELET_BANDS = ("A4", "D4", "D2", "D1")


def _safe_bandpass(values: np.ndarray, sampling_rate: float, low_hz: float, high_hz: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if len(values) < max(int(sampling_rate), 16):
        return robust_zscore(values)

    nyquist = sampling_rate / 2.0
    low = max(low_hz / nyquist, 1e-4)
    high = min(high_hz / nyquist, 0.99)
    if low >= high:
        return robust_zscore(values)

    b, a = signal.butter(3, [low, high], btype="bandpass")
    filtered = signal.filtfilt(b, a, values).astype(np.float32)
    return robust_zscore(filtered)


def preprocess_gw_ppg(values: np.ndarray, sampling_rate: float) -> np.ndarray:
    return _safe_bandpass(values, sampling_rate=sampling_rate, low_hz=0.5, high_hz=8.0)


def preprocess_e4_bvp(values: np.ndarray, sampling_rate: float) -> np.ndarray:
    return _safe_bandpass(values, sampling_rate=sampling_rate, low_hz=0.5, high_hz=8.0)


def preprocess_ecg(values: np.ndarray, sampling_rate: float) -> np.ndarray:
    return _safe_bandpass(values, sampling_rate=sampling_rate, low_hz=0.5, high_hz=40.0)


def resample_array(values: np.ndarray, orig_sr: float, target_sr: float) -> np.ndarray:
    if math.isclose(orig_sr, target_sr):
        return values.astype(np.float32)
    target_len = int(round(len(values) * target_sr / orig_sr))
    if values.ndim == 1:
        return signal.resample(values, target_len).astype(np.float32)
    channels = [signal.resample(values[:, idx], target_len).astype(np.float32) for idx in range(values.shape[1])]
    return np.stack(channels, axis=1)


def ensure_length(values: np.ndarray, target_len: int) -> np.ndarray:
    if len(values) == target_len:
        return values.astype(np.float32)
    if len(values) > target_len:
        return values[:target_len].astype(np.float32)
    if values.ndim == 1:
        out = np.zeros((target_len,), dtype=np.float32)
        out[: len(values)] = values
        return out
    out = np.zeros((target_len, values.shape[1]), dtype=np.float32)
    out[: len(values)] = values
    return out


def compute_wavelet_band_ratios(
    signal_1d: np.ndarray,
    wavelet: str = "db4",
    level: int = 4,
    selected_bands: Sequence[str] = DEFAULT_WAVELET_BANDS,
) -> np.ndarray:
    coeffs = pywt.wavedec(np.asarray(signal_1d, dtype=np.float32), wavelet=wavelet, level=level)
    names = [f"A{level}"] + [f"D{i}" for i in range(level, 0, -1)]
    energy_map = {
        name: float(np.sum(np.square(coeff)))
        for name, coeff in zip(names, coeffs)
    }
    total = float(sum(energy_map.values())) + 1e-12
    return np.asarray([energy_map[name] / total for name in selected_bands], dtype=np.float32)


class GalaxyPPGWindowDataset(Dataset):
    def __init__(
        self,
        manifest_csv: Path,
        split: str,
        dataset_root: Path,
        target_rates: Optional[Dict[str, int]] = None,
        cache_tables: bool = True,
    ) -> None:
        self.manifest_csv = Path(manifest_csv)
        self.dataset_root = Path(dataset_root)
        self.manifest = pd.read_csv(self.manifest_csv)
        self.manifest = self.manifest[self.manifest["split"] == split].reset_index(drop=True)
        self.target_rates = {**DEFAULT_TARGET_RATES, **(target_rates or {})}
        self.cache_tables = cache_tables
        self._table_cache: Dict[Path, pd.DataFrame] = {}
        self._baseline_watch_reference: Dict[str, torch.Tensor] = {}
        self._baseline_wavelet_reference: Dict[str, torch.Tensor] = {}
        self._baseline_quality_reference: Dict[str, float] = {}

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> Dict[str, object]:
        row = self.manifest.iloc[index]
        start_ms = int(row["window_start_ms"])
        end_ms = int(row["window_end_ms"])
        duration_s = float(row["window_duration_s"])

        watch, watch_quality = self._load_watch(row, start_ms, end_ms, duration_s)
        e4 = self._load_e4(row, start_ms, end_ms, duration_s)
        polar, polar_coverage = self._load_polar(row, start_ms, end_ms, duration_s)

        return {
            "watch_signal": watch,
            "e4_signal": e4,
            "polar_signal": polar,
            "watch_quality": watch_quality,
            "polar_coverage": polar_coverage,
            "label": int(row["label"]),
            "subject_id": row["subject_id"],
            "subject_index": int(row["subject_index"]),
            "session": row["session"],
            "group_name": row["group_name"],
            "window_start_ms": start_ms,
            "window_end_ms": end_ms,
        }

    def _read_csv(self, rel_path: str) -> pd.DataFrame:
        path = self.dataset_root / Path(rel_path)
        if self.cache_tables and path in self._table_cache:
            return self._table_cache[path]
        table = pd.read_csv(path)
        if self.cache_tables:
            self._table_cache[path] = table
        return table

    def _slice_window(
        self,
        table: pd.DataFrame,
        time_col: str,
        start_ms: int,
        end_ms: int,
        time_divisor: float = 1.0,
        time_offset_ms: int = 0,
    ) -> pd.DataFrame:
        ts_ms = table[time_col].to_numpy(dtype=np.float64) / time_divisor - time_offset_ms
        mask = (ts_ms >= start_ms) & (ts_ms < end_ms)
        return table.loc[mask].reset_index(drop=True)

    def _load_watch(self, row: pd.Series, start_ms: int, end_ms: int, duration_s: float) -> tuple[torch.Tensor, float]:
        ppg_df = self._read_csv(row["gw_ppg_path"])
        acc_df = self._read_csv(row["gw_acc_path"])

        ppg_window = self._slice_window(ppg_df, "timestamp", start_ms, end_ms)
        acc_window = self._slice_window(acc_df, "timestamp", start_ms, end_ms)
        if ppg_window.empty or acc_window.empty:
            raise ValueError(f"Empty Galaxy Watch window for {row['subject_id']} {row['session']} {start_ms}-{end_ms}")

        ppg = preprocess_gw_ppg(ppg_window["ppg"].to_numpy(dtype=np.float32), sampling_rate=self.target_rates["gw_ppg"])
        acc = preprocess_acc(acc_window[["x", "y", "z"]].to_numpy(dtype=np.float32))

        ppg = resample_array(ppg, orig_sr=25, target_sr=self.target_rates["gw_ppg"])
        acc = resample_array(acc, orig_sr=25, target_sr=self.target_rates["gw_acc"])

        ppg = ensure_length(ppg, int(round(duration_s * self.target_rates["gw_ppg"])))
        acc = ensure_length(acc, int(round(duration_s * self.target_rates["gw_acc"])))

        min_len = min(len(ppg), len(acc))
        ppg = ppg[:min_len, None]
        acc = acc[:min_len]
        watch = np.concatenate([ppg, acc], axis=1).astype(np.float32)

        status = ppg_window["status"].to_numpy(dtype=np.int32)
        watch_quality = float(np.mean((status == 0) | (status == 500)))
        return torch.from_numpy(watch.T.copy()), watch_quality

    def _load_e4(self, row: pd.Series, start_ms: int, end_ms: int, duration_s: float) -> torch.Tensor:
        bvp_df = self._read_csv(row["e4_bvp_path"])
        acc_df = self._read_csv(row["e4_acc_path"])

        bvp_window = self._slice_window(bvp_df, "timestamp", start_ms, end_ms, time_divisor=1000.0)
        acc_window = self._slice_window(acc_df, "timestamp", start_ms, end_ms, time_divisor=1000.0)
        if bvp_window.empty or acc_window.empty:
            raise ValueError(f"Empty E4 window for {row['subject_id']} {row['session']} {start_ms}-{end_ms}")

        bvp = preprocess_e4_bvp(bvp_window["value"].to_numpy(dtype=np.float32), sampling_rate=64)
        acc = preprocess_acc(acc_window[["x", "y", "z"]].to_numpy(dtype=np.float32))

        bvp = resample_array(bvp, orig_sr=64, target_sr=self.target_rates["e4_bvp"])
        acc = resample_array(acc, orig_sr=32, target_sr=self.target_rates["e4_acc"])

        bvp = ensure_length(bvp, int(round(duration_s * self.target_rates["e4_bvp"])))
        acc = ensure_length(acc, int(round(duration_s * self.target_rates["e4_acc"])))

        # Align to the shorter sequence after modality-specific resampling.
        min_len = min(len(bvp), len(acc))
        bvp = bvp[:min_len, None]
        acc = acc[:min_len]
        e4 = np.concatenate([bvp, acc], axis=1).astype(np.float32)
        return torch.from_numpy(e4.T.copy())

    def _load_polar(self, row: pd.Series, start_ms: int, end_ms: int, duration_s: float) -> tuple[torch.Tensor, float]:
        ecg_df = self._read_csv(row["polar_ecg_path"])
        ecg_window = self._slice_window(
            ecg_df,
            "phoneTimestamp",
            start_ms,
            end_ms,
            time_divisor=1.0,
            time_offset_ms=POLAR_TIME_OFFSET_MS,
        )
        if ecg_window.empty:
            raise ValueError(f"Empty Polar window for {row['subject_id']} {row['session']} {start_ms}-{end_ms}")

        ecg = preprocess_ecg(ecg_window["ecg"].to_numpy(dtype=np.float32), sampling_rate=130)
        ecg = resample_array(ecg, orig_sr=130, target_sr=self.target_rates["polar_ecg"])
        target_len = int(round(duration_s * self.target_rates["polar_ecg"]))
        coverage = float(min(len(ecg), target_len) / max(target_len, 1))
        ecg = ensure_length(ecg, target_len)
        return torch.from_numpy(ecg[None, :].copy()), coverage

    def _build_baseline_reference_cache(self) -> None:
        if "session" not in self.manifest.columns:
            return
        baseline_rows = self.manifest[self.manifest["session"] == "baseline"]
        if baseline_rows.empty:
            return

        for subject_id, group in baseline_rows.groupby("subject_id", sort=False):
            watch_list: list[torch.Tensor] = []
            wavelet_list: list[np.ndarray] = []
            quality_list: list[float] = []
            for _, row in group.iterrows():
                start_ms = int(row["window_start_ms"])
                end_ms = int(row["window_end_ms"])
                duration_s = float(row["window_duration_s"])
                watch, watch_quality = self._load_watch(row, start_ms, end_ms, duration_s)
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


class GalaxyWatchWindowDataset(GalaxyPPGWindowDataset):
    def __init__(
        self,
        manifest_csv: Path,
        split: str,
        dataset_root: Path,
        include_sessions: Optional[Sequence[str]] = None,
        target_rates: Optional[Dict[str, int]] = None,
        cache_tables: bool = True,
        wavelet: str = "db4",
        wavelet_level: int = 4,
        wavelet_bands: Sequence[str] = DEFAULT_WAVELET_BANDS,
        baseline_reference: bool = False,
    ) -> None:
        super().__init__(
            manifest_csv=manifest_csv,
            split=split,
            dataset_root=dataset_root,
            target_rates=target_rates,
            cache_tables=cache_tables,
        )
        if include_sessions is not None:
            include_sessions = {str(name) for name in include_sessions}
            self.manifest = self.manifest[self.manifest["session"].isin(include_sessions)].reset_index(drop=True)
        self.wavelet = wavelet
        self.wavelet_level = wavelet_level
        self.wavelet_bands = tuple(wavelet_bands)
        self.baseline_reference = baseline_reference
        if self.baseline_reference:
            self._build_baseline_reference_cache()

    def __getitem__(self, index: int) -> Dict[str, object]:
        row = self.manifest.iloc[index]
        start_ms = int(row["window_start_ms"])
        end_ms = int(row["window_end_ms"])
        duration_s = float(row["window_duration_s"])

        watch, watch_quality = self._load_watch(row, start_ms, end_ms, duration_s)
        wavelet_features = compute_wavelet_band_ratios(
            watch[0].detach().cpu().numpy(),
            wavelet=self.wavelet,
            level=self.wavelet_level,
            selected_bands=self.wavelet_bands,
        )
        baseline_pack = None
        if self.baseline_reference:
            baseline_pack = self._get_baseline_reference(
                row["subject_id"],
                watch,
                wavelet_features,
                watch_quality,
            )
        return {
            "signal": watch,
            "wavelet_features": torch.from_numpy(wavelet_features.copy()),
            "watch_quality": torch.tensor([watch_quality], dtype=torch.float32),
            "label": int(row["label"]),
            "subject_id": row["subject_id"],
            "subject_index": int(row["subject_index"]),
            "session": row["session"],
            "group_name": row["group_name"],
            "window_start_ms": start_ms,
            "window_end_ms": end_ms,
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


class GalaxyPrivilegedWindowDataset(GalaxyPPGWindowDataset):
    def __init__(
        self,
        manifest_csv: Path,
        split: str,
        dataset_root: Path,
        include_sessions: Optional[Sequence[str]] = None,
        target_rates: Optional[Dict[str, int]] = None,
        cache_tables: bool = True,
        wavelet: str = "db4",
        wavelet_level: int = 4,
        wavelet_bands: Sequence[str] = DEFAULT_WAVELET_BANDS,
        baseline_reference: bool = False,
    ) -> None:
        super().__init__(
            manifest_csv=manifest_csv,
            split=split,
            dataset_root=dataset_root,
            target_rates=target_rates,
            cache_tables=cache_tables,
        )
        if include_sessions is not None:
            include_sessions = {str(name) for name in include_sessions}
            self.manifest = self.manifest[self.manifest["session"].isin(include_sessions)].reset_index(drop=True)
        self.wavelet = wavelet
        self.wavelet_level = wavelet_level
        self.wavelet_bands = tuple(wavelet_bands)
        self.baseline_reference = baseline_reference
        if self.baseline_reference:
            self._build_baseline_reference_cache()

    def __getitem__(self, index: int) -> Dict[str, object]:
        row = self.manifest.iloc[index]
        start_ms = int(row["window_start_ms"])
        end_ms = int(row["window_end_ms"])
        duration_s = float(row["window_duration_s"])

        watch, watch_quality = self._load_watch(row, start_ms, end_ms, duration_s)
        e4 = self._load_e4(row, start_ms, end_ms, duration_s)
        polar, polar_coverage = self._load_polar(row, start_ms, end_ms, duration_s)
        polar_targets, polar_mask = self._load_polar_targets(row, start_ms, end_ms)

        wavelet_features = compute_wavelet_band_ratios(
            watch[0].detach().cpu().numpy(),
            wavelet=self.wavelet,
            level=self.wavelet_level,
            selected_bands=self.wavelet_bands,
        )
        baseline_pack = None
        if self.baseline_reference:
            baseline_pack = self._get_baseline_reference(
                row["subject_id"],
                watch,
                wavelet_features,
                watch_quality,
            )
        return {
            "watch_signal": watch,
            "e4_signal": e4,
            "polar_signal": polar,
            "wavelet_features": torch.from_numpy(wavelet_features.copy()),
            "watch_quality": torch.tensor([watch_quality], dtype=torch.float32),
            "polar_coverage": torch.tensor([polar_coverage], dtype=torch.float32),
            "polar_targets": torch.from_numpy(polar_targets.copy()),
            "polar_target_mask": torch.from_numpy(polar_mask.copy()),
            "label": int(row["label"]),
            "subject_id": row["subject_id"],
            "subject_index": int(row["subject_index"]),
            "session": row["session"],
            "group_name": row["group_name"],
            "window_start_ms": start_ms,
            "window_end_ms": end_ms,
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

    def _load_polar_targets(self, row: pd.Series, start_ms: int, end_ms: int) -> tuple[np.ndarray, np.ndarray]:
        polar_ibi_path = self.dataset_root / Path(row["polar_ecg_path"]).parent / "IBI.csv"
        target_names = ("hr_bpm", "ibi_mean_ms", "ibi_std_ms")
        if not polar_ibi_path.exists():
            return np.zeros((len(target_names),), dtype=np.float32), np.zeros((len(target_names),), dtype=np.float32)

        ibi_df = self._read_csv(str(polar_ibi_path.relative_to(self.dataset_root)))
        ibi_window = self._slice_window(
            ibi_df,
            "phoneTimestamp",
            start_ms,
            end_ms,
            time_divisor=1.0,
            time_offset_ms=POLAR_TIME_OFFSET_MS,
        )
        if ibi_window.empty:
            return np.zeros((len(target_names),), dtype=np.float32), np.zeros((len(target_names),), dtype=np.float32)

        durations = ibi_window["duration"].to_numpy(dtype=np.float32)
        mean_ibi = float(np.mean(durations))
        std_ibi = float(np.std(durations))
        hr_bpm = 60000.0 / max(mean_ibi, 1e-3)

        targets = np.asarray([hr_bpm, mean_ibi, std_ibi], dtype=np.float32)
        mask = np.asarray([1.0, 1.0, 1.0], dtype=np.float32)
        return targets, mask
