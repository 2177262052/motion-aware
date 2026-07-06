from __future__ import annotations

from collections import OrderedDict
from pathlib import Path, PureWindowsPath
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd
import pickle
import torch
from scipy import signal
from torch.utils.data import Dataset

from .dataset import preprocess_acc, robust_zscore
from .galaxy_dataset import (
    DEFAULT_WAVELET_BANDS,
    compute_wavelet_band_ratios,
    ensure_length,
    preprocess_e4_bvp,
    preprocess_ecg,
    resample_array,
)


WESAD_CHEST_SAMPLE_RATE = 700
WESAD_WRIST_SAMPLE_RATES = {
    "ACC": 32,
    "BVP": 64,
    "EDA": 4,
    "TEMP": 4,
}

DEFAULT_WESAD_TARGET_RATES = {
    "watch": 32,
    "privileged": 64,
}

DEFAULT_CHEST_MODALITIES = ("ACC", "ECG", "EMG", "EDA", "Temp", "Resp")


def _safe_bandpass(values: np.ndarray, sampling_rate: float, low_hz: float, high_hz: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if len(values) < max(int(sampling_rate), 16):
        return np.nan_to_num(robust_zscore(values), nan=0.0, posinf=0.0, neginf=0.0)

    nyquist = sampling_rate / 2.0
    low = max(low_hz / nyquist, 1e-4)
    high = min(high_hz / nyquist, 0.99)
    if low >= high:
        return np.nan_to_num(robust_zscore(values), nan=0.0, posinf=0.0, neginf=0.0)

    b, a = signal.butter(3, [low, high], btype="bandpass")
    filtered = signal.filtfilt(b, a, values).astype(np.float32)
    normalized = robust_zscore(filtered)
    if not np.isfinite(normalized).all():
        normalized = robust_zscore(values)
    return np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)


def preprocess_resp(values: np.ndarray, sampling_rate: float) -> np.ndarray:
    return _safe_bandpass(values, sampling_rate=sampling_rate, low_hz=0.05, high_hz=1.0)


def preprocess_emg(values: np.ndarray, sampling_rate: float) -> np.ndarray:
    return _safe_bandpass(values, sampling_rate=sampling_rate, low_hz=10.0, high_hz=45.0)


def preprocess_scalar(values: np.ndarray) -> np.ndarray:
    normalized = robust_zscore(np.asarray(values, dtype=np.float32))
    return np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)


class WESADBaseWindowDataset(Dataset):
    def __init__(
        self,
        manifest_csv: Path,
        split: str,
        wesad_root: Path,
        include_sessions: Optional[Sequence[str]] = None,
        target_rates: Optional[Dict[str, int]] = None,
        cache_subjects: int = 2,
        wavelet: str = "db4",
        wavelet_level: int = 4,
        wavelet_bands: Sequence[str] = DEFAULT_WAVELET_BANDS,
        baseline_reference: bool = False,
    ) -> None:
        self.manifest_csv = Path(manifest_csv)
        self.wesad_root = Path(wesad_root)
        self.manifest = pd.read_csv(self.manifest_csv)
        self.manifest = self.manifest[self.manifest["split"] == split].reset_index(drop=True)
        if include_sessions is not None:
            include_sessions = {str(name) for name in include_sessions}
            self.manifest = self.manifest[self.manifest["session"].isin(include_sessions)].reset_index(drop=True)

        self.target_rates = {**DEFAULT_WESAD_TARGET_RATES, **(target_rates or {})}
        self.cache_subjects = max(int(cache_subjects), 0)
        self.wavelet = wavelet
        self.wavelet_level = wavelet_level
        self.wavelet_bands = tuple(wavelet_bands)
        self.baseline_reference = baseline_reference

        self._watch_cache: OrderedDict[Path, np.ndarray] = OrderedDict()
        self._privileged_cache: OrderedDict[Path, np.ndarray] = OrderedDict()
        self._baseline_watch_reference: Dict[str, torch.Tensor] = {}
        self._baseline_wavelet_reference: Dict[str, torch.Tensor] = {}
        self._baseline_quality_reference: Dict[str, float] = {}
        if self.baseline_reference:
            self._build_baseline_reference_cache()

    def __len__(self) -> int:
        return len(self.manifest)

    def _resolve_subject_pkl(self, rel_path: str) -> Path:
        rel = PureWindowsPath(str(rel_path))
        parts = list(rel.parts)
        if parts and parts[0].lower() == self.wesad_root.name.lower():
            parts = parts[1:]
        return self.wesad_root.joinpath(*parts)

    @staticmethod
    def _touch_cache(cache: OrderedDict[Path, np.ndarray], key: Path, value: np.ndarray, max_items: int) -> None:
        if max_items <= 0:
            return
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > max_items:
            cache.popitem(last=False)

    def _load_subject_payload(self, pkl_path: Path) -> dict[str, Any]:
        with pkl_path.open("rb") as handle:
            return pickle.load(handle, encoding="latin1")

    @staticmethod
    def _slice_array(values: np.ndarray, sampling_rate: float, start_s: float, end_s: float) -> np.ndarray:
        start_idx = max(int(round(start_s * sampling_rate)), 0)
        end_idx = min(int(round(end_s * sampling_rate)), len(values))
        return np.asarray(values[start_idx:end_idx], dtype=np.float32)

    @staticmethod
    def _slice_processed_stream(stream: np.ndarray, sampling_rate: float, start_s: float, end_s: float) -> np.ndarray:
        start_idx = max(int(round(start_s * sampling_rate)), 0)
        end_idx = min(int(round(end_s * sampling_rate)), len(stream))
        return np.asarray(stream[start_idx:end_idx], dtype=np.float32)

    def _get_processed_watch_stream(self, rel_path: str) -> np.ndarray:
        pkl_path = self._resolve_subject_pkl(rel_path)
        cached = self._watch_cache.get(pkl_path)
        if cached is not None:
            self._watch_cache.move_to_end(pkl_path)
            return cached

        payload = self._load_subject_payload(pkl_path)
        wrist = payload["signal"]["wrist"]
        target_sr = self.target_rates["watch"]

        bvp = np.asarray(wrist["BVP"], dtype=np.float32).reshape(-1)
        acc = np.asarray(wrist["ACC"], dtype=np.float32)

        bvp = preprocess_e4_bvp(bvp, sampling_rate=WESAD_WRIST_SAMPLE_RATES["BVP"])
        acc = preprocess_acc(acc)

        bvp = resample_array(bvp, orig_sr=WESAD_WRIST_SAMPLE_RATES["BVP"], target_sr=target_sr)
        acc = resample_array(acc, orig_sr=WESAD_WRIST_SAMPLE_RATES["ACC"], target_sr=target_sr)

        min_len = min(len(bvp), len(acc))
        stream = np.concatenate([bvp[:min_len, None], acc[:min_len]], axis=1).astype(np.float32)
        self._touch_cache(self._watch_cache, pkl_path, stream, self.cache_subjects)
        return stream

    def _get_processed_privileged_stream(self, rel_path: str) -> np.ndarray:
        pkl_path = self._resolve_subject_pkl(rel_path)
        cached = self._privileged_cache.get(pkl_path)
        if cached is not None:
            self._privileged_cache.move_to_end(pkl_path)
            return cached

        payload = self._load_subject_payload(pkl_path)
        chest = payload["signal"]["chest"]
        target_sr = self.target_rates["privileged"]

        pieces: list[np.ndarray] = []

        acc = np.asarray(chest["ACC"], dtype=np.float32)
        acc = preprocess_acc(acc)
        acc = resample_array(acc, orig_sr=WESAD_CHEST_SAMPLE_RATE, target_sr=target_sr)
        pieces.append(acc)

        ecg = preprocess_ecg(np.asarray(chest["ECG"], dtype=np.float32).reshape(-1), sampling_rate=WESAD_CHEST_SAMPLE_RATE)
        emg = preprocess_emg(np.asarray(chest["EMG"], dtype=np.float32).reshape(-1), sampling_rate=WESAD_CHEST_SAMPLE_RATE)
        eda = preprocess_scalar(np.asarray(chest["EDA"], dtype=np.float32).reshape(-1))
        temp = preprocess_scalar(np.asarray(chest["Temp"], dtype=np.float32).reshape(-1))
        resp = preprocess_resp(np.asarray(chest["Resp"], dtype=np.float32).reshape(-1), sampling_rate=WESAD_CHEST_SAMPLE_RATE)

        scalar_streams = [ecg, emg, eda, temp, resp]
        for stream in scalar_streams:
            stream = resample_array(stream, orig_sr=WESAD_CHEST_SAMPLE_RATE, target_sr=target_sr)
            stream = np.nan_to_num(stream, nan=0.0, posinf=0.0, neginf=0.0)
            pieces.append(stream[:, None])

        min_len = min(len(piece) for piece in pieces)
        stream = np.concatenate([piece[:min_len] for piece in pieces], axis=1).astype(np.float32)
        self._touch_cache(self._privileged_cache, pkl_path, stream, self.cache_subjects)
        return stream

    def _load_watch(
        self,
        row: pd.Series,
        start_s: float,
        end_s: float,
        duration_s: float,
    ) -> tuple[torch.Tensor, float]:
        target_sr = self.target_rates["watch"]
        target_len = int(round(duration_s * target_sr))
        watch_stream = self._get_processed_watch_stream(row["subject_pkl_path"])
        watch = self._slice_processed_stream(watch_stream, target_sr, start_s, end_s)
        if len(watch) == 0:
            raise ValueError(
                f"Empty WESAD wrist window for {row['subject_id']} {row['session']} {start_s:.3f}-{end_s:.3f}s"
            )
        watch = ensure_length(watch, target_len)
        watch_quality = 1.0
        return torch.from_numpy(watch.T.copy()), watch_quality

    def _load_privileged(
        self,
        row: pd.Series,
        start_s: float,
        end_s: float,
        duration_s: float,
    ) -> torch.Tensor:
        target_sr = self.target_rates["privileged"]
        target_len = int(round(duration_s * target_sr))
        privileged_stream = self._get_processed_privileged_stream(row["subject_pkl_path"])
        privileged = self._slice_processed_stream(privileged_stream, target_sr, start_s, end_s)
        if len(privileged) == 0:
            raise ValueError(
                f"Empty WESAD chest window for {row['subject_id']} {row['session']} {start_s:.3f}-{end_s:.3f}s"
            )
        privileged = ensure_length(privileged, target_len)
        return torch.from_numpy(privileged.T.copy())

    def _build_baseline_reference_cache(self) -> None:
        baseline_rows = self.manifest[self.manifest["session"] == "baseline"]
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


class WESADWatchWindowDataset(WESADBaseWindowDataset):
    def __getitem__(self, index: int) -> Dict[str, object]:
        row = self.manifest.iloc[index]
        start_s = float(row["window_start_s"])
        end_s = float(row["window_end_s"])
        duration_s = float(row["window_duration_s"])
        start_ms = int(row["window_start_ms"])
        end_ms = int(row["window_end_ms"])

        watch, watch_quality = self._load_watch(row, start_s, end_s, duration_s)
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
            "subject_pkl_path": row["subject_pkl_path"],
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


class WESADPrivilegedWindowDataset(WESADBaseWindowDataset):
    def __getitem__(self, index: int) -> Dict[str, object]:
        row = self.manifest.iloc[index]
        start_s = float(row["window_start_s"])
        end_s = float(row["window_end_s"])
        duration_s = float(row["window_duration_s"])
        start_ms = int(row["window_start_ms"])
        end_ms = int(row["window_end_ms"])

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
            baseline_pack = self._get_baseline_reference(
                row["subject_id"],
                watch,
                wavelet_features,
                watch_quality,
            )

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
            "window_start_ms": start_ms,
            "window_end_ms": end_ms,
            "subject_pkl_path": row["subject_pkl_path"],
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
