from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from scipy import signal
from pandas.errors import EmptyDataError
from torch.utils.data import Dataset
from tqdm.auto import tqdm


DEFAULT_MODALITIES = ("BVP", "ACC")
DEFAULT_SAMPLING_RATES = {"BVP": 64, "ACC": 32}


def _safe_literal_eval(value: str):
    if isinstance(value, str) and value.startswith("[") and value.endswith("]"):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value
    return value


def parse_e4_signal(csv_path: Path, modality: str) -> Tuple[pd.Timestamp, float, np.ndarray]:
    raw = pd.read_csv(csv_path, header=None)
    start_ts = pd.to_datetime(raw.iloc[0, 0], utc=True)
    sampling_rate = float(raw.iloc[1, 0])
    values = raw.iloc[2:].astype(float).to_numpy()
    if modality == "ACC":
        return start_ts, sampling_rate, values
    return start_ts, sampling_rate, values[:, 0]


def parse_tags(csv_path: Path) -> List[pd.Timestamp]:
    try:
        series = pd.read_csv(csv_path, header=None).iloc[:, 0]
    except EmptyDataError:
        return []
    return [pd.to_datetime(x, utc=True) for x in series.tolist()]


def design_filter(sampling_rate: float) -> Tuple[np.ndarray, np.ndarray]:
    nyquist = sampling_rate / 2.0
    low = 0.5 / nyquist
    high = min(8.0 / nyquist, 0.99)
    b, a = signal.butter(3, [low, high], btype="bandpass")
    return b, a


def preprocess_bvp(values: np.ndarray, sampling_rate: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    b, a = design_filter(sampling_rate)
    filtered = signal.filtfilt(b, a, values).astype(np.float32)
    return robust_zscore(filtered)


def preprocess_acc(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 1:
        values = values[:, None]
    magnitude = np.linalg.norm(values, axis=1, keepdims=True)
    merged = np.concatenate([values, magnitude], axis=1)
    return robust_zscore(merged)


def robust_zscore(values: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    median = np.median(values, axis=0, keepdims=True)
    mad = np.median(np.abs(values - median), axis=0, keepdims=True)
    scale = 1.4826 * mad + eps
    return (values - median) / scale


def resample_array(values: np.ndarray, orig_sr: float, target_sr: float) -> np.ndarray:
    if math.isclose(orig_sr, target_sr):
        return values.astype(np.float32)
    target_len = int(round(len(values) * target_sr / orig_sr))
    if values.ndim == 1:
        return signal.resample(values, target_len).astype(np.float32)
    channels = [
        signal.resample(values[:, idx], target_len).astype(np.float32)
        for idx in range(values.shape[1])
    ]
    return np.stack(channels, axis=1)


def crop_by_seconds(values: np.ndarray, sampling_rate: float, start_s: float, end_s: float) -> np.ndarray:
    start_idx = max(int(round(start_s * sampling_rate)), 0)
    end_idx = min(int(round(end_s * sampling_rate)), len(values))
    return values[start_idx:end_idx]


def stack_modalities(modal_arrays: Dict[str, np.ndarray], target_sr: float) -> np.ndarray:
    pieces: List[np.ndarray] = []
    for modality, arr in modal_arrays.items():
        if modality == "BVP":
            arr = preprocess_bvp(arr, target_sr)[:, None]
        elif modality == "ACC":
            arr = preprocess_acc(arr)
        else:
            raise ValueError(f"Unsupported modality: {modality}")
        pieces.append(arr)
    return np.concatenate(pieces, axis=1).astype(np.float32)


def random_scaling(x: torch.Tensor, scale_std: float = 0.1) -> torch.Tensor:
    scale = torch.randn(x.shape[0], 1, device=x.device) * scale_std + 1.0
    return x * scale


def random_jitter(x: torch.Tensor, sigma: float = 0.02) -> torch.Tensor:
    return x + torch.randn_like(x) * sigma


def random_channel_dropout(x: torch.Tensor, p: float = 0.1) -> torch.Tensor:
    if p <= 0:
        return x
    keep = (torch.rand(x.shape[0], 1, device=x.device) > p).float()
    return x * keep


def augment_timeseries(x: torch.Tensor) -> torch.Tensor:
    x = random_scaling(x)
    x = random_jitter(x)
    x = random_channel_dropout(x)
    return x


class StressWindowDataset(Dataset):
    def __init__(
        self,
        manifest_csv: Path,
        split: str,
        modalities: Sequence[str] = DEFAULT_MODALITIES,
        target_sr: int = 64,
        ssl: bool = False,
        dataset_root: Optional[Path] = None,
        cache_mode: str = "none",
        cache_device: Optional[str] = None,
    ) -> None:
        self.manifest_csv = Path(manifest_csv)
        self.dataset_root = Path(dataset_root) if dataset_root is not None else None
        self.manifest = pd.read_csv(self.manifest_csv)
        self.manifest = self.manifest[self.manifest["split"] == split].reset_index(drop=True)
        self.modalities = tuple(modalities)
        self.target_sr = target_sr
        self.ssl = ssl
        self.cache_mode = cache_mode
        self.cache_device = cache_device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cached_signals: Optional[List[torch.Tensor]] = None
        self._cached_bytes = 0

        if self.cache_mode not in {"none", "ram", "gpu"}:
            raise ValueError(f"Unsupported cache_mode: {self.cache_mode}")

        if self.cache_mode != "none":
            self._build_cache()

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int):
        row = self.manifest.iloc[index]
        tensor = self.cached_signals[index] if self.cached_signals is not None else self._load_window(row)
        if self.ssl:
            return {
                "view1": augment_timeseries(tensor.clone()),
                "view2": augment_timeseries(tensor.clone()),
                "signal": tensor,
                "subject_index": int(row["subject_index"]),
            }
        return {
            "signal": tensor,
            "label": int(row["label"]),
            "subject_index": int(row["subject_index"]),
            "subject_id": row["subject_id"],
            "stage": row["stage"],
        }

    def _load_window(self, row: pd.Series) -> torch.Tensor:
        modal_arrays: Dict[str, np.ndarray] = {}
        for modality in self.modalities:
            path = self._resolve_signal_path(row[f"{modality.lower()}_path"])
            start_ts, sr, values = parse_e4_signal(path, modality)
            segment_start = float(row["segment_start_s"]) + float(row["window_start_s"])
            segment_end = segment_start + float(row["window_duration_s"])
            window = crop_by_seconds(values, sr, segment_start, segment_end)
            if len(window) == 0:
                raise ValueError(f"Empty window for {path} at row {row.name}")
            modal_arrays[modality] = resample_array(window, sr, self.target_sr)

        min_len = min(len(arr) for arr in modal_arrays.values())
        for modality in list(modal_arrays):
            modal_arrays[modality] = modal_arrays[modality][:min_len]
        stacked = stack_modalities(modal_arrays, self.target_sr)
        return torch.from_numpy(stacked.T.copy())

    def _build_cache(self) -> None:
        self.cached_signals = []
        total_bytes = 0
        desc = f"caching {self.cache_mode} {self.manifest_csv.name}:{len(self.manifest)}"
        for idx in tqdm(range(len(self.manifest)), desc=desc, leave=True):
            row = self.manifest.iloc[idx]
            tensor = self._load_window(row)
            if self.cache_mode == "gpu":
                tensor = tensor.to(self.cache_device)
            else:
                tensor = tensor.contiguous()
            total_bytes += tensor.numel() * tensor.element_size()
            self.cached_signals.append(tensor)
        self._cached_bytes = total_bytes
        print(
            f"dataset cache ready mode={self.cache_mode} "
            f"items={len(self.cached_signals)} size_mb={self._cached_bytes / (1024 ** 2):.1f}"
        )

    def _resolve_signal_path(self, raw_path: Path | str) -> Path:
        raw_str = str(raw_path).strip()
        native_path = Path(raw_str)
        if native_path.exists():
            return native_path

        normalized_str = raw_str.replace("\\", "/")
        normalized_path = Path(normalized_str)
        if normalized_path.exists():
            return normalized_path

        windows_parts = PureWindowsPath(raw_str).parts
        has_windows_drive = len(windows_parts) > 0 and windows_parts[0].endswith(":\\")

        if self.dataset_root is not None:
            if not normalized_path.is_absolute() and not has_windows_drive:
                candidate = self.dataset_root / normalized_path
                if candidate.exists():
                    return candidate

            if "Wearable_Dataset" in windows_parts:
                idx = windows_parts.index("Wearable_Dataset")
                candidate = self.dataset_root / Path(*windows_parts[idx:])
                if candidate.exists():
                    return candidate

            if "Wearable_Dataset" in normalized_path.parts:
                idx = normalized_path.parts.index("Wearable_Dataset")
                candidate = self.dataset_root / Path(*normalized_path.parts[idx:])
                if candidate.exists():
                    return candidate

        if not normalized_path.is_absolute() and not has_windows_drive:
            candidate = self.manifest_csv.parent / normalized_path
            if candidate.exists():
                return candidate

        raise FileNotFoundError(f"Could not resolve signal path: {raw_str}")
