from __future__ import annotations

import argparse
import math
import pickle
from pathlib import Path, PureWindowsPath
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from scipy import signal


DEFAULT_GALAXY_CALM_SESSIONS = ["baseline"]
DEFAULT_GALAXY_STRESS_SESSIONS = ["tsst-prep"]
DEFAULT_WESAD_CALM_SESSIONS = ["baseline"]
DEFAULT_WESAD_STRESS_SESSIONS = ["stress"]
DEFAULT_WAVELET_BANDS = ("A4", "D4", "D2", "D1")
POLAR_TIME_OFFSET_MS = 9 * 3600 * 1000
WESAD_CHEST_SAMPLE_RATE = 700
WESAD_WRIST_SAMPLE_RATES = {"ACC": 32, "BVP": 64}
GALAXY_TARGET_RATES = {
    "gw_ppg": 25,
    "gw_acc": 25,
    "e4_bvp": 64,
    "e4_acc": 32,
    "polar_ecg": 130,
}
WESAD_TARGET_RATES = {
    "watch": 32,
    "privileged": 64,
}

WATCH_COLORS = {
    "ppg": "#2563EB",
    "acc_x": "#111827",
    "acc_y": "#F97316",
    "acc_z": "#2563EB",
}
PRIV_COLOR = "#2F855A"


def maybe_parse_sessions(values: Sequence[str] | None, fallback: Sequence[str]) -> list[str]:
    if values is None or len(values) == 0:
        return list(fallback)
    return [str(item) for item in values]


def discover_manifest(manifests_dir: Path, dataset_kind: str, subject: str | None) -> tuple[str, Path]:
    prefix = "galaxy" if dataset_kind == "galaxy" else "wesad"
    pattern = f"{prefix}_*_loso_val.csv"
    candidates = sorted(manifests_dir.glob(pattern))
    if subject:
        wanted = str(subject).strip()
        candidates = [
            path
            for path in candidates
            if path.stem.replace(f"{prefix}_", "").replace("_loso_val", "") == wanted
        ]
    if not candidates:
        raise ValueError(f"No {dataset_kind} LOSO manifest found in {manifests_dir} for subject={subject!r}")
    path = candidates[0]
    fold_subject = path.stem.replace(f"{prefix}_", "").replace("_loso_val", "")
    return fold_subject, path


def robust_zscore(values: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    median = np.median(values, axis=0, keepdims=True)
    mad = np.median(np.abs(values - median), axis=0, keepdims=True)
    scale = 1.4826 * mad + eps
    return np.nan_to_num((values - median) / scale, nan=0.0, posinf=0.0, neginf=0.0)


def preprocess_acc(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 1:
        values = values[:, None]
    magnitude = np.linalg.norm(values, axis=1, keepdims=True)
    return robust_zscore(np.concatenate([values, magnitude], axis=1))


def safe_bandpass(values: np.ndarray, sampling_rate: float, low_hz: float, high_hz: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if len(values) < max(int(sampling_rate), 16):
        return robust_zscore(values)

    nyquist = sampling_rate / 2.0
    low = max(low_hz / nyquist, 1e-4)
    high = min(high_hz / nyquist, 0.99)
    if low >= high:
        return robust_zscore(values)

    b, a = signal.butter(3, [low, high], btype="bandpass")
    try:
        filtered = signal.filtfilt(b, a, values).astype(np.float32)
    except ValueError:
        filtered = values
    return robust_zscore(filtered)


def preprocess_ppg(values: np.ndarray, sampling_rate: float) -> np.ndarray:
    return safe_bandpass(values, sampling_rate=sampling_rate, low_hz=0.5, high_hz=8.0)


def preprocess_ecg(values: np.ndarray, sampling_rate: float) -> np.ndarray:
    return safe_bandpass(values, sampling_rate=sampling_rate, low_hz=0.5, high_hz=40.0)


def preprocess_resp(values: np.ndarray, sampling_rate: float) -> np.ndarray:
    return safe_bandpass(values, sampling_rate=sampling_rate, low_hz=0.05, high_hz=1.0)


def preprocess_emg(values: np.ndarray, sampling_rate: float) -> np.ndarray:
    return safe_bandpass(values, sampling_rate=sampling_rate, low_hz=10.0, high_hz=45.0)


def preprocess_scalar(values: np.ndarray) -> np.ndarray:
    return robust_zscore(np.asarray(values, dtype=np.float32).reshape(-1))


def resample_array(values: np.ndarray, orig_sr: float, target_sr: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if math.isclose(orig_sr, target_sr):
        return values.astype(np.float32)
    target_len = int(round(len(values) * target_sr / orig_sr))
    if values.ndim == 1:
        return signal.resample(values, target_len).astype(np.float32)
    channels = [signal.resample(values[:, idx], target_len).astype(np.float32) for idx in range(values.shape[1])]
    return np.stack(channels, axis=1)


def ensure_length(values: np.ndarray, target_len: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
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


class RawWaveformWindowDataset:
    """Lightweight waveform-only loader used when full training datasets cannot import locally."""

    def __init__(
        self,
        dataset_kind: str,
        manifest_csv: Path,
        split: str,
        dataset_root: Path,
        include_sessions: list[str],
    ) -> None:
        self.dataset_kind = dataset_kind
        self.manifest_csv = Path(manifest_csv)
        self.dataset_root = Path(dataset_root)
        self.manifest = pd.read_csv(self.manifest_csv)
        self.manifest = self.manifest[self.manifest["split"] == split].reset_index(drop=True)
        if include_sessions:
            self.manifest = self.manifest[self.manifest["session"].astype(str).isin(include_sessions)].reset_index(drop=True)
        self._csv_cache: dict[Path, pd.DataFrame] = {}
        self._pkl_cache: dict[Path, dict[str, Any]] = {}

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.manifest.iloc[index]
        if self.dataset_kind == "galaxy":
            return self._get_galaxy(row)
        return self._get_wesad(row)

    def _read_csv(self, rel_path: str) -> pd.DataFrame:
        path = self.dataset_root / Path(str(rel_path))
        cached = self._csv_cache.get(path)
        if cached is not None:
            return cached
        table = pd.read_csv(path)
        self._csv_cache[path] = table
        return table

    @staticmethod
    def _slice_csv_window(
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

    def _get_galaxy(self, row: pd.Series) -> dict[str, Any]:
        start_ms = int(row["window_start_ms"])
        end_ms = int(row["window_end_ms"])
        duration_s = float(row["window_duration_s"])

        ppg_window = self._slice_csv_window(self._read_csv(row["gw_ppg_path"]), "timestamp", start_ms, end_ms)
        acc_window = self._slice_csv_window(self._read_csv(row["gw_acc_path"]), "timestamp", start_ms, end_ms)
        e4_bvp_window = self._slice_csv_window(
            self._read_csv(row["e4_bvp_path"]),
            "timestamp",
            start_ms,
            end_ms,
            time_divisor=1000.0,
        )
        e4_acc_window = self._slice_csv_window(
            self._read_csv(row["e4_acc_path"]),
            "timestamp",
            start_ms,
            end_ms,
            time_divisor=1000.0,
        )
        polar_window = self._slice_csv_window(
            self._read_csv(row["polar_ecg_path"]),
            "phoneTimestamp",
            start_ms,
            end_ms,
            time_offset_ms=POLAR_TIME_OFFSET_MS,
        )
        if ppg_window.empty or acc_window.empty or e4_bvp_window.empty or e4_acc_window.empty or polar_window.empty:
            raise ValueError(f"Empty Galaxy window for {row['subject_id']} {row['session']} {start_ms}-{end_ms}")

        ppg = preprocess_ppg(ppg_window["ppg"].to_numpy(dtype=np.float32), sampling_rate=25)
        acc = preprocess_acc(acc_window[["x", "y", "z"]].to_numpy(dtype=np.float32))
        ppg = resample_array(ppg, orig_sr=25, target_sr=GALAXY_TARGET_RATES["gw_ppg"])
        acc = resample_array(acc, orig_sr=25, target_sr=GALAXY_TARGET_RATES["gw_acc"])
        ppg = ensure_length(ppg, int(round(duration_s * GALAXY_TARGET_RATES["gw_ppg"])))
        acc = ensure_length(acc, int(round(duration_s * GALAXY_TARGET_RATES["gw_acc"])))
        min_len = min(len(ppg), len(acc))
        watch = np.concatenate([ppg[:min_len, None], acc[:min_len]], axis=1).T

        e4_bvp = preprocess_ppg(e4_bvp_window["value"].to_numpy(dtype=np.float32), sampling_rate=64)
        e4_acc = preprocess_acc(e4_acc_window[["x", "y", "z"]].to_numpy(dtype=np.float32))
        e4_bvp = resample_array(e4_bvp, orig_sr=64, target_sr=GALAXY_TARGET_RATES["e4_bvp"])
        e4_acc = resample_array(e4_acc, orig_sr=32, target_sr=GALAXY_TARGET_RATES["e4_acc"])
        e4_bvp = ensure_length(e4_bvp, int(round(duration_s * GALAXY_TARGET_RATES["e4_bvp"])))
        e4_acc = ensure_length(e4_acc, int(round(duration_s * GALAXY_TARGET_RATES["e4_acc"])))
        min_len = min(len(e4_bvp), len(e4_acc))
        e4 = np.concatenate([e4_bvp[:min_len, None], e4_acc[:min_len]], axis=1).T

        polar = preprocess_ecg(polar_window["ecg"].to_numpy(dtype=np.float32), sampling_rate=130)
        polar = ensure_length(
            resample_array(polar, orig_sr=130, target_sr=GALAXY_TARGET_RATES["polar_ecg"]),
            int(round(duration_s * GALAXY_TARGET_RATES["polar_ecg"])),
        )[None, :]

        return self._base_sample(row, start_ms, end_ms) | {
            "watch_signal": watch.astype(np.float32),
            "e4_signal": e4.astype(np.float32),
            "polar_signal": polar.astype(np.float32),
        }

    def _resolve_subject_pkl(self, rel_path: str) -> Path:
        rel = PureWindowsPath(str(rel_path))
        parts = list(rel.parts)
        if parts and parts[0].lower() == self.dataset_root.name.lower():
            parts = parts[1:]
        return self.dataset_root.joinpath(*parts)

    def _load_payload(self, rel_path: str) -> dict[str, Any]:
        path = self._resolve_subject_pkl(rel_path)
        cached = self._pkl_cache.get(path)
        if cached is not None:
            return cached
        with path.open("rb") as handle:
            payload = pickle.load(handle, encoding="latin1")
        self._pkl_cache[path] = payload
        return payload

    @staticmethod
    def _slice_array(values: np.ndarray, sampling_rate: float, start_s: float, end_s: float) -> np.ndarray:
        start_idx = max(int(round(start_s * sampling_rate)), 0)
        end_idx = min(int(round(end_s * sampling_rate)), len(values))
        return np.asarray(values[start_idx:end_idx], dtype=np.float32)

    def _get_wesad(self, row: pd.Series) -> dict[str, Any]:
        start_s = float(row["window_start_s"])
        end_s = float(row["window_end_s"])
        start_ms = int(row["window_start_ms"])
        end_ms = int(row["window_end_ms"])
        duration_s = float(row["window_duration_s"])
        payload = self._load_payload(row["subject_pkl_path"])

        wrist = payload["signal"]["wrist"]
        bvp = self._slice_array(np.asarray(wrist["BVP"], dtype=np.float32).reshape(-1), 64, start_s, end_s)
        acc = self._slice_array(np.asarray(wrist["ACC"], dtype=np.float32), 32, start_s, end_s)
        bvp = resample_array(preprocess_ppg(bvp, sampling_rate=64), orig_sr=64, target_sr=WESAD_TARGET_RATES["watch"])
        acc = resample_array(preprocess_acc(acc), orig_sr=32, target_sr=WESAD_TARGET_RATES["watch"])
        bvp = ensure_length(bvp, int(round(duration_s * WESAD_TARGET_RATES["watch"])))
        acc = ensure_length(acc, int(round(duration_s * WESAD_TARGET_RATES["watch"])))
        min_len = min(len(bvp), len(acc))
        watch = np.concatenate([bvp[:min_len, None], acc[:min_len]], axis=1).T

        chest = payload["signal"]["chest"]
        acc_chest = self._slice_array(np.asarray(chest["ACC"], dtype=np.float32), WESAD_CHEST_SAMPLE_RATE, start_s, end_s)
        acc_chest = resample_array(preprocess_acc(acc_chest), WESAD_CHEST_SAMPLE_RATE, WESAD_TARGET_RATES["privileged"])
        scalar_specs = [
            ("ECG", preprocess_ecg),
            ("EMG", preprocess_emg),
            ("EDA", lambda values, sampling_rate: preprocess_scalar(values)),
            ("Temp", lambda values, sampling_rate: preprocess_scalar(values)),
            ("Resp", preprocess_resp),
        ]
        pieces = [acc_chest]
        for name, preprocess_fn in scalar_specs:
            stream = self._slice_array(
                np.asarray(chest[name], dtype=np.float32).reshape(-1),
                WESAD_CHEST_SAMPLE_RATE,
                start_s,
                end_s,
            )
            stream = preprocess_fn(stream, WESAD_CHEST_SAMPLE_RATE)
            stream = resample_array(stream, WESAD_CHEST_SAMPLE_RATE, WESAD_TARGET_RATES["privileged"])
            pieces.append(stream[:, None])
        target_len = int(round(duration_s * WESAD_TARGET_RATES["privileged"]))
        pieces = [ensure_length(piece, target_len) for piece in pieces]
        min_len = min(len(piece) for piece in pieces)
        privileged = np.concatenate([piece[:min_len] for piece in pieces], axis=1).T

        return self._base_sample(row, start_ms, end_ms) | {
            "watch_signal": watch.astype(np.float32),
            "privileged_signal": privileged.astype(np.float32),
        }

    @staticmethod
    def _base_sample(row: pd.Series, start_ms: int, end_ms: int) -> dict[str, Any]:
        return {
            "label": int(row["label"]),
            "subject_id": row["subject_id"],
            "subject_index": int(row["subject_index"]),
            "session": row["session"],
            "group_name": row["group_name"],
            "window_start_ms": start_ms,
            "window_end_ms": end_ms,
        }


def build_dataset(args: argparse.Namespace, manifest_path: Path, include_sessions: list[str]) -> Any:
    try:
        if args.dataset_kind == "galaxy":
            from .galaxy_dataset import GalaxyPrivilegedWindowDataset

            return GalaxyPrivilegedWindowDataset(
                manifest_csv=manifest_path,
                split=args.split,
                dataset_root=args.dataset_root,
                include_sessions=include_sessions,
                cache_tables=True,
                wavelet=args.wavelet,
                wavelet_level=args.wavelet_level,
                wavelet_bands=DEFAULT_WAVELET_BANDS,
            )

        from .wesad_dataset import WESADPrivilegedWindowDataset

        return WESADPrivilegedWindowDataset(
            manifest_csv=manifest_path,
            split=args.split,
            wesad_root=args.dataset_root,
            include_sessions=include_sessions,
            cache_subjects=args.cache_subjects,
            wavelet=args.wavelet,
            wavelet_level=args.wavelet_level,
            wavelet_bands=DEFAULT_WAVELET_BANDS,
        )
    except Exception as exc:
        print(f"dataset_import_fallback=raw_waveform_loader reason={type(exc).__name__}: {exc}")
        return RawWaveformWindowDataset(
            dataset_kind=args.dataset_kind,
            manifest_csv=manifest_path,
            split=args.split,
            dataset_root=args.dataset_root,
            include_sessions=include_sessions,
        )



def as_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def robust_scale(values: np.ndarray, target_amp: float = 1.0) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    values = values - np.nanmedian(values)
    scale = np.nanpercentile(np.abs(values), 95)
    if not math.isfinite(float(scale)) or scale < 1e-6:
        scale = np.nanstd(values)
    if not math.isfinite(float(scale)) or scale < 1e-6:
        return np.zeros_like(values)
    return np.clip(values / scale, -2.5, 2.5) * float(target_amp)


def choose_index(dataset: Any, args: argparse.Namespace) -> int:
    frame = dataset.manifest.reset_index(drop=True).copy()
    if args.session:
        frame = frame[frame["session"].astype(str) == str(args.session)]
    if args.label is not None:
        frame = frame[frame["label"].astype(int) == int(args.label)]
    if frame.empty:
        raise ValueError("No windows match the requested session/label filters.")

    if args.window_index is not None:
        if args.window_index < 0 or args.window_index >= len(frame):
            raise IndexError(f"--window-index {args.window_index} outside filtered rows 0..{len(frame) - 1}")
        return int(frame.index[int(args.window_index)])

    if args.selection == "first":
        return int(frame.index[0])
    if args.selection == "middle":
        return int(frame.index[len(frame) // 2])
    if args.selection == "random":
        rng = np.random.default_rng(args.seed)
        return int(rng.choice(frame.index.to_numpy()))

    # Motion-heavy selector uses the already preprocessed watch ACC window.
    best_idx = int(frame.index[0])
    best_score = -float("inf")
    scan = frame
    if args.max_scan_windows and args.max_scan_windows > 0:
        scan = scan.head(args.max_scan_windows)
    for idx in scan.index:
        sample = dataset[int(idx)]
        watch = as_numpy(sample["watch_signal"])
        if watch.shape[0] < 4:
            continue
        acc = watch[1:4]
        score = float(np.sqrt(np.mean(np.square(acc))))
        if score > best_score:
            best_score = score
            best_idx = int(idx)
    return best_idx


def time_axis(num_samples: int, duration_s: float) -> np.ndarray:
    if num_samples <= 1:
        return np.zeros((num_samples,), dtype=np.float32)
    return np.linspace(0.0, float(duration_s), num_samples, endpoint=False, dtype=np.float32)


def sample_duration_s(sample: dict[str, Any]) -> float:
    start_ms = sample.get("window_start_ms")
    end_ms = sample.get("window_end_ms")
    if start_ms is not None and end_ms is not None:
        return max((float(end_ms) - float(start_ms)) / 1000.0, 1e-6)
    return 20.0


def collect_traces(sample: dict[str, Any], dataset_kind: str) -> list[dict[str, Any]]:
    duration_s = sample_duration_s(sample)
    traces: list[dict[str, Any]] = []

    watch = as_numpy(sample["watch_signal"])
    if watch.ndim == 2 and watch.shape[0] >= 4:
        names = ["PPG/BVP", "ACC x", "ACC y", "ACC z"]
        keys = ["ppg", "acc_x", "acc_y", "acc_z"]
        for idx, (name, key) in enumerate(zip(names, keys)):
            values = robust_scale(watch[idx])
            traces.append(
                {
                    "panel": "deployable",
                    "name": name,
                    "key": key,
                    "color": WATCH_COLORS[key],
                    "time_s": time_axis(len(values), duration_s),
                    "value": values,
                }
            )

    if dataset_kind == "galaxy":
        e4 = as_numpy(sample.get("e4_signal", np.empty((0, 0))))
        if e4.ndim == 2 and e4.shape[0] >= 1:
            values = robust_scale(e4[0])
            traces.append(
                {
                    "panel": "privileged",
                    "name": "E4 BVP",
                    "key": "e4_bvp",
                    "color": PRIV_COLOR,
                    "time_s": time_axis(len(values), duration_s),
                    "value": values,
                }
            )
        polar = as_numpy(sample.get("polar_signal", np.empty((0, 0))))
        if polar.ndim == 2 and polar.shape[0] >= 1:
            values = robust_scale(polar[0])
            traces.append(
                {
                    "panel": "privileged",
                    "name": "Polar ECG",
                    "key": "polar_ecg",
                    "color": PRIV_COLOR,
                    "time_s": time_axis(len(values), duration_s),
                    "value": values,
                }
            )
    else:
        privileged = as_numpy(sample.get("privileged_signal", np.empty((0, 0))))
        names = ["Chest ACC x", "Chest ACC y", "Chest ACC z", "ECG", "EMG", "EDA", "Temp", "Resp"]
        keys = ["chest_acc_x", "chest_acc_y", "chest_acc_z", "ecg", "emg", "eda", "temp", "resp"]
        if privileged.ndim == 2:
            for idx in range(min(privileged.shape[0], len(names))):
                values = robust_scale(privileged[idx])
                traces.append(
                    {
                        "panel": "privileged",
                        "name": names[idx],
                        "key": keys[idx],
                        "color": PRIV_COLOR,
                        "time_s": time_axis(len(values), duration_s),
                        "value": values,
                    }
                )
    return traces


def trace_rows(traces: list[dict[str, Any]], sample: dict[str, Any], fold_subject: str, dataset_kind: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for trace in traces:
        for t, v in zip(trace["time_s"], trace["value"]):
            rows.append(
                {
                    "dataset": dataset_kind,
                    "fold_subject": fold_subject,
                    "subject_id": sample.get("subject_id", ""),
                    "session": sample.get("session", ""),
                    "group_name": sample.get("group_name", ""),
                    "label": int(sample.get("label", -1)),
                    "window_start_ms": sample.get("window_start_ms", ""),
                    "window_end_ms": sample.get("window_end_ms", ""),
                    "panel": trace["panel"],
                    "signal": trace["name"],
                    "key": trace["key"],
                    "time_s": float(t),
                    "value": float(v),
                }
            )
    return pd.DataFrame(rows)


def plot_stack(
    traces: list[dict[str, Any]],
    output_path: Path,
    title: str,
    panel_filter: str | None = None,
    transparent: bool = True,
    dpi: int = 450,
) -> None:
    selected = [trace for trace in traces if panel_filter is None or trace["panel"] == panel_filter]
    if not selected:
        print(f"plot_skip={output_path} reason=no_traces")
        return

    if output_path.suffix.lower() == ".svg":
        write_stack_svg(selected, output_path, title=title, transparent=transparent)
        return

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:
        fallback_path = output_path.with_suffix(".svg")
        write_stack_svg(selected, fallback_path, title=title, transparent=transparent)
        print(f"plot_fallback={fallback_path} reason=matplotlib_unavailable:{type(exc).__name__}")
        return

    height = max(1.8, 0.48 * len(selected) + 0.35)
    fig, ax = plt.subplots(figsize=(4.2, height), dpi=180)
    offset_step = 1.25
    y_ticks = []
    y_labels = []
    for row_idx, trace in enumerate(selected):
        offset = (len(selected) - row_idx - 1) * offset_step
        ax.plot(trace["time_s"], trace["value"] + offset, color=trace["color"], linewidth=1.35)
        y_ticks.append(offset)
        y_labels.append(trace["name"])

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=9, fontweight="bold")
    ax.set_xlabel("Time (s)", fontsize=8)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=8)
    ax.set_ylim(-0.65, (len(selected) - 1) * offset_step + 0.65)
    fig.tight_layout(pad=0.35)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", transparent=transparent)
    plt.close(fig)


def svg_escape(text: Any) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def polyline_points(time_s: np.ndarray, values: np.ndarray, width: float, center_y: float, amp: float, x0: float) -> str:
    time_s = np.asarray(time_s, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    if len(time_s) == 0:
        return ""
    t_min = float(np.nanmin(time_s))
    t_max = float(np.nanmax(time_s))
    denom = max(t_max - t_min, 1e-6)
    xs = x0 + (time_s - t_min) / denom * width
    ys = center_y - values * amp
    return " ".join(f"{float(x):.2f},{float(y):.2f}" for x, y in zip(xs, ys))


def write_stack_svg(
    traces: list[dict[str, Any]],
    output_path: Path,
    title: str,
    transparent: bool = True,
) -> None:
    row_h = 54.0
    label_w = 92.0
    plot_w = 360.0
    top = 38.0 if title else 14.0
    bottom = 18.0
    width = label_w + plot_w + 18.0
    height = top + row_h * len(traces) + bottom
    bg = "" if transparent else f'<rect width="{width:.0f}" height="{height:.0f}" fill="white"/>'
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" viewBox="0 0 {width:.0f} {height:.0f}">',
        bg,
    ]
    if title:
        parts.append(
            f'<text x="{width / 2:.1f}" y="20" text-anchor="middle" '
            'font-family="Arial, Helvetica, sans-serif" font-size="15" font-weight="700" fill="#111827">'
            f"{svg_escape(title)}</text>"
        )
    for row_idx, trace in enumerate(traces):
        center_y = top + row_idx * row_h + row_h * 0.5
        parts.append(
            f'<text x="{label_w - 10:.1f}" y="{center_y + 4:.1f}" text-anchor="end" '
            'font-family="Arial, Helvetica, sans-serif" font-size="13" font-weight="700" fill="#111827">'
            f"{svg_escape(trace['name'])}</text>"
        )
        points = polyline_points(trace["time_s"], trace["value"], plot_w, center_y, amp=13.0, x0=label_w)
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="{trace["color"]}" '
            'stroke-width="2.0" stroke-linecap="round" stroke-linejoin="round"/>'
        )
    parts.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(parts), encoding="utf-8")


def write_individual_svg(trace: dict[str, Any], output_path: Path, transparent: bool = True) -> None:
    width = 300.0
    height = 64.0
    center_y = height / 2.0
    bg = "" if transparent else f'<rect width="{width:.0f}" height="{height:.0f}" fill="white"/>'
    points = polyline_points(trace["time_s"], trace["value"], width - 8.0, center_y, amp=18.0, x0=4.0)
    svg = "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" viewBox="0 0 {width:.0f} {height:.0f}">',
            bg,
            f'<polyline points="{points}" fill="none" stroke="{trace["color"]}" '
            'stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>',
            "</svg>",
        ]
    )
    output_path.write_text(svg, encoding="utf-8")


def plot_individual_traces(traces: list[dict[str, Any]], output_dir: Path, prefix: str, transparent: bool, dpi: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:
        for trace in traces:
            safe_name = trace["key"].replace("/", "_")
            write_individual_svg(trace, output_dir / f"{prefix}_{safe_name}.svg", transparent=transparent)
        print(f"individual_plot_fallback={output_dir} reason=matplotlib_unavailable:{type(exc).__name__}")
        return

    for trace in traces:
        fig, ax = plt.subplots(figsize=(2.6, 0.62), dpi=180)
        ax.plot(trace["time_s"], trace["value"], color=trace["color"], linewidth=1.45)
        ax.axis("off")
        fig.tight_layout(pad=0.0)
        safe_name = trace["key"].replace("/", "_")
        for suffix in ("png", "svg"):
            fig.savefig(
                output_dir / f"{prefix}_{safe_name}.{suffix}",
                dpi=dpi,
                bbox_inches="tight",
                pad_inches=0.01,
                transparent=transparent,
            )
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export clean waveform snippets for paper figures.")
    parser.add_argument("--dataset-kind", choices=["galaxy", "wesad"], required=True)
    parser.add_argument("--manifests-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", "--wesad-root", dest="dataset_root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--subject", type=str, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--session", type=str, default=None)
    parser.add_argument("--label", type=int, choices=[0, 1], default=None)
    parser.add_argument("--selection", choices=["first", "middle", "random", "high_motion"], default="middle")
    parser.add_argument("--window-index", type=int, default=None, help="Index within the filtered manifest rows.")
    parser.add_argument("--max-scan-windows", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wavelet", type=str, default="db4")
    parser.add_argument("--wavelet-level", type=int, default=4)
    parser.add_argument("--cache-subjects", type=int, default=4)
    parser.add_argument("--calm-sessions", nargs="*", default=None)
    parser.add_argument("--stress-sessions", nargs="*", default=None)
    parser.add_argument("--prefix", type=str, default=None)
    parser.add_argument("--white-background", action="store_true")
    parser.add_argument("--dpi", type=int, default=450)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dataset_kind == "galaxy":
        calm_sessions = maybe_parse_sessions(args.calm_sessions, DEFAULT_GALAXY_CALM_SESSIONS)
        stress_sessions = maybe_parse_sessions(args.stress_sessions, DEFAULT_GALAXY_STRESS_SESSIONS)
    else:
        calm_sessions = maybe_parse_sessions(args.calm_sessions, DEFAULT_WESAD_CALM_SESSIONS)
        stress_sessions = maybe_parse_sessions(args.stress_sessions, DEFAULT_WESAD_STRESS_SESSIONS)
    include_sessions = calm_sessions + stress_sessions

    fold_subject, manifest_path = discover_manifest(args.manifests_dir, args.dataset_kind, args.subject)
    dataset = build_dataset(args, manifest_path, include_sessions)
    idx = choose_index(dataset, args)
    sample = dataset[idx]
    traces = collect_traces(sample, args.dataset_kind)

    prefix = args.prefix
    if prefix is None:
        prefix = (
            f"{args.dataset_kind}_{sample.get('subject_id', fold_subject)}_"
            f"{sample.get('session', 'session')}_label{sample.get('label', 'x')}_idx{idx}"
        )
    prefix = str(prefix).replace("/", "_").replace("\\", "_").replace(" ", "_")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / f"{prefix}_waveforms.csv"
    meta_path = args.output_dir / f"{prefix}_metadata.txt"
    all_png = args.output_dir / f"{prefix}_all_waveforms.png"
    all_svg = args.output_dir / f"{prefix}_all_waveforms.svg"
    deploy_png = args.output_dir / f"{prefix}_deployable_waveforms.png"
    deploy_svg = args.output_dir / f"{prefix}_deployable_waveforms.svg"
    priv_png = args.output_dir / f"{prefix}_privileged_waveforms.png"
    priv_svg = args.output_dir / f"{prefix}_privileged_waveforms.svg"

    trace_rows(traces, sample, fold_subject, args.dataset_kind).to_csv(csv_path, index=False)
    transparent = not args.white_background
    title = f"{args.dataset_kind.upper()} {sample.get('subject_id', fold_subject)} {sample.get('session', '')}"
    plot_stack(traces, all_png, title=title, panel_filter=None, transparent=transparent, dpi=args.dpi)
    plot_stack(traces, all_svg, title=title, panel_filter=None, transparent=transparent, dpi=args.dpi)
    plot_stack(traces, deploy_png, title="Deployable watch sensors", panel_filter="deployable", transparent=transparent, dpi=args.dpi)
    plot_stack(traces, deploy_svg, title="Deployable watch sensors", panel_filter="deployable", transparent=transparent, dpi=args.dpi)
    plot_stack(traces, priv_png, title="Privileged sensors", panel_filter="privileged", transparent=transparent, dpi=args.dpi)
    plot_stack(traces, priv_svg, title="Privileged sensors", panel_filter="privileged", transparent=transparent, dpi=args.dpi)
    plot_individual_traces(traces, args.output_dir / f"{prefix}_individual", prefix, transparent=transparent, dpi=args.dpi)

    metadata = {
        "dataset": args.dataset_kind,
        "fold_subject": fold_subject,
        "manifest": str(manifest_path),
        "dataset_index": idx,
        "subject_id": sample.get("subject_id", ""),
        "session": sample.get("session", ""),
        "group_name": sample.get("group_name", ""),
        "label": sample.get("label", ""),
        "window_start_ms": sample.get("window_start_ms", ""),
        "window_end_ms": sample.get("window_end_ms", ""),
        "signals": ", ".join(trace["name"] for trace in traces),
    }
    meta_path.write_text("\n".join(f"{key}={value}" for key, value in metadata.items()) + "\n", encoding="utf-8")

    print(f"selected_index={idx}")
    for key, value in metadata.items():
        print(f"{key}={value}")
    print(f"saved_csv={csv_path}")
    print(f"saved_metadata={meta_path}")
    print(f"saved_all={all_png}")
    print(f"saved_deployable={deploy_png}")
    print(f"saved_privileged={priv_png}")
    print(f"saved_individual_dir={args.output_dir / f'{prefix}_individual'}")


if __name__ == "__main__":
    main()
