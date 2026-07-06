from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import pywt
from tqdm.auto import tqdm

from .galaxy_dataset import GalaxyPPGWindowDataset
from .galaxy_protocols import CALM_SESSIONS, STRESS_SESSIONS


def wavelet_energy_profile(signal_1d: np.ndarray, wavelet: str, level: int) -> Dict[str, float]:
    coeffs = pywt.wavedec(signal_1d, wavelet=wavelet, level=level)
    names = [f"A{level}"] + [f"D{i}" for i in range(level, 0, -1)]
    energies = np.array([float(np.sum(np.square(c))) for c in coeffs], dtype=np.float64)
    total = float(energies.sum()) + 1e-12
    return {
        f"{name}_energy": energy
        for name, energy in zip(names, energies)
    } | {
        f"{name}_ratio": energy / total
        for name, energy in zip(names, energies)
    }


def cohens_d(x0: np.ndarray, x1: np.ndarray) -> float:
    if len(x0) < 2 or len(x1) < 2:
        return float("nan")
    var0 = np.var(x0, ddof=1)
    var1 = np.var(x1, ddof=1)
    pooled = ((len(x0) - 1) * var0 + (len(x1) - 1) * var1) / max(len(x0) + len(x1) - 2, 1)
    if pooled <= 0:
        return float("nan")
    return float((np.mean(x1) - np.mean(x0)) / np.sqrt(pooled))


def paired_effect_size(deltas: np.ndarray) -> float:
    if len(deltas) < 2:
        return float("nan")
    std = np.std(deltas, ddof=1)
    if std <= 0:
        return float("nan")
    return float(np.mean(deltas) / std)


def select_indices(
    dataset: GalaxyPPGWindowDataset,
    max_windows: int | None,
    max_per_subject_session: int | None,
    random_state: int,
) -> List[int]:
    manifest = dataset.manifest.reset_index().rename(columns={"index": "dataset_index"})

    if max_per_subject_session is not None:
        sampled_groups = []
        for _, group in manifest.groupby(["subject_id", "session"], sort=False):
            n = min(len(group), max_per_subject_session)
            sampled_groups.append(group.sample(n=n, random_state=random_state, replace=False))
        manifest = pd.concat(sampled_groups, axis=0, ignore_index=True)

    manifest = manifest.sort_values(["subject_id", "session", "window_start_ms"]).reset_index(drop=True)
    if max_windows is not None and len(manifest) > max_windows:
        manifest = manifest.sample(n=max_windows, random_state=random_state, replace=False).reset_index(drop=True)

    return manifest["dataset_index"].astype(int).tolist()


def build_window_records(
    dataset: GalaxyPPGWindowDataset,
    wavelet: str,
    level: int,
    max_windows: int | None = None,
    max_per_subject_session: int | None = None,
    random_state: int = 7,
) -> pd.DataFrame:
    records: List[Dict[str, float | int | str]] = []
    selected_indices = select_indices(
        dataset,
        max_windows=max_windows,
        max_per_subject_session=max_per_subject_session,
        random_state=random_state,
    )

    for index in tqdm(selected_indices, desc="wavelet analysis", leave=True):
        sample = dataset[index]

        watch_ppg = sample["watch_signal"][0].detach().cpu().numpy()
        e4_bvp = sample["e4_signal"][0].detach().cpu().numpy()
        polar_ecg = sample["polar_signal"][0].detach().cpu().numpy()

        row: Dict[str, float | int | str] = {
            "subject_id": sample["subject_id"],
            "subject_index": sample["subject_index"],
            "session": sample["session"],
            "group_name": sample["group_name"],
            "label": sample["label"],
            "watch_quality": float(sample["watch_quality"]),
            "polar_coverage": float(sample["polar_coverage"]),
        }
        row.update({f"watch_{k}": v for k, v in wavelet_energy_profile(watch_ppg, wavelet=wavelet, level=level).items()})
        row.update({f"e4_{k}": v for k, v in wavelet_energy_profile(e4_bvp, wavelet=wavelet, level=level).items()})
        row.update({f"polar_{k}": v for k, v in wavelet_energy_profile(polar_ecg, wavelet=wavelet, level=level).items()})
        records.append(row)

    return pd.DataFrame(records)


def summarize_effects(frame: pd.DataFrame, ratio_cols: List[str]) -> pd.DataFrame:
    rows = []
    calm = frame[frame["label"] == 0]
    stress = frame[frame["label"] == 1]

    for col in ratio_cols:
        x0 = calm[col].to_numpy(dtype=np.float64)
        x1 = stress[col].to_numpy(dtype=np.float64)
        rows.append(
            {
                "feature": col,
                "calm_mean": float(np.mean(x0)) if len(x0) else float("nan"),
                "stress_mean": float(np.mean(x1)) if len(x1) else float("nan"),
                "delta": float(np.mean(x1) - np.mean(x0)) if len(x0) and len(x1) else float("nan"),
                "cohens_d": cohens_d(x0, x1),
            }
        )
    summary = pd.DataFrame(rows).sort_values("cohens_d", key=lambda s: np.abs(s), ascending=False)
    return summary


def normalize_within_subject(frame: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    normalized = frame.copy()
    for subject_id, idx in normalized.groupby("subject_id").groups.items():
        subject_slice = normalized.loc[idx, feature_cols]
        means = subject_slice.mean(axis=0)
        stds = subject_slice.std(axis=0, ddof=0).replace(0.0, np.nan)
        normalized.loc[idx, feature_cols] = (subject_slice - means) / stds
    normalized[feature_cols] = normalized[feature_cols].fillna(0.0)
    return normalized


def summarize_session_pairs(
    session_frame: pd.DataFrame,
    feature_cols: List[str],
    baseline_session: str = "baseline",
) -> pd.DataFrame:
    rows = []
    baseline = session_frame[session_frame["session"] == baseline_session].set_index("subject_id")
    stress_sessions = [session for session in sorted(session_frame["session"].unique()) if session in STRESS_SESSIONS]

    for session in stress_sessions:
        current = session_frame[session_frame["session"] == session].set_index("subject_id")
        shared_subjects = baseline.index.intersection(current.index)
        if len(shared_subjects) == 0:
            continue
        base_vals = baseline.loc[shared_subjects, feature_cols]
        cur_vals = current.loc[shared_subjects, feature_cols]

        for col in feature_cols:
            deltas = (cur_vals[col] - base_vals[col]).to_numpy(dtype=np.float64)
            rows.append(
                {
                    "session": session,
                    "feature": col,
                    "n_subjects": int(len(shared_subjects)),
                    "baseline_mean": float(base_vals[col].mean()),
                    "session_mean": float(cur_vals[col].mean()),
                    "mean_delta": float(np.mean(deltas)),
                    "paired_effect": paired_effect_size(deltas),
                }
            )

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    summary["abs_paired_effect"] = summary["paired_effect"].abs()
    summary = summary.sort_values(["session", "abs_paired_effect"], ascending=[True, False]).reset_index(drop=True)
    return summary


def summarize_alignment(frame: pd.DataFrame, level: int) -> pd.DataFrame:
    rows = []
    names = [f"A{level}"] + [f"D{i}" for i in range(level, 0, -1)]
    for name in names:
        watch_col = f"watch_{name}_ratio"
        e4_col = f"e4_{name}_ratio"
        polar_col = f"polar_{name}_ratio"
        rows.append(
            {
                "band": name,
                "watch_e4_corr": float(frame[watch_col].corr(frame[e4_col])),
                "watch_polar_corr": float(frame[watch_col].corr(frame[polar_col])),
                "e4_polar_corr": float(frame[e4_col].corr(frame[polar_col])),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze GalaxyPPG wavelet energy distributions across calm and stress windows.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--wavelet", type=str, default="db4")
    parser.add_argument("--level", type=int, default=4)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--max-per-subject-session", type=int, default=24)
    parser.add_argument("--random-state", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    dataset = GalaxyPPGWindowDataset(
        manifest_csv=args.manifest,
        split=args.split,
        dataset_root=args.dataset_root,
        cache_tables=True,
    )
    frame = build_window_records(
        dataset,
        wavelet=args.wavelet,
        level=args.level,
        max_windows=args.max_windows,
        max_per_subject_session=args.max_per_subject_session,
        random_state=args.random_state,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    windows_path = args.output_dir / f"wavelet_windows_{args.split}.csv"
    frame.to_csv(windows_path, index=False)

    ratio_cols = [col for col in frame.columns if col.startswith("watch_") and col.endswith("_ratio")]
    normalized_frame = normalize_within_subject(frame, ratio_cols)
    normalized_windows_path = args.output_dir / f"wavelet_windows_subjectnorm_{args.split}.csv"
    normalized_frame.to_csv(normalized_windows_path, index=False)

    subject_label_frame = (
        normalized_frame.groupby(["subject_id", "label"], as_index=False)[ratio_cols]
        .mean()
        .sort_values(["subject_id", "label"])
        .reset_index(drop=True)
    )
    subject_label_path = args.output_dir / f"wavelet_subject_label_{args.split}.csv"
    subject_label_frame.to_csv(subject_label_path, index=False)

    effect_summary = summarize_effects(subject_label_frame, ratio_cols)
    effect_path = args.output_dir / f"wavelet_effects_{args.split}.csv"
    effect_summary.to_csv(effect_path, index=False)

    session_summary = (
        normalized_frame.groupby(["label", "session"])[ratio_cols + ["watch_quality", "polar_coverage"]]
        .mean()
        .reset_index()
        .sort_values(["label", "session"])
    )
    session_path = args.output_dir / f"wavelet_sessions_{args.split}.csv"
    session_summary.to_csv(session_path, index=False)

    subject_session_frame = (
        normalized_frame.groupby(["subject_id", "label", "session"], as_index=False)[ratio_cols]
        .mean()
        .sort_values(["subject_id", "session"])
        .reset_index(drop=True)
    )
    subject_session_path = args.output_dir / f"wavelet_subject_session_{args.split}.csv"
    subject_session_frame.to_csv(subject_session_path, index=False)

    session_pair_summary = summarize_session_pairs(subject_session_frame, ratio_cols, baseline_session="baseline")
    session_pair_path = args.output_dir / f"wavelet_session_pairs_{args.split}.csv"
    session_pair_summary.to_csv(session_pair_path, index=False)

    alignment_summary = summarize_alignment(normalized_frame, level=args.level)
    alignment_path = args.output_dir / f"wavelet_alignment_{args.split}.csv"
    alignment_summary.to_csv(alignment_path, index=False)

    print("Top subject-normalized watch-band effect sizes:")
    print(effect_summary.head(8).to_string(index=False))
    print()
    print("Top session-specific paired effects:")
    top_session_rows = (
        session_pair_summary.groupby("session", as_index=False)
        .head(3)
        .reset_index(drop=True)
    )
    if not top_session_rows.empty:
        print(top_session_rows.to_string(index=False))
        print()
    print("Cross-device alignment by band:")
    print(alignment_summary.to_string(index=False))
    print()
    print(f"Saved per-window features to {windows_path}")
    print(f"Saved subject-normalized per-window features to {normalized_windows_path}")
    print(f"Saved subject-label aggregated features to {subject_label_path}")
    print(f"Saved calm-vs-stress effect summary to {effect_path}")
    print(f"Saved per-session summary to {session_path}")
    print(f"Saved subject-session summary to {subject_session_path}")
    print(f"Saved baseline-vs-session paired summary to {session_pair_path}")
    print(f"Saved cross-device alignment summary to {alignment_path}")


if __name__ == "__main__":
    main()
