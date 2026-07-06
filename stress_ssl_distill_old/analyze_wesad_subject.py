from __future__ import annotations

import argparse
import pickle
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_SAMPLE_RATES = {
    "chest": {
        "ACC": 700,
        "ECG": 700,
        "EMG": 700,
        "EDA": 700,
        "Temp": 700,
        "Resp": 700,
        "label": 700,
    },
    "wrist": {
        "ACC": 32,
        "BVP": 64,
        "EDA": 4,
        "TEMP": 4,
    },
}

LABEL_NAME_MAP = {
    0: "undefined_or_transition",
    1: "baseline",
    2: "stress",
    3: "amusement",
    4: "meditation",
    6: "unknown_extra_6",
    7: "unknown_extra_7",
}


def load_subject_pickle(subject_dir: Path) -> dict[str, Any]:
    pkl_path = subject_dir / f"{subject_dir.name}.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"Could not find subject pickle: {pkl_path}")
    with pkl_path.open("rb") as handle:
        return pickle.load(handle, encoding="latin1")


def summarize_signal_dict(signal_dict: dict[str, np.ndarray], sample_rates: dict[str, int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, array in signal_dict.items():
        shape = tuple(array.shape)
        samples = int(shape[0]) if len(shape) >= 1 else 0
        channels = int(shape[1]) if len(shape) > 1 else 1
        sample_rate = sample_rates.get(name)
        duration_seconds = samples / sample_rate if sample_rate else float("nan")
        rows.append(
            {
                "modality": name,
                "shape": shape,
                "samples": samples,
                "channels": channels,
                "sample_rate_hz": sample_rate,
                "duration_seconds": duration_seconds,
            }
        )
    return rows


def summarize_labels(labels: np.ndarray) -> list[dict[str, Any]]:
    values, counts = np.unique(labels.astype(int), return_counts=True)
    rows: list[dict[str, Any]] = []
    for value, count in zip(values.tolist(), counts.tolist()):
        rows.append(
            {
                "label_id": int(value),
                "label_name": LABEL_NAME_MAP.get(int(value), f"unknown_{int(value)}"),
                "samples": int(count),
                "duration_seconds_at_700hz": float(count) / 700.0,
            }
        )
    return rows


def load_questionnaire(subject_dir: Path) -> pd.DataFrame | None:
    quest_path = subject_dir / f"{subject_dir.name}_quest.csv"
    if not quest_path.exists():
        return None
    try:
        return pd.read_csv(quest_path, sep=";", header=None, engine="python")
    except Exception:
        try:
            return pd.read_csv(quest_path, header=None)
        except Exception:
            return None


def format_seconds(seconds: float) -> str:
    if not np.isfinite(seconds):
        return "nan"
    minutes = seconds / 60.0
    return f"{seconds:.1f}s ({minutes:.1f} min)"


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a single WESAD subject and print a compact data report.")
    parser.add_argument("--wesad-root", type=Path, required=True)
    parser.add_argument("--subject", type=str, default="S2")
    args = parser.parse_args()

    subject_dir = args.wesad_root / args.subject
    if not subject_dir.exists():
        raise FileNotFoundError(f"Could not find subject directory: {subject_dir}")

    payload = load_subject_pickle(subject_dir)
    subject_id = str(payload.get("subject", args.subject))
    labels = np.asarray(payload["label"])
    signal = payload["signal"]
    chest = signal["chest"]
    wrist = signal["wrist"]

    chest_rows = summarize_signal_dict(chest, DEFAULT_SAMPLE_RATES["chest"])
    wrist_rows = summarize_signal_dict(wrist, DEFAULT_SAMPLE_RATES["wrist"])
    label_rows = summarize_labels(labels)
    questionnaire = load_questionnaire(subject_dir)

    print("WESAD Subject Data Report")
    print(f"subject={subject_id}")
    print(f"subject_dir={subject_dir}")
    print(f"files={[path.name for path in sorted(subject_dir.iterdir())]}")
    print("")

    print("Chest modalities")
    for row in chest_rows:
        print(
            f"{row['modality']}: shape={row['shape']} "
            f"rate={row['sample_rate_hz']}Hz "
            f"duration={format_seconds(row['duration_seconds'])}"
        )
    print("")

    print("Wrist modalities")
    for row in wrist_rows:
        print(
            f"{row['modality']}: shape={row['shape']} "
            f"rate={row['sample_rate_hz']}Hz "
            f"duration={format_seconds(row['duration_seconds'])}"
        )
    print("")

    print("Labels")
    for row in label_rows:
        print(
            f"label={row['label_id']} ({row['label_name']}): "
            f"samples={row['samples']} "
            f"duration={format_seconds(row['duration_seconds_at_700hz'])}"
        )
    print("")

    core_binary_ids = {1, 2}
    core_binary_counts = Counter(labels.astype(int).tolist())
    baseline_count = core_binary_counts.get(1, 0)
    stress_count = core_binary_counts.get(2, 0)
    core_total = baseline_count + stress_count
    if core_total > 0:
        print("Candidate binary task: baseline vs stress")
        print(f"baseline_samples={baseline_count} duration={format_seconds(baseline_count / 700.0)}")
        print(f"stress_samples={stress_count} duration={format_seconds(stress_count / 700.0)}")
        print(f"baseline_fraction={baseline_count / core_total:.4f}")
        print(f"stress_fraction={stress_count / core_total:.4f}")
    else:
        print("Candidate binary task: baseline vs stress")
        print("No baseline/stress labels found.")
    print("")

    print("Proposed modality split")
    print("deployable_branch=wrist BVP + wrist ACC")
    print("privileged_branch=chest ACC + ECG + EMG + EDA + Temp + Resp")
    print("optional_extra_wrist=wrist EDA + wrist TEMP")
    print("")

    print("Notes")
    print("- WESAD labels live inside the subject pickle file, not just in the CSV files.")
    print("- Wrist and chest streams have different sample rates, so window alignment must be handled explicitly.")
    print("- A first clean benchmark should use label 1 (baseline) vs label 2 (stress).")
    if questionnaire is not None:
        preview = questionnaire.head(8).fillna("").astype(str)
        print("")
        print("Questionnaire preview")
        for row in preview.itertuples(index=False):
            print(" | ".join(row))


if __name__ == "__main__":
    main()
