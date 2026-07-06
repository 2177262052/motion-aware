from __future__ import annotations

import argparse
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm


CHEST_SAMPLE_RATE = 700
WRIST_SAMPLE_RATES = {
    "ACC": 32,
    "BVP": 64,
    "EDA": 4,
    "TEMP": 4,
}

LABEL_NAME_MAP = {
    0: "undefined_or_transition",
    1: "baseline",
    2: "stress",
    3: "amusement",
    4: "meditation",
    5: "unknown_extra_5",
    6: "unknown_extra_6",
    7: "unknown_extra_7",
}

EXPECTED_CHEST_MODALITIES = ("ACC", "ECG", "EMG", "EDA", "Temp", "Resp")
EXPECTED_WRIST_MODALITIES = ("ACC", "BVP", "EDA", "TEMP")


def load_subject_pickle(subject_dir: Path) -> dict[str, Any]:
    pkl_path = subject_dir / f"{subject_dir.name}.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"Missing subject pickle: {pkl_path}")
    with pkl_path.open("rb") as handle:
        return pickle.load(handle, encoding="latin1")


def format_hours(seconds: float) -> str:
    return f"{seconds / 3600.0:.3f}h"


def contiguous_label_durations(labels: np.ndarray, target_ids: set[int]) -> tuple[Counter, Counter]:
    sample_counts: Counter = Counter()
    window_counts: Counter = Counter()
    if labels.size == 0:
        return sample_counts, window_counts

    win = 30.0
    stride = 10.0

    start = 0
    prev = int(labels[0])
    for idx in range(1, len(labels)):
        cur = int(labels[idx])
        if cur != prev:
            if prev in target_ids:
                count = idx - start
                sample_counts[prev] += count
                duration_s = count / CHEST_SAMPLE_RATE
                if duration_s >= win:
                    window_counts[prev] += int((duration_s - win) // stride + 1)
            start = idx
            prev = cur

    if prev in target_ids:
        count = len(labels) - start
        sample_counts[prev] += count
        duration_s = count / CHEST_SAMPLE_RATE
        if duration_s >= win:
            window_counts[prev] += int((duration_s - win) // stride + 1)

    return sample_counts, window_counts


def summarize_subject(subject_dir: Path) -> dict[str, Any]:
    payload = load_subject_pickle(subject_dir)
    signal = payload["signal"]
    labels = np.asarray(payload["label"]).reshape(-1).astype(int)

    chest = signal["chest"]
    wrist = signal["wrist"]
    chest_keys = sorted(chest.keys())
    wrist_keys = sorted(wrist.keys())

    label_counts = Counter(labels.tolist())
    core_counts, core_window_counts = contiguous_label_durations(labels, {1, 2})

    baseline_seconds = core_counts.get(1, 0) / CHEST_SAMPLE_RATE
    stress_seconds = core_counts.get(2, 0) / CHEST_SAMPLE_RATE
    total_seconds = len(labels) / CHEST_SAMPLE_RATE

    wrist_duration_seconds = {}
    for key, rate in WRIST_SAMPLE_RATES.items():
        if key in wrist:
            wrist_duration_seconds[key] = float(np.asarray(wrist[key]).shape[0]) / rate

    return {
        "subject": str(payload.get("subject", subject_dir.name)),
        "subject_dir": str(subject_dir),
        "total_seconds": total_seconds,
        "baseline_seconds": baseline_seconds,
        "stress_seconds": stress_seconds,
        "baseline_windows_30_10": int(core_window_counts.get(1, 0)),
        "stress_windows_30_10": int(core_window_counts.get(2, 0)),
        "label_counts": dict(label_counts),
        "chest_keys": chest_keys,
        "wrist_keys": wrist_keys,
        "missing_chest": [key for key in EXPECTED_CHEST_MODALITIES if key not in chest_keys],
        "missing_wrist": [key for key in EXPECTED_WRIST_MODALITIES if key not in wrist_keys],
        "wrist_duration_seconds": wrist_duration_seconds,
    }


def build_report(subject_rows: list[dict[str, Any]], wesad_root: Path) -> str:
    total_label_counts: Counter = Counter()
    total_baseline_seconds = 0.0
    total_stress_seconds = 0.0
    total_seconds = 0.0
    total_baseline_windows = 0
    total_stress_windows = 0
    missing_chest_subjects: dict[str, list[str]] = {}
    missing_wrist_subjects: dict[str, list[str]] = {}

    for row in subject_rows:
        total_label_counts.update(row["label_counts"])
        total_baseline_seconds += row["baseline_seconds"]
        total_stress_seconds += row["stress_seconds"]
        total_seconds += row["total_seconds"]
        total_baseline_windows += row["baseline_windows_30_10"]
        total_stress_windows += row["stress_windows_30_10"]
        if row["missing_chest"]:
            missing_chest_subjects[row["subject"]] = row["missing_chest"]
        if row["missing_wrist"]:
            missing_wrist_subjects[row["subject"]] = row["missing_wrist"]

    lines: list[str] = []
    lines.append("# WESAD Data Report")
    lines.append("")
    lines.append("## Dataset Overview")
    lines.append(f"- root: `{wesad_root}`")
    lines.append(f"- subjects: `{len(subject_rows)}`")
    lines.append(f"- total_duration: `{format_hours(total_seconds)}`")
    lines.append(f"- baseline_duration: `{format_hours(total_baseline_seconds)}`")
    lines.append(f"- stress_duration: `{format_hours(total_stress_seconds)}`")
    lines.append(f"- baseline_plus_stress_duration: `{format_hours(total_baseline_seconds + total_stress_seconds)}`")
    lines.append(f"- baseline_windows_30s_stride10s: `{total_baseline_windows}`")
    lines.append(f"- stress_windows_30s_stride10s: `{total_stress_windows}`")
    lines.append(f"- baseline_plus_stress_windows_30s_stride10s: `{total_baseline_windows + total_stress_windows}`")
    lines.append("")
    lines.append("## Label Inventory")
    for label_id in sorted(total_label_counts):
        name = LABEL_NAME_MAP.get(label_id, f"unknown_{label_id}")
        seconds = total_label_counts[label_id] / CHEST_SAMPLE_RATE
        lines.append(
            f"- label `{label_id}` ({name}): samples=`{total_label_counts[label_id]}` duration=`{format_hours(seconds)}`"
        )
    lines.append("")
    lines.append("## Proposed First Benchmark")
    lines.append("- task: `baseline (1)` vs `stress (2)`")
    lines.append("- split: `subject-wise LOSO`")
    lines.append("- deployable branch: `wrist BVP + wrist ACC`")
    lines.append("- privileged branch: `chest ACC + ECG + EMG + EDA + Temp + Resp`")
    lines.append("- optional extra wrist branch later: `wrist EDA + wrist TEMP`")
    lines.append("")
    lines.append("## Modality Completeness")
    if not missing_chest_subjects and not missing_wrist_subjects:
        lines.append("- all subjects contain the expected chest and wrist modalities")
    else:
        if missing_chest_subjects:
            lines.append("- missing chest modalities:")
            for subject, keys in sorted(missing_chest_subjects.items()):
                lines.append(f"  - {subject}: {', '.join(keys)}")
        if missing_wrist_subjects:
            lines.append("- missing wrist modalities:")
            for subject, keys in sorted(missing_wrist_subjects.items()):
                lines.append(f"  - {subject}: {', '.join(keys)}")
    lines.append("")
    lines.append("## Per-Subject Summary")
    for row in sorted(subject_rows, key=lambda item: item["subject"]):
        lines.append(
            "- "
            f"{row['subject']}: total=`{format_hours(row['total_seconds'])}`, "
            f"baseline=`{format_hours(row['baseline_seconds'])}`, "
            f"stress=`{format_hours(row['stress_seconds'])}`, "
            f"windows=`{row['baseline_windows_30_10'] + row['stress_windows_30_10']}`"
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("- WESAD labels come from the subject `pkl` files, not only from the E4 CSV archive.")
    lines.append("- Chest labels are sampled at `700 Hz`; wrist modalities use their own rates and must be synchronized during window extraction.")
    lines.append("- A first clean comparison with GalaxyPPG should use calm-vs-stress binary classification rather than multi-class WESAD labels.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan the full WESAD dataset and print a compact data report.")
    parser.add_argument("--wesad-root", type=Path, required=True)
    parser.add_argument("--save-report", type=Path, default=None)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    subject_dirs = sorted(path for path in args.wesad_root.glob("S*") if path.is_dir())
    if not subject_dirs:
        raise FileNotFoundError(f"No WESAD subject directories found under: {args.wesad_root}")

    subject_rows: list[dict[str, Any]] = []
    iterator = tqdm(
        subject_dirs,
        desc="wesad subjects",
        unit="subject",
        disable=args.no_progress,
    )
    for subject_dir in iterator:
        iterator.set_postfix_str(subject_dir.name)
        subject_rows.append(summarize_subject(subject_dir))
    report = build_report(subject_rows, args.wesad_root)
    print(report)

    if args.save_report is not None:
        args.save_report.parent.mkdir(parents=True, exist_ok=True)
        args.save_report.write_text(report + "\n", encoding="utf-8")
        print(f"\nSaved report to {args.save_report}")


if __name__ == "__main__":
    main()
