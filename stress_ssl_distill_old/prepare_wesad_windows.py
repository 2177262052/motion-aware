from __future__ import annotations

import argparse
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


CHEST_SAMPLE_RATE = 700

LABEL_TO_SESSION = {
    1: "baseline",
    2: "stress",
}

LABEL_TO_BINARY = {
    1: 0,
    2: 1,
}

SESSION_TO_GROUP = {
    "baseline": "calm",
    "stress": "stress",
}


@dataclass
class WESADWindowRecord:
    subject_id: str
    subject_index: int
    session: str
    group_name: str
    label: int
    wesad_label_id: int
    window_start_s: float
    window_end_s: float
    window_start_ms: int
    window_end_ms: int
    window_start_index_700hz: int
    window_end_index_700hz: int
    window_duration_s: float
    stride_s: float
    subject_pkl_path: str


def load_subject_pickle(subject_dir: Path) -> dict[str, Any]:
    pkl_path = subject_dir / f"{subject_dir.name}.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"Missing subject pickle: {pkl_path}")
    with pkl_path.open("rb") as handle:
        return pickle.load(handle, encoding="latin1")


def build_window_records_for_subject(
    subject_dir: Path,
    subject_index: int,
    window_s: float,
    stride_s: float,
) -> list[WESADWindowRecord]:
    payload = load_subject_pickle(subject_dir)
    labels = np.asarray(payload["label"]).reshape(-1).astype(int)

    window_samples = int(round(window_s * CHEST_SAMPLE_RATE))
    stride_samples = int(round(stride_s * CHEST_SAMPLE_RATE))
    subject_id = str(payload.get("subject", subject_dir.name))
    rel_pkl_path = str((subject_dir / f"{subject_dir.name}.pkl").relative_to(subject_dir.parent.parent))

    records: list[WESADWindowRecord] = []
    if labels.size == 0:
        return records

    segment_start = 0
    prev = int(labels[0])
    for idx in range(1, len(labels) + 1):
        is_boundary = idx == len(labels) or int(labels[idx]) != prev
        if not is_boundary:
            continue

        if prev in LABEL_TO_SESSION:
            segment_len = idx - segment_start
            offset = 0
            while offset + window_samples <= segment_len:
                start_idx = segment_start + offset
                end_idx = start_idx + window_samples
                session = LABEL_TO_SESSION[prev]
                records.append(
                    WESADWindowRecord(
                        subject_id=subject_id,
                        subject_index=subject_index,
                        session=session,
                        group_name=SESSION_TO_GROUP[session],
                        label=LABEL_TO_BINARY[prev],
                        wesad_label_id=prev,
                        window_start_s=start_idx / CHEST_SAMPLE_RATE,
                        window_end_s=end_idx / CHEST_SAMPLE_RATE,
                        window_start_ms=int(round(start_idx * 1000.0 / CHEST_SAMPLE_RATE)),
                        window_end_ms=int(round(end_idx * 1000.0 / CHEST_SAMPLE_RATE)),
                        window_start_index_700hz=start_idx,
                        window_end_index_700hz=end_idx,
                        window_duration_s=window_s,
                        stride_s=stride_s,
                        subject_pkl_path=rel_pkl_path,
                    )
                )
                offset += stride_samples

        if idx < len(labels):
            segment_start = idx
            prev = int(labels[idx])

    return records


def build_wesad_windows(
    wesad_root: Path,
    output_csv: Path,
    window_s: float,
    stride_s: float,
    show_progress: bool = True,
) -> pd.DataFrame:
    subject_dirs = sorted(path for path in wesad_root.glob("S*") if path.is_dir())
    if not subject_dirs:
        raise FileNotFoundError(f"No WESAD subject directories found under: {wesad_root}")

    subject_index_map = {subject_dir.name: idx for idx, subject_dir in enumerate(subject_dirs)}
    rows: list[WESADWindowRecord] = []
    iterator = tqdm(subject_dirs, desc="wesad windows", unit="subject", disable=not show_progress)
    for subject_dir in iterator:
        iterator.set_postfix_str(subject_dir.name)
        rows.extend(
            build_window_records_for_subject(
                subject_dir=subject_dir,
                subject_index=subject_index_map[subject_dir.name],
                window_s=window_s,
                stride_s=stride_s,
            )
        )

    df = pd.DataFrame([asdict(row) for row in rows])
    if not df.empty:
        df = df.sort_values(["subject_index", "window_start_index_700hz"]).reset_index(drop=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a base window manifest for WESAD.")
    parser.add_argument("--wesad-root", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--window-s", type=float, default=20.0)
    parser.add_argument("--stride-s", type=float, default=10.0)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    df = build_wesad_windows(
        wesad_root=args.wesad_root,
        output_csv=args.output_csv,
        window_s=args.window_s,
        stride_s=args.stride_s,
        show_progress=not args.no_progress,
    )
    if df.empty:
        print("No WESAD windows were generated.")
        return

    print(df.groupby(["label", "session"]).size())
    print(f"subjects={df['subject_id'].nunique()}")
    print(f"windows={len(df)}")
    print(f"Saved WESAD base windows to {args.output_csv}")


if __name__ == "__main__":
    main()
