from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

from .galaxy_protocols import pair_event_intervals, session_to_group, session_to_label


@dataclass
class GalaxyWindowRecord:
    subject_id: str
    subject_index: int
    split: str
    session: str
    group_name: str
    label: int
    window_start_ms: int
    window_end_ms: int
    window_duration_s: float
    stride_s: float
    tsst_score: float
    ssst_score: float
    gw_ppg_path: str
    gw_acc_path: str
    gw_hr_path: str
    e4_bvp_path: str
    e4_acc_path: str
    polar_ecg_path: str
    polar_acc_path: str


def subject_splits(subject_ids: List[str], held_out_subject: str, val_subjects: Optional[Sequence[str]]) -> Dict[str, str]:
    splits: Dict[str, str] = {}
    val_set = set(val_subjects or [])
    for subject_id in subject_ids:
        if subject_id == held_out_subject:
            splits[subject_id] = "test"
        elif subject_id in val_set:
            splits[subject_id] = "val"
        else:
            splits[subject_id] = "train"
    return splits


def _safe_score(value: object) -> float:
    if pd.isna(value) or value == "-":
        return float("nan")
    return float(value)


def build_manifest(
    dataset_root: Path,
    output_csv: Path,
    held_out_subject: str,
    val_subject: Optional[str] = None,
    val_subjects: Optional[Sequence[str]] = None,
    window_s: float = 30.0,
    stride_s: float = 10.0,
    min_coverage: float = 0.95,
) -> pd.DataFrame:
    meta = pd.read_csv(dataset_root / "Meta.csv")
    meta = meta.set_index("UID")

    all_subject_ids = sorted(path.name for path in dataset_root.iterdir() if path.is_dir() and path.name.startswith("P"))
    subject_index = {subject_id: idx for idx, subject_id in enumerate(all_subject_ids)}
    resolved_val_subjects: list[str] = []
    if val_subjects is not None:
        resolved_val_subjects.extend(str(item) for item in val_subjects if str(item))
    elif val_subject is not None:
        resolved_val_subjects.append(val_subject)
    resolved_val_subjects = [subject_id for subject_id in resolved_val_subjects if subject_id != held_out_subject]
    splits = subject_splits(all_subject_ids, held_out_subject=held_out_subject, val_subjects=resolved_val_subjects)

    records: List[GalaxyWindowRecord] = []
    for subject_id in all_subject_ids:
        subject_dir = dataset_root / subject_id
        event_path = subject_dir / "Event.csv"
        gw_ppg_path = subject_dir / "GalaxyWatch" / "PPG.csv"
        gw_acc_path = subject_dir / "GalaxyWatch" / "ACC.csv"
        gw_hr_path = subject_dir / "GalaxyWatch" / "HR.csv"
        e4_bvp_path = subject_dir / "E4" / "BVP.csv"
        e4_acc_path = subject_dir / "E4" / "ACC.csv"
        polar_ecg_path = subject_dir / "PolarH10" / "ECG.csv"
        polar_acc_path = subject_dir / "PolarH10" / "ACC.csv"

        required = [event_path, gw_ppg_path, gw_acc_path, e4_bvp_path, e4_acc_path, polar_ecg_path, polar_acc_path]
        if not all(path.exists() for path in required):
            continue

        events = pd.read_csv(event_path)
        intervals = pair_event_intervals(events.to_dict("records"))
        if subject_id not in meta.index:
            continue

        tsst_score = _safe_score(meta.loc[subject_id, "TSST"])
        ssst_score = _safe_score(meta.loc[subject_id, "SSST"])

        for interval in intervals:
            label = session_to_label(interval.session)
            group_name = session_to_group(interval.session)
            if label is None:
                continue

            duration_ms = interval.duration_ms
            window_ms = int(round(window_s * 1000))
            stride_ms = int(round(stride_s * 1000))
            if duration_ms < int(round(window_ms * min_coverage)):
                continue

            offset_ms = 0
            while offset_ms + window_ms <= duration_ms:
                window_start_ms = interval.start_ms + offset_ms
                window_end_ms = window_start_ms + window_ms
                records.append(
                    GalaxyWindowRecord(
                        subject_id=subject_id,
                        subject_index=subject_index[subject_id],
                        split=splits[subject_id],
                        session=interval.session,
                        group_name=group_name,
                        label=label,
                        window_start_ms=window_start_ms,
                        window_end_ms=window_end_ms,
                        window_duration_s=window_s,
                        stride_s=stride_s,
                        tsst_score=tsst_score,
                        ssst_score=ssst_score,
                        gw_ppg_path=str(gw_ppg_path.relative_to(dataset_root)),
                        gw_acc_path=str(gw_acc_path.relative_to(dataset_root)),
                        gw_hr_path=str(gw_hr_path.relative_to(dataset_root)),
                        e4_bvp_path=str(e4_bvp_path.relative_to(dataset_root)),
                        e4_acc_path=str(e4_acc_path.relative_to(dataset_root)),
                        polar_ecg_path=str(polar_ecg_path.relative_to(dataset_root)),
                        polar_acc_path=str(polar_acc_path.relative_to(dataset_root)),
                    )
                )
                offset_ms += stride_ms

    df = pd.DataFrame([asdict(record) for record in records])
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a conservative LOSO manifest for GalaxyPPG.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--held-out-subject", type=str, required=True)
    parser.add_argument("--val-subject", type=str, default=None)
    parser.add_argument("--val-subjects", nargs="*", default=None)
    parser.add_argument("--window-s", type=float, default=30.0)
    parser.add_argument("--stride-s", type=float, default=10.0)
    args = parser.parse_args()

    df = build_manifest(
        dataset_root=args.dataset_root,
        output_csv=args.output_csv,
        held_out_subject=args.held_out_subject,
        val_subject=args.val_subject,
        val_subjects=args.val_subjects,
        window_s=args.window_s,
        stride_s=args.stride_s,
    )
    if df.empty:
        print("No records were generated.")
        return
    print(df.groupby(["split", "label", "session"]).size())
    print(f"Saved GalaxyPPG manifest to {args.output_csv}")


if __name__ == "__main__":
    main()
