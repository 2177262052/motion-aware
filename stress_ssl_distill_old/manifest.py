from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .dataset import parse_e4_signal, parse_tags
from .protocols import canonical_stage_to_score_key, expected_segments, protocol_version_from_subject


DEFAULT_BAD_SUBJECTS = {"f07"}


@dataclass
class WindowRecord:
    subject_id: str
    subject_index: int
    protocol_version: str
    stage: str
    label: int
    stress_score: float
    split: str
    segment_start_s: float
    segment_end_s: float
    window_start_s: float
    window_duration_s: float
    stride_s: float
    bvp_path: str
    acc_path: str


def subject_splits(subject_ids: List[str], held_out_subject: str) -> Dict[str, str]:
    return {
        sid: ("test" if sid == held_out_subject else "train")
        for sid in subject_ids
    }


def build_manifest(
    dataset_root: Path,
    output_csv: Path,
    held_out_subject: str,
    window_s: float = 30.0,
    stride_s: float = 10.0,
) -> pd.DataFrame:
    stress_root = dataset_root / "Wearable_Dataset" / "STRESS"
    subject_info = pd.read_csv(dataset_root / "subject-info.csv")
    stress_v1 = pd.read_csv(dataset_root / "Stress_Level_v1.csv", index_col=0)
    stress_v2 = pd.read_csv(dataset_root / "Stress_Level_v2.csv", index_col=0)

    all_subject_ids = sorted(
        [
            path.name
            for path in stress_root.iterdir()
            if path.is_dir() and path.name not in DEFAULT_BAD_SUBJECTS
        ]
    )
    splits = subject_splits(all_subject_ids, held_out_subject)
    subject_index = {sid: idx for idx, sid in enumerate(all_subject_ids)}

    records: List[WindowRecord] = []
    for subject_id in all_subject_ids:
        subject_dir = stress_root / subject_id
        bvp_path = subject_dir / "BVP.csv"
        acc_path = subject_dir / "ACC.csv"
        tags_path = subject_dir / "tags.csv"
        if not (bvp_path.exists() and acc_path.exists() and tags_path.exists()):
            continue

        _, _, bvp_values = parse_e4_signal(bvp_path, "BVP")
        _, _, acc_values = parse_e4_signal(acc_path, "ACC")
        tags = parse_tags(tags_path)
        protocol_version = protocol_version_from_subject(subject_id)
        segments = expected_segments(subject_id)
        score_map = canonical_stage_to_score_key(protocol_version)
        score_table = stress_v1 if protocol_version == "V1" else stress_v2
        if subject_id not in score_table.index:
            continue

        session_start = tags[0]
        for segment in segments:
            if segment.end_idx >= len(tags):
                continue
            seg_start = (tags[segment.start_idx] - session_start).total_seconds()
            seg_end = (tags[segment.end_idx] - session_start).total_seconds()
            duration = seg_end - seg_start
            if duration < window_s or not segment.keep_for_training:
                continue
            score_key = score_map[segment.name]
            stress_score = float(score_table.loc[subject_id, score_key])

            offset = 0.0
            while offset + window_s <= duration + 1e-6:
                records.append(
                    WindowRecord(
                        subject_id=subject_id,
                        subject_index=subject_index[subject_id],
                        protocol_version=protocol_version,
                        stage=segment.name,
                        label=int(segment.is_stress),
                        stress_score=stress_score,
                        split=splits[subject_id],
                        segment_start_s=seg_start,
                        segment_end_s=seg_end,
                        window_start_s=offset,
                        window_duration_s=window_s,
                        stride_s=stride_s,
                        bvp_path=str(bvp_path.relative_to(dataset_root)),
                        acc_path=str(acc_path.relative_to(dataset_root)),
                    )
                )
                offset += stride_s

    df = pd.DataFrame([asdict(record) for record in records])
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a conservative window-level STRESS manifest.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--held-out-subject", type=str, required=True)
    parser.add_argument("--window-s", type=float, default=30.0)
    parser.add_argument("--stride-s", type=float, default=10.0)
    args = parser.parse_args()

    df = build_manifest(
        dataset_root=args.dataset_root,
        output_csv=args.output_csv,
        held_out_subject=args.held_out_subject,
        window_s=args.window_s,
        stride_s=args.stride_s,
    )
    print(df.groupby(["split", "label"]).size())
    print(f"Saved manifest to {args.output_csv}")


if __name__ == "__main__":
    main()
