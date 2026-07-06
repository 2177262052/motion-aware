from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd

from .galaxy_manifest import build_manifest


def parse_subjects(values: Sequence[str] | None) -> list[str]:
    if values is None:
        return []
    return [str(item).strip() for item in values if str(item).strip()]


def build_fixed_split_manifest(
    dataset_root: Path,
    output_csv: Path,
    train_subjects: Sequence[str],
    test_subjects: Sequence[str],
    window_s: float = 30.0,
    stride_s: float = 30.0,
) -> pd.DataFrame:
    train_subjects = list(train_subjects)
    test_subjects = list(test_subjects)
    if not train_subjects or not test_subjects:
        raise ValueError("train_subjects and test_subjects must both be non-empty.")

    frames: list[pd.DataFrame] = []
    for subject_id in test_subjects:
        df = build_manifest(
            dataset_root=dataset_root,
            output_csv=output_csv,
            held_out_subject=subject_id,
            val_subject=None,
            window_s=window_s,
            stride_s=stride_s,
        )
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)
    keep_subjects = set(train_subjects) | set(test_subjects)
    merged = merged[merged["subject_id"].isin(keep_subjects)].copy()
    merged["split"] = merged["subject_id"].map(lambda sid: "test" if sid in test_subjects else "train")
    merged = merged.sort_values(["split", "subject_index", "window_start_ms"]).reset_index(drop=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a fixed train/test subject split manifest for GalaxyPPG.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--train-subjects", nargs="+", required=True)
    parser.add_argument("--test-subjects", nargs="+", required=True)
    parser.add_argument("--window-s", type=float, default=30.0)
    parser.add_argument("--stride-s", type=float, default=30.0)
    args = parser.parse_args()

    df = build_fixed_split_manifest(
        dataset_root=args.dataset_root,
        output_csv=args.output_csv,
        train_subjects=parse_subjects(args.train_subjects),
        test_subjects=parse_subjects(args.test_subjects),
        window_s=args.window_s,
        stride_s=args.stride_s,
    )
    if df.empty:
        print("No records were generated.")
        return
    print("train_subjects=", parse_subjects(args.train_subjects))
    print("test_subjects=", parse_subjects(args.test_subjects))
    print(df.groupby(["split", "label", "session"]).size())
    print(f"Saved GalaxyPPG split manifest to {args.output_csv}")


if __name__ == "__main__":
    main()
