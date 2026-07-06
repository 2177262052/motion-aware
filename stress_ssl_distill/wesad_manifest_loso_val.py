from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def sorted_subject_ids(windows_df: pd.DataFrame) -> list[str]:
    return sorted(windows_df["subject_id"].dropna().astype(str).unique().tolist())


def choose_val_subjects(subject_ids: list[str], held_out_subject: str, val_count: int, offset: int) -> list[str]:
    if held_out_subject not in subject_ids:
        raise ValueError(f"Held-out subject {held_out_subject} not found in windows CSV.")
    if val_count <= 0:
        return []

    start_idx = subject_ids.index(held_out_subject)
    chosen: list[str] = []
    cursor = 1
    while len(chosen) < val_count and cursor < len(subject_ids):
        candidate = subject_ids[(start_idx + offset + cursor - 1) % len(subject_ids)]
        cursor += 1
        if candidate == held_out_subject or candidate in chosen:
            continue
        chosen.append(candidate)
    return chosen


def is_valid_fold(df: pd.DataFrame) -> tuple[bool, dict[str, int]]:
    counts = {
        split: int((df["split"] == split).sum())
        for split in ("train", "val", "test")
    }
    for split in ("train", "val", "test"):
        split_df = df[df["split"] == split]
        labels = set(split_df["label"].tolist())
        if labels != {0, 1}:
            return False, counts
    return True, counts


def build_all_loso_manifests(
    windows_csv: Path,
    output_dir: Path,
    val_count: int,
    val_offset: int,
) -> None:
    df = pd.read_csv(windows_csv)
    if df.empty:
        raise ValueError(f"No rows found in {windows_csv}")

    subject_ids = sorted_subject_ids(df)
    output_dir.mkdir(parents=True, exist_ok=True)
    valid_subjects: list[str] = []
    invalid_subjects: list[str] = []

    for held_out_subject in subject_ids:
        val_subjects = choose_val_subjects(subject_ids, held_out_subject, val_count=val_count, offset=val_offset)
        fold_df = df.copy()
        fold_df["split"] = "train"
        fold_df.loc[fold_df["subject_id"] == held_out_subject, "split"] = "test"
        if val_subjects:
            fold_df.loc[fold_df["subject_id"].isin(val_subjects), "split"] = "val"

        is_valid, counts = is_valid_fold(fold_df)
        output_csv = output_dir / f"wesad_{held_out_subject}_loso_val.csv"
        if not is_valid:
            invalid_subjects.append(held_out_subject)
            print(f"{held_out_subject}: skipped invalid fold counts={counts}")
            continue

        fold_df = fold_df.sort_values(["split", "subject_index", "window_start_index_700hz"]).reset_index(drop=True)
        fold_df.to_csv(output_csv, index=False)
        valid_subjects.append(held_out_subject)
        print(
            f"{held_out_subject}: "
            f"val_subjects={val_subjects} "
            f"train={counts['train']} "
            f"val={counts['val']} "
            f"test={counts['test']} "
            f"-> {output_csv}"
        )

    print(f"valid_subjects={valid_subjects}")
    print(f"invalid_subjects={invalid_subjects}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build LOSO + validation manifests for WESAD.")
    parser.add_argument("--windows-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--val-count", type=int, default=1)
    parser.add_argument("--val-offset", type=int, default=1)
    args = parser.parse_args()

    build_all_loso_manifests(
        windows_csv=args.windows_csv,
        output_dir=args.output_dir,
        val_count=args.val_count,
        val_offset=args.val_offset,
    )


if __name__ == "__main__":
    main()
