from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .galaxy_manifest import build_manifest


def sorted_subject_ids(dataset_root: Path) -> list[str]:
    return sorted(path.name for path in dataset_root.iterdir() if path.is_dir() and path.name.startswith("P"))


def choose_val_subjects(subject_ids: list[str], held_out_subject: str, val_count: int, offset: int) -> list[str]:
    if held_out_subject not in subject_ids:
        raise ValueError(f"Held-out subject {held_out_subject} not found in dataset.")
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


def is_valid_fold(
    df: pd.DataFrame,
    calm_sessions: list[str],
    stress_sessions: list[str],
) -> tuple[bool, dict[str, int]]:
    filtered = df[df["session"].isin(set(calm_sessions) | set(stress_sessions))].copy()
    counts = {
        split: int((filtered["split"] == split).sum())
        for split in ("train", "val", "test")
    }
    for split in ("train", "val", "test"):
        split_df = filtered[filtered["split"] == split]
        labels = set(split_df["label"].tolist())
        if labels != {0, 1}:
            return False, counts
    return True, counts


def build_all_loso_manifests(
    dataset_root: Path,
    output_dir: Path,
    window_s: float,
    stride_s: float,
    val_count: int,
    val_offset: int,
    calm_sessions: list[str],
    stress_sessions: list[str],
) -> None:
    subject_ids = sorted_subject_ids(dataset_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    valid_subjects: list[str] = []
    invalid_subjects: list[str] = []

    for held_out_subject in subject_ids:
        val_subjects = choose_val_subjects(subject_ids, held_out_subject, val_count=val_count, offset=val_offset)
        output_csv = output_dir / f"galaxy_{held_out_subject}_loso_val.csv"
        df = build_manifest(
            dataset_root=dataset_root,
            output_csv=output_csv,
            held_out_subject=held_out_subject,
            val_subjects=val_subjects,
            window_s=window_s,
            stride_s=stride_s,
        )
        is_valid, counts = is_valid_fold(df, calm_sessions=calm_sessions, stress_sessions=stress_sessions)
        if not is_valid:
            if output_csv.exists():
                output_csv.unlink()
            invalid_subjects.append(held_out_subject)
            print(
                f"{held_out_subject}: skipped "
                f"(invalid fold for sessions calm={calm_sessions}, stress={stress_sessions}; "
                f"counts={counts})"
            )
            continue
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
    parser = argparse.ArgumentParser(description="Build LOSO + validation manifests for GalaxyPPG.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--window-s", type=float, default=30.0)
    parser.add_argument("--stride-s", type=float, default=30.0)
    parser.add_argument("--val-count", type=int, default=2)
    parser.add_argument("--val-offset", type=int, default=1)
    parser.add_argument("--calm-sessions", nargs="*", default=["baseline"])
    parser.add_argument("--stress-sessions", nargs="*", default=["tsst-prep"])
    args = parser.parse_args()

    build_all_loso_manifests(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        window_s=args.window_s,
        stride_s=args.stride_s,
        val_count=args.val_count,
        val_offset=args.val_offset,
        calm_sessions=[str(item) for item in args.calm_sessions],
        stress_sessions=[str(item) for item in args.stress_sessions],
    )


if __name__ == "__main__":
    main()
