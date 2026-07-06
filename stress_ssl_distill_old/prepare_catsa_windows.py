from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm


CATSA_SAMPLE_RATES = {
    "BVP": 64.0,
    "ACC": 32.0,
    "EDA": 4.0,
    "TEMP": 4.0,
    "HR": 1.0,
}

SESSION_TO_LABEL = {
    "Baseline": 0,
    "Logic": 1,
    "Stroop": 1,
    "Sudoku": 1,
}

SESSION_TO_GROUP = {
    "Baseline": "calm",
    "Logic": "stress",
    "Stroop": "stress",
    "Sudoku": "stress",
}


@dataclass
class CATSAWindowRecord:
    subject_id: str
    subject_index: int
    session: str
    group_name: str
    label: int
    window_start_s: float
    window_end_s: float
    window_start_ms: int
    window_end_ms: int
    window_duration_s: float
    stride_s: float
    subject_session_path: str


def subject_sort_key(path: Path) -> tuple[int, str]:
    suffix = path.name.replace("Sub", "")
    try:
        return int(suffix), path.name
    except ValueError:
        return 10_000, path.name


def read_signal_length(csv_path: Path) -> int:
    if not csv_path.exists():
        return 0
    try:
        return max(len(pd.read_csv(csv_path)), 0)
    except Exception:
        return 0


def session_duration_s(session_dir: Path, modalities: list[str]) -> float:
    durations = []
    for modality in modalities:
        length = read_signal_length(session_dir / f"{modality}.csv")
        if length <= 0:
            return 0.0
        durations.append(length / CATSA_SAMPLE_RATES[modality])
    return float(min(durations)) if durations else 0.0


def build_window_records_for_session(
    catsa_root: Path,
    subject_dir: Path,
    subject_index: int,
    session: str,
    window_s: float,
    stride_s: float,
    required_modalities: list[str],
) -> list[CATSAWindowRecord]:
    session_dir = subject_dir / session
    if not session_dir.exists():
        return []
    duration_s = session_duration_s(session_dir, required_modalities)
    if duration_s < window_s:
        return []

    rel_session_path = str(session_dir.relative_to(catsa_root))
    label = SESSION_TO_LABEL[session]
    group_name = SESSION_TO_GROUP[session]
    records: list[CATSAWindowRecord] = []
    start_s = 0.0
    while start_s + window_s <= duration_s + 1e-6:
        end_s = start_s + window_s
        records.append(
            CATSAWindowRecord(
                subject_id=subject_dir.name,
                subject_index=subject_index,
                session=session,
                group_name=group_name,
                label=label,
                window_start_s=float(start_s),
                window_end_s=float(end_s),
                window_start_ms=int(round(start_s * 1000.0)),
                window_end_ms=int(round(end_s * 1000.0)),
                window_duration_s=float(window_s),
                stride_s=float(stride_s),
                subject_session_path=rel_session_path,
            )
        )
        start_s += stride_s
    return records


def build_catsa_windows(
    catsa_root: Path,
    output_csv: Path,
    window_s: float,
    stride_s: float,
    calm_sessions: list[str],
    stress_sessions: list[str],
    required_modalities: list[str],
    exclude_subjects: list[str] | None = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    excluded = {str(subject) for subject in exclude_subjects or []}
    subject_dirs = sorted(
        [path for path in catsa_root.glob("Sub*") if path.is_dir() and path.name not in excluded],
        key=subject_sort_key,
    )
    if not subject_dirs:
        raise FileNotFoundError(f"No CATSA subject directories found under: {catsa_root}")

    sessions = calm_sessions + stress_sessions
    unknown = [session for session in sessions if session not in SESSION_TO_LABEL]
    if unknown:
        raise ValueError(f"Unsupported CATSA session names: {unknown}")

    rows: list[CATSAWindowRecord] = []
    iterator = tqdm(subject_dirs, desc="catsa windows", unit="subject", disable=not show_progress)
    for subject_index, subject_dir in enumerate(iterator):
        iterator.set_postfix_str(subject_dir.name)
        for session in sessions:
            rows.extend(
                build_window_records_for_session(
                    catsa_root=catsa_root,
                    subject_dir=subject_dir,
                    subject_index=subject_index,
                    session=session,
                    window_s=window_s,
                    stride_s=stride_s,
                    required_modalities=required_modalities,
                )
            )

    df = pd.DataFrame([asdict(row) for row in rows])
    if not df.empty:
        df = df.sort_values(["subject_index", "session", "window_start_s"]).reset_index(drop=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a base window manifest for CATSA.")
    parser.add_argument("--catsa-root", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--window-s", type=float, default=20.0)
    parser.add_argument("--stride-s", type=float, default=10.0)
    parser.add_argument("--calm-sessions", nargs="*", default=["Baseline"])
    parser.add_argument("--stress-sessions", nargs="*", default=["Stroop", "Logic", "Sudoku"])
    parser.add_argument("--required-modalities", nargs="*", default=["BVP", "ACC", "EDA", "TEMP"])
    parser.add_argument("--exclude-subjects", nargs="*", default=None)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    df = build_catsa_windows(
        catsa_root=args.catsa_root,
        output_csv=args.output_csv,
        window_s=args.window_s,
        stride_s=args.stride_s,
        calm_sessions=[str(item) for item in args.calm_sessions],
        stress_sessions=[str(item) for item in args.stress_sessions],
        required_modalities=[str(item) for item in args.required_modalities],
        exclude_subjects=[str(item) for item in args.exclude_subjects or []],
        show_progress=not args.no_progress,
    )
    if df.empty:
        print("No CATSA windows were generated.")
        return

    print(df.groupby(["label", "session"]).size())
    print(f"subjects={df['subject_id'].nunique()}")
    print(f"windows={len(df)}")
    print(f"Saved CATSA base windows to {args.output_csv}")


if __name__ == "__main__":
    main()
