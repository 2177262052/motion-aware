from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DATASET_SENSOR_TEXT = {
    "galaxy": {
        "deployable_sensors": "Galaxy Watch PPG + ACC",
        "privileged_sensors": "E4 BVP/ACC + Polar H10 ECG/ACC",
        "sessions": "baseline vs tsst-prep",
        "include_sessions": ["baseline", "tsst-prep"],
    },
    "wesad": {
        "deployable_sensors": "Wrist BVP + ACC",
        "privileged_sensors": "Chest ECG/EDA/Resp/EMG/Temp/ACC",
        "sessions": "baseline vs stress",
        "include_sessions": ["baseline", "stress"],
    },
}


def parse_named_dir(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"Expected NAME=DIR, got: {value}")
    name, path = value.split("=", 1)
    name = name.strip().lower()
    if not name:
        raise ValueError(f"Missing dataset name in: {value}")
    return name, Path(path)


def unique_window_frame(df: pd.DataFrame) -> pd.DataFrame:
    required = ["subject_id", "session", "label"]
    for column in required:
        if column not in df.columns:
            raise ValueError(f"Manifest is missing required column: {column}")

    key_columns = ["subject_id", "session", "label"]
    for column in ("window_start_ms", "window_end_ms"):
        if column in df.columns:
            key_columns.append(column)
    if "window_start_ms" not in df.columns and "window_start_s" in df.columns:
        key_columns.append("window_start_s")
    if "window_end_ms" not in df.columns and "window_end_s" in df.columns:
        key_columns.append("window_end_s")

    return df.drop_duplicates(subset=key_columns).copy()


def summarize_dataset(name: str, manifests_dir: Path, use_default_session_filter: bool) -> dict[str, object]:
    manifest_paths = sorted(manifests_dir.glob("*.csv"))
    if not manifest_paths:
        raise ValueError(f"No manifest CSV files found in {manifests_dir}")

    frames = []
    for path in manifest_paths:
        frame = pd.read_csv(path)
        frame["source_manifest"] = path.name
        frames.append(frame)
    all_rows = pd.concat(frames, ignore_index=True)
    sensor_text = DATASET_SENSOR_TEXT.get(name, {})
    include_sessions = sensor_text.get("include_sessions", []) if use_default_session_filter else []
    if include_sessions:
        all_rows = all_rows[all_rows["session"].astype(str).isin([str(item) for item in include_sessions])].copy()
        if len(all_rows) == 0:
            raise ValueError(
                f"No rows remain for dataset={name} after filtering sessions={include_sessions}. "
                "Use --all-sessions if you want to summarize every manifest row."
            )
    unique_rows = unique_window_frame(all_rows)

    subjects = sorted(str(item) for item in unique_rows["subject_id"].dropna().unique())
    labels = pd.to_numeric(unique_rows["label"], errors="coerce")
    calm_windows = int((labels == 0).sum())
    stress_windows = int((labels == 1).sum())
    total_windows = int(len(unique_rows))
    stress_rate = float(stress_windows / total_windows) if total_windows else float("nan")

    window_duration = sorted(pd.to_numeric(unique_rows.get("window_duration_s"), errors="coerce").dropna().unique())
    stride = sorted(pd.to_numeric(unique_rows.get("stride_s"), errors="coerce").dropna().unique())
    sessions = sorted(str(item) for item in unique_rows["session"].dropna().unique())

    return {
        "dataset": name,
        "subjects": len(subjects),
        "loso_folds": len(manifest_paths),
        "unique_windows": total_windows,
        "calm_windows": calm_windows,
        "stress_windows": stress_windows,
        "stress_rate": stress_rate,
        "window_duration_s": ", ".join(f"{value:g}" for value in window_duration),
        "stride_s": ", ".join(f"{value:g}" for value in stride),
        "sessions_in_manifest": ", ".join(sessions),
        "paper_sessions": sensor_text.get("sessions", ""),
        "session_filter": ", ".join(str(item) for item in include_sessions) if include_sessions else "all",
        "deployable_sensors": sensor_text.get("deployable_sensors", ""),
        "privileged_sensors": sensor_text.get("privileged_sensors", ""),
        "manifest_dir": str(manifests_dir),
    }


def write_markdown(path: Path, summary: pd.DataFrame) -> None:
    columns = [
        "dataset",
        "subjects",
        "loso_folds",
        "unique_windows",
        "calm_windows",
        "stress_windows",
        "stress_rate",
        "window_duration_s",
        "stride_s",
        "deployable_sensors",
        "privileged_sensors",
        "paper_sessions",
        "session_filter",
    ]
    available = [column for column in columns if column in summary.columns]
    lines = ["# Dataset Summary", ""]
    lines.append("| " + " | ".join(available) + " |")
    lines.append("|" + "|".join(["---"] * len(available)) + "|")
    for row in summary[available].itertuples(index=False):
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    lines.append("")
    lines.append("Note: unique_windows are de-duplicated across LOSO manifest files.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize unique windows and sensors from LOSO manifests.")
    parser.add_argument("--dataset", action="append", required=True, help="NAME=MANIFEST_DIR. Repeat for each dataset.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--all-sessions",
        action="store_true",
        help="Summarize all manifest sessions instead of the default paper protocol sessions.",
    )
    args = parser.parse_args()

    rows = [
        summarize_dataset(name, path, use_default_session_filter=not args.all_sessions)
        for name, path in (parse_named_dir(item) for item in args.dataset)
    ]
    out = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "dataset_summary.csv"
    md_path = args.output_dir / "dataset_summary.md"
    out.to_csv(csv_path, index=False)
    write_markdown(md_path, out)
    print(out.to_string(index=False))
    print(f"Saved dataset summary CSV to {csv_path}")
    print(f"Saved dataset summary markdown to {md_path}")


if __name__ == "__main__":
    main()
