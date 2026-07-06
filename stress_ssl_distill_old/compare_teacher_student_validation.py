from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


METRIC_SPECS = [
    ("auroc", "AUROC"),
    ("balanced_acc", "BA"),
    ("f1", "F1"),
]


def load_summary(path: Path, name: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "subject" not in frame.columns:
        raise ValueError(f"{name} CSV must contain a subject column: {path}")
    required = [f"val_{metric}" for metric, _ in METRIC_SPECS]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} CSV is missing required columns {missing}: {path}")
    out = frame.copy()
    out["subject"] = out["subject"].astype(str)
    for column in required:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def bootstrap_ci(values: np.ndarray, repeats: int, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(repeats, len(values)), replace=True).mean(axis=1)
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def win_loss_tie(values: pd.Series, tie_eps: float) -> tuple[int, int, int]:
    values = pd.to_numeric(values, errors="coerce").dropna()
    wins = int((values > tie_eps).sum())
    losses = int((values < -tie_eps).sum())
    ties = int((values.abs() <= tie_eps).sum())
    return wins, losses, ties


def build_delta_table(
    teacher: pd.DataFrame,
    student: pd.DataFrame,
    student_name: str,
    tie_eps: float,
) -> pd.DataFrame:
    teacher_cols = ["subject"] + [f"val_{metric}" for metric, _ in METRIC_SPECS]
    student_cols = ["subject"] + [f"val_{metric}" for metric, _ in METRIC_SPECS]
    merged = teacher[teacher_cols].merge(
        student[student_cols],
        on="subject",
        how="inner",
        suffixes=("_teacher", "_student"),
    )
    if merged.empty:
        raise ValueError("No overlapping subjects/folds between teacher and student summaries.")

    rows: list[dict[str, object]] = []
    for _, row in merged.sort_values("subject").iterrows():
        item: dict[str, object] = {
            "fold": row["subject"],
            "student_name": student_name,
        }
        all_positive = True
        for metric, label in METRIC_SPECS:
            teacher_value = float(row[f"val_{metric}_teacher"])
            student_value = float(row[f"val_{metric}_student"])
            delta = teacher_value - student_value
            item[f"teacher_val_{metric}"] = teacher_value
            item[f"student_val_{metric}"] = student_value
            item[f"delta_{metric}"] = delta
            all_positive = all_positive and delta > tie_eps
        item["teacher_improves_all_auroc_ba_f1"] = int(all_positive)
        rows.append(item)
    return pd.DataFrame(rows)


def summarize(delta: pd.DataFrame, repeats: int, seed: int, tie_eps: float) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metric, label in METRIC_SPECS:
        values = pd.to_numeric(delta[f"delta_{metric}"], errors="coerce")
        ci_low, ci_high = bootstrap_ci(values.to_numpy(dtype=float), repeats=repeats, seed=seed)
        wins, losses, ties = win_loss_tie(values, tie_eps=tie_eps)
        rows.append(
            {
                "metric": label,
                "delta_definition": "teacher_minus_student",
                "n_folds": int(values.notna().sum()),
                "mean_delta": float(values.mean()),
                "median_delta": float(values.median()),
                "bootstrap95_low": ci_low,
                "bootstrap95_high": ci_high,
                "teacher_wins": wins,
                "teacher_losses": losses,
                "ties": ties,
            }
        )
    rows.append(
        {
            "metric": "AUROC+BA+F1",
            "delta_definition": "teacher_better_on_all_three",
            "n_folds": int(len(delta)),
            "mean_delta": float(delta["teacher_improves_all_auroc_ba_f1"].mean()),
            "median_delta": float(delta["teacher_improves_all_auroc_ba_f1"].median()),
            "bootstrap95_low": float("nan"),
            "bootstrap95_high": float("nan"),
            "teacher_wins": int(delta["teacher_improves_all_auroc_ba_f1"].sum()),
            "teacher_losses": int(len(delta) - delta["teacher_improves_all_auroc_ba_f1"].sum()),
            "ties": 0,
        }
    )
    return pd.DataFrame(rows)


def format_float(value: float) -> str:
    if pd.isna(value):
        return "nan"
    return f"{float(value):.4f}"


def write_markdown(
    delta: pd.DataFrame,
    summary: pd.DataFrame,
    output_path: Path,
    dataset_kind: str,
    student_name: str,
    teacher_csv: Path,
    student_csv: Path,
) -> None:
    lines: list[str] = [
        f"# {dataset_kind.upper()} Teacher vs {student_name} Validation Comparison",
        "",
        "Delta is defined as teacher validation metric minus student validation metric.",
        "This comparison is validation-side only and is suitable for regime diagnosis; it should not use held-out test metrics.",
        "",
        f"- Teacher CSV: `{teacher_csv}`",
        f"- Student CSV: `{student_csv}`",
        f"- Overlapping folds: {len(delta)}",
        "",
        "## Per-Fold Deltas",
        "",
        "| Fold | Teacher val AUROC | Student val AUROC | Delta AUROC | Teacher val BA | Student val BA | Delta BA | Teacher val F1 | Student val F1 | Delta F1 |",
        "| ---- | ----------------: | ----------------: | ----------: | -------------: | -------------: | -------: | -------------: | -------------: | -------: |",
    ]
    for _, row in delta.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["fold"]),
                    format_float(row["teacher_val_auroc"]),
                    format_float(row["student_val_auroc"]),
                    format_float(row["delta_auroc"]),
                    format_float(row["teacher_val_balanced_acc"]),
                    format_float(row["student_val_balanced_acc"]),
                    format_float(row["delta_balanced_acc"]),
                    format_float(row["teacher_val_f1"]),
                    format_float(row["student_val_f1"]),
                    format_float(row["delta_f1"]),
                ]
            )
            + " |"
        )

    lines.extend(["", "## Paired Summary", ""])
    for _, row in summary.iterrows():
        metric = str(row["metric"])
        if metric == "AUROC+BA+F1":
            lines.append(
                f"- Teacher improves AUROC, BA, and F1 simultaneously in "
                f"{int(row['teacher_wins'])}/{int(row['n_folds'])} folds."
            )
            continue
        lines.append(
            f"- {metric}: mean Delta={format_float(row['mean_delta'])}, "
            f"median Delta={format_float(row['median_delta'])}, "
            f"95% bootstrap CI [{format_float(row['bootstrap95_low'])}, {format_float(row['bootstrap95_high'])}], "
            f"W/L/T={int(row['teacher_wins'])}/{int(row['teacher_losses'])}/{int(row['ties'])}."
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare teacher-only and student/motion-aware validation summaries by fold."
    )
    parser.add_argument("--teacher-csv", type=Path, required=True)
    parser.add_argument("--student-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-kind", type=str, required=True)
    parser.add_argument("--student-name", type=str, default="motion_aware")
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tie-eps", type=float, default=1e-6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    teacher = load_summary(args.teacher_csv, name="teacher")
    student = load_summary(args.student_csv, name="student")
    delta = build_delta_table(teacher, student, student_name=args.student_name, tie_eps=args.tie_eps)
    summary = summarize(delta, repeats=args.bootstrap_repeats, seed=args.seed, tie_eps=args.tie_eps)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    safe_student = args.student_name.replace("/", "_").replace(" ", "_")
    delta_path = args.output_dir / f"{args.dataset_kind}_teacher_vs_{safe_student}_validation_deltas.csv"
    summary_path = args.output_dir / f"{args.dataset_kind}_teacher_vs_{safe_student}_validation_summary.csv"
    md_path = args.output_dir / f"{args.dataset_kind}_teacher_vs_{safe_student}_validation_summary.md"
    delta.to_csv(delta_path, index=False)
    summary.to_csv(summary_path, index=False)
    write_markdown(delta, summary, md_path, args.dataset_kind, args.student_name, args.teacher_csv, args.student_csv)
    print(f"saved_delta_csv={delta_path}")
    print(f"saved_summary_csv={summary_path}")
    print(f"saved_summary_md={md_path}")


if __name__ == "__main__":
    main()
