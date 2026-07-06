from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


FOLD_HEADER_RE = re.compile(r"^\[(P\d+)\]\s*$")
KV_RE = re.compile(r"([A-Za-z_]+)=([-+]?\d*\.?\d+)")


def parse_summary(summary_path: Path) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    current_subject: str | None = None
    current: dict[str, float | str] | None = None

    for raw_line in summary_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line == "[summary]":
            break

        fold_match = FOLD_HEADER_RE.match(line)
        if fold_match:
            if current is not None:
                rows.append(current)
            current_subject = fold_match.group(1)
            current = {"subject": current_subject}
            continue

        if current is None:
            continue

        if line.startswith("watch_only "):
            metrics = {f"watch_{key}": float(value) for key, value in KV_RE.findall(line)}
            current.update(metrics)
        elif line.startswith("teacher "):
            metrics = {f"teacher_{key}": float(value) for key, value in KV_RE.findall(line)}
            current.update(metrics)

    if current is not None:
        rows.append(current)

    if not rows:
        raise ValueError(f"No fold records found in {summary_path}")
    return pd.DataFrame(rows).sort_values("subject").reset_index(drop=True)


def add_deltas(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    required_numeric_cols = [
        "watch_threshold",
        "watch_acc",
        "watch_balanced_acc",
        "watch_f1",
        "watch_auroc",
        "watch_positive_rate",
        "teacher_threshold",
        "teacher_acc",
        "teacher_balanced_acc",
        "teacher_f1",
        "teacher_auroc",
        "teacher_positive_rate",
    ]
    for col in required_numeric_cols:
        if col not in out.columns:
            out[col] = pd.NA
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["delta_balanced_acc"] = out["teacher_balanced_acc"] - out["watch_balanced_acc"]
    out["delta_auroc"] = out["teacher_auroc"] - out["watch_auroc"]
    out["delta_acc"] = out["teacher_acc"] - out["watch_acc"]
    out["delta_f1"] = out["teacher_f1"] - out["watch_f1"]
    out["delta_positive_rate"] = out["teacher_positive_rate"] - out["watch_positive_rate"]

    out["teacher_ba_result"] = out["delta_balanced_acc"].apply(
        lambda x: "win" if x > 1e-9 else ("loss" if x < -1e-9 else "tie")
    )
    out["teacher_auroc_result"] = out["delta_auroc"].apply(
        lambda x: "win" if x > 1e-9 else ("loss" if x < -1e-9 else "tie")
    )

    out["watch_extreme_positive_rate"] = out["watch_positive_rate"].apply(lambda x: x <= 0.05 or x >= 0.95)
    out["teacher_extreme_positive_rate"] = out["teacher_positive_rate"].apply(lambda x: x <= 0.05 or x >= 0.95)
    out["either_extreme_positive_rate"] = out["watch_extreme_positive_rate"] | out["teacher_extreme_positive_rate"]

    out["watch_threshold_extreme"] = out["watch_threshold"].apply(lambda x: x <= 0.05 or x >= 0.95)
    out["teacher_threshold_extreme"] = out["teacher_threshold"].apply(lambda x: x <= 0.05 or x >= 0.95)
    out["either_threshold_extreme"] = out["watch_threshold_extreme"] | out["teacher_threshold_extreme"]
    return out


def format_mean_std(series: pd.Series) -> str:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return "nan ± nan"
    std = clean.std(ddof=1) if len(clean) > 1 else 0.0
    return f"{clean.mean():.4f} ± {std:.4f}"


def build_report(df: pd.DataFrame) -> str:
    lines: list[str] = []
    lines.append("Galaxy LOSO Fold Analysis")
    lines.append("")
    lines.append(f"num_folds={len(df)}")
    lines.append(f"watch_balanced_acc={format_mean_std(df['watch_balanced_acc'])}")
    lines.append(f"watch_auroc={format_mean_std(df['watch_auroc'])}")
    lines.append(f"teacher_balanced_acc={format_mean_std(df['teacher_balanced_acc'])}")
    lines.append(f"teacher_auroc={format_mean_std(df['teacher_auroc'])}")
    lines.append("")

    ba_counts = df["teacher_ba_result"].value_counts().to_dict()
    auroc_counts = df["teacher_auroc_result"].value_counts().to_dict()
    lines.append(
        "teacher_vs_watch_balanced_acc "
        f"wins={ba_counts.get('win', 0)} losses={ba_counts.get('loss', 0)} ties={ba_counts.get('tie', 0)}"
    )
    lines.append(
        "teacher_vs_watch_auroc "
        f"wins={auroc_counts.get('win', 0)} losses={auroc_counts.get('loss', 0)} ties={auroc_counts.get('tie', 0)}"
    )
    lines.append("")

    top_teacher_ba = df.nlargest(5, "delta_balanced_acc")[["subject", "delta_balanced_acc", "teacher_balanced_acc", "watch_balanced_acc"]]
    worst_teacher_ba = df.nsmallest(5, "delta_balanced_acc")[["subject", "delta_balanced_acc", "teacher_balanced_acc", "watch_balanced_acc"]]
    top_teacher_auroc = df.nlargest(5, "delta_auroc")[["subject", "delta_auroc", "teacher_auroc", "watch_auroc"]]
    worst_teacher_auroc = df.nsmallest(5, "delta_auroc")[["subject", "delta_auroc", "teacher_auroc", "watch_auroc"]]

    lines.append("top_teacher_balanced_acc_gains")
    for row in top_teacher_ba.itertuples(index=False):
        lines.append(
            f"{row.subject}: delta_balanced_acc={row.delta_balanced_acc:.4f} "
            f"teacher={row.teacher_balanced_acc:.4f} watch={row.watch_balanced_acc:.4f}"
        )
    lines.append("")

    lines.append("worst_teacher_balanced_acc_losses")
    for row in worst_teacher_ba.itertuples(index=False):
        lines.append(
            f"{row.subject}: delta_balanced_acc={row.delta_balanced_acc:.4f} "
            f"teacher={row.teacher_balanced_acc:.4f} watch={row.watch_balanced_acc:.4f}"
        )
    lines.append("")

    lines.append("top_teacher_auroc_gains")
    for row in top_teacher_auroc.itertuples(index=False):
        lines.append(
            f"{row.subject}: delta_auroc={row.delta_auroc:.4f} "
            f"teacher={row.teacher_auroc:.4f} watch={row.watch_auroc:.4f}"
        )
    lines.append("")

    lines.append("worst_teacher_auroc_losses")
    for row in worst_teacher_auroc.itertuples(index=False):
        lines.append(
            f"{row.subject}: delta_auroc={row.delta_auroc:.4f} "
            f"teacher={row.teacher_auroc:.4f} watch={row.watch_auroc:.4f}"
        )
    lines.append("")

    unstable = df[df["either_extreme_positive_rate"] | df["either_threshold_extreme"]][
        [
            "subject",
            "watch_threshold",
            "teacher_threshold",
            "watch_positive_rate",
            "teacher_positive_rate",
            "watch_balanced_acc",
            "teacher_balanced_acc",
            "watch_auroc",
            "teacher_auroc",
        ]
    ]
    lines.append("folds_with_extreme_threshold_or_positive_rate")
    if unstable.empty:
        lines.append("none")
    else:
        for row in unstable.itertuples(index=False):
            lines.append(
                f"{row.subject}: "
                f"watch_threshold={row.watch_threshold:.4f} teacher_threshold={row.teacher_threshold:.4f} "
                f"watch_positive_rate={row.watch_positive_rate:.4f} teacher_positive_rate={row.teacher_positive_rate:.4f} "
                f"watch_ba={row.watch_balanced_acc:.4f} teacher_ba={row.teacher_balanced_acc:.4f} "
                f"watch_auroc={row.watch_auroc:.4f} teacher_auroc={row.teacher_auroc:.4f}"
            )

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze per-fold watch-only vs teacher LOSO summary results.")
    parser.add_argument("--summary-txt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    df = add_deltas(parse_summary(args.summary_txt))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = args.output_dir / "galaxy_loso_fold_analysis.csv"
    report_path = args.output_dir / "galaxy_loso_fold_analysis.txt"

    df.to_csv(csv_path, index=False)
    report_path.write_text(build_report(df), encoding="utf-8")

    print(f"Saved fold CSV to {csv_path}")
    print(f"Saved fold report to {report_path}")


if __name__ == "__main__":
    main()
