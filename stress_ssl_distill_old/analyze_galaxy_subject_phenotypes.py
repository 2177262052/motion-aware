from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DELTA_COLS = [
    "delta_wavelet_a4",
    "delta_wavelet_d4",
    "delta_wavelet_d2",
    "delta_wavelet_d1",
    "delta_acc_mag_mean",
]


def load_audit(audit_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(audit_csv)
    if "subject_id" not in df.columns:
        raise ValueError(f"subject_id column missing in {audit_csv}")
    return df.sort_values("subject_id").reset_index(drop=True)


def load_fold_results(fold_csv: Path | None) -> pd.DataFrame | None:
    if fold_csv is None:
        return None
    df = pd.read_csv(fold_csv)
    if "subject" not in df.columns:
        raise ValueError(f"subject column missing in {fold_csv}")
    renamed = df.rename(columns={"subject": "subject_id"}).copy()
    needed = [
        "subject_id",
        "watch_balanced_acc",
        "watch_auroc",
        "teacher_balanced_acc",
        "teacher_auroc",
        "delta_balanced_acc",
        "delta_auroc",
        "teacher_ba_result",
        "teacher_auroc_result",
    ]
    for col in needed:
        if col not in renamed.columns:
            renamed[col] = pd.NA
    return renamed[needed]


def compute_strength(df: pd.DataFrame) -> pd.Series:
    return (
        df["delta_wavelet_a4"].abs()
        + df["delta_wavelet_d4"].abs()
        + 0.5 * df["delta_wavelet_d2"].abs()
        + 0.5 * df["delta_wavelet_d1"].abs()
    )


def classify_direction(row: pd.Series) -> str:
    a4 = row["delta_wavelet_a4"]
    d4 = row["delta_wavelet_d4"]
    if pd.isna(a4) or pd.isna(d4):
        return "missing"
    if a4 > 0 and d4 < 0:
        return "canonical"
    if a4 < 0 and d4 > 0:
        return "inverse"
    if abs(a4) < 1e-9 and abs(d4) < 1e-9:
        return "flat"
    return "mixed"


def classify_strength_group(strength: pd.Series) -> pd.Series:
    valid = strength.dropna()
    if valid.empty:
        return pd.Series(["unknown"] * len(strength), index=strength.index)
    low_cut = float(valid.quantile(1 / 3))
    high_cut = float(valid.quantile(2 / 3))

    def _label(value: float) -> str:
        if pd.isna(value):
            return "unknown"
        if value <= low_cut:
            return "low"
        if value >= high_cut:
            return "high"
        return "medium"

    return strength.apply(_label)


def classify_phenotype(row: pd.Series) -> str:
    if row["subject_id"] == "P01":
        return "missing_task_data"

    strength = row["response_strength"]
    direction = row["response_direction"]

    if pd.isna(strength) or direction == "missing":
        return "missing_task_data"
    if strength < 0.08:
        return "flat_responder"
    if direction == "canonical":
        return "canonical_responder"
    if direction == "inverse":
        return "inverse_responder"
    return "mixed_responder"


def add_phenotypes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in DELTA_COLS:
        if col not in out.columns:
            out[col] = pd.NA
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["response_strength"] = compute_strength(out)
    out["response_direction"] = out.apply(classify_direction, axis=1)
    out["strength_group"] = classify_strength_group(out["response_strength"])
    out["phenotype"] = out.apply(classify_phenotype, axis=1)
    return out


def merge_with_fold_results(audit_df: pd.DataFrame, fold_df: pd.DataFrame | None) -> pd.DataFrame:
    if fold_df is None:
        return audit_df
    return audit_df.merge(fold_df, on="subject_id", how="left")


def format_mean_std(series: pd.Series) -> str:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return "nan +/- nan"
    std = clean.std(ddof=1) if len(clean) > 1 else 0.0
    return f"{clean.mean():.4f} +/- {std:.4f}"


def build_group_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for phenotype, group in df.groupby("phenotype", dropna=False):
        row: dict[str, object] = {
            "phenotype": phenotype,
            "num_subjects": int(len(group)),
            "subjects": ",".join(group["subject_id"].astype(str).tolist()),
            "response_strength_mean": pd.to_numeric(group["response_strength"], errors="coerce").mean(),
            "delta_wavelet_a4_mean": pd.to_numeric(group["delta_wavelet_a4"], errors="coerce").mean(),
            "delta_wavelet_d4_mean": pd.to_numeric(group["delta_wavelet_d4"], errors="coerce").mean(),
            "delta_acc_mag_mean_mean": pd.to_numeric(group["delta_acc_mag_mean"], errors="coerce").mean(),
        }
        if "watch_balanced_acc" in group.columns:
            row["watch_balanced_acc_mean"] = pd.to_numeric(group["watch_balanced_acc"], errors="coerce").mean()
            row["teacher_balanced_acc_mean"] = pd.to_numeric(group["teacher_balanced_acc"], errors="coerce").mean()
            row["watch_auroc_mean"] = pd.to_numeric(group["watch_auroc"], errors="coerce").mean()
            row["teacher_auroc_mean"] = pd.to_numeric(group["teacher_auroc"], errors="coerce").mean()
            row["delta_balanced_acc_mean"] = pd.to_numeric(group["delta_balanced_acc"], errors="coerce").mean()
            row["delta_auroc_mean"] = pd.to_numeric(group["delta_auroc"], errors="coerce").mean()
            row["teacher_ba_wins"] = int((group["teacher_ba_result"] == "win").sum())
            row["teacher_ba_losses"] = int((group["teacher_ba_result"] == "loss").sum())
            row["teacher_auroc_wins"] = int((group["teacher_auroc_result"] == "win").sum())
            row["teacher_auroc_losses"] = int((group["teacher_auroc_result"] == "loss").sum())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["num_subjects", "phenotype"], ascending=[False, True]).reset_index(drop=True)


def build_report(df: pd.DataFrame, summary_df: pd.DataFrame, has_fold_results: bool) -> str:
    lines: list[str] = []
    lines.append("Galaxy Subject Phenotype Analysis")
    lines.append("")
    lines.append(f"num_subjects={len(df)}")
    lines.append(f"phenotype_counts={df['phenotype'].value_counts().to_dict()}")
    lines.append(f"strength_group_counts={df['strength_group'].value_counts().to_dict()}")
    lines.append("")

    lines.append("phenotype_members")
    for row in summary_df.itertuples(index=False):
        lines.append(f"{row.phenotype}: n={row.num_subjects} subjects={row.subjects}")
    lines.append("")

    lines.append("strongest_subjects")
    top_strength = df.nlargest(8, "response_strength")[
        ["subject_id", "phenotype", "response_strength", "delta_wavelet_a4", "delta_wavelet_d4", "delta_acc_mag_mean"]
    ]
    for row in top_strength.itertuples(index=False):
        lines.append(
            f"{row.subject_id}: phenotype={row.phenotype} "
            f"strength={row.response_strength:.4f} "
            f"dA4={row.delta_wavelet_a4:.4f} dD4={row.delta_wavelet_d4:.4f} "
            f"d_acc={row.delta_acc_mag_mean:.4f}"
        )
    lines.append("")

    lines.append("flattest_subjects")
    flat_strength = df.nsmallest(8, "response_strength")[
        ["subject_id", "phenotype", "response_strength", "delta_wavelet_a4", "delta_wavelet_d4", "delta_acc_mag_mean"]
    ]
    for row in flat_strength.itertuples(index=False):
        lines.append(
            f"{row.subject_id}: phenotype={row.phenotype} "
            f"strength={row.response_strength:.4f} "
            f"dA4={row.delta_wavelet_a4:.4f} dD4={row.delta_wavelet_d4:.4f} "
            f"d_acc={row.delta_acc_mag_mean:.4f}"
        )
    lines.append("")

    if has_fold_results:
        lines.append("performance_by_phenotype")
        for row in summary_df.itertuples(index=False):
            if pd.isna(getattr(row, "watch_balanced_acc_mean", np.nan)):
                continue
            lines.append(
                f"{row.phenotype}: "
                f"watch_ba_mean={row.watch_balanced_acc_mean:.4f} "
                f"teacher_ba_mean={row.teacher_balanced_acc_mean:.4f} "
                f"delta_ba_mean={row.delta_balanced_acc_mean:.4f} "
                f"watch_auroc_mean={row.watch_auroc_mean:.4f} "
                f"teacher_auroc_mean={row.teacher_auroc_mean:.4f} "
                f"delta_auroc_mean={row.delta_auroc_mean:.4f} "
                f"teacher_ba_wins={row.teacher_ba_wins} "
                f"teacher_ba_losses={row.teacher_ba_losses}"
            )
        lines.append("")

        lines.append("overall_performance")
        lines.append(f"watch_balanced_acc={format_mean_std(df['watch_balanced_acc'])}")
        lines.append(f"teacher_balanced_acc={format_mean_std(df['teacher_balanced_acc'])}")
        lines.append(f"watch_auroc={format_mean_std(df['watch_auroc'])}")
        lines.append(f"teacher_auroc={format_mean_std(df['teacher_auroc'])}")
        lines.append("")

        lines.append("subjects_where_teacher_gains_most")
        best = df.nlargest(5, "delta_balanced_acc")[
            ["subject_id", "phenotype", "delta_balanced_acc", "delta_auroc", "watch_balanced_acc", "teacher_balanced_acc"]
        ]
        for row in best.itertuples(index=False):
            lines.append(
                f"{row.subject_id}: phenotype={row.phenotype} "
                f"delta_ba={row.delta_balanced_acc:.4f} delta_auroc={row.delta_auroc:.4f} "
                f"watch_ba={row.watch_balanced_acc:.4f} teacher_ba={row.teacher_balanced_acc:.4f}"
            )
        lines.append("")

        lines.append("subjects_where_teacher_loses_most")
        worst = df.nsmallest(5, "delta_balanced_acc")[
            ["subject_id", "phenotype", "delta_balanced_acc", "delta_auroc", "watch_balanced_acc", "teacher_balanced_acc"]
        ]
        for row in worst.itertuples(index=False):
            lines.append(
                f"{row.subject_id}: phenotype={row.phenotype} "
                f"delta_ba={row.delta_balanced_acc:.4f} delta_auroc={row.delta_auroc:.4f} "
                f"watch_ba={row.watch_balanced_acc:.4f} teacher_ba={row.teacher_balanced_acc:.4f}"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze GalaxyPPG subject response phenotypes and optionally align them with LOSO fold results."
    )
    parser.add_argument("--audit-csv", type=Path, required=True)
    parser.add_argument("--fold-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    audit_df = add_phenotypes(load_audit(args.audit_csv))
    fold_df = load_fold_results(args.fold_csv)
    merged_df = merge_with_fold_results(audit_df, fold_df)
    summary_df = build_group_summary(merged_df)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    subject_csv = args.output_dir / "galaxy_subject_phenotypes.csv"
    group_csv = args.output_dir / "galaxy_subject_phenotype_groups.csv"
    report_txt = args.output_dir / "galaxy_subject_phenotype_analysis.txt"

    merged_df.to_csv(subject_csv, index=False)
    summary_df.to_csv(group_csv, index=False)
    report_txt.write_text(build_report(merged_df, summary_df, fold_df is not None), encoding="utf-8")

    print(f"Saved phenotype CSV to {subject_csv}")
    print(f"Saved phenotype group CSV to {group_csv}")
    print(f"Saved phenotype report to {report_txt}")


if __name__ == "__main__":
    main()
