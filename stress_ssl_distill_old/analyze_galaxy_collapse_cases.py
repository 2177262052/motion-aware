from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def classify_collapse_level(positive_rate: float) -> str:
    if pd.isna(positive_rate):
        return "unknown"
    if positive_rate <= 0.05 or positive_rate >= 0.95:
        return "full_collapse"
    if positive_rate <= 0.15 or positive_rate >= 0.85:
        return "near_collapse"
    return "stable"


def classify_failure_mode(balanced_acc: float, auroc: float) -> str:
    if pd.isna(balanced_acc) or pd.isna(auroc):
        return "unknown"
    if auroc >= 0.80 and balanced_acc <= 0.55:
        return "threshold_failure"
    if auroc < 0.70 and balanced_acc <= 0.55:
        return "representation_failure"
    if auroc >= 0.80 and balanced_acc > 0.55:
        return "ranking_ok"
    return "mixed_failure"


def load_results(results_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(results_csv)
    required = [
        "subject",
        "watch_only_balanced_acc",
        "watch_only_auroc",
        "watch_only_positive_rate",
        "deploy_watch_balanced_acc",
        "deploy_watch_auroc",
        "deploy_watch_positive_rate",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in results CSV: {missing}")
    return df.copy()


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "test_positive_prior" not in out.columns:
        out["test_positive_prior"] = 0.5

    for prefix in ("watch_only", "deploy_watch", "teacher"):
        pr_col = f"{prefix}_positive_rate"
        ba_col = f"{prefix}_balanced_acc"
        auroc_col = f"{prefix}_auroc"
        if pr_col not in out.columns:
            continue
        out[f"{prefix}_collapse_level"] = out[pr_col].apply(classify_collapse_level)
        out[f"{prefix}_failure_mode"] = [
            classify_failure_mode(ba, auroc)
            for ba, auroc in zip(out.get(ba_col, pd.Series(dtype=float)), out.get(auroc_col, pd.Series(dtype=float)))
        ]
        err_col = f"{prefix}_positive_rate_error"
        if err_col not in out.columns:
            out[err_col] = (pd.to_numeric(out[pr_col], errors="coerce") - pd.to_numeric(out["test_positive_prior"], errors="coerce")).abs()

    out["deploy_rescues_watch_collapse"] = (
        out["watch_only_collapse_level"].isin(["full_collapse", "near_collapse"])
        & (out["deploy_watch_collapse_level"] == "stable")
    ).astype(int)
    out["deploy_new_collapse"] = (
        (out["watch_only_collapse_level"] == "stable")
        & out["deploy_watch_collapse_level"].isin(["full_collapse", "near_collapse"])
    ).astype(int)
    out["deploy_improves_pre"] = (
        pd.to_numeric(out["deploy_watch_positive_rate_error"], errors="coerce")
        < pd.to_numeric(out["watch_only_positive_rate_error"], errors="coerce") - 1e-12
    ).astype(int)
    out["deploy_worsens_pre"] = (
        pd.to_numeric(out["deploy_watch_positive_rate_error"], errors="coerce")
        > pd.to_numeric(out["watch_only_positive_rate_error"], errors="coerce") + 1e-12
    ).astype(int)
    return out


def format_subject_block(row: pd.Series) -> str:
    return (
        f"{row['subject']}: "
        f"prior={row['test_positive_prior']:.4f} "
        f"watch[level={row['watch_only_collapse_level']}, mode={row['watch_only_failure_mode']}, "
        f"ba={row['watch_only_balanced_acc']:.4f}, auroc={row['watch_only_auroc']:.4f}, "
        f"pr={row['watch_only_positive_rate']:.4f}, pre={row['watch_only_positive_rate_error']:.4f}] "
        f"deploy[level={row['deploy_watch_collapse_level']}, mode={row['deploy_watch_failure_mode']}, "
        f"ba={row['deploy_watch_balanced_acc']:.4f}, auroc={row['deploy_watch_auroc']:.4f}, "
        f"pr={row['deploy_watch_positive_rate']:.4f}, pre={row['deploy_watch_positive_rate_error']:.4f}]"
    )


def build_report(df: pd.DataFrame) -> str:
    lines: list[str] = []
    lines.append("Galaxy Collapse Analysis")
    lines.append("")
    lines.append(f"num_folds={len(df)}")
    lines.append(f"watch_full_collapse={(df['watch_only_collapse_level'] == 'full_collapse').sum()}")
    lines.append(f"watch_near_collapse={(df['watch_only_collapse_level'] == 'near_collapse').sum()}")
    lines.append(f"deploy_full_collapse={(df['deploy_watch_collapse_level'] == 'full_collapse').sum()}")
    lines.append(f"deploy_near_collapse={(df['deploy_watch_collapse_level'] == 'near_collapse').sum()}")
    lines.append(f"deploy_rescues_watch_collapse={int(df['deploy_rescues_watch_collapse'].sum())}")
    lines.append(f"deploy_new_collapse={int(df['deploy_new_collapse'].sum())}")
    lines.append(f"deploy_improves_positive_rate_error={int(df['deploy_improves_pre'].sum())}")
    lines.append(f"deploy_worsens_positive_rate_error={int(df['deploy_worsens_pre'].sum())}")
    lines.append("")

    lines.append("watch_collapsed_or_near")
    watch_bad = df[df["watch_only_collapse_level"].isin(["full_collapse", "near_collapse"])]
    if watch_bad.empty:
        lines.append("none")
    else:
        for _, row in watch_bad.sort_values(["watch_only_positive_rate_error", "watch_only_balanced_acc"], ascending=[False, True]).iterrows():
            lines.append(format_subject_block(row))
    lines.append("")

    lines.append("deploy_collapsed_or_near")
    deploy_bad = df[df["deploy_watch_collapse_level"].isin(["full_collapse", "near_collapse"])]
    if deploy_bad.empty:
        lines.append("none")
    else:
        for _, row in deploy_bad.sort_values(["deploy_watch_positive_rate_error", "deploy_watch_balanced_acc"], ascending=[False, True]).iterrows():
            lines.append(format_subject_block(row))
    lines.append("")

    lines.append("rescued_subjects")
    rescued = df[df["deploy_rescues_watch_collapse"] == 1]
    if rescued.empty:
        lines.append("none")
    else:
        for _, row in rescued.sort_values("subject").iterrows():
            lines.append(format_subject_block(row))
    lines.append("")

    lines.append("new_failures_under_deploy")
    new_failures = df[df["deploy_new_collapse"] == 1]
    if new_failures.empty:
        lines.append("none")
    else:
        for _, row in new_failures.sort_values("subject").iterrows():
            lines.append(format_subject_block(row))
    lines.append("")

    lines.append("likely_threshold_failures")
    threshold_bad = df[
        (df["watch_only_failure_mode"] == "threshold_failure")
        | (df["deploy_watch_failure_mode"] == "threshold_failure")
    ]
    if threshold_bad.empty:
        lines.append("none")
    else:
        for _, row in threshold_bad.sort_values("subject").iterrows():
            lines.append(format_subject_block(row))
    lines.append("")

    lines.append("likely_representation_failures")
    rep_bad = df[
        (df["watch_only_failure_mode"] == "representation_failure")
        | (df["deploy_watch_failure_mode"] == "representation_failure")
    ]
    if rep_bad.empty:
        lines.append("none")
    else:
        for _, row in rep_bad.sort_values("subject").iterrows():
            lines.append(format_subject_block(row))

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze collapsed and near-collapsed LOSO folds for watch-only vs deploy-watch.")
    parser.add_argument("--results-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    df = enrich(load_results(args.results_csv))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    enriched_csv = args.output_dir / "galaxy_collapse_analysis.csv"
    report_txt = args.output_dir / "galaxy_collapse_analysis.txt"

    df.to_csv(enriched_csv, index=False)
    report_txt.write_text(build_report(df), encoding="utf-8")

    print(f"Saved collapse analysis CSV to {enriched_csv}")
    print(f"Saved collapse analysis TXT to {report_txt}")


if __name__ == "__main__":
    main()
