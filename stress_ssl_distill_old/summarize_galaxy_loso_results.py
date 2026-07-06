from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def mean_std(series: pd.Series) -> tuple[float, float]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return float("nan"), float("nan")
    if len(clean) == 1:
        value = float(clean.iloc[0])
        return value, 0.0
    return float(clean.mean()), float(clean.std(ddof=1))


def append_model_block(lines: list[str], title: str, df: pd.DataFrame, prefix: str) -> None:
    ba_mean, ba_std = mean_std(df[f"{prefix}_balanced_acc"])
    auroc_mean, auroc_std = mean_std(df[f"{prefix}_auroc"])
    f1_mean, f1_std = mean_std(df[f"{prefix}_f1"])
    collapse_mean, _ = mean_std(df[f"{prefix}_collapse"])
    lines.append(f"{title} balanced_acc_mean={ba_mean:.4f} balanced_acc_std={ba_std:.4f}")
    lines.append(f"{title} auroc_mean={auroc_mean:.4f} auroc_std={auroc_std:.4f}")
    lines.append(f"{title} f1_mean={f1_mean:.4f} f1_std={f1_std:.4f}")
    lines.append(f"{title} collapse_rate={collapse_mean:.4f}")

    pre_col = f"{prefix}_positive_rate_error"
    if pre_col in df.columns:
        pre_mean, pre_std = mean_std(df[pre_col])
        lines.append(f"{title} positive_rate_error_mean={pre_mean:.4f} positive_rate_error_std={pre_std:.4f}")


def win_loss_tie(
    left: pd.Series,
    right: pd.Series,
    *,
    smaller_is_better: bool = False,
) -> tuple[int, int, int]:
    wins = 0
    losses = 0
    ties = 0
    for left_value, right_value in zip(left, right):
        if pd.isna(left_value) or pd.isna(right_value):
            continue
        if smaller_is_better:
            if left_value < right_value - 1e-12:
                wins += 1
            elif left_value > right_value + 1e-12:
                losses += 1
            else:
                ties += 1
        else:
            if left_value > right_value + 1e-12:
                wins += 1
            elif left_value < right_value - 1e-12:
                losses += 1
            else:
                ties += 1
    return wins, losses, ties


def build_summary(df: pd.DataFrame) -> str:
    lines: list[str] = []
    lines.append("[summary]")
    append_model_block(lines, "watch_only", df, "watch_only")
    append_model_block(lines, "deploy_watch", df, "deploy_watch")
    append_model_block(lines, "teacher", df, "teacher")

    ba_wins, ba_losses, ba_ties = win_loss_tie(
        df["deploy_watch_balanced_acc"],
        df["watch_only_balanced_acc"],
    )
    auroc_wins, auroc_losses, auroc_ties = win_loss_tie(
        df["deploy_watch_auroc"],
        df["watch_only_auroc"],
    )
    lines.append(
        f"deploy_vs_watch balanced_acc_wins={ba_wins} balanced_acc_losses={ba_losses} balanced_acc_ties={ba_ties}"
    )
    lines.append(
        f"deploy_vs_watch auroc_wins={auroc_wins} auroc_losses={auroc_losses} auroc_ties={auroc_ties}"
    )

    if "deploy_watch_positive_rate_error" in df.columns and "watch_only_positive_rate_error" in df.columns:
        pre_wins, pre_losses, pre_ties = win_loss_tie(
            df["deploy_watch_positive_rate_error"],
            df["watch_only_positive_rate_error"],
            smaller_is_better=True,
        )
        lines.append(
            f"deploy_vs_watch positive_rate_error_wins={pre_wins} positive_rate_error_losses={pre_losses} positive_rate_error_ties={pre_ties}"
        )

    return "\n".join(lines) + "\n"


def maybe_add_positive_rate_error(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "test_positive_prior" not in out.columns:
        return out

    triplets = [
        ("watch_only_positive_rate", "watch_only_positive_rate_error"),
        ("deploy_watch_positive_rate", "deploy_watch_positive_rate_error"),
        ("teacher_positive_rate", "teacher_positive_rate_error"),
    ]
    for rate_col, err_col in triplets:
        if rate_col in out.columns and err_col not in out.columns:
            out[err_col] = (pd.to_numeric(out[rate_col], errors="coerce") - pd.to_numeric(out["test_positive_prior"], errors="coerce")).abs()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute LOSO summary statistics from an existing formal results CSV.")
    parser.add_argument("--results-csv", type=Path, required=True)
    parser.add_argument("--output-txt", type=Path, required=True)
    parser.add_argument("--overwrite-csv", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.results_csv)
    df = maybe_add_positive_rate_error(df)

    if args.overwrite_csv:
        df.to_csv(args.results_csv, index=False)

    args.output_txt.parent.mkdir(parents=True, exist_ok=True)
    args.output_txt.write_text(build_summary(df), encoding="utf-8")

    print(f"Saved recomputed LOSO summary to {args.output_txt}")
    if args.overwrite_csv:
        print(f"Updated LOSO CSV in place at {args.results_csv}")


if __name__ == "__main__":
    main()
