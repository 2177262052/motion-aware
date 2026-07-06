from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MOTION_ORDER = ["low", "mid", "high"]


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, formats: list[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(output_dir / f"{stem}.{fmt}", bbox_inches="tight")
    plt.close(fig)


def add_motion_bin(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    out = frame.copy()
    values = pd.to_numeric(out[column], errors="coerce")
    out["motion_bin"] = "missing"
    finite = values.notna() & np.isfinite(values)
    if values[finite].nunique() < 3:
        out.loc[finite, "motion_bin"] = "all"
        return out
    bins = pd.qcut(values[finite], q=3, duplicates="drop")
    labels = MOTION_ORDER[: len(bins.cat.categories)]
    bins = bins.cat.rename_categories(labels)
    out.loc[finite, "motion_bin"] = bins.astype(str)
    return out


def prepare_frame(
    windows: pd.DataFrame,
    model_name: str,
    motion_column: str,
) -> pd.DataFrame:
    required = {"model_name", "label", "teacher_prob"}
    missing = required.difference(windows.columns)
    if missing:
        raise ValueError(f"Failure windows CSV missing columns: {sorted(missing)}")
    frame = windows[windows["model_name"] == model_name].copy()
    if frame.empty:
        available = sorted(windows["model_name"].dropna().astype(str).unique())
        raise ValueError(f"Model {model_name!r} not found. Available: {available}")
    if motion_column not in frame.columns:
        raise ValueError(f"Failure windows CSV missing motion column: {motion_column}")

    frame["label"] = pd.to_numeric(frame["label"], errors="coerce").astype("Int64")
    frame["teacher_prob"] = pd.to_numeric(frame["teacher_prob"], errors="coerce")
    frame = frame.dropna(subset=["label", "teacher_prob"]).copy()
    frame["label"] = frame["label"].astype(int)
    frame["teacher_pred"] = (frame["teacher_prob"] >= 0.5).astype(int)
    frame["teacher_correct"] = (frame["teacher_pred"] == frame["label"]).astype(int)
    frame["teacher_confidence"] = np.maximum(frame["teacher_prob"], 1.0 - frame["teacher_prob"])
    frame["teacher_margin_abs"] = np.abs(frame["teacher_prob"] - 0.5) * 2.0
    frame = add_motion_bin(frame, motion_column)
    return frame


def build_confidence_bins(frame: pd.DataFrame, n_bins: int) -> pd.DataFrame:
    out = frame.copy()
    labels = [f"Q{i + 1}" for i in range(n_bins)]
    try:
        out["confidence_bin"] = pd.qcut(out["teacher_confidence"], q=n_bins, labels=labels, duplicates="drop")
    except ValueError:
        out["confidence_bin"] = "all"
    out["confidence_bin"] = out["confidence_bin"].astype(str)
    rows: list[dict[str, object]] = []
    for confidence_bin, group in out.groupby("confidence_bin", dropna=False):
        rows.append(
            {
                "confidence_bin": str(confidence_bin),
                "n": int(len(group)),
                "confidence_mean": float(group["teacher_confidence"].mean()),
                "confidence_min": float(group["teacher_confidence"].min()),
                "confidence_max": float(group["teacher_confidence"].max()),
                "teacher_error_rate": float(1.0 - group["teacher_correct"].mean()),
                "teacher_accuracy": float(group["teacher_correct"].mean()),
                "positive_prior": float(group["label"].mean()),
            }
        )
    return pd.DataFrame(rows)


def build_motion_bins(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for motion_bin, group in frame.groupby("motion_bin", dropna=False):
        rows.append(
            {
                "motion_bin": str(motion_bin),
                "n": int(len(group)),
                "confidence_mean": float(group["teacher_confidence"].mean()),
                "teacher_error_rate": float(1.0 - group["teacher_correct"].mean()),
                "teacher_accuracy": float(group["teacher_correct"].mean()),
                "positive_prior": float(group["label"].mean()),
            }
        )
    out = pd.DataFrame(rows)
    out["motion_bin"] = pd.Categorical(out["motion_bin"], categories=MOTION_ORDER, ordered=True)
    return out.sort_values("motion_bin").reset_index(drop=True)


def summarize_high_conf_wrong(frame: pd.DataFrame, quantile: float) -> pd.DataFrame:
    threshold = float(frame["teacher_confidence"].quantile(quantile))
    high = frame[frame["teacher_confidence"] >= threshold].copy()
    wrong = high[high["teacher_correct"] == 0].copy()
    rows = [
        {
            "confidence_quantile_threshold": quantile,
            "confidence_threshold": threshold,
            "n_high_confidence": int(len(high)),
            "n_high_confidence_wrong": int(len(wrong)),
            "high_confidence_wrong_rate": float(len(wrong) / max(len(high), 1)),
            "overall_error_rate": float(1.0 - frame["teacher_correct"].mean()),
        }
    ]
    return pd.DataFrame(rows)


def plot_teacher_confidence(
    confidence_bins: pd.DataFrame,
    motion_bins: pd.DataFrame,
    high_conf_summary: pd.DataFrame,
    output_dir: Path,
    formats: list[str],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.3))

    x = np.arange(len(confidence_bins))
    axes[0].bar(
        x,
        confidence_bins["teacher_error_rate"],
        color="#dd6b20",
        edgecolor="#111827",
        linewidth=0.5,
    )
    for idx, row in confidence_bins.iterrows():
        axes[0].text(
            idx,
            row["teacher_error_rate"] + 0.018,
            f"n={int(row['n'])}\nconf={row['confidence_mean']:.2f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    axes[0].set_title("(a) Teacher error by confidence bin")
    axes[0].set_ylabel("Teacher error rate")
    axes[0].set_xlabel("Teacher confidence quantile")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(confidence_bins["confidence_bin"])
    axes[0].set_ylim(0, min(1.0, max(0.25, float(confidence_bins["teacher_error_rate"].max()) + 0.15)))
    axes[0].grid(axis="y", color="#e5e7eb", linewidth=0.8)

    motion_bins = motion_bins[motion_bins["motion_bin"].isin(MOTION_ORDER)].copy()
    x2 = np.arange(len(motion_bins))
    width = 0.36
    axes[1].bar(
        x2 - width / 2,
        motion_bins["teacher_error_rate"],
        width=width,
        color="#dd6b20",
        edgecolor="#111827",
        linewidth=0.5,
        label="Error rate",
    )
    axes[1].bar(
        x2 + width / 2,
        motion_bins["confidence_mean"],
        width=width,
        color="#4c78a8",
        edgecolor="#111827",
        linewidth=0.5,
        label="Mean confidence",
    )
    axes[1].set_title("(b) Teacher behavior across motion bins")
    axes[1].set_ylabel("Rate / confidence")
    axes[1].set_xlabel("Motion level")
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels([str(item).capitalize() for item in motion_bins["motion_bin"]])
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", color="#e5e7eb", linewidth=0.8)

    if not high_conf_summary.empty:
        row = high_conf_summary.iloc[0]
        fig.text(
            0.5,
            -0.03,
            (
                f"High-confidence threshold (Q{int(row['confidence_quantile_threshold'] * 100)}): "
                f"{row['confidence_threshold']:.3f}; "
                f"wrong among high-confidence windows: "
                f"{int(row['n_high_confidence_wrong'])}/{int(row['n_high_confidence'])} "
                f"({row['high_confidence_wrong_rate']:.1%})."
            ),
            ha="center",
            va="top",
            fontsize=8,
        )

    fig.tight_layout()
    save_figure(fig, output_dir, "teacher_confidence_correctness", formats)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot teacher confidence versus correctness to sanity-check whether the privileged teacher is a reliable oracle."
    )
    parser.add_argument("--failure-windows-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", type=str, default="Ours")
    parser.add_argument("--motion-column", type=str, default="acc_jerk_rms")
    parser.add_argument("--confidence-bins", type=int, default=4)
    parser.add_argument("--high-confidence-quantile", type=float, default=0.8)
    parser.add_argument("--formats", nargs="*", default=["png", "pdf"])
    args = parser.parse_args()

    set_plot_style()
    windows = pd.read_csv(args.failure_windows_csv)
    frame = prepare_frame(windows, args.model_name, args.motion_column)
    confidence_bins = build_confidence_bins(frame, args.confidence_bins)
    motion_bins = build_motion_bins(frame)
    high_conf_summary = summarize_high_conf_wrong(frame, args.high_confidence_quantile)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "teacher_confidence_windows.csv", index=False)
    confidence_bins.to_csv(args.output_dir / "teacher_confidence_bins.csv", index=False)
    motion_bins.to_csv(args.output_dir / "teacher_confidence_motion_bins.csv", index=False)
    high_conf_summary.to_csv(args.output_dir / "teacher_high_confidence_wrong_summary.csv", index=False)
    plot_teacher_confidence(confidence_bins, motion_bins, high_conf_summary, args.output_dir, args.formats)

    print(confidence_bins.to_string(index=False))
    print()
    print(motion_bins.to_string(index=False))
    print()
    print(high_conf_summary.to_string(index=False))
    print(f"Saved teacher confidence diagnostics to {args.output_dir}")


if __name__ == "__main__":
    main()
