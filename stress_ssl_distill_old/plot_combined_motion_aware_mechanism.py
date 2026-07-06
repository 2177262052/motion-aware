from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


BIN_ORDER = ["low_motion", "mid_motion", "high_motion"]
BIN_LABELS = {
    "low_motion": "Low",
    "mid_motion": "Mid",
    "high_motion": "High",
}
METHOD_ORDER = ["watch_only", "motion_aware"]
METHOD_LABELS = {
    "watch_only": "Watch-only",
    "motion_aware": "Motion-aware",
}
DATASET_LABELS = {
    "galaxy": "Galaxy PPG",
    "wesad": "WESAD",
}
DATASET_COLORS = {
    "galaxy": "#2E6FBA",
    "wesad": "#3C8D3F",
}
POSITIVE = "#5AA05A"
NEGATIVE = "#D65F3D"
GRID = "#D8DDE6"
TEXT = "#111827"


def read_csv(path: Path, dataset: str, required: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}. Available: {list(df.columns)}")
    df = df.copy()
    df["dataset"] = dataset
    return df


def bootstrap_ci(values: np.ndarray, n_boot: int = 5000, seed: int = 42) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    if values.size == 1 or n_boot <= 0:
        value = float(values.mean())
        return value, value
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    means = values[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def metric_ylim(values: pd.Series, pad: float = 0.06) -> tuple[float, float]:
    finite = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if finite.size == 0:
        return 0.0, 1.0
    low = max(0.0, float(finite.min()) - pad)
    high = min(1.0, float(finite.max()) + pad)
    if high - low < 0.08:
        center = 0.5 * (high + low)
        low = max(0.0, center - 0.04)
        high = min(1.0, center + 0.04)
    return low, high


def setup_panel(ax, title: str) -> None:
    ax.set_title(title, fontsize=11, fontweight="bold", loc="left", pad=8)
    ax.grid(True, axis="y", color=GRID, alpha=0.72, linewidth=0.8)
    ax.grid(True, axis="x", color=GRID, alpha=0.35, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#4B5563")
    ax.spines["bottom"].set_color("#4B5563")
    ax.tick_params(labelsize=8)


def plot_metric_lines(ax, bin_summary: pd.DataFrame, metric: str, title: str) -> None:
    setup_panel(ax, title)
    x = np.arange(len(BIN_ORDER), dtype=float)
    for dataset in ["galaxy", "wesad"]:
        ds = bin_summary[bin_summary["dataset"] == dataset]
        if ds.empty:
            continue
        color = DATASET_COLORS[dataset]
        for method in METHOD_ORDER:
            rows = ds[ds["method"].astype(str) == method].copy()
            values: list[float] = []
            for motion_bin in BIN_ORDER:
                row = rows[rows["motion_bin"].astype(str) == motion_bin]
                values.append(float(row[metric].iloc[0]) if not row.empty else float("nan"))
            linestyle = "-" if method == "motion_aware" else "--"
            marker = "o" if method == "motion_aware" else "s"
            alpha = 0.95 if method == "motion_aware" else 0.62
            label = f"{DATASET_LABELS[dataset]} {METHOD_LABELS[method]}"
            ax.plot(
                x,
                values,
                color=color,
                linestyle=linestyle,
                marker=marker,
                markersize=5.5,
                linewidth=2.1,
                alpha=alpha,
                markeredgecolor="white",
                markeredgewidth=0.8,
                label=label,
            )
    ax.set_xticks(x, [BIN_LABELS[item] for item in BIN_ORDER])
    ax.set_xlabel("ACC jerk bin", fontsize=9)
    ylabel = "Balanced accuracy" if metric == "balanced_acc" else metric.upper()
    if metric == "auroc":
        ylabel = "AUROC"
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_ylim(*metric_ylim(bin_summary[metric]))


def subject_sort_key(subject: object) -> tuple[str, int, str]:
    import re

    text = str(subject)
    match = re.search(r"(\d+)", text)
    prefix = text[: match.start()] if match else text
    number = int(match.group(1)) if match else 10_000
    return prefix, number, text


def plot_gap_bars(ax, gap_df: pd.DataFrame, dataset: str, title: str, n_boot: int, seed: int) -> None:
    setup_panel(ax, title)
    ds = gap_df[gap_df["dataset"] == dataset].copy()
    if ds.empty:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
        return
    ds["motion_sensitivity_gap_reduction"] = pd.to_numeric(
        ds["motion_sensitivity_gap_reduction"], errors="coerce"
    )
    ds = ds.dropna(subset=["motion_sensitivity_gap_reduction"]).copy()
    ds["_subject_sort"] = ds["subject_id"].map(subject_sort_key)
    ds = ds.sort_values("motion_sensitivity_gap_reduction", ascending=True).drop(columns="_subject_sort")

    values = ds["motion_sensitivity_gap_reduction"].to_numpy(dtype=float)
    subjects = ds["subject_id"].astype(str).tolist()
    y = np.arange(len(ds))
    colors = np.where(values >= 0.0, POSITIVE, NEGATIVE)
    ax.barh(y, values, color=colors, height=0.66, edgecolor="white", linewidth=0.4)
    ax.axvline(0.0, color="#2F3640", linestyle="--", linewidth=1.0, alpha=0.75)
    ax.set_yticks(y, subjects, fontsize=7)
    ax.set_xlabel("Motion-sensitivity gap reduction", fontsize=9)
    ax.set_ylabel("Held-out subject", fontsize=9)

    mean = float(np.mean(values))
    ci_low, ci_high = bootstrap_ci(values, n_boot=n_boot, seed=seed)
    wins = int(np.sum(values > 1e-12))
    losses = int(np.sum(values < -1e-12))
    ties = int(len(values) - wins - losses)
    ax.text(
        0.975,
        0.065,
        f"Mean = {mean:+.3f}\n95% CI [{ci_low:+.3f}, {ci_high:+.3f}]\nW/L/T = {wins}/{losses}/{ties}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color=TEXT,
        bbox={
            "boxstyle": "round,pad=0.45,rounding_size=0.2",
            "facecolor": "white",
            "edgecolor": "#B8BDC8",
            "linewidth": 0.8,
            "alpha": 0.95,
        },
    )


def make_plot(
    bin_summary: pd.DataFrame,
    subject_gap: pd.DataFrame,
    output_prefix: Path,
    title: str,
    subtitle: str,
    font_family: str,
    n_boot: int,
    seed: int,
    layout: str,
) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": font_family,
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "figure.dpi": 160,
            "savefig.dpi": 450,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    for column in ["balanced_acc", "auroc"]:
        bin_summary[column] = pd.to_numeric(bin_summary[column], errors="coerce")

    if layout == "three_panel":
        fig = plt.figure(figsize=(11.2, 7.15))
        gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.28], hspace=0.40, wspace=0.24)
        ax_a = fig.add_subplot(gs[0, 0])
        ax_b = fig.add_subplot(gs[0, 1])
        ax_c = fig.add_subplot(gs[1, :])

        plot_metric_lines(ax_a, bin_summary, "balanced_acc", "A. Balanced accuracy vs motion bin")
        plot_metric_lines(ax_b, bin_summary, "auroc", "B. AUROC vs motion bin")
        plot_gap_bars(ax_c, subject_gap, "galaxy", "C. Galaxy PPG per-subject gap reduction", n_boot, seed)

        handles, labels = ax_b.get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.908),
            ncol=4,
            fontsize=7.8,
            frameon=True,
            framealpha=0.96,
            edgecolor="#C8CDD8",
            columnspacing=1.3,
            handlelength=2.2,
        )

        fig.suptitle(title, fontsize=14, fontweight="bold", y=0.992)
        if subtitle:
            fig.text(0.5, 0.948, subtitle, ha="center", va="center", fontsize=9.5, color=TEXT)
        fig.subplots_adjust(left=0.075, right=0.985, top=0.835, bottom=0.075)
    else:
        fig = plt.figure(figsize=(11.2, 7.2))
        gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.22], hspace=0.36, wspace=0.24)
        ax_a = fig.add_subplot(gs[0, 0])
        ax_b = fig.add_subplot(gs[0, 1])
        ax_c = fig.add_subplot(gs[1, 0])
        ax_d = fig.add_subplot(gs[1, 1])

        plot_metric_lines(ax_a, bin_summary, "balanced_acc", "A. Balanced accuracy vs motion bin")
        plot_metric_lines(ax_b, bin_summary, "auroc", "B. AUROC vs motion bin")
        handles, labels = ax_b.get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.912),
            ncol=4,
            fontsize=7.8,
            frameon=True,
            framealpha=0.96,
            edgecolor="#C8CDD8",
            columnspacing=1.3,
            handlelength=2.2,
        )

        plot_gap_bars(ax_c, subject_gap, "galaxy", "C. Galaxy PPG per-subject gap reduction", n_boot, seed)
        plot_gap_bars(ax_d, subject_gap, "wesad", "D. WESAD per-subject gap reduction", n_boot, seed + 19)

        fig.suptitle(title, fontsize=14, fontweight="bold", y=0.985)
        if subtitle:
            fig.text(0.5, 0.943, subtitle, ha="center", va="center", fontsize=9.5, color=TEXT)
        fig.subplots_adjust(left=0.075, right=0.985, top=0.86, bottom=0.075)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=450, bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot a combined Galaxy/WESAD motion-aware mechanism figure.")
    parser.add_argument("--galaxy-bin-csv", type=Path, required=True)
    parser.add_argument("--galaxy-gap-csv", type=Path, required=True)
    parser.add_argument("--wesad-bin-csv", type=Path, required=True)
    parser.add_argument("--wesad-gap-csv", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--font-family", type=str, default="Liberation Sans")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layout", type=str, default="three_panel", choices=["three_panel", "four_panel"])
    parser.add_argument("--baseline-label", type=str, default="Watch-only")
    parser.add_argument("--motion-label", type=str, default="Motion-aware")
    parser.add_argument("--title", type=str, default="Motion-aware watch encoder mechanism")
    parser.add_argument(
        "--subtitle",
        type=str,
        default="Motion-aware modeling reduces sensitivity to high-motion windows, with stronger gains on Galaxy PPG.",
    )
    args = parser.parse_args()
    METHOD_LABELS["watch_only"] = args.baseline_label
    METHOD_LABELS["motion_aware"] = args.motion_label

    galaxy_bin = read_csv(
        args.galaxy_bin_csv,
        "galaxy",
        required=["dataset", "method", "motion_bin", "balanced_acc", "auroc"],
    )
    wesad_bin = read_csv(
        args.wesad_bin_csv,
        "wesad",
        required=["dataset", "method", "motion_bin", "balanced_acc", "auroc"],
    )
    galaxy_gap = read_csv(
        args.galaxy_gap_csv,
        "galaxy",
        required=["dataset", "subject_id", "motion_sensitivity_gap_reduction"],
    )
    wesad_gap = read_csv(
        args.wesad_gap_csv,
        "wesad",
        required=["dataset", "subject_id", "motion_sensitivity_gap_reduction"],
    )
    bin_summary = pd.concat([galaxy_bin, wesad_bin], ignore_index=True)
    subject_gap = pd.concat([galaxy_gap, wesad_gap], ignore_index=True)

    make_plot(
        bin_summary=bin_summary,
        subject_gap=subject_gap,
        output_prefix=args.output_prefix,
        title=args.title,
        subtitle=args.subtitle,
        font_family=args.font_family,
        n_boot=args.bootstrap,
        seed=args.seed,
        layout=args.layout,
    )
    print(f"Saved figure to {args.output_prefix.with_suffix('.png')}")
    print(f"Saved figure to {args.output_prefix.with_suffix('.pdf')}")
    print(f"Saved figure to {args.output_prefix.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
