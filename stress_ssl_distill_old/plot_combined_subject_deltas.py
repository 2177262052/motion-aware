from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


METRIC_LABELS = {
    "balanced_acc": "Balanced accuracy",
    "auroc": "AUROC",
    "f1": "F1",
    "collapse": "Collapse rate",
    "positive_rate_error": "Positive-rate error",
}

X_LABELS = {
    "balanced_acc": r"$\Delta$ Balanced accuracy",
    "auroc": r"$\Delta$ AUROC",
    "f1": r"$\Delta$ F1",
    "collapse": r"$\Delta$ Collapse rate",
    "positive_rate_error": r"$\Delta$ Positive-rate error",
}

POSITIVE = "#4F82BD"
NEGATIVE = "#D94A26"
GRID = "#D9DFEA"
TEXT = "#1F2937"
GALAXY = "#2F66B3"
WESAD = "#3D7E2A"


def natural_subject_key(subject: object) -> tuple[str, int, str]:
    text = str(subject)
    match = re.search(r"(\d+)", text)
    prefix = text[: match.start()] if match else text
    number = int(match.group(1)) if match else 10_000
    return prefix, number, text


def normalize_text(value: object) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def read_delta_csv(
    path: Path,
    dataset_name: str,
    candidate_keyword: str,
    reference_keyword: str,
    comparison_keyword: str | None,
) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"subject", "metric"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    if "delta" not in df.columns:
        if {"candidate_value", "reference_value"}.issubset(df.columns):
            df = df.copy()
            df["delta"] = pd.to_numeric(df["candidate_value"], errors="coerce") - pd.to_numeric(
                df["reference_value"], errors="coerce"
            )
        else:
            raise ValueError(f"{path} needs either delta or candidate_value/reference_value columns.")

    filtered = df.copy()
    if comparison_keyword and "comparison" in filtered.columns:
        key = normalize_text(comparison_keyword)
        mask = filtered["comparison"].map(normalize_text).str.contains(key, regex=False)
        if mask.any():
            filtered = filtered[mask].copy()

    if {"candidate", "reference"}.issubset(filtered.columns):
        candidate_key = normalize_text(candidate_keyword)
        reference_key = normalize_text(reference_keyword)
        mask = (
            filtered["candidate"].map(normalize_text).str.contains(candidate_key, regex=False)
            & filtered["reference"].map(normalize_text).str.contains(reference_key, regex=False)
        )
        if mask.any():
            filtered = filtered[mask].copy()

    if "comparison" in filtered.columns and filtered["comparison"].nunique() > 1:
        # Keep the first comparison in file order, but print enough context to catch accidental mixed inputs.
        first = filtered["comparison"].dropna().astype(str).iloc[0]
        available = ", ".join(filtered["comparison"].dropna().astype(str).unique()[:8])
        print(f"warning_multiple_comparisons dataset={dataset_name} keeping={first!r}; available={available}")
        filtered = filtered[filtered["comparison"].astype(str) == first].copy()

    filtered["dataset"] = dataset_name
    filtered["delta"] = pd.to_numeric(filtered["delta"], errors="coerce")
    filtered["metric"] = filtered["metric"].astype(str)
    filtered["subject"] = filtered["subject"].astype(str)
    filtered = filtered[np.isfinite(filtered["delta"])].copy()
    return filtered


def metric_frame(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    out = df[df["metric"] == metric].copy()
    if out.empty:
        raise ValueError(f"No rows found for metric={metric!r}. Available metrics: {sorted(df['metric'].unique())}")
    if out["subject"].duplicated().any():
        out = out.groupby(["dataset", "subject", "metric"], as_index=False)["delta"].mean()
    out["_sort"] = out["delta"]
    out["_subject_sort"] = out["subject"].map(natural_subject_key)
    return out.sort_values(["_sort", "_subject_sort"], ascending=[False, True]).drop(
        columns=["_sort", "_subject_sort"]
    )


def padded_xlim(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return -1.0, 1.0
    low = min(float(finite.min()), 0.0)
    high = max(float(finite.max()), 0.0)
    span = max(high - low, 1e-6)
    pad = max(0.025, span * 0.10)
    return low - pad, high + pad


def format_delta(value: float) -> str:
    return f"{value:+.3f}"


def add_summary_box(ax, deltas: np.ndarray) -> None:
    wins = int(np.sum(deltas > 1e-12))
    losses = int(np.sum(deltas < -1e-12))
    ties = int(len(deltas) - wins - losses)
    mean = float(np.mean(deltas)) if len(deltas) else float("nan")
    text = (
        "Mean $\\Delta$ = "
        + rf"$\mathbf{{{format_delta(mean)}}}$"
        + "\n\n"
        + "Wins / Losses / Ties\n"
        + rf"$\mathbf{{{wins}}}$ / {losses} / {ties}"
    )
    ax.text(
        0.965,
        0.105,
        text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.5,
        fontweight="normal",
        color=TEXT,
        bbox={
            "boxstyle": "round,pad=0.55,rounding_size=0.25",
            "facecolor": "white",
            "edgecolor": "#B8BDC8",
            "linewidth": 0.8,
            "alpha": 0.96,
        },
    )


def draw_panel(ax, df: pd.DataFrame, metric: str, accent: str) -> None:
    data = metric_frame(df, metric)
    y = np.arange(len(data))
    deltas = data["delta"].to_numpy(dtype=float)
    colors = np.where(deltas >= 0, POSITIVE, NEGATIVE)

    ax.barh(y, deltas, color=colors, height=0.66, edgecolor="white", linewidth=0.4)
    ax.axvline(0, color="#20242B", linestyle="--", linewidth=1.0, alpha=0.82, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(data["subject"].tolist(), fontsize=8, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlabel(X_LABELS.get(metric, rf"$\Delta$ {metric}"), fontsize=9, fontweight="bold")
    ax.set_title(
        f"{METRIC_LABELS.get(metric, metric)} ($\\Delta$)",
        loc="center",
        fontsize=10.5,
        fontweight="bold",
        color=accent,
        pad=8,
    )
    ax.set_xlim(*padded_xlim(deltas))
    ax.xaxis.grid(True, color=GRID, linewidth=0.8, alpha=0.75)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", length=0)
    for label in ax.get_xticklabels():
        label.set_fontweight("bold")
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#555A64")
    ax.spines["bottom"].set_linewidth(0.8)
    add_summary_box(ax, deltas)


def make_figure(
    galaxy: pd.DataFrame,
    wesad: pd.DataFrame,
    output_prefix: Path,
    metrics: list[str],
    title: str,
    subtitle: str,
) -> None:
    import matplotlib.pyplot as plt

    if len(metrics) != 2:
        raise ValueError("This combined paper figure expects exactly two metrics, e.g. balanced_acc and auroc.")

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.weight": "normal",
            "axes.titlesize": 10.5,
            "axes.labelsize": 9,
            "axes.titleweight": "normal",
            "axes.labelweight": "normal",
            "figure.dpi": 160,
            "savefig.dpi": 450,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(11.6, 6.6), constrained_layout=False)
    draw_panel(axes[0, 0], galaxy, metrics[0], GALAXY)
    draw_panel(axes[0, 1], galaxy, metrics[1], GALAXY)
    draw_panel(axes[1, 0], wesad, metrics[0], WESAD)
    draw_panel(axes[1, 1], wesad, metrics[1], WESAD)

    if subtitle:
        fig.text(0.5, 0.967, subtitle, ha="center", va="center", fontsize=10.5, fontweight="bold", color="#111827")
    if title:
        fig.text(0.5, 0.025, title, ha="center", va="center", fontsize=14, fontweight="bold", color="#111827")

    galaxy_n = galaxy[galaxy["metric"] == metrics[0]]["subject"].nunique()
    wesad_n = wesad[wesad["metric"] == metrics[0]]["subject"].nunique()
    fig.text(
        0.018,
        0.705,
        f"Galaxy PPG (N = {galaxy_n})",
        rotation=90,
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color="white",
        bbox={"boxstyle": "round,pad=0.35,rounding_size=0.18", "facecolor": GALAXY, "edgecolor": GALAXY},
    )
    fig.text(
        0.018,
        0.295,
        f"WESAD (N = {wesad_n})",
        rotation=90,
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color="white",
        bbox={"boxstyle": "round,pad=0.35,rounding_size=0.18", "facecolor": WESAD, "edgecolor": WESAD},
    )

    fig.subplots_adjust(left=0.095, right=0.985, top=0.925, bottom=0.115, hspace=0.38, wspace=0.18)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=450, bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw a paper-style 2x2 per-subject delta figure for Galaxy and WESAD.")
    parser.add_argument("--galaxy-csv", type=Path, required=True)
    parser.add_argument("--wesad-csv", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--metric", action="append", default=None, help="Repeat exactly twice. Default: balanced_acc, auroc.")
    parser.add_argument("--candidate-keyword", type=str, default="ours")
    parser.add_argument("--reference-keyword", type=str, default="watch")
    parser.add_argument("--galaxy-comparison", type=str, default=None)
    parser.add_argument("--wesad-comparison", type=str, default=None)
    parser.add_argument("--title", type=str, default="Per-subject paired deltas (ours vs watch-only)")
    parser.add_argument("--subtitle", type=str, default="Positive values indicate improvement")
    args = parser.parse_args()

    metrics = args.metric or ["balanced_acc", "auroc"]
    galaxy = read_delta_csv(
        args.galaxy_csv,
        dataset_name="galaxy",
        candidate_keyword=args.candidate_keyword,
        reference_keyword=args.reference_keyword,
        comparison_keyword=args.galaxy_comparison,
    )
    wesad = read_delta_csv(
        args.wesad_csv,
        dataset_name="wesad",
        candidate_keyword=args.candidate_keyword,
        reference_keyword=args.reference_keyword,
        comparison_keyword=args.wesad_comparison,
    )
    make_figure(
        galaxy=galaxy,
        wesad=wesad,
        output_prefix=args.output_prefix,
        metrics=metrics,
        title=args.title,
        subtitle=args.subtitle,
    )
    print(f"Saved {args.output_prefix.with_suffix('.png')}")
    print(f"Saved {args.output_prefix.with_suffix('.pdf')}")
    print(f"Saved {args.output_prefix.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
