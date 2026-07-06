from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_METHOD_ORDER = [
    "Watch-only",
    "Watch-only+Motion",
    "Pure-KD",
    "KD+Motion",
    "SGPC+Motion",
    "Ours",
]

DEFAULT_METRICS = ["balanced_acc", "auroc", "f1", "collapse", "positive_rate_error"]

METRIC_LABELS = {
    "balanced_acc": "Balanced accuracy",
    "auroc": "AUROC",
    "f1": "F1",
    "collapse": "Collapse",
    "positive_rate_error": "Positive-rate error",
}

LOWER_IS_BETTER = {"collapse", "positive_rate_error"}

METHOD_COLORS = {
    "Watch-only": "#8c8c8c",
    "Watch-only+Motion": "#4c78a8",
    "Pure-KD": "#b279a2",
    "KD+Motion": "#72b7b2",
    "SGPC+Motion": "#f58518",
    "Ours": "#2f855a",
}


def parse_order(values: list[str] | None) -> list[str]:
    if not values:
        return DEFAULT_METHOD_ORDER
    return [str(item) for item in values]


def ensure_numeric(frame: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for metric in metrics:
        if metric not in out.columns:
            raise ValueError(f"Missing metric column: {metric}")
        out[metric] = pd.to_numeric(out[metric], errors="coerce")
    return out


def paired_wide(frame: pd.DataFrame, metric: str, method_order: list[str]) -> pd.DataFrame:
    selected = frame[frame["method"].isin(method_order)][["subject", "method", metric]].copy()
    wide = selected.pivot_table(index="subject", columns="method", values=metric, aggfunc="first")
    available = [method for method in method_order if method in wide.columns]
    return wide[available].sort_index()


def method_summary(wide: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    return wide.mean(axis=0, skipna=True), wide.std(axis=0, skipna=True)


def plot_metric(
    frame: pd.DataFrame,
    metric: str,
    method_order: list[str],
    output_path: Path,
    reference_method: str,
    title: str | None = None,
    dpi: int = 300,
) -> None:
    wide = paired_wide(frame, metric, method_order)
    if wide.empty:
        raise ValueError(f"No data available for metric {metric}.")
    methods = list(wide.columns)
    x = np.arange(len(methods))
    means, stds = method_summary(wide)

    fig, ax = plt.subplots(figsize=(max(7.5, 1.15 * len(methods)), 4.8))
    for subject, row in wide.iterrows():
        values = row.to_numpy(dtype=float)
        mask = np.isfinite(values)
        if mask.sum() < 2:
            continue
        ax.plot(
            x[mask],
            values[mask],
            color="#c7c7c7",
            linewidth=0.9,
            alpha=0.55,
            zorder=1,
        )

    colors = [METHOD_COLORS.get(method, "#4a5568") for method in methods]
    ax.errorbar(
        x,
        means.loc[methods],
        yerr=stds.loc[methods],
        fmt="none",
        ecolor="#2d3748",
        elinewidth=1.0,
        capsize=3,
        alpha=0.65,
        zorder=2,
    )
    ax.scatter(
        x,
        means.loc[methods],
        s=[74 if method == reference_method else 58 for method in methods],
        color=colors,
        edgecolor="#1a202c",
        linewidth=0.8,
        zorder=3,
    )

    best_idx = None
    if metric in LOWER_IS_BETTER:
        best_idx = int(np.nanargmin(means.loc[methods].to_numpy(dtype=float)))
    else:
        best_idx = int(np.nanargmax(means.loc[methods].to_numpy(dtype=float)))
    ax.scatter(
        [best_idx],
        [means.loc[methods].iloc[best_idx]],
        s=140,
        facecolors="none",
        edgecolors="#111827",
        linewidth=1.3,
        zorder=4,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=25, ha="right")
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.set_title(title or f"Per-subject paired {METRIC_LABELS.get(metric, metric)}")
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if metric not in {"positive_rate_error"}:
        ax.set_ylim(bottom=max(0.0, float(np.nanmin(wide.to_numpy(dtype=float))) - 0.08))
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_panel(
    frame: pd.DataFrame,
    metrics: list[str],
    method_order: list[str],
    output_path: Path,
    reference_method: str,
    dpi: int = 300,
) -> None:
    ncols = 2
    nrows = int(np.ceil(len(metrics) / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12.5, 4.0 * nrows))
    axes_arr = np.asarray(axes).reshape(-1)

    for ax, metric in zip(axes_arr, metrics):
        wide = paired_wide(frame, metric, method_order)
        methods = list(wide.columns)
        x = np.arange(len(methods))
        means, stds = method_summary(wide)
        for _, row in wide.iterrows():
            values = row.to_numpy(dtype=float)
            mask = np.isfinite(values)
            if mask.sum() < 2:
                continue
            ax.plot(x[mask], values[mask], color="#c7c7c7", linewidth=0.8, alpha=0.5, zorder=1)
        colors = [METHOD_COLORS.get(method, "#4a5568") for method in methods]
        ax.errorbar(
            x,
            means.loc[methods],
            yerr=stds.loc[methods],
            fmt="none",
            ecolor="#2d3748",
            elinewidth=0.9,
            capsize=2.5,
            alpha=0.65,
            zorder=2,
        )
        ax.scatter(
            x,
            means.loc[methods],
            s=[68 if method == reference_method else 52 for method in methods],
            color=colors,
            edgecolor="#1a202c",
            linewidth=0.75,
            zorder=3,
        )
        ax.set_title(METRIC_LABELS.get(metric, metric))
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=25, ha="right")
        ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for ax in axes_arr[len(metrics) :]:
        ax.axis("off")

    fig.suptitle("Per-subject paired core ablation results", y=1.0, fontsize=14)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot per-subject paired Galaxy core ablation results.")
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metrics", nargs="*", default=DEFAULT_METRICS)
    parser.add_argument("--method-order", nargs="*", default=None)
    parser.add_argument("--reference-method", type=str, default="Ours")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    method_order = parse_order(args.method_order)
    metrics = [str(metric) for metric in args.metrics]
    frame = pd.read_csv(args.input_csv)
    if not {"method", "subject"}.issubset(frame.columns):
        raise ValueError("--input-csv must contain method and subject columns.")
    frame = ensure_numeric(frame, metrics)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for metric in metrics:
        plot_metric(
            frame,
            metric=metric,
            method_order=method_order,
            output_path=args.output_dir / f"paired_{metric}.png",
            reference_method=args.reference_method,
            dpi=args.dpi,
        )
    plot_panel(
        frame,
        metrics=metrics,
        method_order=method_order,
        output_path=args.output_dir / "paired_core_ablation_panel.png",
        reference_method=args.reference_method,
        dpi=args.dpi,
    )
    plot_panel(
        frame,
        metrics=metrics,
        method_order=method_order,
        output_path=args.output_dir / "paired_core_ablation_panel.pdf",
        reference_method=args.reference_method,
        dpi=args.dpi,
    )
    print(f"Saved paired plots to {args.output_dir}")


if __name__ == "__main__":
    main()
