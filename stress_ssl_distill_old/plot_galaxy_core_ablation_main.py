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

DEFAULT_FOCUS_METHODS = ["Watch-only", "Watch-only+Motion", "Ours"]
DEFAULT_CONTRAST_METHODS = ["Pure-KD"]
DEFAULT_METRICS = ["balanced_acc", "auroc", "f1"]

METRIC_LABELS = {
    "balanced_acc": "Balanced accuracy",
    "auroc": "AUROC",
    "f1": "F1",
}

METHOD_COLORS = {
    "Watch-only": "#8c8c8c",
    "Watch-only+Motion": "#4c78a8",
    "Pure-KD": "#b279a2",
    "KD+Motion": "#72b7b2",
    "SGPC+Motion": "#f58518",
    "Ours": "#2f855a",
}


def parse_values(values: list[str] | None, fallback: list[str], allow_empty: bool = False) -> list[str]:
    if values is None:
        return list(fallback)
    if len(values) == 0 and allow_empty:
        return []
    if len(values) == 0:
        return list(fallback)
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


def plot_main_panel(
    frame: pd.DataFrame,
    metrics: list[str],
    method_order: list[str],
    focus_methods: list[str],
    contrast_methods: list[str],
    output_path: Path,
    dpi: int,
    title: str | None,
) -> None:
    missing_focus = [method for method in focus_methods if method not in method_order]
    if missing_focus:
        raise ValueError(f"Focus methods must be included in method order: {missing_focus}")
    missing_contrast = [method for method in contrast_methods if method not in method_order]
    if missing_contrast:
        raise ValueError(f"Contrast methods must be included in method order: {missing_contrast}")
    if not focus_methods:
        raise ValueError("At least one focus method is required.")

    ncols = len(metrics)
    fig, axes = plt.subplots(1, ncols, figsize=(4.25 * ncols, 3.8), sharey=False)
    axes_arr = np.asarray(axes).reshape(-1)
    x_all = np.arange(len(method_order))
    baseline_method = focus_methods[0]

    for ax, metric in zip(axes_arr, metrics):
        wide = paired_wide(frame, metric, method_order)
        if wide.empty:
            raise ValueError(f"No data available for metric {metric}.")

        available_methods = list(wide.columns)
        x_available = np.asarray([method_order.index(method) for method in available_methods], dtype=float)
        focus_available = [method for method in focus_methods if method in wide.columns]
        focus_x_available = np.asarray([method_order.index(method) for method in focus_available], dtype=float)

        for _, row in wide.iterrows():
            values = row.reindex(focus_available).to_numpy(dtype=float)
            mask = np.isfinite(values)
            if mask.sum() < 2:
                continue
            ax.plot(
                focus_x_available[mask],
                values[mask],
                color="#c9cdd3",
                linewidth=0.85,
                alpha=0.42,
                zorder=1,
            )

        means = wide.mean(axis=0, skipna=True)
        if baseline_method in wide.columns:
            baseline_x = float(method_order.index(baseline_method))
            for contrast_method in contrast_methods:
                if contrast_method not in wide.columns:
                    continue
                contrast_x = float(method_order.index(contrast_method))
                for _, row in wide[[baseline_method, contrast_method]].iterrows():
                    values = row.to_numpy(dtype=float)
                    if not np.all(np.isfinite(values)):
                        continue
                    ax.plot(
                        [baseline_x, contrast_x],
                        values,
                        color=METHOD_COLORS.get(contrast_method, "#9f7aea"),
                        linewidth=0.75,
                        alpha=0.23,
                        linestyle=(0, (3, 2)),
                        zorder=1,
                    )
                if np.isfinite(means.loc[[baseline_method, contrast_method]].to_numpy(dtype=float)).all():
                    ax.plot(
                        [baseline_x, contrast_x],
                        [means.loc[baseline_method], means.loc[contrast_method]],
                        color=METHOD_COLORS.get(contrast_method, "#9f7aea"),
                        linewidth=1.25,
                        alpha=0.78,
                        linestyle=(0, (3, 2)),
                        zorder=2,
                    )

        focus_means = means.reindex(focus_available).to_numpy(dtype=float)
        ax.plot(
            focus_x_available,
            focus_means,
            color="#1f2937",
            linewidth=1.45,
            alpha=0.72,
            zorder=2,
        )

        for method, x in zip(available_methods, x_available):
            is_focus = method in focus_methods
            ax.scatter(
                [x],
                [means.loc[method]],
                s=90 if method == "Ours" else (72 if is_focus else 50),
                color=METHOD_COLORS.get(method, "#4a5568"),
                edgecolor="#111827",
                linewidth=0.8 if is_focus else 0.6,
                alpha=1.0 if is_focus else 0.78,
                zorder=3,
            )
            if is_focus or method in contrast_methods:
                ax.text(
                    x,
                    float(means.loc[method]) + 0.025,
                    f"{means.loc[method]:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="#1f2937",
                )

        ax.set_title(METRIC_LABELS.get(metric, metric), fontsize=12)
        ax.set_xticks(x_all)
        ax.set_xticklabels(method_order, rotation=30, ha="right", fontsize=8)
        ax.set_xlim(-0.45, len(method_order) - 0.55)
        values = wide.to_numpy(dtype=float)
        finite_values = values[np.isfinite(values)]
        if len(finite_values):
            y_min = max(0.0, float(np.nanmin(finite_values)) - 0.08)
            y_max = min(1.05, float(np.nanmax(finite_values)) + 0.08)
            if y_max - y_min < 0.25:
                center = (y_min + y_max) * 0.5
                y_min = max(0.0, center - 0.15)
                y_max = min(1.05, center + 0.15)
            ax.set_ylim(y_min, y_max)
        ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylabel("Score")

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=METHOD_COLORS.get(method, "#4a5568"),
            markeredgecolor="#111827",
            markersize=7,
            label=method,
        )
        for method in method_order
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=min(len(method_order), 6),
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, -0.05),
    )
    if title:
        fig.suptitle(title, y=1.04, fontsize=13)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot a clean main-paper Galaxy core ablation figure with subject-level paired focus lines."
    )
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metrics", nargs="*", default=DEFAULT_METRICS)
    parser.add_argument("--method-order", nargs="*", default=None)
    parser.add_argument("--focus-methods", nargs="*", default=None)
    parser.add_argument("--contrast-methods", nargs="*", default=None)
    parser.add_argument("--formats", nargs="*", default=["png", "pdf"])
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--connect-all", action="store_true")
    parser.add_argument("--separate-metrics", action="store_true")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    method_order = parse_values(args.method_order, DEFAULT_METHOD_ORDER)
    if args.connect_all:
        focus_methods = list(method_order)
        contrast_methods = []
    else:
        focus_methods = parse_values(args.focus_methods, DEFAULT_FOCUS_METHODS)
        contrast_methods = parse_values(args.contrast_methods, DEFAULT_CONTRAST_METHODS, allow_empty=True)
    metrics = [str(metric) for metric in args.metrics]

    frame = pd.read_csv(args.input_csv)
    if not {"method", "subject"}.issubset(frame.columns):
        raise ValueError("--input-csv must contain method and subject columns.")
    frame = ensure_numeric(frame, metrics)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in args.formats:
        output_path = args.output_dir / f"core_ablation_main_paired.{fmt.lower()}"
        plot_main_panel(
            frame=frame,
            metrics=metrics,
            method_order=method_order,
            focus_methods=focus_methods,
            contrast_methods=contrast_methods,
            output_path=output_path,
            dpi=args.dpi,
            title=args.title,
        )
        if args.separate_metrics:
            for metric in metrics:
                metric_output_path = args.output_dir / f"core_ablation_main_{metric}.{fmt.lower()}"
                plot_main_panel(
                    frame=frame,
                    metrics=[metric],
                    method_order=method_order,
                    focus_methods=focus_methods,
                    contrast_methods=contrast_methods,
                    output_path=metric_output_path,
                    dpi=args.dpi,
                    title=args.title,
                )
    print(f"Saved main-paper core ablation paired figure to {args.output_dir}")


if __name__ == "__main__":
    main()
