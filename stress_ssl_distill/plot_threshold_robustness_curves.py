from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METHOD_LABELS = {
    "ppg_only": "PPG/BVP-only",
    "bvp_only": "BVP-only",
    "simple_concat": "Simple concat",
    "motion_aware": "Motion-aware",
    "no_kd": "No KD",
    "standard_kd": "Standard KD",
    "pure_kd": "Standard KD",
    "cross_gated_kd": "Cross-gated KD",
    "teacher_gated_kd": "Teacher-gated KD",
    "student_gated_kd": "Student-gated KD",
}

DATASET_LABELS = {
    "galaxy": "Galaxy PPG",
    "wesad": "WESAD",
}

COLORS = {
    "ppg_only": "#7AA6D9",
    "bvp_only": "#7AA6D9",
    "simple_concat": "#9B79C6",
    "motion_aware": "#2F73C8",
    "no_kd": "#5DA5DA",
    "standard_kd": "#4C9A5A",
    "pure_kd": "#4C9A5A",
    "cross_gated_kd": "#D55E00",
    "teacher_gated_kd": "#CC79A7",
    "student_gated_kd": "#F0A202",
}

MARKERS = {
    "ppg_only": "s",
    "bvp_only": "s",
    "simple_concat": "D",
    "motion_aware": "o",
    "no_kd": "s",
    "standard_kd": "o",
    "pure_kd": "o",
    "cross_gated_kd": "^",
    "teacher_gated_kd": "v",
    "student_gated_kd": "P",
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#B8C2CC",
            "grid.alpha": 0.28,
            "grid.linewidth": 0.8,
            "legend.frameon": True,
            "legend.framealpha": 0.96,
            "legend.edgecolor": "#C7CED8",
            "savefig.dpi": 450,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _ordered_present(values: list[str], present: set[str]) -> list[str]:
    return [value for value in values if value in present]


def _sem(std: pd.Series, n: pd.Series) -> np.ndarray:
    return (std.astype(float) / np.sqrt(n.astype(float).clip(lower=1))).to_numpy()


def _plot_metric(
    ax: plt.Axes,
    frame: pd.DataFrame,
    metric: str,
    methods: list[str],
    show_ylabel: bool,
) -> list[tuple[object, str]]:
    metric_mean = f"{metric}_mean"
    metric_std = f"{metric}_std"
    handles: list[tuple[object, str]] = []

    for method in methods:
        sub = frame[frame["method"] == method].sort_values("offset")
        if sub.empty:
            continue
        x = sub["offset"].astype(float).to_numpy()
        y = sub[metric_mean].astype(float).to_numpy()
        err = _sem(sub[metric_std], sub["n_subjects"])
        color = COLORS.get(method, "#4E79A7")
        marker = MARKERS.get(method, "o")
        (line,) = ax.plot(
            x,
            y,
            marker=marker,
            markersize=5.0,
            linewidth=2.0,
            color=color,
            label=METHOD_LABELS.get(method, method),
        )
        ax.fill_between(x, y - err, y + err, color=color, alpha=0.14, linewidth=0)
        handles.append((line, METHOD_LABELS.get(method, method)))

    ax.axvline(0.0, color="#424B57", linewidth=1.0, linestyle="--", alpha=0.75)
    ax.set_xlabel("Threshold offset")
    if show_ylabel:
        ylabel = "Balanced accuracy" if metric == "balanced_accuracy" else "F1"
        ax.set_ylabel(ylabel)
    else:
        ax.set_ylabel("")
    ax.set_xticks(sorted(frame["offset"].astype(float).unique()))
    ax.tick_params(axis="both", labelsize=9)
    return handles


def plot_curves(
    summary: pd.DataFrame,
    output_dir: Path,
    group: str | None,
    datasets: list[str] | None,
    methods: list[str] | None,
    title: str | None,
    stem: str,
) -> None:
    setup_style()
    frame = summary.copy()
    if group is not None:
        frame = frame[frame["group"].astype(str) == group]
    if datasets:
        frame = frame[frame["dataset"].astype(str).isin(datasets)]
    if methods:
        frame = frame[frame["method"].astype(str).isin(methods)]
    if frame.empty:
        raise ValueError("No rows left after filtering. Check --group/--datasets/--methods.")

    required = {
        "dataset",
        "group",
        "method",
        "offset",
        "n_subjects",
        "balanced_accuracy_mean",
        "balanced_accuracy_std",
        "f1_mean",
        "f1_std",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Summary CSV missing columns: {sorted(missing)}")

    present_datasets = list(dict.fromkeys(frame["dataset"].astype(str).tolist()))
    if datasets:
        present_datasets = _ordered_present(datasets, set(present_datasets))
    present_methods = list(dict.fromkeys(frame["method"].astype(str).tolist()))
    if methods:
        present_methods = _ordered_present(methods, set(present_methods))

    n_rows = len(present_datasets)
    fig, axes = plt.subplots(
        n_rows,
        2,
        figsize=(7.2, max(2.7 * n_rows, 2.9)),
        sharex=True,
        squeeze=False,
    )

    legend_items: dict[str, object] = {}
    for row_idx, dataset in enumerate(present_datasets):
        dataset_frame = frame[frame["dataset"].astype(str) == dataset]
        row_label = DATASET_LABELS.get(dataset, dataset)
        for col_idx, metric in enumerate(("balanced_accuracy", "f1")):
            ax = axes[row_idx][col_idx]
            handles = _plot_metric(
                ax,
                dataset_frame,
                metric=metric,
                methods=present_methods,
                show_ylabel=col_idx == 0,
            )
            for handle, label in handles:
                legend_items.setdefault(label, handle)
            panel = chr(ord("A") + row_idx * 2 + col_idx)
            metric_title = "Balanced accuracy" if metric == "balanced_accuracy" else "F1"
            ax.set_title(f"{panel}. {row_label}: {metric_title}", loc="left", fontweight="bold")

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)

    if legend_items:
        fig.legend(
            list(legend_items.values()),
            list(legend_items.keys()),
            loc="upper center",
            bbox_to_anchor=(0.5, 1.035 if not title else 1.075),
            ncol=min(len(legend_items), 5),
            fontsize=9,
        )

    fig.tight_layout(h_pad=1.2, w_pad=1.8)
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(output_dir / f"{stem}.{ext}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot BA/F1 curves under threshold offsets.")
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--group", type=str, default=None)
    parser.add_argument("--datasets", nargs="*", default=None, help="Dataset order, e.g. galaxy wesad.")
    parser.add_argument("--methods", nargs="*", default=None, help="Method order to plot.")
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--stem", type=str, default=None)
    args = parser.parse_args()

    summary = pd.read_csv(args.summary_csv)
    stem = args.stem
    if stem is None:
        parts = ["threshold_robustness_curves"]
        if args.group:
            parts.append(str(args.group))
        stem = "_".join(parts)
    plot_curves(
        summary=summary,
        output_dir=args.output_dir,
        group=args.group,
        datasets=args.datasets,
        methods=args.methods,
        title=args.title,
        stem=stem,
    )
    print(f"saved={args.output_dir / (stem + '.png')}")
    print(f"saved={args.output_dir / (stem + '.pdf')}")
    print(f"saved={args.output_dir / (stem + '.svg')}")


if __name__ == "__main__":
    main()
