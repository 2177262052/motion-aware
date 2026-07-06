from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


BLUE = "#2F6FBB"
LIGHT_BLUE = "#7EA6D8"
ORANGE = "#E6863B"
GREEN = "#4C9A4C"
RED = "#D85C3A"
PURPLE = "#7B61B3"
GRID = "#D8DDE6"
TEXT = "#111827"

COUPLING_ORDER = ["low", "mid", "high"]
COUPLING_LABELS = ["Low", "Mid", "High"]
VALIDITY_ORDER = ["low", "high"]
MODULES = [
    ("adapt_strength", "Adapt", BLUE),
    ("clean_strength", "Clean", ORANGE),
    ("refine_strength", "Refine", GREEN),
]
SIGNATURE_VARIABLES = [
    ("acc_jerk", "ACC jerk"),
    ("ppg_acc_coupling", "PPG--ACC coupling"),
    ("ppg_validity", "PPG validity"),
    ("adapt_strength", "Adapt strength"),
    ("clean_strength", "Clean strength"),
]
BASELINE_COLORS = {
    "PPG-only": BLUE,
    "BVP-only": BLUE,
    "PPG/BVP-only": BLUE,
    "Simple concat": PURPLE,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a cross-dataset 2x3 motion-aware internal mechanism figure from exported CSV files."
    )
    parser.add_argument("--galaxy-dir", type=Path, required=True)
    parser.add_argument("--wesad-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--font-family", type=str, default="Arial")
    parser.add_argument("--dpi", type=int, default=450)
    return parser.parse_args()


def _read_csv(output_dir: Path, dataset_kind: str, suffix: str) -> pd.DataFrame:
    path = output_dir / f"{dataset_kind}_motion_internal_{suffix}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing required CSV: {path}")
    return pd.read_csv(path)


def load_dataset(output_dir: Path, dataset_kind: str) -> dict[str, pd.DataFrame]:
    response = _read_csv(output_dir, dataset_kind, "response_by_coupling")
    heatmap = _read_csv(output_dir, dataset_kind, "gain_heatmap")
    signature = _read_csv(output_dir, dataset_kind, "signature_dot_whisker")
    if "true_class_prob_gain_mean" not in heatmap.columns:
        raise ValueError(
            f"{output_dir} was produced by an older analyzer. Rerun "
            "analyze_motion_internal_response.py so Panel B can use true-class probability gain."
        )
    return {"response": response, "heatmap": heatmap, "signature": signature}


def setup_matplotlib(font_family: str):
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    resolved_font = font_family if font_family in available_fonts else "DejaVu Sans"

    plt.rcParams.update(
        {
            "font.family": resolved_font,
            "font.size": 8.5,
            "axes.titlesize": 10,
            "axes.labelsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.alpha": 0.42,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )
    return plt


def baseline_order(heatmap: pd.DataFrame) -> list[str]:
    names = list(dict.fromkeys(heatmap["baseline_name"].astype(str).tolist()))
    preferred = ["PPG-only", "BVP-only", "PPG/BVP-only", "Simple concat"]
    ordered = [name for name in preferred if name in names]
    ordered.extend(name for name in names if name not in ordered)
    return ordered


def response_value(row: pd.DataFrame, metric: str, stat: str) -> float:
    col = f"{metric}_{stat}"
    if row.empty or col not in row.columns:
        return float("nan")
    return float(row[col].iloc[0])


def plot_response(ax, response: pd.DataFrame, panel: str, title: str | None, show_xlabel: bool):
    x = np.arange(len(COUPLING_ORDER))
    for metric, label, color in MODULES:
        means: list[float] = []
        sems: list[float] = []
        for cbin in COUPLING_ORDER:
            row = response[response["coupling_bin"].astype(str) == cbin]
            means.append(response_value(row, metric, "mean"))
            sem = response_value(row, metric, "sem")
            sems.append(0.0 if not np.isfinite(sem) else sem)
        ax.errorbar(
            x,
            means,
            yerr=sems,
            color=color,
            marker="o",
            linewidth=1.9,
            markersize=4.2,
            capsize=2.5,
            label=label,
        )
    ax.set_xticks(x, COUPLING_LABELS)
    ax.set_ylabel("Relative feature change")
    ax.set_xlabel("PPG--ACC coupling bin" if show_xlabel else "")
    if title:
        ax.set_title(title, loc="center", fontweight="bold", pad=8)
    ax.text(-0.14, 1.06, panel, transform=ax.transAxes, fontweight="bold", fontsize=11, va="bottom")


def heatmap_matrix(heatmap: pd.DataFrame, order: list[str]) -> tuple[np.ndarray, list[str]]:
    rows: list[tuple[str, str]] = []
    for baseline_name in order:
        for validity in VALIDITY_ORDER:
            rows.append((baseline_name, validity))

    matrix = np.full((len(rows), len(COUPLING_ORDER)), np.nan, dtype=float)
    for i, (baseline_name, validity) in enumerate(rows):
        for j, coupling in enumerate(COUPLING_ORDER):
            selected = heatmap[
                (heatmap["baseline_name"].astype(str) == baseline_name)
                & (heatmap["validity_bin"].astype(str) == validity)
                & (heatmap["coupling_bin"].astype(str) == coupling)
            ]
            if not selected.empty:
                matrix[i, j] = float(selected["true_class_prob_gain_mean"].iloc[0])
    labels = [f"{baseline}\n{validity} validity" for baseline, validity in rows]
    return matrix, labels


def plot_gain_heatmap(
    ax,
    heatmap: pd.DataFrame,
    panel: str,
    title: str | None,
    vmin: float,
    vmax: float,
    baseline_names: list[str],
    show_xlabel: bool,
):
    matrix, row_labels = heatmap_matrix(heatmap, baseline_names)
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(COUPLING_ORDER)), COUPLING_LABELS)
    ax.set_yticks(np.arange(len(row_labels)), row_labels)
    ax.set_xlabel("PPG--ACC coupling" if show_xlabel else "")
    if title:
        ax.set_title(title, loc="center", fontweight="bold", pad=8)
    ax.text(-0.14, 1.06, panel, transform=ax.transAxes, fontweight="bold", fontsize=11, va="bottom")
    ax.grid(False)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:+.3f}", ha="center", va="center", fontsize=7.2, color=TEXT)
    return im


def signature_xlim(*signatures: pd.DataFrame) -> tuple[float, float]:
    lows: list[float] = []
    highs: list[float] = []
    allowed = {var for var, _ in SIGNATURE_VARIABLES}
    for signature in signatures:
        if signature.empty:
            continue
        subset = signature[signature["variable"].astype(str).isin(allowed)].copy()
        lows.extend(pd.to_numeric(subset["ci_low"], errors="coerce").dropna().tolist())
        highs.extend(pd.to_numeric(subset["ci_high"], errors="coerce").dropna().tolist())
    if not lows or not highs:
        return (-1.0, 1.0)
    low = float(np.nanmin(lows))
    high = float(np.nanmax(highs))
    span = max(high - low, 0.5)
    pad = span * 0.10
    return low - pad, high + pad


def plot_signature(
    ax,
    signature: pd.DataFrame,
    panel: str,
    title: str | None,
    xlim: tuple[float, float],
    baseline_names: list[str],
    show_xlabel: bool,
):
    y = np.arange(len(SIGNATURE_VARIABLES), dtype=float)
    offsets = np.linspace(-0.13 * (len(baseline_names) - 1), 0.13 * (len(baseline_names) - 1), len(baseline_names))
    for idx, baseline_name in enumerate(baseline_names):
        rows = signature[signature["baseline_name"].astype(str) == baseline_name].set_index("variable")
        xs: list[float] = []
        xerr_low: list[float] = []
        xerr_high: list[float] = []
        for variable, _ in SIGNATURE_VARIABLES:
            if variable not in rows.index:
                xs.append(float("nan"))
                xerr_low.append(0.0)
                xerr_high.append(0.0)
                continue
            row = rows.loc[variable]
            value = float(row["corrected_minus_harmed_z"])
            ci_low = float(row["ci_low"])
            ci_high = float(row["ci_high"])
            xs.append(value)
            xerr_low.append(max(0.0, value - ci_low) if np.isfinite(ci_low) else 0.0)
            xerr_high.append(max(0.0, ci_high - value) if np.isfinite(ci_high) else 0.0)
        color = BASELINE_COLORS.get(baseline_name, PURPLE if idx else BLUE)
        ax.errorbar(
            np.asarray(xs, dtype=float),
            y + offsets[idx],
            xerr=np.vstack([xerr_low, xerr_high]),
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=1.45,
            capsize=2.8,
            markersize=4.6,
            alpha=0.95,
            label=baseline_name,
        )
    ax.axvline(0, color="#4B5563", linewidth=1.0)
    ax.set_xlim(*xlim)
    ax.set_yticks(y, [label for _, label in SIGNATURE_VARIABLES])
    ax.invert_yaxis()
    ax.set_xlabel("Corrected minus harmed standardized mean difference" if show_xlabel else "")
    if title:
        ax.set_title(title, loc="center", fontweight="bold", pad=8)
    ax.text(-0.14, 1.06, panel, transform=ax.transAxes, fontweight="bold", fontsize=11, va="bottom")
    ax.grid(True, axis="x", color=GRID, alpha=0.55)
    ax.grid(False, axis="y")


def shared_heatmap_limits(*datasets: dict[str, pd.DataFrame]) -> tuple[float, float]:
    values: list[float] = []
    for data in datasets:
        values.extend(pd.to_numeric(data["heatmap"]["true_class_prob_gain_mean"], errors="coerce").dropna().tolist())
    if not values:
        return (-0.01, 0.01)
    vmax = max(abs(float(np.nanmin(values))), abs(float(np.nanmax(values))), 0.01)
    return -vmax, vmax


def main() -> None:
    args = parse_args()
    plt = setup_matplotlib(args.font_family)
    galaxy = load_dataset(args.galaxy_dir, "galaxy")
    wesad = load_dataset(args.wesad_dir, "wesad")

    galaxy_baselines = baseline_order(galaxy["heatmap"])
    wesad_baselines = baseline_order(wesad["heatmap"])

    heat_vmin, heat_vmax = shared_heatmap_limits(galaxy, wesad)
    forest_xlim = signature_xlim(galaxy["signature"], wesad["signature"])

    fig = plt.figure(figsize=(12.0, 7.0))
    gs = fig.add_gridspec(
        2,
        4,
        width_ratios=[1.0, 1.15, 0.055, 1.45],
        height_ratios=[1.0, 1.0],
        left=0.095,
        right=0.975,
        top=0.895,
        bottom=0.13,
        hspace=0.42,
        wspace=0.35,
    )
    axes = np.array(
        [
            [fig.add_subplot(gs[r, 0]), fig.add_subplot(gs[r, 1]), fig.add_subplot(gs[r, 3])]
            for r in range(2)
        ]
    )
    cax = fig.add_subplot(gs[:, 2])

    plot_response(
        axes[0, 0],
        galaxy["response"],
        "A",
        "Internal response vs PPG--ACC coupling",
        show_xlabel=False,
    )
    plot_gain_heatmap(
        axes[0, 1],
        galaxy["heatmap"],
        "B",
        "True-class probability gain",
        heat_vmin,
        heat_vmax,
        galaxy_baselines,
        show_xlabel=False,
    )
    plot_signature(
        axes[0, 2],
        galaxy["signature"],
        "C",
        "Corrected vs harmed signatures",
        forest_xlim,
        galaxy_baselines,
        show_xlabel=False,
    )

    plot_response(axes[1, 0], wesad["response"], "D", None, show_xlabel=True)
    im = plot_gain_heatmap(
        axes[1, 1],
        wesad["heatmap"],
        "E",
        None,
        heat_vmin,
        heat_vmax,
        wesad_baselines,
        show_xlabel=True,
    )
    plot_signature(axes[1, 2], wesad["signature"], "F", None, forest_xlim, wesad_baselines, show_xlabel=True)

    fig.text(0.028, 0.68, "Galaxy PPG", rotation=90, va="center", ha="center", fontsize=11, fontweight="bold")
    fig.text(0.028, 0.30, "WESAD", rotation=90, va="center", ha="center", fontsize=11, fontweight="bold")

    module_handles, module_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        module_handles,
        module_labels,
        loc="upper left",
        bbox_to_anchor=(0.105, 0.992),
        ncol=3,
        frameon=False,
        fontsize=8.2,
    )

    from matplotlib.lines import Line2D

    baseline_handles = [
        Line2D([0], [0], marker="o", linestyle="", color=BLUE, label="PPG/BVP-only"),
        Line2D([0], [0], marker="o", linestyle="", color=PURPLE, label="Simple concat"),
    ]
    fig.legend(
        baseline_handles,
        [handle.get_label() for handle in baseline_handles],
        loc="lower center",
        bbox_to_anchor=(0.73, 0.025),
        ncol=2,
        frameon=False,
        title="Input baseline",
        title_fontsize=8.5,
        fontsize=8.2,
    )

    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("True-class probability gain", fontsize=8.5)
    cbar.ax.tick_params(labelsize=7.5)

    output_prefix = args.output_prefix
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=args.dpi, bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_prefix.with_suffix('.png')}")
    print(f"Saved {output_prefix.with_suffix('.pdf')}")
    print(f"Saved {output_prefix.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
