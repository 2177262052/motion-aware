from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


COLORS = {
    "trust": "#2E86AB",
    "gate": "#F18F01",
    "third": "#59A14F",
    "ci": "#A7C7E7",
    "zero": "#4B5563",
    "grid": "#D8DDE6",
    "text": "#111827",
}


def finite_frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for column in columns:
        if column not in out.columns:
            raise ValueError(f"Missing required column {column!r}. Available columns include: {list(out.columns)[:30]}")
        out[column] = pd.to_numeric(out[column], errors="coerce")
    mask = np.ones(len(out), dtype=bool)
    for column in columns:
        mask &= np.isfinite(out[column].to_numpy(dtype=float))
    return out.loc[mask].copy()


def bootstrap_ci(values: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    if values.size == 1 or n_boot <= 0:
        value = float(values.mean())
        return value, value
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(n_boot, values.size))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def rank_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3:
        return float("nan")
    xr = pd.Series(x).rank(method="average").to_numpy(dtype=np.float64)
    yr = pd.Series(y).rank(method="average").to_numpy(dtype=np.float64)
    if np.std(xr) <= 1e-12 or np.std(yr) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(xr, yr)[0, 1])


def make_bins(values: pd.Series, n_bins: int, mode: str) -> pd.Series:
    if mode == "uniform":
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        return pd.cut(values.clip(0.0, 1.0), bins=bins, labels=False, include_lowest=True)
    if mode == "quantile":
        ranked = values.rank(method="first")
        return pd.qcut(ranked, q=min(n_bins, values.nunique()), labels=False, duplicates="drop")
    raise ValueError(f"Unsupported binning mode: {mode}")


def summarize_bins(
    df: pd.DataFrame,
    trust_col: str,
    teacher_correct_col: str,
    third_col: str,
    weight_col: str | None,
    min_weight: float,
    n_bins: int,
    binning: str,
    n_boot: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    columns = [trust_col, teacher_correct_col, third_col]
    if weight_col:
        columns.append(weight_col)
    work = finite_frame(df, columns)
    work[trust_col] = work[trust_col].clip(0.0, 1.0)
    min_weight = min(max(float(min_weight), 0.0), 1.0)
    if weight_col:
        work["gated_kd_weight"] = work[weight_col].clip(0.0, 1.0)
    else:
        work["gated_kd_weight"] = min_weight + (1.0 - min_weight) * work[trust_col]
    work["trust_bin"] = make_bins(work[trust_col], n_bins=n_bins, mode=binning)
    work = work.dropna(subset=["trust_bin"]).copy()
    work["trust_bin"] = work["trust_bin"].astype(int)

    correlations = {
        "rho_trust_teacher_correct": rank_corr(work[trust_col].to_numpy(), work[teacher_correct_col].to_numpy()),
        "rho_trust_weight": rank_corr(work[trust_col].to_numpy(), work["gated_kd_weight"].to_numpy()),
        "rho_trust_third": rank_corr(work[trust_col].to_numpy(), work[third_col].to_numpy()),
    }

    rows: list[dict[str, float | int]] = []
    for bin_id, group in work.groupby("trust_bin", sort=True):
        trust = group[trust_col].to_numpy(dtype=float)
        correct = group[teacher_correct_col].to_numpy(dtype=float)
        gate = group["gated_kd_weight"].to_numpy(dtype=float)
        third = group[third_col].to_numpy(dtype=float)
        c_low, c_high = bootstrap_ci(correct, n_boot=n_boot, seed=seed + int(bin_id) * 17 + 1)
        g_low, g_high = bootstrap_ci(gate, n_boot=n_boot, seed=seed + int(bin_id) * 17 + 2)
        third_low, third_high = bootstrap_ci(third, n_boot=n_boot, seed=seed + int(bin_id) * 17 + 3)
        rows.append(
            {
                "bin": int(bin_id),
                "n": int(len(group)),
                "trust_mean": float(np.mean(trust)),
                "trust_min": float(np.min(trust)),
                "trust_max": float(np.max(trust)),
                "teacher_correct_rate": float(np.mean(correct)),
                "teacher_correct_ci_low": c_low,
                "teacher_correct_ci_high": c_high,
                "gated_kd_weight_mean": float(np.mean(gate)),
                "gated_kd_weight_ci_low": g_low,
                "gated_kd_weight_ci_high": g_high,
                "third_metric_mean": float(np.mean(third)),
                "third_metric_ci_low": third_low,
                "third_metric_ci_high": third_high,
            }
        )
    return pd.DataFrame(rows), correlations


def setup_axis(ax, title: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=11, fontweight="bold", pad=9)
    ax.set_xlabel("Cross-confidence trust bin", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, axis="y", color=COLORS["grid"], alpha=0.72, linewidth=0.8)
    ax.grid(True, axis="x", color=COLORS["grid"], alpha=0.35, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#4B5563")
    ax.spines["bottom"].set_color("#4B5563")
    ax.tick_params(labelsize=8)


def plot_line_with_ci(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    color: str,
    label: str,
) -> None:
    ax.fill_between(x, low, high, color=color, alpha=0.18, linewidth=0)
    ax.plot(
        x,
        y,
        color=color,
        marker="o",
        markersize=5.5,
        linewidth=2.2,
        markeredgecolor="white",
        markeredgewidth=0.8,
        label=label,
    )


def rho_text(value: float) -> str:
    if not np.isfinite(value):
        return "rho=n/a"
    return f"rho={value:+.2f}"


def make_plot(
    summary: pd.DataFrame,
    correlations: dict[str, float],
    output_prefix: Path,
    title: str,
    subtitle: str,
    font_family: str,
    third_title: str,
    third_ylabel: str,
    third_axis: str,
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

    x = summary["trust_mean"].to_numpy(dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.35), constrained_layout=False)

    setup_axis(
        axes[0],
        f"A. Trust tracks teacher correctness ({rho_text(correlations.get('rho_trust_teacher_correct', float('nan')))})",
        "Teacher correct rate",
    )
    plot_line_with_ci(
        axes[0],
        x,
        summary["teacher_correct_rate"].to_numpy(dtype=float),
        summary["teacher_correct_ci_low"].to_numpy(dtype=float),
        summary["teacher_correct_ci_high"].to_numpy(dtype=float),
        COLORS["trust"],
        "teacher correct",
    )
    axes[0].set_ylim(0.0, 1.02)

    setup_axis(
        axes[1],
        f"B. Gated KD follows reliability ({rho_text(correlations.get('rho_trust_weight', float('nan')))})",
        "Gated KD weight",
    )
    plot_line_with_ci(
        axes[1],
        x,
        summary["gated_kd_weight_mean"].to_numpy(dtype=float),
        summary["gated_kd_weight_ci_low"].to_numpy(dtype=float),
        summary["gated_kd_weight_ci_high"].to_numpy(dtype=float),
        COLORS["gate"],
        "KD weight",
    )
    axes[1].set_ylim(0.0, 1.02)

    setup_axis(
        axes[2],
        f"{third_title} ({rho_text(correlations.get('rho_trust_third', float('nan')))})",
        third_ylabel,
    )
    if third_axis != "rate":
        axes[2].axhline(0.0, color=COLORS["zero"], linewidth=1.0, linestyle="--", alpha=0.75)
    plot_line_with_ci(
        axes[2],
        x,
        summary["third_metric_mean"].to_numpy(dtype=float),
        summary["third_metric_ci_low"].to_numpy(dtype=float),
        summary["third_metric_ci_high"].to_numpy(dtype=float),
        COLORS["third"],
        third_ylabel,
    )
    if third_axis == "rate":
        axes[2].set_ylim(0.0, 1.02)
    else:
        third_values = np.concatenate(
            [
                summary["third_metric_ci_low"].to_numpy(dtype=float),
                summary["third_metric_ci_high"].to_numpy(dtype=float),
                np.array([0.0]),
            ]
        )
        third_min = float(np.nanmin(third_values))
        third_max = float(np.nanmax(third_values))
        third_pad = max((third_max - third_min) * 0.12, 0.005)
        axes[2].set_ylim(third_min - third_pad, third_max + third_pad)

    for ax in axes:
        ax.set_xlim(max(0.0, float(np.nanmin(x)) - 0.04), min(1.0, float(np.nanmax(x)) + 0.04))
        for bin_row in summary.itertuples(index=False):
            ax.text(
                float(bin_row.trust_mean),
                0.025 if ax is not axes[2] else ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.035,
                f"n={int(bin_row.n)}",
                ha="center",
                va="bottom",
                fontsize=6.8,
                color="#6B7280",
            )

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.995)
    if subtitle:
        fig.text(0.5, 0.91, subtitle, ha="center", va="center", fontsize=9.5, color=COLORS["text"])
    fig.subplots_adjust(left=0.065, right=0.99, top=0.81, bottom=0.19, wspace=0.32)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=450, bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot a 3-panel gated-KD reliability mechanism curve.")
    parser.add_argument("--windows-csv", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--trust-col", type=str, default="trust_final_teacher")
    parser.add_argument("--teacher-correct-col", type=str, default="teacher_correct")
    parser.add_argument("--third-col", type=str, default="teacher_closer_than_base")
    parser.add_argument("--weight-col", type=str, default=None, help="Optional explicit KD weight column. If omitted, weight is computed from trust.")
    parser.add_argument("--third-title", type=str, default="C. Trusted teachers are more useful")
    parser.add_argument("--third-ylabel", type=str, default="Teacher closer than watch path")
    parser.add_argument("--third-axis", type=str, default="rate", choices=["rate", "signed"])
    parser.add_argument("--min-weight", type=float, default=0.0, help="Gated KD minimum weight. Use 0.0 for min0.0.")
    parser.add_argument("--bins", type=int, default=6)
    parser.add_argument("--binning", type=str, default="quantile", choices=["quantile", "uniform"])
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--font-family", type=str, default="Arial")
    parser.add_argument("--title", type=str, default="Galaxy PPG gated KD reliability mechanism")
    parser.add_argument(
        "--subtitle",
        type=str,
        default="Windows are grouped by cross-confidence trust; shaded bands show bootstrap 95% CIs.",
    )
    args = parser.parse_args()

    windows = pd.read_csv(args.windows_csv)
    summary, correlations = summarize_bins(
        windows,
        trust_col=args.trust_col,
        teacher_correct_col=args.teacher_correct_col,
        third_col=args.third_col,
        weight_col=args.weight_col,
        min_weight=args.min_weight,
        n_bins=args.bins,
        binning=args.binning,
        n_boot=args.bootstrap,
        seed=args.seed,
    )
    if summary.empty:
        raise ValueError("No finite rows remain after filtering.")

    summary_path = args.output_prefix.with_name(args.output_prefix.name + "_binned_summary.csv")
    corr_path = args.output_prefix.with_name(args.output_prefix.name + "_correlations.csv")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    pd.DataFrame([correlations]).to_csv(corr_path, index=False)
    make_plot(
        summary,
        correlations,
        args.output_prefix,
        title=args.title,
        subtitle=args.subtitle,
        font_family=args.font_family,
        third_title=args.third_title,
        third_ylabel=args.third_ylabel,
        third_axis=args.third_axis,
    )

    print(summary.to_string(index=False))
    print(f"Saved binned summary to {summary_path}")
    print(f"Saved correlations to {corr_path}")
    print(f"Saved figure to {args.output_prefix.with_suffix('.png')}")
    print(f"Saved figure to {args.output_prefix.with_suffix('.pdf')}")
    print(f"Saved figure to {args.output_prefix.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
