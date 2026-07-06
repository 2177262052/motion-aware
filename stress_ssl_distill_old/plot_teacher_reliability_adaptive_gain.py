from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


DATASET_COLORS = {
    "Galaxy PPG": "#2563eb",
    "WESAD": "#16a34a",
}


def read_results(path: Path, method: str, dataset: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = ["subject", "deploy_watch_auroc", "deploy_watch_balanced_acc", "teacher_auroc"]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    out = frame.copy()
    out["dataset"] = dataset
    out["method"] = method
    return out


def build_paired_frame(result_dir: Path) -> pd.DataFrame:
    specs = [
        (
            "Galaxy PPG",
            result_dir / "galaxy" / "galaxy_loso_ours_results.csv",
            result_dir / "galaxy" / "galaxy_loso_purekd_results.csv",
        ),
        (
            "WESAD",
            result_dir / "wesad" / "wesad_loso_ours_results.csv",
            result_dir / "wesad" / "wesad_loso_purekd_results.csv",
        ),
    ]
    rows: list[pd.DataFrame] = []
    for dataset, ours_path, purekd_path in specs:
        ours = read_results(ours_path, "ours", dataset)
        purekd = read_results(purekd_path, "purekd", dataset)
        paired = ours.merge(purekd, on="subject", suffixes=("_ours", "_purekd"))
        paired["dataset"] = dataset
        paired["teacher_auroc_mean"] = paired[["teacher_auroc_ours", "teacher_auroc_purekd"]].mean(axis=1)
        if "teacher_balanced_acc_ours" in paired.columns and "teacher_balanced_acc_purekd" in paired.columns:
            paired["teacher_balanced_acc_mean"] = paired[
                ["teacher_balanced_acc_ours", "teacher_balanced_acc_purekd"]
            ].mean(axis=1)
        else:
            paired["teacher_balanced_acc_mean"] = np.nan
        if "teacher_positive_rate_error_ours" in paired.columns and "teacher_positive_rate_error_purekd" in paired.columns:
            paired["teacher_positive_rate_error_mean"] = paired[
                ["teacher_positive_rate_error_ours", "teacher_positive_rate_error_purekd"]
            ].mean(axis=1)
        else:
            paired["teacher_positive_rate_error_mean"] = np.nan
        paired["adaptive_minus_purekd_auroc"] = paired["deploy_watch_auroc_ours"] - paired["deploy_watch_auroc_purekd"]
        paired["adaptive_minus_purekd_balanced_acc"] = (
            paired["deploy_watch_balanced_acc_ours"] - paired["deploy_watch_balanced_acc_purekd"]
        )
        paired["adaptive_minus_purekd_f1"] = paired.get("deploy_watch_f1_ours", np.nan) - paired.get(
            "deploy_watch_f1_purekd",
            np.nan,
        )
        rows.append(paired)
    return pd.concat(rows, axis=0, ignore_index=True)


def safe_corr(x: pd.Series, y: pd.Series, method: str) -> float:
    frame = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(frame) < 3:
        return float("nan")
    if frame["x"].std() <= 1e-12 or frame["y"].std() <= 1e-12:
        return float("nan")
    return float(frame["x"].corr(frame["y"], method=method))


def summarize(paired: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset, group in paired.groupby("dataset", sort=False):
        for x_col, label in [
            ("teacher_auroc_mean", "teacher AUROC"),
            ("teacher_balanced_acc_mean", "teacher BA"),
            ("teacher_positive_rate_error_mean", "teacher positive-rate error"),
        ]:
            rows.append(
                {
                    "dataset": dataset,
                    "x": label,
                    "y": "Ours - PureKD deploy AUROC",
                    "n": int(group[[x_col, "adaptive_minus_purekd_auroc"]].dropna().shape[0]),
                    "x_mean": float(pd.to_numeric(group[x_col], errors="coerce").mean()),
                    "delta_mean": float(group["adaptive_minus_purekd_auroc"].mean()),
                    "delta_median": float(group["adaptive_minus_purekd_auroc"].median()),
                    "wins": int((group["adaptive_minus_purekd_auroc"] > 1e-12).sum()),
                    "losses": int((group["adaptive_minus_purekd_auroc"] < -1e-12).sum()),
                    "ties": int((group["adaptive_minus_purekd_auroc"].abs() <= 1e-12).sum()),
                    "pearson_r": safe_corr(group[x_col], group["adaptive_minus_purekd_auroc"], method="pearson"),
                    "spearman_r": safe_corr(group[x_col], group["adaptive_minus_purekd_auroc"], method="spearman"),
                }
            )
    rows.append(
        {
            "dataset": "Combined",
            "x": "teacher AUROC",
            "y": "Ours - PureKD deploy AUROC",
            "n": int(paired[["teacher_auroc_mean", "adaptive_minus_purekd_auroc"]].dropna().shape[0]),
            "x_mean": float(paired["teacher_auroc_mean"].mean()),
            "delta_mean": float(paired["adaptive_minus_purekd_auroc"].mean()),
            "delta_median": float(paired["adaptive_minus_purekd_auroc"].median()),
            "wins": int((paired["adaptive_minus_purekd_auroc"] > 1e-12).sum()),
            "losses": int((paired["adaptive_minus_purekd_auroc"] < -1e-12).sum()),
            "ties": int((paired["adaptive_minus_purekd_auroc"].abs() <= 1e-12).sum()),
            "pearson_r": safe_corr(paired["teacher_auroc_mean"], paired["adaptive_minus_purekd_auroc"], method="pearson"),
            "spearman_r": safe_corr(paired["teacher_auroc_mean"], paired["adaptive_minus_purekd_auroc"], method="spearman"),
        }
    )
    return pd.DataFrame(rows)


def fmt(value: object, digits: int = 3) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "nan"
    if not math.isfinite(x):
        return "nan"
    return f"{x:.{digits}f}"


def write_summary_markdown(summary: pd.DataFrame, output_path: Path) -> None:
    lines = ["# Teacher Reliability vs Adaptive Gain", ""]
    lines.append("| Dataset | Reliability axis | n | x mean | delta mean | wins/losses/ties | Pearson r | Spearman r |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.dataset} | {row.x} | {int(row.n)} | {fmt(row.x_mean)} | {fmt(row.delta_mean)} | "
            f"{int(row.wins)}/{int(row.losses)}/{int(row.ties)} | {fmt(row.pearson_r)} | {fmt(row.spearman_r)} |"
        )
    lines.append("")
    lines.append(
        "Interpretation note: this plot is a mechanism/heterogeneity analysis. "
        "Use it to show how adaptive correction behaves under varying teacher reliability, not as the sole main-performance claim."
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def try_plot_with_matplotlib(paired: pd.DataFrame, output_path: Path) -> bool:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:
        print(f"matplotlib_unavailable={exc!r}; writing SVG fallback")
        return False

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), dpi=180)
    panels = [
        ("teacher_auroc_mean", "Teacher AUROC", "Higher teacher ranking reliability"),
        ("teacher_positive_rate_error_mean", "Teacher positive-rate error", "Lower calibration/threshold reliability"),
    ]
    for ax, (x_col, x_label, subtitle) in zip(axes, panels):
        for dataset, group in paired.groupby("dataset", sort=False):
            ax.scatter(
                group[x_col],
                group["adaptive_minus_purekd_auroc"],
                s=48,
                alpha=0.85,
                color=DATASET_COLORS.get(dataset, "#64748b"),
                label=dataset,
                edgecolors="white",
                linewidths=0.8,
            )
        ax.axhline(0.0, color="#475569", linewidth=1.0, linestyle="--")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Ours - PureKD deploy AUROC")
        ax.set_title(subtitle, fontsize=10)
        ax.grid(True, color="#e2e8f0", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].legend(frameon=False, loc="best")
    fig.suptitle("Teacher Reliability and Adaptive Correction Gain", y=1.02, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return True


def scale(values: pd.Series, left: float, right: float, invert: bool = False) -> list[float]:
    finite = pd.to_numeric(values, errors="coerce")
    lo = float(finite.min())
    hi = float(finite.max())
    if not math.isfinite(lo) or not math.isfinite(hi) or abs(hi - lo) <= 1e-12:
        return [(left + right) * 0.5 for _ in values]
    out = []
    for value in finite:
        if not math.isfinite(float(value)):
            out.append((left + right) * 0.5)
            continue
        t = (float(value) - lo) / (hi - lo)
        if invert:
            t = 1.0 - t
        out.append(left + t * (right - left))
    return out


def write_svg_fallback(paired: pd.DataFrame, output_path: Path) -> None:
    width, height = 1200, 450
    margin = 70
    panel_gap = 80
    panel_w = (width - 2 * margin - panel_gap) / 2
    panel_h = 270
    top = 95
    y_min = min(-0.05, float(paired["adaptive_minus_purekd_auroc"].min()) - 0.03)
    y_max = max(0.05, float(paired["adaptive_minus_purekd_auroc"].max()) + 0.03)

    def y_to_px(y: float) -> float:
        return top + (y_max - y) / (y_max - y_min) * panel_h

    def panel(x0: float, x_col: str, title: str, invert_x: bool = False) -> list[str]:
        lines = [
            f'<rect x="{x0}" y="{top}" width="{panel_w}" height="{panel_h}" fill="white" stroke="#cbd5e1"/>',
            f'<text x="{x0 + panel_w / 2}" y="{top - 24}" text-anchor="middle" font-size="18" font-weight="600">{title}</text>',
        ]
        zero_y = y_to_px(0.0)
        lines.append(f'<line x1="{x0}" y1="{zero_y}" x2="{x0 + panel_w}" y2="{zero_y}" stroke="#64748b" stroke-dasharray="6,5"/>')
        xs = scale(paired[x_col], x0 + 35, x0 + panel_w - 35, invert=invert_x)
        ys = [y_to_px(float(v)) for v in paired["adaptive_minus_purekd_auroc"]]
        for (_, row), x, y in zip(paired.iterrows(), xs, ys):
            color = DATASET_COLORS.get(str(row["dataset"]), "#64748b")
            lines.append(f'<circle cx="{x}" cy="{y}" r="6" fill="{color}" fill-opacity="0.85" stroke="white" stroke-width="1"/>')
        lines.append(f'<text x="{x0 + panel_w / 2}" y="{top + panel_h + 44}" text-anchor="middle" font-size="14">{title}</text>')
        return lines

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="600" y="42" text-anchor="middle" font-family="Arial" font-size="26" font-weight="700">Teacher Reliability and Adaptive Correction Gain</text>',
        *panel(margin, "teacher_auroc_mean", "Teacher AUROC"),
        *panel(margin + panel_w + panel_gap, "teacher_positive_rate_error_mean", "Teacher positive-rate error", invert_x=False),
        f'<text x="25" y="{top + panel_h / 2}" text-anchor="middle" transform="rotate(-90 25 {top + panel_h / 2})" font-family="Arial" font-size="15">Ours - PureKD deploy AUROC</text>',
        f'<circle cx="{width - 230}" cy="398" r="6" fill="{DATASET_COLORS["Galaxy PPG"]}"/><text x="{width - 214}" y="403" font-family="Arial" font-size="14">Galaxy PPG</text>',
        f'<circle cx="{width - 115}" cy="398" r="6" fill="{DATASET_COLORS["WESAD"]}"/><text x="{width - 99}" y="403" font-family="Arial" font-size="14">WESAD</text>',
        "</svg>",
    ]
    output_path.write_text("\n".join(svg), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot teacher reliability against adaptive correction gain.")
    parser.add_argument("--result-dir", type=Path, default=Path("result"))
    parser.add_argument("--output-dir", type=Path, default=Path("result") / "figures")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paired = build_paired_frame(args.result_dir)
    paired_path = args.output_dir / "teacher_reliability_adaptive_gain_points.csv"
    summary_path = args.output_dir / "teacher_reliability_adaptive_gain_summary.csv"
    markdown_path = args.output_dir / "teacher_reliability_adaptive_gain_summary.md"
    figure_path = args.output_dir / "teacher_reliability_adaptive_gain.svg"

    summary = summarize(paired)
    paired.to_csv(paired_path, index=False)
    summary.to_csv(summary_path, index=False)
    write_summary_markdown(summary, markdown_path)
    if not try_plot_with_matplotlib(paired, figure_path):
        write_svg_fallback(paired, figure_path)

    print(summary.to_string(index=False))
    print(f"Saved paired points to {paired_path}")
    print(f"Saved summary to {summary_path}")
    print(f"Saved markdown summary to {markdown_path}")
    print(f"Saved figure to {figure_path}")


if __name__ == "__main__":
    main()
