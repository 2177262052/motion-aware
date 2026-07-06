from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


COLORS = {
    "blue": "#4E79A7",
    "orange": "#F28E2B",
    "red": "#E15759",
    "green": "#59A14F",
    "purple": "#7B61FF",
    "gray": "#6B7280",
    "light_gray": "#E5E7EB",
    "dark": "#111827",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a focused teacher-reliability mechanism figure from exported window-level CSV."
    )
    parser.add_argument("--windows-csv", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--title", type=str, default="Galaxy Teacher Reliability Mechanism")
    parser.add_argument("--max-scatter", type=int, default=3000)
    parser.add_argument("--dpi", type=int, default=450)
    parser.add_argument("--figure-width", type=float, default=12.0)
    parser.add_argument("--figure-height", type=float, default=7.0)
    return parser.parse_args()


def numeric(frame: pd.DataFrame, col: str) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan)


def ensure_labels(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()

    def bool_label(col: str, pos: str, neg: str) -> pd.Series:
        vals = numeric(out, col)

        def assign(value: object) -> str:
            try:
                x = float(value)
            except (TypeError, ValueError):
                return "missing"
            if not math.isfinite(x):
                return "missing"
            return pos if x >= 0.5 else neg

        return vals.map(assign)

    def normalize_label(value: object) -> str:
        text = str(value).strip()
        mapping = {
            "teacher_correct": "teacher correct",
            "teacher wrong": "teacher wrong",
            "teacher_wrong": "teacher wrong",
            "teacher correct": "teacher correct",
            "teacher_closer": "teacher closer",
            "teacher closer": "teacher closer",
            "teacher_not_closer": "teacher not closer",
            "teacher not closer": "teacher not closer",
            "teacher_student_agree": "teacher/student agree",
            "teacher/student/agree": "teacher/student agree",
            "teacher/student agree": "teacher/student agree",
            "teacher_student_disagree": "teacher/student disagree",
            "teacher/student/disagree": "teacher/student disagree",
            "teacher/student disagree": "teacher/student disagree",
        }
        return mapping.get(text, text.replace("_", " "))

    if "teacher_correct_label" not in out.columns:
        out["teacher_correct_label"] = bool_label("teacher_correct", "teacher correct", "teacher wrong")
    else:
        out["teacher_correct_label"] = out["teacher_correct_label"].map(normalize_label)

    if "teacher_help_label" not in out.columns:
        out["teacher_help_label"] = bool_label("teacher_closer_than_base", "teacher closer", "teacher not closer")
    else:
        out["teacher_help_label"] = out["teacher_help_label"].map(normalize_label)

    if "teacher_base_agreement_label" not in out.columns:
        out["teacher_base_agreement_label"] = bool_label(
            "teacher_base_disagree",
            "teacher/student disagree",
            "teacher/student agree",
        )
    else:
        out["teacher_base_agreement_label"] = out["teacher_base_agreement_label"].map(normalize_label)

    if "transfer_signal" not in out.columns:
        alpha = numeric(out, "deploy_alpha")
        gate = numeric(out, "deploy_gate_mean")
        out["transfer_signal"] = alpha.where(alpha.notna(), gate)

    out["final_gain_abs_error"] = numeric(out, "final_gain_abs_error")
    if out["final_gain_abs_error"].isna().all():
        out["final_gain_abs_error"] = numeric(out, "base_abs_error") - numeric(out, "final_abs_error")

    return out


def finite_mean(frame: pd.DataFrame, col: str) -> float:
    values = numeric(frame, col).dropna()
    if values.empty:
        return float("nan")
    return float(values.mean())


def fmt(value: object, digits: int = 3) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "nan"
    if not math.isfinite(x):
        return "nan"
    return f"{x:.{digits}f}"


def spearman_rho(frame: pd.DataFrame, x_col: str, y_col: str) -> float:
    valid = pd.DataFrame(
        {
            "x": numeric(frame, x_col),
            "y": numeric(frame, y_col),
        }
    ).dropna()
    if len(valid) < 3 or valid["x"].nunique() < 2 or valid["y"].nunique() < 2:
        return float("nan")
    return float(valid["x"].rank(method="average").corr(valid["y"].rank(method="average")))


def setup_style(dpi: int) -> None:
    import matplotlib.pyplot as plt  # type: ignore

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.color": COLORS["light_gray"],
            "legend.frameon": False,
            "savefig.dpi": dpi,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def annotate_bar_values(ax: Any, values: Sequence[float], x_positions: Sequence[float], y_offset: float = 0.02) -> None:
    for x, y in zip(x_positions, values):
        if not math.isfinite(float(y)):
            continue
        ax.text(x, y + y_offset, fmt(y, 2), ha="center", va="bottom", fontsize=8, color=COLORS["dark"])


def boxplot_by_group(
    ax: Any,
    frame: pd.DataFrame,
    group_col: str,
    value_col: str,
    order: Sequence[str],
    title: str,
    ylabel: str,
    colors: Sequence[str],
) -> None:
    groups: list[np.ndarray] = []
    labels: list[str] = []
    for group_name in order:
        values = numeric(frame[frame[group_col] == group_name], value_col).dropna().to_numpy()
        if len(values) == 0:
            continue
        groups.append(values)
        labels.append(group_name)
    if not groups:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return
    box = ax.boxplot(groups, labels=labels, patch_artist=True, showfliers=False, widths=0.55)
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)
    for median in box["medians"]:
        median.set_color(COLORS["dark"])
        median.set_linewidth(1.4)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", labelrotation=8)


def plot_focused_figure(frame: pd.DataFrame, output_prefix: Path, title: str, dpi: int, width: float, height: float, max_scatter: int) -> None:
    import matplotlib.pyplot as plt  # type: ignore

    setup_style(dpi)
    fig, axes = plt.subplots(2, 3, figsize=(width, height), dpi=180)

    # A. Teacher fallibility summary.
    ax = axes[0, 0]
    rates = [
        finite_mean(frame, "teacher_correct"),
        finite_mean(frame, "teacher_closer_than_base"),
        finite_mean(frame, "teacher_base_disagree"),
    ]
    labels = ["teacher\ncorrect", "teacher closer\nthan base", "teacher/student\ndisagree"]
    x_pos = np.arange(len(labels))
    ax.bar(
        x_pos,
        rates,
        color=[COLORS["green"], COLORS["blue"], COLORS["red"]],
        alpha=0.82,
        width=0.65,
    )
    annotate_bar_values(ax, rates, x_pos)
    ax.axhline(0.5, color=COLORS["gray"], linewidth=1.0, linestyle="--", alpha=0.75)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Rate")
    ax.set_title("A. Privileged teacher is fallible")

    # B. Cross-confidence trust separates correct/wrong teacher windows.
    rho_teacher = spearman_rho(frame, "trust_final_teacher", "teacher_correct")
    boxplot_by_group(
        axes[0, 1],
        frame,
        "teacher_correct_label",
        "trust_final_teacher",
        ["teacher wrong", "teacher correct"],
        f"B. Trust tracks teacher correctness (rho={fmt(rho_teacher, 2)})",
        "Cross-confidence trust",
        [COLORS["red"], COLORS["green"]],
    )

    # C. Transfer signal is suppressed under teacher/student disagreement.
    rho_disagree = spearman_rho(frame, "transfer_signal", "teacher_base_disagree")
    boxplot_by_group(
        axes[0, 2],
        frame,
        "teacher_base_agreement_label",
        "transfer_signal",
        ["teacher/student agree", "teacher/student disagree"],
        f"C. Transfer is reduced when signals conflict (rho={fmt(rho_disagree, 2)})",
        "Transfer signal",
        [COLORS["blue"], COLORS["orange"]],
    )

    # D. Final gain is higher when the teacher is actually closer than the base student.
    boxplot_by_group(
        axes[1, 0],
        frame,
        "teacher_help_label",
        "final_gain_abs_error",
        ["teacher not closer", "teacher closer"],
        "D. Gain concentrates when teacher helps",
        "Base abs. error - final abs. error",
        [COLORS["red"], COLORS["green"]],
    )
    axes[1, 0].axhline(0.0, color=COLORS["gray"], linewidth=1.0)

    # E. Per-subject gain.
    ax = axes[1, 1]
    subject_summary = (
        frame.groupby("fold_subject", as_index=False)
        .agg(final_gain_abs_error_mean=("final_gain_abs_error", "mean"))
        .sort_values("final_gain_abs_error_mean", ascending=True)
    )
    bar_colors = np.where(subject_summary["final_gain_abs_error_mean"] >= 0, COLORS["blue"], COLORS["red"])
    ax.barh(
        subject_summary["fold_subject"].astype(str),
        subject_summary["final_gain_abs_error_mean"],
        color=bar_colors,
        alpha=0.82,
    )
    ax.axvline(0.0, color=COLORS["gray"], linewidth=1.0)
    ax.set_title("E. Subject-level final gain")
    ax.set_xlabel("Base abs. error - final abs. error")

    # F. Trust-vs-gain scatter.
    ax = axes[1, 2]
    scatter = frame.dropna(subset=["trust_final_teacher", "final_gain_abs_error"])
    if len(scatter) > max_scatter:
        scatter = scatter.sample(max_scatter, random_state=42)
    rho_gain = spearman_rho(frame, "trust_final_teacher", "final_gain_abs_error")
    colors = np.where(numeric(scatter, "teacher_correct") >= 0.5, COLORS["green"], COLORS["red"])
    ax.axhline(0.0, color=COLORS["gray"], linewidth=1.0)
    ax.scatter(
        scatter["trust_final_teacher"],
        scatter["final_gain_abs_error"],
        s=13,
        alpha=0.42,
        c=colors,
        linewidths=0,
    )
    ax.set_title(f"F. Gain rises with trusted teacher signal (rho={fmt(rho_gain, 2)})")
    ax.set_xlabel("Cross-confidence trust")
    ax.set_ylabel("Final gain")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_prefix.with_suffix(".png")
    svg_path = output_prefix.with_suffix(".svg")
    pdf_path = output_prefix.with_suffix(".pdf")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"saved_png={png_path}")
    print(f"saved_svg={svg_path}")
    print(f"saved_pdf={pdf_path}")


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.windows_csv)
    frame = ensure_labels(frame)
    if frame.empty:
        raise ValueError(f"No rows found in {args.windows_csv}")
    plot_focused_figure(
        frame,
        output_prefix=args.output_prefix,
        title=args.title,
        dpi=args.dpi,
        width=args.figure_width,
        height=args.figure_height,
        max_scatter=args.max_scatter,
    )


if __name__ == "__main__":
    main()
