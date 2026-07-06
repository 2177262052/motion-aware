from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRIC_DIRECTIONS = {
    "balanced_acc": "higher",
    "auroc": "higher",
    "f1": "higher",
    "collapse": "lower",
    "positive_rate_error": "lower",
}

METRIC_LABELS = {
    "balanced_acc": "Balanced accuracy",
    "auroc": "AUROC",
    "f1": "F1",
    "collapse": "Collapse",
    "positive_rate_error": "Positive-rate error",
    "positive_rate": "Predicted positive rate",
}


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 7,
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


def try_wilcoxon(diff: np.ndarray, alternative: str) -> float:
    clean = diff[np.isfinite(diff)]
    if clean.size == 0:
        return float("nan")
    if np.allclose(clean, 0.0):
        return 1.0
    try:
        from scipy.stats import wilcoxon

        return float(wilcoxon(clean, alternative=alternative, zero_method="wilcox").pvalue)
    except Exception:
        return float("nan")


def paired_frame(
    subject_metrics: pd.DataFrame,
    baseline_method: str,
    kd_method: str,
) -> pd.DataFrame:
    required = {"method", "subject", *METRIC_DIRECTIONS.keys(), "positive_rate"}
    missing = required.difference(subject_metrics.columns)
    if missing:
        raise ValueError(f"Subject metrics CSV missing columns: {sorted(missing)}")

    base = subject_metrics[subject_metrics["method"] == baseline_method].copy()
    kd = subject_metrics[subject_metrics["method"] == kd_method].copy()
    if base.empty:
        raise ValueError(f"Baseline method not found: {baseline_method}")
    if kd.empty:
        raise ValueError(f"KD method not found: {kd_method}")

    cols = ["subject", *METRIC_DIRECTIONS.keys(), "positive_rate"]
    base = base[cols].rename(columns={col: f"{col}_baseline" for col in cols if col != "subject"})
    kd = kd[cols].rename(columns={col: f"{col}_kd" for col in cols if col != "subject"})
    paired = base.merge(kd, on="subject", how="inner").sort_values("subject").reset_index(drop=True)
    for col in paired.columns:
        if col != "subject":
            paired[col] = pd.to_numeric(paired[col], errors="coerce")

    for metric, direction in METRIC_DIRECTIONS.items():
        raw_delta = paired[f"{metric}_kd"] - paired[f"{metric}_baseline"]
        paired[f"delta_{metric}_kd_minus_baseline"] = raw_delta
        paired[f"{metric}_kd_harms"] = raw_delta < -1e-12 if direction == "higher" else raw_delta > 1e-12
        paired[f"{metric}_kd_helps"] = raw_delta > 1e-12 if direction == "higher" else raw_delta < -1e-12

    paired["kd_induced_collapse"] = (paired["collapse_baseline"] < 0.5) & (paired["collapse_kd"] > 0.5)
    paired["kd_resolved_collapse"] = (paired["collapse_baseline"] > 0.5) & (paired["collapse_kd"] < 0.5)
    paired["both_collapsed"] = (paired["collapse_baseline"] > 0.5) & (paired["collapse_kd"] > 0.5)
    paired["neither_collapsed"] = (paired["collapse_baseline"] < 0.5) & (paired["collapse_kd"] < 0.5)
    paired["delta_positive_rate_kd_minus_baseline"] = paired["positive_rate_kd"] - paired["positive_rate_baseline"]
    return paired


def summarize_pairwise(paired: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metric, direction in METRIC_DIRECTIONS.items():
        delta = paired[f"delta_{metric}_kd_minus_baseline"].to_numpy(dtype=float)
        if direction == "higher":
            signed_baseline_better = -delta
        else:
            signed_baseline_better = delta
        rows.append(
            {
                "comparison": "Pure KD - Watch-only",
                "metric": metric,
                "direction": direction,
                "n_subjects": int(np.isfinite(delta).sum()),
                "baseline_mean": float(paired[f"{metric}_baseline"].mean()),
                "kd_mean": float(paired[f"{metric}_kd"].mean()),
                "delta_kd_minus_baseline": float(np.nanmean(delta)),
                "kd_harms_subjects": int(paired[f"{metric}_kd_harms"].sum()),
                "kd_helps_subjects": int(paired[f"{metric}_kd_helps"].sum()),
                "ties": int((~paired[f"{metric}_kd_harms"] & ~paired[f"{metric}_kd_helps"]).sum()),
                "wilcoxon_p_two_sided": try_wilcoxon(signed_baseline_better, alternative="two-sided"),
                "wilcoxon_p_baseline_greater": try_wilcoxon(signed_baseline_better, alternative="greater"),
            }
        )
    rows.append(
        {
            "comparison": "Pure KD - Watch-only",
            "metric": "collapse_transition",
            "direction": "lower",
            "n_subjects": int(len(paired)),
            "baseline_mean": float(paired["collapse_baseline"].mean()),
            "kd_mean": float(paired["collapse_kd"].mean()),
            "delta_kd_minus_baseline": float((paired["collapse_kd"] - paired["collapse_baseline"]).mean()),
            "kd_harms_subjects": int(paired["kd_induced_collapse"].sum()),
            "kd_helps_subjects": int(paired["kd_resolved_collapse"].sum()),
            "ties": int((paired["both_collapsed"] | paired["neither_collapsed"]).sum()),
            "wilcoxon_p_two_sided": try_wilcoxon(
                (paired["collapse_kd"] - paired["collapse_baseline"]).to_numpy(dtype=float),
                alternative="two-sided",
            ),
            "wilcoxon_p_baseline_greater": try_wilcoxon(
                (paired["collapse_kd"] - paired["collapse_baseline"]).to_numpy(dtype=float),
                alternative="greater",
            ),
        }
    )
    return pd.DataFrame(rows)


def plot_kd_harm(paired: pd.DataFrame, output_dir: Path, formats: list[str]) -> None:
    ordered = paired.sort_values("delta_balanced_acc_kd_minus_baseline")["subject"].tolist()
    frame = paired.set_index("subject").reindex(ordered)
    x = np.arange(len(frame))

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.5))

    ba_delta = frame["delta_balanced_acc_kd_minus_baseline"].to_numpy(dtype=float)
    ba_colors = np.where(ba_delta >= 0, "#2f855a", "#c53030")
    axes[0].bar(x, ba_delta, color=ba_colors, edgecolor="#111827", linewidth=0.35)
    axes[0].axhline(0.0, color="#111827", linewidth=0.8)
    axes[0].set_title("(a) Pure KD change in BA")
    axes[0].set_ylabel("Pure KD - Watch-only")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(ordered, rotation=70, ha="right")
    axes[0].grid(axis="y", color="#e5e7eb", linewidth=0.8)

    pre_delta = frame["delta_positive_rate_error_kd_minus_baseline"].to_numpy(dtype=float)
    pre_colors = np.where(pre_delta <= 0, "#2f855a", "#c53030")
    axes[1].bar(x, pre_delta, color=pre_colors, edgecolor="#111827", linewidth=0.35)
    axes[1].axhline(0.0, color="#111827", linewidth=0.8)
    axes[1].set_title("(b) Positive-rate error change")
    axes[1].set_ylabel("Pure KD - Watch-only")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(ordered, rotation=70, ha="right")
    axes[1].grid(axis="y", color="#e5e7eb", linewidth=0.8)

    categories = [
        ("Neither", "neither_collapsed", "#a0aec0"),
        ("KD induced", "kd_induced_collapse", "#c53030"),
        ("KD resolved", "kd_resolved_collapse", "#2f855a"),
        ("Both", "both_collapsed", "#dd6b20"),
    ]
    counts = [int(frame[col].sum()) for _, col, _ in categories]
    colors = [color for _, _, color in categories]
    bars = axes[2].bar(np.arange(len(categories)), counts, color=colors, edgecolor="#111827", linewidth=0.5)
    for bar, count in zip(bars, counts):
        axes[2].text(bar.get_x() + bar.get_width() / 2, count + 0.15, str(count), ha="center", va="bottom")
    axes[2].set_title("(c) Collapse transitions")
    axes[2].set_ylabel("# subjects")
    axes[2].set_xticks(np.arange(len(categories)))
    axes[2].set_xticklabels([name for name, _, _ in categories], rotation=20, ha="right")
    axes[2].grid(axis="y", color="#e5e7eb", linewidth=0.8)

    fig.tight_layout()
    save_figure(fig, output_dir, "kd_harm_diagnostics", formats)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose subject-level harm caused by vanilla KD compared with a watch-only baseline."
    )
    parser.add_argument("--subject-metrics-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-method", type=str, default="Watch-only")
    parser.add_argument("--kd-method", type=str, default="Pure-KD")
    parser.add_argument("--formats", nargs="*", default=["png", "pdf"])
    args = parser.parse_args()

    set_plot_style()
    subject_metrics = pd.read_csv(args.subject_metrics_csv)
    paired = paired_frame(subject_metrics, args.baseline_method, args.kd_method)
    summary = summarize_pairwise(paired)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paired.to_csv(args.output_dir / "kd_harm_subject_deltas.csv", index=False)
    summary.to_csv(args.output_dir / "kd_harm_summary.csv", index=False)
    plot_kd_harm(paired, args.output_dir, args.formats)

    print(summary.to_string(index=False))
    print(f"Saved KD harm diagnostics to {args.output_dir}")


if __name__ == "__main__":
    main()
