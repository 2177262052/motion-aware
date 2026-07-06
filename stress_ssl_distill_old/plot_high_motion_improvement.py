from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


POSITIVE = "#5AA05A"
NEGATIVE = "#D65F3D"
GRID = "#D8DDE6"
TEXT = "#111827"

METRIC_LABELS = {
    "balanced_acc": "Balanced accuracy",
    "auroc": "AUROC",
    "f1": "F1",
}


def subject_sort_key(subject: object) -> tuple[str, int, str]:
    text = str(subject)
    match = re.search(r"(\d+)", text)
    prefix = text[: match.start()] if match else text
    number = int(match.group(1)) if match else 10_000
    return prefix, number, text


def bootstrap_ci(values: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
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


def compute_improvement(
    subject_bin: pd.DataFrame,
    metric: str,
    motion_bin: str,
    candidate: str,
    reference: str,
) -> pd.DataFrame:
    required = {"subject_id", "method", "motion_bin", metric}
    missing = required - set(subject_bin.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}. Available columns: {list(subject_bin.columns)}")

    frame = subject_bin[subject_bin["motion_bin"].astype(str) == motion_bin].copy()
    frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
    pivot = frame.pivot_table(index="subject_id", columns="method", values=metric, aggfunc="first")
    if candidate not in pivot.columns or reference not in pivot.columns:
        raise ValueError(
            f"Need methods {candidate!r} and {reference!r}; available methods in {motion_bin}: {list(pivot.columns)}"
        )
    out = pivot[[reference, candidate]].reset_index()
    out = out.rename(columns={reference: "reference_value", candidate: "candidate_value"})
    out["delta"] = out["candidate_value"] - out["reference_value"]
    out = out.dropna(subset=["reference_value", "candidate_value", "delta"]).copy()
    return out


def make_plot(
    deltas: pd.DataFrame,
    output_prefix: Path,
    dataset_label: str,
    metric: str,
    motion_bin_label: str,
    candidate_label: str,
    reference_label: str,
    font_family: str,
    n_boot: int,
    seed: int,
) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": font_family,
            "font.size": 9,
            "axes.titlesize": 12,
            "axes.labelsize": 9,
            "figure.dpi": 160,
            "savefig.dpi": 450,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    frame = deltas.copy()
    frame["_subject_sort"] = frame["subject_id"].map(subject_sort_key)
    frame = frame.sort_values("delta", ascending=True).drop(columns="_subject_sort")
    values = frame["delta"].to_numpy(dtype=np.float64)
    subjects = frame["subject_id"].astype(str).tolist()
    colors = np.where(values >= 0.0, POSITIVE, NEGATIVE)
    y = np.arange(len(frame))

    fig_height = max(4.4, 0.22 * len(frame) + 1.7)
    fig, ax = plt.subplots(figsize=(6.4, fig_height))
    ax.barh(y, values, color=colors, height=0.68, edgecolor="white", linewidth=0.45)
    ax.axvline(0.0, color="#2F3640", linestyle="--", linewidth=1.0, alpha=0.78)
    ax.set_yticks(y, subjects, fontsize=7.5)
    metric_label = METRIC_LABELS.get(metric, metric)
    ax.set_xlabel(f"High-motion Δ {metric_label} ({candidate_label} - {reference_label})", fontsize=9)
    ax.set_ylabel("Held-out subject", fontsize=9)
    ax.set_title(f"{dataset_label}: high-motion {metric_label} improvement", fontsize=12, fontweight="bold", pad=10)
    ax.grid(True, axis="x", color=GRID, alpha=0.75, linewidth=0.8)
    ax.grid(True, axis="y", color=GRID, alpha=0.35, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#4B5563")
    ax.spines["bottom"].set_color("#4B5563")
    ax.tick_params(axis="x", labelsize=8)

    mean = float(np.mean(values)) if len(values) else float("nan")
    ci_low, ci_high = bootstrap_ci(values, n_boot=n_boot, seed=seed)
    wins = int(np.sum(values > 1e-12))
    losses = int(np.sum(values < -1e-12))
    ties = int(len(values) - wins - losses)
    ax.text(
        0.975,
        0.055,
        f"{motion_bin_label}\nMean Δ = {mean:+.3f}\n95% CI [{ci_low:+.3f}, {ci_high:+.3f}]\nW/L/T = {wins}/{losses}/{ties}",
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

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=450, bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot per-subject high-motion improvement for motion-aware models.")
    parser.add_argument("--subject-bin-csv", type=Path, required=True)
    parser.add_argument("--dataset-label", type=str, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--metric", type=str, default="balanced_acc")
    parser.add_argument("--motion-bin", type=str, default="high_motion")
    parser.add_argument("--motion-bin-label", type=str, default="High-motion windows")
    parser.add_argument("--candidate", type=str, default="motion_aware")
    parser.add_argument("--reference", type=str, default="watch_only")
    parser.add_argument("--candidate-label", type=str, default="Motion-aware")
    parser.add_argument("--reference-label", type=str, default="Watch-only")
    parser.add_argument("--font-family", type=str, default="Liberation Sans")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    subject_bin = pd.read_csv(args.subject_bin_csv)
    deltas = compute_improvement(
        subject_bin,
        metric=args.metric,
        motion_bin=args.motion_bin,
        candidate=args.candidate,
        reference=args.reference,
    )
    summary_path = args.output_prefix.with_name(args.output_prefix.name + "_deltas.csv")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    deltas.to_csv(summary_path, index=False)
    make_plot(
        deltas,
        output_prefix=args.output_prefix,
        dataset_label=args.dataset_label,
        metric=args.metric,
        motion_bin_label=args.motion_bin_label,
        candidate_label=args.candidate_label,
        reference_label=args.reference_label,
        font_family=args.font_family,
        n_boot=args.bootstrap,
        seed=args.seed,
    )
    print(deltas.to_string(index=False))
    print(f"Saved deltas to {summary_path}")
    print(f"Saved figure to {args.output_prefix.with_suffix('.png')}")
    print(f"Saved figure to {args.output_prefix.with_suffix('.pdf')}")
    print(f"Saved figure to {args.output_prefix.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
