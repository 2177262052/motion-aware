from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

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
    "collapse": "Collapse rate",
    "positive_rate_error": "Positive-rate error",
}

COLORS = {
    "positive": "#2E86AB",
    "negative": "#C73E1D",
    "zero": "#424242",
    "grid": "#D8DDE6",
}


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"Expected NAME=PATH, got: {value}")
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"Missing method name in: {value}")
    return name, Path(path)


def parse_comparison(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise ValueError(f"Expected CANDIDATE=REFERENCE, got: {value}")
    candidate, reference = value.split("=", 1)
    candidate = candidate.strip()
    reference = reference.strip()
    if not candidate or not reference:
        raise ValueError(f"Invalid comparison: {value}")
    return candidate, reference


def natural_subject_key(subject: object) -> tuple[str, int, str]:
    text = str(subject)
    match = re.search(r"(\d+)", text)
    prefix = text[: match.start()] if match else text
    number = int(match.group(1)) if match else 10_000
    return prefix, number, text


def subject_column(df: pd.DataFrame, path: Path) -> str:
    for column in ("subject", "subject_id", "heldout_subject"):
        if column in df.columns:
            return column
    raise ValueError(f"{path} does not contain a subject column.")


def read_method_frame(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    subj_col = subject_column(df, path)
    if "model" in df.columns and "deploy_watch" in set(df["model"].astype(str)):
        dropped = int((df["model"].astype(str) != "deploy_watch").sum())
        if dropped:
            print(f"info_filter_model path={path} keeping deploy_watch rows; dropped={dropped}")
        df = df[df["model"].astype(str) == "deploy_watch"].copy()

    out = pd.DataFrame({"subject": df[subj_col].astype(str)})
    for metric in METRIC_DIRECTIONS:
        candidates = [
            metric,
            f"deploy_watch_{metric}",
            f"watch_only_{metric}",
        ]
        source = next((column for column in candidates if column in df.columns), None)
        if source is None:
            raise ValueError(f"{path} does not contain any of these columns: {candidates}")
        out[metric] = pd.to_numeric(df[source], errors="coerce")

    duplicates = int(out["subject"].duplicated().sum())
    if duplicates:
        print(f"warning_duplicate_subject_rows path={path} duplicates={duplicates}; averaging by subject")
        out = out.groupby("subject", as_index=False).mean(numeric_only=True)
    return out


def build_delta_frame(
    frames: dict[str, pd.DataFrame],
    comparisons: list[tuple[str, str]],
    metrics: list[str],
    signed_better: bool,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for candidate_name, reference_name in comparisons:
        if candidate_name not in frames:
            raise ValueError(f"Unknown candidate method {candidate_name!r}. Available: {sorted(frames)}")
        if reference_name not in frames:
            raise ValueError(f"Unknown reference method {reference_name!r}. Available: {sorted(frames)}")
        candidate = frames[candidate_name]
        reference = frames[reference_name]
        merged = reference[["subject", *metrics]].merge(
            candidate[["subject", *metrics]],
            on="subject",
            suffixes=("_reference", "_candidate"),
        )
        for metric in metrics:
            delta = merged[f"{metric}_candidate"] - merged[f"{metric}_reference"]
            direction = METRIC_DIRECTIONS[metric]
            plotted_delta = -delta if signed_better and direction == "lower" else delta
            frame = pd.DataFrame(
                {
                    "candidate": candidate_name,
                    "reference": reference_name,
                    "comparison": f"{candidate_name} vs {reference_name}",
                    "metric": metric,
                    "direction": direction,
                    "subject": merged["subject"],
                    "reference_value": merged[f"{metric}_reference"],
                    "candidate_value": merged[f"{metric}_candidate"],
                    "delta": delta,
                    "plotted_delta": plotted_delta,
                }
            )
            rows.append(frame)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def bootstrap_ci(values: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    if len(values) == 1 or n_boot <= 0:
        mean = float(values.mean())
        return mean, mean
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def summarize_delta_frame(delta_frame: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (comparison, metric), group in delta_frame.groupby(["comparison", "metric"], sort=False):
        plotted = group["plotted_delta"].to_numpy(dtype=np.float64)
        raw = group["delta"].to_numpy(dtype=np.float64)
        wins = int(np.sum(plotted > 1e-12))
        losses = int(np.sum(plotted < -1e-12))
        ties = int(len(plotted) - wins - losses)
        ci_low, ci_high = bootstrap_ci(raw, n_boot=n_boot, seed=seed)
        plotted_ci_low, plotted_ci_high = bootstrap_ci(plotted, n_boot=n_boot, seed=seed)
        rows.append(
            {
                "comparison": comparison,
                "metric": metric,
                "n": int(len(group)),
                "reference_mean": float(group["reference_value"].mean()),
                "candidate_mean": float(group["candidate_value"].mean()),
                "delta_mean": float(np.mean(raw)),
                "delta_median": float(np.median(raw)),
                "delta_ci_low": ci_low,
                "delta_ci_high": ci_high,
                "plotted_delta_mean": float(np.mean(plotted)),
                "plotted_delta_ci_low": plotted_ci_low,
                "plotted_delta_ci_high": plotted_ci_high,
                "candidate_wins": wins,
                "candidate_losses": losses,
                "ties": ties,
            }
        )
    return pd.DataFrame(rows)


def panel_sort(data: pd.DataFrame, sort_by: str) -> pd.DataFrame:
    if sort_by == "delta":
        return data.sort_values("plotted_delta", ascending=True).reset_index(drop=True)
    if sort_by == "subject":
        tmp = data.copy()
        tmp["_subject_sort"] = tmp["subject"].map(natural_subject_key)
        return tmp.sort_values("_subject_sort").drop(columns="_subject_sort").reset_index(drop=True)
    raise ValueError(f"Unsupported sort mode: {sort_by}")


def format_float(value: float) -> str:
    if not np.isfinite(value):
        return "nan"
    return f"{value:+.3f}"


def make_plot(
    delta_frame: pd.DataFrame,
    output_prefix: Path,
    sort_by: str,
    title: str,
    signed_better: bool,
) -> None:
    import matplotlib.pyplot as plt

    panels = list(delta_frame.groupby(["comparison", "metric"], sort=False))
    n_panels = len(panels)
    ncols = min(2, n_panels)
    nrows = int(math.ceil(n_panels / ncols))
    fig_width = 7.2 * ncols
    fig_height = max(3.4 * nrows, 4.0)

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 450,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height), squeeze=False)
    axes_flat = axes.reshape(-1)
    for ax in axes_flat[n_panels:]:
        ax.axis("off")

    for ax, ((comparison, metric), data) in zip(axes_flat, panels):
        data = panel_sort(data, sort_by=sort_by)
        y = np.arange(len(data))
        values = data["plotted_delta"].to_numpy(dtype=np.float64)
        colors = np.where(values >= 0, COLORS["positive"], COLORS["negative"])
        ax.barh(y, values, color=colors, edgecolor="white", linewidth=0.4)
        ax.axvline(0.0, color=COLORS["zero"], linewidth=0.9)
        ax.set_yticks(y)
        ax.set_yticklabels(data["subject"], fontsize=7)
        ax.grid(axis="x", color=COLORS["grid"], linewidth=0.7, alpha=0.8)
        ax.set_axisbelow(True)
        direction = data["direction"].iloc[0]
        x_label = f"Delta {METRIC_LABELS.get(metric, metric)}"
        if direction == "lower":
            x_label += " (positive favors candidate)" if signed_better else " (negative favors candidate)"
        ax.set_xlabel(x_label)
        ax.set_title(f"{comparison}: {METRIC_LABELS.get(metric, metric)}")

        mean_value = float(values.mean()) if len(values) else 0.0
        wins = int(np.sum(values > 1e-12))
        losses = int(np.sum(values < -1e-12))
        ties = int(len(values) - wins - losses)
        ax.text(
            0.02,
            0.98,
            f"mean={format_float(mean_value)}; W/L/T={wins}/{losses}/{ties}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#C8CDD8", "alpha": 0.92},
        )

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=450)
    fig.savefig(output_prefix.with_suffix(".svg"))
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def write_markdown(path: Path, summary: pd.DataFrame) -> None:
    lines = [
        "# Per-subject Delta Summary",
        "",
        "| Comparison | Metric | n | Reference mean | Candidate mean | Delta mean | 95% CI | Wins/Losses/Ties |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.comparison} | {row.metric} | {int(row.n)} | "
            f"{row.reference_mean:.4f} | {row.candidate_mean:.4f} | {row.delta_mean:+.4f} | "
            f"[{row.delta_ci_low:+.4f}, {row.delta_ci_high:+.4f}] | "
            f"{int(row.candidate_wins)}/{int(row.candidate_losses)}/{int(row.ties)} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot named per-subject paired deltas from LOSO result CSVs.")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--method", action="append", required=True, help="NAME=CSV. Repeat for each method.")
    parser.add_argument("--comparison", action="append", required=True, help="CANDIDATE=REFERENCE. Repeat as needed.")
    parser.add_argument(
        "--metric",
        action="append",
        choices=sorted(METRIC_DIRECTIONS),
        default=None,
        help="Metric to plot. Defaults to balanced_acc and auroc.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", type=str, default=None)
    parser.add_argument("--sort-by", choices=["delta", "subject"], default="delta")
    parser.add_argument("--signed-better", action="store_true", help="Flip lower-is-better metrics so positive always favors candidate.")
    parser.add_argument("--bootstrap", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    frames = {name: read_method_frame(path) for name, path in (parse_named_path(item) for item in args.method)}
    comparisons = [parse_comparison(item) for item in args.comparison]
    metrics = args.metric or ["balanced_acc", "auroc"]

    delta_frame = build_delta_frame(frames, comparisons, metrics, signed_better=args.signed_better)
    summary = summarize_delta_frame(delta_frame, n_boot=args.bootstrap, seed=args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or f"{args.dataset}_per_subject_delta"
    delta_path = args.output_dir / f"{prefix}_deltas.csv"
    summary_path = args.output_dir / f"{prefix}_summary.csv"
    markdown_path = args.output_dir / f"{prefix}_summary.md"
    figure_prefix = args.output_dir / prefix

    delta_frame.to_csv(delta_path, index=False)
    summary.to_csv(summary_path, index=False)
    write_markdown(markdown_path, summary)
    make_plot(
        delta_frame,
        figure_prefix,
        sort_by=args.sort_by,
        title=f"{args.dataset}: per-subject paired deltas",
        signed_better=args.signed_better,
    )

    print(summary.to_string(index=False))
    print(f"Saved deltas to {delta_path}")
    print(f"Saved summary to {summary_path}")
    print(f"Saved markdown to {markdown_path}")
    print(
        "Saved figure to "
        f"{figure_prefix.with_suffix('.png')}, "
        f"{figure_prefix.with_suffix('.svg')}, and "
        f"{figure_prefix.with_suffix('.pdf')}"
    )


if __name__ == "__main__":
    main()
