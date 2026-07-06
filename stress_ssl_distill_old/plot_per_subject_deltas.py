from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

try:
    from scipy.stats import wilcoxon
except Exception:  # pragma: no cover - optional dependency guard
    wilcoxon = None


METRICS = ("balanced_acc", "auroc", "f1", "collapse", "positive_rate_error")


def _read_simple(path: Path, prefix: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    columns = {"subject": "subject"}
    for metric in METRICS:
        columns[metric] = f"{prefix}_{metric}"
    return frame[list(columns)].rename(columns=columns)


def _read_deploy(path: Path, prefix: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    columns = {"subject": "subject"}
    for metric in METRICS:
        columns[f"deploy_watch_{metric}"] = f"{prefix}_{metric}"
    return frame[list(columns)].rename(columns=columns)


def load_dataset_frames(dataset_dir: Path, dataset: str) -> pd.DataFrame:
    if dataset == "galaxy":
        files = {
            "watch": dataset_dir / "galaxy_watch_only_results.csv",
            "motion": dataset_dir / "galaxy_watch_moiton_results.csv",
            "purekd": dataset_dir / "galaxy_loso_purekd_results.csv",
            "ours": dataset_dir / "galaxy_loso_ours_results.csv",
        }
    elif dataset == "wesad":
        # The current result bundle uses swapped filenames for these two WESAD
        # watch-only exports: the file named "watch_only" contains the motion
        # run, while the file named "watch_motion" contains the no-motion run.
        files = {
            "watch": dataset_dir / "wesad_watch_motion_results.csv",
            "motion": dataset_dir / "wesad_watch_only_results.csv",
            "purekd": dataset_dir / "wesad_loso_purekd_results.csv",
            "ours": dataset_dir / "wesad_loso_ours_results.csv",
        }
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    frames = [
        _read_simple(files["watch"], "watch"),
        _read_simple(files["motion"], "motion"),
        _read_deploy(files["purekd"], "purekd"),
        _read_deploy(files["ours"], "ours"),
    ]
    out = frames[0]
    for frame in frames[1:]:
        out = out.merge(frame, on="subject", how="inner", validate="one_to_one")
    out.insert(0, "dataset", dataset)
    return out


def build_delta_frame(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    comparisons = {
        "Motion - Watch-only": ("motion", "watch"),
        "Ours - PureKD": ("ours", "purekd"),
        "Ours - Motion": ("ours", "motion"),
    }
    for _, row in frame.iterrows():
        for comparison, (candidate, reference) in comparisons.items():
            for metric in METRICS:
                rows.append(
                    {
                        "dataset": row["dataset"],
                        "subject": row["subject"],
                        "comparison": comparison,
                        "metric": metric,
                        "reference": reference,
                        "candidate": candidate,
                        "reference_value": float(row[f"{reference}_{metric}"]),
                        "candidate_value": float(row[f"{candidate}_{metric}"]),
                        "delta": float(row[f"{candidate}_{metric}"] - row[f"{reference}_{metric}"]),
                    }
                )
    return pd.DataFrame(rows)


def summarize_delta_frame(delta_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    lower_is_better = {"collapse", "positive_rate_error"}
    for (dataset, comparison, metric), group in delta_frame.groupby(["dataset", "comparison", "metric"], sort=False):
        deltas = group["delta"].to_numpy(dtype=np.float64)
        signed = -deltas if metric in lower_is_better else deltas
        wins = int(np.sum(signed > 1e-12))
        losses = int(np.sum(signed < -1e-12))
        ties = int(len(signed) - wins - losses)
        p_two_sided = np.nan
        p_better = np.nan
        if wilcoxon is not None and np.any(np.abs(signed) > 1e-12):
            try:
                p_two_sided = float(wilcoxon(signed, alternative="two-sided", zero_method="wilcox").pvalue)
                p_better = float(wilcoxon(signed, alternative="greater", zero_method="wilcox").pvalue)
            except ValueError:
                pass
        rows.append(
            {
                "dataset": dataset,
                "comparison": comparison,
                "metric": metric,
                "n": int(len(deltas)),
                "mean_delta": float(np.mean(deltas)),
                "median_delta": float(np.median(deltas)),
                "signed_better_mean": float(np.mean(signed)),
                "wins": wins,
                "losses": losses,
                "ties": ties,
                "wilcoxon_p_better": p_better,
                "wilcoxon_p_two_sided": p_two_sided,
            }
        )
    return pd.DataFrame(rows)


def _subject_sort_key(value: object) -> tuple[str, int, str]:
    text = str(value)
    digits = "".join(ch for ch in text if ch.isdigit())
    return ("".join(ch for ch in text if not ch.isdigit()), int(digits) if digits else 10_000, text)


def _auroc_panel_data(delta_frame: pd.DataFrame) -> list[tuple[str, str, str, pd.DataFrame]]:
    panel_specs = [
        ("galaxy", "Motion - Watch-only", "Galaxy: Motion - Watch-only"),
        ("galaxy", "Ours - PureKD", "Galaxy: Ours - PureKD"),
        ("wesad", "Motion - Watch-only", "WESAD: Motion - Watch-only"),
        ("wesad", "Ours - PureKD", "WESAD: Ours - PureKD"),
    ]
    out: list[tuple[str, str, str, pd.DataFrame]] = []
    for dataset, comparison, title in panel_specs:
        data = delta_frame[
            (delta_frame["dataset"] == dataset)
            & (delta_frame["comparison"] == comparison)
            & (delta_frame["metric"] == "auroc")
        ].copy()
        data = data.sort_values("delta", ascending=True).reset_index(drop=True)
        out.append((dataset, comparison, title, data))
    return out


def plot_auroc_panels_matplotlib(delta_frame: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    panels = _auroc_panel_data(delta_frame)
    colors = {"galaxy": "#2563eb", "wesad": "#059669"}
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 6.4), sharey=False)
    axes = axes.reshape(-1)

    for ax, (dataset, _comparison, title, data) in zip(axes, panels):
        x = np.arange(len(data))
        bar_colors = np.where(data["delta"].to_numpy() >= 0, colors[dataset], "#dc6b6b")
        ax.bar(x, data["delta"], color=bar_colors, width=0.78)
        ax.axhline(0.0, color="#333333", linewidth=0.8)
        ax.set_title(title, fontsize=11, pad=8)
        ax.set_ylabel("Delta AUROC", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(data["subject"], rotation=65, ha="right", fontsize=7)
        mean_delta = float(data["delta"].mean())
        wins = int((data["delta"] > 1e-12).sum())
        losses = int((data["delta"] < -1e-12).sum())
        ties = int(len(data) - wins - losses)
        ax.text(
            0.02,
            0.96,
            f"mean={mean_delta:+.3f}; wins/losses/ties={wins}/{losses}/{ties}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8.5,
            bbox={"facecolor": "white", "edgecolor": "#d0d0d0", "boxstyle": "round,pad=0.25", "alpha": 0.9},
        )
        ax.grid(axis="y", color="#dddddd", linewidth=0.6, alpha=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Per-subject paired AUROC deltas", fontsize=14, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def plot_auroc_panels_svg(delta_frame: pd.DataFrame, output_path: Path) -> Path:
    panels = _auroc_panel_data(delta_frame)
    colors = {"galaxy": "#2563eb", "wesad": "#059669"}
    neg_color = "#dc6b6b"
    width, height = 1280, 760
    margin_x, margin_top = 70, 70
    panel_gap_x, panel_gap_y = 70, 80
    panel_w = (width - 2 * margin_x - panel_gap_x) / 2
    panel_h = (height - margin_top - 70 - panel_gap_y) / 2
    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="640" y="34" text-anchor="middle" font-family="Arial, sans-serif" font-size="24" font-weight="700">Per-subject paired AUROC deltas</text>',
        '<text x="640" y="58" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="#555">Positive bars indicate the candidate method improves over its paired reference for the same held-out subject.</text>',
    ]

    for idx, (dataset, _comparison, title, data) in enumerate(panels):
        col = idx % 2
        row = idx // 2
        x0 = margin_x + col * (panel_w + panel_gap_x)
        y0 = margin_top + row * (panel_h + panel_gap_y)
        deltas = data["delta"].to_numpy(dtype=float)
        max_abs = max(float(np.max(np.abs(deltas))) if len(deltas) else 0.0, 0.02)
        max_abs *= 1.15
        zero_y = y0 + panel_h / 2
        y_scale = (panel_h * 0.42) / max_abs
        n = max(len(data), 1)
        bar_gap = 2.0
        bar_w = max((panel_w - 2 * 18 - (n - 1) * bar_gap) / n, 2.0)
        plot_x0 = x0 + 18

        svg.append(f'<g font-family="Arial, sans-serif">')
        svg.append(f'<text x="{x0}" y="{y0 - 18}" font-size="16" font-weight="700">{title}</text>')
        svg.append(f'<rect x="{x0}" y="{y0}" width="{panel_w}" height="{panel_h}" fill="#fbfbfb" stroke="#d9d9d9" stroke-width="1"/>')
        svg.append(f'<line x1="{x0}" y1="{zero_y}" x2="{x0 + panel_w}" y2="{zero_y}" stroke="#333" stroke-width="1"/>')
        for tick in (-max_abs, -max_abs / 2, max_abs / 2, max_abs):
            ty = zero_y - tick * y_scale
            svg.append(f'<line x1="{x0}" y1="{ty}" x2="{x0 + panel_w}" y2="{ty}" stroke="#e6e6e6" stroke-width="0.8"/>')
            svg.append(f'<text x="{x0 - 8}" y="{ty + 4}" text-anchor="end" font-size="10" fill="#555">{tick:+.2f}</text>')

        for i, item in data.iterrows():
            delta = float(item["delta"])
            bx = plot_x0 + i * (bar_w + bar_gap)
            by = zero_y - max(delta, 0.0) * y_scale
            bh = abs(delta) * y_scale
            if delta < 0:
                by = zero_y
            fill = colors[dataset] if delta >= 0 else neg_color
            svg.append(f'<rect x="{bx:.2f}" y="{by:.2f}" width="{bar_w:.2f}" height="{bh:.2f}" fill="{fill}"/>')
            tx = bx + bar_w / 2
            svg.append(
                f'<text x="{tx:.2f}" y="{y0 + panel_h + 14}" text-anchor="end" font-size="9" fill="#333" '
                f'transform="rotate(-60 {tx:.2f} {y0 + panel_h + 14})">{item["subject"]}</text>'
            )

        mean_delta = float(data["delta"].mean()) if len(data) else 0.0
        wins = int((data["delta"] > 1e-12).sum())
        losses = int((data["delta"] < -1e-12).sum())
        ties = int(len(data) - wins - losses)
        note = f"mean={mean_delta:+.3f}; wins/losses/ties={wins}/{losses}/{ties}"
        svg.append(f'<rect x="{x0 + 8}" y="{y0 + 8}" width="250" height="24" rx="5" fill="white" stroke="#d0d0d0" opacity="0.94"/>')
        svg.append(f'<text x="{x0 + 16}" y="{y0 + 25}" font-size="12" fill="#333">{note}</text>')
        svg.append(f'<text x="{x0 - 48}" y="{y0 + panel_h / 2}" font-size="12" fill="#333" transform="rotate(-90 {x0 - 48} {y0 + panel_h / 2})">Delta AUROC</text>')
        svg.append("</g>")

    svg.append("</svg>")
    svg_path = output_path.with_suffix(".svg")
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.write_text("\n".join(svg) + "\n", encoding="utf-8")
    return svg_path


def plot_auroc_panels(delta_frame: pd.DataFrame, output_path: Path) -> Path:
    try:
        plot_auroc_panels_matplotlib(delta_frame, output_path)
        return output_path
    except Exception as exc:
        print(f"matplotlib unavailable ({exc}); writing SVG fallback.")
        return plot_auroc_panels_svg(delta_frame, output_path)


def write_markdown_summary(summary: pd.DataFrame, output_path: Path) -> None:
    lines: list[str] = ["# Per-subject Delta Summary", ""]
    for dataset in ("galaxy", "wesad"):
        lines.append(f"## {dataset.upper()}")
        lines.append("")
        subset = summary[(summary["dataset"] == dataset) & (summary["metric"] == "auroc")]
        lines.append("| Comparison | Mean delta | Median delta | Wins | Losses | Ties | Wilcoxon p better |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for row in subset.itertuples(index=False):
            lines.append(
                f"| {row.comparison} | {row.mean_delta:+.4f} | {row.median_delta:+.4f} | "
                f"{row.wins} | {row.losses} | {row.ties} | {row.wilcoxon_p_better:.4g} |"
            )
        lines.append("")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot per-subject paired deltas for final stress experiments.")
    parser.add_argument("--result-dir", type=Path, default=Path("result"))
    parser.add_argument("--output-dir", type=Path, default=Path("result") / "figures")
    args = parser.parse_args()

    galaxy = load_dataset_frames(args.result_dir / "galaxy", "galaxy")
    wesad = load_dataset_frames(args.result_dir / "wesad", "wesad")
    frame = pd.concat([galaxy, wesad], axis=0, ignore_index=True)
    frame["subject_sort"] = frame["subject"].map(_subject_sort_key)
    frame = frame.sort_values(["dataset", "subject_sort"]).drop(columns=["subject_sort"]).reset_index(drop=True)

    delta_frame = build_delta_frame(frame)
    summary = summarize_delta_frame(delta_frame)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged_path = args.output_dir / "per_subject_metrics_merged.csv"
    delta_path = args.output_dir / "per_subject_deltas.csv"
    summary_path = args.output_dir / "per_subject_delta_summary.csv"
    markdown_path = args.output_dir / "per_subject_delta_summary.md"
    figure_path = args.output_dir / "per_subject_auroc_deltas.png"

    frame.to_csv(merged_path, index=False)
    delta_frame.to_csv(delta_path, index=False)
    summary.to_csv(summary_path, index=False)
    write_markdown_summary(summary, markdown_path)
    saved_figure_path = plot_auroc_panels(delta_frame, figure_path)

    print(f"Saved merged metrics to {merged_path}")
    print(f"Saved per-subject deltas to {delta_path}")
    print(f"Saved delta summary to {summary_path}")
    print(f"Saved markdown summary to {markdown_path}")
    print(f"Saved AUROC delta figure to {saved_figure_path}")


if __name__ == "__main__":
    main()
