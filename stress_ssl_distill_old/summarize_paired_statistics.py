from __future__ import annotations

import argparse
import math
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

DATASET_FILES = {
    "galaxy": {
        "Watch-only": ["galaxy_watch_only_results.csv"],
        "Watch-only + Motion": ["galaxy_watch_motion_results.csv", "galaxy_watch_moiton_results.csv"],
        "PureKD + Motion": ["galaxy_loso_purekd_results.csv"],
        "Ours": ["galaxy_loso_ours_results.csv"],
    },
    "wesad": {
        # These two local files were exported with swapped names. Keep the
        # semantic mapping aligned with the recorded experiment means:
        # Watch-only ~= 0.7912 BA, Watch-only + Motion ~= 0.8002 BA.
        "Watch-only": ["wesad_watch_motion_results.csv"],
        "Watch-only + Motion": ["wesad_watch_only_results.csv"],
        "PureKD + Motion": ["wesad_loso_purekd_results.csv"],
        "Ours": ["wesad_loso_ours_results.csv"],
    },
}

COMPARISONS = [
    ("Watch-only + Motion", "Watch-only"),
    ("PureKD + Motion", "Watch-only + Motion"),
    ("Ours", "PureKD + Motion"),
    ("Ours", "Watch-only + Motion"),
    ("Ours", "Watch-only"),
]


def find_first_existing(root: Path, names: list[str]) -> Path:
    for name in names:
        path = root / name
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find any of these files under {root}: {names}")


def read_method_frame(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "subject" not in df.columns:
        raise ValueError(f"{path} does not contain a subject column.")
    out = pd.DataFrame({"subject": df["subject"].astype(str)})
    for metric in METRIC_DIRECTIONS:
        if metric in df.columns:
            source = metric
        elif f"deploy_watch_{metric}" in df.columns:
            source = f"deploy_watch_{metric}"
        elif f"watch_only_{metric}" in df.columns:
            source = f"watch_only_{metric}"
        else:
            raise ValueError(f"{path} does not contain {metric} or a recognized prefixed column.")
        out[metric] = pd.to_numeric(df[source], errors="coerce")
    return out


def exact_sign_p_two_sided(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return float("nan")
    k = min(wins, losses)
    cdf = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return float(min(1.0, 2.0 * cdf))


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int) -> tuple[float, float]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan"), float("nan")
    if len(x) == 1 or n_boot <= 0:
        value = float(np.mean(x))
        return value, value
    indices = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def wilcoxon_p(values: np.ndarray, direction: str) -> tuple[float, float]:
    try:
        from scipy.stats import wilcoxon
    except Exception:
        return float("nan"), float("nan")

    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    x = x[np.abs(x) > 1e-12]
    if len(x) == 0:
        return float("nan"), float("nan")
    try:
        two_sided = float(wilcoxon(x, alternative="two-sided", zero_method="wilcox").pvalue)
        better_alt = "greater" if direction == "higher" else "less"
        candidate_better = float(wilcoxon(x, alternative=better_alt, zero_method="wilcox").pvalue)
    except ValueError:
        return float("nan"), float("nan")
    return two_sided, candidate_better


def summarize_pair(
    dataset: str,
    candidate_name: str,
    reference_name: str,
    metric: str,
    candidate: pd.DataFrame,
    reference: pd.DataFrame,
    rng: np.random.Generator,
    n_boot: int,
) -> tuple[dict[str, object], pd.DataFrame]:
    merged = reference[["subject", metric]].merge(
        candidate[["subject", metric]],
        on="subject",
        suffixes=("_reference", "_candidate"),
    )
    merged = merged.dropna(subset=[f"{metric}_reference", f"{metric}_candidate"]).copy()
    reference_values = merged[f"{metric}_reference"].to_numpy(dtype=np.float64)
    candidate_values = merged[f"{metric}_candidate"].to_numpy(dtype=np.float64)
    delta = candidate_values - reference_values
    direction = METRIC_DIRECTIONS[metric]
    signed_better = delta if direction == "higher" else -delta
    eps = 1e-12
    wins = int(np.sum(signed_better > eps))
    losses = int(np.sum(signed_better < -eps))
    ties = int(len(signed_better) - wins - losses)
    raw_ci_low, raw_ci_high = bootstrap_ci(delta, rng=rng, n_boot=n_boot)
    better_ci_low, better_ci_high = bootstrap_ci(signed_better, rng=rng, n_boot=n_boot)
    wilcoxon_two_sided, wilcoxon_candidate_better = wilcoxon_p(delta, direction)

    summary = {
        "dataset": dataset,
        "candidate": candidate_name,
        "reference": reference_name,
        "comparison": f"{candidate_name} vs {reference_name}",
        "metric": metric,
        "direction": direction,
        "n": int(len(merged)),
        "reference_mean": float(np.mean(reference_values)) if len(merged) else float("nan"),
        "candidate_mean": float(np.mean(candidate_values)) if len(merged) else float("nan"),
        "delta_mean": float(np.mean(delta)) if len(merged) else float("nan"),
        "delta_median": float(np.median(delta)) if len(merged) else float("nan"),
        "delta_bootstrap_ci_low": raw_ci_low,
        "delta_bootstrap_ci_high": raw_ci_high,
        "signed_candidate_better_mean": float(np.mean(signed_better)) if len(merged) else float("nan"),
        "signed_candidate_better_ci_low": better_ci_low,
        "signed_candidate_better_ci_high": better_ci_high,
        "candidate_wins": wins,
        "candidate_losses": losses,
        "ties": ties,
        "sign_test_p_two_sided": exact_sign_p_two_sided(wins, losses),
        "wilcoxon_p_two_sided": wilcoxon_two_sided,
        "wilcoxon_p_candidate_better": wilcoxon_candidate_better,
    }

    deltas = merged[["subject"]].copy()
    deltas["dataset"] = dataset
    deltas["candidate"] = candidate_name
    deltas["reference"] = reference_name
    deltas["metric"] = metric
    deltas["direction"] = direction
    deltas["reference_value"] = reference_values
    deltas["candidate_value"] = candidate_values
    deltas["delta"] = delta
    deltas["signed_candidate_better"] = signed_better
    return summary, deltas


def format_float(value: object, digits: int = 4) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "nan"
    if not np.isfinite(x):
        return "nan"
    return f"{x:.{digits}f}"


def write_markdown(path: Path, summary: pd.DataFrame) -> None:
    lines: list[str] = []
    lines.append("# Paired Statistical Tests")
    lines.append("")
    lines.append(
        "Unit of analysis is the held-out LOSO subject. Delta is candidate minus reference; "
        "for lower-is-better metrics, `signed better` flips the sign so positive still favors the candidate."
    )
    lines.append("")
    main = summary[summary["metric"].isin(["balanced_acc", "auroc"])].copy()
    for dataset, dataset_group in main.groupby("dataset", sort=False):
        lines.append(f"## {dataset}")
        lines.append("")
        lines.append("| Comparison | Metric | n | Ref mean | Cand mean | Delta | 95% CI | Wins/Losses/Ties | Wilcoxon p | Sign p |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in dataset_group.itertuples(index=False):
            lines.append(
                f"| {row.comparison} | {row.metric} | {int(row.n)} | "
                f"{format_float(row.reference_mean)} | {format_float(row.candidate_mean)} | "
                f"{format_float(row.delta_mean)} | "
                f"[{format_float(row.delta_bootstrap_ci_low)}, {format_float(row.delta_bootstrap_ci_high)}] | "
                f"{int(row.candidate_wins)}/{int(row.candidate_losses)}/{int(row.ties)} | "
                f"{format_float(row.wilcoxon_p_two_sided)} | {format_float(row.sign_test_p_two_sided)} |"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize paired LOSO subject statistics with bootstrap CIs.")
    parser.add_argument("--result-root", type=Path, default=Path("result"))
    parser.add_argument("--output-dir", type=Path, default=Path("result/statistics"))
    parser.add_argument("--bootstrap", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    method_frames: dict[str, dict[str, pd.DataFrame]] = {}
    for dataset, files in DATASET_FILES.items():
        dataset_root = args.result_root / dataset
        method_frames[dataset] = {}
        for method, names in files.items():
            path = find_first_existing(dataset_root, names)
            method_frames[dataset][method] = read_method_frame(path)

    summary_rows: list[dict[str, object]] = []
    delta_frames: list[pd.DataFrame] = []
    for dataset, frames in method_frames.items():
        for candidate_name, reference_name in COMPARISONS:
            if candidate_name not in frames or reference_name not in frames:
                continue
            for metric in METRIC_DIRECTIONS:
                summary, deltas = summarize_pair(
                    dataset=dataset,
                    candidate_name=candidate_name,
                    reference_name=reference_name,
                    metric=metric,
                    candidate=frames[candidate_name],
                    reference=frames[reference_name],
                    rng=rng,
                    n_boot=args.bootstrap,
                )
                summary_rows.append(summary)
                delta_frames.append(deltas)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(summary_rows)
    deltas_df = pd.concat(delta_frames, ignore_index=True) if delta_frames else pd.DataFrame()
    summary_path = args.output_dir / "paired_statistical_tests.csv"
    deltas_path = args.output_dir / "paired_subject_deltas.csv"
    markdown_path = args.output_dir / "paired_statistical_tests.md"
    summary_df.to_csv(summary_path, index=False)
    deltas_df.to_csv(deltas_path, index=False)
    write_markdown(markdown_path, summary_df)

    print(f"Saved paired statistical tests to {summary_path}")
    print(f"Saved paired subject deltas to {deltas_path}")
    print(f"Saved markdown table to {markdown_path}")
    print()
    print(summary_df[summary_df["metric"].isin(["balanced_acc", "auroc"])].to_string(index=False))


if __name__ == "__main__":
    main()
