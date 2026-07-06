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


def read_method_frame(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "subject" not in df.columns:
        raise ValueError(f"{path} does not contain a subject column.")
    if "model" in df.columns:
        model_values = set(df["model"].astype(str))
        if "deploy_watch" in model_values:
            teacher_rows = int((df["model"].astype(str) != "deploy_watch").sum())
            if teacher_rows:
                print(
                    f"info_filter_model path={path} keeping model=deploy_watch "
                    f"and dropping non-deploy rows={teacher_rows}"
                )
            df = df[df["model"].astype(str) == "deploy_watch"].copy()
    out = pd.DataFrame({"subject": df["subject"].astype(str)})
    for metric in METRIC_DIRECTIONS:
        candidates = [
            metric,
            f"deploy_watch_{metric}",
            f"watch_only_{metric}",
        ]
        source = next((column for column in candidates if column in df.columns), None)
        if source is None:
            raise ValueError(f"{path} does not contain one of {candidates}.")
        out[metric] = pd.to_numeric(df[source], errors="coerce")
    duplicate_count = int(out["subject"].duplicated().sum())
    if duplicate_count:
        print(
            f"warning_duplicate_subject_rows path={path} duplicates={duplicate_count}; "
            "averaging rows per subject before paired tests"
        )
        out = out.groupby("subject", as_index=False).mean(numeric_only=True)
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
    lines = [
        "# Paired Statistical Tests",
        "",
        "Unit of analysis is the held-out LOSO subject. Delta is candidate minus reference.",
        "For lower-is-better metrics, signed-better flips the sign so positive still favors the candidate.",
        "",
        "| Comparison | Metric | n | Ref mean | Cand mean | Delta | 95% CI | Wins/Losses/Ties | Wilcoxon p | Sign p |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.comparison} | {row.metric} | {int(row.n)} | "
            f"{format_float(row.reference_mean)} | {format_float(row.candidate_mean)} | "
            f"{format_float(row.delta_mean)} | "
            f"[{format_float(row.delta_bootstrap_ci_low)}, {format_float(row.delta_bootstrap_ci_high)}] | "
            f"{int(row.candidate_wins)}/{int(row.candidate_losses)}/{int(row.ties)} | "
            f"{format_float(row.wilcoxon_p_two_sided)} | {format_float(row.sign_test_p_two_sided)} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Named paired LOSO statistics with bootstrap CIs.")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--method", action="append", required=True, help="NAME=CSV. Repeat for each method.")
    parser.add_argument("--comparison", action="append", required=True, help="CANDIDATE=REFERENCE. Repeat as needed.")
    parser.add_argument(
        "--metric",
        action="append",
        default=None,
        choices=sorted(METRIC_DIRECTIONS),
        help="Metric to summarize. Defaults to all metrics.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    frames = {name: read_method_frame(path) for name, path in (parse_named_path(item) for item in args.method)}
    metrics = args.metric or list(METRIC_DIRECTIONS)

    summary_rows: list[dict[str, object]] = []
    delta_frames: list[pd.DataFrame] = []
    for comparison in args.comparison:
        candidate_name, reference_name = parse_comparison(comparison)
        if candidate_name not in frames:
            raise ValueError(f"Unknown candidate method {candidate_name!r}. Available: {sorted(frames)}")
        if reference_name not in frames:
            raise ValueError(f"Unknown reference method {reference_name!r}. Available: {sorted(frames)}")
        for metric in metrics:
            summary, deltas = summarize_pair(
                dataset=args.dataset,
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
    summary_path = args.output_dir / f"{args.dataset}_paired_statistical_tests.csv"
    deltas_path = args.output_dir / f"{args.dataset}_paired_subject_deltas.csv"
    markdown_path = args.output_dir / f"{args.dataset}_paired_statistical_tests.md"
    summary_df.to_csv(summary_path, index=False)
    deltas_df.to_csv(deltas_path, index=False)
    write_markdown(markdown_path, summary_df)
    print(f"Saved paired statistical tests to {summary_path}")
    print(f"Saved paired subject deltas to {deltas_path}")
    print(f"Saved markdown table to {markdown_path}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
