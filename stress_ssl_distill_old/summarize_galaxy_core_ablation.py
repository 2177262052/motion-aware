from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


METRIC_DIRECTIONS = {
    "acc": "higher",
    "balanced_acc": "higher",
    "auroc": "higher",
    "f1": "higher",
    "positive_rate": "none",
    "positive_rate_error": "lower",
    "collapse": "lower",
}

METRICS = list(METRIC_DIRECTIONS)


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.name, path
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"Missing method name in {value!r}")
    return name, Path(path)


def resolve_result_csv(path: Path) -> Path:
    if path.is_file():
        return path
    candidates = [
        path / "galaxy_loso_formal_results.csv",
        path / "galaxy_watch_loso_results.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find a Galaxy LOSO result CSV in {path}. "
        "Expected galaxy_loso_formal_results.csv or galaxy_watch_loso_results.csv."
    )


def numeric_column(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series([np.nan] * len(df), index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def normalize_result_frame(method_name: str, csv_path: Path) -> pd.DataFrame:
    raw = pd.read_csv(csv_path)
    if "subject" not in raw.columns:
        raise ValueError(f"{csv_path} does not contain a subject column.")

    if "deploy_watch_balanced_acc" in raw.columns:
        prefix = "deploy_watch_"
    elif "balanced_acc" in raw.columns:
        prefix = ""
    elif "watch_only_balanced_acc" in raw.columns:
        prefix = "watch_only_"
    else:
        raise ValueError(f"Could not infer metric columns in {csv_path}.")

    rows = pd.DataFrame(
        {
            "method": method_name,
            "subject": raw["subject"].astype(str),
            "source_csv": str(csv_path),
        }
    )
    for metric in METRICS:
        rows[metric] = numeric_column(raw, f"{prefix}{metric}" if prefix else metric)
    if prefix:
        rows["threshold"] = numeric_column(raw, f"{prefix}threshold")
    else:
        rows["threshold"] = numeric_column(raw, "threshold")
    return rows


def mean_std(values: pd.Series) -> tuple[float, float]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return float("nan"), float("nan")
    return float(clean.mean()), float(clean.std(ddof=1)) if len(clean) > 1 else 0.0


def iqr(values: pd.Series) -> tuple[float, float, float]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return float("nan"), float("nan"), float("nan")
    q1 = float(clean.quantile(0.25))
    q3 = float(clean.quantile(0.75))
    return q1, q3, q3 - q1


def summarize_methods(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method, group in frame.groupby("method", sort=False):
        row: dict[str, object] = {
            "method": method,
            "n_subjects": int(group["subject"].nunique()),
        }
        for metric in METRICS:
            values = pd.to_numeric(group[metric], errors="coerce")
            mean, std = mean_std(values)
            q1, q3, spread = iqr(values)
            direction = METRIC_DIRECTIONS[metric]
            if direction == "lower":
                worst = float(values.max()) if values.notna().any() else float("nan")
                best = float(values.min()) if values.notna().any() else float("nan")
            else:
                worst = float(values.min()) if values.notna().any() else float("nan")
                best = float(values.max()) if values.notna().any() else float("nan")
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_median"] = float(values.median()) if values.notna().any() else float("nan")
            row[f"{metric}_q1"] = q1
            row[f"{metric}_q3"] = q3
            row[f"{metric}_iqr"] = spread
            row[f"{metric}_best_fold"] = best
            row[f"{metric}_worst_fold"] = worst
        rows.append(row)
    return pd.DataFrame(rows)


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


def bootstrap_ci(
    diff: np.ndarray,
    iterations: int,
    seed: int,
    alpha: float = 0.05,
) -> tuple[float, float]:
    clean = diff[np.isfinite(diff)]
    if clean.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.choice(clean, size=(iterations, clean.size), replace=True).mean(axis=1)
    lo = float(np.quantile(draws, alpha / 2.0))
    hi = float(np.quantile(draws, 1.0 - alpha / 2.0))
    return lo, hi


def paired_metric_frames(frame: pd.DataFrame, reference: str, candidate: str, metric: str) -> pd.DataFrame:
    ref = frame[frame["method"] == reference][["subject", metric]].rename(columns={metric: "reference"})
    cand = frame[frame["method"] == candidate][["subject", metric]].rename(columns={metric: "candidate"})
    paired = ref.merge(cand, on="subject", how="inner")
    paired["reference"] = pd.to_numeric(paired["reference"], errors="coerce")
    paired["candidate"] = pd.to_numeric(paired["candidate"], errors="coerce")
    return paired.dropna(subset=["reference", "candidate"])


def signed_better_diff(paired: pd.DataFrame, metric: str) -> pd.Series:
    direction = METRIC_DIRECTIONS[metric]
    if direction == "lower":
        return paired["candidate"] - paired["reference"]
    return paired["reference"] - paired["candidate"]


def compare_to_reference(
    frame: pd.DataFrame,
    reference: str,
    metrics: Iterable[str],
    bootstrap_iterations: int,
    seed: int,
) -> pd.DataFrame:
    methods = [method for method in frame["method"].drop_duplicates().tolist() if method != reference]
    rows: list[dict[str, object]] = []
    for candidate in methods:
        for metric in metrics:
            direction = METRIC_DIRECTIONS[metric]
            if direction == "none":
                continue
            paired = paired_metric_frames(frame, reference, candidate, metric)
            diff = signed_better_diff(paired, metric).to_numpy(dtype=float)
            wins = int(np.sum(diff > 1e-12))
            losses = int(np.sum(diff < -1e-12))
            ties = int(len(diff) - wins - losses)
            ci_low, ci_high = bootstrap_ci(diff, bootstrap_iterations, seed)
            rows.append(
                {
                    "reference_method": reference,
                    "candidate_method": candidate,
                    "metric": metric,
                    "direction": direction,
                    "n_paired": int(len(diff)),
                    "reference_mean": float(paired["reference"].mean()) if len(paired) else float("nan"),
                    "candidate_mean": float(paired["candidate"].mean()) if len(paired) else float("nan"),
                    "signed_mean_diff_ref_better": float(np.mean(diff)) if len(diff) else float("nan"),
                    "bootstrap_ci_low": ci_low,
                    "bootstrap_ci_high": ci_high,
                    "reference_wins": wins,
                    "reference_losses": losses,
                    "ties": ties,
                    "wilcoxon_p_two_sided": try_wilcoxon(diff, alternative="two-sided"),
                    "wilcoxon_p_reference_greater": try_wilcoxon(diff, alternative="greater"),
                }
            )
    return pd.DataFrame(rows)


def format_float(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(number):
        return ""
    return f"{number:.4f}"


def write_markdown_tables(summary: pd.DataFrame, pairwise: pd.DataFrame, output_path: Path) -> None:
    main_cols = [
        "method",
        "balanced_acc_mean",
        "balanced_acc_std",
        "auroc_mean",
        "auroc_std",
        "f1_mean",
        "f1_std",
        "collapse_mean",
        "positive_rate_error_mean",
    ]
    lines = ["# Galaxy Core Ablation Summary", ""]
    lines.append("## Main Metrics")
    lines.append("")
    header = ["Method", "BA", "AUROC", "F1", "Collapse", "PRE"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for _, row in summary.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["method"]),
                    f"{format_float(row['balanced_acc_mean'])} +/- {format_float(row['balanced_acc_std'])}",
                    f"{format_float(row['auroc_mean'])} +/- {format_float(row['auroc_std'])}",
                    f"{format_float(row['f1_mean'])} +/- {format_float(row['f1_std'])}",
                    format_float(row["collapse_mean"]),
                    format_float(row["positive_rate_error_mean"]),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## Pairwise Tests Against Reference")
    lines.append("")
    lines.append(
        "| Reference | Candidate | Metric | Mean Diff | 95% CI | Wins/Losses/Ties | Wilcoxon p |"
    )
    lines.append("| --- | --- | --- | ---: | --- | --- | ---: |")
    for _, row in pairwise.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["reference_method"]),
                    str(row["candidate_method"]),
                    str(row["metric"]),
                    format_float(row["signed_mean_diff_ref_better"]),
                    f"[{format_float(row['bootstrap_ci_low'])}, {format_float(row['bootstrap_ci_high'])}]",
                    f"{int(row['reference_wins'])}/{int(row['reference_losses'])}/{int(row['ties'])}",
                    format_float(row["wilcoxon_p_two_sided"]),
                ]
            )
            + " |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize Galaxy core ablation per-subject results and paired significance tests."
    )
    parser.add_argument(
        "--method",
        action="append",
        required=True,
        help="Named result directory or CSV as NAME=PATH. Repeat in table order.",
    )
    parser.add_argument("--reference-method", type=str, required=True, help="Method name treated as ours/reference.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    frames = []
    seen: set[str] = set()
    for item in args.method:
        name, path = parse_named_path(item)
        if name in seen:
            raise ValueError(f"Duplicate method name: {name}")
        seen.add(name)
        csv_path = resolve_result_csv(path)
        frame = normalize_result_frame(name, csv_path)
        frames.append(frame)
        print(f"loaded method={name} n={len(frame)} csv={csv_path}")
    if args.reference_method not in seen:
        raise ValueError(f"Reference method {args.reference_method!r} was not provided as --method.")

    all_metrics = pd.concat(frames, axis=0, ignore_index=True)
    summary = summarize_methods(all_metrics)
    pairwise = compare_to_reference(
        all_metrics,
        reference=args.reference_method,
        metrics=METRICS,
        bootstrap_iterations=args.bootstrap_iterations,
        seed=args.seed,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_metrics_path = args.output_dir / "core_ablation_subject_metrics.csv"
    summary_path = args.output_dir / "core_ablation_summary.csv"
    pairwise_path = args.output_dir / "core_ablation_pairwise_vs_reference.csv"
    markdown_path = args.output_dir / "core_ablation_report.md"
    all_metrics.to_csv(all_metrics_path, index=False)
    summary.to_csv(summary_path, index=False)
    pairwise.to_csv(pairwise_path, index=False)
    write_markdown_tables(summary, pairwise, markdown_path)

    print()
    print("Main summary:")
    print(
        summary[
            [
                "method",
                "balanced_acc_mean",
                "balanced_acc_std",
                "auroc_mean",
                "auroc_std",
                "f1_mean",
                "f1_std",
                "collapse_mean",
                "positive_rate_error_mean",
            ]
        ].to_string(index=False)
    )
    print()
    print("Pairwise tests against reference:")
    print(pairwise.to_string(index=False))
    print()
    print(f"Saved subject metrics to {all_metrics_path}")
    print(f"Saved method summary to {summary_path}")
    print(f"Saved pairwise tests to {pairwise_path}")
    print(f"Saved markdown report to {markdown_path}")


if __name__ == "__main__":
    main()
