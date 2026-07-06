from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score


CANONICAL_ALIASES = {
    "dataset": ("dataset", "dataset_kind"),
    "group": ("group", "analysis_group", "rq"),
    "method": ("method", "model", "objective"),
    "fold": ("fold", "loso_fold", "heldout_fold"),
    "subject_id": ("subject_id", "subject", "heldout_subject", "test_subject"),
    "window_id": ("window_id", "window_uid", "sample_id", "uid"),
    "label": ("label", "y_true", "test_label", "target"),
    "prob": ("prob", "y_prob", "positive_prob", "test_prob", "deploy_watch_prob", "watch_prob"),
    "threshold": (
        "threshold",
        "base_threshold",
        "selected_threshold",
        "watch_threshold",
        "deploy_watch_threshold",
        "val_threshold",
    ),
}

REQUIRED_COLUMNS = ("subject_id", "window_id", "label", "prob", "threshold")
METRICS = ("balanced_acc", "auroc", "f1")


@dataclass(frozen=True)
class MethodSpec:
    dataset: str
    group: str
    method: str
    path: Path


@dataclass(frozen=True)
class ComparisonSpec:
    dataset: str
    group: str
    model_a: str
    model_b: str
    name: str


def _find_column(df: pd.DataFrame, canonical: str) -> str | None:
    lower_to_original = {str(column).lower(): str(column) for column in df.columns}
    for alias in CANONICAL_ALIASES[canonical]:
        if alias.lower() in lower_to_original:
            return lower_to_original[alias.lower()]
    return None


def _canonicalize_prediction_frame(spec: MethodSpec) -> pd.DataFrame:
    if not spec.path.exists():
        raise FileNotFoundError(spec.path)
    df = pd.read_csv(spec.path)

    values: dict[str, pd.Series | str] = {}
    for canonical in CANONICAL_ALIASES:
        source = _find_column(df, canonical)
        if source is not None:
            values[canonical] = df[source]

    missing = [column for column in REQUIRED_COLUMNS if column not in values]
    if missing:
        raise ValueError(
            f"{spec.path} is not a window-level prediction CSV. Missing required columns: {missing}. "
            "Expected at least subject_id/window_id/label/prob/threshold."
        )

    def column_or_default(name: str, default: str) -> pd.Series | str:
        value = values.get(name, default)
        return value

    out = pd.DataFrame(
        {
            # The registry/--method spec is authoritative. This lets the same
            # prediction file be reused under a different analysis group/name
            # (e.g., motion_aware as the no-KD baseline for KD comparisons)
            # without duplicating rows under the CSV's original metadata.
            "dataset": spec.dataset,
            "group": spec.group,
            "method": spec.method,
            "fold": column_or_default("fold", values["subject_id"]),
            "subject_id": values["subject_id"],
            "window_id": values["window_id"],
            "label": values["label"],
            "prob": values["prob"],
            "threshold": values["threshold"],
        }
    )

    out["dataset"] = out["dataset"].astype(str).replace({"nan": spec.dataset})
    out["group"] = out["group"].astype(str).replace({"nan": spec.group})
    out["method"] = out["method"].astype(str).replace({"nan": spec.method})
    out["fold"] = out["fold"].astype(str)
    out["subject_id"] = out["subject_id"].astype(str)
    out["window_id"] = out["window_id"].astype(str)
    out["label"] = pd.to_numeric(out["label"], errors="raise").astype(int)
    out["prob"] = pd.to_numeric(out["prob"], errors="raise").astype(float)
    out["threshold"] = pd.to_numeric(out["threshold"], errors="raise").astype(float)

    if not out["label"].isin([0, 1]).all():
        bad = sorted(out.loc[~out["label"].isin([0, 1]), "label"].unique().tolist())
        raise ValueError(f"{spec.path} has non-binary labels: {bad}")
    if ((out["prob"] < 0.0) | (out["prob"] > 1.0)).any():
        raise ValueError(f"{spec.path} contains probabilities outside [0, 1].")
    if ((out["threshold"] < 0.0) | (out["threshold"] > 1.0)).any():
        raise ValueError(f"{spec.path} contains thresholds outside [0, 1].")

    key = ["dataset", "group", "method", "subject_id", "window_id"]
    dup = out.duplicated(key).sum()
    if dup:
        duplicated = out.loc[out.duplicated(key, keep=False), key].head(10)
        raise ValueError(f"{spec.path} has duplicate prediction rows for {key}; first rows:\n{duplicated}")

    conflicts = (
        out.groupby(["dataset", "group", "method", "subject_id"])["threshold"]
        .nunique(dropna=False)
        .reset_index(name="n_thresholds")
    )
    conflicts = conflicts[conflicts["n_thresholds"] > 1]
    if not conflicts.empty:
        raise ValueError(f"{spec.path} has conflicting thresholds within subject/model:\n{conflicts.head(20)}")

    return out


def _parse_method(value: str) -> MethodSpec:
    if "=" not in value:
        raise ValueError(f"Expected DATASET:GROUP:METHOD=CSV, got {value!r}")
    lhs, rhs = value.split("=", 1)
    parts = lhs.split(":")
    if len(parts) != 3:
        raise ValueError(f"Expected DATASET:GROUP:METHOD=CSV, got {value!r}")
    dataset, group, method = (part.strip() for part in parts)
    if not dataset or not group or not method:
        raise ValueError(f"Empty dataset/group/method in {value!r}")
    return MethodSpec(dataset=dataset, group=group, method=method, path=Path(rhs.strip()))


def _parse_comparison(value: str) -> ComparisonSpec:
    if "=" not in value:
        raise ValueError(f"Expected DATASET:GROUP:MODEL_A=MODEL_B, got {value!r}")
    lhs, rhs = value.split("=", 1)
    parts = lhs.split(":")
    if len(parts) != 3:
        raise ValueError(f"Expected DATASET:GROUP:MODEL_A=MODEL_B, got {value!r}")
    dataset, group, model_a = (part.strip() for part in parts)
    model_b = rhs.strip()
    if not dataset or not group or not model_a or not model_b:
        raise ValueError(f"Invalid comparison {value!r}")
    return ComparisonSpec(
        dataset=dataset,
        group=group,
        model_a=model_a,
        model_b=model_b,
        name=f"{model_a} - {model_b}",
    )


def _read_registry(path: Path) -> list[MethodSpec]:
    df = pd.read_csv(path)
    required = {"dataset", "group", "method", "path"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing registry columns: {sorted(missing)}")
    return [
        MethodSpec(
            dataset=str(row.dataset),
            group=str(row.group),
            method=str(row.method),
            path=Path(str(row.path)),
        )
        for row in df.itertuples(index=False)
    ]


def _metric_values(labels: np.ndarray, probs: np.ndarray, threshold: float) -> dict[str, float]:
    if len(np.unique(labels)) < 2:
        raise ValueError("AUROC undefined because labels contain a single class.")
    preds = (probs >= threshold).astype(int)
    return {
        "balanced_acc": float(balanced_accuracy_score(labels, preds)),
        "auroc": float(roc_auc_score(labels, probs)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
    }


def _subject_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for key, group in frame.groupby(["dataset", "group", "method", "subject_id"], sort=True):
        dataset, group_name, method, subject_id = key
        labels = group["label"].to_numpy(dtype=int)
        probs = group["prob"].to_numpy(dtype=float)
        thresholds = group["threshold"].unique()
        if len(thresholds) != 1:
            raise ValueError(f"Conflicting thresholds for {key}: {thresholds}")
        try:
            metrics = _metric_values(labels, probs, float(thresholds[0]))
        except ValueError as exc:
            raise ValueError(f"{key}: {exc}") from exc
        rows.append(
            {
                "dataset": dataset,
                "group": group_name,
                "method": method,
                "subject_id": subject_id,
                "n_windows": int(len(group)),
                "threshold": float(thresholds[0]),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def _paired_windows(a: pd.DataFrame, b: pd.DataFrame, comparison: ComparisonSpec) -> None:
    subjects_a = set(a["subject_id"].astype(str))
    subjects_b = set(b["subject_id"].astype(str))
    if subjects_a != subjects_b:
        raise ValueError(
            f"{comparison.name} subject sets differ. "
            f"Only A={sorted(subjects_a - subjects_b)}, only B={sorted(subjects_b - subjects_a)}"
        )
    for subject_id in sorted(subjects_a):
        aa = a[a["subject_id"].astype(str) == subject_id][["window_id", "label"]].sort_values("window_id")
        bb = b[b["subject_id"].astype(str) == subject_id][["window_id", "label"]].sort_values("window_id")
        if set(aa["window_id"]) != set(bb["window_id"]):
            raise ValueError(f"{comparison.name} window_id mismatch for subject {subject_id}")
        merged = aa.merge(bb, on="window_id", suffixes=("_a", "_b"), validate="one_to_one")
        if not (merged["label_a"].to_numpy() == merged["label_b"].to_numpy()).all():
            bad = merged[merged["label_a"] != merged["label_b"]].head(10)
            raise ValueError(f"{comparison.name} label mismatch for subject {subject_id}:\n{bad}")


def _bootstrap_ci(values: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return float("nan"), float("nan")
    if n_boot <= 0 or len(values) == 1:
        mean = float(values.mean())
        return mean, mean
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(int(n_boot), len(values)))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _summarize_comparison(
    subject_metrics: pd.DataFrame,
    windows: pd.DataFrame,
    comparison: ComparisonSpec,
    metrics: list[str],
    n_boot: int,
    seed: int,
    tolerance: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    a_key = (comparison.dataset, comparison.group, comparison.model_a)
    b_key = (comparison.dataset, comparison.group, comparison.model_b)
    a_win = windows[
        (windows["dataset"] == a_key[0])
        & (windows["group"] == a_key[1])
        & (windows["method"] == a_key[2])
    ].copy()
    b_win = windows[
        (windows["dataset"] == b_key[0])
        & (windows["group"] == b_key[1])
        & (windows["method"] == b_key[2])
    ].copy()
    if a_win.empty or b_win.empty:
        raise ValueError(f"Missing predictions for comparison {comparison}")
    _paired_windows(a_win, b_win, comparison)

    a = subject_metrics[
        (subject_metrics["dataset"] == a_key[0])
        & (subject_metrics["group"] == a_key[1])
        & (subject_metrics["method"] == a_key[2])
    ]
    b = subject_metrics[
        (subject_metrics["dataset"] == b_key[0])
        & (subject_metrics["group"] == b_key[1])
        & (subject_metrics["method"] == b_key[2])
    ]
    rows: list[dict[str, object]] = []
    detail_frames: list[pd.DataFrame] = []
    for metric in metrics:
        merged = a[["subject_id", metric]].merge(
            b[["subject_id", metric]],
            on="subject_id",
            suffixes=("_a", "_b"),
            validate="one_to_one",
        )
        if len(merged) != a["subject_id"].nunique() or len(merged) != b["subject_id"].nunique():
            raise ValueError(f"Subject-level metric merge lost rows for {comparison.name} / {metric}")
        merged["delta"] = merged[f"{metric}_a"] - merged[f"{metric}_b"]
        deltas = merged["delta"].to_numpy(dtype=float)
        ci_low, ci_high = _bootstrap_ci(deltas, n_boot=n_boot, seed=seed)
        wins = int(np.sum(deltas > tolerance))
        losses = int(np.sum(deltas < -tolerance))
        ties = int(len(deltas) - wins - losses)
        rows.append(
            {
                "dataset": comparison.dataset,
                "group": comparison.group,
                "comparison": comparison.name,
                "model_a": comparison.model_a,
                "model_b": comparison.model_b,
                "metric": metric,
                "n_subjects": int(len(deltas)),
                "mean_delta": float(np.mean(deltas)),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "wins": wins,
                "losses": losses,
                "ties": ties,
                "n_boot": int(n_boot),
                "seed": int(seed),
                "tolerance": float(tolerance),
            }
        )
        details = pd.DataFrame(
            {
                "dataset": comparison.dataset,
                "group": comparison.group,
                "comparison": comparison.name,
                "model_a": comparison.model_a,
                "model_b": comparison.model_b,
                "subject_id": merged["subject_id"],
                "metric": metric,
                "metric_a": merged[f"{metric}_a"],
                "metric_b": merged[f"{metric}_b"],
                "delta": merged["delta"],
            }
        )
        detail_frames.append(details)
    return pd.DataFrame(rows), pd.concat(detail_frames, ignore_index=True)


def _write_compact_table(summary: pd.DataFrame, output_path: Path) -> None:
    metric_names = {"balanced_acc": "BA", "auroc": "AUROC", "f1": "F1"}
    rows: list[dict[str, object]] = []
    for (dataset, group, comparison), frame in summary.groupby(["dataset", "group", "comparison"], sort=True):
        row: dict[str, object] = {"Dataset": dataset, "Group": group, "Comparison": comparison}
        for metric, label in metric_names.items():
            metric_row = frame[frame["metric"] == metric]
            if metric_row.empty:
                continue
            item = metric_row.iloc[0]
            row[f"Delta {label} [95% CI]"] = f"{item.mean_delta:+.3f} [{item.ci_low:+.3f}, {item.ci_high:+.3f}]"
            if metric == "balanced_acc":
                row["BA W/L/T"] = f"{int(item.wins)}/{int(item.losses)}/{int(item.ties)}"
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired subject-level bootstrap CIs from window-level predictions.")
    parser.add_argument("--registry", type=Path, default=None, help="CSV with dataset,group,method,path columns.")
    parser.add_argument(
        "--method",
        action="append",
        default=[],
        help="DATASET:GROUP:METHOD=window_predictions.csv. Can be repeated.",
    )
    parser.add_argument(
        "--comparison",
        action="append",
        required=True,
        help="DATASET:GROUP:MODEL_A=MODEL_B. Delta is MODEL_A minus MODEL_B.",
    )
    parser.add_argument("--metric", action="append", choices=METRICS, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tolerance", type=float, default=1e-12)
    args = parser.parse_args()

    specs: list[MethodSpec] = []
    if args.registry is not None:
        specs.extend(_read_registry(args.registry))
    specs.extend(_parse_method(item) for item in args.method)
    if not specs:
        raise ValueError("Provide --registry or at least one --method.")

    frames = [_canonicalize_prediction_frame(spec) for spec in specs]
    windows = pd.concat(frames, ignore_index=True)
    metrics = list(args.metric or METRICS)
    subject_metrics = _subject_metrics(windows)
    comparisons = [_parse_comparison(item) for item in args.comparison]

    summary_frames: list[pd.DataFrame] = []
    detail_frames: list[pd.DataFrame] = []
    for comparison in comparisons:
        summary, details = _summarize_comparison(
            subject_metrics=subject_metrics,
            windows=windows,
            comparison=comparison,
            metrics=metrics,
            n_boot=args.bootstrap,
            seed=args.seed,
            tolerance=args.tolerance,
        )
        summary_frames.append(summary)
        detail_frames.append(details)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_df = pd.concat(summary_frames, ignore_index=True)
    details_df = pd.concat(detail_frames, ignore_index=True)
    subject_metrics.to_csv(args.output_dir / "subject_level_metrics.csv", index=False)
    summary_df.to_csv(args.output_dir / "paired_bootstrap_ci.csv", index=False)
    details_df.to_csv(args.output_dir / "paired_subject_deltas.csv", index=False)
    _write_compact_table(summary_df, args.output_dir / "paired_bootstrap_compact.csv")

    print(f"saved_subject_metrics={args.output_dir / 'subject_level_metrics.csv'}")
    print(f"saved_summary={args.output_dir / 'paired_bootstrap_ci.csv'}")
    print(f"saved_deltas={args.output_dir / 'paired_subject_deltas.csv'}")


if __name__ == "__main__":
    main()
