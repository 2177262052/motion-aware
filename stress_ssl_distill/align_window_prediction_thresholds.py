from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score


SUBJECT_COLUMNS = ("subject", "subject_id", "heldout_subject", "test_subject", "fold")
PLAIN_METRICS = ("acc", "balanced_acc", "f1", "auroc", "positive_rate")


def find_subject_column(frame: pd.DataFrame, path: Path) -> str:
    for column in SUBJECT_COLUMNS:
        if column in frame.columns:
            return column
    raise ValueError(f"{path} has no subject column among {SUBJECT_COLUMNS}")


def pick_column(frame: pd.DataFrame, result_model: str | None, suffix: str) -> str:
    candidates: list[str] = []
    if result_model:
        candidates.append(f"{result_model}_{suffix}")
    candidates.append(suffix)
    if suffix == "threshold":
        candidates.extend(["deploy_watch_threshold", "watch_only_threshold", "val_threshold"])
    for column in candidates:
        if column in frame.columns:
            return column
    raise ValueError(f"Could not find {suffix!r} column. Tried {candidates}. Available={list(frame.columns)}")


def subject_metrics(group: pd.DataFrame) -> dict[str, float]:
    labels = group["label"].to_numpy(dtype=int)
    probs = group["prob"].to_numpy(dtype=float)
    threshold = float(group["threshold"].iloc[0])
    preds = (probs >= threshold).astype(int)
    metrics = {
        "acc": float(accuracy_score(labels, preds)),
        "balanced_acc": float(balanced_accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "positive_rate": float(np.mean(preds)),
    }
    metrics["auroc"] = float(roc_auc_score(labels, probs)) if len(np.unique(labels)) > 1 else float("nan")
    return metrics


def build_result_table(results_csv: Path, result_model: str | None) -> pd.DataFrame:
    result = pd.read_csv(results_csv)
    if result_model is not None and "model" in result.columns:
        result = result[result["model"].astype(str) == str(result_model)].copy()
        if result.empty:
            raise ValueError(f"{results_csv} has a model column, but no rows with model={result_model!r}")
    subject_col = find_subject_column(result, results_csv)
    threshold_col = pick_column(result, result_model, "threshold")

    columns = {
        "subject_id": result[subject_col].astype(str),
        "source_threshold": pd.to_numeric(result[threshold_col], errors="coerce"),
    }
    for metric in PLAIN_METRICS:
        try:
            metric_col = pick_column(result, result_model, metric)
        except ValueError:
            continue
        columns[f"reported_{metric}"] = pd.to_numeric(result[metric_col], errors="coerce")

    table = pd.DataFrame(columns)
    table = table.dropna(subset=["source_threshold"])
    if table.empty:
        raise ValueError(f"No usable thresholds found in {results_csv} with result_model={result_model!r}")
    if table["subject_id"].duplicated().any():
        collapsed_rows: list[pd.Series] = []
        for subject_id, group in table.groupby("subject_id", sort=True):
            threshold_values = group["source_threshold"].dropna().unique()
            if len(threshold_values) != 1:
                raise ValueError(
                    f"{results_csv} has duplicate threshold rows with conflicting thresholds "
                    f"for subject {subject_id}: {threshold_values}"
                )
            collapsed_rows.append(group.iloc[0])
        table = pd.DataFrame(collapsed_rows).reset_index(drop=True)
    bad = table.loc[(table["source_threshold"] < 0.0) | (table["source_threshold"] > 1.0)]
    if not bad.empty:
        raise ValueError(f"{results_csv} has thresholds outside [0,1]:\n{bad.head(10)}")
    return table


def find_metrics_file(metrics_dir: Path, subject_id: str) -> Path:
    candidates = []
    for pattern in (
        f"{subject_id}_deploy_watch_metrics.csv",
        f"{subject_id}_watch_only_metrics.csv",
        f"{subject_id}_metrics.csv",
        f"{subject_id}.csv",
        f"*{subject_id}*metrics.csv",
    ):
        candidates.extend(sorted(metrics_dir.glob(pattern)))
    seen: set[Path] = set()
    unique = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen and candidate.exists():
            unique.append(candidate)
            seen.add(resolved)
    if not unique:
        raise FileNotFoundError(f"No metrics CSV found for subject {subject_id} under {metrics_dir}")
    return unique[0]


def threshold_from_metrics_file(
    metrics_path: Path,
    *,
    result_model: str | None,
    monitor: str,
) -> float:
    history = pd.read_csv(metrics_path)
    if history.empty:
        raise ValueError(f"{metrics_path} is empty.")

    if result_model in {"deploy_watch", "watch", "watch_only", None}:
        threshold_candidates = ["val_watch_threshold", "val_threshold"]
        metric_candidates = [f"val_watch_{monitor}", f"val_{monitor}"]
    elif result_model == "teacher":
        threshold_candidates = ["val_teacher_threshold"]
        metric_candidates = [f"val_teacher_{monitor}"]
    else:
        threshold_candidates = [f"val_{result_model}_threshold", "val_watch_threshold", "val_threshold"]
        metric_candidates = [f"val_{result_model}_{monitor}", f"val_watch_{monitor}", f"val_{monitor}"]

    threshold_col = next((col for col in threshold_candidates if col in history.columns), None)
    metric_col = next((col for col in metric_candidates if col in history.columns), None)
    if threshold_col is None:
        raise ValueError(f"{metrics_path} has no threshold column among {threshold_candidates}")
    if metric_col is None:
        raise ValueError(f"{metrics_path} has no monitor column among {metric_candidates}")

    values = pd.to_numeric(history[metric_col], errors="coerce")
    if values.notna().sum() == 0:
        raise ValueError(f"{metrics_path} monitor column {metric_col} is all NaN.")
    # EarlyStopping keeps the first epoch that reaches a best score because
    # equal scores are not improvements. idxmax returns the first maximum.
    idx = values.idxmax()
    threshold = float(pd.to_numeric(history.loc[idx, threshold_col], errors="raise"))
    if not (0.0 <= threshold <= 1.0):
        raise ValueError(f"{metrics_path} selected threshold outside [0,1]: {threshold}")
    return threshold


def apply_exact_thresholds_from_metrics(
    result_table: pd.DataFrame,
    *,
    metrics_dir: Path,
    result_model: str | None,
    monitor: str,
) -> pd.DataFrame:
    table = result_table.copy()
    exact_thresholds: dict[str, float] = {}
    metrics_paths: dict[str, str] = {}
    for subject_id in table["subject_id"].astype(str):
        metrics_path = find_metrics_file(metrics_dir, subject_id)
        exact_thresholds[subject_id] = threshold_from_metrics_file(
            metrics_path,
            result_model=result_model,
            monitor=monitor,
        )
        metrics_paths[subject_id] = str(metrics_path)
    table["source_threshold_rounded"] = table["source_threshold"]
    table["source_threshold"] = table["subject_id"].map(exact_thresholds).astype(float)
    table["metrics_path"] = table["subject_id"].map(metrics_paths)
    return table


def _reported_error(metrics: dict[str, float], reported: pd.Series) -> float:
    error = 0.0
    count = 0
    for metric in PLAIN_METRICS:
        reported_col = f"reported_{metric}"
        if reported_col not in reported or pd.isna(reported[reported_col]):
            continue
        error += abs(float(metrics[metric]) - float(reported[reported_col]))
        count += 1
    return error / max(count, 1)


def choose_thresholds_by_reported_fit(
    windows: pd.DataFrame,
    result_table: pd.DataFrame,
    *,
    metrics_dir: Path,
    result_model: str | None,
    monitor: str,
) -> pd.DataFrame:
    metrics_table = apply_exact_thresholds_from_metrics(
        result_table,
        metrics_dir=metrics_dir,
        result_model=result_model,
        monitor=monitor,
    )
    result_lookup = result_table.set_index("subject_id")
    chosen_rows: list[pd.Series] = []

    for _, metrics_row in metrics_table.iterrows():
        subject_id = str(metrics_row["subject_id"])
        result_row = result_lookup.loc[subject_id].copy()
        if isinstance(result_row, pd.DataFrame):
            result_row = result_row.iloc[0].copy()
        subject_windows = windows[windows["subject_id"].astype(str) == subject_id].copy()
        if subject_windows.empty:
            chosen_rows.append(metrics_row)
            continue

        result_windows = subject_windows.copy()
        result_windows["threshold"] = float(result_row["source_threshold"])
        metrics_windows = subject_windows.copy()
        metrics_windows["threshold"] = float(metrics_row["source_threshold"])

        result_error = _reported_error(subject_metrics(result_windows), result_row)
        metrics_error = _reported_error(subject_metrics(metrics_windows), metrics_row)
        if result_error <= metrics_error + 1e-12:
            chosen = result_row.copy()
            chosen["source_threshold_rounded"] = result_row["source_threshold"]
            chosen["metrics_threshold"] = metrics_row["source_threshold"]
            chosen["metrics_path"] = metrics_row.get("metrics_path", "")
            chosen["threshold_source"] = "results_csv"
            chosen["result_threshold_error"] = result_error
            chosen["metrics_threshold_error"] = metrics_error
        else:
            chosen = metrics_row.copy()
            chosen["metrics_threshold"] = metrics_row["source_threshold"]
            chosen["threshold_source"] = "metrics_log"
            chosen["result_threshold_error"] = result_error
            chosen["metrics_threshold_error"] = metrics_error
        chosen_rows.append(chosen)

    return pd.DataFrame(chosen_rows).reset_index(drop=True)


def align_thresholds(
    window_csv: Path,
    results_csv: Path,
    output_csv: Path,
    diagnostics_csv: Path,
    result_model: str | None,
    metrics_dir: Path | None,
    monitor: str,
) -> None:
    windows = pd.read_csv(window_csv)
    required = {"subject_id", "label", "prob", "threshold"}
    missing = required - set(windows.columns)
    if missing:
        raise ValueError(f"{window_csv} missing columns: {sorted(missing)}")
    windows["subject_id"] = windows["subject_id"].astype(str)
    windows["label"] = pd.to_numeric(windows["label"], errors="raise").astype(int)
    windows["prob"] = pd.to_numeric(windows["prob"], errors="raise").astype(float)
    windows["threshold"] = pd.to_numeric(windows["threshold"], errors="raise").astype(float)

    result_table = build_result_table(results_csv, result_model=result_model)
    if metrics_dir is not None:
        result_table = choose_thresholds_by_reported_fit(
            windows,
            result_table,
            metrics_dir=metrics_dir,
            result_model=result_model,
            monitor=monitor,
        )
    thresholds = dict(zip(result_table["subject_id"], result_table["source_threshold"]))
    missing_subjects = sorted(set(windows["subject_id"]) - set(thresholds))
    extra_subjects = sorted(set(thresholds) - set(windows["subject_id"]))
    if missing_subjects:
        raise ValueError(f"{results_csv} has no threshold for window subjects: {missing_subjects[:20]}")
    if extra_subjects:
        print(f"note=extra_result_subjects_not_in_windows count={len(extra_subjects)} first={extra_subjects[:10]}")

    before_rows: list[dict[str, object]] = []
    after_rows: list[dict[str, object]] = []
    for subject, group in windows.groupby("subject_id", sort=True):
        before = subject_metrics(group)
        before_rows.append({"subject_id": subject, "stage": "before", **before})

    aligned = windows.copy()
    aligned["original_export_threshold"] = aligned["threshold"]
    aligned["threshold"] = aligned["subject_id"].map(thresholds).astype(float)

    for subject, group in aligned.groupby("subject_id", sort=True):
        after = subject_metrics(group)
        after_rows.append({"subject_id": subject, "stage": "after", **after})

    diagnostics = pd.concat([pd.DataFrame(before_rows), pd.DataFrame(after_rows)], ignore_index=True)
    diagnostics = diagnostics.merge(result_table, on="subject_id", how="left", validate="many_to_one")
    for metric in PLAIN_METRICS:
        reported_col = f"reported_{metric}"
        if reported_col in diagnostics.columns:
            diagnostics[f"delta_vs_reported_{metric}"] = diagnostics[metric] - diagnostics[reported_col]
    diagnostics["threshold"] = diagnostics["subject_id"].map(thresholds).astype(float)
    diagnostics["window_csv"] = str(window_csv)
    diagnostics["results_csv"] = str(results_csv)
    diagnostics["result_model"] = result_model or ""

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_csv.parent.mkdir(parents=True, exist_ok=True)
    aligned.to_csv(output_csv, index=False)
    diagnostics.to_csv(diagnostics_csv, index=False)

    after = diagnostics[diagnostics["stage"] == "after"].copy()
    print(f"saved_aligned={output_csv}")
    print(f"saved_diagnostics={diagnostics_csv}")
    for metric in ("balanced_acc", "f1", "auroc"):
        col = f"delta_vs_reported_{metric}"
        if col in after.columns:
            max_abs = after[col].abs().max()
            mean_abs = after[col].abs().mean()
            print(f"{metric}_after_vs_reported_mean_abs={mean_abs:.8f} max_abs={max_abs:.8f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Align exported window prediction thresholds to original LOSO result CSVs.")
    parser.add_argument("--window-csv", type=Path, required=True)
    parser.add_argument("--results-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--diagnostics-csv", type=Path, required=True)
    parser.add_argument(
        "--result-model",
        type=str,
        default=None,
        help="Use deploy_watch/watch_only/teacher for prefixed result columns; omit for plain threshold columns.",
    )
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=None,
        help="Optional per-fold metrics log directory. If provided, exact best-epoch thresholds are read from logs.",
    )
    parser.add_argument("--monitor", type=str, default="auroc", choices=["acc", "balanced_acc", "f1", "auroc"])
    args = parser.parse_args()
    align_thresholds(
        window_csv=args.window_csv,
        results_csv=args.results_csv,
        output_csv=args.output_csv,
        diagnostics_csv=args.diagnostics_csv,
        result_model=args.result_model,
        metrics_dir=args.metrics_dir,
        monitor=args.monitor,
    )


if __name__ == "__main__":
    main()
