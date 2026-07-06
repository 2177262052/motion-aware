from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score


DEFAULT_METHOD_ORDER = ["Watch-only", "Ours"]
MOTION_ORDER = ["low", "mid", "high"]

COLORS = {
    "Watch-only": "#4c78a8",
    "WatchMotion": "#4c78a8",
    "Ours": "#2f855a",
    "teacher": "#dd6b20",
    "prior": "#111827",
    "collapse": "#c53030",
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


def parse_order(values: list[str] | None, fallback: Iterable[str]) -> list[str]:
    if not values:
        return list(fallback)
    return [str(item) for item in values]


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, formats: list[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(output_dir / f"{stem}.{fmt}", bbox_inches="tight")
    plt.close(fig)


def safe_numeric(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def safe_binary_metrics(labels: pd.Series, preds: pd.Series, probs: pd.Series | None = None) -> dict[str, float]:
    y_true = pd.to_numeric(labels, errors="coerce")
    y_pred = pd.to_numeric(preds, errors="coerce")
    mask = y_true.notna() & y_pred.notna() & np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        out = {
            "n": 0,
            "positive_prior": float("nan"),
            "acc": float("nan"),
            "balanced_acc": float("nan"),
            "f1": float("nan"),
            "positive_rate": float("nan"),
            "positive_rate_error": float("nan"),
        }
    else:
        yt = y_true[mask].astype(int)
        yp = y_pred[mask].astype(int)
        prior = float(yt.mean())
        positive_rate = float(yp.mean())
        out = {
            "n": int(mask.sum()),
            "positive_prior": prior,
            "acc": float(accuracy_score(yt, yp)),
            "balanced_acc": float(balanced_accuracy_score(yt, yp)) if yt.nunique() > 1 else float("nan"),
            "f1": float(f1_score(yt, yp, zero_division=0)),
            "positive_rate": positive_rate,
            "positive_rate_error": abs(positive_rate - prior),
        }

    if probs is None:
        out["auroc"] = float("nan")
        return out

    y_prob = pd.to_numeric(probs, errors="coerce")
    prob_mask = y_true.notna() & y_prob.notna() & np.isfinite(y_true) & np.isfinite(y_prob)
    if prob_mask.sum() == 0:
        out["auroc"] = float("nan")
    else:
        yt_prob = y_true[prob_mask].astype(int)
        yp_prob = y_prob[prob_mask].astype(float)
        if yt_prob.nunique() < 2 or yp_prob.nunique() < 2:
            out["auroc"] = float("nan")
        else:
            out["auroc"] = float(roc_auc_score(yt_prob, yp_prob))
    return out


def true_prior_by_subject(windows: pd.DataFrame) -> pd.DataFrame:
    required = {"fold_subject", "label"}
    missing = required.difference(windows.columns)
    if missing:
        raise ValueError(f"Failure windows CSV missing columns: {sorted(missing)}")
    priors = (
        windows.drop_duplicates(["fold_subject", "window_start_ms", "window_end_ms", "label"])
        .assign(label=lambda df: pd.to_numeric(df["label"], errors="coerce"))
        .groupby("fold_subject", as_index=False)
        .agg(true_positive_prior=("label", "mean"), n_windows=("label", "size"))
        .rename(columns={"fold_subject": "subject"})
    )
    return priors


def build_subject_positive_rate(
    subject_metrics: pd.DataFrame,
    priors: pd.DataFrame,
    method_order: list[str],
) -> pd.DataFrame:
    required = {"method", "subject", "positive_rate", "collapse", "positive_rate_error"}
    missing = required.difference(subject_metrics.columns)
    if missing:
        raise ValueError(f"Subject metrics CSV missing columns: {sorted(missing)}")

    selected = subject_metrics[subject_metrics["method"].isin(method_order)].copy()
    selected = safe_numeric(selected, ["positive_rate", "collapse", "positive_rate_error", "balanced_acc", "auroc", "f1"])
    selected = selected.merge(priors, on="subject", how="left")
    return selected


def add_motion_bin(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    if column not in frame.columns:
        raise ValueError(f"Failure windows CSV missing motion column: {column}")
    out = frame.copy()
    values = pd.to_numeric(out[column], errors="coerce")
    out["motion_bin"] = "missing"
    finite = values.notna() & np.isfinite(values)
    if values[finite].nunique() < 3:
        out.loc[finite, "motion_bin"] = "all"
        return out
    bins = pd.qcut(values[finite], q=3, duplicates="drop")
    labels = MOTION_ORDER[: len(bins.cat.categories)]
    bins = bins.cat.rename_categories(labels)
    out.loc[finite, "motion_bin"] = bins.astype(str)
    return out


def build_motion_failure(windows: pd.DataFrame, baseline_model: str, motion_column: str) -> pd.DataFrame:
    windows = add_motion_bin(windows, motion_column)
    selected = windows[windows["model_name"] == baseline_model].copy()
    if selected.empty:
        available = sorted(windows["model_name"].dropna().astype(str).unique())
        raise ValueError(f"Baseline model {baseline_model!r} not found in failure windows. Available: {available}")
    if "deploy_pred" not in selected.columns:
        raise ValueError("Failure windows CSV missing deploy_pred.")

    rows: list[dict[str, object]] = []
    for motion_bin, group in selected.groupby("motion_bin", dropna=False):
        metrics = safe_binary_metrics(group["label"], group["deploy_pred"], group.get("deploy_prob"))
        rows.append({"model_name": baseline_model, "motion_bin": str(motion_bin), **metrics})
    out = pd.DataFrame(rows)
    out["motion_bin"] = pd.Categorical(out["motion_bin"], categories=MOTION_ORDER, ordered=True)
    return out.sort_values("motion_bin").reset_index(drop=True)


def build_teacher_oracle(windows: pd.DataFrame, ours_model: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = windows[windows["model_name"] == ours_model].copy()
    if selected.empty:
        available = sorted(windows["model_name"].dropna().astype(str).unique())
        raise ValueError(f"Ours model {ours_model!r} not found in failure windows. Available: {available}")
    required = {"fold_subject", "label", "deploy_pred", "teacher_pred"}
    missing = required.difference(selected.columns)
    if missing:
        raise ValueError(f"Failure windows CSV missing teacher/deploy columns: {sorted(missing)}")

    rows: list[dict[str, object]] = []
    for subject, group in selected.groupby("fold_subject", dropna=False):
        deploy = safe_binary_metrics(group["label"], group["deploy_pred"], group.get("deploy_prob"))
        teacher = safe_binary_metrics(group["label"], group["teacher_pred"], group.get("teacher_prob"))
        row = {
            "subject": str(subject),
            "deploy_balanced_acc": deploy["balanced_acc"],
            "teacher_balanced_acc": teacher["balanced_acc"],
            "deploy_f1": deploy["f1"],
            "teacher_f1": teacher["f1"],
            "deploy_auroc": deploy["auroc"],
            "teacher_auroc": teacher["auroc"],
            "deploy_positive_rate": deploy["positive_rate"],
            "teacher_positive_rate": teacher["positive_rate"],
            "positive_prior": deploy["positive_prior"],
            "teacher_minus_deploy_balanced_acc": teacher["balanced_acc"] - deploy["balanced_acc"],
            "teacher_better": int(teacher["balanced_acc"] > deploy["balanced_acc"] + 1e-12),
            "deploy_better": int(deploy["balanced_acc"] > teacher["balanced_acc"] + 1e-12),
        }
        row["tie"] = int(row["teacher_better"] == 0 and row["deploy_better"] == 0)
        rows.append(row)
    subjects = pd.DataFrame(rows).sort_values("subject").reset_index(drop=True)

    selected = safe_numeric(
        selected,
        ["deploy_correct", "teacher_correct", "teacher_rescue_possible", "teacher_harm_risk", "deploy_teacher_pred_disagree"],
    )
    summary = pd.DataFrame(
        [
            {
                "model_name": ours_model,
                "n_windows": int(len(selected)),
                "teacher_better_subjects": int(subjects["teacher_better"].sum()),
                "deploy_better_subjects": int(subjects["deploy_better"].sum()),
                "tie_subjects": int(subjects["tie"].sum()),
                "teacher_rescue_possible_rate": float(selected.get("teacher_rescue_possible", pd.Series(dtype=float)).mean()),
                "teacher_harm_risk_rate": float(selected.get("teacher_harm_risk", pd.Series(dtype=float)).mean()),
                "deploy_teacher_disagree_rate": float(selected.get("deploy_teacher_pred_disagree", pd.Series(dtype=float)).mean()),
            }
        ]
    )
    return subjects, summary


def plot_subject_positive_rate(ax: plt.Axes, data: pd.DataFrame, method_order: list[str]) -> None:
    if data.empty:
        raise ValueError("No subject positive-rate data to plot.")
    subject_order = (
        data[["subject", "true_positive_prior"]]
        .drop_duplicates("subject")
        .sort_values(["true_positive_prior", "subject"], na_position="last")["subject"]
        .tolist()
    )
    x = np.arange(len(subject_order))
    prior = (
        data[["subject", "true_positive_prior"]]
        .drop_duplicates("subject")
        .set_index("subject")
        .reindex(subject_order)["true_positive_prior"]
        .to_numpy(dtype=float)
    )
    ax.plot(x, prior, color=COLORS["prior"], linewidth=1.2, marker="_", markersize=8, label="True prior", zorder=2)

    offsets = np.linspace(-0.14, 0.14, num=max(len(method_order), 1))
    markers = ["o", "^", "s", "D"]
    for idx, method in enumerate(method_order):
        method_data = data[data["method"] == method].set_index("subject").reindex(subject_order)
        values = method_data["positive_rate"].to_numpy(dtype=float)
        collapse = method_data["collapse"].fillna(0).to_numpy(dtype=float) > 0.5
        xx = x + offsets[idx]
        ax.scatter(
            xx,
            values,
            s=32,
            marker=markers[idx % len(markers)],
            color=COLORS.get(method, "#4a5568"),
            edgecolor="#1a202c",
            linewidth=0.4,
            label=method,
            zorder=3,
        )
        if collapse.any():
            ax.scatter(
                xx[collapse],
                values[collapse],
                s=78,
                marker="o",
                facecolors="none",
                edgecolors=COLORS["collapse"],
                linewidth=1.2,
                label=f"{method} collapse" if idx == 0 else None,
                zorder=4,
            )

    ax.axhspan(0.0, 0.05, color="#fed7d7", alpha=0.35, linewidth=0)
    ax.axhspan(0.95, 1.0, color="#fed7d7", alpha=0.35, linewidth=0)
    ax.set_title("(a) Subject-level prediction collapse")
    ax.set_ylabel("Predicted positive rate")
    ax.set_ylim(-0.03, 1.03)
    ax.set_xticks(x)
    ax.set_xticklabels(subject_order, rotation=70, ha="right")
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
    ax.legend(frameon=False, loc="upper left", ncol=2)


def plot_motion_failure(ax: plt.Axes, data: pd.DataFrame, baseline_model: str) -> None:
    data = data.set_index("motion_bin").reindex(MOTION_ORDER)
    x = np.arange(len(MOTION_ORDER))
    pred = data["positive_rate"].to_numpy(dtype=float)
    prior = data["positive_prior"].to_numpy(dtype=float)
    pre = data["positive_rate_error"].to_numpy(dtype=float)

    width = 0.34
    bars = ax.bar(
        x - width / 2,
        pred,
        width=width,
        color=COLORS.get(baseline_model, "#4c78a8"),
        edgecolor="#1a202c",
        linewidth=0.5,
        label="Predicted positive rate",
    )
    ax.scatter(x + width / 2, prior, color=COLORS["prior"], marker="D", s=38, label="True positive prior", zorder=3)
    for idx, value in enumerate(pre):
        if np.isfinite(value):
            ax.text(x[idx], max(pred[idx], prior[idx]) + 0.045, f"err={value:.2f}", ha="center", fontsize=7)

    for bar, value in zip(bars, pred):
        if np.isfinite(value):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.018, f"{value:.2f}", ha="center", fontsize=7)

    ax.set_title(f"(b) Motion bins expose {baseline_model} mismatch")
    ax.set_ylabel("Rate")
    ax.set_xlabel("Motion level")
    ax.set_xticks(x)
    ax.set_xticklabels([item.capitalize() for item in MOTION_ORDER])
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
    ax.legend(frameon=False, loc="upper left")


def plot_teacher_not_oracle(ax: plt.Axes, subjects: pd.DataFrame, summary: pd.DataFrame) -> None:
    ordered = subjects.sort_values("teacher_minus_deploy_balanced_acc")["subject"].tolist()
    x = np.arange(len(ordered))
    frame = subjects.set_index("subject").reindex(ordered)
    teacher = frame["teacher_balanced_acc"].to_numpy(dtype=float)
    deploy = frame["deploy_balanced_acc"].to_numpy(dtype=float)

    for idx in range(len(ordered)):
        if np.isfinite(teacher[idx]) and np.isfinite(deploy[idx]):
            color = "#a0aec0"
            ax.plot([idx, idx], [teacher[idx], deploy[idx]], color=color, linewidth=0.9, zorder=1)
    ax.scatter(x - 0.08, teacher, s=28, color=COLORS["teacher"], edgecolor="#1a202c", linewidth=0.4, label="Teacher", zorder=2)
    ax.scatter(x + 0.08, deploy, s=28, color=COLORS["Ours"], edgecolor="#1a202c", linewidth=0.4, label="Deploy", zorder=3)

    counts = summary.iloc[0]
    note = (
        f"Deploy better: {int(counts['deploy_better_subjects'])}\n"
        f"Teacher better: {int(counts['teacher_better_subjects'])}\n"
        f"Ties: {int(counts['tie_subjects'])}"
    )
    ax.text(
        0.02,
        0.04,
        note,
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
        ha="left",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#e2e8f0"},
    )
    ax.set_title("(c) Privileged teacher is not an oracle")
    ax.set_ylabel("Balanced accuracy")
    ax.set_ylim(0.0, 1.03)
    ax.set_xticks(x)
    ax.set_xticklabels(ordered, rotation=70, ha="right")
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
    ax.legend(frameon=False, loc="upper right")


def plot_panel(
    subject_rates: pd.DataFrame,
    motion_failure: pd.DataFrame,
    teacher_subjects: pd.DataFrame,
    teacher_summary: pd.DataFrame,
    method_order: list[str],
    motion_baseline: str,
    output_dir: Path,
    formats: list[str],
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.4))
    plot_subject_positive_rate(axes[0], subject_rates, method_order)
    plot_motion_failure(axes[1], motion_failure, motion_baseline)
    plot_teacher_not_oracle(axes[2], teacher_subjects, teacher_summary)
    fig.suptitle("Empirical characterization of watch-only deployment failures", y=1.03, fontsize=12)
    fig.tight_layout()
    save_figure(fig, output_dir, "galaxy_problem_characterization", formats)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Galaxy problem characterization panels.")
    parser.add_argument("--subject-metrics-csv", type=Path, required=True)
    parser.add_argument("--failure-windows-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method-order", nargs="*", default=DEFAULT_METHOD_ORDER)
    parser.add_argument("--motion-baseline", type=str, default="WatchMotion")
    parser.add_argument("--ours-name", type=str, default="Ours")
    parser.add_argument("--motion-column", type=str, default="acc_jerk_rms")
    parser.add_argument("--formats", nargs="*", default=["png", "pdf"])
    args = parser.parse_args()

    set_plot_style()
    method_order = parse_order(args.method_order, DEFAULT_METHOD_ORDER)

    subject_metrics = pd.read_csv(args.subject_metrics_csv)
    failure_windows = pd.read_csv(args.failure_windows_csv)

    priors = true_prior_by_subject(failure_windows)
    subject_rates = build_subject_positive_rate(subject_metrics, priors, method_order)
    motion_failure = build_motion_failure(failure_windows, args.motion_baseline, args.motion_column)
    teacher_subjects, teacher_summary = build_teacher_oracle(failure_windows, args.ours_name)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    subject_rates.to_csv(args.output_dir / "problem_subject_positive_rate.csv", index=False)
    motion_failure.to_csv(args.output_dir / "problem_motion_bin_failure.csv", index=False)
    teacher_subjects.to_csv(args.output_dir / "problem_teacher_not_oracle_subjects.csv", index=False)
    teacher_summary.to_csv(args.output_dir / "problem_teacher_not_oracle_summary.csv", index=False)

    plot_panel(
        subject_rates=subject_rates,
        motion_failure=motion_failure,
        teacher_subjects=teacher_subjects,
        teacher_summary=teacher_summary,
        method_order=method_order,
        motion_baseline=args.motion_baseline,
        output_dir=args.output_dir,
        formats=args.formats,
    )

    print(f"Saved subject positive-rate characterization to {args.output_dir / 'problem_subject_positive_rate.csv'}")
    print(f"Saved motion-bin failure characterization to {args.output_dir / 'problem_motion_bin_failure.csv'}")
    print(f"Saved teacher-not-oracle subjects to {args.output_dir / 'problem_teacher_not_oracle_subjects.csv'}")
    print(f"Saved teacher-not-oracle summary to {args.output_dir / 'problem_teacher_not_oracle_summary.csv'}")
    print(f"Saved panel figure to {args.output_dir}")


if __name__ == "__main__":
    main()
