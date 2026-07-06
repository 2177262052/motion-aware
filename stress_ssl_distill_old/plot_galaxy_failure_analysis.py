from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score


MOTION_ORDER = ["low", "mid", "high"]
DEFAULT_MODEL_ORDER = ["WatchMotion", "Ours"]

MODEL_COLORS = {
    "WatchMotion": "#4c78a8",
    "Ours": "#2f855a",
    "base": "#8c8c8c",
    "deploy": "#2f855a",
}

RATE_COLORS = {
    "rescue_rate": "#2f855a",
    "harm_rate": "#d95f02",
}

METRIC_LABELS = {
    "acc": "Accuracy",
    "balanced_acc": "Balanced accuracy",
    "f1": "F1",
    "auroc": "AUROC",
    "positive_rate": "Positive rate",
}


def parse_order(values: list[str] | None, fallback: Iterable[str]) -> list[str]:
    if not values:
        return list(fallback)
    return [str(item) for item in values]


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def add_motion_bin(frame: pd.DataFrame, column: str = "acc_jerk_rms") -> pd.DataFrame:
    if column not in frame.columns:
        raise ValueError(f"Missing motion column: {column}")
    out = frame.copy()
    values = pd.to_numeric(out[column], errors="coerce")
    out["motion_bin"] = "missing"
    finite_mask = values.notna() & np.isfinite(values)
    if values[finite_mask].nunique() < 3:
        out.loc[finite_mask, "motion_bin"] = "all"
        return out

    try:
        bins = pd.qcut(values[finite_mask], q=3, duplicates="drop")
    except ValueError:
        out.loc[finite_mask, "motion_bin"] = "all"
        return out

    categories = list(bins.cat.categories)
    labels = MOTION_ORDER[: len(categories)]
    bins = bins.cat.rename_categories(labels)
    out.loc[finite_mask, "motion_bin"] = bins.astype(str)
    return out


def available_motion_order(frame: pd.DataFrame) -> list[str]:
    present = set(frame["motion_bin"].astype(str))
    ordered = [item for item in MOTION_ORDER if item in present]
    extras = sorted(present.difference(MOTION_ORDER).difference({"missing"}))
    if "missing" in present:
        extras.append("missing")
    return ordered + extras


def safe_binary_metrics(
    labels: pd.Series,
    preds: pd.Series,
    probs: pd.Series | None = None,
) -> dict[str, float]:
    y_true = pd.to_numeric(labels, errors="coerce")
    y_pred = pd.to_numeric(preds, errors="coerce")
    pred_mask = y_true.notna() & y_pred.notna() & np.isfinite(y_true) & np.isfinite(y_pred)
    if pred_mask.sum() == 0:
        metrics = {
            "n": 0,
            "positive_prior": float("nan"),
            "acc": float("nan"),
            "balanced_acc": float("nan"),
            "f1": float("nan"),
            "positive_rate": float("nan"),
        }
    else:
        yt = y_true[pred_mask].astype(int)
        yp = y_pred[pred_mask].astype(int)
        metrics = {
            "n": int(pred_mask.sum()),
            "positive_prior": float(yt.mean()),
            "acc": float(accuracy_score(yt, yp)),
            "balanced_acc": float(balanced_accuracy_score(yt, yp)) if yt.nunique() > 1 else float("nan"),
            "f1": float(f1_score(yt, yp, zero_division=0)),
            "positive_rate": float(yp.mean()),
        }

    if probs is None:
        metrics["auroc"] = float("nan")
        return metrics

    y_prob = pd.to_numeric(probs, errors="coerce")
    prob_mask = y_true.notna() & y_prob.notna() & np.isfinite(y_true) & np.isfinite(y_prob)
    if prob_mask.sum() == 0:
        metrics["auroc"] = float("nan")
    else:
        yt_prob = y_true[prob_mask].astype(int)
        yp_prob = y_prob[prob_mask].astype(float)
        if yt_prob.nunique() < 2 or yp_prob.nunique() < 2:
            metrics["auroc"] = float("nan")
        else:
            metrics["auroc"] = float(roc_auc_score(yt_prob, yp_prob))
    return metrics


def build_motion_performance(windows: pd.DataFrame, model_order: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    selected = windows[windows["model_name"].isin(model_order)].copy()
    for (model_name, motion_bin), group in selected.groupby(["model_name", "motion_bin"], dropna=False):
        if "deploy_pred" not in group.columns:
            continue
        metrics = safe_binary_metrics(group["label"], group["deploy_pred"], group.get("deploy_prob"))
        rows.append({"model_name": model_name, "motion_bin": str(motion_bin), **metrics})
    return pd.DataFrame(rows)


def build_sgpc_rates(windows: pd.DataFrame, ours_name: str) -> pd.DataFrame:
    selected = windows[windows["model_name"] == ours_name].copy()
    required = {"sgpc_rescue", "sgpc_harm"}
    missing = required.difference(selected.columns)
    if missing:
        raise ValueError(f"Missing SGPC columns in windows CSV: {sorted(missing)}")

    rows: list[dict[str, object]] = []
    for motion_bin, group in selected.groupby("motion_bin", dropna=False):
        rescue = pd.to_numeric(group["sgpc_rescue"], errors="coerce")
        harm = pd.to_numeric(group["sgpc_harm"], errors="coerce")
        valid_rescue = rescue.notna() & np.isfinite(rescue)
        valid_harm = harm.notna() & np.isfinite(harm)
        rows.append(
            {
                "model_name": ours_name,
                "motion_bin": str(motion_bin),
                "n": int(len(group)),
                "positive_prior": float(pd.to_numeric(group["label"], errors="coerce").mean()),
                "rescue_rate": float(rescue[valid_rescue].mean()) if valid_rescue.any() else float("nan"),
                "harm_rate": float(harm[valid_harm].mean()) if valid_harm.any() else float("nan"),
                "net_rescue_rate": (
                    float(rescue[valid_rescue].mean() - harm[valid_harm].mean())
                    if valid_rescue.any() and valid_harm.any()
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def build_base_deploy_performance(windows: pd.DataFrame, ours_name: str) -> pd.DataFrame:
    selected = windows[windows["model_name"] == ours_name].copy()
    rows: list[dict[str, object]] = []
    outputs = [
        ("base", "base_pred", "base_prob"),
        ("deploy", "deploy_pred", "deploy_prob"),
    ]
    for motion_bin, group in selected.groupby("motion_bin", dropna=False):
        for output_name, pred_col, prob_col in outputs:
            if pred_col not in group.columns:
                continue
            metrics = safe_binary_metrics(group["label"], group[pred_col], group.get(prob_col))
            rows.append({"model_name": ours_name, "output": output_name, "motion_bin": str(motion_bin), **metrics})
    return pd.DataFrame(rows)


def value_label(ax: plt.Axes, x: float, y: float, text: str, dy: float = 0.012) -> None:
    if not np.isfinite(y):
        return
    ax.text(x, y + dy, text, ha="center", va="bottom", fontsize=8, color="#2d3748")


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, formats: list[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(output_dir / f"{stem}.{fmt}", bbox_inches="tight")
    plt.close(fig)


def plot_motion_performance(
    performance: pd.DataFrame,
    motion_order: list[str],
    model_order: list[str],
    output_dir: Path,
    formats: list[str],
) -> None:
    metrics = [("balanced_acc", "Balanced accuracy"), ("f1", "F1")]
    fig, axes = plt.subplots(1, len(metrics), figsize=(10.8, 4.1), sharey=True)
    axes_arr = np.asarray(axes).reshape(-1)
    x = np.arange(len(motion_order))
    width = 0.34

    for ax, (metric, title) in zip(axes_arr, metrics):
        for idx, model_name in enumerate(model_order):
            values = (
                performance[performance["model_name"] == model_name]
                .set_index("motion_bin")
                .reindex(motion_order)[metric]
                .to_numpy(dtype=float)
            )
            offset = (idx - (len(model_order) - 1) / 2.0) * width
            bars = ax.bar(
                x + offset,
                values,
                width=width,
                label=model_name,
                color=MODEL_COLORS.get(model_name, "#4a5568"),
                edgecolor="#1a202c",
                linewidth=0.5,
            )
            for bar, value in zip(bars, values):
                value_label(ax, bar.get_x() + bar.get_width() / 2.0, value, f"{value:.2f}")

        ax.axhline(0.5, color="#a0aec0", linewidth=0.9, linestyle="--", zorder=0)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([item.capitalize() for item in motion_order])
        ax.set_ylim(0.0, 1.0)
        ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
        ax.set_xlabel("Motion level")
    axes_arr[0].set_ylabel("Score")
    axes_arr[-1].legend(frameon=False, loc="lower right")
    fig.suptitle("Deploy performance across motion levels", y=1.03, fontsize=12)
    fig.tight_layout()
    save_figure(fig, output_dir, "motion_bin_deploy_performance", formats)


def plot_sgpc_rescue_harm(
    rates: pd.DataFrame,
    motion_order: list[str],
    output_dir: Path,
    formats: list[str],
) -> None:
    rates = rates.set_index("motion_bin").reindex(motion_order).reset_index()
    x = np.arange(len(motion_order))
    width = 0.36
    fig, ax = plt.subplots(figsize=(6.7, 4.1))
    for idx, (column, label) in enumerate([("rescue_rate", "Rescue"), ("harm_rate", "Harm")]):
        values = rates[column].to_numpy(dtype=float)
        offset = (idx - 0.5) * width
        bars = ax.bar(
            x + offset,
            values,
            width=width,
            label=label,
            color=RATE_COLORS[column],
            edgecolor="#1a202c",
            linewidth=0.5,
        )
        for bar, value in zip(bars, values):
            value_label(ax, bar.get_x() + bar.get_width() / 2.0, value, f"{100 * value:.1f}%", dy=0.004)

    ax.axhline(0.0, color="#2d3748", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([item.capitalize() for item in motion_order])
    ax.set_ylabel("Window rate")
    ax.set_xlabel("Motion level")
    ax.set_ylim(0.0, max(0.16, float(np.nanmax(rates[["rescue_rate", "harm_rate"]].to_numpy(dtype=float))) + 0.04))
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("SGPC correction outcomes")
    fig.tight_layout()
    save_figure(fig, output_dir, "sgpc_rescue_harm_by_motion", formats)


def plot_base_to_deploy(
    performance: pd.DataFrame,
    motion_order: list[str],
    output_dir: Path,
    formats: list[str],
) -> None:
    metrics = [("balanced_acc", "Balanced accuracy"), ("f1", "F1")]
    output_order = ["base", "deploy"]
    output_labels = {"base": "Base path", "deploy": "SGPC deploy"}
    fig, axes = plt.subplots(1, len(metrics), figsize=(10.8, 4.1), sharey=True)
    axes_arr = np.asarray(axes).reshape(-1)
    x = np.arange(len(motion_order))
    width = 0.34

    for ax, (metric, title) in zip(axes_arr, metrics):
        for idx, output_name in enumerate(output_order):
            values = (
                performance[performance["output"] == output_name]
                .set_index("motion_bin")
                .reindex(motion_order)[metric]
                .to_numpy(dtype=float)
            )
            offset = (idx - 0.5) * width
            bars = ax.bar(
                x + offset,
                values,
                width=width,
                label=output_labels.get(output_name, output_name),
                color=MODEL_COLORS.get(output_name, "#4a5568"),
                edgecolor="#1a202c",
                linewidth=0.5,
            )
            for bar, value in zip(bars, values):
                value_label(ax, bar.get_x() + bar.get_width() / 2.0, value, f"{value:.2f}")
        ax.axhline(0.5, color="#a0aec0", linewidth=0.9, linestyle="--", zorder=0)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([item.capitalize() for item in motion_order])
        ax.set_xlabel("Motion level")
        ax.set_ylim(0.0, 1.0)
        ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
    axes_arr[0].set_ylabel("Score")
    axes_arr[-1].legend(frameon=False, loc="lower right")
    fig.suptitle("Internal SGPC effect: base path to deploy output", y=1.03, fontsize=12)
    fig.tight_layout()
    save_figure(fig, output_dir, "sgpc_base_to_deploy_by_motion", formats)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Galaxy failure-analysis figures for motion-aware SGPC.")
    parser.add_argument("--windows-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-order", nargs="*", default=None)
    parser.add_argument("--ours-name", type=str, default="Ours")
    parser.add_argument("--motion-column", type=str, default="acc_jerk_rms")
    parser.add_argument("--formats", nargs="*", default=["png", "pdf"])
    args = parser.parse_args()

    set_plot_style()
    model_order = parse_order(args.model_order, DEFAULT_MODEL_ORDER)
    windows = pd.read_csv(args.windows_csv)
    windows = add_motion_bin(windows, column=args.motion_column)
    motion_order = available_motion_order(windows)
    motion_order = [item for item in motion_order if item in MOTION_ORDER]
    if not motion_order:
        raise ValueError("No low/mid/high motion bins were created.")

    performance = build_motion_performance(windows, model_order)
    sgpc_rates = build_sgpc_rates(windows, args.ours_name)
    base_deploy = build_base_deploy_performance(windows, args.ours_name)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    performance.to_csv(args.output_dir / "motion_bin_deploy_performance.csv", index=False)
    sgpc_rates.to_csv(args.output_dir / "sgpc_rescue_harm_by_motion.csv", index=False)
    base_deploy.to_csv(args.output_dir / "sgpc_base_to_deploy_by_motion.csv", index=False)

    plot_motion_performance(performance, motion_order, model_order, args.output_dir, args.formats)
    plot_sgpc_rescue_harm(sgpc_rates, motion_order, args.output_dir, args.formats)
    plot_base_to_deploy(base_deploy, motion_order, args.output_dir, args.formats)

    print(f"Saved motion-bin deploy performance to {args.output_dir / 'motion_bin_deploy_performance.csv'}")
    print(f"Saved SGPC rescue/harm rates to {args.output_dir / 'sgpc_rescue_harm_by_motion.csv'}")
    print(f"Saved base-to-deploy performance to {args.output_dir / 'sgpc_base_to_deploy_by_motion.csv'}")
    print(f"Saved figures to {args.output_dir}")


if __name__ == "__main__":
    main()
