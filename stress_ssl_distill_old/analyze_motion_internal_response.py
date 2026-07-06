from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .analyze_motion_aware_mechanism import (
    DEFAULT_GALAXY_CALM_SESSIONS,
    DEFAULT_GALAXY_STRESS_SESSIONS,
    DEFAULT_WESAD_CALM_SESSIONS,
    DEFAULT_WESAD_STRESS_SESSIONS,
    build_dataset,
    build_loader,
    build_model_from_state,
    discover_manifests,
    find_checkpoint,
    load_state,
    motion_scores,
    select_model_threshold,
)


KEY_COLUMNS = [
    "dataset",
    "fold_subject",
    "row_order",
    "subject_id",
    "session",
    "window_start_ms",
    "window_end_ms",
]

BLUE = "#2F6FBB"
ORANGE = "#E6863B"
GREEN = "#4C9A4C"
RED = "#D85C3A"
PURPLE = "#7B61B3"
GRID = "#D8DDE6"
TEXT = "#111827"


def _safe_norm_ratio(numer: torch.Tensor, denom: torch.Tensor) -> torch.Tensor:
    numer_norm = numer.float().flatten(1).norm(dim=1)
    denom_norm = denom.float().flatten(1).norm(dim=1).clamp(min=1e-6)
    return numer_norm / denom_norm


def _nan_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return 0.0
    x = x[mask]
    y = y[mask]
    if float(np.std(x)) < 1e-8 or float(np.std(y)) < 1e-8:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def signal_quality_metrics(signal_tensor: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return PPG validity, PPG-ACC coupling, and ACC jerk.

    The validity score is intentionally relative rather than clinical: it rewards
    periodic PPG morphology and penalizes high-frequency jitter. The downstream
    figure bins this score within the evaluated data, so its absolute scale is
    not interpreted.
    """

    signal = signal_tensor.detach().cpu().float()
    ppg = signal[:, 0, :].numpy()
    acc = signal[:, 1:4, :].numpy()
    jerk, _ = motion_scores(signal_tensor)

    validities: list[float] = []
    couplings: list[float] = []
    for ppg_i, acc_i in zip(ppg, acc):
        ppg_i = ppg_i.astype(np.float64)
        ppg_centered = ppg_i - np.nanmean(ppg_i)
        if np.nanstd(ppg_centered) < 1e-8:
            validities.append(float("-inf"))
            couplings.append(0.0)
            continue

        # Approximate 20 s windows at 25 Hz in the current manifests. Lags 10-45
        # cover roughly 0.4-1.8 s periods, broad enough for stressed/calm HR.
        n = len(ppg_centered)
        min_lag = max(3, int(round(n / 20.0 * 0.4)))
        max_lag = min(n - 2, int(round(n / 20.0 * 1.8)))
        ac_scores: list[float] = []
        for lag in range(min_lag, max_lag + 1):
            ac_scores.append(_nan_corr(ppg_centered[:-lag], ppg_centered[lag:]))
        periodicity = float(np.nanmax(ac_scores)) if ac_scores else 0.0

        first = np.diff(ppg_centered)
        second = np.diff(first)
        noise_ratio = float(np.sqrt(np.mean(second**2)) / (np.sqrt(np.mean(first**2)) + 1e-6)) if first.size > 2 else 0.0
        validities.append(periodicity - 0.35 * math.log1p(max(noise_ratio, 0.0)))

        acc_mag = np.sqrt(np.sum(acc_i.astype(np.float64) ** 2, axis=0))
        ppg_change = np.abs(np.diff(ppg_centered))
        acc_change = np.abs(np.diff(acc_mag))
        couplings.append(abs(_nan_corr(ppg_change, acc_change)))

    return np.asarray(validities, dtype=np.float32), np.asarray(couplings, dtype=np.float32), jerk


class MotionInternalRecorder:
    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.handles: list[torch.utils.hooks.RemovableHandle] = []
        self.gate_ratios: list[torch.Tensor] = []
        self.enhancer_input: torch.Tensor | None = None
        self.enhancer_output: torch.Tensor | None = None
        self.enhancer_aux: dict[str, torch.Tensor] = {}
        self._register()

    def _register(self) -> None:
        modules = dict(self.model.named_modules())
        for name, module in modules.items():
            is_motion_film = (
                ("motion_gates" in name or "motion_film" in name)
                and hasattr(module, "to_gamma")
                and hasattr(module, "to_beta")
            )
            if is_motion_film:
                self.handles.append(module.register_forward_hook(self._gate_hook))
            if name.endswith("ppg_enhancer"):
                self.handles.append(module.register_forward_hook(self._enhancer_hook))

    def _gate_hook(self, module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        if not inputs:
            return
        x = inputs[0]
        self.gate_ratios.append(_safe_norm_ratio(output.detach() - x.detach(), x.detach()).cpu())

    def _enhancer_hook(self, module: nn.Module, inputs: tuple[torch.Tensor, ...], output: object) -> None:
        if inputs:
            self.enhancer_input = inputs[0].detach().cpu()
        if isinstance(output, tuple) and len(output) == 2 and isinstance(output[1], dict):
            self.enhancer_output = output[0].detach().cpu()
            self.enhancer_aux = {
                str(key): value.detach().cpu()
                for key, value in output[1].items()
                if torch.is_tensor(value)
            }

    def reset(self) -> None:
        self.gate_ratios = []
        self.enhancer_input = None
        self.enhancer_output = None
        self.enhancer_aux = {}

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def summarize_batch(self, batch_size: int) -> dict[str, np.ndarray]:
        zeros = np.zeros((batch_size,), dtype=np.float32)
        if self.gate_ratios:
            min_len = min(item.shape[0] for item in self.gate_ratios)
            stacked = torch.stack([item[:min_len] for item in self.gate_ratios], dim=1)
            adapt = stacked.mean(dim=1).numpy().astype(np.float32)
            if min_len < batch_size:
                adapt = np.pad(adapt, (0, batch_size - min_len), constant_values=np.nan)
        else:
            adapt = zeros.copy()

        ppg_in = self.enhancer_input
        enhanced = self.enhancer_output
        clean = self.enhancer_aux.get("motion_clean_ppg_feats")
        mask = self.enhancer_aux.get("motion_artifact_mask")
        basis = self.enhancer_aux.get("motion_artifact_basis")

        if ppg_in is not None and clean is not None:
            clean_strength = _safe_norm_ratio(clean - ppg_in, ppg_in).numpy().astype(np.float32)
        else:
            clean_strength = zeros.copy()

        if clean is not None and enhanced is not None:
            refine_strength = _safe_norm_ratio(enhanced - clean, clean).numpy().astype(np.float32)
        else:
            refine_strength = zeros.copy()

        if ppg_in is not None and mask is not None and basis is not None:
            artifact_energy = _safe_norm_ratio(mask * basis, ppg_in).numpy().astype(np.float32)
            mask_mean = mask.float().flatten(1).mean(dim=1).numpy().astype(np.float32)
        else:
            artifact_energy = zeros.copy()
            mask_mean = zeros.copy()

        def _fit(values: np.ndarray) -> np.ndarray:
            values = np.asarray(values, dtype=np.float32)
            if len(values) == batch_size:
                return values
            out = np.full((batch_size,), np.nan, dtype=np.float32)
            out[: min(batch_size, len(values))] = values[:batch_size]
            return out

        return {
            "adapt_strength": _fit(adapt),
            "clean_strength": _fit(clean_strength),
            "refine_strength": _fit(refine_strength),
            "artifact_energy": _fit(artifact_energy),
            "mask_mean": _fit(mask_mean),
        }


def collect_basic_predictions(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: str,
    pin_memory: bool,
    threshold: float,
    dataset_kind: str,
    fold_subject: str,
    method: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    model.eval()
    with torch.no_grad():
        for batch_index, batch in enumerate(tqdm(loader, desc=f"{dataset_kind}:{fold_subject}:{method}", leave=False)):
            signal = batch["signal"].to(device, non_blocking=pin_memory)
            wavelet = batch["wavelet_features"].to(device, non_blocking=pin_memory)
            quality = batch["watch_quality"].to(device, non_blocking=pin_memory)
            out = model(signal, wavelet, quality)
            probs = torch.softmax(out["logits"], dim=1)[:, 1].detach().cpu().numpy()
            labels = batch["label"].detach().cpu().numpy().astype(int)
            preds = (probs >= float(threshold)).astype(int)
            ppg_validity, ppg_acc_coupling, acc_jerk = signal_quality_metrics(batch["signal"])
            batch_size = len(labels)
            for idx in range(batch_size):
                rows.append(
                    {
                        "dataset": dataset_kind,
                        "fold_subject": fold_subject,
                        "row_order": batch_index * loader.batch_size + idx if loader.batch_size is not None else len(rows),
                        "subject_id": str(batch["subject_id"][idx]),
                        "session": str(batch["session"][idx]),
                        "window_start_ms": int(batch["window_start_ms"][idx]),
                        "window_end_ms": int(batch["window_end_ms"][idx]),
                        "label": int(labels[idx]),
                        "method": method,
                        "prob": float(probs[idx]),
                        "pred": int(preds[idx]),
                        "threshold": float(threshold),
                        "ppg_validity": float(ppg_validity[idx]),
                        "ppg_acc_coupling": float(ppg_acc_coupling[idx]),
                        "acc_jerk": float(acc_jerk[idx]),
                        "watch_quality": float(quality.detach().cpu()[idx].reshape(-1)[0].item()),
                    }
                )
    return pd.DataFrame(rows)


def collect_motion_internal_predictions(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: str,
    pin_memory: bool,
    threshold: float,
    dataset_kind: str,
    fold_subject: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    recorder = MotionInternalRecorder(model)
    model.eval()
    try:
        with torch.no_grad():
            for batch_index, batch in enumerate(tqdm(loader, desc=f"{dataset_kind}:{fold_subject}:motion_internal", leave=False)):
                recorder.reset()
                signal = batch["signal"].to(device, non_blocking=pin_memory)
                wavelet = batch["wavelet_features"].to(device, non_blocking=pin_memory)
                quality = batch["watch_quality"].to(device, non_blocking=pin_memory)
                out = model(signal, wavelet, quality)
                probs = torch.softmax(out["logits"], dim=1)[:, 1].detach().cpu().numpy()
                labels = batch["label"].detach().cpu().numpy().astype(int)
                preds = (probs >= float(threshold)).astype(int)
                ppg_validity, ppg_acc_coupling, acc_jerk = signal_quality_metrics(batch["signal"])
                internal = recorder.summarize_batch(len(labels))
                batch_size = len(labels)
                for idx in range(batch_size):
                    row = {
                        "dataset": dataset_kind,
                        "fold_subject": fold_subject,
                        "row_order": batch_index * loader.batch_size + idx if loader.batch_size is not None else len(rows),
                        "subject_id": str(batch["subject_id"][idx]),
                        "session": str(batch["session"][idx]),
                        "window_start_ms": int(batch["window_start_ms"][idx]),
                        "window_end_ms": int(batch["window_end_ms"][idx]),
                        "label": int(labels[idx]),
                        "motion_prob": float(probs[idx]),
                        "motion_pred": int(preds[idx]),
                        "motion_threshold": float(threshold),
                        "ppg_validity": float(ppg_validity[idx]),
                        "ppg_acc_coupling": float(ppg_acc_coupling[idx]),
                        "acc_jerk": float(acc_jerk[idx]),
                        "watch_quality": float(quality.detach().cpu()[idx].reshape(-1)[0].item()),
                    }
                    for key, values in internal.items():
                        row[key] = float(values[idx])
                    rows.append(row)
    finally:
        recorder.close()
    return pd.DataFrame(rows)


def merge_baseline_motion(baseline: pd.DataFrame, motion: pd.DataFrame, baseline_name: str) -> pd.DataFrame:
    base_cols = KEY_COLUMNS + ["label", "prob", "pred", "threshold"]
    base = baseline[base_cols].rename(
        columns={
            "prob": "baseline_prob",
            "pred": "baseline_pred",
            "threshold": "baseline_threshold",
        }
    )
    merged = motion.merge(base, on=KEY_COLUMNS + ["label"], how="inner")
    merged["baseline_abs_error"] = (merged["label"].astype(float) - merged["baseline_prob"].astype(float)).abs()
    merged["motion_abs_error"] = (merged["label"].astype(float) - merged["motion_prob"].astype(float)).abs()
    merged["soft_gain_abs_error"] = merged["baseline_abs_error"] - merged["motion_abs_error"]
    labels = merged["label"].astype(int)
    merged["baseline_true_class_prob"] = np.where(labels == 1, merged["baseline_prob"], 1.0 - merged["baseline_prob"])
    merged["motion_true_class_prob"] = np.where(labels == 1, merged["motion_prob"], 1.0 - merged["motion_prob"])
    merged["true_class_prob_gain"] = merged["motion_true_class_prob"] - merged["baseline_true_class_prob"]
    merged["prediction_shift"] = (merged["motion_prob"].astype(float) - merged["baseline_prob"].astype(float)).abs()
    merged["corrected_window"] = (merged["baseline_pred"] != merged["label"]) & (merged["motion_pred"] == merged["label"])
    merged["harmed_window"] = (merged["baseline_pred"] == merged["label"]) & (merged["motion_pred"] != merged["label"])
    merged["outcome_group"] = np.where(
        merged["corrected_window"],
        "corrected",
        np.where(merged["harmed_window"], "harmed", "unchanged"),
    )
    merged["response_strength"] = merged[["adapt_strength", "clean_strength", "refine_strength"]].astype(float).mean(axis=1)
    merged["baseline_name"] = baseline_name
    return merged


def add_bins(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    unique = frame[KEY_COLUMNS + ["ppg_acc_coupling", "ppg_validity"]].drop_duplicates(subset=KEY_COLUMNS).copy()
    unique["coupling_bin"] = pd.qcut(
        unique["ppg_acc_coupling"].rank(method="first"),
        q=3,
        labels=["low", "mid", "high"],
    )
    unique["validity_bin"] = pd.qcut(
        unique["ppg_validity"].rank(method="first"),
        q=2,
        labels=["low", "high"],
    )
    return frame.drop(columns=["coupling_bin", "validity_bin"], errors="ignore").merge(
        unique[KEY_COLUMNS + ["coupling_bin", "validity_bin"]],
        on=KEY_COLUMNS,
        how="left",
    )


def zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    std = values.std(ddof=0)
    if not np.isfinite(std) or std < 1e-12:
        return values * 0.0
    return (values - values.mean()) / std


def summarize_groups(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    unique_motion = frame.drop_duplicates(subset=KEY_COLUMNS).copy()
    response_summary = (
        unique_motion.groupby("coupling_bin", observed=True)[["adapt_strength", "clean_strength", "refine_strength"]]
        .agg(["mean", "sem", "count"])
        .reset_index()
    )
    response_summary.columns = ["_".join([str(part) for part in col if str(part)]) for col in response_summary.columns]

    heatmap_summary = (
        frame.groupby(["baseline_name", "validity_bin", "coupling_bin"], observed=True)
        .agg(
            true_class_prob_gain_mean=("true_class_prob_gain", "mean"),
            soft_gain_abs_error_mean=("soft_gain_abs_error", "mean"),
            n=("true_class_prob_gain", "count"),
        )
        .reset_index()
    )

    compare_cols = ["acc_jerk", "ppg_acc_coupling", "ppg_validity", "adapt_strength", "clean_strength", "refine_strength"]
    standardized = frame.copy()
    for col in compare_cols:
        standardized[f"{col}_z"] = zscore(standardized[col])
    rows: list[dict[str, object]] = []
    for col in compare_cols:
        for (baseline_name, group_name), group in standardized.groupby(["baseline_name", "outcome_group"], observed=True):
            rows.append(
                {
                    "baseline_name": baseline_name,
                    "variable": col,
                    "outcome_group": group_name,
                    "n": int(len(group)),
                    "mean": float(pd.to_numeric(group[col], errors="coerce").mean()),
                    "z_mean": float(pd.to_numeric(group[f"{col}_z"], errors="coerce").mean()),
                }
            )
    outcome_summary = pd.DataFrame(rows)
    return response_summary, heatmap_summary, outcome_summary


def bootstrap_diff_ci(
    corrected: np.ndarray,
    harmed: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
) -> tuple[float, float]:
    corrected = np.asarray(corrected, dtype=np.float64)
    harmed = np.asarray(harmed, dtype=np.float64)
    corrected = corrected[np.isfinite(corrected)]
    harmed = harmed[np.isfinite(harmed)]
    if corrected.size == 0 or harmed.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    corrected_idx = rng.integers(0, corrected.size, size=(n_boot, corrected.size))
    harmed_idx = rng.integers(0, harmed.size, size=(n_boot, harmed.size))
    diffs = corrected[corrected_idx].mean(axis=1) - harmed[harmed_idx].mean(axis=1)
    return float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))


def make_signature_summary(
    frame: pd.DataFrame,
    variables: list[tuple[str, str]],
    n_boot: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for baseline_idx, (baseline_name, base_frame) in enumerate(frame.groupby("baseline_name", sort=False, observed=True)):
        standardized = base_frame.copy()
        for var, label in variables:
            standardized[f"{var}_z"] = zscore(standardized[var])
            corrected = pd.to_numeric(
                standardized.loc[standardized["outcome_group"] == "corrected", f"{var}_z"],
                errors="coerce",
            ).dropna()
            harmed = pd.to_numeric(
                standardized.loc[standardized["outcome_group"] == "harmed", f"{var}_z"],
                errors="coerce",
            ).dropna()
            if corrected.empty or harmed.empty:
                diff = float("nan")
                ci_low = float("nan")
                ci_high = float("nan")
            else:
                diff = float(corrected.mean() - harmed.mean())
                ci_low, ci_high = bootstrap_diff_ci(
                    corrected.to_numpy(),
                    harmed.to_numpy(),
                    seed=seed + baseline_idx * 1009 + len(rows),
                )
            rows.append(
                {
                    "baseline_name": baseline_name,
                    "variable": var,
                    "label": label,
                    "corrected_n": int((standardized["outcome_group"] == "corrected").sum()),
                    "harmed_n": int((standardized["outcome_group"] == "harmed").sum()),
                    "corrected_minus_harmed_z": diff,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                }
            )
    return pd.DataFrame(rows)


def _plot_errorbar(ax, x: np.ndarray, summary: pd.DataFrame, metric: str, label: str, color: str) -> None:
    means = []
    sems = []
    for bin_name in ["low", "mid", "high"]:
        row = summary[summary["coupling_bin"] == bin_name]
        means.append(float(row[f"{metric}_mean"].iloc[0]) if not row.empty else np.nan)
        sems.append(float(row[f"{metric}_sem"].iloc[0]) if not row.empty else 0.0)
    ax.errorbar(
        x,
        means,
        yerr=sems,
        color=color,
        marker="o",
        linewidth=2.0,
        markersize=5,
        capsize=3,
        label=label,
    )


def plot_figure(
    frame: pd.DataFrame,
    response_summary: pd.DataFrame,
    heatmap_summary: pd.DataFrame,
    outcome_summary: pd.DataFrame,
    output_prefix: Path,
    title: str,
    baseline_label: str,
    font_family: str,
) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": font_family,
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.alpha": 0.45,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig = plt.figure(figsize=(9.4, 6.9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.12], hspace=0.46, wspace=0.34)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    x = np.arange(3)
    _plot_errorbar(ax_a, x, response_summary, "adapt_strength", "Adapt", BLUE)
    _plot_errorbar(ax_a, x, response_summary, "clean_strength", "Clean", ORANGE)
    _plot_errorbar(ax_a, x, response_summary, "refine_strength", "Refine", GREEN)
    ax_a.set_xticks(x, ["Low", "Mid", "High"])
    ax_a.set_xlabel("PPG-ACC coupling bin")
    ax_a.set_ylabel("Relative feature change")
    ax_a.set_title("A. Internal response vs coupling", loc="left", fontweight="bold")
    ax_a.legend(frameon=False, fontsize=8)

    baseline_names = list(dict.fromkeys(heatmap_summary["baseline_name"].astype(str).tolist()))
    heat_rows: list[tuple[str, str]] = []
    for baseline_name in baseline_names:
        for vbin in ["low", "high"]:
            heat_rows.append((baseline_name, vbin))
    heat = np.full((len(heat_rows), 3), np.nan, dtype=float)
    for i, (baseline_name, vbin) in enumerate(heat_rows):
        for j, cbin in enumerate(["low", "mid", "high"]):
            row = heatmap_summary[
                (heatmap_summary["baseline_name"].astype(str) == baseline_name)
                & (heatmap_summary["validity_bin"].astype(str) == vbin)
                & (heatmap_summary["coupling_bin"].astype(str) == cbin)
            ]
            if not row.empty:
                heat[i, j] = float(row["true_class_prob_gain_mean"].iloc[0])
    vmax = np.nanmax(np.abs(heat)) if np.isfinite(heat).any() else 0.01
    vmax = max(float(vmax), 1e-4)
    im = ax_b.imshow(heat, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    ax_b.set_xticks(np.arange(3), ["Low", "Mid", "High"])
    ylabels = [f"{baseline_name}\n{vbin} validity" for baseline_name, vbin in heat_rows]
    ax_b.set_yticks(np.arange(len(heat_rows)), ylabels)
    ax_b.set_xlabel("PPG-ACC coupling")
    ax_b.set_ylabel("")
    ax_b.set_title("B. True-class probability gain", loc="left", fontweight="bold")
    for i in range(len(heat_rows)):
        for j in range(3):
            value = heat[i, j]
            if np.isfinite(value):
                ax_b.text(j, i, f"{value:+.3f}", ha="center", va="center", fontsize=8, color=TEXT)
    cbar = fig.colorbar(im, ax=ax_b, fraction=0.046, pad=0.03)
    cbar.set_label("True-class probability gain", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    variables = [
        ("acc_jerk", "ACC jerk"),
        ("ppg_acc_coupling", "PPG--ACC coupling"),
        ("ppg_validity", "PPG validity"),
        ("adapt_strength", "Adapt strength"),
        ("clean_strength", "Clean strength"),
    ]
    signature = make_signature_summary(frame, variables)
    y = np.arange(len(variables), dtype=float)
    baseline_order = baseline_names or [baseline_label]
    offsets = np.linspace(-0.16 * (len(baseline_order) - 1), 0.16 * (len(baseline_order) - 1), len(baseline_order))
    color_cycle = [BLUE, PURPLE, ORANGE]
    max_abs = 0.0
    for baseline_idx, baseline_name in enumerate(baseline_order):
        rows = signature[signature["baseline_name"].astype(str) == baseline_name].set_index("variable")
        xs: list[float] = []
        xerr_low: list[float] = []
        xerr_high: list[float] = []
        for var, _ in variables:
            if var not in rows.index:
                xs.append(float("nan"))
                xerr_low.append(0.0)
                xerr_high.append(0.0)
                continue
            row = rows.loc[var]
            value = float(row["corrected_minus_harmed_z"])
            ci_low = float(row["ci_low"])
            ci_high = float(row["ci_high"])
            xs.append(value)
            xerr_low.append(max(0.0, value - ci_low) if np.isfinite(ci_low) else 0.0)
            xerr_high.append(max(0.0, ci_high - value) if np.isfinite(ci_high) else 0.0)
        xs_array = np.asarray(xs, dtype=float)
        if np.isfinite(xs_array).any():
            finite_lows = xs_array - np.asarray(xerr_low)
            finite_highs = xs_array + np.asarray(xerr_high)
            max_abs = max(max_abs, float(np.nanmax(np.abs(np.concatenate([finite_lows, finite_highs])))))
        ax_c.errorbar(
            xs_array,
            y + offsets[baseline_idx],
            xerr=np.vstack([xerr_low, xerr_high]),
            fmt="o",
            color=color_cycle[baseline_idx % len(color_cycle)],
            ecolor=color_cycle[baseline_idx % len(color_cycle)],
            elinewidth=1.6,
            capsize=3,
            markersize=5.5,
            label=baseline_name,
            alpha=0.95,
        )
    ax_c.axvline(0, color="#374151", linewidth=1.0)
    ax_c.set_yticks(y, [label for _, label in variables])
    ax_c.invert_yaxis()
    ax_c.set_xlabel("Corrected minus harmed signature (z-score mean difference, 95% bootstrap CI)")
    ax_c.set_title("C. Corrected vs harmed window signatures", loc="left", fontweight="bold")
    ax_c.grid(True, axis="x", color=GRID, alpha=0.55)
    ax_c.grid(False, axis="y")
    if max_abs > 0:
        ax_c.set_xlim(-max_abs * 1.18, max_abs * 1.18)
    ax_c.legend(frameon=False, fontsize=8, loc="lower right")

    fig.suptitle(title, y=0.995, fontsize=13, fontweight="bold")
    fig.subplots_adjust(top=0.90, bottom=0.11, left=0.09, right=0.985)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=450, bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def dataframe_to_markdown(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    if df.empty:
        return ""
    headers = list(df.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in df.iterrows():
        values = []
        for col in headers:
            value = row[col]
            if isinstance(value, (float, np.floating)):
                values.append(f"{value:{floatfmt}}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_summary(
    path: Path,
    frame: pd.DataFrame,
    response_summary: pd.DataFrame,
    heatmap_summary: pd.DataFrame,
    outcome_summary: pd.DataFrame,
) -> None:
    lines = ["# Motion-Aware Internal Mechanism", ""]
    lines.extend(
        [
            f"- Windows: {len(frame)}",
            f"- Corrected windows: {int((frame['outcome_group'] == 'corrected').sum())}",
            f"- Harmed windows: {int((frame['outcome_group'] == 'harmed').sum())}",
            f"- Mean abs-error gain: {frame['soft_gain_abs_error'].mean():.4f}",
            f"- Mean true-class probability gain: {frame['true_class_prob_gain'].mean():.4f}",
            f"- Mean adapt strength: {frame['adapt_strength'].mean():.4f}",
            f"- Mean clean strength: {frame['clean_strength'].mean():.4f}",
            f"- Mean refine strength: {frame['refine_strength'].mean():.4f}",
            "",
            "## Response by PPG-ACC Coupling",
            "",
            dataframe_to_markdown(response_summary, ".4f"),
            "",
            "## Gain Heatmap Values",
            "",
            dataframe_to_markdown(heatmap_summary, ".4f"),
            "",
            "## Corrected/Harmed Summary",
            "",
            dataframe_to_markdown(outcome_summary, ".4f"),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze how motion-aware internal responses relate to PPG-motion coupling and corrected/harmed windows."
    )
    parser.add_argument("--dataset-kind", choices=["galaxy", "wesad"], required=True)
    parser.add_argument("--manifests-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", "--wesad-root", dest="dataset_root", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--extra-baseline-checkpoint-dir", type=Path, default=None)
    parser.add_argument("--motion-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    ablation_choices = ["none", "ppg_only", "simple_concat", "ppg_only_refine", "simple_concat_refine"]
    parser.add_argument("--baseline-input-ablation", choices=ablation_choices, default="ppg_only")
    parser.add_argument("--extra-baseline-input-ablation", choices=ablation_choices, default="simple_concat")
    parser.add_argument("--motion-input-ablation", choices=ablation_choices, default="none")
    parser.add_argument("--baseline-label", type=str, default="PPG-only")
    parser.add_argument("--extra-baseline-label", type=str, default="Simple concat")
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--cache-subjects", type=int, default=2)
    parser.add_argument("--threshold-metric", choices=["acc", "balanced_acc", "f1", "auroc"], default="balanced_acc")
    parser.add_argument("--calm-sessions", nargs="*", default=None)
    parser.add_argument("--stress-sessions", nargs="*", default=None)
    parser.add_argument("--watch-model-dim", type=int, default=192)
    parser.add_argument("--watch-transformer-layers", type=int, default=2)
    parser.add_argument("--watch-transformer-heads", type=int, default=4)
    parser.add_argument("--watch-fusion-hidden-dim", type=int, default=256)
    parser.add_argument("--watch-embed-dim", type=int, default=160)
    parser.add_argument("--font-family", type=str, default="Liberation Sans")
    args = parser.parse_args()

    if args.dataset_kind == "galaxy":
        calm_sessions = args.calm_sessions or DEFAULT_GALAXY_CALM_SESSIONS
        stress_sessions = args.stress_sessions or DEFAULT_GALAXY_STRESS_SESSIONS
    else:
        calm_sessions = args.calm_sessions or DEFAULT_WESAD_CALM_SESSIONS
        stress_sessions = args.stress_sessions or DEFAULT_WESAD_STRESS_SESSIONS
    include_sessions = list(calm_sessions) + list(stress_sessions)

    manifests = discover_manifests(args.manifests_dir, args.dataset_kind, args.subjects)
    if not manifests:
        raise ValueError(f"No {args.dataset_kind} manifests found in {args.manifests_dir}")

    all_frames: list[pd.DataFrame] = []
    for subject, manifest in manifests:
        print(f"subject={subject} manifest={manifest}")
        val_ds = build_dataset(
            args.dataset_kind,
            manifest,
            "val",
            args.dataset_root,
            include_sessions,
            cache_subjects=args.cache_subjects,
        )
        test_ds = build_dataset(
            args.dataset_kind,
            manifest,
            "test",
            args.dataset_root,
            include_sessions,
            cache_subjects=args.cache_subjects,
        )
        if len(test_ds) == 0:
            print(f"skip_subject={subject} reason=empty_test_split")
            continue
        val_loader = build_loader(val_ds if len(val_ds) > 0 else test_ds, args.batch_size, args.num_workers, args.pin_memory)
        test_loader = build_loader(test_ds, args.batch_size, args.num_workers, args.pin_memory)

        baseline_model = build_model_from_state(
            load_state(find_checkpoint(args.baseline_checkpoint_dir, subject)),
            device=args.device,
            model_dim=args.watch_model_dim,
            transformer_layers=args.watch_transformer_layers,
            transformer_heads=args.watch_transformer_heads,
            fusion_hidden_dim=args.watch_fusion_hidden_dim,
            embed_dim=args.watch_embed_dim,
            input_ablation=args.baseline_input_ablation,
        )
        extra_baseline_model = None
        if args.extra_baseline_checkpoint_dir is not None:
            extra_baseline_model = build_model_from_state(
                load_state(find_checkpoint(args.extra_baseline_checkpoint_dir, subject)),
                device=args.device,
                model_dim=args.watch_model_dim,
                transformer_layers=args.watch_transformer_layers,
                transformer_heads=args.watch_transformer_heads,
                fusion_hidden_dim=args.watch_fusion_hidden_dim,
                embed_dim=args.watch_embed_dim,
                input_ablation=args.extra_baseline_input_ablation,
            )
        motion_model = build_model_from_state(
            load_state(find_checkpoint(args.motion_checkpoint_dir, subject)),
            device=args.device,
            model_dim=args.watch_model_dim,
            transformer_layers=args.watch_transformer_layers,
            transformer_heads=args.watch_transformer_heads,
            fusion_hidden_dim=args.watch_fusion_hidden_dim,
            embed_dim=args.watch_embed_dim,
            input_ablation=args.motion_input_ablation,
        )

        baseline_threshold = select_model_threshold(
            baseline_model,
            val_loader,
            device=args.device,
            pin_memory=args.pin_memory,
            metric=args.threshold_metric,
        )
        extra_baseline_threshold = None
        if extra_baseline_model is not None:
            extra_baseline_threshold = select_model_threshold(
                extra_baseline_model,
                val_loader,
                device=args.device,
                pin_memory=args.pin_memory,
                metric=args.threshold_metric,
            )
        motion_threshold = select_model_threshold(
            motion_model,
            val_loader,
            device=args.device,
            pin_memory=args.pin_memory,
            metric=args.threshold_metric,
        )
        if extra_baseline_threshold is None:
            print(f"subject={subject} baseline_threshold={baseline_threshold:.4f} motion_threshold={motion_threshold:.4f}")
        else:
            print(
                f"subject={subject} baseline_threshold={baseline_threshold:.4f} "
                f"extra_baseline_threshold={extra_baseline_threshold:.4f} "
                f"motion_threshold={motion_threshold:.4f}"
            )

        baseline_frame = collect_basic_predictions(
            baseline_model,
            test_loader,
            device=args.device,
            pin_memory=args.pin_memory,
            threshold=baseline_threshold,
            dataset_kind=args.dataset_kind,
            fold_subject=subject,
            method="baseline",
        )
        extra_baseline_frame = None
        if extra_baseline_model is not None and extra_baseline_threshold is not None:
            extra_baseline_frame = collect_basic_predictions(
                extra_baseline_model,
                test_loader,
                device=args.device,
                pin_memory=args.pin_memory,
                threshold=extra_baseline_threshold,
                dataset_kind=args.dataset_kind,
                fold_subject=subject,
                method="extra_baseline",
            )
        motion_frame = collect_motion_internal_predictions(
            motion_model,
            test_loader,
            device=args.device,
            pin_memory=args.pin_memory,
            threshold=motion_threshold,
            dataset_kind=args.dataset_kind,
            fold_subject=subject,
        )
        all_frames.append(merge_baseline_motion(baseline_frame, motion_frame, args.baseline_label))
        if extra_baseline_frame is not None:
            all_frames.append(merge_baseline_motion(extra_baseline_frame, motion_frame, args.extra_baseline_label))

    if not all_frames:
        raise ValueError("No windows were collected.")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.concat(all_frames, ignore_index=True)
    frame = add_bins(frame)
    response_summary, heatmap_summary, outcome_summary = summarize_groups(frame)
    signature_summary = make_signature_summary(
        frame,
        [
            ("acc_jerk", "ACC jerk"),
            ("ppg_acc_coupling", "PPG--ACC coupling"),
            ("ppg_validity", "PPG validity"),
            ("adapt_strength", "Adapt strength"),
            ("clean_strength", "Clean strength"),
        ],
    )

    prefix = output_dir / f"{args.dataset_kind}_motion_internal_mechanism"
    frame.to_csv(output_dir / f"{args.dataset_kind}_motion_internal_windows.csv", index=False)
    response_summary.to_csv(output_dir / f"{args.dataset_kind}_motion_internal_response_by_coupling.csv", index=False)
    heatmap_summary.to_csv(output_dir / f"{args.dataset_kind}_motion_internal_gain_heatmap.csv", index=False)
    outcome_summary.to_csv(output_dir / f"{args.dataset_kind}_motion_internal_corrected_harmed.csv", index=False)
    signature_summary.to_csv(output_dir / f"{args.dataset_kind}_motion_internal_signature_dot_whisker.csv", index=False)
    write_summary(output_dir / f"{args.dataset_kind}_motion_internal_mechanism_summary.md", frame, response_summary, heatmap_summary, outcome_summary)
    plot_figure(
        frame,
        response_summary,
        heatmap_summary,
        outcome_summary,
        prefix,
        title=f"{args.dataset_kind.upper()} motion-aware internal mechanism",
        baseline_label=args.baseline_label,
        font_family=args.font_family,
    )
    print(f"Saved windows to {output_dir / f'{args.dataset_kind}_motion_internal_windows.csv'}")
    print(f"Saved figure to {prefix.with_suffix('.png')}")
    print(f"Saved PDF figure to {prefix.with_suffix('.pdf')}")
    print(f"Saved SVG figure to {prefix.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
