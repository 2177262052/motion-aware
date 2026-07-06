from __future__ import annotations

import argparse
import importlib
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


DEFAULT_GALAXY_CALM_SESSIONS = ["baseline"]
DEFAULT_GALAXY_STRESS_SESSIONS = ["tsst-prep"]
DEFAULT_WESAD_CALM_SESSIONS = ["baseline"]
DEFAULT_WESAD_STRESS_SESSIONS = ["stress"]
BIN_ORDER = ["low_motion", "mid_motion", "high_motion"]
METHOD_ORDER = ["watch_only", "motion_aware"]


class ScaledMotionFiLM(nn.Module):
    """Compatibility module for historical scaled-motion checkpoints."""

    def __init__(
        self,
        cond_dim: int,
        target_dim: int,
        *args: object,
        scale_logit_init: float = -2.0,
        **kwargs: object,
    ) -> None:
        super().__init__()
        self.to_gamma = nn.Sequential(
            nn.Linear(cond_dim, target_dim),
            nn.GELU(),
            nn.Linear(target_dim, target_dim),
        )
        self.to_beta = nn.Sequential(
            nn.Linear(cond_dim, target_dim),
            nn.GELU(),
            nn.Linear(target_dim, target_dim),
        )
        self.scale_logit = nn.Parameter(torch.tensor(float(scale_logit_init)))

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        gamma = self.to_gamma(condition).unsqueeze(-1)
        beta = self.to_beta(condition).unsqueeze(-1)
        scale = torch.sigmoid(self.scale_logit)
        return x * (1.0 + scale * torch.tanh(gamma)) + scale * beta


class DeployWatchFromPrivilegedState(nn.Module):
    """Minimal deployable wrapper for checkpoints saved from privileged models."""

    def __init__(self, encoder: nn.Module, embed_dim: int, num_classes: int = 2) -> None:
        super().__init__()
        self.watch_encoder = encoder
        self.watch_classifier = nn.Linear(embed_dim, num_classes)

    def forward(
        self,
        signal: torch.Tensor,
        wavelet_features: torch.Tensor,
        quality: torch.Tensor,
        **kwargs: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        out = self.watch_encoder(signal, wavelet_features, quality, **kwargs)
        logits = self.watch_classifier(out["embedding"])
        return {**out, "logits": logits}


def discover_manifests(manifests_dir: Path, dataset_kind: str, subjects: Iterable[str] | None) -> list[tuple[str, Path]]:
    requested = {str(subject).strip() for subject in subjects or [] if str(subject).strip()}
    prefixes = ["galaxy"] if dataset_kind == "galaxy" else ["wesad"]
    manifests: dict[str, Path] = {}
    for prefix in prefixes:
        for pattern in (f"{prefix}_*_loso_val.csv", "*_loso_val.csv"):
            for path in sorted(manifests_dir.glob(pattern)):
                subject = path.stem
                if subject.startswith(f"{prefix}_"):
                    subject = subject[len(prefix) + 1 :]
                if subject.endswith("_loso_val"):
                    subject = subject[: -len("_loso_val")]
                if dataset_kind == "galaxy" and not subject.upper().startswith("P"):
                    continue
                if dataset_kind == "wesad" and not subject.upper().startswith("S"):
                    continue
                if requested and subject not in requested:
                    continue
                manifests.setdefault(subject, path)
    return sorted(manifests.items())


def find_checkpoint(checkpoint_dir: Path, subject: str) -> Path:
    candidates = [
        checkpoint_dir / f"{subject}.pt",
        checkpoint_dir / f"{subject.upper()}.pt",
        checkpoint_dir / f"{subject.lower()}.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = sorted(checkpoint_dir.glob(f"*{subject}*.pt"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"No checkpoint for subject {subject} under {checkpoint_dir}")


def load_state(path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported checkpoint format: {path}")
    state: dict[str, torch.Tensor] = {}
    for key, value in checkpoint.items():
        if not torch.is_tensor(value):
            continue
        clean_key = key[len("module.") :] if key.startswith("module.") else key
        state[clean_key] = value
    return state


def has_key_fragment(state: dict[str, torch.Tensor], fragment: str) -> bool:
    return any(fragment in key for key in state)


def infer_watch_enhancement(state: dict[str, torch.Tensor]) -> str:
    return "motion_disentangled" if has_key_fragment(state, "ppg_enhancer") else "none"


def infer_motion_mode(state: dict[str, torch.Tensor]) -> str:
    if has_key_fragment(state, "residual_scale"):
        return "residual"
    return "strong"


def uses_scaled_motion(state: dict[str, torch.Tensor]) -> bool:
    return has_key_fragment(state, "scale_logit")


def uses_privileged_prefix(state: dict[str, torch.Tensor]) -> bool:
    return any(key.startswith("watch_encoder.") for key in state) and any(key.startswith("watch_classifier.") for key in state)


def set_scaled_motion_compat(enabled: bool) -> None:
    from . import galaxy_models

    if not hasattr(set_scaled_motion_compat, "_original_motion_film"):
        setattr(set_scaled_motion_compat, "_original_motion_film", galaxy_models.MotionFiLM)
    original_motion_film = getattr(set_scaled_motion_compat, "_original_motion_film")
    galaxy_models.MotionFiLM = ScaledMotionFiLM if enabled else original_motion_film


def restore_galaxy_model_module() -> None:
    from . import galaxy_models

    importlib.reload(galaxy_models)
    if hasattr(set_scaled_motion_compat, "_original_motion_film"):
        delattr(set_scaled_motion_compat, "_original_motion_film")


def install_input_ablation(input_ablation: str) -> None:
    if input_ablation == "none":
        return
    from .run_watch_input_ablation_loso import install_ablation

    install_ablation(input_ablation)


def build_model_from_state(
    state: dict[str, torch.Tensor],
    *,
    device: str,
    model_dim: int,
    transformer_layers: int,
    transformer_heads: int,
    fusion_hidden_dim: int,
    embed_dim: int,
    input_ablation: str,
) -> nn.Module:
    restore_galaxy_model_module()
    set_scaled_motion_compat(uses_scaled_motion(state))
    install_input_ablation(input_ablation)
    from . import galaxy_models

    watch_enhancement = infer_watch_enhancement(state)
    watch_motion_mode = infer_motion_mode(state)
    if uses_privileged_prefix(state):
        encoder = galaxy_models.WaveletGuidedWatchEncoder(
            model_dim=model_dim,
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads,
            fusion_hidden_dim=fusion_hidden_dim,
            embed_dim=embed_dim,
            watch_enhancement=watch_enhancement,
            watch_motion_mode=watch_motion_mode,
        )
        model = DeployWatchFromPrivilegedState(encoder=encoder, embed_dim=embed_dim)
    else:
        model = galaxy_models.WaveletGuidedWatchNet(
            model_dim=model_dim,
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads,
            fusion_hidden_dim=fusion_hidden_dim,
            embed_dim=embed_dim,
            watch_enhancement=watch_enhancement,
            watch_motion_mode=watch_motion_mode,
        )
    missing, unexpected = model.load_state_dict(state, strict=False)
    critical_missing = [
        key for key in missing
        if not key.endswith("scale_logit")
    ]
    if critical_missing:
        print(f"load_warning=missing_keys count={len(critical_missing)} first={critical_missing[:5]}")
    if unexpected:
        print(f"load_note=unexpected_keys count={len(unexpected)} first={unexpected[:5]}")
    return model.to(device).eval()


def build_dataset(
    dataset_kind: str,
    manifest: Path,
    split: str,
    dataset_root: Path,
    include_sessions: list[str],
    cache_subjects: int,
):
    if dataset_kind == "galaxy":
        from .galaxy_dataset import GalaxyWatchWindowDataset

        return GalaxyWatchWindowDataset(
            manifest_csv=manifest,
            split=split,
            dataset_root=dataset_root,
            include_sessions=include_sessions,
            cache_tables=True,
        )
    if dataset_kind == "wesad":
        from .wesad_dataset import WESADWatchWindowDataset

        return WESADWatchWindowDataset(
            manifest_csv=manifest,
            split=split,
            wesad_root=dataset_root,
            include_sessions=include_sessions,
            cache_subjects=cache_subjects,
        )
    raise ValueError(f"Unsupported dataset kind: {dataset_kind}")


def build_loader(dataset, batch_size: int, num_workers: int, pin_memory: bool) -> DataLoader:
    kwargs: dict[str, object] = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **kwargs)


def motion_scores(signal_tensor: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    signal = signal_tensor.detach().cpu().float()
    acc = signal[:, 1:4, :]
    if acc.shape[1] == 0:
        zeros = np.zeros((signal.shape[0],), dtype=np.float32)
        return zeros, zeros
    energy = torch.sqrt(torch.mean(torch.sum(acc.pow(2), dim=1), dim=1).clamp(min=0.0))
    if acc.shape[-1] > 1:
        diff = acc.diff(dim=-1)
        jerk = torch.sqrt(torch.mean(torch.sum(diff.pow(2), dim=1), dim=1).clamp(min=0.0))
    else:
        jerk = torch.zeros_like(energy)
    return jerk.numpy(), energy.numpy()


def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: str,
    pin_memory: bool,
    threshold: float | None = None,
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
            logits = model(signal, wavelet, quality)["logits"]
            probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            labels = batch["label"].detach().cpu().numpy().astype(int)
            jerk, energy = motion_scores(batch["signal"])
            preds = (probs >= float(threshold)).astype(int) if threshold is not None else np.zeros_like(labels)
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
                        "threshold": float(threshold) if threshold is not None else float("nan"),
                        "acc_jerk": float(jerk[idx]),
                        "acc_energy": float(energy[idx]),
                        "watch_quality": float(quality.detach().cpu()[idx].reshape(-1)[0].item()),
                    }
                )
    return pd.DataFrame(rows)


def select_model_threshold(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: str,
    pin_memory: bool,
    metric: str,
) -> float:
    frame = collect_predictions(
        model,
        loader,
        device=device,
        pin_memory=pin_memory,
        threshold=None,
        dataset_kind="val",
        fold_subject="val",
        method="val",
    )
    if frame.empty:
        return 0.5
    threshold, _ = select_threshold_local(
        frame["label"].astype(int).tolist(),
        frame["prob"].astype(float).tolist(),
        metric=metric,
    )
    return float(threshold)


def assign_subject_motion_bins(frame: pd.DataFrame, score_col: str) -> pd.DataFrame:
    frame = frame.copy()
    bins = pd.Series(index=frame.index, dtype="object")
    ranks = pd.Series(index=frame.index, dtype="float")
    for _, indices in frame.groupby("subject_id").groups.items():
        idx = list(indices)
        rank = frame.loc[idx, score_col].rank(method="first", pct=True)
        labels = np.where(rank <= 1.0 / 3.0, "low_motion", np.where(rank <= 2.0 / 3.0, "mid_motion", "high_motion"))
        bins.loc[idx] = labels
        ranks.loc[idx] = rank
    frame["motion_score"] = frame[score_col]
    frame["motion_score_name"] = score_col
    frame["motion_bin"] = bins
    frame["motion_rank_within_subject"] = ranks
    return frame


def safe_metrics(labels: list[int], probs: list[float], preds: list[int]) -> dict[str, float]:
    if not labels:
        return {
            "n": 0,
            "stress_rate": float("nan"),
            "acc": float("nan"),
            "balanced_acc": float("nan"),
            "f1": float("nan"),
            "auroc": float("nan"),
            "positive_rate": float("nan"),
            "has_both_classes": 0.0,
        }
    y_true = np.asarray(labels, dtype=int)
    y_prob = np.asarray(probs, dtype=float)
    y_pred = np.asarray(preds, dtype=int)
    has_both = int(len(np.unique(y_true)) == 2)
    return {
        "n": int(len(y_true)),
        "stress_rate": float(np.mean(y_true)),
        "acc": float(accuracy_score(y_true, y_pred)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auroc": float(roc_auc_score(y_true, y_prob)) if has_both else float("nan"),
        "positive_rate": float(np.mean(y_pred)),
        "has_both_classes": float(has_both),
    }


def evaluate_with_threshold_local(y_true: list[int], y_prob: list[float], threshold: float) -> dict[str, float]:
    y_pred = [1 if prob >= threshold else 0 for prob in y_prob]
    metrics = safe_metrics(y_true, y_prob, y_pred)
    metrics["threshold"] = float(threshold)
    return metrics


def select_threshold_local(
    y_true: list[int],
    y_prob: list[float],
    metric: str = "balanced_acc",
) -> tuple[float, dict[str, float]]:
    if len(set(y_true)) < 2:
        return 0.5, evaluate_with_threshold_local(y_true, y_prob, threshold=0.5)

    candidates = sorted(set([0.0, 1.0] + [round(float(prob), 6) for prob in y_prob]))
    best_threshold = 0.5
    best_metrics = evaluate_with_threshold_local(y_true, y_prob, threshold=0.5)
    best_score = best_metrics[metric]
    for threshold in candidates:
        metrics = evaluate_with_threshold_local(y_true, y_prob, threshold=threshold)
        score = metrics[metric]
        if score > best_score + 1e-12:
            best_score = score
            best_threshold = threshold
            best_metrics = metrics
    return float(best_threshold), best_metrics


def summarize_by_bin(long_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_cols = ["dataset", "method", "motion_bin"]
    for keys, group in long_frame.groupby(group_cols, sort=False):
        dataset, method, motion_bin = keys
        metrics = safe_metrics(
            group["label"].astype(int).tolist(),
            group["prob"].astype(float).tolist(),
            group["pred"].astype(int).tolist(),
        )
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "motion_bin": motion_bin,
                **metrics,
                "acc_jerk_mean": float(group["acc_jerk"].mean()),
                "acc_energy_mean": float(group["acc_energy"].mean()),
                "watch_quality_mean": float(group["watch_quality"].mean()),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["method"] = pd.Categorical(out["method"], categories=METHOD_ORDER, ordered=True)
        out["motion_bin"] = pd.Categorical(out["motion_bin"], categories=BIN_ORDER, ordered=True)
        out = out.sort_values(["dataset", "method", "motion_bin"]).reset_index(drop=True)
    return out


def summarize_subject_bins(long_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_cols = ["dataset", "subject_id", "method", "motion_bin"]
    for keys, group in long_frame.groupby(group_cols, sort=False):
        dataset, subject_id, method, motion_bin = keys
        metrics = safe_metrics(
            group["label"].astype(int).tolist(),
            group["prob"].astype(float).tolist(),
            group["pred"].astype(int).tolist(),
        )
        rows.append(
            {
                "dataset": dataset,
                "subject_id": subject_id,
                "method": method,
                "motion_bin": motion_bin,
                **metrics,
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["method"] = pd.Categorical(out["method"], categories=METHOD_ORDER, ordered=True)
        out["motion_bin"] = pd.Categorical(out["motion_bin"], categories=BIN_ORDER, ordered=True)
        out = out.sort_values(["dataset", "subject_id", "method", "motion_bin"]).reset_index(drop=True)
    return out


def bootstrap_ci(values: np.ndarray, repeats: int = 5000, seed: int = 42) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(repeats, len(values)), replace=True).mean(axis=1)
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def summarize_motion_gap(subject_bin_summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for (dataset, subject_id), group in subject_bin_summary.groupby(["dataset", "subject_id"], sort=False):
        pivot = group.pivot_table(index="method", columns="motion_bin", values="balanced_acc", aggfunc="first")
        if not {"watch_only", "motion_aware"}.issubset(set(pivot.index)):
            continue
        if not {"low_motion", "high_motion"}.issubset(set(pivot.columns)):
            continue
        watch_gap = float(pivot.loc["watch_only", "low_motion"] - pivot.loc["watch_only", "high_motion"])
        motion_gap = float(pivot.loc["motion_aware", "low_motion"] - pivot.loc["motion_aware", "high_motion"])
        rows.append(
            {
                "dataset": dataset,
                "subject_id": subject_id,
                "watch_only_low_high_gap": watch_gap,
                "motion_aware_low_high_gap": motion_gap,
                "motion_sensitivity_gap_reduction": watch_gap - motion_gap,
            }
        )
    subject_gap = pd.DataFrame(rows)
    summary_rows: list[dict[str, object]] = []
    for dataset, group in subject_gap.groupby("dataset", sort=False):
        values = group["motion_sensitivity_gap_reduction"].to_numpy(dtype=float)
        ci_low, ci_high = bootstrap_ci(values)
        summary_rows.append(
            {
                "dataset": dataset,
                "subjects": int(len(values)),
                "watch_only_gap_mean": float(group["watch_only_low_high_gap"].mean()),
                "motion_aware_gap_mean": float(group["motion_aware_low_high_gap"].mean()),
                "gap_reduction_mean": float(np.nanmean(values)) if len(values) else float("nan"),
                "gap_reduction_ci_low": ci_low,
                "gap_reduction_ci_high": ci_high,
                "gap_reduction_positive_subjects": int(np.sum(values > 0)),
                "gap_reduction_negative_subjects": int(np.sum(values < 0)),
                "gap_reduction_zero_subjects": int(np.sum(values == 0)),
            }
        )
    return subject_gap, pd.DataFrame(summary_rows)


def add_global_gap_columns(gap_summary: pd.DataFrame, bin_summary: pd.DataFrame) -> pd.DataFrame:
    gap_summary = gap_summary.copy()
    for dataset in gap_summary["dataset"].astype(str).unique():
        ds_bins = bin_summary[bin_summary["dataset"].astype(str) == dataset]
        pivot = ds_bins.pivot_table(index="method", columns="motion_bin", values="balanced_acc", aggfunc="first")
        if not {"watch_only", "motion_aware"}.issubset(set(pivot.index)):
            continue
        if not {"low_motion", "high_motion"}.issubset(set(pivot.columns)):
            continue
        watch_gap = float(pivot.loc["watch_only", "low_motion"] - pivot.loc["watch_only", "high_motion"])
        motion_gap = float(pivot.loc["motion_aware", "low_motion"] - pivot.loc["motion_aware", "high_motion"])
        mask = gap_summary["dataset"].astype(str) == dataset
        gap_summary.loc[mask, "global_watch_only_low_high_gap"] = watch_gap
        gap_summary.loc[mask, "global_motion_aware_low_high_gap"] = motion_gap
        gap_summary.loc[mask, "global_gap_reduction"] = watch_gap - motion_gap
    return gap_summary


def plot_dataset(
    dataset: str,
    bin_summary: pd.DataFrame,
    subject_gap: pd.DataFrame,
    gap_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    ds_bins = bin_summary[bin_summary["dataset"] == dataset].copy()
    ds_gap = subject_gap[subject_gap["dataset"] == dataset].copy()
    ds_gap_summary = gap_summary[gap_summary["dataset"] == dataset].copy()

    colors = {"watch_only": "#4E79A7", "motion_aware": "#F28E2B"}
    labels = {"watch_only": "Watch-only", "motion_aware": "Motion-aware"}
    bin_labels = {"low_motion": "Low", "mid_motion": "Mid", "high_motion": "High"}

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=150)
    fig.suptitle(f"{dataset.upper()} Motion-Aware Mechanism", fontsize=15, fontweight="bold")

    x = np.arange(len(BIN_ORDER))
    width = 0.36
    for ax, metric, ylabel in [
        (axes[0, 0], "balanced_acc", "Balanced accuracy"),
        (axes[0, 1], "auroc", "AUROC"),
    ]:
        for offset, method in [(-width / 2, "watch_only"), (width / 2, "motion_aware")]:
            vals = []
            for motion_bin in BIN_ORDER:
                row = ds_bins[(ds_bins["method"].astype(str) == method) & (ds_bins["motion_bin"].astype(str) == motion_bin)]
                vals.append(float(row[metric].iloc[0]) if not row.empty else np.nan)
            ax.bar(x + offset, vals, width=width, color=colors[method], alpha=0.86, label=labels[method])
        ax.set_xticks(x, [bin_labels[item] for item in BIN_ORDER])
        ax.set_xlabel("ACC jerk bin")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0.0, 1.0)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False, fontsize=9)

    ax = axes[1, 0]
    if not ds_gap_summary.empty:
        watch_gap = float(ds_gap_summary["watch_only_gap_mean"].iloc[0])
        motion_gap = float(ds_gap_summary["motion_aware_gap_mean"].iloc[0])
        reduction = float(ds_gap_summary["gap_reduction_mean"].iloc[0])
        ci_low = float(ds_gap_summary["gap_reduction_ci_low"].iloc[0])
        ci_high = float(ds_gap_summary["gap_reduction_ci_high"].iloc[0])
        ax.bar([0, 1], [watch_gap, motion_gap], color=[colors["watch_only"], colors["motion_aware"]], alpha=0.86)
        ax.axhline(0, color="#444444", linewidth=1)
        ax.set_xticks([0, 1], ["Watch-only", "Motion-aware"])
        ax.set_ylabel("BA low-high gap")
        ax.set_title(f"Subject-mean gap reduction = {reduction:+.3f} [{ci_low:+.3f}, {ci_high:+.3f}]")
        ax.grid(axis="y", alpha=0.25)
    else:
        ax.text(0.5, 0.5, "No gap data", ha="center", va="center")
        ax.axis("off")

    ax = axes[1, 1]
    if not ds_gap.empty:
        ordered = ds_gap.sort_values("motion_sensitivity_gap_reduction")
        y = np.arange(len(ordered))
        vals = ordered["motion_sensitivity_gap_reduction"].to_numpy(dtype=float)
        ax.barh(y, vals, color=np.where(vals >= 0, "#59A14F", "#E15759"), alpha=0.86)
        ax.axvline(0, color="#444444", linewidth=1)
        ax.set_yticks(y, ordered["subject_id"].astype(str).tolist(), fontsize=7)
        ax.set_xlabel("Motion-sensitivity gap reduction")
        ax.set_title("Per-subject reduction")
        ax.grid(axis="x", alpha=0.25)
    else:
        ax.text(0.5, 0.5, "No subject gap data", ha="center", va="center")
        ax.axis("off")

    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=450, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def dataframe_to_markdown(frame: pd.DataFrame, floatfmt: str = ".4f") -> str:
    """Small markdown writer to avoid an optional pandas tabulate dependency."""
    if frame.empty:
        return ""
    headers = [str(col) for col in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in frame.iterrows():
        values: list[str] = []
        for value in row.tolist():
            if isinstance(value, (float, np.floating)):
                values.append(f"{float(value):{floatfmt}}" if np.isfinite(value) else "nan")
            elif isinstance(value, (int, np.integer)):
                values.append(str(int(value)))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_summary(
    output_path: Path,
    dataset: str,
    bin_summary: pd.DataFrame,
    gap_summary: pd.DataFrame,
) -> None:
    lines = [f"# Motion-Aware Mechanism Summary: {dataset}", ""]
    ds_gap = gap_summary[gap_summary["dataset"] == dataset]
    if not ds_gap.empty:
        row = ds_gap.iloc[0]
        lines.extend(
            [
                f"- Subjects with gap estimate: {int(row['subjects'])}",
                f"- Watch-only low-high BA gap: {row['watch_only_gap_mean']:.4f}",
                f"- Motion-aware low-high BA gap: {row['motion_aware_gap_mean']:.4f}",
                (
                    "- Subject-mean motion-sensitivity gap reduction: "
                    f"{row['gap_reduction_mean']:.4f} "
                    f"[{row['gap_reduction_ci_low']:.4f}, {row['gap_reduction_ci_high']:.4f}]"
                ),
                (
                    "- Positive / negative / zero subjects: "
                    f"{int(row['gap_reduction_positive_subjects'])} / "
                    f"{int(row['gap_reduction_negative_subjects'])} / "
                    f"{int(row['gap_reduction_zero_subjects'])}"
                ),
                "",
            ]
        )
        if "global_gap_reduction" in row.index:
            lines.extend(
                [
                    f"- Global watch-only low-high BA gap: {row['global_watch_only_low_high_gap']:.4f}",
                    f"- Global motion-aware low-high BA gap: {row['global_motion_aware_low_high_gap']:.4f}",
                    f"- Global motion-sensitivity gap reduction: {row['global_gap_reduction']:.4f}",
                    "",
                ]
            )
    lines.append("## Bin Summary")
    lines.append("")
    ds_bins = bin_summary[bin_summary["dataset"] == dataset].copy()
    if ds_bins.empty:
        lines.append("No bin summary available.")
    else:
        display_cols = [
            "method",
            "motion_bin",
            "n",
            "stress_rate",
            "balanced_acc",
            "auroc",
            "f1",
            "positive_rate",
            "acc_jerk_mean",
        ]
        lines.append(dataframe_to_markdown(ds_bins[display_cols], floatfmt=".4f"))
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze whether a motion-aware watch encoder reduces performance sensitivity "
            "between low- and high-motion windows."
        )
    )
    parser.add_argument("--dataset-kind", choices=["galaxy", "wesad"], required=True)
    parser.add_argument("--manifests-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", "--wesad-root", dest="dataset_root", type=Path, required=True)
    parser.add_argument("--watch-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--motion-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--watch-input-ablation",
        choices=[
            "none",
            "acc_only",
            "ppg_only",
            "simple_concat",
            "ppg_only_refine",
            "simple_concat_refine",
            "gated_fusion",
            "gated_fusion_refine",
        ],
        default="none",
        help="Architecture used by the baseline checkpoint. Use ppg_only for PPG/BVP-only ablation checkpoints.",
    )
    parser.add_argument(
        "--motion-input-ablation",
        choices=[
            "none",
            "acc_only",
            "ppg_only",
            "simple_concat",
            "ppg_only_refine",
            "simple_concat_refine",
            "gated_fusion",
            "gated_fusion_refine",
        ],
        default="none",
        help="Architecture used by the motion-aware checkpoint. Usually none.",
    )
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--cache-subjects", type=int, default=2)
    parser.add_argument("--threshold-metric", choices=["acc", "balanced_acc", "f1", "auroc"], default="balanced_acc")
    parser.add_argument("--motion-score", choices=["acc_jerk", "acc_energy"], default="acc_jerk")
    parser.add_argument("--calm-sessions", nargs="*", default=None)
    parser.add_argument("--stress-sessions", nargs="*", default=None)
    parser.add_argument("--watch-model-dim", type=int, default=192)
    parser.add_argument("--watch-transformer-layers", type=int, default=2)
    parser.add_argument("--watch-transformer-heads", type=int, default=4)
    parser.add_argument("--watch-fusion-hidden-dim", type=int, default=256)
    parser.add_argument("--watch-embed-dim", type=int, default=160)
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

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[pd.DataFrame] = []
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

        watch_state = load_state(find_checkpoint(args.watch_checkpoint_dir, subject))
        watch_model = build_model_from_state(
            watch_state,
            device=args.device,
            model_dim=args.watch_model_dim,
            transformer_layers=args.watch_transformer_layers,
            transformer_heads=args.watch_transformer_heads,
            fusion_hidden_dim=args.watch_fusion_hidden_dim,
            embed_dim=args.watch_embed_dim,
            input_ablation=args.watch_input_ablation,
        )
        watch_threshold = select_model_threshold(
            watch_model,
            val_loader,
            device=args.device,
            pin_memory=args.pin_memory,
            metric=args.threshold_metric,
        )
        watch_frame = collect_predictions(
            watch_model,
            test_loader,
            device=args.device,
            pin_memory=args.pin_memory,
            threshold=watch_threshold,
            dataset_kind=args.dataset_kind,
            fold_subject=subject,
            method="watch_only",
        )
        del watch_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        motion_state = load_state(find_checkpoint(args.motion_checkpoint_dir, subject))
        motion_model = build_model_from_state(
            motion_state,
            device=args.device,
            model_dim=args.watch_model_dim,
            transformer_layers=args.watch_transformer_layers,
            transformer_heads=args.watch_transformer_heads,
            fusion_hidden_dim=args.watch_fusion_hidden_dim,
            embed_dim=args.watch_embed_dim,
            input_ablation=args.motion_input_ablation,
        )
        motion_threshold = select_model_threshold(
            motion_model,
            val_loader,
            device=args.device,
            pin_memory=args.pin_memory,
            metric=args.threshold_metric,
        )
        motion_frame = collect_predictions(
            motion_model,
            test_loader,
            device=args.device,
            pin_memory=args.pin_memory,
            threshold=motion_threshold,
            dataset_kind=args.dataset_kind,
            fold_subject=subject,
            method="motion_aware",
        )
        del motion_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        fold_frame = pd.concat([watch_frame, motion_frame], ignore_index=True)
        fold_frame["watch_threshold"] = watch_threshold
        fold_frame["motion_threshold"] = motion_threshold
        all_rows.append(fold_frame)

    if not all_rows:
        raise ValueError("No prediction rows were collected.")

    long_frame = pd.concat(all_rows, ignore_index=True)
    # Assign bins once from unique windows, then merge back to both methods.
    window_cols = [
        "dataset",
        "fold_subject",
        "row_order",
        "subject_id",
        "session",
        "window_start_ms",
        "window_end_ms",
        "label",
        "acc_jerk",
        "acc_energy",
        "watch_quality",
    ]
    unique_windows = long_frame[window_cols].drop_duplicates().reset_index(drop=True)
    unique_windows = assign_subject_motion_bins(unique_windows, args.motion_score)
    long_frame = long_frame.merge(
        unique_windows[
            [
                "dataset",
                "fold_subject",
                "row_order",
                "subject_id",
                "window_start_ms",
                "window_end_ms",
                "motion_score",
                "motion_score_name",
                "motion_bin",
                "motion_rank_within_subject",
            ]
        ],
        on=["dataset", "fold_subject", "row_order", "subject_id", "window_start_ms", "window_end_ms"],
        how="left",
    )

    bin_summary = summarize_by_bin(long_frame)
    subject_bin_summary = summarize_subject_bins(long_frame)
    subject_gap, gap_summary = summarize_motion_gap(subject_bin_summary)
    gap_summary = add_global_gap_columns(gap_summary, bin_summary)

    prefix = args.dataset_kind
    windows_path = output_dir / f"{prefix}_motion_mechanism_windows.csv"
    bin_path = output_dir / f"{prefix}_motion_bin_summary.csv"
    subject_bin_path = output_dir / f"{prefix}_motion_subject_bin_summary.csv"
    subject_gap_path = output_dir / f"{prefix}_motion_subject_gap_reduction.csv"
    gap_path = output_dir / f"{prefix}_motion_gap_summary.csv"
    plot_path = output_dir / f"{prefix}_motion_aware_mechanism.png"
    summary_path = output_dir / f"{prefix}_motion_aware_mechanism_summary.md"

    long_frame.to_csv(windows_path, index=False)
    bin_summary.to_csv(bin_path, index=False)
    subject_bin_summary.to_csv(subject_bin_path, index=False)
    subject_gap.to_csv(subject_gap_path, index=False)
    gap_summary.to_csv(gap_path, index=False)
    plot_dataset(args.dataset_kind, bin_summary, subject_gap, gap_summary, plot_path)
    write_summary(summary_path, args.dataset_kind, bin_summary, gap_summary)

    print(f"Saved window predictions to {windows_path}")
    print(f"Saved bin summary to {bin_path}")
    print(f"Saved subject bin summary to {subject_bin_path}")
    print(f"Saved subject gap reduction to {subject_gap_path}")
    print(f"Saved gap summary to {gap_path}")
    print(f"Saved figure to {plot_path}")
    print(f"Saved SVG figure to {plot_path.with_suffix('.svg')}")
    print(f"Saved PDF figure to {plot_path.with_suffix('.pdf')}")
    print(f"Saved markdown summary to {summary_path}")


if __name__ == "__main__":
    main()
