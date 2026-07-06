from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch

from .analyze_adaptive_alpha_behavior import (
    DEFAULT_GALAXY_CALM_SESSIONS,
    DEFAULT_GALAXY_STRESS_SESSIONS,
    DEFAULT_WESAD_CALM_SESSIONS,
    DEFAULT_WESAD_STRESS_SESSIONS,
    build_dataset,
    build_loader,
    discover_manifests,
    forward_batch,
    list_from_batch,
    load_model,
    maybe_parse_sessions,
    motion_features,
    positive_prob,
    resolve_checkpoint,
)
from .reliability import cross_calibrated_trust, true_class_confidence


def margin_from_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.shape[1] != 2:
        raise ValueError("Teacher reliability analysis expects binary logits.")
    return logits[:, 1] - logits[:, 0]


def prediction_from_prob(prob: torch.Tensor) -> torch.Tensor:
    return (prob >= 0.5).long()


def true_conf_from_prob(prob: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    labels_float = labels.float()
    return torch.where(labels_float >= 0.5, prob, 1.0 - prob).clamp(0.0, 1.0)


def full_like(reference: torch.Tensor, value: float) -> torch.Tensor:
    return torch.full_like(reference, float(value), dtype=torch.float32)


def optional_prob(out: dict[str, torch.Tensor], key: str, reference: torch.Tensor) -> torch.Tensor:
    logits = out.get(key)
    if logits is None:
        return full_like(reference, float("nan"))
    return positive_prob(logits)


def optional_margin(out: dict[str, torch.Tensor], key: str, reference: torch.Tensor) -> torch.Tensor:
    logits = out.get(key)
    if logits is None:
        return full_like(reference, float("nan"))
    return margin_from_logits(logits).float()


def optional_vector_stat(
    out: dict[str, torch.Tensor],
    key: str,
    reference: torch.Tensor,
    reducer: str,
) -> torch.Tensor:
    value = out.get(key)
    if value is None:
        return full_like(reference, float("nan"))
    value = value.float()
    flat = value.reshape(value.shape[0], -1)
    if reducer == "mean":
        return flat.mean(dim=1)
    if reducer == "std":
        return flat.std(dim=1, unbiased=False)
    if reducer == "max":
        return flat.max(dim=1).values
    if reducer == "l2":
        return torch.linalg.vector_norm(flat, ord=2, dim=1)
    raise ValueError(f"Unsupported reducer: {reducer}")


def optional_alpha(out: dict[str, torch.Tensor], key: str, reference: torch.Tensor) -> torch.Tensor:
    value = out.get(key)
    if value is None:
        return full_like(reference, float("nan"))
    return value.reshape(-1).float()


def optional_embedding_shift(out: dict[str, torch.Tensor], reference: torch.Tensor) -> torch.Tensor:
    base = out.get("watch_embedding")
    corrected = out.get("deploy_correction_embedding")
    if base is None or corrected is None:
        return full_like(reference, float("nan"))
    return torch.linalg.vector_norm((corrected - base).reshape(base.shape[0], -1), ord=2, dim=1)


def metadata_list(batch: dict[str, Any], key: str, n: int, default: Any) -> list[Any]:
    if key not in batch:
        return [default for _ in range(n)]
    values = list_from_batch(batch[key])
    if len(values) == n:
        return values
    if len(values) == 1:
        return values * n
    return (values + [default for _ in range(n)])[:n]


def tensor_dict_to_arrays(values: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    return {key: value.detach().cpu().numpy() for key, value in values.items()}


def collect_fold_rows(
    subject: str,
    manifest_path: Path,
    checkpoint_path: Path,
    include_sessions: list[str],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    dataset = build_dataset(args.dataset_kind, manifest_path, args.dataset_root, include_sessions, args)
    if args.max_windows_per_fold is not None and args.max_windows_per_fold > 0:
        dataset.manifest = dataset.manifest.head(args.max_windows_per_fold).reset_index(drop=True)
    loader = build_loader(dataset, args)
    model = load_model(checkpoint_path, args)

    rows: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            watch_signal = batch["watch_signal"].to(args.device, non_blocking=args.pin_memory)
            quality = batch["watch_quality"].to(args.device, non_blocking=args.pin_memory)
            labels_long = batch["label"].to(args.device, non_blocking=args.pin_memory).long()
            labels_float = labels_long.float()
            out = forward_batch(model, batch, args)

            final_logits = out["logits"]
            base_logits = out.get("base_logits", final_logits)
            teacher_logits = out.get("teacher_logits")
            corrected_logits = out.get("deploy_corrected_logits")
            teacher_available = teacher_logits is not None
            corrected_available = corrected_logits is not None

            final_prob = positive_prob(final_logits)
            base_prob = positive_prob(base_logits)
            teacher_prob = positive_prob(teacher_logits) if teacher_logits is not None else full_like(final_prob, float("nan"))
            corrected_prob = (
                positive_prob(corrected_logits) if corrected_logits is not None else full_like(final_prob, float("nan"))
            )

            final_margin = margin_from_logits(final_logits)
            base_margin = margin_from_logits(base_logits)
            teacher_margin = (
                margin_from_logits(teacher_logits) if teacher_logits is not None else full_like(final_prob, float("nan"))
            )
            corrected_margin = (
                margin_from_logits(corrected_logits) if corrected_logits is not None else full_like(final_prob, float("nan"))
            )

            final_pred = prediction_from_prob(final_prob)
            base_pred = prediction_from_prob(base_prob)
            teacher_pred = prediction_from_prob(teacher_prob) if teacher_available else torch.zeros_like(labels_long)
            corrected_pred = prediction_from_prob(corrected_prob) if corrected_available else torch.zeros_like(labels_long)

            final_abs_error = torch.abs(final_prob - labels_float)
            base_abs_error = torch.abs(base_prob - labels_float)
            teacher_abs_error = torch.abs(teacher_prob - labels_float)
            corrected_abs_error = torch.abs(corrected_prob - labels_float)

            acc_energy, acc_jerk, ppg_std = motion_features(watch_signal)
            deploy_gate_mean = optional_vector_stat(out, "deploy_correction_gate", final_prob, "mean")
            deploy_gate_std = optional_vector_stat(out, "deploy_correction_gate", final_prob, "std")
            deploy_gate_max = optional_vector_stat(out, "deploy_correction_gate", final_prob, "max")
            deploy_delta_l2 = optional_vector_stat(out, "deploy_correction_delta", final_prob, "l2")
            deploy_alpha = optional_alpha(out, "deploy_correction_alpha", final_prob)
            deploy_alpha_unit = optional_alpha(out, "deploy_correction_alpha_unit", final_prob)
            correction_embedding_shift = optional_embedding_shift(out, final_prob)

            if teacher_logits is not None:
                teacher_true_conf = true_class_confidence(teacher_logits, labels_long)
                trust_final_teacher = cross_calibrated_trust(final_logits, teacher_logits, labels_long, quality)
                trust_base_teacher = cross_calibrated_trust(base_logits, teacher_logits, labels_long, quality)
            else:
                teacher_true_conf = full_like(final_prob, float("nan"))
                trust_final_teacher = full_like(final_prob, float("nan"))
                trust_base_teacher = full_like(final_prob, float("nan"))

            reliability_prob = out.get("reliability")
            if reliability_prob is None:
                reliability_prob_values = full_like(final_prob, float("nan"))
            else:
                reliability_prob_values = reliability_prob.reshape(-1).float()

            teacher_correct = (
                (teacher_pred == labels_long).float() if teacher_available else full_like(final_prob, float("nan"))
            )
            corrected_correct = (
                (corrected_pred == labels_long).float() if corrected_available else full_like(final_prob, float("nan"))
            )
            teacher_closer_than_base = (
                (teacher_abs_error < base_abs_error).float() if teacher_available else full_like(final_prob, float("nan"))
            )
            corrected_better_than_base = (
                (corrected_abs_error < base_abs_error).float()
                if corrected_available
                else full_like(final_prob, float("nan"))
            )
            teacher_base_disagree = (
                (teacher_pred != base_pred).float() if teacher_available else full_like(final_prob, float("nan"))
            )
            teacher_final_disagree = (
                (teacher_pred != final_pred).float() if teacher_available else full_like(final_prob, float("nan"))
            )

            transfer_signal = torch.where(
                torch.isfinite(deploy_alpha),
                deploy_alpha,
                deploy_gate_mean,
            )

            tensors = {
                "label": labels_float,
                "watch_quality": quality.reshape(-1).float(),
                "acc_energy": acc_energy.float(),
                "acc_jerk": acc_jerk.float(),
                "ppg_std": ppg_std.float(),
                "base_prob": base_prob.float(),
                "final_prob": final_prob.float(),
                "teacher_prob": teacher_prob.float(),
                "corrected_prob": corrected_prob.float(),
                "base_margin": base_margin.float(),
                "final_margin": final_margin.float(),
                "teacher_margin": teacher_margin.float(),
                "corrected_margin": corrected_margin.float(),
                "base_pred": base_pred.float(),
                "final_pred": final_pred.float(),
                "teacher_pred": teacher_pred.float() if teacher_available else full_like(final_prob, float("nan")),
                "corrected_pred": corrected_pred.float() if corrected_available else full_like(final_prob, float("nan")),
                "base_correct": (base_pred == labels_long).float(),
                "final_correct": (final_pred == labels_long).float(),
                "teacher_correct": teacher_correct,
                "corrected_correct": corrected_correct,
                "base_true_conf": true_conf_from_prob(base_prob, labels_long),
                "final_true_conf": true_conf_from_prob(final_prob, labels_long),
                "teacher_true_conf": teacher_true_conf.float(),
                "base_abs_error": base_abs_error.float(),
                "final_abs_error": final_abs_error.float(),
                "teacher_abs_error": teacher_abs_error.float(),
                "corrected_abs_error": corrected_abs_error.float(),
                "final_gain_abs_error": (base_abs_error - final_abs_error).float(),
                "teacher_gain_abs_error": (base_abs_error - teacher_abs_error).float(),
                "corrected_gain_abs_error": (base_abs_error - corrected_abs_error).float(),
                "teacher_closer_than_base": teacher_closer_than_base,
                "final_better_than_base": (final_abs_error < base_abs_error).float(),
                "corrected_better_than_base": corrected_better_than_base,
                "teacher_base_disagree": teacher_base_disagree,
                "teacher_final_disagree": teacher_final_disagree,
                "final_base_disagree": (final_pred != base_pred).float(),
                "teacher_base_prob_gap": torch.abs(teacher_prob - base_prob).float(),
                "teacher_final_prob_gap": torch.abs(teacher_prob - final_prob).float(),
                "final_base_prob_shift": torch.abs(final_prob - base_prob).float(),
                "corrected_base_prob_shift": torch.abs(corrected_prob - base_prob).float(),
                "teacher_margin_abs": torch.abs(teacher_margin).float(),
                "base_margin_abs": torch.abs(base_margin).float(),
                "final_margin_abs": torch.abs(final_margin).float(),
                "teacher_base_margin_gap": torch.abs(teacher_margin - base_margin).float(),
                "teacher_final_margin_gap": torch.abs(teacher_margin - final_margin).float(),
                "final_base_margin_shift": torch.abs(final_margin - base_margin).float(),
                "deploy_gate_mean": deploy_gate_mean.float(),
                "deploy_gate_std": deploy_gate_std.float(),
                "deploy_gate_max": deploy_gate_max.float(),
                "deploy_delta_l2": deploy_delta_l2.float(),
                "deploy_alpha": deploy_alpha.float(),
                "deploy_alpha_unit": deploy_alpha_unit.float(),
                "transfer_signal": transfer_signal.float(),
                "correction_embedding_shift": correction_embedding_shift.float(),
                "trust_final_teacher": trust_final_teacher.float(),
                "trust_base_teacher": trust_base_teacher.float(),
                "cross_confidence_weight_min05": (0.05 + 0.95 * trust_final_teacher).float(),
                "reliability_head_prob": reliability_prob_values.float(),
            }
            arrays = tensor_dict_to_arrays(tensors)

            n = int(labels_long.shape[0])
            subjects = metadata_list(batch, "subject_id", n, "")
            sessions = metadata_list(batch, "session", n, "")
            groups = metadata_list(batch, "group_name", n, "")
            starts = metadata_list(batch, "window_start_ms", n, None)
            ends = metadata_list(batch, "window_end_ms", n, None)

            for idx in range(n):
                rows.append(
                    {
                        "dataset": args.dataset_kind,
                        "fold_subject": subject,
                        "checkpoint": str(checkpoint_path),
                        "manifest": str(manifest_path),
                        "subject_id": subjects[idx],
                        "session": sessions[idx],
                        "group_name": groups[idx],
                        "window_start_ms": starts[idx],
                        "window_end_ms": ends[idx],
                        **{key: float(value[idx]) for key, value in arrays.items()},
                    }
                )
    return rows


def fmt(value: object, digits: int = 4) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "nan"
    if not math.isfinite(x):
        return "nan"
    return f"{x:.{digits}f}"


def label_bool(values: pd.Series, positive: str, negative: str) -> pd.Series:
    def assign(value: object) -> str:
        try:
            x = float(value)
        except (TypeError, ValueError):
            return "missing"
        if not math.isfinite(x):
            return "missing"
        return positive if x >= 0.5 else negative

    return pd.to_numeric(values, errors="coerce").map(assign)


def add_quantile_bin(frame: pd.DataFrame, source_col: str, target_col: str, labels: Sequence[str]) -> None:
    values = pd.to_numeric(frame[source_col], errors="coerce")
    valid = values.dropna()
    if valid.nunique() < 2:
        frame[target_col] = "all"
        return
    q1 = float(valid.quantile(1.0 / len(labels)))
    q2 = float(valid.quantile(2.0 / len(labels))) if len(labels) >= 3 else q1

    def assign(value: object) -> str:
        try:
            x = float(value)
        except (TypeError, ValueError):
            return "missing"
        if not math.isfinite(x):
            return "missing"
        if len(labels) == 2:
            return labels[0] if x <= q1 else labels[1]
        if x <= q1:
            return labels[0]
        if x <= q2:
            return labels[1]
        return labels[2]

    frame[target_col] = values.map(assign)


def add_mechanism_labels(windows: pd.DataFrame) -> pd.DataFrame:
    out = windows.copy()
    out["teacher_correct_label"] = label_bool(out["teacher_correct"], "teacher_correct", "teacher_wrong")
    out["teacher_help_label"] = label_bool(out["teacher_closer_than_base"], "teacher_closer", "teacher_not_closer")
    out["teacher_base_agreement_label"] = label_bool(
        out["teacher_base_disagree"],
        "teacher_student_disagree",
        "teacher_student_agree",
    )
    out["final_help_label"] = label_bool(out["final_better_than_base"], "final_better", "final_not_better")
    add_quantile_bin(out, "acc_jerk", "motion_jerk_bin", ["low_motion", "mid_motion", "high_motion"])
    add_quantile_bin(out, "acc_energy", "motion_energy_bin", ["low_energy", "mid_energy", "high_energy"])
    add_quantile_bin(out, "watch_quality", "quality_bin", ["low_quality", "mid_quality", "high_quality"])
    return out


def finite_mean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return float("nan")
    return float(values.mean())


def summarize_group(windows: pd.DataFrame, group_col: str) -> pd.DataFrame:
    metrics = {
        "n": ("label", "size"),
        "teacher_correct_rate": ("teacher_correct", "mean"),
        "teacher_closer_rate": ("teacher_closer_than_base", "mean"),
        "teacher_student_disagree_rate": ("teacher_base_disagree", "mean"),
        "final_better_rate": ("final_better_than_base", "mean"),
        "base_abs_error_mean": ("base_abs_error", "mean"),
        "final_abs_error_mean": ("final_abs_error", "mean"),
        "final_gain_abs_error_mean": ("final_gain_abs_error", "mean"),
        "teacher_abs_error_mean": ("teacher_abs_error", "mean"),
        "trust_final_teacher_mean": ("trust_final_teacher", "mean"),
        "cross_confidence_weight_min05_mean": ("cross_confidence_weight_min05", "mean"),
        "deploy_gate_mean": ("deploy_gate_mean", "mean"),
        "deploy_alpha_mean": ("deploy_alpha", finite_mean),
        "transfer_signal_mean": ("transfer_signal", finite_mean),
        "final_base_prob_shift_mean": ("final_base_prob_shift", "mean"),
        "final_base_margin_shift_mean": ("final_base_margin_shift", "mean"),
        "correction_embedding_shift_mean": ("correction_embedding_shift", finite_mean),
        "watch_quality_mean": ("watch_quality", "mean"),
        "acc_energy_mean": ("acc_energy", "mean"),
        "acc_jerk_mean": ("acc_jerk", "mean"),
    }
    summary = windows.groupby(["dataset", group_col], as_index=False).agg(**metrics)
    summary.insert(1, "grouping", group_col)
    summary = summary.rename(columns={group_col: "group"})
    return summary


def summarize_windows(windows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    windows = add_mechanism_labels(windows)
    subject_summary = (
        windows.groupby(["dataset", "fold_subject"], as_index=False)
        .agg(
            n_windows=("label", "size"),
            teacher_correct_rate=("teacher_correct", "mean"),
            teacher_closer_rate=("teacher_closer_than_base", "mean"),
            teacher_student_disagree_rate=("teacher_base_disagree", "mean"),
            final_better_rate=("final_better_than_base", "mean"),
            base_abs_error_mean=("base_abs_error", "mean"),
            final_abs_error_mean=("final_abs_error", "mean"),
            final_gain_abs_error_mean=("final_gain_abs_error", "mean"),
            teacher_abs_error_mean=("teacher_abs_error", "mean"),
            trust_final_teacher_mean=("trust_final_teacher", "mean"),
            cross_confidence_weight_min05_mean=("cross_confidence_weight_min05", "mean"),
            deploy_gate_mean=("deploy_gate_mean", "mean"),
            deploy_alpha_mean=("deploy_alpha", finite_mean),
            transfer_signal_mean=("transfer_signal", finite_mean),
            final_base_prob_shift_mean=("final_base_prob_shift", "mean"),
            final_base_margin_shift_mean=("final_base_margin_shift", "mean"),
            watch_quality_mean=("watch_quality", "mean"),
            acc_energy_mean=("acc_energy", "mean"),
            acc_jerk_mean=("acc_jerk", "mean"),
        )
        .sort_values(["dataset", "final_gain_abs_error_mean"], ascending=[True, False])
    )

    group_summaries = pd.concat(
        [
            summarize_group(windows, "teacher_correct_label"),
            summarize_group(windows, "teacher_help_label"),
            summarize_group(windows, "teacher_base_agreement_label"),
            summarize_group(windows, "motion_jerk_bin"),
            summarize_group(windows, "motion_energy_bin"),
            summarize_group(windows, "quality_bin"),
            summarize_group(windows, "final_help_label"),
        ],
        ignore_index=True,
    )

    correlations = summarize_correlations(windows)
    return subject_summary, group_summaries, correlations


def rank_corr(frame: pd.DataFrame, x_col: str, y_col: str) -> tuple[int, float]:
    x = pd.to_numeric(frame[x_col], errors="coerce")
    y = pd.to_numeric(frame[y_col], errors="coerce")
    valid = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(valid) < 3 or valid["x"].nunique() < 2 or valid["y"].nunique() < 2:
        return int(len(valid)), float("nan")
    corr = valid["x"].rank(method="average").corr(valid["y"].rank(method="average"), method="pearson")
    return int(len(valid)), float(corr)


def summarize_correlations(windows: pd.DataFrame) -> pd.DataFrame:
    pairs = [
        ("trust_final_teacher", "teacher_correct"),
        ("trust_final_teacher", "teacher_closer_than_base"),
        ("trust_final_teacher", "final_gain_abs_error"),
        ("cross_confidence_weight_min05", "final_gain_abs_error"),
        ("deploy_gate_mean", "teacher_correct"),
        ("deploy_gate_mean", "teacher_closer_than_base"),
        ("deploy_gate_mean", "teacher_base_disagree"),
        ("deploy_gate_mean", "final_gain_abs_error"),
        ("deploy_gate_mean", "acc_jerk"),
        ("deploy_gate_mean", "watch_quality"),
        ("transfer_signal", "teacher_correct"),
        ("transfer_signal", "teacher_closer_than_base"),
        ("transfer_signal", "teacher_base_disagree"),
        ("transfer_signal", "final_gain_abs_error"),
        ("final_base_prob_shift", "acc_jerk"),
        ("final_base_prob_shift", "teacher_base_disagree"),
        ("final_base_prob_shift", "trust_final_teacher"),
        ("acc_jerk", "final_gain_abs_error"),
        ("watch_quality", "final_gain_abs_error"),
    ]
    rows: list[dict[str, Any]] = []
    for x_col, y_col in pairs:
        if x_col not in windows.columns or y_col not in windows.columns:
            continue
        n, rho = rank_corr(windows, x_col, y_col)
        rows.append(
            {
                "dataset": str(windows["dataset"].iloc[0]) if not windows.empty else "",
                "x": x_col,
                "y": y_col,
                "n": n,
                "spearman_rho": rho,
            }
        )
    return pd.DataFrame(rows)


def write_markdown_report(
    output_path: Path,
    windows: pd.DataFrame,
    subject_summary: pd.DataFrame,
    group_summary: pd.DataFrame,
    correlations: pd.DataFrame,
) -> None:
    lines = ["# Teacher Reliability Mechanism Analysis", ""]
    lines.append(f"- Dataset: {windows['dataset'].iloc[0] if not windows.empty else 'unknown'}")
    lines.append(f"- Windows: {len(windows)}")
    lines.append(f"- Fold subjects: {windows['fold_subject'].nunique()}")
    lines.append(f"- Teacher correct rate: {fmt(windows['teacher_correct'].mean())}")
    lines.append(f"- Teacher closer-than-base rate: {fmt(windows['teacher_closer_than_base'].mean())}")
    lines.append(f"- Teacher/student disagreement rate: {fmt(windows['teacher_base_disagree'].mean())}")
    lines.append(f"- Final better-than-base rate: {fmt(windows['final_better_than_base'].mean())}")
    lines.append(f"- Mean final gain in absolute error: {fmt(windows['final_gain_abs_error'].mean())}")
    lines.append(f"- Mean cross-confidence trust: {fmt(windows['trust_final_teacher'].mean())}")
    lines.append(f"- Mean deploy gate: {fmt(windows['deploy_gate_mean'].mean())}")
    lines.append(f"- Mean adaptive alpha: {fmt(windows['deploy_alpha'].mean())}")
    lines.append("")

    lines.append("## Key Group Summaries")
    lines.append("")
    lines.append(
        "| Grouping | Group | n | teacher correct | teacher closer | disagree | final better | "
        "final gain | trust | gate | alpha | motion jerk | quality |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in group_summary.itertuples(index=False):
        lines.append(
            f"| {row.grouping} | {row.group} | {int(row.n)} | {fmt(row.teacher_correct_rate)} | "
            f"{fmt(row.teacher_closer_rate)} | {fmt(row.teacher_student_disagree_rate)} | "
            f"{fmt(row.final_better_rate)} | {fmt(row.final_gain_abs_error_mean)} | "
            f"{fmt(row.trust_final_teacher_mean)} | {fmt(row.deploy_gate_mean)} | "
            f"{fmt(row.deploy_alpha_mean)} | {fmt(row.acc_jerk_mean)} | {fmt(row.watch_quality_mean)} |"
        )

    lines.append("")
    lines.append("## Subject-Level Summary")
    lines.append("")
    lines.append("| Subject | n | teacher correct | disagree | final gain | trust | gate | motion jerk | quality |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in subject_summary.itertuples(index=False):
        lines.append(
            f"| {row.fold_subject} | {int(row.n_windows)} | {fmt(row.teacher_correct_rate)} | "
            f"{fmt(row.teacher_student_disagree_rate)} | {fmt(row.final_gain_abs_error_mean)} | "
            f"{fmt(row.trust_final_teacher_mean)} | {fmt(row.deploy_gate_mean)} | "
            f"{fmt(row.acc_jerk_mean)} | {fmt(row.watch_quality_mean)} |"
        )

    lines.append("")
    lines.append("## Spearman Correlations")
    lines.append("")
    lines.append("| x | y | n | rho |")
    lines.append("|---|---|---:|---:|")
    for row in correlations.itertuples(index=False):
        lines.append(f"| {row.x} | {row.y} | {int(row.n)} | {fmt(row.spearman_rho)} |")

    lines.append("")
    lines.append("## Reading Guide")
    lines.append("")
    lines.append(
        "Use this report as mechanism evidence, not as a main performance table. The useful story is strongest when "
        "trust/gate/transfer signals decrease on unreliable or disagreeing teacher windows, and when final gain is "
        "positive in the windows or subjects where the teacher carries usable information."
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _boxplot(ax: Any, frame: pd.DataFrame, group_col: str, value_col: str, title: str, ylabel: str) -> None:
    groups = []
    labels = []
    for label, group in frame.groupby(group_col, sort=False):
        values = pd.to_numeric(group[value_col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if values.empty:
            continue
        groups.append(values.to_numpy())
        labels.append(str(label))
    if not groups:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return
    box = ax.boxplot(groups, labels=labels, patch_artist=True, showfliers=False)
    colors = ["#4E79A7", "#F28E2B", "#59A14F", "#E15759"]
    for patch, color in zip(box["boxes"], colors * 4):
        patch.set_facecolor(color)
        patch.set_alpha(0.68)
    ax.set_title(title)
    ax.set_ylabel(ylabel)


def try_plot(windows: pd.DataFrame, subject_summary: pd.DataFrame, output_prefix: Path, title: str) -> bool:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:
        print(f"matplotlib_unavailable={exc!r}; skipping reliability figure")
        return False

    plot_frame = add_mechanism_labels(windows)
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.26,
            "legend.frameon": False,
            "savefig.dpi": 450,
            "savefig.bbox": "tight",
        }
    )
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.2), dpi=180)

    _boxplot(
        axes[0, 0],
        plot_frame,
        "teacher_correct_label",
        "trust_final_teacher",
        "Cross-confidence by teacher correctness",
        "Trust",
    )
    _boxplot(
        axes[0, 1],
        plot_frame,
        "teacher_base_agreement_label",
        "transfer_signal",
        "Transfer signal by teacher/student agreement",
        "Gate or alpha",
    )
    _boxplot(
        axes[0, 2],
        plot_frame,
        "motion_jerk_bin",
        "final_base_prob_shift",
        "Prediction shift across motion bins",
        "|final - base prob|",
    )

    ax = axes[1, 0]
    if not subject_summary.empty:
        ax.axvline(0.0, color="#475569", linewidth=1.0)
        subj = subject_summary.sort_values("final_gain_abs_error_mean")
        colors = np.where(subj["final_gain_abs_error_mean"] >= 0, "#4E79A7", "#E15759")
        ax.barh(subj["fold_subject"], subj["final_gain_abs_error_mean"], color=colors, alpha=0.85)
    ax.set_title("Per-subject final gain over base")
    ax.set_xlabel("Base abs. error - final abs. error")

    ax = axes[1, 1]
    scatter_frame = plot_frame.dropna(subset=["trust_final_teacher", "final_gain_abs_error"])
    if len(scatter_frame) > 3000:
        scatter_frame = scatter_frame.sample(3000, random_state=42)
    ax.axhline(0.0, color="#475569", linewidth=1.0)
    ax.scatter(
        scatter_frame["trust_final_teacher"],
        scatter_frame["final_gain_abs_error"],
        s=12,
        alpha=0.42,
        color="#4E79A7",
        linewidths=0,
    )
    ax.set_title("Final gain vs teacher trust")
    ax.set_xlabel("Cross-confidence trust")
    ax.set_ylabel("Final gain")

    ax = axes[1, 2]
    labels = ["teacher wrong", "teacher correct"]
    wrong = plot_frame[plot_frame["teacher_correct"] < 0.5]
    correct = plot_frame[plot_frame["teacher_correct"] >= 0.5]
    rates = [
        float(wrong["final_better_than_base"].mean()) if len(wrong) else float("nan"),
        float(correct["final_better_than_base"].mean()) if len(correct) else float("nan"),
    ]
    ax.bar(labels, rates, color=["#E15759", "#59A14F"], alpha=0.82)
    ax.set_ylim(0, 1)
    ax.set_title("Final better-than-base rate")
    ax.set_ylabel("Rate")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    png_path = output_prefix.with_suffix(".png")
    svg_path = output_prefix.with_suffix(".svg")
    pdf_path = output_prefix.with_suffix(".pdf")
    fig.savefig(png_path, dpi=450, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export teacher reliability, gate, disagreement, motion, and quality mechanism analyses."
    )
    parser.add_argument("--dataset-kind", type=str, required=True, choices=["galaxy", "wesad"])
    parser.add_argument("--manifests-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", "--wesad-root", dest="dataset_root", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--cache-subjects", type=int, default=4)
    parser.add_argument("--wavelet", type=str, default="db4")
    parser.add_argument("--wavelet-level", type=int, default=4)
    parser.add_argument("--baseline-reference", action="store_true")
    parser.add_argument("--correction-scale-init", type=float, default=0.05)
    parser.add_argument("--correction-alpha-init-bias", type=float, default=-3.0)
    parser.add_argument("--correction-alpha-max", type=float, default=1.0)
    parser.add_argument("--calm-sessions", nargs="*", default=None)
    parser.add_argument("--stress-sessions", nargs="*", default=None)
    parser.add_argument("--max-windows-per-fold", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dataset_kind == "galaxy":
        calm_sessions = maybe_parse_sessions(args.calm_sessions, DEFAULT_GALAXY_CALM_SESSIONS)
        stress_sessions = maybe_parse_sessions(args.stress_sessions, DEFAULT_GALAXY_STRESS_SESSIONS)
    else:
        calm_sessions = maybe_parse_sessions(args.calm_sessions, DEFAULT_WESAD_CALM_SESSIONS)
        stress_sessions = maybe_parse_sessions(args.stress_sessions, DEFAULT_WESAD_STRESS_SESSIONS)
    include_sessions = calm_sessions + stress_sessions

    manifests = discover_manifests(args.manifests_dir, args.dataset_kind, args.subjects)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    for subject, manifest_path in manifests:
        checkpoint_path = resolve_checkpoint(args.checkpoint_dir, subject)
        print(f"[{subject}] manifest={manifest_path} checkpoint={checkpoint_path}")
        rows = collect_fold_rows(subject, manifest_path, checkpoint_path, include_sessions, args)
        print(f"[{subject}] windows={len(rows)}")
        all_rows.extend(rows)

    windows = pd.DataFrame(all_rows)
    if windows.empty:
        raise ValueError("No windows were collected.")
    windows = add_mechanism_labels(windows)
    subject_summary, group_summary, correlations = summarize_windows(windows)

    prefix = args.dataset_kind
    windows_path = args.output_dir / f"{prefix}_teacher_reliability_windows.csv"
    subject_path = args.output_dir / f"{prefix}_teacher_reliability_subject_summary.csv"
    group_path = args.output_dir / f"{prefix}_teacher_reliability_group_summary.csv"
    corr_path = args.output_dir / f"{prefix}_teacher_reliability_correlation_summary.csv"
    report_path = args.output_dir / f"{prefix}_teacher_reliability_mechanism_report.md"
    figure_prefix = args.output_dir / f"{prefix}_teacher_reliability_mechanism"

    windows.to_csv(windows_path, index=False)
    subject_summary.to_csv(subject_path, index=False)
    group_summary.to_csv(group_path, index=False)
    correlations.to_csv(corr_path, index=False)
    write_markdown_report(report_path, windows, subject_summary, group_summary, correlations)
    plotted = try_plot(windows, subject_summary, figure_prefix, title=f"{args.dataset_kind.upper()} Teacher Reliability Mechanism")

    print()
    print(
        "summary="
        f"windows:{len(windows)} "
        f"subjects:{windows['fold_subject'].nunique()} "
        f"teacher_correct:{windows['teacher_correct'].mean():.4f} "
        f"teacher_closer:{windows['teacher_closer_than_base'].mean():.4f} "
        f"disagree:{windows['teacher_base_disagree'].mean():.4f} "
        f"final_gain:{windows['final_gain_abs_error'].mean():.4f} "
        f"trust:{windows['trust_final_teacher'].mean():.4f} "
        f"gate:{windows['deploy_gate_mean'].mean():.4f}"
    )
    print(f"Saved windows to {windows_path}")
    print(f"Saved subject summary to {subject_path}")
    print(f"Saved group summary to {group_path}")
    print(f"Saved correlations to {corr_path}")
    print(f"Saved report to {report_path}")
    if plotted:
        print(
            "Saved figures to "
            f"{figure_prefix.with_suffix('.png')}, "
            f"{figure_prefix.with_suffix('.svg')}, and "
            f"{figure_prefix.with_suffix('.pdf')}"
        )


if __name__ == "__main__":
    main()
