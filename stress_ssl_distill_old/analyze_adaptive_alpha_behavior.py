from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .measure_deployment_efficiency import build_model_from_state, normalize_state_dict


DEFAULT_WAVELET_BANDS = ("A4", "D4", "D2", "D1")
DEFAULT_GALAXY_CALM_SESSIONS = [
    "baseline",
    "meditation-1",
    "meditation-2",
    "rest-1",
    "rest-2",
    "rest-3",
    "rest-4",
    "rest-5",
]
DEFAULT_GALAXY_STRESS_SESSIONS = ["tsst-prep"]
DEFAULT_WESAD_CALM_SESSIONS = ["baseline"]
DEFAULT_WESAD_STRESS_SESSIONS = ["stress"]


def maybe_parse_sessions(values: Sequence[str] | None, fallback: Iterable[str]) -> list[str]:
    if values is None or len(values) == 0:
        return list(fallback)
    return [str(item) for item in values]


def discover_manifests(manifests_dir: Path, dataset_kind: str, subjects: Sequence[str] | None) -> list[tuple[str, Path]]:
    requested = {str(subject).strip() for subject in subjects or [] if str(subject).strip()}
    prefix = "galaxy" if dataset_kind == "galaxy" else "wesad"
    manifests: list[tuple[str, Path]] = []
    for path in sorted(manifests_dir.glob(f"{prefix}_*_loso_val.csv")):
        subject = path.stem.replace(f"{prefix}_", "").replace("_loso_val", "")
        if requested and subject not in requested:
            continue
        manifests.append((subject, path))
    if not manifests:
        raise ValueError(f"No {dataset_kind} LOSO manifests found in {manifests_dir}")
    return manifests


def resolve_checkpoint(checkpoint_dir: Path, subject: str) -> Path:
    roots = [checkpoint_dir]
    if (checkpoint_dir / "checkpoints").exists():
        roots.insert(0, checkpoint_dir / "checkpoints")
    candidates: list[Path] = []
    for root in roots:
        candidates.extend(
            [
                root / f"{subject}_deploy_watch.pt",
                root / f"{subject}.pt",
                root / f"{subject}_watch_only.pt",
            ]
        )
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Could not find checkpoint for {subject} under {checkpoint_dir}. "
        f"Tried: {', '.join(str(path) for path in candidates[:6])}"
    )


def build_dataset(
    dataset_kind: str,
    manifest_path: Path,
    dataset_root: Path,
    include_sessions: list[str],
    args: argparse.Namespace,
) -> Any:
    if dataset_kind == "galaxy":
        from .galaxy_dataset import GalaxyPrivilegedWindowDataset

        return GalaxyPrivilegedWindowDataset(
            manifest_csv=manifest_path,
            split="test",
            dataset_root=dataset_root,
            include_sessions=include_sessions,
            cache_tables=True,
            wavelet=args.wavelet,
            wavelet_level=args.wavelet_level,
            wavelet_bands=DEFAULT_WAVELET_BANDS,
            baseline_reference=args.baseline_reference,
        )

    from .wesad_dataset import WESADPrivilegedWindowDataset

    return WESADPrivilegedWindowDataset(
        manifest_csv=manifest_path,
        split="test",
        wesad_root=dataset_root,
        include_sessions=include_sessions,
        cache_subjects=args.cache_subjects,
        wavelet=args.wavelet,
        wavelet_level=args.wavelet_level,
        wavelet_bands=DEFAULT_WAVELET_BANDS,
        baseline_reference=args.baseline_reference,
    )


def build_loader(dataset: Any, args: argparse.Namespace) -> DataLoader:
    kwargs: dict[str, Any] = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
    }
    if args.num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **kwargs)


def list_from_batch(value: Any) -> list[Any]:
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def positive_prob(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=1)[:, 1]


def sigmoid_margin(logits: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(logits[:, 1] - logits[:, 0])


def motion_features(watch_signal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ppg = watch_signal[:, :1]
    acc = watch_signal[:, 1:]
    acc_energy = torch.sqrt(torch.mean(torch.square(acc), dim=(1, 2)).clamp_min(1e-12))
    if acc.shape[-1] > 1:
        acc_diff = torch.diff(acc, dim=-1)
        acc_jerk = torch.sqrt(torch.mean(torch.square(acc_diff), dim=(1, 2)).clamp_min(1e-12))
    else:
        acc_jerk = torch.zeros_like(acc_energy)
    ppg_std = torch.std(ppg, dim=(1, 2), unbiased=False)
    return acc_energy, acc_jerk, ppg_std


def model_args_for_loader(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        dataset_kind=args.dataset_kind,
        device=args.device,
        correction_scale_init=args.correction_scale_init,
        correction_alpha_init_bias=args.correction_alpha_init_bias,
        correction_alpha_max=args.correction_alpha_max,
    )


def load_model(checkpoint_path: Path, args: argparse.Namespace) -> torch.nn.Module:
    state = normalize_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model = build_model_from_state(state, model_args_for_loader(args))
    model.eval()
    return model


def forward_batch(model: torch.nn.Module, batch: dict[str, Any], args: argparse.Namespace) -> dict[str, torch.Tensor]:
    watch_signal = batch["watch_signal"].to(args.device, non_blocking=args.pin_memory)
    wavelet = batch["wavelet_features"].to(args.device, non_blocking=args.pin_memory)
    quality = batch["watch_quality"].to(args.device, non_blocking=args.pin_memory)
    baseline_kwargs: dict[str, torch.Tensor] = {}
    if args.baseline_reference:
        baseline_kwargs = {
            "baseline_watch_signal": batch["baseline_watch_signal"].to(args.device, non_blocking=args.pin_memory),
            "baseline_wavelet_features": batch["baseline_wavelet_features"].to(args.device, non_blocking=args.pin_memory),
            "baseline_quality": batch["baseline_watch_quality"].to(args.device, non_blocking=args.pin_memory),
        }

    if args.dataset_kind == "galaxy":
        e4_signal = batch["e4_signal"].to(args.device, non_blocking=args.pin_memory)
        return model(
            watch_signal,
            wavelet,
            quality,
            e4_signal=e4_signal,
            **baseline_kwargs,
            return_aux=True,
        )

    privileged_signal = batch["privileged_signal"].to(args.device, non_blocking=args.pin_memory)
    return model(
        watch_signal,
        wavelet,
        quality,
        privileged_signal=privileged_signal,
        **baseline_kwargs,
    )


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
            labels = batch["label"].to(args.device, non_blocking=args.pin_memory).float()
            out = forward_batch(model, batch, args)

            final_prob = positive_prob(out["logits"])
            base_prob = positive_prob(out["base_logits"]) if "base_logits" in out else final_prob
            corrected_logits = out.get("deploy_corrected_logits")
            corrected_prob = positive_prob(corrected_logits) if corrected_logits is not None else torch.full_like(final_prob, float("nan"))
            teacher_logits = out.get("teacher_logits")
            teacher_prob = positive_prob(teacher_logits) if teacher_logits is not None else torch.full_like(final_prob, float("nan"))
            alpha = out.get("deploy_correction_alpha")
            if alpha is None:
                alpha_values = torch.zeros_like(final_prob)
            else:
                alpha_values = alpha.reshape(-1).float()
            alpha_unit = out.get("deploy_correction_alpha_unit")
            if alpha_unit is None:
                alpha_unit_values = torch.full_like(final_prob, float("nan"))
            else:
                alpha_unit_values = alpha_unit.reshape(-1).float()

            acc_energy, acc_jerk, ppg_std = motion_features(watch_signal)
            base_error = torch.abs(base_prob - labels)
            corrected_error = torch.abs(corrected_prob - labels)
            final_error = torch.abs(final_prob - labels)
            teacher_error = torch.abs(teacher_prob - labels)
            correction_helpful = corrected_error < base_error
            final_helpful = final_error < base_error
            teacher_closer_than_base = teacher_error < base_error
            correction_shift = torch.abs(corrected_prob - base_prob)
            final_shift = torch.abs(final_prob - base_prob)

            subjects = list_from_batch(batch["subject_id"])
            sessions = list_from_batch(batch["session"])
            groups = list_from_batch(batch.get("group_name", [""] * len(subjects)))
            starts = list_from_batch(batch.get("window_start_ms", [None] * len(subjects)))
            ends = list_from_batch(batch.get("window_end_ms", [None] * len(subjects)))

            tensors = {
                "label": labels,
                "alpha": alpha_values,
                "alpha_unit": alpha_unit_values,
                "base_prob": base_prob,
                "corrected_prob": corrected_prob,
                "final_prob": final_prob,
                "teacher_prob": teacher_prob,
                "base_abs_error": base_error,
                "corrected_abs_error": corrected_error,
                "final_abs_error": final_error,
                "teacher_abs_error": teacher_error,
                "correction_helpful": correction_helpful.float(),
                "final_helpful": final_helpful.float(),
                "teacher_closer_than_base": teacher_closer_than_base.float(),
                "correction_shift": correction_shift,
                "final_shift": final_shift,
                "watch_quality": quality.reshape(-1).float(),
                "acc_energy": acc_energy,
                "acc_jerk": acc_jerk,
                "ppg_std": ppg_std,
            }
            arrays = {key: value.detach().cpu().numpy() for key, value in tensors.items()}

            for idx in range(len(subjects)):
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


def safe_quantile(values: pd.Series, q: float) -> float:
    x = pd.to_numeric(values, errors="coerce").dropna()
    if x.empty:
        return float("nan")
    return float(x.quantile(q))


def summarize_windows(windows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    subject_rows: list[dict[str, Any]] = []
    for (dataset, fold_subject), group in windows.groupby(["dataset", "fold_subject"], sort=False):
        subject_rows.append(
            {
                "dataset": dataset,
                "fold_subject": fold_subject,
                "n_windows": int(len(group)),
                "alpha_mean": float(group["alpha"].mean()),
                "alpha_std": float(group["alpha"].std(ddof=1)) if len(group) > 1 else 0.0,
                "alpha_p25": safe_quantile(group["alpha"], 0.25),
                "alpha_p50": safe_quantile(group["alpha"], 0.50),
                "alpha_p75": safe_quantile(group["alpha"], 0.75),
                "quality_mean": float(group["watch_quality"].mean()),
                "acc_energy_mean": float(group["acc_energy"].mean()),
                "acc_jerk_mean": float(group["acc_jerk"].mean()),
                "correction_helpful_rate": float(group["correction_helpful"].mean()),
                "final_helpful_rate": float(group["final_helpful"].mean()),
                "teacher_closer_than_base_rate": float(group["teacher_closer_than_base"].mean()),
                "base_abs_error_mean": float(group["base_abs_error"].mean()),
                "corrected_abs_error_mean": float(group["corrected_abs_error"].mean()),
                "final_abs_error_mean": float(group["final_abs_error"].mean()),
                "teacher_abs_error_mean": float(group["teacher_abs_error"].mean()),
            }
        )
    subject_summary = pd.DataFrame(subject_rows)

    helpful_summary = (
        windows.assign(correction_helpful_label=windows["correction_helpful"].map(lambda x: "helpful" if x >= 0.5 else "not_helpful"))
        .groupby(["dataset", "correction_helpful_label"], as_index=False)
        .agg(
            n=("alpha", "size"),
            alpha_mean=("alpha", "mean"),
            alpha_median=("alpha", "median"),
            acc_energy_mean=("acc_energy", "mean"),
            final_abs_error_mean=("final_abs_error", "mean"),
            base_abs_error_mean=("base_abs_error", "mean"),
        )
    )

    label_summary = (
        windows.groupby(["dataset", "label"], as_index=False)
        .agg(
            n=("alpha", "size"),
            alpha_mean=("alpha", "mean"),
            alpha_median=("alpha", "median"),
            acc_energy_mean=("acc_energy", "mean"),
            final_abs_error_mean=("final_abs_error", "mean"),
        )
        .sort_values(["dataset", "label"])
    )
    return subject_summary, helpful_summary, label_summary


def fmt(value: object, digits: int = 4) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "nan"
    if not math.isfinite(x):
        return "nan"
    return f"{x:.{digits}f}"


def write_markdown_report(
    output_path: Path,
    windows: pd.DataFrame,
    subject_summary: pd.DataFrame,
    helpful_summary: pd.DataFrame,
    label_summary: pd.DataFrame,
) -> None:
    lines = ["# Adaptive Alpha Behavior", ""]
    lines.append(f"- Windows: {len(windows)}")
    lines.append(f"- Fold subjects: {windows['fold_subject'].nunique()}")
    lines.append(f"- Alpha mean: {fmt(windows['alpha'].mean())}")
    lines.append(f"- Alpha median: {fmt(windows['alpha'].median())}")
    lines.append(f"- Correction helpful rate: {fmt(windows['correction_helpful'].mean())}")
    lines.append(f"- Final helpful rate: {fmt(windows['final_helpful'].mean())}")
    lines.append("")
    lines.append("## Alpha By Correction Helpfulness")
    lines.append("")
    lines.append("| Group | n | alpha mean | alpha median | final error | base error |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in helpful_summary.itertuples(index=False):
        lines.append(
            f"| {row.correction_helpful_label} | {int(row.n)} | {fmt(row.alpha_mean)} | {fmt(row.alpha_median)} | "
            f"{fmt(row.final_abs_error_mean)} | {fmt(row.base_abs_error_mean)} |"
        )
    lines.append("")
    lines.append("## Alpha By Label")
    lines.append("")
    lines.append("| Label | n | alpha mean | alpha median | ACC energy | final error |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in label_summary.itertuples(index=False):
        lines.append(
            f"| {int(row.label)} | {int(row.n)} | {fmt(row.alpha_mean)} | {fmt(row.alpha_median)} | "
            f"{fmt(row.acc_energy_mean)} | {fmt(row.final_abs_error_mean)} |"
        )
    lines.append("")
    lines.append("## Subject-Level Alpha")
    lines.append("")
    lines.append("| Subject | n | alpha mean | correction helpful rate | ACC energy |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in subject_summary.sort_values("alpha_mean", ascending=False).head(20).itertuples(index=False):
        lines.append(
            f"| {row.fold_subject} | {int(row.n_windows)} | {fmt(row.alpha_mean)} | "
            f"{fmt(row.correction_helpful_rate)} | {fmt(row.acc_energy_mean)} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def try_plot(windows: pd.DataFrame, subject_summary: pd.DataFrame, output_path: Path, title: str) -> bool:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:
        print(f"matplotlib_unavailable={exc!r}; skipping alpha figure")
        return False

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.4), dpi=180)
    label_colors = {0.0: "#2563eb", 1.0: "#dc2626"}

    ax = axes[0, 0]
    for label, group in windows.groupby("label"):
        ax.hist(
            group["alpha"],
            bins=24,
            alpha=0.58,
            density=True,
            color=label_colors.get(float(label), "#64748b"),
            label=f"label={int(label)}",
        )
    ax.set_title("Alpha distribution by class")
    ax.set_xlabel("Adaptive correction weight alpha")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    groups = [
        windows[windows["correction_helpful"] < 0.5]["alpha"].dropna(),
        windows[windows["correction_helpful"] >= 0.5]["alpha"].dropna(),
    ]
    ax.boxplot(groups, labels=["not helpful", "helpful"], patch_artist=True)
    ax.set_title("Alpha when correction helps")
    ax.set_ylabel("Alpha")

    ax = axes[1, 0]
    plot_frame = windows.dropna(subset=["acc_jerk", "alpha"])
    if len(plot_frame) > 3000:
        plot_frame = plot_frame.sample(3000, random_state=42)
    ax.scatter(plot_frame["acc_jerk"], plot_frame["alpha"], s=8, alpha=0.30, color="#7c3aed", linewidths=0)
    ax.set_title("Alpha vs ACC jerk")
    ax.set_xlabel("ACC jerk RMS")
    ax.set_ylabel("Alpha")

    ax = axes[1, 1]
    subj = subject_summary.sort_values("alpha_mean", ascending=True)
    ax.barh(subj["fold_subject"], subj["alpha_mean"], color="#7c3aed", alpha=0.82)
    ax.set_title("Mean alpha by held-out subject")
    ax.set_xlabel("Mean alpha")

    for ax in axes.reshape(-1):
        ax.grid(True, color="#e2e8f0", linewidth=0.8, alpha=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze adaptive correction alpha behavior on LOSO test windows.")
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
    args = parser.parse_args()

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
        fold_rows = collect_fold_rows(subject, manifest_path, checkpoint_path, include_sessions, args)
        print(f"[{subject}] windows={len(fold_rows)}")
        all_rows.extend(fold_rows)

    windows = pd.DataFrame(all_rows)
    if windows.empty:
        raise ValueError("No adaptive alpha rows were collected.")
    subject_summary, helpful_summary, label_summary = summarize_windows(windows)

    prefix = args.dataset_kind
    windows_path = args.output_dir / f"{prefix}_adaptive_alpha_windows.csv"
    subject_path = args.output_dir / f"{prefix}_adaptive_alpha_subject_summary.csv"
    helpful_path = args.output_dir / f"{prefix}_adaptive_alpha_helpfulness_summary.csv"
    label_path = args.output_dir / f"{prefix}_adaptive_alpha_label_summary.csv"
    report_path = args.output_dir / f"{prefix}_adaptive_alpha_behavior.md"
    figure_path = args.output_dir / f"{prefix}_adaptive_alpha_behavior.svg"

    windows.to_csv(windows_path, index=False)
    subject_summary.to_csv(subject_path, index=False)
    helpful_summary.to_csv(helpful_path, index=False)
    label_summary.to_csv(label_path, index=False)
    write_markdown_report(report_path, windows, subject_summary, helpful_summary, label_summary)
    try_plot(windows, subject_summary, figure_path, title=f"{args.dataset_kind.upper()} Adaptive Alpha Behavior")

    print()
    print(f"windows={len(windows)} subjects={windows['fold_subject'].nunique()} alpha_mean={windows['alpha'].mean():.4f}")
    print("Alpha by correction helpfulness:")
    print(helpful_summary.to_string(index=False))
    print()
    print(f"Saved window outputs to {windows_path}")
    print(f"Saved subject summary to {subject_path}")
    print(f"Saved helpfulness summary to {helpful_path}")
    print(f"Saved label summary to {label_path}")
    print(f"Saved report to {report_path}")
    print(f"Saved figure to {figure_path}")


if __name__ == "__main__":
    main()
