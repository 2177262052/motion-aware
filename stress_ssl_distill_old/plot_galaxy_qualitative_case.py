from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .analyze_galaxy_model_failures import build_model_from_state, normalize_state_dict
from .galaxy_dataset import DEFAULT_WAVELET_BANDS, GalaxyPrivilegedWindowDataset


DEFAULT_CALM_SESSIONS = ["baseline"]
DEFAULT_STRESS_SESSIONS = ["tsst-prep"]


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, formats: list[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(output_dir / f"{stem}.{fmt}", bbox_inches="tight")
    plt.close(fig)


def safe_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def select_case(
    windows: pd.DataFrame,
    model_name: str,
    case_type: str,
    subject: str | None,
    window_start_ms: int | None,
) -> pd.Series:
    frame = windows[windows["model_name"].astype(str) == model_name].copy()
    if frame.empty:
        available = sorted(windows["model_name"].dropna().astype(str).unique())
        raise ValueError(f"Model {model_name!r} not found. Available: {available}")
    if subject is not None:
        frame = frame[frame["fold_subject"].astype(str) == subject]
    if window_start_ms is not None:
        frame = frame[pd.to_numeric(frame["window_start_ms"], errors="coerce") == int(window_start_ms)]
    if frame.empty:
        raise ValueError("No matching rows after subject/window filtering.")

    numeric_cols = [
        "acc_jerk_rms",
        "sgpc_rescue",
        "sgpc_harm",
        "base_correct",
        "deploy_correct",
        "deploy_abs_shift_from_base",
        "deploy_delta_norm",
        "deploy_gate_mean",
        "label",
    ]
    for column in numeric_cols:
        if column in frame.columns:
            frame[column] = safe_num(frame[column])

    if case_type == "manual":
        return frame.iloc[0]

    if case_type == "high_motion_rescue":
        required = {"sgpc_rescue", "acc_jerk_rms"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Case type {case_type} requires columns: {sorted(missing)}")
        candidates = frame[frame["sgpc_rescue"] > 0.5].copy()
        if candidates.empty:
            raise ValueError("No SGPC rescue case found. Try --case-type high_motion_error or manual.")
        return candidates.sort_values(["acc_jerk_rms", "deploy_abs_shift_from_base"], ascending=False).iloc[0]

    if case_type == "high_motion_error":
        candidates = frame[pd.to_numeric(frame.get("deploy_correct", 1), errors="coerce") < 0.5].copy()
        if candidates.empty:
            raise ValueError("No deploy error case found. Try another case type.")
        return candidates.sort_values(["acc_jerk_rms", "deploy_abs_shift_from_base"], ascending=False).iloc[0]

    if case_type == "stable_low_motion":
        required = {"base_correct", "deploy_correct", "acc_jerk_rms"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Case type {case_type} requires columns: {sorted(missing)}")
        candidates = frame[(frame["base_correct"] > 0.5) & (frame["deploy_correct"] > 0.5)].copy()
        if candidates.empty:
            raise ValueError("No stable correct case found. Try another case type.")
        return candidates.sort_values(["acc_jerk_rms", "deploy_abs_shift_from_base"], ascending=[True, True]).iloc[0]

    if case_type == "harm":
        if "sgpc_harm" not in frame.columns:
            raise ValueError("Case type harm requires sgpc_harm column.")
        candidates = frame[frame["sgpc_harm"] > 0.5].copy()
        if candidates.empty:
            raise ValueError("No SGPC harm case found. Try another case type.")
        return candidates.sort_values(["acc_jerk_rms", "deploy_abs_shift_from_base"], ascending=False).iloc[0]

    raise ValueError(f"Unsupported case type: {case_type}")


def checkpoint_for_subject(checkpoint_dir: Path, subject: str) -> Path:
    candidates = [
        checkpoint_dir / f"{subject}_deploy_watch.pt",
        checkpoint_dir / f"{subject}.pt",
    ]
    for path in candidates:
        if path.exists():
            return path
    matches = sorted(checkpoint_dir.glob(f"{subject}*.pt"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"No checkpoint found for {subject} in {checkpoint_dir}")


def manifest_for_subject(manifests_dir: Path, subject: str) -> Path:
    path = manifests_dir / f"galaxy_{subject}_loso_val.csv"
    if not path.exists():
        raise FileNotFoundError(f"No manifest found for {subject}: {path}")
    return path


def find_dataset_index(dataset: GalaxyPrivilegedWindowDataset, row: pd.Series) -> int:
    start_ms = int(row["window_start_ms"])
    end_ms = int(row["window_end_ms"])
    subject = str(row["fold_subject"])
    mask = (
        (dataset.manifest["subject_id"].astype(str) == subject)
        & (pd.to_numeric(dataset.manifest["window_start_ms"], errors="coerce") == start_ms)
        & (pd.to_numeric(dataset.manifest["window_end_ms"], errors="coerce") == end_ms)
    )
    matches = np.flatnonzero(mask.to_numpy())
    if len(matches) == 0:
        raise ValueError(f"Could not find selected window in dataset: {subject} {start_ms}-{end_ms}")
    return int(matches[0])


def tensor_batch(sample: dict[str, object], device: str) -> dict[str, object]:
    out: dict[str, object] = {}
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.unsqueeze(0).to(device)
        else:
            out[key] = [value]
    return out


def probability_from_logits(logits: torch.Tensor) -> float:
    return float(torch.softmax(logits, dim=1)[0, 1].detach().cpu().item())


def forward_case(model: torch.nn.Module, batch: dict[str, object], device: str) -> dict[str, torch.Tensor]:
    watch_signal = batch["watch_signal"].to(device)
    wavelet = batch["wavelet_features"].to(device)
    quality = batch["watch_quality"].to(device)
    e4_signal = batch["e4_signal"].to(device)
    with torch.no_grad():
        out = model(watch_signal, wavelet, quality, e4_signal=e4_signal, return_aux=True)
    return out


def acc_magnitude_and_jerk(acc: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    if acc.ndim != 2:
        raise ValueError(f"Expected ACC shape [channels, time], got {acc.shape}")
    if acc.shape[0] >= 3:
        mag = np.sqrt(np.sum(acc[:3] ** 2, axis=0))
    else:
        mag = np.abs(acc[0])
    jerk = np.diff(mag, prepend=mag[0]) * fs
    return mag, jerk


def upsample_mask(mask: np.ndarray, target_len: int) -> np.ndarray:
    if mask.ndim == 3:
        mask = mask[0]
    if mask.ndim == 2:
        mask = mask.mean(axis=0)
    mask = np.asarray(mask, dtype=float).reshape(-1)
    if len(mask) == target_len:
        return mask
    src_x = np.linspace(0.0, 1.0, num=len(mask))
    dst_x = np.linspace(0.0, 1.0, num=target_len)
    return np.interp(dst_x, src_x, mask)


def plot_case(
    sample: dict[str, object],
    out: dict[str, torch.Tensor],
    selected: pd.Series,
    output_dir: Path,
    formats: list[str],
    fs: float,
    case_type: str,
) -> None:
    watch = sample["watch_signal"].detach().cpu().numpy()
    ppg = watch[0]
    acc = watch[1:]
    n = ppg.shape[-1]
    time_s = np.arange(n) / fs
    acc_mag, jerk = acc_magnitude_and_jerk(acc, fs)

    base_prob = probability_from_logits(out["base_logits"]) if "base_logits" in out else float("nan")
    deploy_prob = probability_from_logits(out["logits"])
    teacher_prob = probability_from_logits(out["teacher_logits"]) if "teacher_logits" in out else float("nan")
    priv_prob = probability_from_logits(out["privileged_correction_logits"]) if "privileged_correction_logits" in out else float("nan")

    label = int(sample["label"])
    session = str(sample["session"])
    subject = str(sample["subject_id"])
    start_ms = int(sample["window_start_ms"])

    mask_line = None
    if "motion_artifact_mask" in out:
        mask_line = upsample_mask(out["motion_artifact_mask"].detach().cpu().numpy(), n)

    gate_mean = float("nan")
    gate_max = float("nan")
    delta_norm = float("nan")
    if "deploy_correction_gate" in out:
        gate = out["deploy_correction_gate"].detach().cpu().numpy()[0]
        gate_mean = float(np.mean(gate))
        gate_max = float(np.max(gate))
    if "deploy_correction_delta" in out:
        delta = out["deploy_correction_delta"].detach().cpu().numpy()[0]
        delta_norm = float(np.linalg.norm(delta))

    fig, axes = plt.subplots(4, 1, figsize=(8.2, 7.6), gridspec_kw={"height_ratios": [1.1, 1.1, 1.0, 1.1]})
    ax = axes[0]
    ax.plot(time_s, ppg, color="#2b6cb0", linewidth=1.0)
    ax.set_title(f"(A) Raw watch PPG | {subject} {session} | label={'stress' if label else 'baseline'} | {case_type}")
    ax.set_ylabel("PPG\n(z-score)")
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)

    ax = axes[1]
    ax.plot(time_s, acc_mag, color="#4a5568", linewidth=1.0, label="ACC magnitude")
    ax2 = ax.twinx()
    ax2.plot(time_s, np.abs(jerk), color="#dd6b20", linewidth=0.8, alpha=0.75, label="|jerk|")
    ax.set_title("(B) Watch motion context")
    ax.set_ylabel("ACC mag.")
    ax2.set_ylabel("|jerk|")
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, frameon=False, loc="upper right")

    ax = axes[2]
    if mask_line is not None:
        ax.plot(time_s, mask_line, color="#805ad5", linewidth=1.0, label="motion artifact mask")
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Mask")
    else:
        ax.bar(["gate mean", "gate max"], [gate_mean, gate_max], color=["#805ad5", "#b794f4"])
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Gate")
    ax.axhline(gate_mean, color="#2f855a", linestyle="--", linewidth=1.0, label=f"gate mean={gate_mean:.2f}")
    ax.text(
        0.99,
        0.08,
        f"correction norm={delta_norm:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#e2e8f0"},
    )
    ax.set_title("(C) Motion-aware mask and deploy correction behavior")
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
    ax.legend(frameon=False, loc="upper right")

    ax = axes[3]
    names = ["Base", "Teacher", "Priv. Corr.", "Ours"]
    probs = [base_prob, teacher_prob, priv_prob, deploy_prob]
    colors = ["#4c78a8", "#dd6b20", "#68a06a", "#2f855a"]
    bars = ax.bar(names, probs, color=colors, edgecolor="#1a202c", linewidth=0.5)
    ax.axhline(0.5, color="#718096", linestyle="--", linewidth=1.0, label="threshold")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Stress probability")
    ax.set_title("(D) Prediction probabilities")
    for bar, prob in zip(bars, probs):
        if np.isfinite(prob):
            ax.text(bar.get_x() + bar.get_width() / 2, prob + 0.025, f"{prob:.2f}", ha="center", fontsize=8)
    truth_text = "Ground truth: stress" if label else "Ground truth: baseline"
    ax.text(
        0.02,
        0.92,
        truth_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#e2e8f0"},
    )
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)

    axes[-1].set_xlabel("Time in window (s)")
    for axis in axes[:3]:
        axis.set_xlim(time_s[0], time_s[-1])
        axis.set_xlabel("")
    fig.tight_layout()

    stem = f"qualitative_{case_type}_{subject}_{start_ms}"
    save_figure(fig, output_dir, stem, formats)

    row = {
        "case_type": case_type,
        "subject_id": subject,
        "session": session,
        "label": label,
        "window_start_ms": start_ms,
        "window_end_ms": int(sample["window_end_ms"]),
        "base_prob": base_prob,
        "teacher_prob": teacher_prob,
        "privileged_correction_prob": priv_prob,
        "deploy_prob": deploy_prob,
        "gate_mean": gate_mean,
        "gate_max": gate_max,
        "deploy_delta_norm": delta_norm,
        "selected_acc_jerk_rms": float(selected.get("acc_jerk_rms", np.nan)),
        "selected_sgpc_rescue": float(selected.get("sgpc_rescue", np.nan)),
        "selected_sgpc_harm": float(selected.get("sgpc_harm", np.nan)),
    }
    pd.DataFrame([row]).to_csv(output_dir / f"{stem}.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot a qualitative Galaxy PPG/ACC/SGPC case study.")
    parser.add_argument("--failure-windows-csv", type=Path, required=True)
    parser.add_argument("--manifests-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case-type", type=str, default="high_motion_rescue", choices=["high_motion_rescue", "high_motion_error", "stable_low_motion", "harm", "manual"])
    parser.add_argument("--model-name", type=str, default="Ours")
    parser.add_argument("--subject", type=str, default=None)
    parser.add_argument("--window-start-ms", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--watch-backbone", type=str, default="wavelet_guided")
    parser.add_argument("--correction-scale-init", type=float, default=0.05)
    parser.add_argument("--fs", type=float, default=25.0)
    parser.add_argument("--calm-sessions", nargs="*", default=DEFAULT_CALM_SESSIONS)
    parser.add_argument("--stress-sessions", nargs="*", default=DEFAULT_STRESS_SESSIONS)
    parser.add_argument("--formats", nargs="*", default=["png", "pdf"])
    args = parser.parse_args()

    set_plot_style()
    windows = pd.read_csv(args.failure_windows_csv)
    selected = select_case(
        windows,
        model_name=args.model_name,
        case_type=args.case_type,
        subject=args.subject,
        window_start_ms=args.window_start_ms,
    )
    subject = str(selected["fold_subject"])
    manifest_path = manifest_for_subject(args.manifests_dir, subject)
    checkpoint_path = checkpoint_for_subject(args.checkpoint_dir, subject)

    include_sessions = list(args.calm_sessions) + list(args.stress_sessions)
    dataset = GalaxyPrivilegedWindowDataset(
        manifest_csv=manifest_path,
        split="test",
        dataset_root=args.dataset_root,
        include_sessions=include_sessions,
        cache_tables=True,
        wavelet_bands=DEFAULT_WAVELET_BANDS,
    )
    sample = dataset[find_dataset_index(dataset, selected)]

    state = normalize_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model = build_model_from_state(
        state,
        watch_backbone=args.watch_backbone,
        correction_scale_init=args.correction_scale_init,
        device=args.device,
    )
    model.eval()
    batch = tensor_batch(sample, args.device)
    out = forward_case(model, batch, args.device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_case(sample, out, selected, args.output_dir, args.formats, fs=args.fs, case_type=args.case_type)
    print(
        "selected_case="
        f"subject:{subject} "
        f"start_ms:{int(selected['window_start_ms'])} "
        f"session:{selected.get('session', '')} "
        f"label:{int(selected['label'])} "
        f"acc_jerk_rms:{float(selected.get('acc_jerk_rms', np.nan)):.4f} "
        f"sgpc_rescue:{float(selected.get('sgpc_rescue', np.nan)):.1f} "
        f"sgpc_harm:{float(selected.get('sgpc_harm', np.nan)):.1f}"
    )
    print(f"Saved qualitative case figure to {args.output_dir}")


if __name__ == "__main__":
    main()
