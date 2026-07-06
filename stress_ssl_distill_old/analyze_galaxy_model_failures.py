from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .analyze_galaxy_data_quality import (
    DEFAULT_CALM_SESSIONS,
    DEFAULT_STRESS_SESSIONS,
    extract_window_features,
    maybe_parse_sessions,
)
from .galaxy_dataset import DEFAULT_WAVELET_BANDS, GalaxyPrivilegedWindowDataset
from .galaxy_models import PrivilegedGalaxyTeacherNet, WaveletGuidedWatchNet


OUTPUT_NAMES = ("deploy", "base", "teacher", "privileged_correction")


def discover_manifests(manifests_dir: Path, subjects: Iterable[str] | None) -> dict[str, Path]:
    requested = {subject.strip() for subject in subjects or [] if subject.strip()}
    manifests: dict[str, Path] = {}
    for path in sorted(manifests_dir.glob("galaxy_*_loso_val.csv")):
        subject = path.stem.replace("galaxy_", "").replace("_loso_val", "")
        if requested and subject not in requested:
            continue
        manifests[subject] = path
    return manifests


def normalize_state_dict(raw: object) -> dict[str, torch.Tensor]:
    if isinstance(raw, dict) and "state_dict" in raw and isinstance(raw["state_dict"], dict):
        raw = raw["state_dict"]
    if not isinstance(raw, dict):
        raise TypeError(f"Expected a state_dict-like object, got {type(raw)!r}")
    state: dict[str, torch.Tensor] = {}
    for key, value in raw.items():
        if not isinstance(value, torch.Tensor):
            continue
        name = str(key)
        if name.startswith("module."):
            name = name[len("module.") :]
        state[name] = value
    return state


def has_prefix(state: dict[str, torch.Tensor], *prefixes: str) -> bool:
    return any(any(key.startswith(prefix) for prefix in prefixes) for key in state)


def infer_num_phenotypes(state: dict[str, torch.Tensor]) -> int:
    for key, value in state.items():
        if key.endswith("phenotype_router.weight") and value.ndim == 2:
            return int(value.shape[0])
        if key.endswith("phenotype_router.bias") and value.ndim == 1:
            return int(value.shape[0])
    return 0


def infer_wavelet_dim(state: dict[str, torch.Tensor], prefix: str = "") -> int:
    key = f"{prefix}wavelet_mlp.0.weight"
    value = state.get(key)
    if value is not None and value.ndim == 2:
        return int(value.shape[1] - 1)
    return 4


def infer_model_dim(state: dict[str, torch.Tensor], prefix: str = "") -> int:
    key = f"{prefix}cls_token"
    value = state.get(key)
    if value is not None and value.ndim == 3:
        return int(value.shape[-1])
    return 192


def infer_embed_dim(state: dict[str, torch.Tensor], prefix: str = "") -> int:
    for key in (f"{prefix}classifier.weight", "watch_classifier.weight"):
        value = state.get(key)
        if value is not None and value.ndim == 2:
            return int(value.shape[1])
    return 160


def infer_fusion_hidden_dim(state: dict[str, torch.Tensor], prefix: str = "") -> int:
    key = f"{prefix}fusion.0.weight"
    value = state.get(key)
    if value is not None and value.ndim == 2:
        return int(value.shape[0])
    return 256


def infer_transformer_layers(state: dict[str, torch.Tensor], prefix: str = "") -> int:
    pattern = f"{prefix}transformer.layers."
    layer_ids = set()
    for key in state:
        if not key.startswith(pattern):
            continue
        rest = key[len(pattern) :]
        layer_id = rest.split(".", 1)[0]
        if layer_id.isdigit():
            layer_ids.add(int(layer_id))
    return max(layer_ids) + 1 if layer_ids else 2


def is_watch_only_state(state: dict[str, torch.Tensor]) -> bool:
    return has_prefix(state, "ppg_stem.", "acc_stem.") and not has_prefix(state, "watch_encoder.")


def build_model_from_state(
    state: dict[str, torch.Tensor],
    watch_backbone: str,
    correction_scale_init: float,
    device: str,
) -> torch.nn.Module:
    if is_watch_only_state(state):
        watch_enhancement = "motion_disentangled" if has_prefix(state, "ppg_enhancer.") else "none"
        if watch_backbone != "wavelet_guided":
            raise ValueError("Watch-only checkpoint auto-loading currently expects the wavelet_guided backbone.")
        model = WaveletGuidedWatchNet(
            wavelet_dim=infer_wavelet_dim(state),
            model_dim=infer_model_dim(state),
            transformer_layers=infer_transformer_layers(state),
            fusion_hidden_dim=infer_fusion_hidden_dim(state),
            embed_dim=infer_embed_dim(state),
            watch_enhancement=watch_enhancement,
        )
        result = model.load_state_dict(state, strict=False)
        if result.missing_keys:
            print(f"watch_load_missing_keys={result.missing_keys[:12]} total={len(result.missing_keys)}")
        if result.unexpected_keys:
            print(f"watch_load_unexpected_keys={result.unexpected_keys[:12]} total={len(result.unexpected_keys)}")
        model.to(device)
        model.eval()
        return model

    watch_enhancement = "motion_disentangled" if has_prefix(state, "watch_encoder.ppg_enhancer.") else "none"
    model = PrivilegedGalaxyTeacherNet(
        num_phenotypes=infer_num_phenotypes(state),
        watch_backbone=watch_backbone,
        watch_enhancement=watch_enhancement,
        use_reliability_head=has_prefix(state, "reliability_head."),
        use_projection_heads=has_prefix(state, "watch_projector.", "e4_projector."),
        use_e4_classifier=has_prefix(state, "e4_classifier."),
        use_rhythm_heads=has_prefix(state, "rhythm_head.", "teacher_rhythm_head."),
        use_wavelet_head=has_prefix(state, "wavelet_predictor."),
        use_teacher_fused_classifier=has_prefix(state, "teacher_fused_classifier."),
        use_student_gated_correction=has_prefix(
            state,
            "deploy_correction.",
            "deploy_correction_gate.",
            "privileged_correction.",
            "privileged_correction_gate.",
        ),
        correction_scale_init=correction_scale_init,
    )
    result = model.load_state_dict(state, strict=False)
    non_optional_missing = [
        key
        for key in result.missing_keys
        if not key.startswith(
            (
                "reliability_head.",
                "watch_projector.",
                "e4_projector.",
                "e4_classifier.",
                "rhythm_head.",
                "teacher_rhythm_head.",
                "wavelet_predictor.",
                "teacher_fused_classifier.",
                "deploy_correction.",
                "deploy_correction_gate.",
                "privileged_correction.",
                "privileged_correction_gate.",
                "correction_norm.",
                "correction_scale",
            )
        )
    ]
    if non_optional_missing:
        print(f"load_missing_keys={non_optional_missing[:12]} total={len(non_optional_missing)}")
    if result.unexpected_keys:
        print(f"load_unexpected_keys={result.unexpected_keys[:12]} total={len(result.unexpected_keys)}")
    model.to(device)
    model.eval()
    return model


def parse_named_path(value: str, default_name: str | None = None) -> tuple[str, Path]:
    if "=" in value:
        name, path = value.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Missing name in named path: {value}")
        return name, Path(path)
    path = Path(value)
    return default_name or path.stem, path


def discover_checkpoint_sets(args: argparse.Namespace) -> dict[str, dict[str, Path]]:
    sets: dict[str, dict[str, Path]] = {}
    if args.checkpoint:
        if args.manifest is None:
            raise ValueError("--checkpoint requires --manifest.")
        subject = args.subject_name or args.manifest.stem.replace("galaxy_", "").replace("_loso_val", "")
        for item in args.checkpoint:
            name, path = parse_named_path(item)
            sets.setdefault(name, {})[subject] = path
    if args.checkpoint_dir:
        for item in args.checkpoint_dir:
            name, path = parse_named_path(item, default_name="model")
            requested = {subject.strip() for subject in args.subjects or [] if subject.strip()}
            ckpts: dict[str, Path] = {}
            candidate_paths = sorted(path.glob("*_deploy_watch.pt")) + sorted(path.glob("P*.pt"))
            for ckpt in candidate_paths:
                if ckpt.name.endswith("_deploy_watch.pt"):
                    subject = ckpt.name.replace("_deploy_watch.pt", "")
                else:
                    subject = ckpt.stem
                if requested and subject not in requested:
                    continue
                if subject in ckpts and ckpts[subject].name.endswith("_deploy_watch.pt"):
                    continue
                ckpts[subject] = ckpt
            if not ckpts:
                raise ValueError(f"No *_deploy_watch.pt or P*.pt checkpoints found in {path}")
            sets[name] = ckpts
    if not sets:
        raise ValueError("Provide --checkpoint NAME=PATH and/or --checkpoint-dir NAME=DIR.")
    return sets


def build_loader(
    manifest_path: Path,
    dataset_root: Path,
    split: str,
    include_sessions: Sequence[str],
    args: argparse.Namespace,
) -> DataLoader:
    dataset = GalaxyPrivilegedWindowDataset(
        manifest_csv=manifest_path,
        split=split,
        dataset_root=dataset_root,
        include_sessions=include_sessions,
        cache_tables=True,
        wavelet=args.wavelet,
        wavelet_level=args.wavelet_level,
        wavelet_bands=DEFAULT_WAVELET_BANDS,
        baseline_reference=args.baseline_reference,
    )
    kwargs = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
    }
    if args.num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **kwargs)


def safe_metrics(y_true: Sequence[int], y_prob: Sequence[float], threshold: float) -> dict[str, float]:
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.asarray(y_prob, dtype=np.float64)
    y_pred = (y_prob_arr >= threshold).astype(int)
    out = {
        "threshold": float(threshold),
        "acc": float(accuracy_score(y_true_arr, y_pred)) if len(y_true_arr) else float("nan"),
        "balanced_acc": float(balanced_accuracy_score(y_true_arr, y_pred)) if len(y_true_arr) else float("nan"),
        "f1": float(f1_score(y_true_arr, y_pred, zero_division=0)) if len(y_true_arr) else float("nan"),
        "positive_rate": float(np.mean(y_pred)) if len(y_pred) else float("nan"),
    }
    if len(np.unique(y_true_arr)) < 2 or len(np.unique(y_prob_arr)) < 2:
        out["auroc"] = float("nan")
    else:
        out["auroc"] = float(roc_auc_score(y_true_arr, y_prob_arr))
    return out


def select_threshold(y_true: Sequence[int], y_prob: Sequence[float], metric: str) -> float:
    if len(set(y_true)) < 2:
        return 0.5
    candidates = sorted(set([0.0, 1.0] + [round(float(prob), 6) for prob in y_prob]))
    best_threshold = 0.5
    best_score = safe_metrics(y_true, y_prob, best_threshold).get(metric, float("nan"))
    if not np.isfinite(best_score):
        best_score = -float("inf")
    for threshold in candidates:
        metrics = safe_metrics(y_true, y_prob, threshold)
        score = metrics.get(metric, float("nan"))
        if np.isfinite(score) and score > best_score + 1e-12:
            best_score = score
            best_threshold = threshold
    return float(best_threshold)


def probability_from_logits(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=1)[:, 1]


def margin_from_logits(logits: torch.Tensor) -> torch.Tensor:
    return logits[:, 1] - logits[:, 0]


def forward_batch(
    model: torch.nn.Module,
    batch: dict[str, object],
    device: str,
    pin_memory: bool,
    baseline_reference: bool,
) -> dict[str, torch.Tensor]:
    watch_signal = batch["watch_signal"].to(device, non_blocking=pin_memory)
    wavelet = batch["wavelet_features"].to(device, non_blocking=pin_memory)
    quality = batch["watch_quality"].to(device, non_blocking=pin_memory)
    e4_signal = batch["e4_signal"].to(device, non_blocking=pin_memory)
    baseline_kwargs = {}
    if baseline_reference:
        baseline_kwargs = {
            "baseline_watch_signal": batch["baseline_watch_signal"].to(device, non_blocking=pin_memory),
            "baseline_wavelet_features": batch["baseline_wavelet_features"].to(device, non_blocking=pin_memory),
            "baseline_quality": batch["baseline_watch_quality"].to(device, non_blocking=pin_memory),
        }
    with torch.no_grad():
        if isinstance(model, WaveletGuidedWatchNet):
            return model(
                watch_signal,
                wavelet,
                quality,
                baseline_signal=baseline_kwargs.get("baseline_watch_signal"),
                baseline_wavelet_features=baseline_kwargs.get("baseline_wavelet_features"),
                baseline_quality=baseline_kwargs.get("baseline_quality"),
            )
        return model(
            watch_signal,
            wavelet,
            quality,
            e4_signal=e4_signal,
            return_aux=True,
            **baseline_kwargs,
        )


def collect_frame(
    model: torch.nn.Module,
    loader: DataLoader,
    model_name: str,
    subject: str,
    split: str,
    device: str,
    pin_memory: bool,
    baseline_reference: bool,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for batch in tqdm(loader, desc=f"{model_name}:{subject}:{split}", leave=True):
        out = forward_batch(model, batch, device, pin_memory, baseline_reference)
        output_logits: dict[str, torch.Tensor] = {"deploy": out["logits"]}
        if "base_logits" in out:
            output_logits["base"] = out["base_logits"]
        if "teacher_logits" in out:
            output_logits["teacher"] = out["teacher_logits"]
        if "privileged_correction_logits" in out:
            output_logits["privileged_correction"] = out["privileged_correction_logits"]

        probs = {name: probability_from_logits(logits).detach().cpu().numpy() for name, logits in output_logits.items()}
        margins = {name: margin_from_logits(logits).detach().cpu().numpy() for name, logits in output_logits.items()}
        labels = batch["label"].detach().cpu().numpy().astype(int)
        batch_size = int(labels.shape[0])

        deploy_gate = out.get("deploy_correction_gate")
        deploy_delta = out.get("deploy_correction_delta")
        priv_gate = out.get("privileged_correction_gate")
        priv_delta = out.get("privileged_correction_delta")
        motion_artifact_mask = out.get("motion_artifact_mask")
        deploy_gate_np = deploy_gate.detach().cpu().numpy() if deploy_gate is not None else None
        deploy_delta_np = deploy_delta.detach().cpu().numpy() if deploy_delta is not None else None
        priv_gate_np = priv_gate.detach().cpu().numpy() if priv_gate is not None else None
        priv_delta_np = priv_delta.detach().cpu().numpy() if priv_delta is not None else None
        motion_artifact_mask_np = motion_artifact_mask.detach().cpu().numpy() if motion_artifact_mask is not None else None

        for idx in range(batch_size):
            features = extract_window_features(
                batch["watch_signal"][idx],
                batch["wavelet_features"][idx],
                batch["watch_quality"][idx],
                e4_signal=batch["e4_signal"][idx],
                polar_targets=batch["polar_targets"][idx],
                polar_target_mask=batch["polar_target_mask"][idx],
                polar_coverage=batch["polar_coverage"][idx],
                wavelet_bands=DEFAULT_WAVELET_BANDS,
            )
            row: dict[str, object] = {
                "model_name": model_name,
                "fold_subject": subject,
                "split": split,
                "subject_id": str(batch["subject_id"][idx]),
                "session": str(batch["session"][idx]),
                "group_name": str(batch["group_name"][idx]),
                "label": int(labels[idx]),
                "window_start_ms": int(batch["window_start_ms"][idx]),
                "window_end_ms": int(batch["window_end_ms"][idx]),
                **features,
            }
            for name in OUTPUT_NAMES:
                if name in probs:
                    row[f"{name}_prob"] = float(probs[name][idx])
                    row[f"{name}_margin"] = float(margins[name][idx])
            if "deploy_prob" in row and "base_prob" in row:
                row["deploy_minus_base_prob"] = float(row["deploy_prob"] - row["base_prob"])
                row["deploy_abs_shift_from_base"] = abs(float(row["deploy_minus_base_prob"]))
            if "teacher_prob" in row and "deploy_prob" in row:
                row["teacher_minus_deploy_prob"] = float(row["teacher_prob"] - row["deploy_prob"])
            if deploy_gate_np is not None:
                gate = deploy_gate_np[idx]
                row["deploy_gate_mean"] = float(np.mean(gate))
                row["deploy_gate_std"] = float(np.std(gate))
                row["deploy_gate_max"] = float(np.max(gate))
            if deploy_delta_np is not None:
                delta = deploy_delta_np[idx]
                row["deploy_delta_norm"] = float(np.linalg.norm(delta))
            if priv_gate_np is not None:
                gate = priv_gate_np[idx]
                row["privileged_gate_mean"] = float(np.mean(gate))
                row["privileged_gate_std"] = float(np.std(gate))
            if priv_delta_np is not None:
                delta = priv_delta_np[idx]
                row["privileged_delta_norm"] = float(np.linalg.norm(delta))
            if motion_artifact_mask_np is not None:
                mask = motion_artifact_mask_np[idx]
                row["motion_artifact_mask_mean"] = float(np.mean(mask))
                row["motion_artifact_mask_std"] = float(np.std(mask))
                row["motion_artifact_mask_max"] = float(np.max(mask))
            rows.append(row)
    return pd.DataFrame(rows)


def add_predictions(frame: pd.DataFrame, thresholds: dict[str, float]) -> pd.DataFrame:
    out = frame.copy()
    for name, threshold in thresholds.items():
        prob_col = f"{name}_prob"
        if prob_col not in out.columns:
            continue
        pred_col = f"{name}_pred"
        correct_col = f"{name}_correct"
        out[f"{name}_threshold"] = threshold
        out[pred_col] = (pd.to_numeric(out[prob_col], errors="coerce") >= threshold).astype(int)
        out[correct_col] = (out[pred_col].astype(int) == out["label"].astype(int)).astype(int)
    if {"base_correct", "deploy_correct"}.issubset(out.columns):
        out["sgpc_rescue"] = ((out["base_correct"] == 0) & (out["deploy_correct"] == 1)).astype(int)
        out["sgpc_harm"] = ((out["base_correct"] == 1) & (out["deploy_correct"] == 0)).astype(int)
    if {"deploy_correct", "teacher_correct"}.issubset(out.columns):
        out["teacher_rescue_possible"] = ((out["deploy_correct"] == 0) & (out["teacher_correct"] == 1)).astype(int)
        out["teacher_harm_risk"] = ((out["deploy_correct"] == 1) & (out["teacher_correct"] == 0)).astype(int)
        out["deploy_teacher_pred_disagree"] = (out["deploy_pred"] != out["teacher_pred"]).astype(int)
    return out


def summarize_outputs(frame: pd.DataFrame, thresholds: dict[str, float]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    y_true = frame["label"].astype(int).tolist()
    for name, threshold in thresholds.items():
        prob_col = f"{name}_prob"
        if prob_col not in frame.columns:
            continue
        metrics = safe_metrics(y_true, pd.to_numeric(frame[prob_col], errors="coerce").tolist(), threshold)
        rows.append(
            {
                "model_name": str(frame["model_name"].iloc[0]),
                "fold_subject": str(frame["fold_subject"].iloc[0]),
                "output": name,
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def add_quantile_bin(frame: pd.DataFrame, column: str, output_column: str) -> pd.DataFrame:
    out = frame.copy()
    labels = ["low", "mid", "high"]
    values = pd.to_numeric(out[column], errors="coerce")
    try:
        out[output_column] = pd.qcut(values, q=3, labels=labels, duplicates="drop")
    except ValueError:
        out[output_column] = "all"
    out[output_column] = out[output_column].astype(str).replace("nan", "missing")
    return out


def safe_group_ba(labels: pd.Series, preds: pd.Series) -> float:
    labels_num = pd.to_numeric(labels, errors="coerce")
    preds_num = pd.to_numeric(preds, errors="coerce")
    valid = labels_num.notna() & preds_num.notna()
    if not valid.any():
        return float("nan")
    return float(balanced_accuracy_score(labels_num[valid].astype(int), preds_num[valid].astype(int)))


def safe_numeric_mean(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return float("nan")
    return float(clean.mean())


def summarize_failure_bins(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    out = add_quantile_bin(frame, "watch_quality", "watch_quality_bin")
    out = add_quantile_bin(out, "acc_jerk_rms", "motion_bin")
    rows: list[dict[str, object]] = []
    groupings = [
        ["model_name", "watch_quality_bin"],
        ["model_name", "motion_bin"],
        ["model_name", "watch_quality_bin", "motion_bin"],
    ]
    for keys in groupings:
        for values, group in out.groupby(keys, dropna=False):
            if not isinstance(values, tuple):
                values = (values,)
            row = {
                "grouping": "+".join(keys),
                "n": int(len(group)),
                "positive_prior": float(group["label"].mean()),
                "deploy_acc": safe_numeric_mean(group["deploy_correct"]) if "deploy_correct" in group else float("nan"),
                "deploy_balanced_acc": safe_group_ba(group["label"], group["deploy_pred"]) if "deploy_pred" in group else float("nan"),
                "base_acc": safe_numeric_mean(group["base_correct"]) if "base_correct" in group else float("nan"),
                "base_balanced_acc": safe_group_ba(group["label"], group["base_pred"]) if "base_pred" in group else float("nan"),
                "deploy_positive_rate": safe_numeric_mean(group["deploy_pred"]) if "deploy_pred" in group else float("nan"),
                "watch_quality_mean": safe_numeric_mean(group["watch_quality"]),
                "acc_jerk_rms_mean": safe_numeric_mean(group["acc_jerk_rms"]),
                "ppg_noise_ratio_4_8_mean": safe_numeric_mean(group["ppg_noise_ratio_4_8"]),
                "deploy_abs_shift_from_base_mean": safe_numeric_mean(group.get("deploy_abs_shift_from_base", pd.Series(dtype=float))),
                "sgpc_rescue_rate": safe_numeric_mean(group.get("sgpc_rescue", pd.Series(dtype=float))),
                "sgpc_harm_rate": safe_numeric_mean(group.get("sgpc_harm", pd.Series(dtype=float))),
                "teacher_rescue_possible_rate": safe_numeric_mean(group.get("teacher_rescue_possible", pd.Series(dtype=float))),
            }
            for key, value in zip(keys, values):
                row[key] = value
            rows.append(row)
    return pd.DataFrame(rows)


def thresholds_from_val_frame(val_frame: pd.DataFrame, monitor: str) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    y_true = val_frame["label"].astype(int).tolist()
    for name in OUTPUT_NAMES:
        prob_col = f"{name}_prob"
        if prob_col in val_frame.columns:
            thresholds[name] = select_threshold(y_true, pd.to_numeric(val_frame[prob_col], errors="coerce").tolist(), monitor)
    return thresholds


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load Galaxy deploy-watch checkpoints and export per-window failure/gate diagnostics."
    )
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--manifests-dir", type=Path, default=None)
    parser.add_argument("--subject-name", type=str, default=None)
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=None,
        help="Single-fold checkpoint as NAME=PATH. Can be repeated. Requires --manifest.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        action="append",
        default=None,
        help="LOSO checkpoint directory as NAME=DIR. Can be repeated. Looks for *_deploy_watch.pt.",
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--monitor", type=str, default="balanced_acc", choices=["acc", "balanced_acc", "f1", "auroc"])
    parser.add_argument("--watch-backbone", type=str, default="wavelet_guided", choices=["wavelet_guided", "resnet18_1d", "resnet34_1d", "resnet50_1d"])
    parser.add_argument("--correction-scale-init", type=float, default=0.05)
    parser.add_argument("--baseline-reference", action="store_true")
    parser.add_argument("--calm-sessions", nargs="*", default=DEFAULT_CALM_SESSIONS)
    parser.add_argument("--stress-sessions", nargs="*", default=DEFAULT_STRESS_SESSIONS)
    parser.add_argument("--wavelet", type=str, default="db4")
    parser.add_argument("--wavelet-level", type=int, default=4)
    args = parser.parse_args()

    if args.manifest is None and args.manifests_dir is None:
        raise ValueError("Provide --manifest for one fold or --manifests-dir for LOSO directories.")
    if args.manifest is not None and args.manifests_dir is not None:
        raise ValueError("Use only one of --manifest or --manifests-dir.")

    checkpoint_sets = discover_checkpoint_sets(args)
    manifests: dict[str, Path]
    if args.manifest is not None:
        subject = args.subject_name or args.manifest.stem.replace("galaxy_", "").replace("_loso_val", "")
        manifests = {subject: args.manifest}
    else:
        manifests = discover_manifests(args.manifests_dir, args.subjects)
    if not manifests:
        raise ValueError("No manifests found.")

    include_sessions = maybe_parse_sessions(args.calm_sessions, DEFAULT_CALM_SESSIONS) + maybe_parse_sessions(
        args.stress_sessions,
        DEFAULT_STRESS_SESSIONS,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_windows: list[pd.DataFrame] = []
    all_summary: list[pd.DataFrame] = []
    for model_name, ckpts in sorted(checkpoint_sets.items()):
        for subject, ckpt_path in sorted(ckpts.items()):
            manifest_path = manifests.get(subject)
            if manifest_path is None:
                print(f"[{model_name}:{subject}] skipped=no_manifest")
                continue
            print(f"[{model_name}:{subject}] checkpoint={ckpt_path}")
            state = normalize_state_dict(torch.load(ckpt_path, map_location="cpu"))
            model = build_model_from_state(
                state,
                watch_backbone=args.watch_backbone,
                correction_scale_init=args.correction_scale_init,
                device=args.device,
            )
            val_loader = build_loader(manifest_path, args.dataset_root, "val", include_sessions, args)
            test_loader = build_loader(manifest_path, args.dataset_root, "test", include_sessions, args)
            if len(val_loader.dataset) == 0:
                print(f"[{model_name}:{subject}] val split empty; using test split for threshold diagnostics")
                val_loader = test_loader

            val_frame = collect_frame(
                model,
                val_loader,
                model_name=model_name,
                subject=subject,
                split="val",
                device=args.device,
                pin_memory=args.pin_memory,
                baseline_reference=args.baseline_reference,
            )
            thresholds = thresholds_from_val_frame(val_frame, args.monitor)
            test_frame = collect_frame(
                model,
                test_loader,
                model_name=model_name,
                subject=subject,
                split="test",
                device=args.device,
                pin_memory=args.pin_memory,
                baseline_reference=args.baseline_reference,
            )
            test_frame = add_predictions(test_frame, thresholds)
            summary = summarize_outputs(test_frame, thresholds)
            all_windows.append(test_frame)
            all_summary.append(summary)
            deploy_row = summary[summary["output"] == "deploy"]
            if not deploy_row.empty:
                row = deploy_row.iloc[0]
                print(
                    f"  deploy threshold={row['threshold']:.4f} "
                    f"balanced_acc={row['balanced_acc']:.4f} "
                    f"auroc={row['auroc']:.4f} "
                    f"f1={row['f1']:.4f} "
                    f"positive_rate={row['positive_rate']:.4f}"
                )

    if not all_windows:
        raise ValueError("No checkpoint windows were analyzed.")

    windows = pd.concat(all_windows, axis=0, ignore_index=True)
    summary = pd.concat(all_summary, axis=0, ignore_index=True)
    bins = summarize_failure_bins(windows)

    windows_path = args.output_dir / "galaxy_model_failure_windows.csv"
    summary_path = args.output_dir / "galaxy_model_failure_summary.csv"
    bins_path = args.output_dir / "galaxy_model_failure_bins.csv"
    hard_cases_path = args.output_dir / "galaxy_model_failure_hard_cases.csv"

    hard_cases = windows.copy()
    if "deploy_correct" in hard_cases.columns:
        hard_cases = hard_cases[hard_cases["deploy_correct"] == 0]
        sort_cols = [col for col in ["deploy_abs_shift_from_base", "acc_jerk_rms", "ppg_noise_ratio_4_8"] if col in hard_cases.columns]
        if sort_cols:
            hard_cases = hard_cases.sort_values(sort_cols, ascending=[False] * len(sort_cols))

    windows.to_csv(windows_path, index=False)
    summary.to_csv(summary_path, index=False)
    bins.to_csv(bins_path, index=False)
    hard_cases.to_csv(hard_cases_path, index=False)

    print()
    print("Per-fold output summary:")
    print(summary.to_string(index=False))
    print()
    print(f"Saved per-window failure diagnostics to {windows_path}")
    print(f"Saved per-fold summary to {summary_path}")
    print(f"Saved quality/motion failure bins to {bins_path}")
    print(f"Saved misclassified hard cases to {hard_cases_path}")


if __name__ == "__main__":
    main()
