from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .reliability import cross_calibrated_trust, trust_weighted_kl_loss


DEFAULT_GALAXY_CALM_SESSIONS = ["baseline"]
DEFAULT_GALAXY_STRESS_SESSIONS = ["tsst-prep"]
DEFAULT_WESAD_CALM_SESSIONS = ["baseline"]
DEFAULT_WESAD_STRESS_SESSIONS = ["stress"]


class ScaledMotionFiLM(nn.Module):
    """Historical learnable-strength MotionFiLM used by scale_logit checkpoints."""

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


def set_scaled_motion_compat(enabled: bool) -> None:
    from . import galaxy_models

    if not hasattr(set_scaled_motion_compat, "_original_motion_film"):
        setattr(set_scaled_motion_compat, "_original_motion_film", galaxy_models.MotionFiLM)
    original_motion_film = getattr(set_scaled_motion_compat, "_original_motion_film")
    galaxy_models.MotionFiLM = ScaledMotionFiLM if enabled else original_motion_film


def distillation_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    quality: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
    teacher_probs = F.softmax(teacher_logits.detach() / temperature, dim=1)
    kl = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=1)
    weights = 0.35 + 0.65 * quality.squeeze(1).clamp(0.0, 1.0)
    return (kl * weights).sum() / weights.sum().clamp(min=1e-6) * (temperature ** 2)


def apply_min_trust_floor(trust: torch.Tensor, min_weight: float) -> torch.Tensor:
    trust = trust.detach().float().clamp(0.0, 1.0)
    if min_weight <= 0.0:
        return trust
    floor = max(0.0, min(float(min_weight), 1.0))
    return floor + (1.0 - floor) * trust


def discover_manifests(manifests_dir: Path, dataset_kind: str, subjects: Iterable[str] | None) -> list[tuple[str, Path]]:
    requested = {str(subject).strip() for subject in subjects or [] if str(subject).strip()}
    prefix = "galaxy" if dataset_kind == "galaxy" else "wesad"
    manifests: dict[str, Path] = {}
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
        checkpoint_dir / f"{subject}_deploy_watch.pt",
        checkpoint_dir / f"{subject.upper()}_deploy_watch.pt",
        checkpoint_dir / f"{subject.lower()}_deploy_watch.pt",
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


def has_key_prefix(state: dict[str, torch.Tensor], prefix: str) -> bool:
    return any(key.startswith(prefix) for key in state)


def infer_watch_enhancement(state: dict[str, torch.Tensor]) -> str:
    return "motion_disentangled" if has_key_fragment(state, "ppg_enhancer") else "none"


def infer_motion_mode(state: dict[str, torch.Tensor]) -> str:
    if has_key_fragment(state, "scale_logit"):
        return "scaled"
    if has_key_fragment(state, "residual_scale"):
        return "residual"
    return "strong"


def infer_correction_enabled(state: dict[str, torch.Tensor]) -> bool:
    return has_key_prefix(state, "deploy_correction.") or has_key_prefix(state, "deploy_correction_alpha.")


def infer_correction_mode(state: dict[str, torch.Tensor], fallback: str) -> str:
    value = state.get("correction_mode_id")
    if value is None:
        return fallback
    try:
        return "margin_residual" if int(value.item()) == 1 else "logit_mix"
    except Exception:
        return fallback


def infer_num_phenotypes(state: dict[str, torch.Tensor]) -> int:
    weight = state.get("phenotype_router.weight")
    return int(weight.shape[0]) if torch.is_tensor(weight) and weight.ndim == 2 else 0


def build_dataset(
    dataset_kind: str,
    manifest: Path,
    split: str,
    dataset_root: Path,
    include_sessions: list[str],
    cache_subjects: int,
    baseline_reference: bool,
    wavelet: str,
    wavelet_level: int,
):
    if dataset_kind == "galaxy":
        from .galaxy_dataset import GalaxyPrivilegedWindowDataset

        return GalaxyPrivilegedWindowDataset(
            manifest_csv=manifest,
            split=split,
            dataset_root=dataset_root,
            include_sessions=include_sessions,
            cache_tables=True,
            baseline_reference=baseline_reference,
            wavelet=wavelet,
            wavelet_level=wavelet_level,
        )
    if dataset_kind == "wesad":
        from .wesad_dataset import WESADPrivilegedWindowDataset

        return WESADPrivilegedWindowDataset(
            manifest_csv=manifest,
            split=split,
            wesad_root=dataset_root,
            include_sessions=include_sessions,
            cache_subjects=cache_subjects,
            baseline_reference=baseline_reference,
            wavelet=wavelet,
            wavelet_level=wavelet_level,
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


def build_model_from_state(
    dataset_kind: str,
    state: dict[str, torch.Tensor],
    sample: dict[str, object],
    args: argparse.Namespace,
) -> nn.Module:
    set_scaled_motion_compat(has_key_fragment(state, "scale_logit"))
    watch_enhancement = infer_watch_enhancement(state)
    watch_motion_mode = infer_motion_mode(state)
    correction_enabled = infer_correction_enabled(state)
    correction_mode = infer_correction_mode(state, args.correction_mode)

    if dataset_kind == "galaxy":
        from .galaxy_models_adaptive_correction import AdaptiveCorrectionGalaxyTeacherNet

        model = AdaptiveCorrectionGalaxyTeacherNet(
            num_phenotypes=infer_num_phenotypes(state),
            watch_backbone=args.watch_backbone,
            use_reliability_head=has_key_prefix(state, "reliability_head."),
            use_projection_heads=has_key_prefix(state, "watch_projector.") or has_key_prefix(state, "e4_projector."),
            use_e4_classifier=has_key_prefix(state, "e4_classifier."),
            use_rhythm_heads=has_key_prefix(state, "rhythm_head.") or has_key_prefix(state, "teacher_rhythm_head."),
            use_wavelet_head=has_key_prefix(state, "wavelet_predictor."),
            use_teacher_fused_classifier=has_key_prefix(state, "teacher_fused_classifier."),
            use_student_gated_correction=correction_enabled,
            correction_scale_init=args.correction_scale_init,
            correction_alpha_init_bias=args.correction_alpha_init_bias,
            correction_alpha_max=args.correction_alpha_max,
            correction_mode=correction_mode,
            watch_enhancement=watch_enhancement,
            watch_motion_mode=watch_motion_mode,
        )
    else:
        from .wesad_models_safe_sgpc import WESADPrivilegedTeacherNet

        wavelet_dim = int(sample["wavelet_features"].shape[0])
        privileged_channels = int(sample["privileged_signal"].shape[0])
        model = WESADPrivilegedTeacherNet(
            wavelet_dim=wavelet_dim,
            privileged_channels=privileged_channels,
            watch_backbone=args.watch_backbone,
            embed_dim=args.watch_embed_dim,
            align_dim=args.align_proj_dim,
            model_dim=args.watch_model_dim,
            transformer_layers=args.watch_transformer_layers,
            transformer_heads=args.watch_transformer_heads,
            fusion_hidden_dim=args.watch_fusion_hidden_dim,
            watch_enhancement=watch_enhancement,
            watch_motion_mode=watch_motion_mode,
            use_student_gated_correction=correction_enabled,
            correction_scale_init=args.correction_scale_init,
            correction_alpha_init_bias=args.correction_alpha_init_bias,
            correction_alpha_max=args.correction_alpha_max,
            correction_mode=correction_mode,
        )

    result = model.load_state_dict(state, strict=False)
    missing = list(result.missing_keys)
    unexpected = list(result.unexpected_keys)
    allowed_missing_suffixes = ("num_batches_tracked",)
    critical_missing = [key for key in missing if not key.endswith(allowed_missing_suffixes)]
    if critical_missing:
        print(f"load_warning=missing_keys count={len(critical_missing)} first={critical_missing[:5]}")
    if unexpected:
        print(f"load_note=unexpected_keys count={len(unexpected)} first={unexpected[:5]}")
    return model.to(args.device).eval()


def batch_to_device(
    dataset_kind: str,
    batch: dict[str, object],
    device: str,
    pin_memory: bool,
    baseline_reference: bool,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    watch_signal = batch["watch_signal"].to(device, non_blocking=pin_memory)
    wavelet = batch["wavelet_features"].to(device, non_blocking=pin_memory)
    quality = batch["watch_quality"].to(device, non_blocking=pin_memory)
    labels = batch["label"].to(device, non_blocking=pin_memory).long()
    kwargs: dict[str, torch.Tensor] = {
        "watch_signal": watch_signal,
        "wavelet_features": wavelet,
        "quality": quality,
    }
    if dataset_kind == "galaxy":
        kwargs["e4_signal"] = batch["e4_signal"].to(device, non_blocking=pin_memory)
    else:
        kwargs["privileged_signal"] = batch["privileged_signal"].to(device, non_blocking=pin_memory)
    if baseline_reference:
        kwargs.update(
            {
                "baseline_watch_signal": batch["baseline_watch_signal"].to(device, non_blocking=pin_memory),
                "baseline_wavelet_features": batch["baseline_wavelet_features"].to(device, non_blocking=pin_memory),
                "baseline_quality": batch["baseline_watch_quality"].to(device, non_blocking=pin_memory),
            }
        )
    return kwargs, labels


def forward_privileged(model: nn.Module, dataset_kind: str, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if dataset_kind == "galaxy":
        return model(
            inputs["watch_signal"],
            inputs["wavelet_features"],
            inputs["quality"],
            e4_signal=inputs["e4_signal"],
            baseline_watch_signal=inputs.get("baseline_watch_signal"),
            baseline_wavelet_features=inputs.get("baseline_wavelet_features"),
            baseline_quality=inputs.get("baseline_quality"),
        )
    return model(
        inputs["watch_signal"],
        inputs["wavelet_features"],
        inputs["quality"],
        privileged_signal=inputs["privileged_signal"],
        baseline_watch_signal=inputs.get("baseline_watch_signal"),
        baseline_wavelet_features=inputs.get("baseline_wavelet_features"),
        baseline_quality=inputs.get("baseline_quality"),
    )


def safe_auroc(labels: list[int], probs: list[float]) -> float:
    if len(set(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, probs))


def classifier_parameters(model: nn.Module, selector: str) -> list[tuple[str, nn.Parameter]]:
    if selector == "watch_classifier":
        prefixes = ("watch_classifier.",)
    elif selector == "watch_head":
        prefixes = ("watch_classifier.", "deploy_correction", "correction_norm", "correction_scale")
    elif selector == "watch_path":
        prefixes = ("watch_encoder.", "watch_classifier.", "deploy_correction", "correction_norm", "correction_scale")
    else:
        raise ValueError(f"Unsupported gradient parameter selector: {selector}")
    selected = [(name, param) for name, param in model.named_parameters() if name.startswith(prefixes)]
    if not selected:
        raise ValueError(f"No parameters matched selector {selector}")
    return selected


def freeze_except(model: nn.Module, selected: list[tuple[str, nn.Parameter]]) -> None:
    selected_ids = {id(param) for _, param in selected}
    for param in model.parameters():
        param.requires_grad_(id(param) in selected_ids)


def flatten_grads(selected: list[tuple[str, nn.Parameter]]) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    for _, param in selected:
        if param.grad is None:
            chunks.append(torch.zeros_like(param).reshape(-1))
        else:
            chunks.append(param.grad.detach().reshape(-1))
    if not chunks:
        raise ValueError("No selected parameters for gradient flattening.")
    return torch.cat(chunks)


def grad_for_loss(model: nn.Module, selected: list[tuple[str, nn.Parameter]], loss: torch.Tensor, retain_graph: bool) -> torch.Tensor:
    model.zero_grad(set_to_none=True)
    loss.backward(retain_graph=retain_graph)
    return flatten_grads(selected)


def cosine_or_nan(left: torch.Tensor, right: torch.Tensor) -> float:
    left_norm = torch.linalg.vector_norm(left)
    right_norm = torch.linalg.vector_norm(right)
    if float(left_norm.detach().cpu()) <= 1e-12 or float(right_norm.detach().cpu()) <= 1e-12:
        return float("nan")
    return float(F.cosine_similarity(left, right, dim=0).detach().cpu().item())


def mean_or_nan(values: list[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else float("nan")


def collect_fold_diagnostics(
    dataset_kind: str,
    subject: str,
    model: nn.Module,
    loader: DataLoader,
    args: argparse.Namespace,
) -> tuple[dict[str, object], list[dict[str, object]], pd.DataFrame]:
    selected = classifier_parameters(model, args.gradient_params)
    freeze_except(model, selected)
    model.eval()

    all_labels: list[int] = []
    student_probs: list[float] = []
    base_probs: list[float] = []
    teacher_probs: list[float] = []
    sample_rows: list[dict[str, object]] = []
    batch_rows: list[dict[str, object]] = []

    for batch_index, batch in enumerate(tqdm(loader, desc=f"{dataset_kind}:{subject}:kd-diagnosis", leave=False)):
        if args.max_batches is not None and batch_index >= args.max_batches:
            break
        inputs, labels = batch_to_device(dataset_kind, batch, args.device, args.pin_memory, args.baseline_reference)
        out = forward_privileged(model, dataset_kind, inputs)
        if "teacher_logits" not in out:
            raise RuntimeError("Checkpoint/model did not produce teacher_logits; use a privileged checkpoint.")
        student_logits = out[args.student_logits]
        base_logits = out.get("base_logits", student_logits)
        teacher_logits = out["teacher_logits"]
        quality = inputs["quality"]

        with torch.no_grad():
            trust_signal = cross_calibrated_trust(student_logits, teacher_logits, labels, quality)
            gated_trust = apply_min_trust_floor(trust_signal, min_weight=args.cross_confidence_min_weight)

        task_loss = F.cross_entropy(student_logits, labels)
        uniform_kd_loss = distillation_kl_loss(
            student_logits,
            teacher_logits.detach(),
            quality=quality,
            temperature=args.distill_temp,
        )
        gated_kd_loss = trust_weighted_kl_loss(
            student_logits,
            teacher_logits.detach(),
            gated_trust,
            temperature=args.distill_temp,
        )

        task_grad = grad_for_loss(model, selected, task_loss, retain_graph=True)
        uniform_grad = grad_for_loss(model, selected, uniform_kd_loss, retain_graph=True)
        gated_grad = grad_for_loss(model, selected, gated_kd_loss, retain_graph=False)

        labels_cpu = labels.detach().cpu().numpy().astype(int)
        student_prob_cpu = torch.softmax(student_logits.detach(), dim=1)[:, 1].cpu().numpy()
        base_prob_cpu = torch.softmax(base_logits.detach(), dim=1)[:, 1].cpu().numpy()
        teacher_prob_cpu = torch.softmax(teacher_logits.detach(), dim=1)[:, 1].cpu().numpy()
        trust_cpu = trust_signal.detach().cpu().numpy()
        gated_trust_cpu = gated_trust.detach().cpu().numpy()
        quality_cpu = quality.detach().cpu().reshape(len(labels_cpu), -1)[:, 0].numpy()

        batch_rows.append(
            {
                "dataset": dataset_kind,
                "fold_subject": subject,
                "batch_index": batch_index,
                "n": int(len(labels_cpu)),
                "label_rate": float(np.mean(labels_cpu)),
                "task_loss": float(task_loss.detach().cpu().item()),
                "uniform_kd_loss": float(uniform_kd_loss.detach().cpu().item()),
                "gated_kd_loss": float(gated_kd_loss.detach().cpu().item()),
                "uniform_grad_cosine": cosine_or_nan(task_grad, uniform_grad),
                "gated_grad_cosine": cosine_or_nan(task_grad, gated_grad),
                "trust_mean": float(np.mean(trust_cpu)),
                "trust_floor_mean": float(np.mean(gated_trust_cpu)),
                "trust_high_frac": float(np.mean(trust_cpu >= 0.5)),
                "quality_mean": float(np.mean(quality_cpu)),
                "teacher_student_prob_corr": float(np.corrcoef(teacher_prob_cpu, student_prob_cpu)[0, 1])
                if len(labels_cpu) > 1 and np.std(teacher_prob_cpu) > 1e-12 and np.std(student_prob_cpu) > 1e-12
                else float("nan"),
            }
        )

        for idx in range(len(labels_cpu)):
            row = {
                "dataset": dataset_kind,
                "fold_subject": subject,
                "subject_id": str(batch["subject_id"][idx]),
                "session": str(batch["session"][idx]),
                "window_start_ms": int(batch["window_start_ms"][idx]),
                "window_end_ms": int(batch["window_end_ms"][idx]),
                "label": int(labels_cpu[idx]),
                "student_prob": float(student_prob_cpu[idx]),
                "base_prob": float(base_prob_cpu[idx]),
                "teacher_prob": float(teacher_prob_cpu[idx]),
                "trust": float(trust_cpu[idx]),
                "trust_floor": float(gated_trust_cpu[idx]),
                "quality": float(quality_cpu[idx]),
                "teacher_correct": int((teacher_prob_cpu[idx] >= 0.5) == bool(labels_cpu[idx])),
                "student_correct": int((student_prob_cpu[idx] >= 0.5) == bool(labels_cpu[idx])),
                "teacher_student_disagree": int((teacher_prob_cpu[idx] >= 0.5) != (student_prob_cpu[idx] >= 0.5)),
            }
            sample_rows.append(row)
        all_labels.extend(labels_cpu.tolist())
        student_probs.extend(student_prob_cpu.tolist())
        base_probs.extend(base_prob_cpu.tolist())
        teacher_probs.extend(teacher_prob_cpu.tolist())

    student_auroc = safe_auroc(all_labels, student_probs)
    base_auroc = safe_auroc(all_labels, base_probs)
    teacher_auroc = safe_auroc(all_labels, teacher_probs)
    uniform_cos = mean_or_nan([row["uniform_grad_cosine"] for row in batch_rows])
    gated_cos = mean_or_nan([row["gated_grad_cosine"] for row in batch_rows])
    trust_mean = mean_or_nan([row["trust_mean"] for row in batch_rows])
    trust_floor_mean = mean_or_nan([row["trust_floor_mean"] for row in batch_rows])
    teacher_student_corr = mean_or_nan([row["teacher_student_prob_corr"] for row in batch_rows])
    teacher_advantage = teacher_auroc - student_auroc if math.isfinite(teacher_auroc) and math.isfinite(student_auroc) else float("nan")
    gated_cos_gain = gated_cos - uniform_cos if math.isfinite(gated_cos) and math.isfinite(uniform_cos) else float("nan")

    if math.isfinite(teacher_advantage) and teacher_advantage > args.auroc_margin and math.isfinite(uniform_cos) and uniform_cos >= args.cosine_margin:
        suggested_regime = "standard_kd"
    elif math.isfinite(gated_cos) and gated_cos >= args.cosine_margin and (
        not math.isfinite(uniform_cos) or gated_cos >= uniform_cos + args.gated_cosine_margin
    ):
        suggested_regime = "gated_kd"
    else:
        suggested_regime = "watch_or_no_kd"

    fold_row: dict[str, object] = {
        "dataset": dataset_kind,
        "fold_subject": subject,
        "n": int(len(all_labels)),
        "label_rate": float(np.mean(all_labels)) if all_labels else float("nan"),
        "student_auroc": student_auroc,
        "base_auroc": base_auroc,
        "teacher_auroc": teacher_auroc,
        "teacher_minus_student_auroc": teacher_advantage,
        "uniform_grad_cosine_mean": uniform_cos,
        "gated_grad_cosine_mean": gated_cos,
        "gated_minus_uniform_grad_cosine": gated_cos_gain,
        "trust_mean": trust_mean,
        "trust_floor_mean": trust_floor_mean,
        "trust_high_frac_mean": mean_or_nan([row["trust_high_frac"] for row in batch_rows]),
        "teacher_student_prob_corr_mean": teacher_student_corr,
        "teacher_correct_rate_05": float(np.mean([row["teacher_correct"] for row in sample_rows])) if sample_rows else float("nan"),
        "student_correct_rate_05": float(np.mean([row["student_correct"] for row in sample_rows])) if sample_rows else float("nan"),
        "teacher_student_disagree_rate_05": float(np.mean([row["teacher_student_disagree"] for row in sample_rows]))
        if sample_rows
        else float("nan"),
        "suggested_regime_proxy": suggested_regime,
        "student_logits_key": args.student_logits,
        "gradient_params": args.gradient_params,
    }
    return fold_row, batch_rows, pd.DataFrame(sample_rows)


def write_summary(fold_frame: pd.DataFrame, args: argparse.Namespace) -> None:
    lines: list[str] = []
    lines.append("# KD Regime Diagnosis")
    lines.append("")
    lines.append(f"- Dataset: {args.dataset_kind}")
    lines.append(f"- Split: {args.split}")
    lines.append(f"- Folds: {len(fold_frame)}")
    lines.append(f"- Student logits: {args.student_logits}")
    lines.append(f"- Gradient parameters: {args.gradient_params}")
    lines.append(f"- Distill temperature: {args.distill_temp}")
    lines.append(f"- Cross-confidence min weight: {args.cross_confidence_min_weight}")
    lines.append("")

    if not fold_frame.empty:
        for col in [
            "student_auroc",
            "teacher_auroc",
            "teacher_minus_student_auroc",
            "uniform_grad_cosine_mean",
            "gated_grad_cosine_mean",
            "gated_minus_uniform_grad_cosine",
            "trust_mean",
            "teacher_student_disagree_rate_05",
        ]:
            values = pd.to_numeric(fold_frame[col], errors="coerce")
            lines.append(f"- {col}: mean={values.mean():.4f} std={values.std(ddof=1):.4f}")
        lines.append("")
        lines.append("## Regime Proxy Counts")
        counts = fold_frame["suggested_regime_proxy"].value_counts(dropna=False)
        for regime, count in counts.items():
            lines.append(f"- {regime}: {int(count)}")
        lines.append("")
        lines.append("## Reading Guide")
        lines.append(
            "A useful pre-training selector should show teacher_minus_student_auroc > 0 for strong-teacher folds, "
            "and gated_grad_cosine_mean should be higher than uniform_grad_cosine_mean when the teacher is weak or conflicting."
        )
        lines.append(
            "Treat this as a diagnostic pilot: it uses existing trained checkpoints to test whether the AUROC + gradient signal is plausible, "
            "not as final evidence that a regime can be selected without any warm-up."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"{args.dataset_kind}_kd_regime_diagnosis_summary.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose whether teacher AUROC and KD/task gradient compatibility can separate standard KD from gated KD regimes."
    )
    parser.add_argument("--dataset-kind", choices=["galaxy", "wesad"], required=True)
    parser.add_argument("--manifests-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cache-subjects", type=int, default=4)
    parser.add_argument("--calm-sessions", nargs="*", default=None)
    parser.add_argument("--stress-sessions", nargs="*", default=None)
    parser.add_argument("--wavelet", type=str, default="db4")
    parser.add_argument("--wavelet-level", type=int, default=4)
    parser.add_argument("--baseline-reference", action="store_true")
    parser.add_argument("--distill-temp", type=float, default=4.0)
    parser.add_argument("--cross-confidence-min-weight", type=float, default=0.0)
    parser.add_argument("--student-logits", choices=["logits", "base_logits"], default="logits")
    parser.add_argument("--gradient-params", choices=["watch_classifier", "watch_head", "watch_path"], default="watch_classifier")
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--auroc-margin", type=float, default=0.01)
    parser.add_argument("--cosine-margin", type=float, default=0.0)
    parser.add_argument("--gated-cosine-margin", type=float, default=0.02)
    parser.add_argument("--watch-backbone", choices=["wavelet_guided", "resnet18_1d", "resnet34_1d", "resnet50_1d"], default="wavelet_guided")
    parser.add_argument("--watch-model-dim", type=int, default=192)
    parser.add_argument("--watch-transformer-layers", type=int, default=2)
    parser.add_argument("--watch-transformer-heads", type=int, default=4)
    parser.add_argument("--watch-fusion-hidden-dim", type=int, default=256)
    parser.add_argument("--watch-embed-dim", type=int, default=160)
    parser.add_argument("--align-proj-dim", type=int, default=128)
    parser.add_argument("--correction-scale-init", type=float, default=0.05)
    parser.add_argument("--correction-alpha-init-bias", type=float, default=-1.5)
    parser.add_argument("--correction-alpha-max", type=float, default=1.0)
    parser.add_argument("--correction-mode", choices=["logit_mix", "margin_residual"], default="margin_residual")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dataset_kind == "galaxy":
        calm_sessions = args.calm_sessions or DEFAULT_GALAXY_CALM_SESSIONS
        stress_sessions = args.stress_sessions or DEFAULT_GALAXY_STRESS_SESSIONS
    else:
        calm_sessions = args.calm_sessions or DEFAULT_WESAD_CALM_SESSIONS
        stress_sessions = args.stress_sessions or DEFAULT_WESAD_STRESS_SESSIONS
    include_sessions = [str(item) for item in calm_sessions + stress_sessions]

    manifests = discover_manifests(args.manifests_dir, args.dataset_kind, args.subjects)
    if not manifests:
        raise ValueError(f"No LOSO manifests found in {args.manifests_dir}")

    fold_rows: list[dict[str, object]] = []
    batch_rows: list[dict[str, object]] = []
    sample_frames: list[pd.DataFrame] = []
    for subject, manifest in manifests:
        checkpoint = find_checkpoint(args.checkpoint_dir, subject)
        print(f"diagnosing dataset={args.dataset_kind} subject={subject} checkpoint={checkpoint}")
        state = load_state(checkpoint)
        dataset = build_dataset(
            args.dataset_kind,
            manifest,
            args.split,
            args.dataset_root,
            include_sessions,
            args.cache_subjects,
            args.baseline_reference,
            args.wavelet,
            args.wavelet_level,
        )
        if len(dataset) == 0:
            print(f"skip_subject={subject} reason=empty_{args.split}_split")
            continue
        model = build_model_from_state(args.dataset_kind, state, dataset[0], args)
        loader = build_loader(dataset, args.batch_size, args.num_workers, args.pin_memory)
        fold_row, fold_batch_rows, sample_frame = collect_fold_diagnostics(args.dataset_kind, subject, model, loader, args)
        fold_rows.append(fold_row)
        batch_rows.extend(fold_batch_rows)
        sample_frames.append(sample_frame)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fold_frame = pd.DataFrame(fold_rows)
    batch_frame = pd.DataFrame(batch_rows)
    sample_frame = pd.concat(sample_frames, ignore_index=True) if sample_frames else pd.DataFrame()
    fold_path = args.output_dir / f"{args.dataset_kind}_kd_regime_fold_diagnostics.csv"
    batch_path = args.output_dir / f"{args.dataset_kind}_kd_regime_batch_gradients.csv"
    sample_path = args.output_dir / f"{args.dataset_kind}_kd_regime_window_predictions.csv"
    fold_frame.to_csv(fold_path, index=False)
    batch_frame.to_csv(batch_path, index=False)
    sample_frame.to_csv(sample_path, index=False)
    write_summary(fold_frame, args)
    print(f"saved_fold_diagnostics={fold_path}")
    print(f"saved_batch_gradients={batch_path}")
    print(f"saved_window_predictions={sample_path}")


if __name__ == "__main__":
    main()
