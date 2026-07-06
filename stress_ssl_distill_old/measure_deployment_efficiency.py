from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from . import galaxy_models as galaxy_models_module
from .galaxy_models import PrivilegedGalaxyTeacherNet, WaveletGuidedWatchNet
from .galaxy_models_adaptive_correction import AdaptiveCorrectionGalaxyTeacherNet
from .wesad_models import WESADPrivilegedTeacherNet as WESADPrivilegedTeacherNetLegacy
from .wesad_models_safe_sgpc import WESADPrivilegedTeacherNet as WESADPrivilegedTeacherNetAdaptive


BYTES_PER_FLOAT32 = 4
MB = 1024 * 1024
KB = 1024


ORIGINAL_MOTION_FILM = galaxy_models_module.MotionFiLM


class ScaledMotionFiLM(nn.Module):
    """Historical learnable-strength motion FiLM used by scale_logit checkpoints."""

    def __init__(
        self,
        cond_dim: int,
        target_dim: int,
        *args: object,
        scale_logit_init: float = -2.0,
        **kwargs: object,
    ) -> None:
        _ = (args, kwargs)
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


def install_motion_film_for_state(state: dict[str, torch.Tensor], prefix: str = "") -> None:
    if has_prefix(state, f"{prefix}motion_gates.0.scale_logit"):
        galaxy_models_module.MotionFiLM = ScaledMotionFiLM
    else:
        galaxy_models_module.MotionFiLM = ORIGINAL_MOTION_FILM


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
    value = state.get(f"{prefix}wavelet_mlp.0.weight")
    if value is not None and value.ndim == 2:
        return int(value.shape[1] - 1)
    return 4


def infer_model_dim(state: dict[str, torch.Tensor], prefix: str = "") -> int:
    value = state.get(f"{prefix}cls_token")
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
    value = state.get(f"{prefix}fusion.0.weight")
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


def parse_named_path(value: str, default_name: str | None = None) -> tuple[str, Path]:
    if "=" in value:
        name, path = value.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Missing method name in named path: {value}")
        return name, Path(path)
    path = Path(value)
    return default_name or path.stem, path


def checkpoint_subject_sort_key(path: Path) -> tuple[int, str]:
    stem = path.stem
    token = stem.replace("_deploy_watch", "").replace("_watch_only", "")
    token = token.replace("Sub", "").replace("S", "").replace("P", "")
    try:
        return int(token), path.name
    except ValueError:
        return 10_000, path.name


def discover_checkpoints(args: argparse.Namespace) -> list[tuple[str, Path]]:
    checkpoints: list[tuple[str, Path]] = []
    for item in args.checkpoint or []:
        checkpoints.append(parse_named_path(item))

    for item in args.checkpoint_dir or []:
        name, path = parse_named_path(item, default_name=Path(item).stem)
        candidates = sorted(path.glob("*.pt"), key=checkpoint_subject_sort_key)
        if not candidates and (path / "checkpoints").exists():
            candidates = sorted((path / "checkpoints").glob("*.pt"), key=checkpoint_subject_sort_key)
        if not candidates:
            raise ValueError(f"No .pt checkpoints found in {path} or {path / 'checkpoints'}")
        if args.subjects:
            requested = {subject.strip() for subject in args.subjects if subject.strip()}
            candidates = [ckpt for ckpt in candidates if ckpt.stem.split("_", 1)[0] in requested or ckpt.stem in requested]
        if args.max_checkpoints_per_set > 0:
            candidates = candidates[: args.max_checkpoints_per_set]
        checkpoints.extend((name, ckpt) for ckpt in candidates)

    if not checkpoints:
        raise ValueError("Provide --checkpoint NAME=PATH and/or --checkpoint-dir NAME=DIR.")
    return checkpoints


def count_params(model: torch.nn.Module) -> int:
    return int(sum(param.numel() for param in model.parameters()))


def include_watch_encoder_deploy_param(name: str) -> bool:
    training_only_prefixes = (
        "baseline_ref_adapter.",
        "baseline_ref_gate.",
        "contrastive_head.",
        "wavelet_predictor.",
    )
    return not name.startswith(training_only_prefixes)


def include_privileged_deploy_param(name: str) -> bool:
    if name.startswith("watch_encoder."):
        return include_watch_encoder_deploy_param(name[len("watch_encoder.") :])
    if name.startswith("watch_classifier."):
        return True
    if name.startswith(("deploy_correction.", "deploy_correction_gate.", "deploy_correction_alpha.", "correction_norm.")):
        return True
    if name == "correction_scale":
        return True
    return False


def count_deploy_params(model: torch.nn.Module) -> int:
    if isinstance(model, WaveletGuidedWatchNet):
        return int(
            sum(param.numel() for name, param in model.named_parameters() if include_watch_encoder_deploy_param(name))
        )
    if hasattr(model, "watch_encoder") and hasattr(model, "watch_classifier"):
        return int(sum(param.numel() for name, param in model.named_parameters() if include_privileged_deploy_param(name)))
    return count_params(model)


def infer_watch_enhancement(state: dict[str, torch.Tensor], prefix: str = "") -> str:
    return "motion_disentangled" if has_prefix(state, f"{prefix}ppg_enhancer.") else "none"


def infer_watch_motion_mode(state: dict[str, torch.Tensor], prefix: str = "") -> str:
    if has_prefix(state, f"{prefix}motion_gates.0.scale_logit"):
        return "scaled"
    return "residual" if has_prefix(state, f"{prefix}motion_gates.0.residual_scale") else "strong"


def infer_correction_mode(state: dict[str, torch.Tensor]) -> str:
    value = state.get("correction_mode_id")
    if value is None:
        return "logit_mix"
    try:
        mode_id = int(value.reshape(-1)[0].item())
    except (IndexError, RuntimeError, TypeError, ValueError):
        return "logit_mix"
    return "margin_residual" if mode_id == 1 else "logit_mix"


def infer_transformer_heads_from_state(state: dict[str, torch.Tensor], prefix: str = "") -> int:
    # The current checkpoints all use 4 heads. Keep this explicit rather than guessing from packed QKV weights.
    _ = (state, prefix)
    return 4


def infer_privileged_channels(state: dict[str, torch.Tensor]) -> int:
    value = state.get("privileged_encoder.stem.0.weight")
    if value is not None and value.ndim == 3:
        return int(value.shape[1])
    return 9


def is_watch_only_state(state: dict[str, torch.Tensor]) -> bool:
    return has_prefix(state, "ppg_stem.", "acc_stem.") and not has_prefix(state, "watch_encoder.")


def build_watch_only_model(state: dict[str, torch.Tensor], device: str) -> WaveletGuidedWatchNet:
    install_motion_film_for_state(state)
    model = WaveletGuidedWatchNet(
        wavelet_dim=infer_wavelet_dim(state),
        model_dim=infer_model_dim(state),
        transformer_layers=infer_transformer_layers(state),
        transformer_heads=infer_transformer_heads_from_state(state),
        fusion_hidden_dim=infer_fusion_hidden_dim(state),
        embed_dim=infer_embed_dim(state),
        watch_enhancement=infer_watch_enhancement(state),
        watch_motion_mode=infer_watch_motion_mode(state),
    )
    result = model.load_state_dict(state, strict=False)
    if result.missing_keys:
        print(f"watch_load_missing_keys={result.missing_keys[:8]} total={len(result.missing_keys)}")
    if result.unexpected_keys:
        print(f"watch_load_unexpected_keys={result.unexpected_keys[:8]} total={len(result.unexpected_keys)}")
    model.to(device)
    model.eval()
    return model


def build_galaxy_privileged_model(
    state: dict[str, torch.Tensor],
    device: str,
    correction_scale_init: float,
    correction_alpha_init_bias: float = -3.0,
    correction_alpha_max: float = 1.0,
) -> torch.nn.Module:
    watch_prefix = "watch_encoder."
    install_motion_film_for_state(state, watch_prefix)
    kwargs = {
        "wavelet_dim": infer_wavelet_dim(state, watch_prefix),
        "embed_dim": infer_embed_dim(state),
        "num_phenotypes": infer_num_phenotypes(state),
        "watch_backbone": "wavelet_guided",
        "watch_enhancement": infer_watch_enhancement(state, watch_prefix),
        "watch_motion_mode": infer_watch_motion_mode(state, watch_prefix),
        "use_reliability_head": has_prefix(state, "reliability_head."),
        "use_projection_heads": has_prefix(state, "watch_projector.", "e4_projector."),
        "use_e4_classifier": has_prefix(state, "e4_classifier."),
        "use_rhythm_heads": has_prefix(state, "rhythm_head.", "teacher_rhythm_head."),
        "use_wavelet_head": has_prefix(state, "wavelet_predictor."),
        "use_teacher_fused_classifier": has_prefix(state, "teacher_fused_classifier."),
        "use_student_gated_correction": has_prefix(state, "deploy_correction.", "deploy_correction_gate."),
        "correction_scale_init": correction_scale_init,
    }
    cls: type[torch.nn.Module]
    if has_prefix(state, "deploy_correction_alpha."):
        cls = AdaptiveCorrectionGalaxyTeacherNet
        kwargs["correction_alpha_init_bias"] = correction_alpha_init_bias
        kwargs["correction_alpha_max"] = correction_alpha_max
        kwargs["correction_mode"] = infer_correction_mode(state)
    else:
        cls = PrivilegedGalaxyTeacherNet
    model = cls(**kwargs)
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
                "deploy_correction_alpha.",
                "privileged_correction.",
                "privileged_correction_gate.",
                "correction_norm.",
                "correction_scale",
                "correction_mode_id",
            )
        )
    ]
    if non_optional_missing:
        print(f"galaxy_load_missing_keys={non_optional_missing[:8]} total={len(non_optional_missing)}")
    if result.unexpected_keys:
        print(f"galaxy_load_unexpected_keys={result.unexpected_keys[:8]} total={len(result.unexpected_keys)}")
    model.to(device)
    model.eval()
    return model


def build_wesad_privileged_model(
    state: dict[str, torch.Tensor],
    device: str,
    correction_scale_init: float,
    correction_alpha_init_bias: float,
    correction_alpha_max: float,
) -> torch.nn.Module:
    watch_prefix = "watch_encoder."
    install_motion_film_for_state(state, watch_prefix)
    kwargs = {
        "wavelet_dim": infer_wavelet_dim(state, watch_prefix),
        "privileged_channels": infer_privileged_channels(state),
        "embed_dim": infer_embed_dim(state),
        "align_dim": 128,
        "watch_backbone": "wavelet_guided",
        "model_dim": infer_model_dim(state, watch_prefix),
        "transformer_layers": infer_transformer_layers(state, watch_prefix),
        "transformer_heads": infer_transformer_heads_from_state(state, watch_prefix),
        "fusion_hidden_dim": infer_fusion_hidden_dim(state, watch_prefix),
        "watch_enhancement": infer_watch_enhancement(state, watch_prefix),
        "watch_motion_mode": infer_watch_motion_mode(state, watch_prefix),
        "use_student_gated_correction": has_prefix(state, "deploy_correction.", "deploy_correction_gate."),
        "correction_scale_init": correction_scale_init,
    }
    if has_prefix(state, "deploy_correction_alpha."):
        model = WESADPrivilegedTeacherNetAdaptive(
            **kwargs,
            correction_alpha_init_bias=correction_alpha_init_bias,
            correction_alpha_max=correction_alpha_max,
            correction_mode=infer_correction_mode(state),
        )
    else:
        model = WESADPrivilegedTeacherNetLegacy(**kwargs)
    result = model.load_state_dict(state, strict=False)
    non_optional_missing = [
        key
        for key in result.missing_keys
        if not key.startswith(
            (
                "deploy_correction.",
                "deploy_correction_gate.",
                "deploy_correction_alpha.",
                "privileged_correction.",
                "privileged_correction_gate.",
                "correction_norm.",
                "correction_scale",
                "correction_mode_id",
            )
        )
    ]
    if non_optional_missing:
        print(f"wesad_load_missing_keys={non_optional_missing[:8]} total={len(non_optional_missing)}")
    if result.unexpected_keys:
        print(f"wesad_load_unexpected_keys={result.unexpected_keys[:8]} total={len(result.unexpected_keys)}")
    model.to(device)
    model.eval()
    return model


def build_model_from_state(state: dict[str, torch.Tensor], args: argparse.Namespace) -> torch.nn.Module:
    if is_watch_only_state(state):
        return build_watch_only_model(state, args.device)
    if args.dataset_kind == "galaxy":
        return build_galaxy_privileged_model(
            state,
            args.device,
            args.correction_scale_init,
            args.correction_alpha_init_bias,
            args.correction_alpha_max,
        )
    if args.dataset_kind == "wesad":
        return build_wesad_privileged_model(
            state,
            args.device,
            args.correction_scale_init,
            args.correction_alpha_init_bias,
            args.correction_alpha_max,
        )
    raise ValueError(f"Unsupported dataset kind: {args.dataset_kind}")


def infer_wavelet_dim_from_model(model: torch.nn.Module) -> int:
    if isinstance(model, WaveletGuidedWatchNet):
        return int(model.wavelet_mlp[0].weight.shape[1] - 1)
    encoder = getattr(model, "watch_encoder", None)
    if encoder is not None and hasattr(encoder, "wavelet_mlp"):
        return int(encoder.wavelet_mlp[0].weight.shape[1] - 1)
    return 4


def make_dummy_inputs(
    device: str,
    batch_size: int,
    watch_channels: int,
    watch_length: int,
    wavelet_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    signal = torch.randn(batch_size, watch_channels, watch_length, device=device)
    wavelet = torch.randn(batch_size, wavelet_dim, device=device)
    quality = torch.ones(batch_size, 1, device=device)
    return signal, wavelet, quality


def quality_column(model: torch.nn.Module, quality: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "_quality_column"):
        return model._quality_column(quality)  # noqa: SLF001
    if quality.ndim == 1:
        quality = quality.unsqueeze(1)
    return quality.float().clamp(0.0, 1.0)


def correction_scale_value(model: torch.nn.Module) -> torch.Tensor:
    if hasattr(model, "_correction_scale_value"):
        return model._correction_scale_value()  # noqa: SLF001
    correction_scale = getattr(model, "correction_scale", None)
    if correction_scale is None:
        raise RuntimeError("Correction scale requested while correction module is disabled.")
    return torch.tanh(correction_scale)


def deploy_forward_callable(
    model: torch.nn.Module,
    signal: torch.Tensor,
    wavelet: torch.Tensor,
    quality: torch.Tensor,
) -> Callable[[], torch.Tensor]:
    if isinstance(model, WaveletGuidedWatchNet):
        def forward_watch_only() -> torch.Tensor:
            out = model._encode_core(signal, wavelet, quality)  # noqa: SLF001
            return model.classifier(out["embedding"])

        return forward_watch_only

    if hasattr(model, "watch_encoder") and hasattr(model, "watch_classifier"):
        def forward_deploy_watch() -> torch.Tensor:
            watch_out = model.watch_encoder(signal, wavelet, quality)
            embedding = watch_out["embedding"]
            base_logits = model.watch_classifier(embedding)
            if not bool(getattr(model, "use_student_gated_correction", False)):
                return base_logits

            deploy_input = torch.cat([embedding, quality_column(model, quality)], dim=1)
            deploy_delta = model.deploy_correction(deploy_input)
            deploy_gate = model.deploy_correction_gate(deploy_input)
            deploy_embedding = model.correction_norm(
                embedding + correction_scale_value(model) * deploy_gate * deploy_delta
            )
            corrected_logits = model.watch_classifier(deploy_embedding)
            alpha_head = getattr(model, "deploy_correction_alpha", None)
            if alpha_head is None:
                return corrected_logits
            alpha = float(getattr(model, "correction_alpha_max", 1.0)) * torch.sigmoid(alpha_head(deploy_input))
            return base_logits + alpha * (corrected_logits - base_logits)

        return forward_deploy_watch

    def forward_generic() -> torch.Tensor:
        out = model(signal, wavelet, quality)
        if isinstance(out, dict) and "logits" in out:
            return out["logits"]
        if isinstance(out, torch.Tensor):
            return out
        raise TypeError(f"Unsupported model output type: {type(out)!r}")

    return forward_generic


def infer_deploy_forward_kind(model: torch.nn.Module) -> str:
    if isinstance(model, WaveletGuidedWatchNet):
        return "watch_only_direct"
    if hasattr(model, "watch_encoder") and hasattr(model, "watch_classifier"):
        if bool(getattr(model, "use_student_gated_correction", False)):
            has_alpha = hasattr(model, "deploy_correction_alpha") and getattr(model, "deploy_correction_alpha") is not None
            return "privileged_watch_with_adaptive_correction" if has_alpha else "privileged_watch_with_direct_correction"
        return "privileged_watch_no_correction"
    return "generic_model_forward"


def measure_latency(
    forward_fn: Callable[[], torch.Tensor],
    device: str,
    warmup: int,
    repeats: int,
) -> dict[str, float]:
    with torch.inference_mode():
        for _ in range(max(warmup, 0)):
            _ = forward_fn()
        if device.startswith("cuda"):
            torch.cuda.synchronize()
            timings: list[float] = []
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            for _ in range(repeats):
                starter.record()
                _ = forward_fn()
                ender.record()
                torch.cuda.synchronize()
                timings.append(float(starter.elapsed_time(ender)))
        else:
            timings = []
            for _ in range(repeats):
                start = time.perf_counter()
                _ = forward_fn()
                timings.append((time.perf_counter() - start) * 1000.0)
    values = np.asarray(timings, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
        "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def infer_architecture(model: torch.nn.Module, dataset_kind: str) -> dict[str, object]:
    watch_enhancement = getattr(model, "watch_enhancement", "")
    watch_motion_mode = getattr(model, "watch_motion_mode", "")
    use_correction = bool(getattr(model, "use_student_gated_correction", False))
    has_alpha = hasattr(model, "deploy_correction_alpha") and getattr(model, "deploy_correction_alpha") is not None
    if isinstance(model, WaveletGuidedWatchNet):
        family = "watch_only"
        use_correction = False
        has_alpha = False
    else:
        family = "privileged_deploy_watch"
    if dataset_kind == "galaxy":
        watch_input = "Galaxy Watch PPG + ACC + watch-derived wavelet/quality"
        privileged = "E4 BVP/ACC and Polar ECG/rhythm-derived cues during training only"
    else:
        watch_input = "wrist BVP + ACC + watch-derived wavelet/quality"
        privileged = "chest ACC/ECG/EMG/EDA/Temp/Resp during training only"
    return {
        "dataset": dataset_kind,
        "model_family": family,
        "watch_backbone": getattr(model, "watch_backbone", "wavelet_guided"),
        "watch_enhancement": watch_enhancement,
        "watch_motion_mode": watch_motion_mode,
        "deploy_forward_kind": infer_deploy_forward_kind(model),
        "deploy_correction_enabled": bool(use_correction),
        "adaptive_correction": bool(use_correction and has_alpha),
        "legacy_direct_correction": bool(use_correction and not has_alpha),
        "needs_privileged_sensors_at_inference": "no",
        "deployment_input_modality": watch_input,
        "privileged_training_signals": "none" if family == "watch_only" else privileged,
    }


def profile_checkpoint(method: str, checkpoint_path: Path, args: argparse.Namespace) -> dict[str, object]:
    state = normalize_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model = build_model_from_state(state, args)
    wavelet_dim = args.wavelet_dim if args.wavelet_dim > 0 else infer_wavelet_dim_from_model(model)
    signal, wavelet, quality = make_dummy_inputs(
        device=args.device,
        batch_size=args.batch_size,
        watch_channels=args.watch_channels,
        watch_length=args.watch_length,
        wavelet_dim=wavelet_dim,
    )
    forward_fn = deploy_forward_callable(model, signal, wavelet, quality)
    latency = measure_latency(
        forward_fn,
        device=args.device,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    total_params = count_params(model)
    deploy_params = count_deploy_params(model)
    input_values = args.watch_channels * args.watch_length + wavelet_dim + 1
    return {
        "method": method,
        "checkpoint": str(checkpoint_path),
        **infer_architecture(model, args.dataset_kind),
        "batch_size": args.batch_size,
        "watch_channels": args.watch_channels,
        "watch_length": args.watch_length,
        "wavelet_dim": wavelet_dim,
        "total_training_params": total_params,
        "deploy_params": deploy_params,
        "training_only_params": max(total_params - deploy_params, 0),
        "deploy_param_size_mb_fp32": deploy_params * BYTES_PER_FLOAT32 / MB,
        "deploy_param_size_mb_fp16": deploy_params * 2 / MB,
        "deploy_param_size_mb_int8": deploy_params / MB,
        "checkpoint_size_mb": checkpoint_path.stat().st_size / MB,
        "deploy_input_values_per_window": input_values,
        "deploy_input_size_kb_fp32": input_values * BYTES_PER_FLOAT32 / KB,
        "latency_ms_mean": latency["mean"],
        "latency_ms_std": latency["std"],
        "latency_ms_median": latency["median"],
        "latency_ms_p10": latency["p10"],
        "latency_ms_p90": latency["p90"],
        "latency_ms_min": latency["min"],
        "latency_ms_max": latency["max"],
        "throughput_windows_per_s": (args.batch_size * 1000.0 / latency["mean"]) if latency["mean"] > 0 else float("nan"),
        "throughput_windows_per_s_median": (args.batch_size * 1000.0 / latency["median"]) if latency["median"] > 0 else float("nan"),
        "device": args.device,
    }


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        "total_training_params",
        "deploy_params",
        "training_only_params",
        "deploy_param_size_mb_fp32",
        "deploy_param_size_mb_fp16",
        "deploy_param_size_mb_int8",
        "checkpoint_size_mb",
        "deploy_input_values_per_window",
        "deploy_input_size_kb_fp32",
        "latency_ms_mean",
        "latency_ms_std",
        "latency_ms_median",
        "latency_ms_p10",
        "latency_ms_p90",
        "latency_ms_min",
        "latency_ms_max",
        "throughput_windows_per_s",
        "throughput_windows_per_s_median",
    ]
    first_cols = [
        "dataset",
        "model_family",
        "watch_backbone",
        "watch_enhancement",
        "watch_motion_mode",
        "deploy_forward_kind",
        "deploy_correction_enabled",
        "adaptive_correction",
        "legacy_direct_correction",
        "batch_size",
        "watch_channels",
        "watch_length",
        "wavelet_dim",
        "device",
        "deployment_input_modality",
        "needs_privileged_sensors_at_inference",
        "privileged_training_signals",
    ]
    summary_rows: list[dict[str, object]] = []
    for method, group in rows.groupby("method", sort=False):
        row: dict[str, object] = {"method": method, "n_checkpoints": int(len(group))}
        for col in first_cols:
            if col in group.columns:
                row[col] = group[col].iloc[0]
        for col in numeric_cols:
            values = pd.to_numeric(group[col], errors="coerce")
            row[f"{col}_mean"] = float(values.mean())
            row[f"{col}_std"] = float(values.std(ddof=1)) if len(values.dropna()) > 1 else 0.0
        summary_rows.append(row)
    return pd.DataFrame(summary_rows)


def write_markdown_table(summary: pd.DataFrame, output_path: Path) -> None:
    cols = [
        "method",
        "dataset",
        "model_family",
        "watch_enhancement",
        "watch_motion_mode",
        "deploy_forward_kind",
        "adaptive_correction",
        "deploy_params_mean",
        "deploy_param_size_mb_fp32_mean",
        "deploy_param_size_mb_fp16_mean",
        "latency_ms_median_mean",
        "latency_ms_p90_mean",
        "throughput_windows_per_s_median_mean",
        "needs_privileged_sensors_at_inference",
    ]
    available = [col for col in cols if col in summary.columns]
    lines = ["# Deployment Efficiency Summary", ""]
    if summary.empty:
        lines.append("No checkpoints were profiled.")
    else:
        lines.append("| " + " | ".join(available) + " |")
        lines.append("|" + "|".join(["---"] * len(available)) + "|")
        for row in summary[available].itertuples(index=False):
            values = []
            for value in row:
                if isinstance(value, float):
                    values.append(f"{value:.4f}")
                else:
                    values.append(str(value))
            lines.append("| " + " | ".join(values) + " |")
    lines.append("")
    lines.append("Note: deploy_params counts only the stress-probability inference path. Privileged teacher and auxiliary heads are training-time only.")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure deployable watch-path efficiency from saved checkpoints.")
    parser.add_argument("--dataset-kind", type=str, required=True, choices=["galaxy", "wesad"])
    parser.add_argument("--checkpoint", action="append", default=None, help="Single checkpoint as NAME=PATH. Can repeat.")
    parser.add_argument("--checkpoint-dir", action="append", default=None, help="Checkpoint directory as NAME=DIR. Can repeat.")
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--watch-channels", type=int, default=5)
    parser.add_argument("--watch-length", type=int, default=500, help="Galaxy 20s at 25 Hz = 500; WESAD 20s at 32 Hz = 640.")
    parser.add_argument("--wavelet-dim", type=int, default=0, help="0 means infer from checkpoint.")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument(
        "--torch-num-threads",
        type=int,
        default=0,
        help="Set PyTorch CPU intra-op threads for more reproducible CPU latency; 0 leaves the default.",
    )
    parser.add_argument(
        "--torch-num-interop-threads",
        type=int,
        default=0,
        help="Set PyTorch CPU inter-op threads; 0 leaves the default.",
    )
    parser.add_argument("--correction-scale-init", type=float, default=0.05)
    parser.add_argument("--correction-alpha-init-bias", type=float, default=-3.0)
    parser.add_argument("--correction-alpha-max", type=float, default=1.0)
    parser.add_argument(
        "--max-checkpoints-per-set",
        type=int,
        default=1,
        help="Use one checkpoint per method by default because efficiency is architecture-level; set 0 for all.",
    )
    args = parser.parse_args()
    if args.torch_num_threads > 0:
        torch.set_num_threads(args.torch_num_threads)
    if args.torch_num_interop_threads > 0:
        torch.set_num_interop_threads(args.torch_num_interop_threads)

    checkpoints = discover_checkpoints(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for method, checkpoint_path in checkpoints:
        print(f"[{method}] profiling {checkpoint_path}")
        rows.append(profile_checkpoint(method, checkpoint_path, args))

    per_checkpoint = pd.DataFrame(rows)
    method_summary = summarize(per_checkpoint)
    prefix = args.dataset_kind
    per_checkpoint_path = args.output_dir / f"{prefix}_deployment_efficiency_per_checkpoint.csv"
    summary_path = args.output_dir / f"{prefix}_deployment_efficiency_summary.csv"
    markdown_path = args.output_dir / f"{prefix}_deployment_efficiency_summary.md"
    per_checkpoint.to_csv(per_checkpoint_path, index=False)
    method_summary.to_csv(summary_path, index=False)
    write_markdown_table(method_summary, markdown_path)

    display_cols = [
        "method",
        "model_family",
        "watch_enhancement",
        "deploy_forward_kind",
        "adaptive_correction",
        "deploy_params_mean",
        "deploy_param_size_mb_fp32_mean",
        "latency_ms_median_mean",
        "latency_ms_p90_mean",
        "throughput_windows_per_s_median_mean",
        "needs_privileged_sensors_at_inference",
    ]
    print()
    print(method_summary[[col for col in display_cols if col in method_summary.columns]].to_string(index=False))
    print()
    print(f"Saved per-checkpoint efficiency to {per_checkpoint_path}")
    print(f"Saved method efficiency summary to {summary_path}")
    print(f"Saved markdown table to {markdown_path}")


if __name__ == "__main__":
    main()
