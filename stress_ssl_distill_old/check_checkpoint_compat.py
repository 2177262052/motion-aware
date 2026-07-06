from __future__ import annotations

import argparse
import inspect
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from .galaxy_models import (
    PrivilegedGalaxyTeacherNet,
    ResNet18WatchNet,
    ResNet34WatchNet,
    ResNet50WatchNet,
    WaveletGuidedWatchNet,
)
from .galaxy_models_adaptive_correction import AdaptiveCorrectionGalaxyTeacherNet
from .wesad_models_safe_sgpc import WESADPrivilegedTeacherNet


STATE_DICT_CANDIDATES = (
    "state_dict",
    "model_state_dict",
    "model",
    "net",
    "ema_state_dict",
)


def _is_tensor_state_dict(obj: Any) -> bool:
    return isinstance(obj, Mapping) and any(torch.is_tensor(value) for value in obj.values())


def extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if _is_tensor_state_dict(checkpoint):
        raw = checkpoint
    elif isinstance(checkpoint, Mapping):
        raw = None
        for key in STATE_DICT_CANDIDATES:
            value = checkpoint.get(key)
            if _is_tensor_state_dict(value):
                raw = value
                break
        if raw is None:
            top_keys = ", ".join(str(key) for key in checkpoint.keys())
            raise ValueError(f"Could not find tensor state_dict. Top-level keys: {top_keys}")
    else:
        raise ValueError(f"Unsupported checkpoint type: {type(checkpoint)!r}")

    state: dict[str, torch.Tensor] = {}
    for key, value in raw.items():
        if not torch.is_tensor(value):
            continue
        normalized = str(key)
        while normalized.startswith("module."):
            normalized = normalized[len("module.") :]
        state[normalized] = value
    return state


def has_any(keys: set[str], *prefixes: str) -> bool:
    return any(any(key.startswith(prefix) for prefix in prefixes) for key in keys)


def infer_wavelet_dim(state: dict[str, torch.Tensor], default: int = 4) -> int:
    for key in (
        "wavelet_mlp.0.weight",
        "watch_encoder.wavelet_mlp.0.weight",
    ):
        value = state.get(key)
        if value is not None and value.ndim == 2:
            return int(value.shape[1] - 1)
    for key in (
        "wavelet_predictor.2.weight",
        "watch_encoder.wavelet_predictor.2.weight",
    ):
        value = state.get(key)
        if value is not None and value.ndim == 2:
            return int(value.shape[0])
    return default


def infer_model_type(state: dict[str, torch.Tensor]) -> str:
    keys = set(state)
    if has_any(keys, "privileged_encoder."):
        return "wesad-privileged"
    if has_any(keys, "e4_encoder.", "polar_encoder.", "teacher_e4_proj."):
        if has_any(keys, "deploy_correction_alpha."):
            return "galaxy-adaptive"
        return "galaxy-privileged"
    if has_any(keys, "watch_encoder."):
        if has_any(keys, "deploy_correction_alpha."):
            return "galaxy-adaptive"
        return "galaxy-privileged"
    if has_any(keys, "layer1.", "layer2.", "layer3.", "layer4."):
        return "galaxy-watch-resnet"
    return "galaxy-watch"


def infer_watch_backbone(state: dict[str, torch.Tensor], model_type: str) -> str:
    keys = set(state)
    prefix = "watch_encoder." if "privileged" in model_type or "adaptive" in model_type else ""
    if has_any(keys, f"{prefix}layer1.", f"{prefix}layer2.", f"{prefix}layer3."):
        # Depth is hard to distinguish from keys alone without inspecting block count.
        return "resnet18_1d"
    return "wavelet_guided"


def infer_watch_enhancement(state: dict[str, torch.Tensor], model_type: str) -> str:
    prefix = "watch_encoder." if "privileged" in model_type or "adaptive" in model_type else ""
    return "motion_disentangled" if has_any(set(state), f"{prefix}ppg_enhancer.") else "none"


def infer_watch_motion_mode(state: dict[str, torch.Tensor], model_type: str) -> str:
    prefix = "watch_encoder." if "privileged" in model_type or "adaptive" in model_type else ""
    return "residual" if has_any(set(state), f"{prefix}motion_gates.0.residual_scale") else "strong"


def infer_bool_from_prefix(state: dict[str, torch.Tensor], prefix: str) -> bool:
    return has_any(set(state), prefix)


def bool_arg(value: str) -> bool | None:
    if value == "auto":
        return None
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError("Expected auto/true/false.")


def construct(module_cls: type[torch.nn.Module], **kwargs: Any) -> torch.nn.Module:
    """Instantiate a model while tolerating older constructor signatures."""
    signature = inspect.signature(module_cls.__init__)
    parameters = signature.parameters
    accepts_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values())
    if accepts_kwargs:
        return module_cls(**kwargs)
    supported = {key: value for key, value in kwargs.items() if key in parameters}
    return module_cls(**supported)


def resolve_auto(value: str, inferred: str) -> str:
    return inferred if value == "auto" else value


def resolve_auto_bool(value: bool | None, inferred: bool) -> bool:
    return inferred if value is None else value


def build_model(args: argparse.Namespace, state: dict[str, torch.Tensor]) -> torch.nn.Module:
    inferred_model_type = infer_model_type(state)
    model_type = resolve_auto(args.model_type, inferred_model_type)
    wavelet_dim = args.wavelet_dim if args.wavelet_dim is not None else infer_wavelet_dim(state)
    watch_backbone = resolve_auto(args.watch_backbone, infer_watch_backbone(state, model_type))
    watch_enhancement = resolve_auto(args.watch_enhancement, infer_watch_enhancement(state, model_type))
    watch_motion_mode = resolve_auto(args.watch_motion_mode, infer_watch_motion_mode(state, model_type))

    if model_type == "galaxy-watch":
        if watch_backbone == "wavelet_guided":
            return construct(
                WaveletGuidedWatchNet,
                wavelet_dim=wavelet_dim,
                watch_enhancement=watch_enhancement,
                watch_motion_mode=watch_motion_mode,
            )
        if watch_backbone == "resnet18_1d":
            return construct(ResNet18WatchNet, wavelet_dim=wavelet_dim)
        if watch_backbone == "resnet34_1d":
            return construct(ResNet34WatchNet, wavelet_dim=wavelet_dim)
        if watch_backbone == "resnet50_1d":
            return construct(ResNet50WatchNet, wavelet_dim=wavelet_dim)
        raise ValueError(f"Unsupported watch backbone: {watch_backbone}")

    if model_type == "galaxy-watch-resnet":
        return construct(ResNet18WatchNet, wavelet_dim=wavelet_dim)

    if model_type in {"galaxy-privileged", "galaxy-adaptive"}:
        use_reliability_head = resolve_auto_bool(args.reliability_head, infer_bool_from_prefix(state, "reliability_head."))
        use_projection_heads = resolve_auto_bool(args.projection_heads, infer_bool_from_prefix(state, "watch_projector."))
        use_e4_classifier = resolve_auto_bool(args.e4_classifier, infer_bool_from_prefix(state, "e4_classifier."))
        use_rhythm_heads = resolve_auto_bool(args.rhythm_heads, infer_bool_from_prefix(state, "rhythm_head."))
        use_wavelet_head = resolve_auto_bool(args.wavelet_head, infer_bool_from_prefix(state, "wavelet_predictor."))
        use_teacher_fused_classifier = resolve_auto_bool(
            args.teacher_fused_classifier,
            infer_bool_from_prefix(state, "teacher_fused_classifier."),
        )
        use_student_gated_correction = resolve_auto_bool(
            args.student_gated_correction,
            infer_bool_from_prefix(state, "deploy_correction."),
        )
        common = dict(
            wavelet_dim=wavelet_dim,
            watch_backbone=watch_backbone,
            watch_enhancement=watch_enhancement,
            watch_motion_mode=watch_motion_mode,
            use_reliability_head=use_reliability_head,
            use_projection_heads=use_projection_heads,
            use_e4_classifier=use_e4_classifier,
            use_rhythm_heads=use_rhythm_heads,
            use_wavelet_head=use_wavelet_head,
            use_teacher_fused_classifier=use_teacher_fused_classifier,
            use_student_gated_correction=use_student_gated_correction,
            correction_scale_init=args.correction_scale_init,
        )
        if model_type == "galaxy-adaptive":
            return construct(
                AdaptiveCorrectionGalaxyTeacherNet,
                **common,
                correction_alpha_init_bias=args.correction_alpha_init_bias,
                correction_alpha_max=args.correction_alpha_max,
                correction_mode=args.correction_mode,
            )
        return construct(PrivilegedGalaxyTeacherNet, **common)

    if model_type in {"wesad-privileged", "wesad-safe-sgpc"}:
        use_student_gated_correction = resolve_auto_bool(
            args.student_gated_correction,
            infer_bool_from_prefix(state, "deploy_correction."),
        )
        return construct(
            WESADPrivilegedTeacherNet,
            wavelet_dim=wavelet_dim,
            privileged_channels=args.privileged_channels,
            watch_backbone=watch_backbone,
            watch_enhancement=watch_enhancement,
            watch_motion_mode=watch_motion_mode,
            use_student_gated_correction=use_student_gated_correction,
            correction_scale_init=args.correction_scale_init,
            correction_alpha_init_bias=args.correction_alpha_init_bias,
            correction_alpha_max=args.correction_alpha_max,
            correction_mode=args.correction_mode,
        )

    raise ValueError(f"Unsupported model type: {model_type}")


def compare_state_dicts(
    checkpoint_state: dict[str, torch.Tensor],
    model_state: Mapping[str, torch.Tensor],
) -> tuple[list[str], list[str], list[tuple[str, tuple[int, ...], tuple[int, ...]]]]:
    checkpoint_keys = set(checkpoint_state)
    model_keys = set(model_state)
    missing = sorted(model_keys - checkpoint_keys)
    unexpected = sorted(checkpoint_keys - model_keys)
    mismatched = []
    for key in sorted(checkpoint_keys & model_keys):
        ckpt_shape = tuple(checkpoint_state[key].shape)
        model_shape = tuple(model_state[key].shape)
        if ckpt_shape != model_shape:
            mismatched.append((key, ckpt_shape, model_shape))
    return missing, unexpected, mismatched


def print_head(title: str, rows: list[str], limit: int) -> None:
    print(f"{title}={len(rows)}")
    for item in rows[:limit]:
        print(f"  - {item}")
    if len(rows) > limit:
        print(f"  - ... ({len(rows) - limit} more)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check whether a checkpoint matches the current model structure.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--model-type",
        default="auto",
        choices=[
            "auto",
            "galaxy-watch",
            "galaxy-watch-resnet",
            "galaxy-privileged",
            "galaxy-adaptive",
            "wesad-privileged",
            "wesad-safe-sgpc",
        ],
    )
    parser.add_argument("--watch-backbone", default="auto", choices=["auto", "wavelet_guided", "resnet18_1d", "resnet34_1d", "resnet50_1d"])
    parser.add_argument("--watch-enhancement", default="auto", choices=["auto", "none", "motion_disentangled"])
    parser.add_argument("--watch-motion-mode", default="auto", choices=["auto", "strong", "residual", "scaled"])
    parser.add_argument("--wavelet-dim", type=int, default=None)
    parser.add_argument("--privileged-channels", type=int, default=9)
    parser.add_argument("--student-gated-correction", type=bool_arg, default=None)
    parser.add_argument("--reliability-head", type=bool_arg, default=None)
    parser.add_argument("--projection-heads", type=bool_arg, default=None)
    parser.add_argument("--e4-classifier", type=bool_arg, default=None)
    parser.add_argument("--rhythm-heads", type=bool_arg, default=None)
    parser.add_argument("--wavelet-head", type=bool_arg, default=None)
    parser.add_argument("--teacher-fused-classifier", type=bool_arg, default=None)
    parser.add_argument("--correction-scale-init", type=float, default=0.05)
    parser.add_argument("--correction-alpha-init-bias", type=float, default=-3.0)
    parser.add_argument("--correction-alpha-max", type=float, default=1.0)
    parser.add_argument("--correction-mode", default="logit_mix", choices=["logit_mix", "margin_residual"])
    parser.add_argument("--limit", type=int, default=30)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    checkpoint_state = extract_state_dict(checkpoint)
    model = build_model(args, checkpoint_state)
    model_state = model.state_dict()
    missing, unexpected, mismatched = compare_state_dicts(checkpoint_state, model_state)

    print(f"checkpoint={args.checkpoint}")
    print(f"inferred_model_type={infer_model_type(checkpoint_state)}")
    print(f"built_model={model.__class__.__name__}")
    print(f"checkpoint_tensors={len(checkpoint_state)}")
    print(f"model_tensors={len(model_state)}")
    print(f"checkpoint_numel={sum(v.numel() for v in checkpoint_state.values())}")
    print(f"model_numel={sum(v.numel() for v in model_state.values())}")
    print()

    print_head("missing_keys", missing, args.limit)
    print_head("unexpected_keys", unexpected, args.limit)
    print(f"shape_mismatches={len(mismatched)}")
    for key, ckpt_shape, model_shape in mismatched[: args.limit]:
        print(f"  - {key}: checkpoint={ckpt_shape} model={model_shape}")
    if len(mismatched) > args.limit:
        print(f"  - ... ({len(mismatched) - args.limit} more)")
    print()

    strict_ok = not missing and not unexpected and not mismatched
    shape_ok = not mismatched
    print(f"strict_load_compatible={strict_ok}")
    print(f"shape_compatible_for_common_keys={shape_ok}")
    if strict_ok:
        model.load_state_dict(checkpoint_state, strict=True)
        print("strict_load_test=passed")
    else:
        print("strict_load_test=skipped")


if __name__ == "__main__":
    main()
