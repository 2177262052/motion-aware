from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from . import galaxy_models as galaxy_models_module
from .galaxy_models import PrivilegedGalaxyTeacherNet, WaveletGuidedWatchNet
from .wesad_models_safe_sgpc import WESADPrivilegedTeacherNet


BYTES_PER_FLOAT32 = 4
MB = 1024 * 1024
ORIGINAL_MOTION_FILM = galaxy_models_module.MotionFiLM


class ScaledMotionFiLM(nn.Module):
    """Scaled motion FiLM used by the final motion-aware checkpoints."""

    def __init__(self, cond_dim: int, target_dim: int, *args: object, scale_logit_init: float = -2.0, **kwargs: object) -> None:
        super().__init__()
        _ = (args, kwargs)
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


def normalize_state_dict(raw: object) -> dict[str, torch.Tensor]:
    if isinstance(raw, dict) and "state_dict" in raw and isinstance(raw["state_dict"], dict):
        raw = raw["state_dict"]
    if not isinstance(raw, dict):
        raise TypeError(f"Expected a state_dict-like object, got {type(raw)!r}")
    state: dict[str, torch.Tensor] = {}
    for key, value in raw.items():
        if not torch.is_tensor(value):
            continue
        name = str(key)
        if name.startswith("module."):
            name = name[len("module.") :]
        state[name] = value
    return state


def has_prefix(state: dict[str, torch.Tensor], *prefixes: str) -> bool:
    return any(any(key.startswith(prefix) for prefix in prefixes) for key in state)


def install_motion_film_for_state(state: dict[str, torch.Tensor], prefix: str = "") -> None:
    if has_prefix(state, f"{prefix}motion_gates.0.scale_logit"):
        galaxy_models_module.MotionFiLM = ScaledMotionFiLM
    else:
        galaxy_models_module.MotionFiLM = ORIGINAL_MOTION_FILM


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
    for key in (f"{prefix}classifier.weight", f"{prefix}watch_classifier.weight", "watch_classifier.weight"):
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
    layer_prefix = f"{prefix}transformer.layers."
    layer_ids = set()
    for key in state:
        if key.startswith(layer_prefix):
            token = key[len(layer_prefix) :].split(".", 1)[0]
            if token.isdigit():
                layer_ids.add(int(token))
    return max(layer_ids) + 1 if layer_ids else 2


def infer_watch_kwargs(state: dict[str, torch.Tensor], prefix: str = "") -> dict[str, object]:
    return {
        "wavelet_dim": infer_wavelet_dim(state, prefix),
        "model_dim": infer_model_dim(state, prefix),
        "transformer_layers": infer_transformer_layers(state, prefix),
        "transformer_heads": 4,
        "fusion_hidden_dim": infer_fusion_hidden_dim(state, prefix),
        "embed_dim": infer_embed_dim(state, prefix),
        "watch_enhancement": "motion_disentangled" if has_prefix(state, f"{prefix}ppg_enhancer.") else "none",
        "watch_motion_mode": "scaled" if has_prefix(state, f"{prefix}motion_gates.0.scale_logit") else "strong",
    }


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" in value:
        name, path = value.split("=", 1)
        return name.strip(), Path(path)
    path = Path(value)
    return path.stem, path


def checkpoint_subject_sort_key(path: Path) -> tuple[int, str]:
    token = path.stem.replace("_deploy_watch", "").replace("_watch_only", "")
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
        name, path = parse_named_path(item)
        candidates = sorted(path.glob("*.pt"), key=checkpoint_subject_sort_key)
        if not candidates and (path / "checkpoints").exists():
            candidates = sorted((path / "checkpoints").glob("*.pt"), key=checkpoint_subject_sort_key)
        if args.max_checkpoints_per_set > 0:
            candidates = candidates[: args.max_checkpoints_per_set]
        checkpoints.extend((name, ckpt) for ckpt in candidates)
    if not checkpoints:
        raise ValueError("Provide --checkpoint NAME=PATH or --checkpoint-dir NAME=DIR.")
    return checkpoints


def include_watch_deploy_param(name: str) -> bool:
    return not name.startswith(("wavelet_predictor.", "baseline_ref_adapter.", "baseline_ref_gate."))


def count_deploy_params(model: nn.Module, model_family: str) -> int:
    if model_family == "watch_only":
        return int(sum(p.numel() for name, p in model.named_parameters() if include_watch_deploy_param(name)))
    return int(
        sum(
            p.numel()
            for name, p in model.named_parameters()
            if (name.startswith("watch_encoder.") and include_watch_deploy_param(name[len("watch_encoder.") :]))
            or name.startswith("watch_classifier.")
        )
    )


def build_model(dataset_kind: str, state: dict[str, torch.Tensor]) -> tuple[nn.Module, str]:
    if has_prefix(state, "watch_encoder."):
        prefix = "watch_encoder."
        install_motion_film_for_state(state, prefix)
        kwargs = infer_watch_kwargs(state, prefix)
        if dataset_kind == "galaxy":
            model = PrivilegedGalaxyTeacherNet(
                watch_kwargs=kwargs,
                teacher_dim=int(kwargs["model_dim"]),
                embed_dim=int(kwargs["embed_dim"]),
            )
        else:
            model = WESADPrivilegedTeacherNet(
                watch_kwargs=kwargs,
                teacher_dim=int(kwargs["model_dim"]),
                embed_dim=int(kwargs["embed_dim"]),
            )
        model.load_state_dict(state, strict=False)
        return model, "privileged_deploy_watch"

    install_motion_film_for_state(state)
    kwargs = infer_watch_kwargs(state)
    model = WaveletGuidedWatchNet(**kwargs)
    model.load_state_dict(state, strict=False)
    return model, "watch_only"


def forward_deploy(model: nn.Module, model_family: str, signal: torch.Tensor, wavelet: torch.Tensor, quality: torch.Tensor) -> torch.Tensor:
    if model_family == "watch_only":
        return model(signal, wavelet, quality)["logits"]
    return model.forward_watch(signal, wavelet, quality)["logits"]


def time_forward(
    model: nn.Module,
    model_family: str,
    signal: torch.Tensor,
    wavelet: torch.Tensor,
    quality: torch.Tensor,
    warmup: int,
    repeats: int,
) -> tuple[float, float, float, float]:
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = forward_deploy(model, model_family, signal, wavelet, quality)
        durations = []
        for _ in range(repeats):
            start = time.perf_counter()
            _ = forward_deploy(model, model_family, signal, wavelet, quality)
            durations.append((time.perf_counter() - start) * 1000.0)
    arr = np.asarray(durations, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=1) if len(arr) > 1 else 0.0), float(np.median(arr)), float(np.percentile(arr, 90))


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure deployable watch-path parameters and CPU/GPU latency.")
    parser.add_argument("--dataset-kind", type=str, required=True, choices=["galaxy", "wesad"])
    parser.add_argument("--checkpoint", action="append", default=[])
    parser.add_argument("--checkpoint-dir", action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--watch-channels", type=int, default=5)
    parser.add_argument("--watch-length", type=int, default=500)
    parser.add_argument("--wavelet-dim", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=1000)
    parser.add_argument("--max-checkpoints-per-set", type=int, default=0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for method, checkpoint_path in discover_checkpoints(args):
        state = normalize_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        model, family = build_model(args.dataset_kind, state)
        model.to(args.device)
        signal = torch.randn(args.batch_size, args.watch_channels, args.watch_length, device=args.device)
        wavelet = torch.randn(args.batch_size, args.wavelet_dim, device=args.device)
        quality = torch.ones(args.batch_size, 1, device=args.device)
        params = count_deploy_params(model, family)
        latency_mean, latency_std, latency_median, latency_p90 = time_forward(
            model,
            family,
            signal,
            wavelet,
            quality,
            args.warmup,
            args.repeats,
        )
        rows.append(
            {
                "method": method,
                "checkpoint": str(checkpoint_path),
                "dataset": args.dataset_kind,
                "model_family": family,
                "deploy_params": params,
                "deploy_param_size_mb_fp32": params * BYTES_PER_FLOAT32 / MB,
                "deploy_param_size_mb_fp16": params * 2 / MB,
                "latency_ms_mean": latency_mean,
                "latency_ms_std": latency_std,
                "latency_ms_median": latency_median,
                "latency_ms_p90": latency_p90,
                "throughput_windows_per_s_mean": 1000.0 / max(latency_mean, 1e-12),
                "needs_privileged_sensors_at_inference": "no",
            }
        )

    per_checkpoint = pd.DataFrame(rows)
    per_checkpoint.to_csv(args.output_dir / f"{args.dataset_kind}_deployment_efficiency_per_checkpoint.csv", index=False)

    summary = (
        per_checkpoint.groupby(["method", "dataset", "model_family"], as_index=False)
        .agg(
            deploy_params_mean=("deploy_params", "mean"),
            deploy_param_size_mb_fp32_mean=("deploy_param_size_mb_fp32", "mean"),
            deploy_param_size_mb_fp16_mean=("deploy_param_size_mb_fp16", "mean"),
            latency_ms_mean_mean=("latency_ms_mean", "mean"),
            latency_ms_std_mean=("latency_ms_std", "mean"),
            latency_ms_median_mean=("latency_ms_median", "mean"),
            latency_ms_p90_mean=("latency_ms_p90", "mean"),
            throughput_windows_per_s_mean=("throughput_windows_per_s_mean", "mean"),
            needs_privileged_sensors_at_inference=("needs_privileged_sensors_at_inference", "first"),
        )
        .sort_values(["dataset", "method"])
    )
    summary.to_csv(args.output_dir / f"{args.dataset_kind}_deployment_efficiency_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
