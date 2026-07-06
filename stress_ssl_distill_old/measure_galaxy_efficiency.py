from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch

from .analyze_galaxy_model_failures import build_model_from_state, normalize_state_dict
from .galaxy_models import PrivilegedGalaxyTeacherNet, WaveletGuidedWatchNet


BYTES_PER_FLOAT32 = 4
MB = 1024 * 1024


def parse_named_path(value: str, default_name: str | None = None) -> tuple[str, Path]:
    if "=" in value:
        name, path = value.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Missing method name in named path: {value}")
        return name, Path(path)
    path = Path(value)
    return default_name or path.stem, path


def discover_checkpoints(args: argparse.Namespace) -> list[tuple[str, Path]]:
    checkpoints: list[tuple[str, Path]] = []
    for item in args.checkpoint or []:
        checkpoints.append(parse_named_path(item))

    for item in args.checkpoint_dir or []:
        name, path = parse_named_path(item, default_name=Path(item).stem)
        ckpts = sorted(path.glob("*.pt"))
        if not ckpts:
            raise ValueError(f"No .pt checkpoints found in {path}")
        if args.max_checkpoints_per_set > 0:
            ckpts = ckpts[: args.max_checkpoints_per_set]
        checkpoints.extend((name, ckpt) for ckpt in ckpts)

    if not checkpoints:
        raise ValueError("Provide --checkpoint NAME=PATH and/or --checkpoint-dir NAME=DIR.")
    return checkpoints


def count_params(model: torch.nn.Module) -> int:
    return int(sum(param.numel() for param in model.parameters()))


def include_watch_net_deploy_param(name: str) -> bool:
    training_only_prefixes = (
        "baseline_ref_adapter.",
        "baseline_ref_gate.",
        "contrastive_head.",
        "wavelet_predictor.",
    )
    return not name.startswith(training_only_prefixes)


def include_privileged_deploy_param(name: str, use_sgpc: bool) -> bool:
    if name.startswith("watch_encoder."):
        watch_name = name[len("watch_encoder.") :]
        return include_watch_net_deploy_param(watch_name)
    if name.startswith("watch_classifier."):
        return True
    if use_sgpc and name.startswith(("deploy_correction.", "deploy_correction_gate.", "correction_norm.")):
        return True
    if use_sgpc and name == "correction_scale":
        return True
    return False


def count_deploy_params(model: torch.nn.Module) -> int:
    if isinstance(model, WaveletGuidedWatchNet):
        return int(sum(param.numel() for name, param in model.named_parameters() if include_watch_net_deploy_param(name)))
    if isinstance(model, PrivilegedGalaxyTeacherNet):
        return int(
            sum(
                param.numel()
                for name, param in model.named_parameters()
                if include_privileged_deploy_param(name, model.use_student_gated_correction)
            )
        )
    return count_params(model)


def infer_architecture(model: torch.nn.Module) -> dict[str, object]:
    if isinstance(model, WaveletGuidedWatchNet):
        return {
            "model_family": "watch_only",
            "watch_backbone": "wavelet_guided",
            "watch_enhancement": model.watch_enhancement,
            "sgpc": False,
            "needs_e4_or_privileged_at_test": "no",
            "deployment_input_modality": "watch PPG + watch ACC + watch-derived wavelet/quality",
            "privileged_training_signals": "none",
        }
    if isinstance(model, PrivilegedGalaxyTeacherNet):
        return {
            "model_family": "privileged_deploy_watch",
            "watch_backbone": model.watch_backbone,
            "watch_enhancement": model.watch_enhancement,
            "sgpc": bool(model.use_student_gated_correction),
            "needs_e4_or_privileged_at_test": "no",
            "deployment_input_modality": "watch PPG + watch ACC + watch-derived wavelet/quality",
            "privileged_training_signals": "E4 BVP/ACC and auxiliary privileged targets during training only",
        }
    return {
        "model_family": type(model).__name__,
        "watch_backbone": "",
        "watch_enhancement": "",
        "sgpc": "",
        "needs_e4_or_privileged_at_test": "unknown",
        "deployment_input_modality": "unknown",
        "privileged_training_signals": "unknown",
    }


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


def infer_wavelet_dim_from_model(model: torch.nn.Module) -> int:
    if isinstance(model, WaveletGuidedWatchNet):
        weight = model.wavelet_mlp[0].weight
        return int(weight.shape[1] - 1)
    if isinstance(model, PrivilegedGalaxyTeacherNet):
        encoder = model.watch_encoder
        if hasattr(encoder, "wavelet_mlp"):
            weight = encoder.wavelet_mlp[0].weight
            return int(weight.shape[1] - 1)
    return 4


def deploy_forward_callable(
    model: torch.nn.Module,
    signal: torch.Tensor,
    wavelet: torch.Tensor,
    quality: torch.Tensor,
) -> Callable[[], torch.Tensor]:
    if isinstance(model, WaveletGuidedWatchNet):
        def forward_watch_only() -> torch.Tensor:
            # Logit-only deployment path; skips contrastive/wavelet auxiliary heads.
            out = model._encode_core(signal, wavelet, quality)
            return model.classifier(out["embedding"])

        return forward_watch_only

    if isinstance(model, PrivilegedGalaxyTeacherNet):
        def forward_privileged_deploy() -> torch.Tensor:
            out = model.forward_watch(signal, wavelet, quality, return_aux=False)
            return out["logits"]

        return forward_privileged_deploy

    def forward_generic() -> torch.Tensor:
        out = model(signal, wavelet, quality)
        if isinstance(out, dict) and "logits" in out:
            return out["logits"]
        if isinstance(out, torch.Tensor):
            return out
        raise TypeError(f"Unsupported model output type: {type(out)!r}")

    return forward_generic


def measure_latency(
    forward_fn: Callable[[], torch.Tensor],
    device: str,
    warmup: int,
    repeats: int,
) -> tuple[float, float]:
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

    return float(np.mean(timings)), float(np.std(timings))


def profile_checkpoint(method: str, checkpoint_path: Path, args: argparse.Namespace) -> dict[str, object]:
    state = normalize_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model = build_model_from_state(
        state,
        watch_backbone=args.watch_backbone,
        correction_scale_init=args.correction_scale_init,
        device=args.device,
    )
    model.eval()

    wavelet_dim = args.wavelet_dim if args.wavelet_dim > 0 else infer_wavelet_dim_from_model(model)
    signal, wavelet, quality = make_dummy_inputs(
        device=args.device,
        batch_size=args.batch_size,
        watch_channels=args.watch_channels,
        watch_length=args.watch_length,
        wavelet_dim=wavelet_dim,
    )
    forward_fn = deploy_forward_callable(model, signal, wavelet, quality)

    latency_mean, latency_std = measure_latency(
        forward_fn,
        device=args.device,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    total_params = count_params(model)
    deploy_params = count_deploy_params(model)
    arch = infer_architecture(model)

    return {
        "method": method,
        "checkpoint": str(checkpoint_path),
        **arch,
        "batch_size": args.batch_size,
        "watch_channels": args.watch_channels,
        "watch_length": args.watch_length,
        "wavelet_dim": wavelet_dim,
        "total_params": total_params,
        "deploy_params": deploy_params,
        "deploy_param_size_mb_fp32": deploy_params * BYTES_PER_FLOAT32 / MB,
        "deploy_param_size_mb_fp16": deploy_params * 2 / MB,
        "checkpoint_size_mb": checkpoint_path.stat().st_size / MB,
        "latency_ms_mean": latency_mean,
        "latency_ms_std": latency_std,
        "throughput_windows_per_s": (args.batch_size * 1000.0 / latency_mean) if latency_mean > 0 else float("nan"),
    }


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        "total_params",
        "deploy_params",
        "deploy_param_size_mb_fp32",
        "deploy_param_size_mb_fp16",
        "checkpoint_size_mb",
        "latency_ms_mean",
        "latency_ms_std",
        "throughput_windows_per_s",
    ]
    first_cols = [
        "model_family",
        "watch_backbone",
        "watch_enhancement",
        "sgpc",
        "batch_size",
        "watch_channels",
        "watch_length",
        "wavelet_dim",
        "deployment_input_modality",
        "needs_e4_or_privileged_at_test",
        "privileged_training_signals",
    ]

    summary_rows: list[dict[str, object]] = []
    for method, group in rows.groupby("method", sort=False):
        row: dict[str, object] = {"method": method, "n_checkpoints": int(len(group))}
        for col in first_cols:
            if col in group.columns:
                row[col] = group[col].iloc[0]
        for col in numeric_cols:
            if col in group.columns:
                values = pd.to_numeric(group[col], errors="coerce")
                row[f"{col}_mean"] = float(values.mean())
                row[f"{col}_std"] = float(values.std(ddof=1)) if len(values.dropna()) > 1 else 0.0
        summary_rows.append(row)
    return pd.DataFrame(summary_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure Galaxy deployment efficiency from saved checkpoints.")
    parser.add_argument("--checkpoint", action="append", default=None, help="Single checkpoint as NAME=PATH. Can repeat.")
    parser.add_argument("--checkpoint-dir", action="append", default=None, help="Checkpoint directory as NAME=DIR. Can repeat.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--watch-channels", type=int, default=5)
    parser.add_argument("--watch-length", type=int, default=500, help="20 s at 25 Hz by default.")
    parser.add_argument("--wavelet-dim", type=int, default=0, help="0 means infer from checkpoint.")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--watch-backbone", type=str, default="wavelet_guided")
    parser.add_argument("--correction-scale-init", type=float, default=0.05)
    parser.add_argument(
        "--max-checkpoints-per-set",
        type=int,
        default=1,
        help="Use 1 checkpoint per method by default because efficiency is architecture-level; set 0 for all.",
    )
    args = parser.parse_args()

    checkpoints = discover_checkpoints(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for method, checkpoint_path in checkpoints:
        print(f"[{method}] profiling {checkpoint_path}")
        rows.append(profile_checkpoint(method, checkpoint_path, args))

    per_checkpoint = pd.DataFrame(rows)
    method_summary = summarize(per_checkpoint)
    per_checkpoint_path = args.output_dir / "galaxy_efficiency_per_checkpoint.csv"
    summary_path = args.output_dir / "galaxy_efficiency_summary.csv"
    per_checkpoint.to_csv(per_checkpoint_path, index=False)
    method_summary.to_csv(summary_path, index=False)

    display_cols = [
        "method",
        "model_family",
        "watch_enhancement",
        "sgpc",
        "deploy_params_mean",
        "deploy_param_size_mb_fp32_mean",
        "latency_ms_mean_mean",
        "throughput_windows_per_s_mean",
        "deployment_input_modality",
        "needs_e4_or_privileged_at_test",
    ]
    print()
    print(method_summary[[col for col in display_cols if col in method_summary.columns]].to_string(index=False))
    print()
    print(f"Saved per-checkpoint efficiency to {per_checkpoint_path}")
    print(f"Saved method efficiency summary to {summary_path}")


if __name__ == "__main__":
    main()
