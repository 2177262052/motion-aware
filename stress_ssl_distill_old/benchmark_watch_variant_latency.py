from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .analyze_motion_aware_mechanism import build_model_from_state
from .measure_deployment_efficiency import normalize_state_dict


TRAINING_ONLY_PREFIXES = (
    "baseline_ref_adapter.",
    "baseline_ref_gate.",
    "contrastive_head.",
    "wavelet_predictor.",
)
DEPLOY_PREFIX_STRIP = ("encoder.", "watch_encoder.")


def parse_variant(value: str) -> tuple[str, str, Path]:
    parts = value.split("=", 2)
    if len(parts) != 3:
        raise ValueError(
            "--variant must use NAME=INPUT_ABLATION=CHECKPOINT_DIR, "
            f"got: {value!r}"
        )
    name, input_ablation, checkpoint_dir = (part.strip() for part in parts)
    if not name or not input_ablation or not checkpoint_dir:
        raise ValueError(f"Invalid --variant value: {value!r}")
    return name, input_ablation, Path(checkpoint_dir)


def checkpoint_subject_sort_key(path: Path) -> tuple[int, str]:
    stem = path.stem
    token = stem.replace("_deploy_watch", "").replace("_watch_only", "")
    token = token.replace("galaxy_", "").replace("wesad_", "")
    token = token.replace("Sub", "").replace("S", "").replace("P", "")
    try:
        return int(token), path.name
    except ValueError:
        return 10_000, path.name


def discover_checkpoint(checkpoint_dir: Path, subject: str | None) -> Path:
    roots = [checkpoint_dir]
    if (checkpoint_dir / "checkpoints").exists():
        roots.insert(0, checkpoint_dir / "checkpoints")

    if subject:
        for root in roots:
            candidates = [
                root / f"{subject}.pt",
                root / f"{subject}_watch_only.pt",
                root / f"{subject}_deploy_watch.pt",
                root / f"galaxy_{subject}.pt",
                root / f"wesad_{subject}.pt",
                root / f"galaxy_{subject}_watch_only.pt",
                root / f"wesad_{subject}_watch_only.pt",
            ]
            for candidate in candidates:
                if candidate.exists():
                    return candidate
            matches = sorted(root.glob(f"*{subject}*.pt"), key=checkpoint_subject_sort_key)
            if matches:
                return matches[0]

    all_candidates: list[Path] = []
    for root in roots:
        all_candidates.extend(root.glob("*.pt"))
    all_candidates = sorted(set(all_candidates), key=checkpoint_subject_sort_key)
    if not all_candidates:
        raise FileNotFoundError(f"No .pt checkpoint found in {checkpoint_dir}")
    return all_candidates[0]


def load_state(path: Path) -> dict[str, torch.Tensor]:
    raw = torch.load(path, map_location="cpu")
    return normalize_state_dict(raw)


def strip_deploy_prefix(name: str) -> str:
    stripped = name
    for prefix in DEPLOY_PREFIX_STRIP:
        if stripped.startswith(prefix):
            return stripped[len(prefix) :]
    return stripped


def include_deploy_param(name: str) -> bool:
    stripped = strip_deploy_prefix(name)
    if stripped.startswith(TRAINING_ONLY_PREFIXES):
        return False
    return True


def count_total_params(model: torch.nn.Module) -> int:
    return int(sum(param.numel() for param in model.parameters()))


def count_deploy_params(model: torch.nn.Module) -> int:
    return int(
        sum(
            param.numel()
            for name, param in model.named_parameters()
            if include_deploy_param(name)
        )
    )


def module_bucket(name: str) -> str:
    stripped = strip_deploy_prefix(name)
    if stripped.startswith(TRAINING_ONLY_PREFIXES):
        return "training_only"
    if "." not in stripped:
        return stripped
    first, second, *_ = stripped.split(".")
    if first in {"ppg_encoder", "acc_encoder", "ppg_enhancer", "motion_gates", "motion_projections"}:
        return first
    if first in {"acc_to_ppg", "fusion_gate", "fuse_mix"}:
        return first
    if first in {"temporal", "attn_pool", "fusion", "classifier"}:
        return first
    if first in {"wavelet_embed", "quality_embed"}:
        return first
    return f"{first}.{second}"


def module_breakdown(model: torch.nn.Module) -> list[dict[str, object]]:
    buckets: dict[str, dict[str, int]] = {}
    for name, param in model.named_parameters():
        bucket = module_bucket(name)
        item = buckets.setdefault(bucket, {"deploy_params": 0, "total_params": 0})
        item["total_params"] += int(param.numel())
        if include_deploy_param(name):
            item["deploy_params"] += int(param.numel())
    return [
        {
            "module": module,
            "deploy_params": values["deploy_params"],
            "total_params": values["total_params"],
            "deploy_params_m": values["deploy_params"] / 1_000_000.0,
            "total_params_m": values["total_params"] / 1_000_000.0,
        }
        for module, values in sorted(buckets.items())
    ]


def measure_forward_latency(
    model: torch.nn.Module,
    signal: torch.Tensor,
    wavelet: torch.Tensor,
    quality: torch.Tensor,
    warmup: int,
    repeats: int,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(signal, wavelet, quality)["logits"]

        latencies: list[float] = []
        for _ in range(repeats):
            start = time.perf_counter()
            _ = model(signal, wavelet, quality)["logits"]
            end = time.perf_counter()
            latencies.append((end - start) * 1000.0)
    return np.asarray(latencies, dtype=np.float64)


def build_model(
    state: dict[str, torch.Tensor],
    input_ablation: str,
    args: argparse.Namespace,
) -> torch.nn.Module:
    return build_model_from_state(
        state,
        device="cpu",
        model_dim=args.watch_model_dim,
        transformer_layers=args.watch_transformer_layers,
        transformer_heads=args.watch_transformer_heads,
        fusion_hidden_dim=args.watch_fusion_hidden_dim,
        embed_dim=args.watch_embed_dim,
        input_ablation=input_ablation,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "CPU microbenchmark for watch-input variants. Measures only model "
            "forward latency with batch size 1 and synthetic one-window input."
        )
    )
    parser.add_argument(
        "--variant",
        action="append",
        required=True,
        help="NAME=INPUT_ABLATION=CHECKPOINT_DIR, e.g. Pulse-only=ppg_only_refine=artifacts/.../checkpoints",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--subject", type=str, default=None, help="Optional fold checkpoint to benchmark, e.g. P02 or S2.")
    parser.add_argument("--watch-channels", type=int, default=5)
    parser.add_argument("--watch-length", type=int, default=500, help="20s Galaxy at 25Hz=500; WESAD at 32Hz=640.")
    parser.add_argument("--wavelet-dim", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--watch-model-dim", type=int, default=192)
    parser.add_argument("--watch-transformer-layers", type=int, default=2)
    parser.add_argument("--watch-transformer-heads", type=int, default=4)
    parser.add_argument("--watch-fusion-hidden-dim", type=int, default=256)
    parser.add_argument("--watch-embed-dim", type=int, default=160)
    args = parser.parse_args()

    if args.batch_size != 1:
        raise ValueError("This benchmark is intended for batch size = 1.")
    if args.num_threads > 0:
        torch.set_num_threads(args.num_threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

    torch.manual_seed(args.seed)
    signal = torch.randn(args.batch_size, args.watch_channels, args.watch_length)
    wavelet = torch.randn(args.batch_size, args.wavelet_dim)
    quality = torch.ones(args.batch_size, 1)

    rows: list[dict[str, object]] = []
    breakdown_rows: list[dict[str, object]] = []
    for item in args.variant:
        name, input_ablation, checkpoint_dir = parse_variant(item)
        checkpoint = discover_checkpoint(checkpoint_dir, args.subject)
        state = load_state(checkpoint)
        model = build_model(state, input_ablation, args)
        params = count_deploy_params(model)
        total_params = count_total_params(model)
        for part in module_breakdown(model):
            breakdown_rows.append(
                {
                    "variant": name,
                    "input_ablation": input_ablation,
                    "checkpoint": str(checkpoint),
                    **part,
                }
            )
        latencies = measure_forward_latency(
            model,
            signal,
            wavelet,
            quality,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        rows.append(
            {
                "variant": name,
                "input_ablation": input_ablation,
                "checkpoint": str(checkpoint),
                "parameters": params,
                "parameters_m": params / 1_000_000.0,
                "total_model_params": total_params,
                "total_model_params_m": total_params / 1_000_000.0,
                "training_or_unused_params": total_params - params,
                "latency_ms_mean": float(latencies.mean()),
                "latency_ms_std": float(latencies.std(ddof=1)) if len(latencies) > 1 else 0.0,
                "latency_ms_median": float(np.median(latencies)),
                "latency_ms_p10": float(np.percentile(latencies, 10)),
                "latency_ms_p90": float(np.percentile(latencies, 90)),
                "batch_size": args.batch_size,
                "watch_length": args.watch_length,
                "warmup": args.warmup,
                "repeats": args.repeats,
                "num_threads": torch.get_num_threads(),
            }
        )

    result = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "watch_variant_latency.csv"
    md_path = args.output_dir / "watch_variant_latency.md"
    breakdown_path = args.output_dir / "watch_variant_latency_module_breakdown.csv"
    result.to_csv(csv_path, index=False)
    pd.DataFrame(breakdown_rows).to_csv(breakdown_path, index=False)

    table = result.copy()
    table["Parameters"] = table["parameters_m"].map(lambda value: f"{value:.2f}M")
    table["CPU latency per 20-s window"] = table.apply(
        lambda row: f"{row['latency_ms_mean']:.3f} ± {row['latency_ms_std']:.3f} ms",
        axis=1,
    )
    table["CPU latency per 20-s window"] = table.apply(
        lambda row: f"{row['latency_ms_mean']:.3f} +/- {row['latency_ms_std']:.3f} ms",
        axis=1,
    )
    display_rows = table[["variant", "Parameters", "CPU latency per 20-s window"]].rename(
        columns={"variant": "Variant"}
    )
    markdown_lines = [
        "| Variant | Parameters | CPU latency per 20-s window |",
        "|---|---:|---:|",
    ]
    for row in display_rows.to_dict(orient="records"):
        markdown_lines.append(
            f"| {row['Variant']} | {row['Parameters']} | {row['CPU latency per 20-s window']} |"
        )
    markdown = "\n".join(markdown_lines)
    md_path.write_text(markdown + "\n", encoding="utf-8")
    print(markdown)
    print(f"saved_csv={csv_path}")
    print(f"saved_markdown={md_path}")
    print(f"saved_module_breakdown={breakdown_path}")


if __name__ == "__main__":
    main()
