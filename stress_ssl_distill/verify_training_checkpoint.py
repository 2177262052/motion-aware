from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader


def load_tensor_state(path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported checkpoint format: {path}")
    out: dict[str, torch.Tensor] = {}
    for key, value in checkpoint.items():
        if not torch.is_tensor(value):
            continue
        clean_key = key[len("module.") :] if key.startswith("module.") else key
        out[clean_key] = value
    return out


def uses_scaled_motion(state: dict[str, torch.Tensor]) -> bool:
    return any("scale_logit" in key for key in state)


def build_galaxy_training_model(state: dict[str, torch.Tensor], device: str) -> torch.nn.Module:
    if uses_scaled_motion(state):
        from .run_galaxy_loso_eval_elastic_scaled_motion import install_scaled_motion_film

        install_scaled_motion_film()
    from .galaxy_models import PrivilegedGalaxyTeacherNet

    model = PrivilegedGalaxyTeacherNet(
        watch_backbone="wavelet_guided",
        use_e4_classifier=True,
        use_rhythm_heads=True,
        use_wavelet_head=True,
        watch_enhancement="motion_disentangled" if any("ppg_enhancer" in key for key in state) else "none",
    )
    return model.to(device).eval()


def build_wesad_training_model(state: dict[str, torch.Tensor], device: str) -> torch.nn.Module:
    from .wesad_models_safe_sgpc import WESADPrivilegedTeacherNet

    model = WESADPrivilegedTeacherNet(
        use_wavelet_head=True,
        watch_enhancement="motion_disentangled" if any("ppg_enhancer" in key for key in state) else "none",
    )
    return model.to(device).eval()


def build_training_model(dataset_kind: str, state: dict[str, torch.Tensor], device: str) -> torch.nn.Module:
    if dataset_kind == "galaxy":
        return build_galaxy_training_model(state, device)
    if dataset_kind == "wesad":
        return build_wesad_training_model(state, device)
    raise ValueError(f"Unsupported dataset kind: {dataset_kind}")


def build_dataset(dataset_kind: str, manifest: Path, dataset_root: Path, split: str, include_sessions: list[str]):
    if dataset_kind == "galaxy":
        from .galaxy_dataset import GalaxyPrivilegedWindowDataset

        return GalaxyPrivilegedWindowDataset(
            manifest_csv=manifest,
            split=split,
            dataset_root=dataset_root,
            include_sessions=include_sessions,
            cache_tables=True,
        )
    if dataset_kind == "wesad":
        from .wesad_dataset import WESADPrivilegedWindowDataset

        return WESADPrivilegedWindowDataset(
            manifest_csv=manifest,
            split=split,
            wesad_root=dataset_root,
            include_sessions=include_sessions,
            cache_subjects=4,
        )
    raise ValueError(f"Unsupported dataset kind: {dataset_kind}")


def load_full_model(
    model: torch.nn.Module,
    state: dict[str, torch.Tensor],
    *,
    allowed_unexpected: set[str],
) -> tuple[list[str], list[str], list[tuple[str, tuple[int, ...], tuple[int, ...]]]]:
    model_state = model.state_dict()
    missing = [key for key in model_state if key not in state]
    unexpected = [key for key in state if key not in model_state]
    shape_mismatch = [
        (key, tuple(state[key].shape), tuple(model_state[key].shape))
        for key in sorted(state.keys() & model_state.keys())
        if tuple(state[key].shape) != tuple(model_state[key].shape)
    ]
    loadable = {
        key: value
        for key, value in state.items()
        if key in model_state and tuple(value.shape) == tuple(model_state[key].shape)
    }
    model_state.update(loadable)
    model.load_state_dict(model_state, strict=True)

    serious_unexpected = [key for key in unexpected if key not in allowed_unexpected]
    if missing or shape_mismatch or serious_unexpected:
        print("compatibility=FAIL")
    else:
        print("compatibility=OK")
    return missing, unexpected, shape_mismatch


def print_state_report(model: torch.nn.Module, state: dict[str, torch.Tensor], missing: list[str], unexpected: list[str], shape_mismatch: list[tuple[str, tuple[int, ...], tuple[int, ...]]]) -> None:
    model_state = model.state_dict()
    common = [
        key
        for key in state.keys() & model_state.keys()
        if tuple(state[key].shape) == tuple(model_state[key].shape)
    ]
    print(f"checkpoint_tensors={len(state)}")
    print(f"checkpoint_numel={sum(value.numel() for value in state.values())}")
    print(f"model_tensors={len(model_state)}")
    print(f"model_numel={sum(value.numel() for value in model_state.values())}")
    print(f"common_shape_compatible={len(common)}")
    print(f"common_shape_numel={sum(state[key].numel() for key in common)}")
    print(f"missing_keys={len(missing)}")
    for key in missing[:20]:
        print(f"  missing={key}")
    print(f"unexpected_keys={len(unexpected)}")
    for key in unexpected[:40]:
        print(f"  unexpected={key}")
    print(f"shape_mismatches={len(shape_mismatch)}")
    for key, checkpoint_shape, model_shape in shape_mismatch[:20]:
        print(f"  mismatch={key} checkpoint={checkpoint_shape} model={model_shape}")
    if unexpected:
        print("unexpected_prefix_counts=" + ", ".join(f"{prefix}:{count}" for prefix, count in Counter(key.split(".")[0] for key in unexpected).most_common()))


def smoke_forward(dataset_kind: str, model: torch.nn.Module, batch: dict[str, Any], device: str) -> dict[str, torch.Tensor]:
    if dataset_kind == "galaxy":
        return model(
            batch["watch_signal"].to(device),
            batch["wavelet_features"].to(device),
            batch["watch_quality"].to(device),
            e4_signal=batch["e4_signal"].to(device),
        )
    if dataset_kind == "wesad":
        return model(
            batch["watch_signal"].to(device),
            batch["wavelet_features"].to(device),
            batch["watch_quality"].to(device),
            privileged_signal=batch["privileged_signal"].to(device),
        )
    raise ValueError(f"Unsupported dataset kind: {dataset_kind}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify that a checkpoint still matches the full training-time model.")
    parser.add_argument("--dataset-kind", choices=["galaxy", "wesad"], required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--calm-sessions", nargs="*", default=None)
    parser.add_argument("--stress-sessions", nargs="*", default=None)
    args = parser.parse_args()

    state = load_tensor_state(args.checkpoint)
    model = build_training_model(args.dataset_kind, state, args.device)
    allowed_unexpected = {"correction_mode_id"}
    missing, unexpected, shape_mismatch = load_full_model(
        model,
        state,
        allowed_unexpected=allowed_unexpected,
    )
    print_state_report(model, state, missing, unexpected, shape_mismatch)

    if args.manifest is None or args.dataset_root is None:
        print("forward_smoke=SKIP reason=manifest_or_dataset_root_not_provided")
        return

    if args.dataset_kind == "galaxy":
        calm = args.calm_sessions or ["baseline"]
        stress = args.stress_sessions or ["tsst-prep"]
    else:
        calm = args.calm_sessions or ["baseline"]
        stress = args.stress_sessions or ["stress"]
    dataset = build_dataset(args.dataset_kind, args.manifest, args.dataset_root, args.split, calm + stress)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    batch = next(iter(loader))
    with torch.no_grad():
        out = smoke_forward(args.dataset_kind, model, batch, args.device)
    print("forward_smoke=OK")
    for key in sorted(out):
        value = out[key]
        if torch.is_tensor(value):
            finite = bool(torch.isfinite(value).all().detach().cpu().item())
            print(f"  output={key} shape={tuple(value.shape)} finite={int(finite)}")


if __name__ == "__main__":
    main()
