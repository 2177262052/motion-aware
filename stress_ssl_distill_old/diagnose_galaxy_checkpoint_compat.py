from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable

import torch

from .galaxy_dataset import GalaxyPrivilegedWindowDataset
from .galaxy_models import PrivilegedGalaxyTeacherNet
from .train_galaxy_privileged import (
    build_loader,
    collect_outputs,
    evaluate_with_threshold,
    select_threshold,
)


def discover_manifests(manifests_dir: Path, subjects: Iterable[str] | None) -> dict[str, Path]:
    requested = {subject.strip() for subject in subjects or [] if subject.strip()}
    manifests: dict[str, Path] = {}
    for path in sorted(manifests_dir.glob("galaxy_*_loso_val.csv")):
        subject = path.stem.replace("galaxy_", "").replace("_loso_val", "")
        if requested and subject not in requested:
            continue
        manifests[subject] = path
    return manifests


def discover_checkpoints(checkpoint_dir: Path, subjects: Iterable[str] | None) -> dict[str, Path]:
    requested = {subject.strip() for subject in subjects or [] if subject.strip()}
    checkpoints: dict[str, Path] = {}
    for path in sorted(checkpoint_dir.glob("*_deploy_watch.pt")):
        subject = path.name.replace("_deploy_watch.pt", "")
        if requested and subject not in requested:
            continue
        checkpoints[subject] = path
    return checkpoints


def normalize_state_dict(raw: object) -> dict[str, torch.Tensor]:
    if isinstance(raw, dict) and "state_dict" in raw and isinstance(raw["state_dict"], dict):
        raw = raw["state_dict"]
    if not isinstance(raw, dict):
        raise TypeError(f"Expected a state_dict-like object, got {type(raw)!r}")

    state: dict[str, torch.Tensor] = {}
    for key, value in raw.items():
        if not isinstance(value, torch.Tensor):
            continue
        normalized_key = str(key)
        if normalized_key.startswith("module."):
            normalized_key = normalized_key[len("module.") :]
        state[normalized_key] = value
    return state


def shape_fingerprint(state: dict[str, torch.Tensor]) -> str:
    payload = {
        key: list(value.shape)
        for key, value in sorted(state.items())
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def state_numel(state: dict[str, torch.Tensor]) -> int:
    return sum(value.numel() for value in state.values())


def build_model(args: argparse.Namespace) -> PrivilegedGalaxyTeacherNet:
    return PrivilegedGalaxyTeacherNet(
        num_phenotypes=args.num_phenotypes,
        watch_backbone=args.watch_backbone,
        watch_enhancement=args.watch_enhancement,
        use_reliability_head=args.reliability_distill_weight > 0.0,
        use_projection_heads=args.align_weight > 0.0,
        use_e4_classifier=args.e4_cls_weight > 0.0,
        use_rhythm_heads=args.rhythm_weight > 0.0,
        use_wavelet_head=args.wavelet_weight > 0.0,
        use_teacher_fused_classifier=args.teacher_fused_cls_weight > 0.0,
    )


def compare_state_dicts(
    checkpoint_state: dict[str, torch.Tensor],
    model_state: dict[str, torch.Tensor],
) -> dict[str, object]:
    checkpoint_keys = set(checkpoint_state)
    model_keys = set(model_state)
    common = sorted(checkpoint_keys & model_keys)
    missing = sorted(model_keys - checkpoint_keys)
    unexpected = sorted(checkpoint_keys - model_keys)
    mismatched = [
        {
            "key": key,
            "checkpoint_shape": tuple(checkpoint_state[key].shape),
            "model_shape": tuple(model_state[key].shape),
        }
        for key in common
        if tuple(checkpoint_state[key].shape) != tuple(model_state[key].shape)
    ]
    strict_compatible = not missing and not unexpected and not mismatched
    shape_compatible = not mismatched
    return {
        "strict_compatible": strict_compatible,
        "shape_compatible": shape_compatible,
        "missing_count": len(missing),
        "unexpected_count": len(unexpected),
        "mismatched_count": len(mismatched),
        "missing_numel": sum(model_state[key].numel() for key in missing),
        "unexpected_numel": sum(checkpoint_state[key].numel() for key in unexpected),
        "missing": missing,
        "unexpected": unexpected,
        "mismatched": mismatched,
    }


def make_loader(
    manifest_path: Path,
    dataset_root: Path,
    split: str,
    args: argparse.Namespace,
) -> torch.utils.data.DataLoader:
    dataset = GalaxyPrivilegedWindowDataset(
        manifest_csv=manifest_path,
        split=split,
        dataset_root=dataset_root,
        include_sessions=list(args.calm_sessions) + list(args.stress_sessions),
        cache_tables=True,
        wavelet=args.wavelet,
        wavelet_level=args.wavelet_level,
        baseline_reference=args.baseline_reference,
    )
    return build_loader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )


def evaluate_checkpoint(
    checkpoint_state: dict[str, torch.Tensor],
    manifest_path: Path,
    dataset_root: Path,
    args: argparse.Namespace,
    strict: bool = True,
) -> dict[str, float]:
    model = build_model(args)
    model.load_state_dict(checkpoint_state, strict=strict)
    model.to(args.device)
    model.eval()

    val_loader = make_loader(manifest_path, dataset_root, "val", args)
    test_loader = make_loader(manifest_path, dataset_root, "test", args)
    if len(val_loader.dataset) == 0:
        val_loader = test_loader

    val_true, val_prob = collect_outputs(
        model,
        val_loader,
        args.device,
        args.pin_memory,
        mode="watch",
        aggregation=args.eval_aggregation,
        baseline_reference=args.baseline_reference,
    )
    threshold, val_metrics = select_threshold(val_true, val_prob, metric=args.monitor)
    test_true, test_prob = collect_outputs(
        model,
        test_loader,
        args.device,
        args.pin_memory,
        mode="watch",
        aggregation=args.eval_aggregation,
        baseline_reference=args.baseline_reference,
    )
    test_metrics = evaluate_with_threshold(test_true, test_prob, threshold=threshold)

    teacher_val_true, teacher_val_prob = collect_outputs(
        model,
        val_loader,
        args.device,
        args.pin_memory,
        mode="teacher",
        aggregation=args.eval_aggregation,
        baseline_reference=args.baseline_reference,
    )
    teacher_threshold, teacher_val_metrics = select_threshold(
        teacher_val_true,
        teacher_val_prob,
        metric=args.monitor,
    )
    teacher_test_true, teacher_test_prob = collect_outputs(
        model,
        test_loader,
        args.device,
        args.pin_memory,
        mode="teacher",
        aggregation=args.eval_aggregation,
        baseline_reference=args.baseline_reference,
    )
    teacher_test_metrics = evaluate_with_threshold(
        teacher_test_true,
        teacher_test_prob,
        threshold=teacher_threshold,
    )

    return {
        "val_watch_balanced_acc": val_metrics["balanced_acc"],
        "val_watch_auroc": val_metrics["auroc"],
        "watch_threshold": threshold,
        "watch_test_acc": test_metrics["acc"],
        "watch_test_balanced_acc": test_metrics["balanced_acc"],
        "watch_test_f1": test_metrics["f1"],
        "watch_test_auroc": test_metrics["auroc"],
        "watch_test_positive_rate": test_metrics["positive_rate"],
        "val_teacher_balanced_acc": teacher_val_metrics["balanced_acc"],
        "val_teacher_auroc": teacher_val_metrics["auroc"],
        "teacher_threshold": teacher_threshold,
        "teacher_test_acc": teacher_test_metrics["acc"],
        "teacher_test_balanced_acc": teacher_test_metrics["balanced_acc"],
        "teacher_test_f1": teacher_test_metrics["f1"],
        "teacher_test_auroc": teacher_test_metrics["auroc"],
        "teacher_test_positive_rate": teacher_test_metrics["positive_rate"],
    }


def format_examples(values: list[object], limit: int = 8) -> str:
    if not values:
        return ""
    shown = values[:limit]
    suffix = "" if len(values) <= limit else f" ... (+{len(values) - limit} more)"
    return "; ".join(str(item) for item in shown) + suffix


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose whether saved Galaxy privileged checkpoints match the current model/eval code."
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--manifests-dir", type=Path, default=None)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--monitor", type=str, default="balanced_acc", choices=["acc", "balanced_acc", "f1", "auroc"])
    parser.add_argument("--eval-aggregation", type=str, default="window", choices=["window", "session_mean"])
    parser.add_argument("--baseline-reference", action="store_true")
    parser.add_argument("--calm-sessions", nargs="*", default=["baseline"])
    parser.add_argument("--stress-sessions", nargs="*", default=["tsst-prep"])
    parser.add_argument("--wavelet", type=str, default="db4")
    parser.add_argument("--wavelet-level", type=int, default=4)
    parser.add_argument(
        "--watch-backbone",
        type=str,
        default="wavelet_guided",
        choices=["wavelet_guided", "resnet18_1d", "resnet34_1d", "resnet50_1d"],
    )
    parser.add_argument(
        "--watch-enhancement",
        type=str,
        default="none",
        choices=["none", "motion_disentangled"],
    )
    parser.add_argument("--num-phenotypes", type=int, default=0)
    parser.add_argument("--reliability-distill-weight", type=float, default=0.0)
    parser.add_argument("--align-weight", type=float, default=0.0)
    parser.add_argument("--e4-cls-weight", type=float, default=0.05)
    parser.add_argument("--rhythm-weight", type=float, default=0.15)
    parser.add_argument("--wavelet-weight", type=float, default=0.05)
    parser.add_argument("--teacher-fused-cls-weight", type=float, default=0.0)
    parser.add_argument(
        "--allow-partial-load",
        action="store_true",
        help="Evaluate shape-compatible checkpoints with strict=False when only missing/unexpected keys differ.",
    )
    args = parser.parse_args()

    if args.checkpoint is None and args.checkpoint_dir is None:
        raise ValueError("Provide either --checkpoint or --checkpoint-dir.")
    if args.checkpoint is not None and args.checkpoint_dir is not None:
        raise ValueError("Provide only one of --checkpoint or --checkpoint-dir.")
    if args.manifest is not None and args.manifests_dir is not None:
        raise ValueError("Provide only one of --manifest or --manifests-dir.")

    if args.checkpoint is not None:
        subject = args.checkpoint.name.replace("_deploy_watch.pt", "").replace(".pt", "")
        checkpoints = {subject: args.checkpoint}
    else:
        checkpoints = discover_checkpoints(args.checkpoint_dir, args.subjects)
    if not checkpoints:
        raise ValueError("No checkpoints found.")

    manifests: dict[str, Path] = {}
    if args.manifest is not None:
        if len(checkpoints) != 1:
            raise ValueError("--manifest can only be used with a single --checkpoint.")
        manifests[next(iter(checkpoints))] = args.manifest
    elif args.manifests_dir is not None:
        manifests = discover_manifests(args.manifests_dir, args.subjects)

    template_model = build_model(args)
    template_state = template_model.state_dict()
    print(f"current_model_params={state_numel(template_state)}")
    print(f"current_model_shape_fingerprint={shape_fingerprint(template_state)}")
    print(f"watch_backbone={args.watch_backbone}")
    print(
        "active_heads="
        f"reliability:{args.reliability_distill_weight > 0.0} "
        f"projection:{args.align_weight > 0.0} "
        f"e4:{args.e4_cls_weight > 0.0} "
        f"rhythm:{args.rhythm_weight > 0.0} "
        f"wavelet:{args.wavelet_weight > 0.0} "
        f"teacher_fused:{args.teacher_fused_cls_weight > 0.0} "
        f"phenotype:{args.num_phenotypes > 0}"
    )

    rows: list[dict[str, object]] = []
    for subject, checkpoint_path in sorted(checkpoints.items()):
        checkpoint_state = normalize_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        comparison = compare_state_dicts(checkpoint_state, template_state)
        row: dict[str, object] = {
            "subject": subject,
            "checkpoint": str(checkpoint_path),
            "checkpoint_params": state_numel(checkpoint_state),
            "checkpoint_shape_fingerprint": shape_fingerprint(checkpoint_state),
            "strict_compatible": int(bool(comparison["strict_compatible"])),
            "shape_compatible": int(bool(comparison["shape_compatible"])),
            "missing_count": comparison["missing_count"],
            "unexpected_count": comparison["unexpected_count"],
            "mismatched_count": comparison["mismatched_count"],
            "missing_numel": comparison["missing_numel"],
            "unexpected_numel": comparison["unexpected_numel"],
        }
        print(
            f"[{subject}] "
            f"params={row['checkpoint_params']} "
            f"fingerprint={row['checkpoint_shape_fingerprint']} "
            f"strict_compatible={bool(comparison['strict_compatible'])} "
            f"missing={comparison['missing_count']}({comparison['missing_numel']} params) "
            f"unexpected={comparison['unexpected_count']}({comparison['unexpected_numel']} params) "
            f"mismatched={comparison['mismatched_count']}"
        )
        if comparison["missing_count"]:
            print("  missing_examples=" + format_examples(comparison["missing"]))
        if comparison["unexpected_count"]:
            print("  unexpected_examples=" + format_examples(comparison["unexpected"]))
        if comparison["mismatched_count"]:
            print("  mismatched_examples=" + format_examples(comparison["mismatched"]))

        manifest_path = manifests.get(subject)
        can_eval = bool(comparison["strict_compatible"]) or (
            args.allow_partial_load and bool(comparison["shape_compatible"])
        )
        if can_eval and manifest_path is not None and args.dataset_root is not None:
            metrics = evaluate_checkpoint(
                checkpoint_state,
                manifest_path,
                args.dataset_root,
                args,
                strict=bool(comparison["strict_compatible"]),
            )
            row.update(metrics)
            row["partial_load"] = int(not bool(comparison["strict_compatible"]))
            print(
                f"  eval watch_test_balanced_acc={metrics['watch_test_balanced_acc']:.4f} "
                f"watch_test_auroc={metrics['watch_test_auroc']:.4f} "
                f"watch_threshold={metrics['watch_threshold']:.4f} "
                f"partial_load={row['partial_load']}"
            )
        elif manifest_path is not None and args.dataset_root is not None:
            print("  eval_skipped=checkpoint/model compatibility failed")

        rows.append(row)

    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = sorted({key for row in rows for key in row})
        with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"saved_csv={args.output_csv}")


if __name__ == "__main__":
    main()
