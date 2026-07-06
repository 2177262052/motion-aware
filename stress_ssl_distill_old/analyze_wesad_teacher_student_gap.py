from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .metrics import classification_metrics
from .train_galaxy_watch import build_loader, maybe_parse_sessions, select_threshold
from .wesad_dataset import WESADPrivilegedWindowDataset
from .wesad_models import WESADPrivilegedTeacherNet


DEFAULT_CALM_SESSIONS = ["baseline"]
DEFAULT_STRESS_SESSIONS = ["stress"]


def expected_calibration_error(y_true: list[int], y_prob: list[float], bins: int = 10) -> float:
    if not y_true:
        return float("nan")
    y_true_arr = np.asarray(y_true, dtype=np.float32)
    y_prob_arr = np.asarray(y_prob, dtype=np.float32)
    y_pred_arr = (y_prob_arr >= 0.5).astype(np.float32)
    confidences = np.where(y_pred_arr > 0.5, y_prob_arr, 1.0 - y_prob_arr)
    accuracies = (y_pred_arr == y_true_arr).astype(np.float32)

    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for idx in range(bins):
        lo = edges[idx]
        hi = edges[idx + 1]
        if idx == bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        if not np.any(mask):
            continue
        acc = float(np.mean(accuracies[mask]))
        conf = float(np.mean(confidences[mask]))
        frac = float(np.mean(mask))
        ece += frac * abs(acc - conf)
    return float(ece)


def evaluate_with_threshold(y_true: list[int], y_prob: list[float], threshold: float) -> dict[str, float]:
    y_pred = [1 if prob >= threshold else 0 for prob in y_prob]
    metrics = classification_metrics(y_true, y_pred, y_prob)
    metrics["threshold"] = float(threshold)
    metrics["positive_rate"] = float(np.mean(y_pred)) if y_pred else 0.0
    metrics["ece"] = expected_calibration_error(y_true, y_prob)
    return metrics


def collect_prediction_frame(
    model: WESADPrivilegedTeacherNet,
    loader,
    device: str,
    pin_memory: bool,
    baseline_reference: bool = False,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            watch_signal = batch["watch_signal"].to(device, non_blocking=pin_memory)
            privileged_signal = batch["privileged_signal"].to(device, non_blocking=pin_memory)
            wavelet = batch["wavelet_features"].to(device, non_blocking=pin_memory)
            quality = batch["watch_quality"].to(device, non_blocking=pin_memory)
            labels = batch["label"].to(device, non_blocking=pin_memory).long()
            baseline_kwargs = {}
            if baseline_reference:
                baseline_kwargs = {
                    "baseline_watch_signal": batch["baseline_watch_signal"].to(device, non_blocking=pin_memory),
                    "baseline_wavelet_features": batch["baseline_wavelet_features"].to(device, non_blocking=pin_memory),
                    "baseline_quality": batch["baseline_watch_quality"].to(device, non_blocking=pin_memory),
                }

            out = model(
                watch_signal,
                wavelet,
                quality,
                privileged_signal=privileged_signal,
                **baseline_kwargs,
            )
            watch_probs = torch.softmax(out["logits"], dim=1)[:, 1].detach().cpu().numpy()
            teacher_probs = torch.softmax(out["teacher_logits"], dim=1)[:, 1].detach().cpu().numpy()
            watch_scores = (out["logits"][:, 1] - out["logits"][:, 0]).detach().cpu().numpy()
            teacher_scores = (out["teacher_logits"][:, 1] - out["teacher_logits"][:, 0]).detach().cpu().numpy()
            cosine = torch.nn.functional.cosine_similarity(
                out["watch_embedding"],
                out["teacher_embedding"],
                dim=1,
            ).detach().cpu().numpy()
            align_cosine = torch.nn.functional.cosine_similarity(
                out["watch_align"],
                out["teacher_align"],
                dim=1,
            ).detach().cpu().numpy()
            l2 = torch.norm(out["watch_embedding"] - out["teacher_embedding"], dim=1).detach().cpu().numpy()

            for idx in range(labels.shape[0]):
                rows.append(
                    {
                        "subject_id": str(batch["subject_id"][idx]),
                        "subject_index": int(batch["subject_index"][idx]),
                        "session": str(batch["session"][idx]),
                        "group_name": str(batch["group_name"][idx]),
                        "window_start_ms": int(batch["window_start_ms"][idx]),
                        "window_end_ms": int(batch["window_end_ms"][idx]),
                        "label": int(labels[idx].item()),
                        "watch_prob": float(watch_probs[idx]),
                        "teacher_prob": float(teacher_probs[idx]),
                        "watch_score": float(watch_scores[idx]),
                        "teacher_score": float(teacher_scores[idx]),
                        "teacher_minus_watch_prob": float(teacher_probs[idx] - watch_probs[idx]),
                        "teacher_minus_watch_score": float(teacher_scores[idx] - watch_scores[idx]),
                        "watch_teacher_cosine": float(cosine[idx]),
                        "watch_teacher_align_cosine": float(align_cosine[idx]),
                        "watch_teacher_l2": float(l2[idx]),
                    }
                )
    return pd.DataFrame(rows)


def per_subject_metrics(frame: pd.DataFrame, watch_threshold: float, teacher_threshold: float) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for subject_id, group in frame.groupby("subject_id", sort=True):
        y_true = group["label"].astype(int).tolist()
        watch_prob = group["watch_prob"].astype(float).tolist()
        teacher_prob = group["teacher_prob"].astype(float).tolist()
        watch_metrics = evaluate_with_threshold(y_true, watch_prob, watch_threshold)
        teacher_metrics = evaluate_with_threshold(y_true, teacher_prob, teacher_threshold)
        rows.append(
            {
                "subject_id": subject_id,
                "num_windows": len(group),
                "positive_rate_true": float(np.mean(group["label"].astype(float))),
                "watch_balanced_acc": watch_metrics["balanced_acc"],
                "teacher_balanced_acc": teacher_metrics["balanced_acc"],
                "watch_auroc": watch_metrics["auroc"],
                "teacher_auroc": teacher_metrics["auroc"],
                "watch_f1": watch_metrics["f1"],
                "teacher_f1": teacher_metrics["f1"],
                "watch_positive_rate": watch_metrics["positive_rate"],
                "teacher_positive_rate": teacher_metrics["positive_rate"],
                "watch_teacher_prob_corr": float(group["watch_prob"].corr(group["teacher_prob"])),
                "watch_teacher_cosine_mean": float(group["watch_teacher_cosine"].mean()),
                "watch_teacher_align_cosine_mean": float(group["watch_teacher_align_cosine"].mean()),
                "watch_teacher_l2_mean": float(group["watch_teacher_l2"].mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze teacher-student gap for a trained WESAD privileged model.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", "--wesad-root", dest="dataset_root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--cache-subjects", type=int, default=15)
    parser.add_argument("--wavelet", type=str, default="db4")
    parser.add_argument("--wavelet-level", type=int, default=4)
    parser.add_argument("--baseline-reference", action="store_true")
    parser.add_argument("--calm-sessions", nargs="*", default=None)
    parser.add_argument("--stress-sessions", nargs="*", default=None)
    parser.add_argument("--watch-backbone", type=str, default="wavelet_guided", choices=["wavelet_guided", "resnet18_1d", "resnet34_1d", "resnet50_1d"])
    parser.add_argument("--watch-model-dim", type=int, default=192)
    parser.add_argument("--watch-transformer-layers", type=int, default=2)
    parser.add_argument("--watch-transformer-heads", type=int, default=4)
    parser.add_argument("--watch-fusion-hidden-dim", type=int, default=256)
    parser.add_argument("--watch-embed-dim", type=int, default=160)
    parser.add_argument("--align-proj-dim", type=int, default=128)
    args = parser.parse_args()

    calm_sessions = maybe_parse_sessions(args.calm_sessions, DEFAULT_CALM_SESSIONS)
    stress_sessions = maybe_parse_sessions(args.stress_sessions, DEFAULT_STRESS_SESSIONS)
    include_sessions = calm_sessions + stress_sessions

    val_ds = WESADPrivilegedWindowDataset(
        manifest_csv=args.manifest,
        split="val",
        wesad_root=args.dataset_root,
        include_sessions=include_sessions,
        cache_subjects=args.cache_subjects,
        wavelet=args.wavelet,
        wavelet_level=args.wavelet_level,
        baseline_reference=args.baseline_reference,
    )
    test_ds = WESADPrivilegedWindowDataset(
        manifest_csv=args.manifest,
        split="test",
        wesad_root=args.dataset_root,
        include_sessions=include_sessions,
        cache_subjects=args.cache_subjects,
        wavelet=args.wavelet,
        wavelet_level=args.wavelet_level,
        baseline_reference=args.baseline_reference,
    )
    if len(test_ds) == 0:
        raise ValueError("Test split is empty after session filtering.")

    val_loader = build_loader(val_ds, args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory) if len(val_ds) > 0 else None
    test_loader = build_loader(test_ds, args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)

    model = WESADPrivilegedTeacherNet(
        watch_backbone=args.watch_backbone,
        embed_dim=args.watch_embed_dim,
        align_dim=args.align_proj_dim,
        model_dim=args.watch_model_dim,
        transformer_layers=args.watch_transformer_layers,
        transformer_heads=args.watch_transformer_heads,
        fusion_hidden_dim=args.watch_fusion_hidden_dim,
    ).to(args.device)
    state_dict = torch.load(args.checkpoint, map_location=args.device)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    ignored_missing = [
        key
        for key in missing_keys
        if key.startswith("watch_contrastive_head.") or key.startswith("reliability_head.")
    ]
    remaining_missing = [key for key in missing_keys if key not in ignored_missing]
    if remaining_missing or unexpected_keys:
        raise RuntimeError(
            "Checkpoint does not match WESAD model. "
            f"missing={remaining_missing} unexpected={unexpected_keys}"
        )
    model.eval()

    val_frame = collect_prediction_frame(model, val_loader, args.device, args.pin_memory, baseline_reference=args.baseline_reference) if val_loader is not None else pd.DataFrame()
    test_frame = collect_prediction_frame(model, test_loader, args.device, args.pin_memory, baseline_reference=args.baseline_reference)

    if not val_frame.empty:
        watch_threshold, _ = select_threshold(val_frame["label"].astype(int).tolist(), val_frame["watch_prob"].astype(float).tolist(), metric="balanced_acc")
        teacher_threshold, _ = select_threshold(val_frame["label"].astype(int).tolist(), val_frame["teacher_prob"].astype(float).tolist(), metric="balanced_acc")
    else:
        watch_threshold = 0.5
        teacher_threshold = 0.5

    watch_metrics = evaluate_with_threshold(
        test_frame["label"].astype(int).tolist(),
        test_frame["watch_prob"].astype(float).tolist(),
        watch_threshold,
    )
    teacher_metrics = evaluate_with_threshold(
        test_frame["label"].astype(int).tolist(),
        test_frame["teacher_prob"].astype(float).tolist(),
        teacher_threshold,
    )
    watch_with_teacher_threshold = evaluate_with_threshold(
        test_frame["label"].astype(int).tolist(),
        test_frame["watch_prob"].astype(float).tolist(),
        teacher_threshold,
    )
    teacher_with_watch_threshold = evaluate_with_threshold(
        test_frame["label"].astype(int).tolist(),
        test_frame["teacher_prob"].astype(float).tolist(),
        watch_threshold,
    )

    test_frame["watch_pred"] = (test_frame["watch_prob"] >= watch_threshold).astype(int)
    test_frame["teacher_pred"] = (test_frame["teacher_prob"] >= teacher_threshold).astype(int)
    test_frame["watch_correct"] = (test_frame["watch_pred"] == test_frame["label"]).astype(int)
    test_frame["teacher_correct"] = (test_frame["teacher_pred"] == test_frame["label"]).astype(int)
    test_frame["teacher_only_correct"] = ((test_frame["teacher_correct"] == 1) & (test_frame["watch_correct"] == 0)).astype(int)
    test_frame["watch_only_correct"] = ((test_frame["watch_correct"] == 1) & (test_frame["teacher_correct"] == 0)).astype(int)

    per_subject = per_subject_metrics(test_frame, watch_threshold, teacher_threshold)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    val_path = output_dir / "val_predictions.csv"
    test_path = output_dir / "test_predictions.csv"
    per_subject_path = output_dir / "per_subject_summary.csv"
    summary_path = output_dir / "gap_summary.txt"

    if not val_frame.empty:
        val_frame.to_csv(val_path, index=False)
    test_frame.to_csv(test_path, index=False)
    per_subject.to_csv(per_subject_path, index=False)

    overall_corr = float(test_frame["watch_prob"].corr(test_frame["teacher_prob"]))
    summary_lines = [
        f"watch_threshold={watch_threshold:.4f}",
        f"teacher_threshold={teacher_threshold:.4f}",
        "",
        f"watch balanced_acc={watch_metrics['balanced_acc']:.4f} auroc={watch_metrics['auroc']:.4f} f1={watch_metrics['f1']:.4f} ece={watch_metrics['ece']:.4f}",
        f"teacher balanced_acc={teacher_metrics['balanced_acc']:.4f} auroc={teacher_metrics['auroc']:.4f} f1={teacher_metrics['f1']:.4f} ece={teacher_metrics['ece']:.4f}",
        "",
        f"watch@teacher_threshold balanced_acc={watch_with_teacher_threshold['balanced_acc']:.4f} f1={watch_with_teacher_threshold['f1']:.4f}",
        f"teacher@watch_threshold balanced_acc={teacher_with_watch_threshold['balanced_acc']:.4f} f1={teacher_with_watch_threshold['f1']:.4f}",
        "",
        f"watch_teacher_prob_corr={overall_corr:.4f}",
        f"teacher_only_correct_rate={float(test_frame['teacher_only_correct'].mean()):.4f}",
        f"watch_only_correct_rate={float(test_frame['watch_only_correct'].mean()):.4f}",
        f"watch_teacher_cosine_mean={float(test_frame['watch_teacher_cosine'].mean()):.4f}",
        f"watch_teacher_align_cosine_mean={float(test_frame['watch_teacher_align_cosine'].mean()):.4f}",
        f"watch_teacher_l2_mean={float(test_frame['watch_teacher_l2'].mean()):.4f}",
        "",
        f"saved_test_predictions={test_path}",
        f"saved_per_subject_summary={per_subject_path}",
    ]
    summary_text = "\n".join(summary_lines) + "\n"
    summary_path.write_text(summary_text, encoding="utf-8")
    print(summary_text, end="")


if __name__ == "__main__":
    main()
