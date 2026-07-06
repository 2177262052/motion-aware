from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .galaxy_dataset import GalaxyWatchWindowDataset
from .galaxy_models import TinyWaveletDistillNet
from .metrics import classification_metrics


DEFAULT_CALM_SESSIONS = [
    "baseline",
    "meditation-1",
    "meditation-2",
    "rest-1",
    "rest-2",
    "rest-3",
    "rest-4",
    "rest-5",
]

DEFAULT_STRESS_SESSIONS = ["tsst-prep"]


def maybe_parse_sessions(values: Sequence[str] | None, fallback: Sequence[str]) -> list[str]:
    if values is None or len(values) == 0:
        return list(fallback)
    return [str(item) for item in values]


def set_random_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_student(student_path: Path, dataset: GalaxyWatchWindowDataset, device: str) -> TinyWaveletDistillNet:
    sample = dataset[0]
    signal_channels = int(sample["signal"].shape[0])
    wavelet_dim = int(sample["wavelet_features"].shape[0])
    model = TinyWaveletDistillNet(in_channels=signal_channels, wavelet_dim=wavelet_dim).to(device)
    state = torch.load(student_path, map_location=device)
    result = model.load_state_dict(state, strict=False)
    if result.missing_keys:
        print(f"student_load_missing_keys={result.missing_keys}")
    if result.unexpected_keys:
        print(f"student_load_unexpected_keys={result.unexpected_keys}")
    return model


def extract_features(
    model: TinyWaveletDistillNet,
    dataset: GalaxyWatchWindowDataset,
    device: str,
    feature_key: str,
) -> tuple[torch.Tensor, torch.Tensor, list[str], list[str]]:
    model.eval()
    feats = []
    labels = []
    subject_ids: list[str] = []
    sessions: list[str] = []
    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="extract prototype features", leave=True):
            item = dataset[idx]
            signal = item["signal"].unsqueeze(0).to(device)
            wavelet = item["wavelet_features"].unsqueeze(0).to(device)
            quality = item["watch_quality"].unsqueeze(0).to(device)
            out = model(signal, wavelet, quality)
            feats.append(out[feature_key].detach().cpu())
            labels.append(int(item["label"]))
            subject_ids.append(str(item["subject_id"]))
            sessions.append(str(item["session"]))
    features = torch.cat(feats, dim=0)
    labels_t = torch.tensor(labels, dtype=torch.long)
    return features, labels_t, subject_ids, sessions


def sample_episode_indices(
    labels: torch.Tensor,
    shots_per_class: int,
    rng: np.random.Generator,
) -> tuple[list[int], list[int]]:
    support_indices: list[int] = []
    for label in [0, 1]:
        candidates = torch.nonzero(labels == label, as_tuple=False).squeeze(1).tolist()
        if len(candidates) < shots_per_class + 1:
            raise ValueError(
                f"Not enough samples for class {label}: need at least {shots_per_class + 1}, got {len(candidates)}"
            )
        selected = rng.choice(candidates, size=shots_per_class, replace=False).tolist()
        support_indices.extend(int(item) for item in selected)

    support_set = set(support_indices)
    query_indices = [idx for idx in range(len(labels)) if idx not in support_set]
    return sorted(support_indices), query_indices


def build_prototypes(features: torch.Tensor, labels: torch.Tensor, normalize: bool) -> torch.Tensor:
    if normalize:
        features = F.normalize(features, dim=1)
    prototypes = []
    for label in [0, 1]:
        cls_feats = features[labels == label]
        proto = cls_feats.mean(dim=0, keepdim=True)
        if normalize:
            proto = F.normalize(proto, dim=1)
        prototypes.append(proto)
    return torch.cat(prototypes, dim=0)


def classify_with_prototypes(
    query_features: torch.Tensor,
    prototypes: torch.Tensor,
    metric: str,
) -> tuple[np.ndarray, np.ndarray]:
    if metric == "cosine":
        query_features = F.normalize(query_features, dim=1)
        prototypes = F.normalize(prototypes, dim=1)
        logits = query_features @ prototypes.T
    elif metric == "euclidean":
        logits = -torch.cdist(query_features, prototypes)
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    preds = logits.argmax(dim=1).cpu().numpy()
    return preds, probs


def aggregate_predictions(
    y_true: list[int],
    y_pred: list[int],
    y_prob: list[float],
    subject_ids: list[str],
    sessions: list[str],
    aggregation: str,
) -> tuple[list[int], list[int], list[float]]:
    if aggregation == "window":
        return y_true, y_pred, y_prob
    if aggregation != "session_mean":
        raise ValueError(f"Unsupported eval aggregation: {aggregation}")

    frame = pd.DataFrame(
        {
            "subject_id": subject_ids,
            "session": sessions,
            "label": y_true,
            "pred": y_pred,
            "prob": y_prob,
        }
    )
    grouped = (
        frame.groupby(["subject_id", "session"], as_index=False)
        .agg(label=("label", "first"), prob=("prob", "mean"))
        .reset_index(drop=True)
    )
    grouped["pred"] = (grouped["prob"] >= 0.5).astype(int)
    return (
        grouped["label"].astype(int).tolist(),
        grouped["pred"].astype(int).tolist(),
        grouped["prob"].astype(float).tolist(),
    )


def collapse_flag(positive_rate: float, low: float = 0.05, high: float = 0.95) -> int:
    return int(positive_rate <= low or positive_rate >= high)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prototype-based few-shot evaluation for Galaxy tiny student.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--student-path", type=Path, required=True)
    parser.add_argument("--shots-per-class", type=int, default=2)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--metric", type=str, default="cosine", choices=["cosine", "euclidean"])
    parser.add_argument("--feature-key", type=str, default="embedding", choices=["embedding", "distill_features"])
    parser.add_argument("--eval-aggregation", type=str, default="window", choices=["window", "session_mean"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--calm-sessions", nargs="*", default=None)
    parser.add_argument("--stress-sessions", nargs="*", default=None)
    parser.add_argument("--wavelet", type=str, default="db4")
    parser.add_argument("--wavelet-level", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_random_seed(args.seed)
    calm_sessions = maybe_parse_sessions(args.calm_sessions, DEFAULT_CALM_SESSIONS)
    stress_sessions = maybe_parse_sessions(args.stress_sessions, DEFAULT_STRESS_SESSIONS)
    include_sessions = calm_sessions + stress_sessions

    test_ds = GalaxyWatchWindowDataset(
        manifest_csv=args.manifest,
        split="test",
        dataset_root=args.dataset_root,
        include_sessions=include_sessions,
        cache_tables=True,
        wavelet=args.wavelet,
        wavelet_level=args.wavelet_level,
    )
    if len(test_ds) == 0:
        raise ValueError("Test split is empty after session filtering.")

    model = load_student(args.student_path, test_ds, args.device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"tiny_student_params={total_params}")
    print(
        f"fewshot_proto_config shots_per_class={args.shots_per_class} "
        f"episodes={args.episodes} metric={args.metric} feature_key={args.feature_key} "
        f"eval_aggregation={args.eval_aggregation} seed={args.seed}"
    )

    features, labels, subject_ids, sessions = extract_features(
        model,
        test_ds,
        args.device,
        feature_key=args.feature_key,
    )
    rng = np.random.default_rng(args.seed)

    all_metrics: list[dict[str, float]] = []
    for episode in range(args.episodes):
        support_indices, query_indices = sample_episode_indices(
            labels,
            shots_per_class=args.shots_per_class,
            rng=rng,
        )
        support_feats = features[support_indices]
        support_labels = labels[support_indices]
        query_feats = features[query_indices]
        query_labels = labels[query_indices].cpu().numpy().tolist()
        query_subjects = [subject_ids[idx] for idx in query_indices]
        query_sessions = [sessions[idx] for idx in query_indices]

        prototypes = build_prototypes(support_feats, support_labels, normalize=(args.metric == "cosine"))
        preds, probs = classify_with_prototypes(query_feats, prototypes, metric=args.metric)
        agg_true, agg_pred, agg_prob = aggregate_predictions(
            query_labels,
            preds.tolist(),
            probs.tolist(),
            query_subjects,
            query_sessions,
            aggregation=args.eval_aggregation,
        )
        metrics = classification_metrics(agg_true, agg_pred, agg_prob)
        metrics["positive_rate"] = float(np.mean(agg_pred)) if len(agg_pred) else 0.0
        metrics["collapse"] = float(collapse_flag(metrics["positive_rate"]))
        all_metrics.append(metrics)
        print(
            f"episode={episode + 1} "
            f"acc={metrics['acc']:.4f} "
            f"balanced_acc={metrics['balanced_acc']:.4f} "
            f"f1={metrics['f1']:.4f} "
            f"auroc={metrics['auroc']:.4f} "
            f"positive_rate={metrics['positive_rate']:.4f}"
        )

    print("prototype few-shot summary")
    for key in ["acc", "balanced_acc", "f1", "auroc", "positive_rate", "collapse"]:
        values = np.array([metric[key] for metric in all_metrics], dtype=np.float64)
        mean = float(values.mean())
        std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        print(f"{key}_mean={mean:.4f} {key}_std={std:.4f}")


if __name__ == "__main__":
    main()
