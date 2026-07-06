from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .dataset import StressWindowDataset
from .metrics import classification_metrics
from .models import StudentNet


def extract_features(model: StudentNet, dataset: StressWindowDataset, device: str) -> tuple[torch.Tensor, torch.Tensor, List[str]]:
    model.eval()
    feats = []
    labels = []
    stages = []
    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="extract test features", leave=True):
            item = dataset[idx]
            x = item["signal"].unsqueeze(0).to(device)
            out = model(x)
            feats.append(out["pooled"].detach().cpu())
            labels.append(int(item["label"]))
            stages.append(str(item["stage"]))
    features = torch.cat(feats, dim=0)
    labels_t = torch.tensor(labels, dtype=torch.long)
    return features, labels_t, stages


def _sample_positive_indices(indices: List[int], stages: List[str], shots: int, rng: np.random.Generator, balance_stages: bool) -> List[int]:
    if not balance_stages:
        return rng.choice(indices, size=shots, replace=False).tolist()

    by_stage: Dict[str, List[int]] = {}
    for idx in indices:
        by_stage.setdefault(stages[idx], []).append(idx)
    stage_names = list(by_stage.keys())
    rng.shuffle(stage_names)
    for stage in stage_names:
        rng.shuffle(by_stage[stage])

    selected: List[int] = []
    cursor = 0
    while len(selected) < shots:
        progressed = False
        for stage in stage_names:
            bucket = by_stage[stage]
            if cursor < len(bucket):
                selected.append(bucket[cursor])
                progressed = True
                if len(selected) == shots:
                    break
        if not progressed:
            break
        cursor += 1

    if len(selected) < shots:
        remaining = [idx for idx in indices if idx not in selected]
        extra = rng.choice(remaining, size=shots - len(selected), replace=False).tolist()
        selected.extend(extra)
    return selected


def sample_episode_indices(
    labels: torch.Tensor,
    stages: List[str],
    shots_per_class: int,
    rng: np.random.Generator,
    balance_stages: bool = True,
) -> tuple[List[int], List[int]]:
    support_indices: List[int] = []
    for label in [0, 1]:
        candidates = torch.nonzero(labels == label, as_tuple=False).squeeze(1).tolist()
        if len(candidates) < shots_per_class:
            raise ValueError(f"Not enough samples for class {label}: need {shots_per_class}, got {len(candidates)}")
        if label == 1:
            selected = _sample_positive_indices(candidates, stages, shots_per_class, rng, balance_stages)
        else:
            selected = rng.choice(candidates, size=shots_per_class, replace=False).tolist()
        support_indices.extend(selected)

    support_set = set(support_indices)
    query_indices = [idx for idx in range(len(labels)) if idx not in support_set]
    return support_indices, query_indices


def build_prototypes(features: torch.Tensor, labels: torch.Tensor, normalize: bool = True) -> torch.Tensor:
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
    metric: str = "cosine",
) -> tuple[np.ndarray, np.ndarray]:
    if metric == "cosine":
        query_features = F.normalize(query_features, dim=1)
        prototypes = F.normalize(prototypes, dim=1)
        logits = query_features @ prototypes.T
    elif metric == "euclidean":
        dists = torch.cdist(query_features, prototypes)
        logits = -dists
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    preds = logits.argmax(dim=1).cpu().numpy()
    return preds, probs


def main() -> None:
    parser = argparse.ArgumentParser(description="Prototype-based few-shot evaluation on the held-out subject.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--student-path", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--cache-mode", type=str, default="none", choices=["none", "ram", "gpu"])
    parser.add_argument("--shots-per-class", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metric", type=str, default="cosine", choices=["cosine", "euclidean"])
    parser.add_argument("--disable-stage-balance", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    test_ds = StressWindowDataset(
        args.manifest,
        split="test",
        ssl=False,
        dataset_root=args.dataset_root,
        cache_mode=args.cache_mode,
        cache_device=args.device,
    )
    sample = test_ds[0]["signal"]
    model = StudentNet(in_channels=sample.shape[0]).to(args.device)
    model.load_state_dict(torch.load(args.student_path, map_location=args.device), strict=False)

    features, labels, stages = extract_features(model, test_ds, args.device)
    rng = np.random.default_rng(args.seed)

    all_metrics: List[Dict[str, float]] = []
    for episode in range(args.episodes):
        support_indices, query_indices = sample_episode_indices(
            labels,
            stages,
            shots_per_class=args.shots_per_class,
            rng=rng,
            balance_stages=not args.disable_stage_balance,
        )
        support_feats = features[support_indices]
        support_labels = labels[support_indices]
        query_feats = features[query_indices]
        query_labels = labels[query_indices].cpu().numpy()

        prototypes = build_prototypes(support_feats, support_labels, normalize=(args.metric == "cosine"))
        preds, probs = classify_with_prototypes(query_feats, prototypes, metric=args.metric)
        metrics = classification_metrics(query_labels, preds, probs)
        all_metrics.append(metrics)
        print(
            f"episode={episode + 1} "
            f"acc={metrics['acc']:.4f} "
            f"balanced_acc={metrics['balanced_acc']:.4f} "
            f"f1={metrics['f1']:.4f} "
            f"auroc={metrics['auroc']:.4f}"
        )

    summary = {}
    for key in ["acc", "balanced_acc", "f1", "auroc"]:
        values = np.array([metric[key] for metric in all_metrics], dtype=np.float64)
        summary[key] = (values.mean(), values.std())

    print("prototype few-shot summary")
    for key, (mean, std) in summary.items():
        print(f"{key}_mean={mean:.4f} {key}_std={std:.4f}")


if __name__ == "__main__":
    main()
