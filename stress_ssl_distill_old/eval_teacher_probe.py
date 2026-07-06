from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .checkpointing import load_compatible_state_dict
from .dataset import StressWindowDataset
from .models import TeacherSSLModel


def build_loader(dataset: StressWindowDataset, batch_size: int, num_workers: int, pin_memory: bool, shuffle: bool) -> DataLoader:
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **kwargs)


def extract_features(model: TeacherSSLModel, loader: DataLoader, device: str, pin_memory: bool) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    feats = []
    labels = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="extract features", leave=True):
            x = batch["signal"].to(device, non_blocking=pin_memory)
            y = batch["label"].cpu().numpy()
            pooled = model(x)["pooled"].detach().cpu().numpy()
            feats.append(pooled)
            labels.append(y)
    return np.concatenate(feats, axis=0), np.concatenate(labels, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a frozen teacher encoder with a linear probe.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--teacher-path", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--cache-mode", type=str, default="none", choices=["none", "ram", "gpu"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    effective_num_workers = args.num_workers
    effective_pin_memory = args.pin_memory
    if args.cache_mode != "none":
        effective_num_workers = 0
        if args.cache_mode == "gpu":
            effective_pin_memory = False

    train_ds = StressWindowDataset(
        args.manifest,
        split="train",
        ssl=False,
        dataset_root=args.dataset_root,
        cache_mode=args.cache_mode,
        cache_device=args.device,
    )
    test_ds = StressWindowDataset(
        args.manifest,
        split="test",
        ssl=False,
        dataset_root=args.dataset_root,
        cache_mode=args.cache_mode,
        cache_device=args.device,
    )

    sample = train_ds[0]["signal"]
    teacher = TeacherSSLModel(in_channels=sample.shape[0]).to(args.device)
    load_compatible_state_dict(
        teacher,
        checkpoint_path=args.teacher_path,
        device=args.device,
        prefixes=("encoder.", "proj.", "subject_classifier."),
    )

    train_loader = build_loader(train_ds, args.batch_size, effective_num_workers, effective_pin_memory, shuffle=False)
    test_loader = build_loader(test_ds, args.batch_size, effective_num_workers, effective_pin_memory, shuffle=False)

    x_train, y_train = extract_features(teacher, train_loader, args.device, effective_pin_memory)
    x_test, y_test = extract_features(teacher, test_loader, args.device, effective_pin_memory)

    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
    )
    clf.fit(x_train, y_train)

    probs = clf.predict_proba(x_test)[:, 1]
    preds = (probs >= 0.5).astype(np.int64)

    metrics = {
        "acc": accuracy_score(y_test, preds),
        "balanced_acc": balanced_accuracy_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "auroc": roc_auc_score(y_test, probs),
    }
    print("teacher linear probe results")
    for key, value in metrics.items():
        print(f"{key}={value:.4f}")


if __name__ == "__main__":
    main()
