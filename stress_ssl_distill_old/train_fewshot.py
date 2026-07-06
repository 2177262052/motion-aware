from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from .dataset import StressWindowDataset
from .early_stopping import EarlyStopping
from .metrics import classification_metrics
from .models import StudentNet


def sample_fewshot_indices(dataset: StressWindowDataset, shots_per_class: int) -> List[int]:
    by_class: Dict[int, List[int]] = {0: [], 1: []}
    for idx in range(len(dataset)):
        label = int(dataset.manifest.iloc[idx]["label"])
        by_class[label].append(idx)
    picked = []
    for label, indices in by_class.items():
        if len(indices) < shots_per_class:
            raise ValueError(f"Not enough windows for class {label}: need {shots_per_class}, got {len(indices)}")
        picked.extend(indices[:shots_per_class])
    return picked


def evaluate(model: StudentNet, loader: DataLoader, device: str, pin_memory: bool = False) -> float:
    model.eval()
    y_true = []
    y_pred = []
    y_prob = []
    with torch.no_grad():
        for batch in loader:
            x = batch["signal"].to(device, non_blocking=pin_memory)
            y = batch["label"].to(device, non_blocking=pin_memory)
            logits = model(x)["logits"]
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = logits.argmax(dim=1)
            y_true.extend(y.detach().cpu().tolist())
            y_pred.extend(preds.detach().cpu().tolist())
            y_prob.extend(probs.detach().cpu().tolist())
    return classification_metrics(y_true, y_pred, y_prob)


def main() -> None:
    parser = argparse.ArgumentParser(description="Few-shot adaptation on the held-out subject.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--student-path", type=Path, required=True)
    parser.add_argument("--shots-per-class", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--cache-mode", type=str, default="none", choices=["none", "ram", "gpu"])
    parser.add_argument("--save-path", type=Path, default=None)
    parser.add_argument("--monitor", type=str, default="balanced_acc", choices=["acc", "balanced_acc", "f1", "auroc"])
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--freeze-encoder", action="store_true")
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

    if args.freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False
        if hasattr(model, "distill_proj"):
            for param in model.distill_proj.parameters():
                param.requires_grad = False

    fewshot_indices = sample_fewshot_indices(test_ds, args.shots_per_class)
    train_subset = Subset(test_ds, fewshot_indices)
    train_loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "num_workers": effective_num_workers,
        "pin_memory": effective_pin_memory,
    }
    eval_loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": effective_num_workers,
        "pin_memory": effective_pin_memory,
    }
    if effective_num_workers > 0:
        train_loader_kwargs["persistent_workers"] = True
        train_loader_kwargs["prefetch_factor"] = 2
        eval_loader_kwargs["persistent_workers"] = True
        eval_loader_kwargs["prefetch_factor"] = 2
    train_loader = DataLoader(train_subset, **train_loader_kwargs)
    eval_loader = DataLoader(test_ds, **eval_loader_kwargs)
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters left. Disable --freeze-encoder or check the model.")
    optim = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    if args.freeze_encoder:
        print("few-shot mode: encoder frozen, training classification head only")
    else:
        print("few-shot mode: full-model finetuning")
    stopper = EarlyStopping(
        patience=args.early_stop_patience,
        mode="max",
        min_delta=args.min_delta,
    )
    best_state = None

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        progress = tqdm(train_loader, desc=f"fewshot epoch {epoch + 1}/{args.epochs}", leave=True)
        for batch in progress:
            x = batch["signal"].to(args.device, non_blocking=effective_pin_memory)
            y = batch["label"].to(args.device, non_blocking=effective_pin_memory)
            logits = model(x)["logits"]
            loss = F.cross_entropy(logits, y)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            total_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")
        metrics = evaluate(model, eval_loader, args.device, pin_memory=effective_pin_memory)
        avg = total_loss / max(len(train_loader), 1)
        print(
            f"epoch={epoch + 1} "
            f"fewshot_loss={avg:.4f} "
            f"acc={metrics['acc']:.4f} "
            f"balanced_acc={metrics['balanced_acc']:.4f} "
            f"f1={metrics['f1']:.4f} "
            f"auroc={metrics['auroc']:.4f}"
        )

        score = metrics[args.monitor]
        improved = stopper.step(score, epoch + 1)
        if improved:
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            print(f"new best {args.monitor}={score:.4f} at epoch {epoch + 1}")

        if stopper.should_stop():
            print(
                f"early stopping triggered: no improvement in {args.early_stop_patience} epochs; "
                f"best_{args.monitor}={stopper.best_score:.4f} at epoch {stopper.best_epoch}"
            )
            break

    if args.save_path is not None:
        args.save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_state if best_state is not None else model.state_dict(), args.save_path)
        print(f"Saved few-shot checkpoint to {args.save_path}")
    print(f"best_{args.monitor}={stopper.best_score:.4f} at epoch {stopper.best_epoch}")


if __name__ == "__main__":
    main()
