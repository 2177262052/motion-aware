from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .dataset import StressWindowDataset
from .early_stopping import EarlyStopping
from .metrics import classification_metrics
from .models import StudentNet, SupervisedTeacherNet


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


def evaluate(model: StudentNet, loader: DataLoader, device: str, pin_memory: bool = False) -> dict[str, float]:
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


def distillation_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
    return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Distill a supervised teacher into a lightweight student.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--teacher-path", type=Path, required=True)
    parser.add_argument("--save-path", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--cache-mode", type=str, default="none", choices=["none", "ram", "gpu"])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--hard-weight", type=float, default=1.0)
    parser.add_argument("--kd-weight", type=float, default=1.0)
    parser.add_argument("--feat-weight", type=float, default=0.25)
    parser.add_argument("--monitor", type=str, default="balanced_acc", choices=["acc", "balanced_acc", "f1", "auroc"])
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=0.0)
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
    train_loader = build_loader(train_ds, args.batch_size, effective_num_workers, effective_pin_memory, shuffle=True)
    test_loader = build_loader(test_ds, args.batch_size, effective_num_workers, effective_pin_memory, shuffle=False)

    sample = train_ds[0]["signal"]
    teacher = SupervisedTeacherNet(in_channels=sample.shape[0]).to(args.device)
    teacher.load_state_dict(torch.load(args.teacher_path, map_location=args.device))
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    student = StudentNet(in_channels=sample.shape[0]).to(args.device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=1e-4)

    class_counts = train_ds.manifest["label"].value_counts().sort_index()
    neg = float(class_counts.get(0, 1.0))
    pos = float(class_counts.get(1, 1.0))
    class_weights = torch.tensor([1.0, neg / max(pos, 1.0)], device=args.device, dtype=torch.float32)

    stopper = EarlyStopping(
        patience=args.early_stop_patience,
        mode="max",
        min_delta=args.min_delta,
    )
    best_state = None

    for epoch in range(args.epochs):
        student.train()
        total_loss = 0.0
        progress = tqdm(train_loader, desc=f"distill-sup epoch {epoch + 1}/{args.epochs}", leave=True)
        for batch in progress:
            x = batch["signal"].to(args.device, non_blocking=effective_pin_memory)
            y = batch["label"].to(args.device, non_blocking=effective_pin_memory).long()

            with torch.no_grad():
                teacher_out = teacher(x)
            student_out = student(x)

            hard_loss = F.cross_entropy(student_out["logits"], y, weight=class_weights)
            kd_loss = distillation_kl(student_out["logits"], teacher_out["logits"], args.temperature)
            feat_loss = F.mse_loss(student_out["distill_features"], teacher_out["distill_features"])
            loss = args.hard_weight * hard_loss + args.kd_weight * kd_loss + args.feat_weight * feat_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                hard=f"{hard_loss.item():.4f}",
                kd=f"{kd_loss.item():.4f}",
                feat=f"{feat_loss.item():.4f}",
            )

        avg_loss = total_loss / max(len(train_loader), 1)
        metrics = evaluate(student, test_loader, args.device, pin_memory=effective_pin_memory)
        print(
            f"epoch={epoch + 1} "
            f"distill_loss={avg_loss:.4f} "
            f"acc={metrics['acc']:.4f} "
            f"balanced_acc={metrics['balanced_acc']:.4f} "
            f"f1={metrics['f1']:.4f} "
            f"auroc={metrics['auroc']:.4f}"
        )

        score = metrics[args.monitor]
        improved = stopper.step(score, epoch + 1)
        if improved:
            best_state = {key: value.detach().cpu() for key, value in student.state_dict().items()}
            print(f"new best {args.monitor}={score:.4f} at epoch {epoch + 1}")

        if stopper.should_stop():
            print(
                f"early stopping triggered: no improvement in {args.early_stop_patience} epochs; "
                f"best_{args.monitor}={stopper.best_score:.4f} at epoch {stopper.best_epoch}"
            )
            break

    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state if best_state is not None else student.state_dict(), args.save_path)
    print(f"best_{args.monitor}={stopper.best_score:.4f} at epoch {stopper.best_epoch}")
    print(f"Saved distilled student checkpoint to {args.save_path}")


if __name__ == "__main__":
    main()
