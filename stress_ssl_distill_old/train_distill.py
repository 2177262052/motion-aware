from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .checkpointing import load_compatible_state_dict
from .dataset import StressWindowDataset
from .losses import relational_kd_loss
from .models import StudentNet, TeacherSSLModel


def main() -> None:
    parser = argparse.ArgumentParser(description="Distill the SSL teacher into a lightweight student.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--teacher-path", type=Path, required=True)
    parser.add_argument("--save-path", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--cache-mode", type=str, default="none", choices=["none", "ram", "gpu"])
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

    ds = StressWindowDataset(
        args.manifest,
        split="train",
        ssl=False,
        dataset_root=args.dataset_root,
        cache_mode=args.cache_mode,
        cache_device=args.device,
    )
    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "num_workers": effective_num_workers,
        "pin_memory": effective_pin_memory,
    }
    if effective_num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    loader = DataLoader(ds, **loader_kwargs)
    sample = ds[0]["signal"]

    teacher = TeacherSSLModel(in_channels=sample.shape[0]).to(args.device)
    load_compatible_state_dict(
        teacher,
        checkpoint_path=args.teacher_path,
        device=args.device,
        prefixes=("encoder.", "proj.", "subject_classifier."),
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    student = StudentNet(in_channels=sample.shape[0]).to(args.device)
    optim = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=1e-4)

    student.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        progress = tqdm(loader, desc=f"distill epoch {epoch + 1}/{args.epochs}", leave=True)
        for batch in progress:
            x = batch["signal"].to(args.device, non_blocking=effective_pin_memory)
            with torch.no_grad():
                teacher_out = teacher(x)
            student_out = student(x)
            feat_loss = F.mse_loss(student_out["distill_features"], teacher_out["pooled"])
            rel_loss = relational_kd_loss(student_out["distill_features"], teacher_out["pooled"])
            loss = feat_loss + 0.5 * rel_loss

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optim.step()
            total_loss += loss.item()
            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                feat=f"{feat_loss.item():.4f}",
                rel=f"{rel_loss.item():.4f}",
            )

        avg = total_loss / max(len(loader), 1)
        print(f"epoch={epoch + 1} distill_loss={avg:.4f}")

    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(student.state_dict(), args.save_path)
    print(f"Saved student checkpoint to {args.save_path}")


if __name__ == "__main__":
    main()
