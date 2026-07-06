from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .dataset import StressWindowDataset
from .losses import masked_reconstruction_loss, nt_xent_loss
from .models import TeacherSSLModel


def make_mask(x: torch.Tensor, ratio: float = 0.3) -> torch.Tensor:
    mask = (torch.rand_like(x) < ratio).float()
    return mask


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain a self-supervised teacher on STRESS windows.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--save-path", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--recon-weight", type=float, default=0.1)
    parser.add_argument("--adv-weight", type=float, default=0.1)
    parser.add_argument("--adv-coeff", type=float, default=1.0)
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
        ssl=True,
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
    num_subjects = int(ds.manifest["subject_index"].max()) + 1
    model = TeacherSSLModel(
        in_channels=sample.shape[0],
        num_subjects=num_subjects,
        adv_coeff=args.adv_coeff,
    ).to(args.device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        progress = tqdm(loader, desc=f"ssl epoch {epoch + 1}/{args.epochs}", leave=True)
        for batch in progress:
            view1 = batch["view1"].to(args.device, non_blocking=effective_pin_memory)
            view2 = batch["view2"].to(args.device, non_blocking=effective_pin_memory)
            subject_index = batch["subject_index"].to(args.device, non_blocking=effective_pin_memory)
            mask = make_mask(view1)
            masked_input = view1 * (1.0 - mask)

            out1 = model(masked_input)
            out2 = model(view2)
            contrastive = nt_xent_loss(out1["pooled"], out2["pooled"])
            recon = masked_reconstruction_loss(out1["reconstruction"], view1, mask)
            adv = torch.tensor(0.0, device=args.device)
            if "subject_logits" in out1:
                adv = F.cross_entropy(out1["subject_logits"], subject_index)
            loss = contrastive + args.recon_weight * recon + args.adv_weight * adv

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            total_loss += loss.item()
            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                contrastive=f"{contrastive.item():.4f}",
                recon=f"{recon.item():.4f}",
                adv=f"{adv.item():.4f}",
            )

        avg = total_loss / max(len(loader), 1)
        print(f"epoch={epoch + 1} ssl_loss={avg:.4f}")

    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.save_path)
    print(f"Saved teacher checkpoint to {args.save_path}")


if __name__ == "__main__":
    main()
