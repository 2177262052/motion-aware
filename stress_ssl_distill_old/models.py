from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, coeff: float):
        ctx.coeff = coeff
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.coeff * grad_output, None


def grad_reverse(x: torch.Tensor, coeff: float = 1.0) -> torch.Tensor:
    return GradReverse.apply(x, coeff)


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, stride: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=kernel // 2),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TemporalEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        widths: Tuple[int, ...] = (64, 128, 192, 256),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        blocks = []
        c = in_channels
        for width in widths:
            blocks.append(ConvBlock(c, width, kernel=7, stride=2, dropout=dropout))
            c = width
        self.backbone = nn.Sequential(*blocks)
        self.proj = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(c, c),
            nn.GELU(),
            nn.Linear(c, c),
        )
        self.feature_dim = c

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feats = self.backbone(x)
        pooled = self.proj(feats)
        return {"sequence": feats, "pooled": pooled}


class MaskedDecoder(nn.Module):
    def __init__(self, hidden_dim: int, out_channels: int) -> None:
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.ConvTranspose1d(hidden_dim, hidden_dim // 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.ConvTranspose1d(hidden_dim // 2, hidden_dim // 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.GELU(),
            nn.ConvTranspose1d(hidden_dim // 4, hidden_dim // 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(hidden_dim // 8),
            nn.GELU(),
            nn.Conv1d(hidden_dim // 8, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor, target_len: int) -> torch.Tensor:
        out = self.decoder(x)
        if out.shape[-1] > target_len:
            out = out[..., :target_len]
        elif out.shape[-1] < target_len:
            out = F.pad(out, (0, target_len - out.shape[-1]))
        return out


class TeacherSSLModel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        widths: Tuple[int, ...] = (64, 128, 192, 256),
        num_subjects: int = 0,
        adv_coeff: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder = TemporalEncoder(in_channels=in_channels, widths=widths)
        self.decoder = MaskedDecoder(hidden_dim=self.encoder.feature_dim, out_channels=in_channels)
        self.adv_coeff = adv_coeff
        self.subject_classifier = None
        if num_subjects > 0:
            self.subject_classifier = nn.Sequential(
                nn.Linear(self.encoder.feature_dim, self.encoder.feature_dim),
                nn.GELU(),
                nn.Linear(self.encoder.feature_dim, num_subjects),
            )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        enc = self.encoder(x)
        recon = self.decoder(enc["sequence"], target_len=x.shape[-1])
        out = {**enc, "reconstruction": recon}
        if self.subject_classifier is not None:
            reversed_features = grad_reverse(enc["pooled"], self.adv_coeff)
            out["subject_logits"] = self.subject_classifier(reversed_features)
        return out


class StudentNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        widths: Tuple[int, ...] = (32, 64, 96, 128),
        num_classes: int = 2,
        distill_dim: int = 256,
    ) -> None:
        super().__init__()
        self.encoder = TemporalEncoder(in_channels=in_channels, widths=widths, dropout=0.05)
        self.distill_proj = nn.Linear(self.encoder.feature_dim, distill_dim)
        self.head = nn.Linear(self.encoder.feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        enc = self.encoder(x)
        logits = self.head(enc["pooled"])
        distill_features = self.distill_proj(enc["pooled"])
        return {**enc, "logits": logits, "distill_features": distill_features}


class SupervisedTeacherNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        widths: Tuple[int, ...] = (64, 128, 192, 256),
        num_classes: int = 2,
        distill_dim: int = 256,
    ) -> None:
        super().__init__()
        self.encoder = TemporalEncoder(in_channels=in_channels, widths=widths, dropout=0.1)
        self.distill_proj = nn.Identity() if self.encoder.feature_dim == distill_dim else nn.Linear(self.encoder.feature_dim, distill_dim)
        self.head = nn.Linear(self.encoder.feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        enc = self.encoder(x)
        logits = self.head(enc["pooled"])
        distill_features = self.distill_proj(enc["pooled"])
        return {**enc, "logits": logits, "distill_features": distill_features}
