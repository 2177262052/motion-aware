from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .galaxy_models import (
    MultiScaleResidualBlock,
    ResNet18WatchEncoder,
    ResNet34WatchEncoder,
    ResNet50WatchEncoder,
    WaveletGuidedWatchEncoder,
)


class WESADChestEncoder(nn.Module):
    def __init__(self, in_channels: int = 9, out_dim: int = 160) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            [
                MultiScaleResidualBlock(64, 96, stride=2, kernels=(5, 11, 19), dilations=(1, 1, 2)),
                MultiScaleResidualBlock(96, 128, stride=2, kernels=(5, 11, 19), dilations=(1, 1, 2)),
                MultiScaleResidualBlock(128, 160, stride=2, kernels=(5, 11, 19), dilations=(1, 1, 2)),
            ]
        )
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(160, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        feats = self.stem(signal)
        for block in self.blocks:
            feats = block(feats)
        return self.pool(feats)


def build_watch_encoder(
    watch_backbone: str,
    wavelet_dim: int,
    embed_dim: int,
    model_dim: int,
    transformer_layers: int,
    transformer_heads: int,
    fusion_hidden_dim: int,
    watch_enhancement: str = "none",
    watch_motion_mode: str = "strong",
) -> nn.Module:
    if watch_backbone == "wavelet_guided":
        return WaveletGuidedWatchEncoder(
            wavelet_dim=wavelet_dim,
            model_dim=model_dim,
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads,
            fusion_hidden_dim=fusion_hidden_dim,
            embed_dim=embed_dim,
            watch_enhancement=watch_enhancement,
            watch_motion_mode=watch_motion_mode,
        )
    if watch_enhancement != "none":
        raise ValueError("watch_enhancement is currently only supported for the wavelet_guided backbone.")
    if watch_backbone == "resnet18_1d":
        return ResNet18WatchEncoder(wavelet_dim=wavelet_dim, embed_dim=embed_dim)
    if watch_backbone == "resnet34_1d":
        return ResNet34WatchEncoder(wavelet_dim=wavelet_dim, embed_dim=embed_dim)
    if watch_backbone == "resnet50_1d":
        return ResNet50WatchEncoder(wavelet_dim=wavelet_dim, embed_dim=embed_dim)
    raise ValueError(f"Unsupported watch backbone: {watch_backbone}")


class WESADPrivilegedTeacherNet(nn.Module):
    """WESAD privileged teacher with a deployable wrist BVP/ACC path."""

    def __init__(
        self,
        wavelet_dim: int = 4,
        privileged_channels: int = 9,
        num_classes: int = 2,
        embed_dim: int = 160,
        watch_backbone: str = "wavelet_guided",
        model_dim: int = 192,
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        fusion_hidden_dim: int = 256,
        watch_enhancement: str = "none",
        watch_motion_mode: str = "strong",
        watch_kwargs: Dict[str, object] | None = None,
        teacher_dim: int | None = None,
    ) -> None:
        super().__init__()
        del teacher_dim
        self.watch_backbone = watch_backbone
        self.watch_enhancement = watch_enhancement
        self.watch_motion_mode = watch_motion_mode
        if watch_kwargs is not None:
            kwargs = dict(watch_kwargs)
            kwargs.setdefault("wavelet_dim", wavelet_dim)
            kwargs.setdefault("embed_dim", embed_dim)
            kwargs.setdefault("watch_enhancement", watch_enhancement)
            kwargs.setdefault("watch_motion_mode", watch_motion_mode)
            self.watch_encoder = WaveletGuidedWatchEncoder(**kwargs)
        else:
            self.watch_encoder = build_watch_encoder(
                watch_backbone=watch_backbone,
                wavelet_dim=wavelet_dim,
                embed_dim=embed_dim,
                model_dim=model_dim,
                transformer_layers=transformer_layers,
                transformer_heads=transformer_heads,
                fusion_hidden_dim=fusion_hidden_dim,
                watch_enhancement=watch_enhancement,
                watch_motion_mode=watch_motion_mode,
            )
        self.privileged_encoder = WESADChestEncoder(in_channels=privileged_channels, out_dim=embed_dim)
        self.watch_classifier = nn.Linear(embed_dim, num_classes)
        self.privileged_classifier = nn.Linear(embed_dim, num_classes)
        self.teacher_input_dropout = nn.Dropout(0.1)
        self.teacher_gate = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid(),
        )
        self.teacher_fusion = nn.Sequential(
            nn.Linear(embed_dim * 4, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(256, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )
        self.teacher_out_norm = nn.LayerNorm(embed_dim)
        self.teacher_classifier = nn.Linear(embed_dim, num_classes)
        self.wavelet_predictor = nn.Sequential(
            nn.Linear(embed_dim, 96),
            nn.GELU(),
            nn.Linear(96, wavelet_dim),
        )

    def forward_watch(
        self,
        watch_signal: torch.Tensor,
        wavelet_features: torch.Tensor,
        quality: torch.Tensor,
        baseline_signal: torch.Tensor | None = None,
        baseline_wavelet_features: torch.Tensor | None = None,
        baseline_quality: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        watch_out = self.watch_encoder(
            watch_signal,
            wavelet_features,
            quality,
            baseline_signal=baseline_signal,
            baseline_wavelet_features=baseline_wavelet_features,
            baseline_quality=baseline_quality,
        )
        watch_embedding = watch_out["embedding"]
        logits = self.watch_classifier(watch_embedding)
        return {
            **watch_out,
            "watch_embedding": watch_embedding,
            "base_logits": logits,
            "logits": logits,
            "wavelet_pred": self.wavelet_predictor(watch_embedding),
        }

    def forward(
        self,
        watch_signal: torch.Tensor,
        wavelet_features: torch.Tensor,
        quality: torch.Tensor,
        privileged_signal: torch.Tensor | None = None,
        baseline_watch_signal: torch.Tensor | None = None,
        baseline_wavelet_features: torch.Tensor | None = None,
        baseline_quality: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        out = self.forward_watch(
            watch_signal,
            wavelet_features,
            quality,
            baseline_signal=baseline_watch_signal,
            baseline_wavelet_features=baseline_wavelet_features,
            baseline_quality=baseline_quality,
        )
        if privileged_signal is None:
            return out

        watch_embedding = self.teacher_input_dropout(out["watch_embedding"])
        privileged_embedding = self.teacher_input_dropout(self.privileged_encoder(privileged_signal))
        delta = torch.abs(watch_embedding - privileged_embedding)
        gate = self.teacher_gate(torch.cat([watch_embedding, privileged_embedding, delta], dim=1))
        fusion_input = torch.cat(
            [watch_embedding, privileged_embedding, delta, watch_embedding * privileged_embedding],
            dim=1,
        )
        fused = self.teacher_fusion(fusion_input)
        teacher_embedding = self.teacher_out_norm(watch_embedding + gate * fused)
        out["privileged_embedding"] = privileged_embedding
        out["privileged_logits"] = self.privileged_classifier(privileged_embedding)
        out["teacher_embedding"] = teacher_embedding
        out["teacher_logits"] = self.teacher_classifier(teacher_embedding)
        return out
