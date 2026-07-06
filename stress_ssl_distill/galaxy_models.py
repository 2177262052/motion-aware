from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int = 1, dilation: int = 1) -> None:
        super().__init__()
        padding = ((kernel_size - 1) // 2) * dilation
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, in_ch, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=in_ch, bias=False),
            nn.BatchNorm1d(in_ch),
            nn.GELU(),
            nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SqueezeExcite1d(nn.Module):
    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


class MultiScaleResidualBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 1,
        kernels: Tuple[int, ...] = (7, 15, 31),
        dilations: Tuple[int, ...] = (1, 1, 2),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        branch_out = out_ch // len(kernels)
        extras = out_ch - branch_out * len(kernels)
        widths = [branch_out + (1 if idx < extras else 0) for idx in range(len(kernels))]

        self.branches = nn.ModuleList(
            [
                DepthwiseSeparableConv1d(in_ch, width, kernel_size=kernel, stride=stride, dilation=dilation)
                for width, kernel, dilation in zip(widths, kernels, dilations)
            ]
        )
        self.mix = nn.Sequential(
            nn.Conv1d(out_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.se = SqueezeExcite1d(out_ch)
        self.residual = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.residual = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        out = torch.cat([branch(x) for branch in self.branches], dim=1)
        out = self.mix(out)
        out = self.se(out)
        return F.gelu(out + residual)


class MotionFiLM(nn.Module):
    def __init__(
        self,
        cond_dim: int,
        target_dim: int,
        motion_mode: str = "strong",
        residual_scale_init: float = 0.05,
        scale_logit_init: float = -2.0,
    ) -> None:
        super().__init__()
        if motion_mode not in {"strong", "residual", "scaled"}:
            raise ValueError(f"Unsupported motion mode: {motion_mode}")
        self.motion_mode = motion_mode
        self.to_gamma = nn.Sequential(
            nn.Linear(cond_dim, target_dim),
            nn.GELU(),
            nn.Linear(target_dim, target_dim),
        )
        self.to_beta = nn.Sequential(
            nn.Linear(cond_dim, target_dim),
            nn.GELU(),
            nn.Linear(target_dim, target_dim),
        )
        if motion_mode == "residual":
            self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))
        else:
            self.register_parameter("residual_scale", None)
        if motion_mode == "scaled":
            self.scale_logit = nn.Parameter(torch.tensor(float(scale_logit_init)))
        else:
            self.register_parameter("scale_logit", None)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        gamma = self.to_gamma(condition).unsqueeze(-1)
        beta = self.to_beta(condition).unsqueeze(-1)
        if self.motion_mode == "scaled":
            scale = torch.sigmoid(self.scale_logit)
            return x * (1.0 + scale * torch.tanh(gamma)) + scale * beta
        modulated = x * (1.0 + torch.tanh(gamma)) + beta
        if self.motion_mode == "residual":
            return x + torch.tanh(self.residual_scale) * (modulated - x)
        return modulated


class MotionDisentangledPPGEnhancer(nn.Module):
    """Use motion as artifact context while preserving PPG physiology as the main evidence."""

    def __init__(self, ppg_dim: int, acc_dim: int, wavelet_dim: int = 96, scale_init: float = 0.05) -> None:
        super().__init__()
        self.acc_to_ppg = nn.Sequential(
            nn.Conv1d(acc_dim, ppg_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(ppg_dim),
            nn.GELU(),
        )
        joint_dim = ppg_dim * 3
        self.motion_mask = nn.Sequential(
            DepthwiseSeparableConv1d(joint_dim, ppg_dim, kernel_size=5),
            nn.Conv1d(ppg_dim, ppg_dim, kernel_size=1),
            nn.Sigmoid(),
        )
        self.artifact_basis = nn.Sequential(
            DepthwiseSeparableConv1d(joint_dim, ppg_dim, kernel_size=7),
            nn.Dropout(0.05),
            nn.Conv1d(ppg_dim, ppg_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(ppg_dim),
        )
        self.physiology_refine = MultiScaleResidualBlock(
            ppg_dim,
            ppg_dim,
            stride=1,
            kernels=(5, 11, 21),
            dilations=(1, 2, 4),
            dropout=0.05,
        )
        self.wavelet_affine = nn.Sequential(
            nn.Linear(wavelet_dim, ppg_dim * 2),
            nn.LayerNorm(ppg_dim * 2),
        )
        self.artifact_scale = nn.Parameter(torch.tensor(float(scale_init)))
        self.physiology_scale = nn.Parameter(torch.tensor(float(scale_init)))

    def forward(
        self,
        ppg_feats: torch.Tensor,
        acc_feats: torch.Tensor,
        wavelet_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if acc_feats.shape[-1] != ppg_feats.shape[-1]:
            acc_feats = F.interpolate(acc_feats, size=ppg_feats.shape[-1], mode="linear", align_corners=False)
        acc_ppg = self.acc_to_ppg(acc_feats)
        joint = torch.cat([ppg_feats, acc_ppg, torch.abs(ppg_feats - acc_ppg)], dim=1)

        motion_mask = self.motion_mask(joint)
        artifact = self.artifact_basis(joint)
        clean = ppg_feats - torch.tanh(self.artifact_scale) * motion_mask * artifact

        candidate = self.physiology_refine(clean)
        gamma, beta = self.wavelet_affine(wavelet_embedding).chunk(2, dim=1)
        gamma = torch.tanh(gamma).unsqueeze(-1)
        beta = torch.tanh(beta).unsqueeze(-1)
        candidate = candidate * (1.0 + 0.25 * gamma) + 0.25 * beta
        enhanced = clean + torch.tanh(self.physiology_scale) * (candidate - clean)
        return enhanced, {
            "motion_artifact_mask": motion_mask,
            "motion_artifact_basis": artifact,
            "motion_clean_ppg_feats": clean,
        }


class AttentionPool1d(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, 1),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        weights = self.score(tokens).squeeze(-1)
        weights = torch.softmax(weights, dim=1).unsqueeze(-1)
        return torch.sum(tokens * weights, dim=1)


def sinusoidal_positional_encoding(length: int, dim: int, device: torch.device) -> torch.Tensor:
    position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / dim))
    pe = torch.zeros(length, dim, device=device, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class TinyTemporalBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 7, stride: int = 2, dropout: float = 0.05) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TinyWaveletDistillNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 5,
        wavelet_dim: int = 4,
        num_classes: int = 2,
        widths: Tuple[int, ...] = (32, 64, 96, 128),
        wavelet_hidden: int = 32,
        fusion_dim: int = 128,
        distill_dim: int = 160,
    ) -> None:
        super().__init__()
        blocks = []
        current_channels = in_channels
        for width in widths:
            blocks.append(TinyTemporalBlock(current_channels, width, kernel_size=7, stride=2, dropout=0.05))
            current_channels = width
        self.temporal_encoder = nn.Sequential(*blocks)
        self.temporal_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(current_channels, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(0.05),
        )
        self.wavelet_mlp = nn.Sequential(
            nn.Linear(wavelet_dim + 1, wavelet_hidden),
            nn.LayerNorm(wavelet_hidden),
            nn.GELU(),
            nn.Linear(wavelet_hidden, wavelet_hidden),
            nn.LayerNorm(wavelet_hidden),
            nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim + wavelet_hidden, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
        )
        self.classifier = nn.Linear(fusion_dim, num_classes)
        self.distill_proj = nn.Linear(fusion_dim, distill_dim)

    def forward(
        self,
        signal: torch.Tensor,
        wavelet_features: torch.Tensor,
        quality: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        temporal_tokens = self.temporal_encoder(signal)
        temporal_embedding = self.temporal_head(temporal_tokens)
        wavelet_input = torch.cat([wavelet_features, quality], dim=1)
        wavelet_embedding = self.wavelet_mlp(wavelet_input)
        embedding = self.fusion(torch.cat([temporal_embedding, wavelet_embedding], dim=1))
        logits = self.classifier(embedding)
        distill_features = self.distill_proj(embedding)
        return {
            "logits": logits,
            "embedding": embedding,
            "distill_features": distill_features,
            "temporal_embedding": temporal_embedding,
            "wavelet_embedding": wavelet_embedding,
        }


class ResNet1DBasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)
        return out


class ResNet1DBottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv1d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)
        return out


class ResNet1D34Backbone(nn.Module):
    def __init__(
        self,
        in_channels: int = 5,
        layers: Tuple[int, int, int, int] = (3, 4, 6, 3),
    ) -> None:
        super().__init__()
        self.inplanes = 64
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(64, layers[0], stride=1)
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        self.out_dim = 512 * ResNet1DBasicBlock.expansion
        self.pool = nn.AdaptiveAvgPool1d(1)

    def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        downsample = None
        outplanes = planes * ResNet1DBasicBlock.expansion
        if stride != 1 or self.inplanes != outplanes:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, outplanes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(outplanes),
            )

        layers = [ResNet1DBasicBlock(self.inplanes, planes, stride=stride, downsample=downsample)]
        self.inplanes = outplanes
        for _ in range(1, blocks):
            layers.append(ResNet1DBasicBlock(self.inplanes, planes, stride=1, downsample=None))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).flatten(1)
        return x


class ResNet1DBackbone(nn.Module):
    def __init__(
        self,
        in_channels: int = 5,
        layers: Tuple[int, int, int, int] = (3, 4, 6, 3),
    ) -> None:
        super().__init__()
        self.inplanes = 64
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(64, layers[0], stride=1)
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        self.out_dim = 512 * ResNet1DBottleneck.expansion
        self.pool = nn.AdaptiveAvgPool1d(1)

    def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        downsample = None
        outplanes = planes * ResNet1DBottleneck.expansion
        if stride != 1 or self.inplanes != outplanes:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, outplanes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(outplanes),
            )

        layers = [ResNet1DBottleneck(self.inplanes, planes, stride=stride, downsample=downsample)]
        self.inplanes = outplanes
        for _ in range(1, blocks):
            layers.append(ResNet1DBottleneck(self.inplanes, planes, stride=1, downsample=None))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).flatten(1)
        return x


class ResNet34WatchNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 5,
        wavelet_dim: int = 4,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.backbone = ResNet1D34Backbone(in_channels=in_channels, layers=(3, 4, 6, 3))
        self.temporal_head = nn.Sequential(
            nn.Linear(self.backbone.out_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.15),
        )
        self.wavelet_mlp = nn.Sequential(
            nn.Linear(wavelet_dim + 1, 64),
            nn.GELU(),
            nn.Linear(64, 96),
            nn.LayerNorm(96),
            nn.GELU(),
        )
        self.quality_mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.GELU(),
            nn.Linear(16, 16),
        )
        self.fusion = nn.Sequential(
            nn.Linear(256 + 96 + 16, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(256, 160),
            nn.LayerNorm(160),
            nn.GELU(),
        )
        self.classifier = nn.Linear(160, num_classes)
        self.wavelet_predictor = nn.Sequential(
            nn.Linear(256, 96),
            nn.GELU(),
            nn.Linear(96, wavelet_dim),
        )

    def forward(
        self,
        signal: torch.Tensor,
        wavelet_features: torch.Tensor,
        quality: torch.Tensor,
        baseline_signal: torch.Tensor | None = None,
        baseline_wavelet_features: torch.Tensor | None = None,
        baseline_quality: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        del baseline_signal, baseline_wavelet_features, baseline_quality
        temporal_raw = self.backbone(signal)
        temporal_embedding = self.temporal_head(temporal_raw)
        wavelet_input = torch.cat([wavelet_features, quality], dim=1)
        wavelet_embedding = self.wavelet_mlp(wavelet_input)
        quality_embedding = self.quality_mlp(quality)
        embedding = self.fusion(torch.cat([temporal_embedding, wavelet_embedding, quality_embedding], dim=1))
        logits = self.classifier(embedding)
        wavelet_pred = self.wavelet_predictor(temporal_embedding)
        return {
            "logits": logits,
            "embedding": embedding,
            "wavelet_pred": wavelet_pred,
            "temporal_embedding": temporal_embedding,
            "wavelet_embedding": wavelet_embedding,
        }


class ResNet18WatchNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 5,
        wavelet_dim: int = 4,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.backbone = ResNet1D34Backbone(in_channels=in_channels, layers=(2, 2, 2, 2))
        self.temporal_head = nn.Sequential(
            nn.Linear(self.backbone.out_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.15),
        )
        self.wavelet_mlp = nn.Sequential(
            nn.Linear(wavelet_dim + 1, 64),
            nn.GELU(),
            nn.Linear(64, 96),
            nn.LayerNorm(96),
            nn.GELU(),
        )
        self.quality_mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.GELU(),
            nn.Linear(16, 16),
        )
        self.fusion = nn.Sequential(
            nn.Linear(256 + 96 + 16, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(256, 160),
            nn.LayerNorm(160),
            nn.GELU(),
        )
        self.classifier = nn.Linear(160, num_classes)
        self.wavelet_predictor = nn.Sequential(
            nn.Linear(256, 96),
            nn.GELU(),
            nn.Linear(96, wavelet_dim),
        )

    def forward(
        self,
        signal: torch.Tensor,
        wavelet_features: torch.Tensor,
        quality: torch.Tensor,
        baseline_signal: torch.Tensor | None = None,
        baseline_wavelet_features: torch.Tensor | None = None,
        baseline_quality: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        del baseline_signal, baseline_wavelet_features, baseline_quality
        temporal_raw = self.backbone(signal)
        temporal_embedding = self.temporal_head(temporal_raw)
        wavelet_input = torch.cat([wavelet_features, quality], dim=1)
        wavelet_embedding = self.wavelet_mlp(wavelet_input)
        quality_embedding = self.quality_mlp(quality)
        embedding = self.fusion(torch.cat([temporal_embedding, wavelet_embedding, quality_embedding], dim=1))
        logits = self.classifier(embedding)
        wavelet_pred = self.wavelet_predictor(temporal_embedding)
        return {
            "logits": logits,
            "embedding": embedding,
            "wavelet_pred": wavelet_pred,
            "temporal_embedding": temporal_embedding,
            "wavelet_embedding": wavelet_embedding,
        }


class ResNet50WatchNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 5,
        wavelet_dim: int = 4,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.backbone = ResNet1DBackbone(in_channels=in_channels, layers=(3, 4, 6, 3))
        self.temporal_head = nn.Sequential(
            nn.Linear(self.backbone.out_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.15),
        )
        self.wavelet_mlp = nn.Sequential(
            nn.Linear(wavelet_dim + 1, 64),
            nn.GELU(),
            nn.Linear(64, 96),
            nn.LayerNorm(96),
            nn.GELU(),
        )
        self.quality_mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.GELU(),
            nn.Linear(16, 16),
        )
        self.fusion = nn.Sequential(
            nn.Linear(256 + 96 + 16, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(256, 160),
            nn.LayerNorm(160),
            nn.GELU(),
        )
        self.classifier = nn.Linear(160, num_classes)
        self.wavelet_predictor = nn.Sequential(
            nn.Linear(256, 96),
            nn.GELU(),
            nn.Linear(96, wavelet_dim),
        )

    def forward(
        self,
        signal: torch.Tensor,
        wavelet_features: torch.Tensor,
        quality: torch.Tensor,
        baseline_signal: torch.Tensor | None = None,
        baseline_wavelet_features: torch.Tensor | None = None,
        baseline_quality: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        del baseline_signal, baseline_wavelet_features, baseline_quality
        temporal_raw = self.backbone(signal)
        temporal_embedding = self.temporal_head(temporal_raw)
        wavelet_input = torch.cat([wavelet_features, quality], dim=1)
        wavelet_embedding = self.wavelet_mlp(wavelet_input)
        quality_embedding = self.quality_mlp(quality)
        embedding = self.fusion(torch.cat([temporal_embedding, wavelet_embedding, quality_embedding], dim=1))
        logits = self.classifier(embedding)
        wavelet_pred = self.wavelet_predictor(temporal_embedding)
        return {
            "logits": logits,
            "embedding": embedding,
            "wavelet_pred": wavelet_pred,
            "temporal_embedding": temporal_embedding,
            "wavelet_embedding": wavelet_embedding,
        }


class _ResNetWatchEncoderBase(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        wavelet_dim: int = 4,
        embed_dim: int = 160,
        temporal_hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.temporal_head = nn.Sequential(
            nn.Linear(self.backbone.out_dim, temporal_hidden_dim),
            nn.LayerNorm(temporal_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.15),
        )
        self.wavelet_mlp = nn.Sequential(
            nn.Linear(wavelet_dim + 1, 64),
            nn.GELU(),
            nn.Linear(64, 96),
            nn.LayerNorm(96),
            nn.GELU(),
        )
        self.quality_mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.GELU(),
            nn.Linear(16, 16),
        )
        self.fusion = nn.Sequential(
            nn.Linear(temporal_hidden_dim + 96 + 16, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(256, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

    def forward(
        self,
        signal: torch.Tensor,
        wavelet_features: torch.Tensor,
        quality: torch.Tensor,
        baseline_signal: torch.Tensor | None = None,
        baseline_wavelet_features: torch.Tensor | None = None,
        baseline_quality: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        del baseline_signal, baseline_wavelet_features, baseline_quality
        temporal_raw = self.backbone(signal)
        temporal_embedding = self.temporal_head(temporal_raw)
        wavelet_input = torch.cat([wavelet_features, quality], dim=1)
        wavelet_embedding = self.wavelet_mlp(wavelet_input)
        quality_embedding = self.quality_mlp(quality)
        embedding = self.fusion(torch.cat([temporal_embedding, wavelet_embedding, quality_embedding], dim=1))
        return {
            "embedding": embedding,
            "temporal_embedding": temporal_embedding,
            "wavelet_embedding": wavelet_embedding,
        }


class ResNet18WatchEncoder(_ResNetWatchEncoderBase):
    def __init__(self, in_channels: int = 5, wavelet_dim: int = 4, embed_dim: int = 160) -> None:
        super().__init__(
            backbone=ResNet1D34Backbone(in_channels=in_channels, layers=(2, 2, 2, 2)),
            wavelet_dim=wavelet_dim,
            embed_dim=embed_dim,
        )


class ResNet34WatchEncoder(_ResNetWatchEncoderBase):
    def __init__(self, in_channels: int = 5, wavelet_dim: int = 4, embed_dim: int = 160) -> None:
        super().__init__(
            backbone=ResNet1D34Backbone(in_channels=in_channels, layers=(3, 4, 6, 3)),
            wavelet_dim=wavelet_dim,
            embed_dim=embed_dim,
        )


class ResNet50WatchEncoder(_ResNetWatchEncoderBase):
    def __init__(self, in_channels: int = 5, wavelet_dim: int = 4, embed_dim: int = 160) -> None:
        super().__init__(
            backbone=ResNet1DBackbone(in_channels=in_channels, layers=(3, 4, 6, 3)),
            wavelet_dim=wavelet_dim,
            embed_dim=embed_dim,
        )


class WaveletGuidedWatchNet(nn.Module):
    def __init__(
        self,
        ppg_channels: int = 1,
        acc_channels: int = 4,
        wavelet_dim: int = 4,
        num_classes: int = 2,
        model_dim: int = 192,
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        fusion_hidden_dim: int = 256,
        embed_dim: int = 160,
        watch_enhancement: str = "none",
        watch_motion_mode: str = "strong",
    ) -> None:
        super().__init__()
        if watch_enhancement not in {"none", "motion_disentangled", "acc_concat"}:
            raise ValueError(f"Unsupported watch enhancement: {watch_enhancement}")
        self.watch_enhancement = watch_enhancement
        self.watch_motion_mode = watch_motion_mode
        self.ppg_stem = nn.Sequential(
            nn.Conv1d(ppg_channels, 64, kernel_size=11, stride=2, padding=5, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
        )
        self.acc_stem = nn.Sequential(
            nn.Conv1d(acc_channels, 48, kernel_size=9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(48),
            nn.GELU(),
        )

        self.ppg_blocks = nn.ModuleList(
            [
                MultiScaleResidualBlock(64, 96, stride=1),
                MultiScaleResidualBlock(96, 144, stride=2),
                MultiScaleResidualBlock(144, model_dim, stride=2),
            ]
        )
        self.acc_blocks = nn.ModuleList(
            [
                MultiScaleResidualBlock(48, 64, stride=1, kernels=(5, 11, 19), dilations=(1, 1, 2)),
                MultiScaleResidualBlock(64, 96, stride=2, kernels=(5, 11, 19), dilations=(1, 1, 2)),
                MultiScaleResidualBlock(96, 128, stride=2, kernels=(5, 11, 19), dilations=(1, 1, 2)),
            ]
        )
        self.motion_gates = nn.ModuleList()
        self.motion_proj = nn.ModuleList()
        if watch_enhancement != "acc_concat":
            self.motion_gates = nn.ModuleList(
                [
                    MotionFiLM(64, 96, motion_mode=watch_motion_mode),
                    MotionFiLM(96, 144, motion_mode=watch_motion_mode),
                    MotionFiLM(128, model_dim, motion_mode=watch_motion_mode),
                ]
            )
            self.motion_proj = nn.ModuleList(
                [
                    nn.Linear(64, 64),
                    nn.Linear(96, 96),
                    nn.Linear(128, 128),
                ]
            )
        self.acc_concat_fusion = (
            nn.Sequential(
                nn.Conv1d(model_dim + 128, model_dim, kernel_size=1, bias=False),
                nn.BatchNorm1d(model_dim),
                nn.GELU(),
            )
            if watch_enhancement == "acc_concat"
            else None
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=transformer_heads,
            dim_feedforward=model_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, model_dim))

        self.attn_pool = AttentionPool1d(model_dim)
        self.temporal_head = nn.Sequential(
            nn.Linear(model_dim * 2, model_dim),
            nn.LayerNorm(model_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        self.wavelet_mlp = nn.Sequential(
            nn.Linear(wavelet_dim + 1, 64),
            nn.GELU(),
            nn.Linear(64, 96),
            nn.LayerNorm(96),
            nn.GELU(),
        )
        self.wavelet_gate = nn.Sequential(
            nn.Linear(96, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )
        self.ppg_enhancer = (
            MotionDisentangledPPGEnhancer(model_dim, 128, wavelet_dim=96)
            if watch_enhancement == "motion_disentangled"
            else None
        )
        self.quality_mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.GELU(),
            nn.Linear(16, 16),
        )

        fused_dim = model_dim + 96 + 16
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, fusion_hidden_dim),
            nn.LayerNorm(fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(fusion_hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.baseline_ref_adapter = nn.Sequential(
            nn.Linear(embed_dim * 3, fusion_hidden_dim),
            nn.LayerNorm(fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )
        self.baseline_ref_gate = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid(),
        )
        self.wavelet_predictor = nn.Sequential(
            nn.Linear(model_dim, 96),
            nn.GELU(),
            nn.Linear(96, wavelet_dim),
        )

    def _encode_core(
        self,
        signal: torch.Tensor,
        wavelet_features: torch.Tensor,
        quality: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        ppg = signal[:, :1]
        acc = signal[:, 1:]

        ppg_feats = self.ppg_stem(ppg)
        acc_feats = self.acc_stem(acc)

        if self.watch_enhancement == "acc_concat":
            for ppg_block, acc_block in zip(self.ppg_blocks, self.acc_blocks):
                ppg_feats = ppg_block(ppg_feats)
                acc_feats = acc_block(acc_feats)
            if acc_feats.shape[-1] != ppg_feats.shape[-1]:
                acc_feats = F.interpolate(acc_feats, size=ppg_feats.shape[-1], mode="linear", align_corners=False)
            ppg_feats = self.acc_concat_fusion(torch.cat([ppg_feats, acc_feats], dim=1))
        else:
            for ppg_block, acc_block, gate, proj in zip(self.ppg_blocks, self.acc_blocks, self.motion_gates, self.motion_proj):
                ppg_feats = ppg_block(ppg_feats)
                acc_feats = acc_block(acc_feats)
                acc_context = F.adaptive_avg_pool1d(acc_feats, 1).squeeze(-1)
                ppg_feats = gate(ppg_feats, proj(acc_context))

        wavelet_input = torch.cat([wavelet_features, quality], dim=1)
        wavelet_embedding = self.wavelet_mlp(wavelet_input)
        enhancer_aux: Dict[str, torch.Tensor] = {}
        if self.ppg_enhancer is not None:
            ppg_feats, enhancer_aux = self.ppg_enhancer(ppg_feats, acc_feats, wavelet_embedding)

        tokens = ppg_feats.transpose(1, 2)
        cls = self.cls_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        pos = sinusoidal_positional_encoding(tokens.shape[1], tokens.shape[2], tokens.device).unsqueeze(0)
        tokens = self.transformer(tokens + pos)

        cls_token = tokens[:, 0]
        pooled_tokens = self.attn_pool(tokens[:, 1:])
        temporal_embedding = self.temporal_head(torch.cat([cls_token, pooled_tokens], dim=1))

        temporal_embedding = temporal_embedding * (1.0 + torch.tanh(self.wavelet_gate(wavelet_embedding)))
        quality_embedding = self.quality_mlp(quality)

        fused = self.fusion(torch.cat([temporal_embedding, wavelet_embedding, quality_embedding], dim=1))
        return {
            "embedding": fused,
            "temporal_embedding": temporal_embedding,
            "wavelet_embedding": wavelet_embedding,
            **enhancer_aux,
        }

    def forward(
        self,
        signal: torch.Tensor,
        wavelet_features: torch.Tensor,
        quality: torch.Tensor,
        baseline_signal: torch.Tensor | None = None,
        baseline_wavelet_features: torch.Tensor | None = None,
        baseline_quality: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        out = self._encode_core(signal, wavelet_features, quality)
        fused = out["embedding"]
        if baseline_signal is not None and baseline_wavelet_features is not None and baseline_quality is not None:
            baseline_out = self._encode_core(baseline_signal, baseline_wavelet_features, baseline_quality)
            baseline_embedding = baseline_out["embedding"]
            delta = fused - baseline_embedding
            ref_features = torch.cat([fused, delta, torch.abs(delta)], dim=1)
            ref_adapter = self.baseline_ref_adapter(ref_features)
            ref_gate = self.baseline_ref_gate(ref_features)
            fused = fused + ref_gate * ref_adapter
            out["baseline_embedding"] = baseline_embedding
            out["baseline_delta"] = delta
        logits = self.classifier(fused)
        wavelet_pred = self.wavelet_predictor(out["temporal_embedding"])
        out["embedding"] = fused
        out["logits"] = logits
        out["wavelet_pred"] = wavelet_pred
        return out


class WaveletGuidedWatchEncoder(nn.Module):
    def __init__(
        self,
        ppg_channels: int = 1,
        acc_channels: int = 4,
        wavelet_dim: int = 4,
        model_dim: int = 192,
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        fusion_hidden_dim: int = 256,
        embed_dim: int = 160,
        watch_enhancement: str = "none",
        watch_motion_mode: str = "strong",
    ) -> None:
        super().__init__()
        if watch_enhancement not in {"none", "motion_disentangled", "acc_concat"}:
            raise ValueError(f"Unsupported watch enhancement: {watch_enhancement}")
        self.watch_enhancement = watch_enhancement
        self.watch_motion_mode = watch_motion_mode
        self.ppg_stem = nn.Sequential(
            nn.Conv1d(ppg_channels, 64, kernel_size=11, stride=2, padding=5, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
        )
        self.acc_stem = nn.Sequential(
            nn.Conv1d(acc_channels, 48, kernel_size=9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(48),
            nn.GELU(),
        )
        self.ppg_blocks = nn.ModuleList(
            [
                MultiScaleResidualBlock(64, 96, stride=1),
                MultiScaleResidualBlock(96, 144, stride=2),
                MultiScaleResidualBlock(144, model_dim, stride=2),
            ]
        )
        self.acc_blocks = nn.ModuleList(
            [
                MultiScaleResidualBlock(48, 64, stride=1, kernels=(5, 11, 19), dilations=(1, 1, 2)),
                MultiScaleResidualBlock(64, 96, stride=2, kernels=(5, 11, 19), dilations=(1, 1, 2)),
                MultiScaleResidualBlock(96, 128, stride=2, kernels=(5, 11, 19), dilations=(1, 1, 2)),
            ]
        )
        self.motion_gates = nn.ModuleList()
        self.motion_proj = nn.ModuleList()
        if watch_enhancement != "acc_concat":
            self.motion_gates = nn.ModuleList(
                [
                    MotionFiLM(64, 96, motion_mode=watch_motion_mode),
                    MotionFiLM(96, 144, motion_mode=watch_motion_mode),
                    MotionFiLM(128, model_dim, motion_mode=watch_motion_mode),
                ]
            )
            self.motion_proj = nn.ModuleList(
                [
                    nn.Linear(64, 64),
                    nn.Linear(96, 96),
                    nn.Linear(128, 128),
                ]
            )
        self.acc_concat_fusion = (
            nn.Sequential(
                nn.Conv1d(model_dim + 128, model_dim, kernel_size=1, bias=False),
                nn.BatchNorm1d(model_dim),
                nn.GELU(),
            )
            if watch_enhancement == "acc_concat"
            else None
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=transformer_heads,
            dim_feedforward=model_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, model_dim))
        self.attn_pool = AttentionPool1d(model_dim)
        self.temporal_head = nn.Sequential(
            nn.Linear(model_dim * 2, model_dim),
            nn.LayerNorm(model_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.wavelet_mlp = nn.Sequential(
            nn.Linear(wavelet_dim + 1, 64),
            nn.GELU(),
            nn.Linear(64, 96),
            nn.LayerNorm(96),
            nn.GELU(),
        )
        self.wavelet_gate = nn.Sequential(
            nn.Linear(96, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )
        self.ppg_enhancer = (
            MotionDisentangledPPGEnhancer(model_dim, 128, wavelet_dim=96)
            if watch_enhancement == "motion_disentangled"
            else None
        )
        self.quality_mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.GELU(),
            nn.Linear(16, 16),
        )
        self.fusion = nn.Sequential(
            nn.Linear(model_dim + 96 + 16, fusion_hidden_dim),
            nn.LayerNorm(fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(fusion_hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )
        self.baseline_ref_adapter = nn.Sequential(
            nn.Linear(embed_dim * 3, fusion_hidden_dim),
            nn.LayerNorm(fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )
        self.baseline_ref_gate = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid(),
        )

    def _encode_core(self, signal: torch.Tensor, wavelet_features: torch.Tensor, quality: torch.Tensor) -> Dict[str, torch.Tensor]:
        ppg = signal[:, :1]
        acc = signal[:, 1:]
        ppg_feats = self.ppg_stem(ppg)
        acc_feats = self.acc_stem(acc)

        if self.watch_enhancement == "acc_concat":
            for ppg_block, acc_block in zip(self.ppg_blocks, self.acc_blocks):
                ppg_feats = ppg_block(ppg_feats)
                acc_feats = acc_block(acc_feats)
            if acc_feats.shape[-1] != ppg_feats.shape[-1]:
                acc_feats = F.interpolate(acc_feats, size=ppg_feats.shape[-1], mode="linear", align_corners=False)
            ppg_feats = self.acc_concat_fusion(torch.cat([ppg_feats, acc_feats], dim=1))
        else:
            for ppg_block, acc_block, gate, proj in zip(self.ppg_blocks, self.acc_blocks, self.motion_gates, self.motion_proj):
                ppg_feats = ppg_block(ppg_feats)
                acc_feats = acc_block(acc_feats)
                acc_context = F.adaptive_avg_pool1d(acc_feats, 1).squeeze(-1)
                ppg_feats = gate(ppg_feats, proj(acc_context))

        wavelet_input = torch.cat([wavelet_features, quality], dim=1)
        wavelet_embedding = self.wavelet_mlp(wavelet_input)
        enhancer_aux: Dict[str, torch.Tensor] = {}
        if self.ppg_enhancer is not None:
            ppg_feats, enhancer_aux = self.ppg_enhancer(ppg_feats, acc_feats, wavelet_embedding)

        tokens = ppg_feats.transpose(1, 2)
        cls = self.cls_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        pos = sinusoidal_positional_encoding(tokens.shape[1], tokens.shape[2], tokens.device).unsqueeze(0)
        tokens = self.transformer(tokens + pos)

        cls_token = tokens[:, 0]
        pooled_tokens = self.attn_pool(tokens[:, 1:])
        temporal_embedding = self.temporal_head(torch.cat([cls_token, pooled_tokens], dim=1))

        temporal_embedding = temporal_embedding * (1.0 + torch.tanh(self.wavelet_gate(wavelet_embedding)))
        quality_embedding = self.quality_mlp(quality)
        embedding = self.fusion(torch.cat([temporal_embedding, wavelet_embedding, quality_embedding], dim=1))
        return {
            "embedding": embedding,
            "temporal_embedding": temporal_embedding,
            "wavelet_embedding": wavelet_embedding,
            **enhancer_aux,
        }

    def forward(
        self,
        signal: torch.Tensor,
        wavelet_features: torch.Tensor,
        quality: torch.Tensor,
        baseline_signal: torch.Tensor | None = None,
        baseline_wavelet_features: torch.Tensor | None = None,
        baseline_quality: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        out = self._encode_core(signal, wavelet_features, quality)
        embedding = out["embedding"]
        if baseline_signal is not None and baseline_wavelet_features is not None and baseline_quality is not None:
            baseline_out = self._encode_core(baseline_signal, baseline_wavelet_features, baseline_quality)
            baseline_embedding = baseline_out["embedding"]
            delta = embedding - baseline_embedding
            ref_features = torch.cat([embedding, delta, torch.abs(delta)], dim=1)
            ref_adapter = self.baseline_ref_adapter(ref_features)
            ref_gate = self.baseline_ref_gate(ref_features)
            embedding = embedding + ref_gate * ref_adapter
            out["baseline_embedding"] = baseline_embedding
            out["baseline_delta"] = delta
        out["embedding"] = embedding
        return out


class E4ReferenceEncoder(nn.Module):
    def __init__(self, bvp_channels: int = 1, acc_channels: int = 4, out_dim: int = 160) -> None:
        super().__init__()
        self.bvp_stem = nn.Sequential(
            nn.Conv1d(bvp_channels, 48, kernel_size=11, stride=2, padding=5, bias=False),
            nn.BatchNorm1d(48),
            nn.GELU(),
        )
        self.acc_stem = nn.Sequential(
            nn.Conv1d(acc_channels, 32, kernel_size=9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(32),
            nn.GELU(),
        )
        self.bvp_blocks = nn.ModuleList(
            [
                MultiScaleResidualBlock(48, 72, stride=1, kernels=(7, 15, 31), dilations=(1, 1, 2)),
                MultiScaleResidualBlock(72, 112, stride=2, kernels=(7, 15, 31), dilations=(1, 1, 2)),
                MultiScaleResidualBlock(112, 160, stride=2, kernels=(7, 15, 31), dilations=(1, 1, 2)),
            ]
        )
        self.acc_blocks = nn.ModuleList(
            [
                MultiScaleResidualBlock(32, 48, stride=1, kernels=(5, 11, 19), dilations=(1, 1, 2)),
                MultiScaleResidualBlock(48, 80, stride=2, kernels=(5, 11, 19), dilations=(1, 1, 2)),
                MultiScaleResidualBlock(80, 96, stride=2, kernels=(5, 11, 19), dilations=(1, 1, 2)),
            ]
        )
        self.motion_film = nn.ModuleList(
            [
                MotionFiLM(48, 72),
                MotionFiLM(80, 112),
                MotionFiLM(96, 160),
            ]
        )
        self.motion_proj = nn.ModuleList(
            [
                nn.Linear(48, 48),
                nn.Linear(80, 80),
                nn.Linear(96, 96),
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
        bvp = signal[:, :1]
        acc = signal[:, 1:]
        bvp_feats = self.bvp_stem(bvp)
        acc_feats = self.acc_stem(acc)
        for bvp_block, acc_block, gate, proj in zip(self.bvp_blocks, self.acc_blocks, self.motion_film, self.motion_proj):
            bvp_feats = bvp_block(bvp_feats)
            acc_feats = acc_block(acc_feats)
            acc_context = F.adaptive_avg_pool1d(acc_feats, 1).squeeze(-1)
            bvp_feats = gate(bvp_feats, proj(acc_context))
        return self.pool(bvp_feats)


class PrivilegedGalaxyTeacherNet(nn.Module):
    """Training-time Galaxy teacher with a deployable watch-only path.

    The inference path is only ``watch_encoder -> watch_classifier``. The E4
    encoder, teacher fusion, rhythm heads, and wavelet head provide training
    supervision and are removed from deployment.
    """

    def __init__(
        self,
        wavelet_dim: int = 4,
        num_classes: int = 2,
        embed_dim: int = 160,
        rhythm_dim: int = 3,
        watch_backbone: str = "wavelet_guided",
        use_e4_classifier: bool = True,
        use_rhythm_heads: bool = True,
        use_wavelet_head: bool = True,
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
        self.use_e4_classifier = use_e4_classifier
        self.use_rhythm_heads = use_rhythm_heads
        self.use_wavelet_head = use_wavelet_head

        if watch_kwargs is not None:
            kwargs = dict(watch_kwargs)
            kwargs.setdefault("wavelet_dim", wavelet_dim)
            kwargs.setdefault("embed_dim", embed_dim)
            kwargs.setdefault("watch_enhancement", watch_enhancement)
            kwargs.setdefault("watch_motion_mode", watch_motion_mode)
            self.watch_encoder = WaveletGuidedWatchEncoder(**kwargs)
        elif watch_backbone == "wavelet_guided":
            self.watch_encoder = WaveletGuidedWatchEncoder(
                wavelet_dim=wavelet_dim,
                embed_dim=embed_dim,
                watch_enhancement=watch_enhancement,
                watch_motion_mode=watch_motion_mode,
            )
        elif watch_enhancement != "none":
            raise ValueError("watch_enhancement is only supported for the wavelet_guided backbone.")
        elif watch_backbone == "resnet18_1d":
            self.watch_encoder = ResNet18WatchEncoder(wavelet_dim=wavelet_dim, embed_dim=embed_dim)
        elif watch_backbone == "resnet34_1d":
            self.watch_encoder = ResNet34WatchEncoder(wavelet_dim=wavelet_dim, embed_dim=embed_dim)
        elif watch_backbone == "resnet50_1d":
            self.watch_encoder = ResNet50WatchEncoder(wavelet_dim=wavelet_dim, embed_dim=embed_dim)
        else:
            raise ValueError(f"Unsupported watch backbone: {watch_backbone}")

        self.e4_encoder = E4ReferenceEncoder(out_dim=embed_dim)
        self.watch_classifier = nn.Linear(embed_dim, num_classes)
        self.e4_classifier = nn.Linear(embed_dim, num_classes) if use_e4_classifier else None

        self.teacher_watch_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )
        self.teacher_e4_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )
        self.teacher_e4_tokenizer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
        )
        self.teacher_cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True,
        )
        self.teacher_input_dropout = nn.Dropout(0.15)
        self.teacher_attn_dropout = nn.Dropout(0.15)
        self.teacher_gate = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid(),
        )
        self.teacher_watch_path = nn.Sequential(
            nn.Linear(embed_dim * 3, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(256, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )
        self.teacher_fusion = nn.Sequential(
            nn.Linear(embed_dim * 5, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )
        self.teacher_decision_gate = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid(),
        )
        self.teacher_decision = nn.Sequential(
            nn.Linear(embed_dim * 3, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )
        self.teacher_feature_dropout = nn.Dropout(0.15)
        self.teacher_out_norm = nn.LayerNorm(embed_dim)
        self.teacher_classifier = nn.Linear(embed_dim, num_classes)

        self.rhythm_head = (
            nn.Sequential(
                nn.Linear(embed_dim, 128),
                nn.GELU(),
                nn.Linear(128, 64),
                nn.GELU(),
                nn.Linear(64, rhythm_dim),
            )
            if use_rhythm_heads
            else None
        )
        self.teacher_rhythm_head = (
            nn.Sequential(
                nn.Linear(embed_dim, 128),
                nn.GELU(),
                nn.Linear(128, 64),
                nn.GELU(),
                nn.Linear(64, rhythm_dim),
            )
            if use_rhythm_heads
            else None
        )
        self.wavelet_predictor = (
            nn.Sequential(
                nn.Linear(embed_dim, 96),
                nn.GELU(),
                nn.Linear(96, wavelet_dim),
            )
            if use_wavelet_head
            else None
        )

    def forward_watch(
        self,
        watch_signal: torch.Tensor,
        wavelet_features: torch.Tensor,
        quality: torch.Tensor,
        baseline_signal: torch.Tensor | None = None,
        baseline_wavelet_features: torch.Tensor | None = None,
        baseline_quality: torch.Tensor | None = None,
        return_aux: bool = True,
    ) -> Dict[str, torch.Tensor]:
        watch_out = self.watch_encoder(
            watch_signal,
            wavelet_features,
            quality,
            baseline_signal=baseline_signal,
            baseline_wavelet_features=baseline_wavelet_features,
            baseline_quality=baseline_quality,
        )
        embedding = watch_out["embedding"]
        logits = self.watch_classifier(embedding)
        out: Dict[str, torch.Tensor] = {
            **watch_out,
            "watch_embedding": embedding,
            "base_logits": logits,
            "logits": logits,
        }
        if return_aux and self.rhythm_head is not None:
            out["rhythm_pred"] = self.rhythm_head(embedding)
        if return_aux and self.wavelet_predictor is not None:
            out["wavelet_pred"] = self.wavelet_predictor(embedding)
        return out

    def forward(
        self,
        watch_signal: torch.Tensor,
        wavelet_features: torch.Tensor,
        quality: torch.Tensor,
        e4_signal: torch.Tensor | None = None,
        baseline_watch_signal: torch.Tensor | None = None,
        baseline_wavelet_features: torch.Tensor | None = None,
        baseline_quality: torch.Tensor | None = None,
        return_aux: bool = True,
    ) -> Dict[str, torch.Tensor]:
        out = self.forward_watch(
            watch_signal,
            wavelet_features,
            quality,
            baseline_signal=baseline_watch_signal,
            baseline_wavelet_features=baseline_wavelet_features,
            baseline_quality=baseline_quality,
            return_aux=return_aux,
        )
        if e4_signal is None:
            return out

        e4_embedding = self.e4_encoder(e4_signal)
        out["e4_embedding"] = e4_embedding
        if return_aux and self.e4_classifier is not None:
            out["e4_logits"] = self.e4_classifier(e4_embedding)

        watch_teacher = self.teacher_input_dropout(self.teacher_watch_proj(out["watch_embedding"]))
        e4_teacher = self.teacher_input_dropout(self.teacher_e4_proj(e4_embedding))
        e4_tokens = self.teacher_e4_tokenizer(e4_teacher).view(e4_teacher.shape[0], 4, -1)
        attended_e4, _ = self.teacher_cross_attn(
            query=watch_teacher.unsqueeze(1),
            key=e4_tokens,
            value=e4_tokens,
            need_weights=False,
        )
        attended_e4 = self.teacher_attn_dropout(attended_e4.squeeze(1))
        teacher_gate = self.teacher_gate(torch.cat([watch_teacher, e4_teacher, attended_e4], dim=1))
        teacher_features = torch.cat(
            [
                watch_teacher,
                e4_teacher,
                attended_e4,
                watch_teacher * attended_e4,
                torch.abs(watch_teacher - e4_teacher),
            ],
            dim=1,
        )
        teacher_features = self.teacher_feature_dropout(teacher_features)
        watch_specific_features = torch.cat(
            [watch_teacher, teacher_gate * attended_e4, torch.abs(watch_teacher - attended_e4)],
            dim=1,
        )
        teacher_watch_path = self.teacher_watch_path(watch_specific_features)
        teacher_fused_path = self.teacher_fusion(teacher_features)
        decision_gate = self.teacher_decision_gate(
            torch.cat([watch_teacher, teacher_watch_path, teacher_fused_path], dim=1)
        )
        blended_path = decision_gate * teacher_fused_path + (1.0 - decision_gate) * teacher_watch_path
        teacher_embedding = (
            watch_teacher
            + teacher_gate * attended_e4
            + self.teacher_decision(torch.cat([teacher_watch_path, teacher_fused_path, blended_path], dim=1))
        )
        teacher_embedding = self.teacher_out_norm(teacher_embedding)
        out["teacher_embedding"] = teacher_embedding
        out["teacher_watch_path"] = teacher_watch_path
        out["teacher_fused_path"] = teacher_fused_path
        out["teacher_logits"] = self.teacher_classifier(teacher_embedding)
        if return_aux and self.teacher_rhythm_head is not None:
            out["teacher_rhythm_pred"] = self.teacher_rhythm_head(teacher_embedding)
        return out
