from __future__ import annotations

import argparse
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Dict

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


def _pop_hidden_arg(name: str, default: str) -> str:
    if name not in sys.argv:
        return default
    idx = sys.argv.index(name)
    if idx + 1 >= len(sys.argv):
        raise ValueError(f"Missing value for hidden argument {name}")
    value = sys.argv[idx + 1]
    del sys.argv[idx : idx + 2]
    return value


def _remove_flag(name: str) -> bool:
    present = name in sys.argv
    while name in sys.argv:
        sys.argv.remove(name)
    return present


class WaveletPhysiologyRefiner(nn.Module):
    """Wavelet-guided refinement without any direct ACC input.

    This isolates the physiology/refinement stage from the ACC-dependent
    Adapt/Clean path used by the full motion-aware encoder.
    """

    def __init__(self, ppg_dim: int, wavelet_dim: int = 96, scale_init: float = 0.05) -> None:
        super().__init__()
        from .galaxy_models import MultiScaleResidualBlock

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
        self.physiology_scale = nn.Parameter(torch.tensor(float(scale_init)))

    def forward(self, ppg_feats: torch.Tensor, wavelet_embedding: torch.Tensor) -> torch.Tensor:
        candidate = self.physiology_refine(ppg_feats)
        gamma, beta = self.wavelet_affine(wavelet_embedding).chunk(2, dim=1)
        gamma = torch.tanh(gamma).unsqueeze(-1)
        beta = torch.tanh(beta).unsqueeze(-1)
        candidate = candidate * (1.0 + 0.25 * gamma) + 0.25 * beta
        return ppg_feats + torch.tanh(self.physiology_scale) * (candidate - ppg_feats)


class RefineOnlyPPGEnhancer(nn.Module):
    """Drop-in enhancer signature that performs only wavelet-guided refinement."""

    def __init__(self, ppg_dim: int, acc_dim: int, wavelet_dim: int = 96, scale_init: float = 0.05) -> None:
        super().__init__()
        del acc_dim
        self.refiner = WaveletPhysiologyRefiner(ppg_dim, wavelet_dim=wavelet_dim, scale_init=scale_init)

    def forward(
        self,
        ppg_feats: torch.Tensor,
        acc_feats: torch.Tensor,
        wavelet_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        del acc_feats
        refined = self.refiner(ppg_feats, wavelet_embedding)
        return refined, {"refine_only_ppg_feats": refined}


def install_ppg_only_watch_models(use_refine: bool = False) -> None:
    from . import galaxy_models
    from .galaxy_models import (
        AttentionPool1d,
        MultiScaleResidualBlock,
        sinusoidal_positional_encoding,
    )

    OriginalResNet18WatchNet = galaxy_models.ResNet18WatchNet
    OriginalResNet34WatchNet = galaxy_models.ResNet34WatchNet
    OriginalResNet50WatchNet = galaxy_models.ResNet50WatchNet

    class PPGOnlyWaveletGuidedWatchEncoder(nn.Module):
        """Wavelet-guided watch encoder that removes ACC from the deployable path.

        This is intentionally kept outside the production model so it can be used
        as a clean input ablation: PPG/BVP + wavelet + quality, but no ACC branch,
        no motion FiLM, and no artifact-aware PPG enhancer.
        """

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
            del acc_channels, watch_enhancement, watch_motion_mode
            self.watch_enhancement = "ppg_only"
            self.watch_motion_mode = "none"
            self.ppg_stem = nn.Sequential(
                nn.Conv1d(ppg_channels, 64, kernel_size=11, stride=2, padding=5, bias=False),
                nn.BatchNorm1d(64),
                nn.GELU(),
            )
            self.ppg_blocks = nn.ModuleList(
                [
                    MultiScaleResidualBlock(64, 96, stride=1),
                    MultiScaleResidualBlock(96, 144, stride=2),
                    MultiScaleResidualBlock(144, model_dim, stride=2),
                ]
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
            self.quality_mlp = nn.Sequential(
                nn.Linear(1, 16),
                nn.GELU(),
                nn.Linear(16, 16),
            )
            self.refine_only = WaveletPhysiologyRefiner(model_dim, wavelet_dim=96) if use_refine else None
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

        def _encode_core(
            self,
            signal: torch.Tensor,
            wavelet_features: torch.Tensor,
            quality: torch.Tensor,
        ) -> Dict[str, torch.Tensor]:
            ppg_feats = self.ppg_stem(signal[:, :1])
            for block in self.ppg_blocks:
                ppg_feats = block(ppg_feats)

            wavelet_input = torch.cat([wavelet_features, quality], dim=1)
            wavelet_embedding = self.wavelet_mlp(wavelet_input)
            if self.refine_only is not None:
                ppg_feats = self.refine_only(ppg_feats, wavelet_embedding)

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

    class PPGOnlyWaveletGuidedWatchNet(nn.Module):
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
            self.encoder = PPGOnlyWaveletGuidedWatchEncoder(
                ppg_channels=ppg_channels,
                acc_channels=acc_channels,
                wavelet_dim=wavelet_dim,
                model_dim=model_dim,
                transformer_layers=transformer_layers,
                transformer_heads=transformer_heads,
                fusion_hidden_dim=fusion_hidden_dim,
                embed_dim=embed_dim,
                watch_enhancement=watch_enhancement,
                watch_motion_mode=watch_motion_mode,
            )
            self.classifier = nn.Linear(embed_dim, num_classes)
            self.contrastive_head = nn.Sequential(
                nn.Linear(embed_dim, 128),
                nn.GELU(),
                nn.Linear(128, 128),
            )
            self.wavelet_predictor = nn.Sequential(
                nn.Linear(model_dim, 96),
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
            out = self.encoder(
                signal,
                wavelet_features,
                quality,
                baseline_signal=baseline_signal,
                baseline_wavelet_features=baseline_wavelet_features,
                baseline_quality=baseline_quality,
            )
            embedding = out["embedding"]
            out["logits"] = self.classifier(embedding)
            out["contrastive"] = torch.nn.functional.normalize(self.contrastive_head(embedding), dim=1)
            out["wavelet_pred"] = self.wavelet_predictor(out["temporal_embedding"])
            return out

    galaxy_models.WaveletGuidedWatchEncoder = PPGOnlyWaveletGuidedWatchEncoder
    galaxy_models.WaveletGuidedWatchNet = PPGOnlyWaveletGuidedWatchNet

    class PPGOnlyResNet18WatchNet(OriginalResNet18WatchNet):
        def __init__(self, in_channels: int = 5, wavelet_dim: int = 4, num_classes: int = 2) -> None:
            del in_channels
            super().__init__(in_channels=1, wavelet_dim=wavelet_dim, num_classes=num_classes)

        def forward(self, signal: torch.Tensor, *args, **kwargs) -> Dict[str, torch.Tensor]:
            return super().forward(signal[:, :1], *args, **kwargs)

    class PPGOnlyResNet34WatchNet(OriginalResNet34WatchNet):
        def __init__(self, in_channels: int = 5, wavelet_dim: int = 4, num_classes: int = 2) -> None:
            del in_channels
            super().__init__(in_channels=1, wavelet_dim=wavelet_dim, num_classes=num_classes)

        def forward(self, signal: torch.Tensor, *args, **kwargs) -> Dict[str, torch.Tensor]:
            return super().forward(signal[:, :1], *args, **kwargs)

    class PPGOnlyResNet50WatchNet(OriginalResNet50WatchNet):
        def __init__(self, in_channels: int = 5, wavelet_dim: int = 4, num_classes: int = 2) -> None:
            del in_channels
            super().__init__(in_channels=1, wavelet_dim=wavelet_dim, num_classes=num_classes)

        def forward(self, signal: torch.Tensor, *args, **kwargs) -> Dict[str, torch.Tensor]:
            return super().forward(signal[:, :1], *args, **kwargs)

    galaxy_models.ResNet18WatchNet = PPGOnlyResNet18WatchNet
    galaxy_models.ResNet34WatchNet = PPGOnlyResNet34WatchNet
    galaxy_models.ResNet50WatchNet = PPGOnlyResNet50WatchNet


def install_acc_only_watch_models() -> None:
    from . import galaxy_models
    from .galaxy_models import (
        AttentionPool1d,
        MultiScaleResidualBlock,
        sinusoidal_positional_encoding,
    )

    class AccOnlyWaveletGuidedWatchEncoder(nn.Module):
        """ACC-only watch encoder for input-modality ablation.

        This removes the PPG/BVP waveform from the model path. To keep the
        ablation strict, PPG-derived wavelet coefficients are zeroed before the
        wavelet/quality MLP; use wavelet-weight 0.0 when training this variant.
        """

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
            del ppg_channels, watch_enhancement, watch_motion_mode
            self.watch_enhancement = "acc_only"
            self.watch_motion_mode = "none"
            self.acc_stem = nn.Sequential(
                nn.Conv1d(acc_channels, 48, kernel_size=9, stride=2, padding=4, bias=False),
                nn.BatchNorm1d(48),
                nn.GELU(),
            )
            self.acc_blocks = nn.ModuleList(
                [
                    MultiScaleResidualBlock(48, 64, stride=1, kernels=(5, 11, 19), dilations=(1, 1, 2)),
                    MultiScaleResidualBlock(64, 96, stride=2, kernels=(5, 11, 19), dilations=(1, 1, 2)),
                    MultiScaleResidualBlock(96, model_dim, stride=2, kernels=(5, 11, 19), dilations=(1, 1, 2)),
                ]
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

        def _encode_core(
            self,
            signal: torch.Tensor,
            wavelet_features: torch.Tensor,
            quality: torch.Tensor,
        ) -> Dict[str, torch.Tensor]:
            acc_feats = self.acc_stem(signal[:, 1:])
            for block in self.acc_blocks:
                acc_feats = block(acc_feats)

            zero_wavelet = torch.zeros_like(wavelet_features)
            wavelet_input = torch.cat([zero_wavelet, quality], dim=1)
            wavelet_embedding = self.wavelet_mlp(wavelet_input)

            tokens = acc_feats.transpose(1, 2)
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

    class AccOnlyWaveletGuidedWatchNet(nn.Module):
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
            self.encoder = AccOnlyWaveletGuidedWatchEncoder(
                ppg_channels=ppg_channels,
                acc_channels=acc_channels,
                wavelet_dim=wavelet_dim,
                model_dim=model_dim,
                transformer_layers=transformer_layers,
                transformer_heads=transformer_heads,
                fusion_hidden_dim=fusion_hidden_dim,
                embed_dim=embed_dim,
                watch_enhancement=watch_enhancement,
                watch_motion_mode=watch_motion_mode,
            )
            self.classifier = nn.Linear(embed_dim, num_classes)
            self.contrastive_head = nn.Sequential(
                nn.Linear(embed_dim, 128),
                nn.GELU(),
                nn.Linear(128, 128),
            )
            self.wavelet_predictor = nn.Sequential(
                nn.Linear(model_dim, 96),
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
            out = self.encoder(
                signal,
                wavelet_features,
                quality,
                baseline_signal=baseline_signal,
                baseline_wavelet_features=baseline_wavelet_features,
                baseline_quality=baseline_quality,
            )
            embedding = out["embedding"]
            out["logits"] = self.classifier(embedding)
            out["contrastive"] = F.normalize(self.contrastive_head(embedding), dim=1)
            out["wavelet_pred"] = self.wavelet_predictor(out["temporal_embedding"])
            return out

    galaxy_models.WaveletGuidedWatchEncoder = AccOnlyWaveletGuidedWatchEncoder
    galaxy_models.WaveletGuidedWatchNet = AccOnlyWaveletGuidedWatchNet


def install_simple_concat_watch_models(use_refine: bool = False) -> None:
    from . import galaxy_models

    OriginalWaveletGuidedWatchEncoder = galaxy_models.WaveletGuidedWatchEncoder
    OriginalWaveletGuidedWatchNet = galaxy_models.WaveletGuidedWatchNet

    class SimpleConcatWaveletGuidedWatchEncoder(OriginalWaveletGuidedWatchEncoder):
        """ACC branch ablation with late concat only.

        The production parser in some training scripts does not expose
        ``acc_concat`` as a legal command-line choice. This wrapper keeps the
        command-line value as ``none`` and forces the internal model to use the
        existing acc_concat path: independent PPG/BVP and ACC branches followed
        by 1x1 feature fusion, without ACC-to-PPG FiLM or enhancer cleaning.
        """

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
            del watch_enhancement
            super().__init__(
                ppg_channels=ppg_channels,
                acc_channels=acc_channels,
                wavelet_dim=wavelet_dim,
                model_dim=model_dim,
                transformer_layers=transformer_layers,
                transformer_heads=transformer_heads,
                fusion_hidden_dim=fusion_hidden_dim,
                embed_dim=embed_dim,
                watch_enhancement="acc_concat",
                watch_motion_mode=watch_motion_mode,
            )
            if use_refine:
                self.ppg_enhancer = RefineOnlyPPGEnhancer(model_dim, 128, wavelet_dim=96)

    class SimpleConcatWaveletGuidedWatchNet(OriginalWaveletGuidedWatchNet):
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
            del watch_enhancement
            super().__init__(
                ppg_channels=ppg_channels,
                acc_channels=acc_channels,
                wavelet_dim=wavelet_dim,
                num_classes=num_classes,
                model_dim=model_dim,
                transformer_layers=transformer_layers,
                transformer_heads=transformer_heads,
                fusion_hidden_dim=fusion_hidden_dim,
                embed_dim=embed_dim,
                watch_enhancement="acc_concat",
                watch_motion_mode=watch_motion_mode,
            )
            if use_refine:
                self.ppg_enhancer = RefineOnlyPPGEnhancer(model_dim, 128, wavelet_dim=96)

    galaxy_models.WaveletGuidedWatchEncoder = SimpleConcatWaveletGuidedWatchEncoder
    galaxy_models.WaveletGuidedWatchNet = SimpleConcatWaveletGuidedWatchNet


def install_gated_fusion_watch_models(use_refine: bool = False) -> None:
    from . import galaxy_models
    from .galaxy_models import sinusoidal_positional_encoding

    OriginalWaveletGuidedWatchEncoder = galaxy_models.WaveletGuidedWatchEncoder
    OriginalWaveletGuidedWatchNet = galaxy_models.WaveletGuidedWatchNet

    class GatedFusionWaveletGuidedWatchEncoder(OriginalWaveletGuidedWatchEncoder):
        """ACC branch ablation with gated late fusion.

        PPG/BVP and ACC are encoded independently. ACC is projected into the PPG
        feature space, and a learned 1x1 gate mixes the two streams:
        h_fuse = g * h_ppg + (1 - g) * P_acc(h_acc).
        No ACC-to-PPG FiLM adaptation or artifact cleaning is used.
        """

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
            del watch_enhancement
            super().__init__(
                ppg_channels=ppg_channels,
                acc_channels=acc_channels,
                wavelet_dim=wavelet_dim,
                model_dim=model_dim,
                transformer_layers=transformer_layers,
                transformer_heads=transformer_heads,
                fusion_hidden_dim=fusion_hidden_dim,
                embed_dim=embed_dim,
                watch_enhancement="acc_concat",
                watch_motion_mode=watch_motion_mode,
            )
            self.watch_enhancement = "gated_fusion"
            self.acc_to_ppg_gate_proj = nn.Sequential(
                nn.Conv1d(128, model_dim, kernel_size=1, bias=False),
                nn.BatchNorm1d(model_dim),
                nn.GELU(),
            )
            self.acc_concat_fusion = None
            self.acc_ppg_gate = nn.Sequential(
                nn.Conv1d(model_dim * 2, model_dim, kernel_size=1),
                nn.Sigmoid(),
            )
            if use_refine:
                self.ppg_enhancer = RefineOnlyPPGEnhancer(model_dim, 128, wavelet_dim=96)
            else:
                self.ppg_enhancer = None

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
            for ppg_block, acc_block in zip(self.ppg_blocks, self.acc_blocks):
                ppg_feats = ppg_block(ppg_feats)
                acc_feats = acc_block(acc_feats)

            if acc_feats.shape[-1] != ppg_feats.shape[-1]:
                acc_feats = F.interpolate(acc_feats, size=ppg_feats.shape[-1], mode="linear", align_corners=False)
            acc_ppg = self.acc_to_ppg_gate_proj(acc_feats)
            gate = self.acc_ppg_gate(torch.cat([ppg_feats, acc_ppg], dim=1))
            ppg_feats = gate * ppg_feats + (1.0 - gate) * acc_ppg

            wavelet_input = torch.cat([wavelet_features, quality], dim=1)
            wavelet_embedding = self.wavelet_mlp(wavelet_input)
            enhancer_aux: Dict[str, torch.Tensor] = {
                "gated_fusion_gate": gate,
                "gated_fusion_acc_projection": acc_ppg,
            }
            if self.ppg_enhancer is not None:
                ppg_feats, refine_aux = self.ppg_enhancer(ppg_feats, acc_feats, wavelet_embedding)
                enhancer_aux.update(refine_aux)

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

    class GatedFusionWaveletGuidedWatchNet(OriginalWaveletGuidedWatchNet):
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
            del watch_enhancement
            super().__init__(
                ppg_channels=ppg_channels,
                acc_channels=acc_channels,
                wavelet_dim=wavelet_dim,
                num_classes=num_classes,
                model_dim=model_dim,
                transformer_layers=transformer_layers,
                transformer_heads=transformer_heads,
                fusion_hidden_dim=fusion_hidden_dim,
                embed_dim=embed_dim,
                watch_enhancement="acc_concat",
                watch_motion_mode=watch_motion_mode,
            )
            self.watch_enhancement = "gated_fusion"
            self.acc_to_ppg_gate_proj = nn.Sequential(
                nn.Conv1d(128, model_dim, kernel_size=1, bias=False),
                nn.BatchNorm1d(model_dim),
                nn.GELU(),
            )
            self.acc_concat_fusion = None
            self.acc_ppg_gate = nn.Sequential(
                nn.Conv1d(model_dim * 2, model_dim, kernel_size=1),
                nn.Sigmoid(),
            )
            if use_refine:
                self.ppg_enhancer = RefineOnlyPPGEnhancer(model_dim, 128, wavelet_dim=96)
            else:
                self.ppg_enhancer = None

        _encode_core = GatedFusionWaveletGuidedWatchEncoder._encode_core

    galaxy_models.WaveletGuidedWatchEncoder = GatedFusionWaveletGuidedWatchEncoder
    galaxy_models.WaveletGuidedWatchNet = GatedFusionWaveletGuidedWatchNet


def install_ablation(ablation: str) -> str:
    if ablation == "acc_only":
        install_acc_only_watch_models()
        return "none"
    if ablation == "ppg_only":
        install_ppg_only_watch_models(use_refine=False)
        return "none"
    if ablation == "ppg_only_refine":
        install_ppg_only_watch_models(use_refine=True)
        return "none"
    if ablation == "simple_concat":
        install_simple_concat_watch_models(use_refine=False)
        return "none"
    if ablation == "simple_concat_refine":
        install_simple_concat_watch_models(use_refine=True)
        return "none"
    if ablation == "gated_fusion":
        install_gated_fusion_watch_models(use_refine=False)
        return "none"
    if ablation == "gated_fusion_refine":
        install_gated_fusion_watch_models(use_refine=True)
        return "none"
    raise ValueError(f"Unsupported ablation: {ablation}")


def train_one_watch() -> None:
    ablation = _pop_hidden_arg("--_ablation", "ppg_only")
    dataset_kind = _pop_hidden_arg("--_dataset-kind", "galaxy")
    _remove_flag("--_train-watch-one")
    train_enhancement = install_ablation(ablation)
    print(f"watch_input_ablation={ablation}")
    if ablation == "acc_only":
        print("ablation_detail=ACC only plus quality; PPG/BVP waveform and PPG-derived wavelet features are removed from the model path")
    elif ablation == "ppg_only":
        print("ablation_detail=PPG/BVP only; ACC channels are not used by the model")
    elif ablation == "ppg_only_refine":
        print("ablation_detail=PPG/BVP only plus wavelet-guided refine; ACC channels are not used by the model")
    elif ablation == "simple_concat":
        print("ablation_detail=PPG/BVP branch + ACC branch with feature concat and 1x1 fusion; no ACC-to-PPG modulation")
    elif ablation == "simple_concat_refine":
        print("ablation_detail=PPG/BVP branch + ACC branch with feature concat and 1x1 fusion plus wavelet-guided refine; no ACC-to-PPG modulation/cleaning")
    elif ablation == "gated_fusion":
        print("ablation_detail=PPG/BVP branch + ACC branch with gated late fusion; no ACC-to-PPG modulation/cleaning")
    else:
        print("ablation_detail=PPG/BVP branch + ACC branch with gated late fusion plus wavelet-guided refine; no ACC-to-PPG modulation/cleaning")
    _replace_arg_value("--watch-enhancement", train_enhancement)

    if dataset_kind == "galaxy":
        from .train_galaxy_watch import main as train_main
    elif dataset_kind in {"wesad", "catsa"}:
        from .train_wesad_watch import main as train_main
    else:
        raise ValueError(f"Unsupported dataset-kind: {dataset_kind}")
    train_main()


def train_one_privileged() -> None:
    ablation = _pop_hidden_arg("--_ablation", "ppg_only")
    dataset_kind = _pop_hidden_arg("--_dataset-kind", "galaxy")
    _remove_flag("--_train-priv-one")
    train_enhancement = install_ablation(ablation)
    print(f"watch_input_ablation={ablation}")
    if ablation == "acc_only":
        print("ablation_detail=ACC only plus quality; PPG/BVP waveform and PPG-derived wavelet features are removed from the model path")
    elif ablation == "ppg_only":
        print("ablation_detail=PPG/BVP only; ACC channels are not used by the model")
    elif ablation == "ppg_only_refine":
        print("ablation_detail=PPG/BVP only plus wavelet-guided refine; ACC channels are not used by the model")
    elif ablation == "simple_concat":
        print("ablation_detail=PPG/BVP branch + ACC branch with feature concat and 1x1 fusion; no ACC-to-PPG modulation")
    elif ablation == "simple_concat_refine":
        print("ablation_detail=PPG/BVP branch + ACC branch with feature concat and 1x1 fusion plus wavelet-guided refine; no ACC-to-PPG modulation/cleaning")
    elif ablation == "gated_fusion":
        print("ablation_detail=PPG/BVP branch + ACC branch with gated late fusion; no ACC-to-PPG modulation/cleaning")
    else:
        print("ablation_detail=PPG/BVP branch + ACC branch with gated late fusion plus wavelet-guided refine; no ACC-to-PPG modulation/cleaning")
    _replace_arg_value("--watch-enhancement", train_enhancement)

    if dataset_kind == "galaxy":
        from .train_galaxy_privileged_elastic import main as train_main
    elif dataset_kind in {"wesad", "catsa"}:
        from .train_wesad_privileged_elastic import main as train_main
    else:
        raise ValueError(f"Unsupported dataset-kind: {dataset_kind}")
    train_main()


def _replace_arg_value(name: str, value: str) -> None:
    if name in sys.argv:
        idx = sys.argv.index(name)
        if idx + 1 >= len(sys.argv):
            raise ValueError(f"Missing value for {name}")
        sys.argv[idx + 1] = value


def discover_manifests(manifests_dir: Path, subjects: list[str] | None, dataset_kind: str) -> list[tuple[str, Path]]:
    requested = {subject.strip() for subject in subjects or [] if subject.strip()}
    prefix = "galaxy" if dataset_kind == "galaxy" else ("catsa" if dataset_kind == "catsa" else "wesad")
    manifests: dict[str, Path] = {}
    for pattern in (f"{prefix}_*_loso_val.csv", "*_loso_val.csv"):
        for path in sorted(manifests_dir.glob(pattern)):
            subject = path.stem
            if subject.startswith(f"{prefix}_"):
                subject = subject[len(prefix) + 1 :]
            if subject.endswith("_loso_val"):
                subject = subject[: -len("_loso_val")]
            if requested and subject not in requested:
                continue
            manifests.setdefault(subject, path)
    return sorted(manifests.items())


def run_and_capture(command: list[str], cwd: Path, log_path: Path) -> str:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    chunks: list[str] = []
    assert process.stdout is not None
    with log_path.open("w", encoding="utf-8") as log_file:
        while True:
            chunk = process.stdout.read(1)
            if chunk == "":
                break
            sys.stdout.write(chunk)
            sys.stdout.flush()
            log_file.write(chunk)
            log_file.flush()
            chunks.append(chunk)
    return_code = process.wait()
    output = "".join(chunks)
    if return_code != 0:
        raise RuntimeError(f"Command failed with exit code {return_code}: {' '.join(command)}")
    return output


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def _metrics_helpers(dataset_kind: str):
    if dataset_kind == "galaxy":
        from .run_galaxy_loso_eval_elastic import (
            PRIV_WATCH_PATTERN,
            TEACHER_PATTERN,
            WATCH_ONLY_PATTERN,
            collapse_flag,
            load_test_positive_prior,
            parse_last_metrics,
        )
    else:
        from .run_wesad_loso_eval_elastic import (
            PRIV_WATCH_PATTERN,
            TEACHER_PATTERN,
            WATCH_ONLY_PATTERN,
            collapse_flag,
            load_test_positive_prior,
            parse_last_metrics,
        )
    return WATCH_ONLY_PATTERN, PRIV_WATCH_PATTERN, TEACHER_PATTERN, collapse_flag, load_test_positive_prior, parse_last_metrics


def _append_summary_block(lines: list[str], name: str, rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    lines.append(f"[summary:{name}]")
    for key in ("balanced_acc", "auroc", "f1"):
        mean, std = mean_std([row[key] for row in rows])
        lines.append(f"{name} {key}_mean={mean:.4f} {key}_std={std:.4f}")
    lines.append(f"{name} collapse_rate={statistics.mean(row['collapse'] for row in rows):.4f}")
    mean, std = mean_std([row["positive_rate_error"] for row in rows])
    lines.append(f"{name} positive_rate_error_mean={mean:.4f} positive_rate_error_std={std:.4f}")
    lines.append("")


def main() -> None:
    if "--_train-watch-one" in sys.argv:
        train_one_watch()
        return
    if "--_train-priv-one" in sys.argv:
        train_one_privileged()
        return

    parser = argparse.ArgumentParser(description="Run PPG-only and simple ACC-concat LOSO watch-input ablations.")
    parser.add_argument("--dataset-kind", type=str, required=True, choices=["galaxy", "wesad", "catsa"])
    parser.add_argument(
        "--ablation",
        type=str,
        required=True,
        choices=[
            "acc_only",
            "ppg_only",
            "simple_concat",
            "ppg_only_refine",
            "simple_concat_refine",
            "gated_fusion",
            "gated_fusion_refine",
        ],
    )
    parser.add_argument("--manifests-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", "--wesad-root", dest="dataset_root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--skip-watch-only", action="store_true")
    parser.add_argument("--skip-privileged", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--monitor", type=str, default="auroc")
    parser.add_argument("--threshold-metric", type=str, default="balanced_acc", choices=["monitor", "acc", "balanced_acc", "f1", "auroc"])
    parser.add_argument("--selection-target", type=str, default="watch", choices=["watch", "teacher"])
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--watch-batch-size", type=int, default=32)
    parser.add_argument("--deploy-batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--cache-subjects", type=int, default=15)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--calm-sessions", nargs="*", default=None)
    parser.add_argument("--stress-sessions", nargs="*", default=None)
    parser.add_argument("--watch-contrastive-weight", type=float, default=0.0)
    parser.add_argument("--watch-wavelet-weight", type=float, default=0.2)
    parser.add_argument(
        "--model-type",
        type=str,
        default="wavelet_guided",
        choices=["wavelet_guided", "resnet18_1d", "resnet34_1d", "resnet50_1d"],
    )
    parser.add_argument("--priv-wavelet-weight", type=float, default=0.05)
    parser.add_argument("--teacher-cls-weight", type=float, default=0.80)
    parser.add_argument("--teacher-fused-cls-weight", type=float, default=0.0)
    parser.add_argument("--e4-cls-weight", type=float, default=0.05)
    parser.add_argument("--rhythm-weight", type=float, default=0.15)
    parser.add_argument("--privileged-cls-weight", type=float, default=0.05)
    parser.add_argument("--distill-weight", type=float, default=0.08)
    parser.add_argument("--distill-temp", type=float, default=4.0)
    parser.add_argument("--ranking-distill-weight", type=float, default=0.0)
    parser.add_argument("--distribution-weight", type=float, default=0.0)
    parser.add_argument("--session-consistency-weight", type=float, default=0.0)
    parser.add_argument("--embedding-align-weight", "--align-weight", dest="embedding_align_weight", type=float, default=0.0)
    parser.add_argument("--cross-confidence-distill", action="store_true")
    parser.add_argument("--cross-confidence-targets", nargs="*", default=["kd"], choices=["kd", "ranking", "distribution"])
    parser.add_argument("--cross-confidence-min-weight", type=float, default=0.0)
    parser.add_argument("--watch-model-dim", type=int, default=192)
    parser.add_argument("--watch-transformer-layers", type=int, default=2)
    parser.add_argument("--watch-transformer-heads", type=int, default=4)
    parser.add_argument("--watch-fusion-hidden-dim", type=int, default=256)
    parser.add_argument("--watch-embed-dim", type=int, default=160)
    parser.add_argument("--align-proj-dim", type=int, default=128)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    manifests = discover_manifests(args.manifests_dir, args.subjects, args.dataset_kind)
    if not manifests:
        examples = [path.name for path in sorted(args.manifests_dir.glob("*.csv"))[:10]] if args.manifests_dir.exists() else []
        raise ValueError(f"No LOSO manifests found in {args.manifests_dir}; csv_examples={examples}")
    if args.ablation in {"acc_only", "simple_concat", "simple_concat_refine", "gated_fusion", "gated_fusion_refine"} and args.model_type != "wavelet_guided":
        raise ValueError("ACC-branch ablations are only defined for the wavelet_guided model. Use ppg_only for ResNet ablations.")

    calm_sessions = args.calm_sessions or (["baseline"] if args.dataset_kind == "galaxy" else ["baseline"])
    stress_sessions = args.stress_sessions or (["tsst-prep"] if args.dataset_kind == "galaxy" else ["stress"])
    deploy_batch_size = args.deploy_batch_size
    if deploy_batch_size is None:
        deploy_batch_size = 16 if args.dataset_kind == "galaxy" else 24

    output_dir = args.output_dir
    logs_dir = output_dir / "logs"
    ckpt_dir = output_dir / "checkpoints"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.dataset_kind
    csv_path = output_dir / f"{prefix}_{args.ablation}_loso_results.csv"
    summary_path = output_dir / f"{prefix}_{args.ablation}_loso_summary.txt"

    WATCH_ONLY_PATTERN, PRIV_WATCH_PATTERN, TEACHER_PATTERN, collapse_flag, load_test_positive_prior, parse_last_metrics = _metrics_helpers(
        args.dataset_kind
    )
    csv_rows: list[dict[str, object]] = []
    watch_rows: list[dict[str, float]] = []
    deploy_rows: list[dict[str, float]] = []
    teacher_rows: list[dict[str, float]] = []
    summary_lines = [
        f"dataset={args.dataset_kind}",
        f"watch_input_ablation={args.ablation}",
        f"model_type={args.model_type}",
        "acc_only=ACC plus quality only; PPG/BVP waveform and PPG-derived wavelet features are removed from the model path",
        "ppg_only=PPG/BVP plus wavelet/quality only; ACC removed from model path",
        "ppg_only_refine=PPG/BVP plus wavelet/quality and the refine-only module; ACC removed from model path",
        "simple_concat=PPG/BVP and ACC branches encoded separately, concatenated, and fused by 1x1 conv; no ACC-to-PPG modulation/cleaning",
        "simple_concat_refine=simple_concat plus the refine-only module; no ACC-to-PPG modulation/cleaning",
        "gated_fusion=PPG/BVP and ACC branches encoded separately, fused by a learned per-channel gate; no ACC-to-PPG modulation/cleaning",
        "gated_fusion_refine=gated_fusion plus the refine-only module; no ACC-to-PPG modulation/cleaning",
        "",
    ]

    for subject, manifest_path in manifests:
        summary_lines.append(f"[{subject}]")
        prior = load_test_positive_prior(manifest_path, calm_sessions=list(calm_sessions), stress_sessions=list(stress_sessions))
        summary_lines.append(f"test_positive_prior={prior:.4f}")
        watch_metrics: dict[str, float] | None = None

        if not args.skip_watch_only:
            watch_log = logs_dir / f"{subject}_watch_only.log"
            watch_metrics_csv = logs_dir / f"{subject}_watch_only_metrics.csv"
            watch_ckpt = ckpt_dir / f"{subject}_watch_only.pt"
            watch_command = [
                sys.executable,
                "-m",
                "stress_ssl_distill.run_watch_input_ablation_loso",
                "--_train-watch-one",
                "--_ablation",
                args.ablation,
                "--_dataset-kind",
                args.dataset_kind,
                "--manifest",
                str(manifest_path),
                "--dataset-root",
                str(args.dataset_root),
                "--save-path",
                str(watch_ckpt),
                "--metrics-path",
                str(watch_metrics_csv),
                "--device",
                args.device,
                "--epochs",
                str(args.epochs),
                "--selection-mode",
                "early_stop",
                "--monitor",
                args.monitor,
                "--threshold-metric",
                args.threshold_metric,
                "--early-stop-patience",
                str(args.early_stop_patience),
                "--batch-size",
                str(args.watch_batch_size),
                "--lr",
                str(args.lr),
                "--weight-decay",
                str(args.weight_decay),
                "--num-workers",
                str(args.num_workers),
                "--seed",
                str(args.seed),
                "--model-type",
                args.model_type,
                "--watch-enhancement",
                "none",
                "--watch-model-dim",
                str(args.watch_model_dim),
                "--watch-transformer-layers",
                str(args.watch_transformer_layers),
                "--watch-transformer-heads",
                str(args.watch_transformer_heads),
                "--watch-fusion-hidden-dim",
                str(args.watch_fusion_hidden_dim),
                "--watch-embed-dim",
                str(args.watch_embed_dim),
                "--calm-sessions",
                *calm_sessions,
                "--stress-sessions",
                *stress_sessions,
            ]
            if args.dataset_kind != "galaxy":
                watch_command.extend(["--dataset-kind", args.dataset_kind, "--cache-subjects", str(args.cache_subjects)])
                watch_command.extend(["--contrastive-weight", str(args.watch_contrastive_weight), "--wavelet-weight", str(args.watch_wavelet_weight)])
            else:
                watch_command.extend(["--contrastive-weight", str(args.watch_contrastive_weight), "--wavelet-weight", str(args.watch_wavelet_weight)])
            if args.pin_memory:
                watch_command.append("--pin-memory")
            watch_output = run_and_capture(watch_command, cwd=repo_root, log_path=watch_log)
            watch_metrics = parse_last_metrics(watch_output, WATCH_ONLY_PATTERN, "watch-only")
            watch_metrics["collapse"] = float(collapse_flag(watch_metrics["positive_rate"]))
            watch_metrics["positive_rate_error"] = abs(watch_metrics["positive_rate"] - prior)
            watch_rows.append(watch_metrics)
            summary_lines.append("watch_only " + " ".join(f"{k}={v:.4f}" for k, v in watch_metrics.items() if k != "collapse"))
        else:
            summary_lines.append("watch_only skipped=true")

        if not args.skip_privileged:
            deploy_log = logs_dir / f"{subject}_deploy_watch.log"
            deploy_metrics_csv = logs_dir / f"{subject}_deploy_watch_metrics.csv"
            deploy_ckpt = ckpt_dir / f"{subject}_deploy_watch.pt"
            deploy_command = [
                sys.executable,
                "-m",
                "stress_ssl_distill.run_watch_input_ablation_loso",
                "--_train-priv-one",
                "--_ablation",
                args.ablation,
                "--_dataset-kind",
                args.dataset_kind,
                "--manifest",
                str(manifest_path),
                "--dataset-root",
                str(args.dataset_root),
                "--save-path",
                str(deploy_ckpt),
                "--metrics-path",
                str(deploy_metrics_csv),
                "--device",
                args.device,
                "--epochs",
                str(args.epochs),
                "--selection-mode",
                "early_stop",
                "--selection-target",
                args.selection_target,
                "--monitor",
                args.monitor,
                "--threshold-metric",
                args.threshold_metric,
                "--early-stop-patience",
                str(args.early_stop_patience),
                "--batch-size",
                str(deploy_batch_size),
                "--lr",
                str(args.lr),
                "--weight-decay",
                str(args.weight_decay),
                "--num-workers",
                str(args.num_workers),
                "--seed",
                str(args.seed),
                "--watch-backbone",
                "wavelet_guided",
                "--watch-enhancement",
                "none",
                "--teacher-cls-weight",
                str(args.teacher_cls_weight),
                "--distill-weight",
                str(args.distill_weight),
                "--distill-temp",
                str(args.distill_temp),
                "--ranking-distill-weight",
                str(args.ranking_distill_weight),
                "--distribution-weight",
                str(args.distribution_weight),
                "--session-consistency-weight",
                str(args.session_consistency_weight),
                "--calm-sessions",
                *calm_sessions,
                "--stress-sessions",
                *stress_sessions,
            ]
            if args.dataset_kind == "galaxy":
                deploy_command.extend(
                    [
                        "--teacher-fused-cls-weight",
                        str(args.teacher_fused_cls_weight),
                        "--e4-cls-weight",
                        str(args.e4_cls_weight),
                        "--rhythm-weight",
                        str(args.rhythm_weight),
                        "--wavelet-weight",
                        str(args.priv_wavelet_weight),
                        "--align-weight",
                        str(args.embedding_align_weight),
                    ]
                )
            else:
                deploy_command.extend(
                    [
                        "--dataset-kind",
                        args.dataset_kind,
                        "--cache-subjects",
                        str(args.cache_subjects),
                        "--privileged-cls-weight",
                        str(args.privileged_cls_weight),
                        "--embedding-align-weight",
                        str(args.embedding_align_weight),
                        "--margin-match-weight",
                        "0.0",
                        "--normalized-margin-align-weight",
                        "0.0",
                        "--subject-center-stability-weight",
                        "0.0",
                        "--validation-threshold-stability-weight",
                        "0.0",
                        "--distill-gating",
                        "none",
                        "--watch-model-dim",
                        str(args.watch_model_dim),
                        "--watch-transformer-layers",
                        str(args.watch_transformer_layers),
                        "--watch-transformer-heads",
                        str(args.watch_transformer_heads),
                        "--watch-fusion-hidden-dim",
                        str(args.watch_fusion_hidden_dim),
                        "--watch-embed-dim",
                        str(args.watch_embed_dim),
                        "--align-proj-dim",
                        str(args.align_proj_dim),
                    ]
                )
            deploy_command.extend(
                [
                    "--reliability-distill-weight",
                    "0.0",
                    "--correction-cls-weight",
                    "0.0",
                    "--correction-base-anchor-weight",
                    "0.0",
                    "--correction-nondegradation-weight",
                    "0.0",
                    "--correction-align-weight",
                    "0.0",
                    "--alpha-helpfulness-weight",
                    "0.0",
                    "--alpha-sparsity-weight",
                    "0.0",
                    "--elastic-residual-weight",
                    "0.0",
                    "--elastic-alpha-target-weight",
                    "0.0",
                    "--cross-confidence-min-weight",
                    str(args.cross_confidence_min_weight),
                ]
            )
            if args.pin_memory:
                deploy_command.append("--pin-memory")
            if args.cross_confidence_distill:
                deploy_command.append("--cross-confidence-distill")
                deploy_command.append("--cross-confidence-targets")
                deploy_command.extend(args.cross_confidence_targets)
            deploy_output = run_and_capture(deploy_command, cwd=repo_root, log_path=deploy_log)
            deploy_metrics = parse_last_metrics(deploy_output, PRIV_WATCH_PATTERN, "deploy-watch")
            teacher_metrics = parse_last_metrics(deploy_output, TEACHER_PATTERN, "teacher")
            deploy_metrics["collapse"] = float(collapse_flag(deploy_metrics["positive_rate"]))
            teacher_metrics["collapse"] = float(collapse_flag(teacher_metrics["positive_rate"]))
            deploy_metrics["positive_rate_error"] = abs(deploy_metrics["positive_rate"] - prior)
            teacher_metrics["positive_rate_error"] = abs(teacher_metrics["positive_rate"] - prior)
            deploy_rows.append(deploy_metrics)
            teacher_rows.append(teacher_metrics)
            summary_lines.append("deploy_watch " + " ".join(f"{k}={v:.4f}" for k, v in deploy_metrics.items() if k != "collapse"))
            summary_lines.append("teacher " + " ".join(f"{k}={v:.4f}" for k, v in teacher_metrics.items() if k != "collapse"))
        else:
            summary_lines.append("privileged skipped=true")

        summary_lines.append("")
        for model_name, metrics in (("watch_only", watch_metrics),):
            if metrics is not None:
                csv_rows.append({"subject": subject, "model": model_name, "ablation": args.ablation, **metrics})
        if not args.skip_privileged and deploy_rows:
            csv_rows.append({"subject": subject, "model": "deploy_watch", "ablation": args.ablation, **deploy_rows[-1]})
            csv_rows.append({"subject": subject, "model": "teacher", "ablation": args.ablation, **teacher_rows[-1]})

    _append_summary_block(summary_lines, f"watch_only_{args.ablation}", watch_rows)
    _append_summary_block(summary_lines, f"deploy_watch_{args.ablation}", deploy_rows)
    _append_summary_block(summary_lines, f"teacher_{args.ablation}", teacher_rows)

    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"Saved ablation CSV to {csv_path}")
    print(f"Saved ablation summary to {summary_path}")


if __name__ == "__main__":
    main()
