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
    def __init__(
        self,
        wavelet_dim: int = 4,
        privileged_channels: int = 9,
        num_classes: int = 2,
        embed_dim: int = 160,
        align_dim: int = 128,
        watch_backbone: str = "wavelet_guided",
        model_dim: int = 192,
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        fusion_hidden_dim: int = 256,
        watch_enhancement: str = "none",
        watch_motion_mode: str = "strong",
        use_student_gated_correction: bool = False,
        correction_scale_init: float = 0.05,
        correction_alpha_init_bias: float = -2.0,
        correction_alpha_max: float = 0.35,
        correction_mode: str = "logit_mix",
    ) -> None:
        super().__init__()
        if correction_mode not in {"logit_mix", "margin_residual"}:
            raise ValueError(f"Unsupported correction mode: {correction_mode}")
        self.watch_backbone = watch_backbone
        self.watch_enhancement = watch_enhancement
        self.watch_motion_mode = watch_motion_mode
        self.use_student_gated_correction = use_student_gated_correction
        self.correction_alpha_max = float(correction_alpha_max)
        self.correction_mode = correction_mode
        self.register_buffer(
            "correction_mode_id",
            torch.tensor(1 if correction_mode == "margin_residual" else 0, dtype=torch.long),
        )
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
        self.reliability_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        self.watch_contrastive_head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Linear(128, 128),
        )
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
        self.watch_align_proj = nn.Sequential(
            nn.Linear(embed_dim, align_dim),
            nn.LayerNorm(align_dim),
            nn.GELU(),
            nn.Linear(align_dim, align_dim),
        )
        self.teacher_align_proj = nn.Sequential(
            nn.Linear(embed_dim, align_dim),
            nn.LayerNorm(align_dim),
            nn.GELU(),
            nn.Linear(align_dim, align_dim),
        )
        self.wavelet_predictor = nn.Sequential(
            nn.Linear(embed_dim, 96),
            nn.GELU(),
            nn.Linear(96, wavelet_dim),
        )
        if use_student_gated_correction:
            self.deploy_correction = nn.Sequential(
                nn.Linear(embed_dim + 1, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(embed_dim, embed_dim),
            )
            self.deploy_correction_gate = nn.Sequential(
                nn.Linear(embed_dim + 1, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim),
                nn.Sigmoid(),
            )
            self.deploy_correction_alpha = nn.Sequential(
                nn.Linear(embed_dim + 1, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, 1),
            )
            nn.init.constant_(self.deploy_correction_alpha[-1].bias, float(correction_alpha_init_bias))
            self.privileged_correction = nn.Sequential(
                nn.Linear(embed_dim * 3, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(embed_dim, embed_dim),
            )
            self.privileged_correction_gate = nn.Sequential(
                nn.Linear(embed_dim * 2 + 1, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim),
                nn.Sigmoid(),
            )
            self.correction_norm = nn.LayerNorm(embed_dim)
            self.correction_scale = nn.Parameter(torch.tensor(float(correction_scale_init)))
        else:
            self.deploy_correction = None
            self.deploy_correction_gate = None
            self.deploy_correction_alpha = None
            self.privileged_correction = None
            self.privileged_correction_gate = None
            self.correction_norm = None
            self.correction_scale = None

    def _quality_column(self, quality: torch.Tensor) -> torch.Tensor:
        if quality.ndim == 1:
            quality = quality.unsqueeze(1)
        return quality.float().clamp(0.0, 1.0)

    def _correction_scale_value(self) -> torch.Tensor:
        if self.correction_scale is None:
            raise RuntimeError("Correction scale requested while student-gated correction is disabled.")
        return torch.tanh(self.correction_scale)

    def _combine_logits(
        self,
        base_logits: torch.Tensor,
        corrected_logits: torch.Tensor,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        if self.correction_mode == "logit_mix":
            return base_logits + alpha * (corrected_logits - base_logits)
        if base_logits.shape[1] != 2 or corrected_logits.shape[1] != 2:
            raise ValueError("margin_residual correction expects binary logits.")
        center = base_logits.mean(dim=1, keepdim=True)
        base_margin = (base_logits[:, 1] - base_logits[:, 0]).unsqueeze(1)
        corrected_margin = (corrected_logits[:, 1] - corrected_logits[:, 0]).unsqueeze(1)
        final_margin = base_margin + alpha * (corrected_margin - base_margin)
        return torch.cat([center - 0.5 * final_margin, center + 0.5 * final_margin], dim=1)

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
        reliability_logit = self.reliability_head(watch_embedding)
        base_logits = self.watch_classifier(watch_embedding)
        logits = base_logits
        out = {
            **watch_out,
            "watch_embedding": watch_embedding,
            "base_logits": base_logits,
            "logits": logits,
            "reliability_logit": reliability_logit,
            "reliability": torch.sigmoid(reliability_logit),
            "contrastive": F.normalize(self.watch_contrastive_head(watch_embedding), dim=1),
            "wavelet_pred": self.wavelet_predictor(watch_embedding),
        }
        if self.use_student_gated_correction:
            quality_col = self._quality_column(quality)
            deploy_input = torch.cat([watch_embedding, quality_col], dim=1)
            deploy_delta = self.deploy_correction(deploy_input)
            deploy_gate = self.deploy_correction_gate(deploy_input)
            deploy_embedding = self.correction_norm(
                watch_embedding + self._correction_scale_value() * deploy_gate * deploy_delta
            )
            deploy_corrected_logits = self.watch_classifier(deploy_embedding)
            deploy_alpha_unit = torch.sigmoid(self.deploy_correction_alpha(deploy_input))
            deploy_alpha = self.correction_alpha_max * deploy_alpha_unit
            out["logits"] = self._combine_logits(base_logits, deploy_corrected_logits, deploy_alpha)
            out["deploy_corrected_logits"] = deploy_corrected_logits
            out["deploy_correction_alpha"] = deploy_alpha
            out["deploy_correction_alpha_unit"] = deploy_alpha_unit
            out["deploy_correction_delta"] = deploy_delta
            out["deploy_correction_gate"] = deploy_gate
            out["deploy_correction_embedding"] = deploy_embedding
            if base_logits.shape[1] == 2 and deploy_corrected_logits.shape[1] == 2:
                out["base_margin"] = base_logits[:, 1] - base_logits[:, 0]
                out["deploy_corrected_margin"] = deploy_corrected_logits[:, 1] - deploy_corrected_logits[:, 0]
                out["final_margin"] = out["logits"][:, 1] - out["logits"][:, 0]
        return out

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
            [
                watch_embedding,
                privileged_embedding,
                delta,
                watch_embedding * privileged_embedding,
            ],
            dim=1,
        )
        fused = self.teacher_fusion(fusion_input)
        teacher_embedding = self.teacher_out_norm(watch_embedding + gate * fused)

        out["privileged_embedding"] = privileged_embedding
        out["privileged_logits"] = self.privileged_classifier(privileged_embedding)
        out["teacher_embedding"] = teacher_embedding
        out["teacher_logits"] = self.teacher_classifier(teacher_embedding)
        out["watch_align"] = F.normalize(self.watch_align_proj(out["watch_embedding"]), dim=1)
        out["teacher_align"] = F.normalize(self.teacher_align_proj(teacher_embedding), dim=1)
        if self.use_student_gated_correction:
            quality_col = self._quality_column(quality)
            teacher_context = teacher_embedding.detach()
            correction_input = torch.cat(
                [
                    out["watch_embedding"],
                    teacher_context,
                    torch.abs(out["watch_embedding"] - teacher_context),
                ],
                dim=1,
            )
            priv_delta = self.privileged_correction(correction_input)
            priv_gate = self.privileged_correction_gate(
                torch.cat([out["watch_embedding"], teacher_context, quality_col], dim=1)
            )
            priv_embedding = self.correction_norm(
                out["watch_embedding"] + self._correction_scale_value() * priv_gate * priv_delta
            )
            out["privileged_correction_delta"] = priv_delta
            out["privileged_correction_gate"] = priv_gate
            out["privileged_correction_embedding"] = priv_embedding
            out["privileged_correction_logits"] = self.watch_classifier(priv_embedding)
        return out
