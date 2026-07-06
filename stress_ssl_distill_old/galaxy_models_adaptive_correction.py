from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .galaxy_models import PrivilegedGalaxyTeacherNet


class AdaptiveCorrectionGalaxyTeacherNet(PrivilegedGalaxyTeacherNet):
    def __init__(
        self,
        *args,
        correction_alpha_init_bias: float = -3.0,
        correction_alpha_max: float = 0.35,
        correction_mode: str = "logit_mix",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if correction_mode not in {"logit_mix", "margin_residual"}:
            raise ValueError(f"Unsupported correction mode: {correction_mode}")
        self.correction_alpha_max = float(correction_alpha_max)
        self.correction_mode = correction_mode
        self.register_buffer(
            "correction_mode_id",
            torch.tensor(1 if correction_mode == "margin_residual" else 0, dtype=torch.long),
        )
        embed_dim = self.watch_classifier.in_features
        if self.use_student_gated_correction:
            self.deploy_correction_alpha = nn.Sequential(
                nn.Linear(embed_dim + 1, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, 1),
            )
            nn.init.constant_(self.deploy_correction_alpha[-1].bias, float(correction_alpha_init_bias))
        else:
            self.deploy_correction_alpha = None

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
        base_logits = self.watch_classifier(embedding)
        out = {
            **watch_out,
            "logits": base_logits,
            "base_logits": base_logits,
            "watch_embedding": embedding,
        }
        if self.use_student_gated_correction:
            quality_col = self._quality_column(quality)
            deploy_input = torch.cat([embedding, quality_col], dim=1)
            deploy_delta = self.deploy_correction(deploy_input)
            deploy_gate = self.deploy_correction_gate(deploy_input)
            deploy_embedding = self.correction_norm(
                embedding + self._correction_scale_value() * deploy_gate * deploy_delta
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
        if return_aux and self.reliability_head is not None:
            reliability_logit = self.reliability_head(embedding)
            out["reliability_logit"] = reliability_logit
            out["reliability"] = torch.sigmoid(reliability_logit)
        if return_aux and self.watch_projector is not None:
            out["watch_proj"] = nn.functional.normalize(self.watch_projector(embedding), dim=1)
        if return_aux and self.rhythm_head is not None:
            out["rhythm_pred"] = self.rhythm_head(embedding)
        if return_aux and self.wavelet_predictor is not None:
            out["wavelet_pred"] = self.wavelet_predictor(embedding)
        return out
