from __future__ import annotations

import torch
import torch.nn as nn

from . import galaxy_models


class ScaledMotionFiLM(nn.Module):
    """Historical learnable-strength motion FiLM used by scale_logit checkpoints."""

    def __init__(
        self,
        cond_dim: int,
        target_dim: int,
        *args: object,
        scale_logit_init: float = -2.0,
        **kwargs: object,
    ) -> None:
        super().__init__()
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
        self.scale_logit = nn.Parameter(torch.tensor(float(scale_logit_init)))

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        gamma = self.to_gamma(condition).unsqueeze(-1)
        beta = self.to_beta(condition).unsqueeze(-1)
        scale = torch.sigmoid(self.scale_logit)
        return x * (1.0 + scale * torch.tanh(gamma)) + scale * beta


def install_scaled_motion_film() -> None:
    galaxy_models.MotionFiLM = ScaledMotionFiLM


install_scaled_motion_film()

from .train_galaxy_watch import main  # noqa: E402


if __name__ == "__main__":
    print("scaled_motion_compat=on scale_logit_init=-2.0")
    main()
