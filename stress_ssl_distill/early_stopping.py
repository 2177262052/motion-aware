from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EarlyStopping:
    patience: int = 10
    mode: str = "max"
    min_delta: float = 0.0

    def __post_init__(self) -> None:
        if self.mode not in {"max", "min"}:
            raise ValueError(f"Unsupported mode: {self.mode}")
        self.best_score = float("-inf") if self.mode == "max" else float("inf")
        self.best_epoch = 0
        self.num_bad_epochs = 0

    def step(self, score: float, epoch: int) -> bool:
        improved = self._is_improvement(score)
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.num_bad_epochs = 0
            return True

        self.num_bad_epochs += 1
        return False

    def should_stop(self) -> bool:
        return self.num_bad_epochs >= self.patience

    def _is_improvement(self, score: float) -> bool:
        if self.mode == "max":
            return score > self.best_score + self.min_delta
        return score < self.best_score - self.min_delta

