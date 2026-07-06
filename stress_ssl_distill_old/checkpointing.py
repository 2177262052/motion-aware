from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn


def load_compatible_state_dict(
    model: nn.Module,
    checkpoint_path: Path,
    device: str = "cpu",
    prefixes: Tuple[str, ...] | None = None,
) -> Dict[str, List[str]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_state = model.state_dict()

    filtered = {}
    skipped: List[str] = []
    for key, value in checkpoint.items():
        if prefixes is not None and not any(key.startswith(prefix) for prefix in prefixes):
            skipped.append(key)
            continue
        if key not in model_state:
            skipped.append(key)
            continue
        if tuple(value.shape) != tuple(model_state[key].shape):
            skipped.append(key)
            continue
        filtered[key] = value

    missing_before = [key for key in model_state.keys() if key not in filtered]
    result = model.load_state_dict(filtered, strict=False)
    missing_after = list(result.missing_keys)
    unexpected = list(result.unexpected_keys)

    print(
        "compatible checkpoint load:",
        f"loaded={len(filtered)}",
        f"skipped={len(skipped)}",
        f"missing={len(missing_after)}",
        f"unexpected={len(unexpected)}",
    )
    if skipped:
        print("skipped keys sample:", skipped[:8])

    return {
        "loaded": list(filtered.keys()),
        "skipped": skipped,
        "missing": missing_after,
        "unexpected": unexpected,
        "missing_before": missing_before,
    }

