from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class MIRNet(nn.Module):
    """Minimal MIRNet stub."""

    def __init__(self) -> None:
        super().__init__()
        # TODO: build actual MIRNet layers
        self.layer: nn.Module = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return input tensor until model is implemented."""
        return self.layer(x)

    def load_pretrained(
        self, weights_path: Path
    ) -> None:  # pragma: no cover - placeholder
        """Load pretrained weights (not yet implemented)."""
        raise NotImplementedError("Pretrained weights loading not implemented")
