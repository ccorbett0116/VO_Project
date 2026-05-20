"""Models for the fusion pipeline.

ReliabilityNet (Stage 1) -- predicts how wrong the VO twist is right now.
FusionNet      (Stage 2) -- emits per-step blend weights for wheel vs VO.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ReliabilityNet(nn.Module):
    """Predicts log1p(|VO twist error|) for [linear, angular] from an image
    (120x160 RGB) and scalar features."""

    def __init__(self, n_scalar: int, use_image: bool = True):
        super().__init__()
        self.use_image = use_image
        img_dim = 0
        if use_image:
            self.cnn = nn.Sequential(
                nn.Conv2d(3, 16, 5, 2, 2), nn.BatchNorm2d(16), nn.ReLU(),
                nn.Conv2d(16, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
                nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.Conv2d(64, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            )
            img_dim = 64
        self.head = nn.Sequential(
            nn.Linear(img_dim + n_scalar, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, img: torch.Tensor, scalar: torch.Tensor) -> torch.Tensor:
        feat = torch.cat([self.cnn(img), scalar], 1) if self.use_image else scalar
        return self.head(feat)


class FusionNet(nn.Module):
    """Causal GRU over per-step features -> blend weights (alpha_v, alpha_w)
    in [0, 1].  fused_twist = alpha * vo + (1 - alpha) * wheel."""

    def __init__(self, n_feat: int, hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(n_feat, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.gru(x)              # x: (B, T, F)
        return torch.sigmoid(self.out(self.drop(h)))
