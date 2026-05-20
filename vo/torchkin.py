"""Differentiable unicycle integration (Torch), used to train fusion by ATE.

Mirrors the midpoint model in `vo.kinematics` so offline metrics and the
training objective agree.
"""
from __future__ import annotations

import torch


def integrate_torch(dt: torch.Tensor, twist: torch.Tensor,
                    start: torch.Tensor) -> torch.Tensor:
    """Dead-reckon (x, y) from a twist sequence.

    dt:    (T-1,)  per-step durations
    twist: (T, 2)  [v, w]; row i acts over step i -> i+1, last row unused
    start: (3,)    initial [x, y, theta]
    returns (T, 2) positions.
    """
    v, w = twist[:, 0], twist[:, 1]
    z = torch.zeros(1, dtype=v.dtype, device=v.device)
    th = start[2] + torch.cat([z, torch.cumsum(w[:-1] * dt, 0)])
    head = th[:-1] + 0.5 * w[:-1] * dt
    x = start[0] + torch.cat([z, torch.cumsum(v[:-1] * torch.cos(head) * dt, 0)])
    y = start[1] + torch.cat([z, torch.cumsum(v[:-1] * torch.sin(head) * dt, 0)])
    return torch.stack([x, y], dim=1)


def ate_torch(est_xy: torch.Tensor, gt_xy: torch.Tensor) -> torch.Tensor:
    """Mean L2 position error (ATE)."""
    return torch.mean(torch.linalg.vector_norm(est_xy - gt_xy, dim=1))
