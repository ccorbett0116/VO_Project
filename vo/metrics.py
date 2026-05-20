"""Trajectory-error metrics for localization evaluation."""
from __future__ import annotations

import numpy as np


def trajectory_errors(est: np.ndarray, gt: np.ndarray) -> dict:
    """Compare an estimated trajectory to ground truth (both (N, >=2)).

    ate   -- mean L2 position error (primary metric)
    rmse  -- root-mean-square position error
    final -- error at the last sample
    maxe  -- worst-case error
    loop  -- estimate's own loop-closure: |end - start| (route returns home)
    """
    d = np.linalg.norm(est[:, :2] - gt[:, :2], axis=1)
    return {
        "ate": float(d.mean()),
        "rmse": float(np.sqrt(np.mean(d ** 2))),
        "final": float(d[-1]),
        "maxe": float(d.max()),
        "loop": float(np.linalg.norm(est[-1, :2] - est[0, :2])),
    }
