"""Localization baselines: pure wheel, pure VO, and a GT-tuned constant blend.

These are the references every learned model in later stages must beat. The
constant blend peeks at ground truth to pick its two gains, so it is a
*reference ceiling for non-learned fusion*, not a deployable estimator.
"""
from __future__ import annotations

import numpy as np

from .kinematics import integrate, step_dt
from .metrics import trajectory_errors


def vo_with_fallback(wheel: np.ndarray, vo: np.ndarray) -> np.ndarray:
    """Replace NaN VO twist (camera off / VO failure) with wheel twist."""
    return np.where(np.isnan(vo), wheel, vo)


def best_constant_blend(dt, wheel, vo, start, gt, model="midpoint", grid=21):
    """Grid-search a fixed (alpha_v, alpha_w): twist = a*vo + (1-a)*wheel."""
    g = np.linspace(0.0, 1.0, grid)
    best = {"ate": np.inf, "alpha_v": 0.0, "alpha_w": 0.0}
    for av in g:
        v = av * vo[:, 0] + (1 - av) * wheel[:, 0]
        for aw in g:
            w = aw * vo[:, 1] + (1 - aw) * wheel[:, 1]
            est = integrate(dt, np.column_stack([v, w]), start, model)
            ate = trajectory_errors(est, gt)["ate"]
            if ate < best["ate"]:
                best = {"ate": float(ate), "alpha_v": float(av),
                        "alpha_w": float(aw)}
    return best


def run_baselines(ts, gt, wheel, vo, model="midpoint") -> dict:
    """Compute every baseline for a single run. `vo` may be all-NaN (camera off)."""
    dt = step_dt(ts)
    start = gt[0]
    out = {"wheel": trajectory_errors(integrate(dt, wheel, start, model), gt)}

    has_vo = not np.isnan(vo[:, 0]).all()
    if has_vo:
        vo_fb = vo_with_fallback(wheel, vo)
        out["vo"] = trajectory_errors(integrate(dt, vo_fb, start, model), gt)
        out["const_blend"] = best_constant_blend(dt, wheel, vo_fb, start, gt, model)
    return out
