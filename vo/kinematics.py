"""Unicycle dead-reckoning and twist extraction.

A twist is [v, w] (linear, angular velocity). twist[i] acts over step i -> i+1;
the last row is unused. Integration supports two heading conventions:

  euler    -- translate along the heading *after* the angular update
  midpoint -- translate along the mid-step heading (2nd-order, default)
"""
from __future__ import annotations

import numpy as np


def wrap(a):
    """Wrap angle(s) to (-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def step_dt(ts: np.ndarray, floor: float = 1e-3) -> np.ndarray:
    """Per-step durations; non-positive gaps are floored."""
    dt = np.diff(ts)
    dt[dt <= 0] = floor
    return dt


def integrate(dt: np.ndarray, twist: np.ndarray, start,
              model: str = "midpoint") -> np.ndarray:
    """Dead-reckon a pose trajectory (N, 3) from a twist sequence (N, 2)."""
    x0, y0, th0 = (float(s) for s in start)
    v, w = twist[:, 0], twist[:, 1]
    th = np.empty(len(v))
    th[0] = th0
    th[1:] = th0 + np.cumsum(w[:-1] * dt)
    if model == "euler":
        head = th[1:]
    elif model == "midpoint":
        head = th[:-1] + 0.5 * w[:-1] * dt
    else:
        raise ValueError(f"unknown model: {model}")
    x = x0 + np.concatenate([[0.0], np.cumsum(v[:-1] * np.cos(head) * dt)])
    y = y0 + np.concatenate([[0.0], np.cumsum(v[:-1] * np.sin(head) * dt)])
    return np.column_stack([x, y, wrap(th)])


def extract_twist(ts: np.ndarray, poses: np.ndarray) -> np.ndarray:
    """Recover the twist a pose trajectory followed (forward differences)."""
    dt = step_dt(ts)
    n = len(ts)
    tw = np.zeros((n, 2))
    tw[:-1, 0] = np.hypot(np.diff(poses[:, 0]), np.diff(poses[:, 1])) / dt
    tw[:-1, 1] = wrap(np.diff(poses[:, 2])) / dt
    return tw
