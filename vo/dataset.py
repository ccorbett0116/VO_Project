"""Feature building and dataset assembly for Stages 1 and 2.

`prepare_runs` flattens a set of runs into contiguous arrays (keeping run
boundaries) and optionally pulls the matching images into RAM. The scalar
arrays are tiny; images for all 60 VO runs total ~15 GB.
"""
from __future__ import annotations

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from . import data
from .kinematics import extract_twist

# scalar-feature scaling constants (robot twist limits, feature-count maxima)
V_MAX, W_MAX = 0.20, 0.35
VO_V_CLIP, VO_W_CLIP = 1.0, 1.5
INLIER_MAX, MATCH_MAX = 320.0, 586.0
N_SCALAR = 10


def bc_runs(scalars: dict) -> list[int]:
    """Runs that have VO (phases B and C; phase A is camera-off)."""
    return [r for r in (int(x) for x in data.run_ids(scalars))
            if data.run_meta(scalars, r)["phase"] in ("B", "C")]


def sample_features(scalars: dict, idx: np.ndarray) -> np.ndarray:
    """Per-sample scalar feature matrix (N, N_SCALAR), normalized, NaN-safe."""
    pt = scalars["pan_tilt"][idx]
    wh = scalars["wheel_twist"][idx]
    vo = scalars["vo_twist"][idx]
    fe = scalars["vo_features"][idx]
    vo_fb = np.where(np.isnan(vo), wh, vo)        # fallback so features stay finite
    f = np.stack([
        pt[:, 0], pt[:, 1],
        wh[:, 0] / V_MAX, wh[:, 1] / W_MAX,
        np.clip(vo_fb[:, 0], -VO_V_CLIP, VO_V_CLIP),
        np.clip(vo_fb[:, 1], -VO_W_CLIP, VO_W_CLIP) / VO_W_CLIP,
        np.clip(fe[:, 0], 0, INLIER_MAX) / INLIER_MAX,
        np.clip(fe[:, 1], 0, MATCH_MAX) / MATCH_MAX,
        np.clip(np.abs(vo_fb[:, 0] - wh[:, 0]), 0, 1.0),
        np.clip(np.abs(vo_fb[:, 1] - wh[:, 1]), 0, W_MAX) / W_MAX,
    ], axis=1)
    return np.nan_to_num(f).astype(np.float32)


def prepare_runs(scalars: dict, runs: list[int], with_images: bool = False,
                 h5path: str = data.H5_DEFAULT) -> dict:
    """Flatten runs into aligned arrays. Returns a dict with:
      feat (N,N_SCALAR), target (N,2) log1p|VO err| (NaN if VO missing),
      gindex (N,) global sample indices, run_slices {run:(start,stop)},
      runs, per-run gt/ts/wheel/vo, and images (N,120,160,3) uint8 or None.
    """
    runs = sorted(int(r) for r in runs)
    feat, target, gindex, run_of = [], [], [], []
    run_slices, gt, ts, wheel, vo = {}, {}, {}, {}, {}
    blocks, cur = [], 0
    h = h5py.File(h5path, "r") if with_images else None
    for r in runs:
        _, idx = data.run_mask(scalars, r)
        n = len(idx)
        g = scalars["fused_pose"][idx].astype(np.float64)
        t = scalars["timestamps"][idx].astype(np.float64)
        wh = scalars["wheel_twist"][idx].astype(np.float64)
        vt = scalars["vo_twist"][idx].astype(np.float64)
        feat.append(sample_features(scalars, idx))
        target.append(np.log1p(np.abs(vt - extract_twist(t, g))).astype(np.float32))
        gindex.append(idx)
        run_of.append(np.full(n, r))
        run_slices[r] = (cur, cur + n)
        gt[r], ts[r], wheel[r], vo[r] = g, t, wh, vt
        cur += n
        if with_images:
            lo, hi = idx[0], idx[-1] + 1
            block = (h["images"][lo:hi] if np.array_equal(idx, np.arange(lo, hi))
                     else h["images"][np.sort(idx)])
            blocks.append(block)
    if h is not None:
        h.close()
    images = np.concatenate(blocks, axis=0) if with_images else None
    return {
        "feat": np.concatenate(feat),
        "target": np.concatenate(target),
        "gindex": np.concatenate(gindex),
        "run_of": np.concatenate(run_of),
        "run_slices": run_slices, "runs": runs,
        "gt": gt, "ts": ts, "wheel": wheel, "vo": vo,
        "images": images,
    }


class Stage1Dataset(Dataset):
    """Image + scalar features -> VO-error target, restricted to given rows."""

    def __init__(self, feat, target, images, rows):
        self.feat, self.target, self.images = feat, target, images
        self.rows = np.asarray(rows)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        j = self.rows[i]
        if self.images is not None:
            img = (self.images[j].astype(np.float32) / 255.0 - 0.5) / 0.5
            img = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))
        else:
            img = torch.zeros(3, 120, 160)
        return img, torch.from_numpy(self.feat[j]), torch.from_numpy(self.target[j])
