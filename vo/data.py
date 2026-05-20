"""Dataset access: load the non-image HDF5 arrays into memory and split runs.

The scalar datasets total only ~25 MB, so they are loaded fully. Images stay on
disk and are streamed later (Stage 1) via h5py.
"""
from __future__ import annotations

import json
from collections import defaultdict

import h5py
import numpy as np

H5_DEFAULT = "VO_Research/comprehensive_dataset/training_data.hdf5"
SPLITS_PATH = "splits.json"

_NUM_KEYS = ["run_id", "timestamps", "fused_pose", "wheel_twist",
             "vo_twist", "vo_features", "vo_covariance", "pan_tilt"]
_STR_KEYS = ["phase", "condition", "route", "environment"]


def load_scalars(h5path: str = H5_DEFAULT) -> dict:
    """Load every non-image dataset into a dict of numpy arrays."""
    out = {}
    with h5py.File(h5path, "r") as f:
        for k in _NUM_KEYS:
            out[k] = f[k][:]
        for k in _STR_KEYS:
            a = f[k][:]
            out[k] = np.array([x.decode() if isinstance(x, bytes) else x
                               for x in a])
    return out


def run_ids(scalars: dict) -> np.ndarray:
    return np.unique(scalars["run_id"])


def run_mask(scalars: dict, run: int) -> np.ndarray:
    """Boolean mask for one run, ordered by timestamp."""
    m = scalars["run_id"] == run
    idx = np.where(m)[0]
    order = idx[np.argsort(scalars["timestamps"][idx])]
    full = np.zeros(len(m), bool)
    full[order] = True
    return full, order


def run_meta(scalars: dict, run: int) -> dict:
    i = np.where(scalars["run_id"] == run)[0][0]
    return {k: scalars[k][i] for k in ("phase", "condition", "route")}


def make_splits(scalars: dict, seed: int = 0,
                val_per_cond: int = 1, test_per_cond: int = 1) -> dict:
    """Assign whole runs to train/val/test, stratified by condition.

    Holding out whole runs prevents leakage between consecutive samples.
    """
    rng = np.random.default_rng(seed)
    groups = defaultdict(list)
    for r in run_ids(scalars):
        groups[run_meta(scalars, int(r))["condition"]].append(int(r))

    split = {}
    for cond, runs in sorted(groups.items()):
        runs = sorted(runs)
        rng.shuffle(runs)
        for r in runs[:test_per_cond]:
            split[r] = "test"
        for r in runs[test_per_cond:test_per_cond + val_per_cond]:
            split[r] = "val"
        for r in runs[test_per_cond + val_per_cond:]:
            split[r] = "train"
    return dict(sorted(split.items()))


def save_splits(split: dict, path: str = SPLITS_PATH) -> None:
    with open(path, "w") as f:
        json.dump({str(k): v for k, v in split.items()}, f, indent=2)


def load_splits(path: str = SPLITS_PATH) -> dict:
    with open(path) as f:
        return {int(k): v for k, v in json.load(f).items()}
