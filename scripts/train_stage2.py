"""Stage 2 — train the learned fusion model.

A causal GRU emits per-step blend weights (alpha_v, alpha_w); the fused twist
is integrated through a differentiable unicycle model and trained to minimize
ATE vs fused_pose over whole runs. Consumes Stage 1's reliability.npy.

Training is over whole runs so long-horizon drift (the real objective) is
optimized directly. Window-based training and heavier dropout/weight-decay were
both tried and *worsened* held-out ATE, so the config is kept simple. Because a
single training run is a noisy point estimate on only ~8 test runs, several
seeds are trained and the result is reported as mean +/- std; the best-val
model is kept.

  .venv/bin/python scripts/train_stage2.py [--smoke] [--epochs N] [--seeds N]
"""
import argparse
import copy
import json
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vo import data
from vo.dataset import bc_runs, prepare_runs
from vo.kinematics import step_dt
from vo.models import FusionNet
from vo.torchkin import ate_torch, integrate_torch

OUT = "artifacts/stage2"
REL_PATH = "artifacts/stage1/reliability.npy"
VO_FUSE_V, VO_FUSE_W = 0.30, 0.60      # clamp VO twist to plausible robot range
N_FEAT = 9


def log(msg, fh):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    fh.write(line + "\n")
    fh.flush()


def build_run(d, rel_full, run, device):
    """Per-run tensors: features, dt, wheel, clamped VO, gt."""
    s, e = d["run_slices"][run]
    gidx = d["gindex"][s:e]
    feat10 = d["feat"][s:e]
    rel = rel_full[gidx]
    wheel = d["wheel"][run]
    vo = d["vo"][run]
    vo_valid = np.isfinite(vo[:, :1]).astype(np.float32)
    vo_fuse = np.where(np.isnan(vo), wheel, vo)
    vo_fuse = np.clip(vo_fuse, [-VO_FUSE_V, -VO_FUSE_W], [VO_FUSE_V, VO_FUSE_W])
    feat = np.concatenate([feat10[:, 2:6], rel, feat10[:, 6:8], vo_valid], axis=1)
    t = lambda x: torch.tensor(np.asarray(x, np.float32), device=device)
    return {
        "feat": t(feat), "dt": t(step_dt(d["ts"][run])),
        "wheel": t(wheel), "vo": t(vo_fuse),
        "gt": t(d["gt"][run][:, :2]), "start": t(d["gt"][run][0]),
    }


def fuse(model, run):
    alpha = model(run["feat"].unsqueeze(0))[0]
    twist = alpha * run["vo"] + (1 - alpha) * run["wheel"]
    return integrate_torch(run["dt"], twist, run["start"])


def full_ate(model, run):
    model.eval()
    with torch.no_grad():
        return float(ate_torch(fuse(model, run), run["gt"]))


def split_ate(model, R, runs):
    return float(np.mean([full_ate(model, R[r]) for r in runs]))


def train_seed(seed, R, by, device, epochs, fh):
    """Train one model; return (best-val model, best val ATE)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model = FusionNet(N_FEAT, dropout=0.0).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    best_val, best_state = float("inf"), None
    for ep in range(1, epochs + 1):
        model.train()
        order = by["train"][:]
        random.shuffle(order)
        for r in order:
            opt.zero_grad(set_to_none=True)
            ate_torch(fuse(model, R[r]), R[r]["gt"]).backward()
            opt.step()
        if ep % 20 == 0 or ep == epochs:
            va = split_ate(model, R, by["val"])
            if va < best_val:
                best_val, best_state = va, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    log(f"  seed {seed}: best val_ate={best_val:.3f}", fh)
    return model, best_val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    fh = open(f"{OUT}/log.txt", "w")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"device={device} smoke={a.smoke} seeds={a.seeds} epochs={a.epochs}", fh)

    if not os.path.exists(REL_PATH):
        log(f"ERROR: {REL_PATH} missing — run Stage 1 first.", fh)
        sys.exit(1)
    rel_full = np.load(REL_PATH)

    scalars = data.load_scalars()
    splits = data.load_splits()
    runs = bc_runs(scalars)
    if a.smoke:
        runs = (runs[:4] + [r for r in runs if splits[r] == "val"][:1]
                + [r for r in runs if splits[r] == "test"][:1])
        a.epochs, a.seeds = 20, 2
    d = prepare_runs(scalars, runs, with_images=False)
    R = {r: build_run(d, rel_full, r, device) for r in runs}
    by = {s: [r for r in runs if splits[r] == s] for s in ("train", "val", "test")}
    log(f"runs: train={len(by['train'])} val={len(by['val'])} test={len(by['test'])}", fh)

    # ---- train each seed, keep per-seed split ATEs and the best-val model ----
    per_seed, overall = [], (float("inf"), None)
    for seed in range(a.seeds):
        model, best_val = train_seed(seed, R, by, device, a.epochs, fh)
        ates = {s: split_ate(model, R, by[s]) for s in ("train", "val", "test")}
        per_seed.append(ates)
        log(f"  seed {seed}: train={ates['train']:.3f} val={ates['val']:.3f} "
            f"test={ates['test']:.3f}", fh)
        if best_val < overall[0]:
            overall = (best_val, copy.deepcopy(model.state_dict()))
    torch.save(overall[1], f"{OUT}/model.pt")

    # ---- report: mean +/- std across seeds vs baselines ----
    base = json.load(open("artifacts/baselines.json"))["runs"]
    log(f"\nFINAL — fused ATE across {a.seeds} seeds (metres):", fh)
    summary = {"per_seed": per_seed, "splits": {}}
    for s in ("train", "val", "test"):
        rr = by[s]
        vals = np.array([ps[s] for ps in per_seed])
        wheel = np.mean([base[str(r)]["baselines"]["wheel"]["ate"] for r in rr])
        const = np.mean([base[str(r)]["baselines"]["const_blend"]["ate"] for r in rr])
        summary["splits"][s] = {"wheel": float(wheel), "const_blend": float(const),
                                "fused_mean": float(vals.mean()),
                                "fused_std": float(vals.std())}
        log(f"  {s:5s}  wheel={wheel:.3f}  const={const:.3f}  "
            f"FUSED={vals.mean():.3f}+/-{vals.std():.3f}   "
            f"({(1-vals.mean()/wheel)*100:+.0f}% vs wheel)", fh)
    json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=2)
    log("done.", fh)
    fh.close()


if __name__ == "__main__":
    main()
