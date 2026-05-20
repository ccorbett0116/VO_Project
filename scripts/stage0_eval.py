"""Stage 0 — evaluation harness.

  1. Picks the kinematic model by checking which one reproduces fused_pose best
     when fed the true (extracted) twist.
  2. Builds run-level train/val/test splits and writes splits.json.
  3. Computes the wheel / VO / constant-blend baselines and reports ATE by
     split and by condition; writes artifacts/baselines.json.

Run:  .venv/bin/python scripts/stage0_eval.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vo import data
from vo.baselines import run_baselines
from vo.kinematics import extract_twist, integrate, step_dt
from vo.metrics import trajectory_errors

ART = "artifacts"


def section(t):
    print(f"\n{'=' * 74}\n{t}\n{'=' * 74}")


def run_arrays(scalars, run):
    """Time-ordered (ts, gt, wheel, vo) for one run."""
    _, idx = data.run_mask(scalars, run)
    return (scalars["timestamps"][idx], scalars["fused_pose"][idx],
            scalars["wheel_twist"][idx], scalars["vo_twist"][idx])


def pick_model(scalars, runs):
    """Compare kinematic models by the residual of integrating the true twist."""
    section("KINEMATIC MODEL CHECK (integrate true twist vs fused_pose)")
    res = {"euler": [], "midpoint": []}
    for r in runs:
        ts, gt, _, _ = run_arrays(scalars, int(r))
        dt = step_dt(ts)
        tw = extract_twist(ts, gt)
        for m in res:
            res[m].append(trajectory_errors(integrate(dt, tw, gt[0], m), gt)["ate"])
    for m, v in res.items():
        v = np.array(v)
        print(f"  {m:9s}  mean residual ATE = {v.mean():.3f} m   "
              f"median = {np.median(v):.3f}   p90 = {np.percentile(v, 90):.3f}")
    best = min(res, key=lambda m: np.mean(res[m]))
    print(f"  -> using '{best}' model")
    return best


def main():
    os.makedirs(ART, exist_ok=True)
    scalars = data.load_scalars()
    runs = [int(r) for r in data.run_ids(scalars)]
    print(f"Loaded {len(scalars['run_id'])} samples across {len(runs)} runs.")

    model = pick_model(scalars, runs)

    section("RUN SPLITS (stratified by condition, whole runs held out)")
    split = data.make_splits(scalars, seed=0)
    data.save_splits(split)
    for s in ("train", "val", "test"):
        rs = sorted(r for r, v in split.items() if v == s)
        print(f"  {s:5s} ({len(rs):2d} runs): {rs}")
    print(f"  written to {data.SPLITS_PATH}")

    section("BASELINES PER RUN (ATE, metres)")
    print(f"  {'run':>3s} {'split':>5s} {'condition':>15s} "
          f"{'wheel':>7s} {'VO':>7s} {'const':>7s}  const(av,aw)")
    results = {}
    for r in runs:
        ts, gt, wheel, vo = run_arrays(scalars, r)
        b = run_baselines(ts, gt, wheel, vo, model)
        meta = data.run_meta(scalars, r)
        results[r] = {"split": split[r], "condition": meta["condition"],
                      "route": meta["route"], "baselines": b}
        ew = b["wheel"]["ate"]
        if "vo" in b:
            ev, cb = b["vo"]["ate"], b["const_blend"]
            extra = f"  ({cb['alpha_v']:.2f},{cb['alpha_w']:.2f})"
            line = f"{ew:7.2f} {ev:7.2f} {cb['ate']:7.2f}{extra}"
        else:
            line = f"{ew:7.2f} {'--':>7s} {'--':>7s}"
        print(f"  {r:>3d} {split[r]:>5s} {meta['condition']:>15s} {line}")

    section("SUMMARY BY SPLIT (mean ATE over runs)")
    print(f"  {'split':>6s} {'runs':>5s} {'wheel':>7s} {'VO':>7s} "
          f"{'const':>7s}  {'const gain':>11s}")
    for s in ("train", "val", "test"):
        rr = [v for v in results.values() if v["split"] == s]
        ew = np.mean([v["baselines"]["wheel"]["ate"] for v in rr])
        vv = [v["baselines"] for v in rr if "vo" in v["baselines"]]
        ev = np.mean([b["vo"]["ate"] for b in vv])
        ec = np.mean([b["const_blend"]["ate"] for b in vv])
        print(f"  {s:>6s} {len(rr):>5d} {ew:7.2f} {ev:7.2f} {ec:7.2f}  "
              f"{(1 - ec / ew) * 100:9.0f}%")

    section("SUMMARY BY CONDITION (mean ATE over runs)")
    print(f"  {'condition':>15s} {'runs':>5s} {'wheel':>7s} {'VO':>7s} "
          f"{'const':>7s}  {'const gain':>11s}")
    for c in sorted({v["condition"] for v in results.values()}):
        rr = [v for v in results.values() if v["condition"] == c]
        ew = np.mean([v["baselines"]["wheel"]["ate"] for v in rr])
        vv = [v["baselines"] for v in rr if "vo" in v["baselines"]]
        if vv:
            ev = np.mean([b["vo"]["ate"] for b in vv])
            ec = np.mean([b["const_blend"]["ate"] for b in vv])
            g = f"{(1 - ec / ew) * 100:9.0f}%"
            print(f"  {c:>15s} {len(rr):>5d} {ew:7.2f} {ev:7.2f} {ec:7.2f}  {g}")
        else:
            print(f"  {c:>15s} {len(rr):>5d} {ew:7.2f} {'--':>7s} {'--':>7s}")

    out = {"model": model,
           "runs": {str(r): {k: v[k] for k in ("split", "condition", "route")}
                    | {"baselines": v["baselines"]} for r, v in results.items()}}
    with open(f"{ART}/baselines.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwritten to {ART}/baselines.json")


if __name__ == "__main__":
    main()
