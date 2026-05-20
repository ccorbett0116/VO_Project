# VO Project — Mission Statement

## Goal

Train a model that learns to actively control a robot's camera (pan, tilt, on/off) to maximize visual odometry (VO) quality and improve localization accuracy.

The robot navigates figure-8 routes and must decide, in real time, how to orient its camera to get the best possible position estimates from visual odometry. The trained model should output optimal camera actions (pan angle, tilt angle, and whether the camera should be active) given the robot's current state.

## Background

A mobile robot drives repeatable figure-8 paths in an indoor environment ("window_room"). Ground-truth position is derived from fused pose estimates (x, y, theta). The robot has:

- **Wheel odometry** — provides commanded twist (linear velocity, angular velocity), always available but subject to drift
- **A pan/tilt camera** — captures 120x160 RGB images and runs visual odometry, producing twist estimates, feature counts, and covariance values
- **Visual odometry (VO)** — computes motion estimates from consecutive camera frames; quality depends heavily on camera orientation and scene content

The central research question: **How should the robot aim its camera (or turn it off) at each moment to minimize localization error over a trajectory?**

## Dataset Structure

All data lives in `VO_Research/comprehensive_dataset/training_data.hdf5` (~9.9 GB, 311,808 timestamped samples across 70 runs at ~10–42 Hz).

### HDF5 Datasets (all indexed by a shared sample axis):

| Dataset | Shape | Description |
|---|---|---|
| `run_id` | (311808,) int32 | Run identifier (1–70) |
| `phase` | (311808,) string | Experiment phase: A, B, or C |
| `condition` | (311808,) string | Camera condition for this run (see below) |
| `route` | (311808,) string | `figure8_LR` or `figure8_RL` (direction) |
| `environment` | (311808,) string | Always `window_room` |
| `timestamps` | (311808,) float64 | Unix timestamps |
| `fused_pose` | (311808, 3) float32 | Robot pose [x, y, theta] — best available position estimate |
| `wheel_twist` | (311808, 2) float32 | Wheel odometry [linear_vel, angular_vel], range ~[0, 0.2] m/s linear, [-0.35, 0.35] rad/s angular |
| `pan_tilt` | (311808, 2) float32 | Camera orientation [pan, tilt], each normalized to [0.0, 1.0] |
| `images` | (311808, 120, 160, 3) uint8 | RGB camera images |
| `vo_twist` | (311808, 2) float32 | VO-estimated twist [linear_vel, angular_vel]; NaN when camera is off |
| `vo_features` | (311808, 2) float32 | VO feature stats [inliers, total_matches]; NaN when camera is off |
| `vo_covariance` | (311808, 6) float32 | VO uncertainty (6 covariance values); NaN when camera off; high values (~20000) indicate poor estimates |

### Experiment Phases and Conditions

**Phase A — Baseline (camera off)** (runs 1–10, 58,641 samples)
- Condition: `off` — camera at default position (pan=0.5, tilt=0.5), VO inactive
- 5 repeats x 2 directions
- Establishes wheel-odometry-only baseline; all VO fields are NaN

**Phase B — Static camera angles** (runs 11–40, 127,186 samples)
- Camera held fixed at various pan/tilt positions throughout each run
- Conditions (format `static:pan:tilt`):
  - `static:0.5:0.5` — center (default)
  - `static:0.0:0.5` — full left
  - `static:1.0:0.5` — full right
  - `static:0.5:0.0` — full down
  - `static:0.5:1.0` — full up
- 3 repeats x 2 directions per condition
- Shows how fixed camera angle affects VO quality at different points in the route

**Phase C — Dynamic camera motion** (runs 41–70, 125,981 samples)
- Camera actively sweeps during navigation
- Conditions:
  - `slow_pan` — slow horizontal sweep (pan varies 0–1, tilt near 0)
  - `slow_tilt` — slow vertical sweep (tilt varies 0–1, pan varies)
  - `combined_slow` — simultaneous pan and tilt sweeping
- 5 repeats x 2 directions per condition
- Demonstrates effect of camera motion on VO (motion blur, changing features)

### Key Observations from the Data

- VO is **unavailable when camera is off** (Phase A) — all VO fields are NaN
- When camera is on, VO is valid ~99.7–100% of the time
- `vo_features` column 0 (inliers): 0–319, mean ~123; column 1 (total matches): 0–586, mean ~401
- High covariance values (up to ~20,000) indicate unreliable VO estimates
- The robot moves at fixed speeds (linear up to 0.2 m/s, angular up to 0.35 rad/s)
- Trajectories span roughly [-19, 14] meters in X and [-22, 8] meters in Y

## Manifest

`VO_Research/comprehensive_dataset/manifest.json` maps each of the 70 runs to its source experiment file, session date, phase, condition, route, and repeat number.

## What a Model Should Learn

Given the robot's current state (wheel odometry, current pose estimate, current camera orientation, recent images/VO history), the model should output:

1. **Optimal pan/tilt angles** — where to point the camera to maximize VO quality (more inliers, lower covariance) for the current location and heading
2. **Camera on/off decision** — whether VO at the current moment is worth the computational cost, or if wheel odometry alone is sufficient

Additionally, the model (or the fusion pipeline it feeds into) should learn **how much to trust camera-derived data versus wheel odometry**. The optimal weighting between VO and wheel odometry likely varies with context — VO covariance is one signal, but experimenting with different trust proportions (e.g., weighting VO more heavily when features are abundant and covariance is low, falling back toward wheel odometry when VO is unreliable) may significantly improve localization. This trust ratio could be a fixed learned parameter, a dynamic per-timestep output, or even part of the action space the model optimizes over.

The reward signal / training objective should relate to **localization accuracy** — minimizing the error between the estimated position (from fused wheel + VO odometry) and the true position over complete trajectories.
