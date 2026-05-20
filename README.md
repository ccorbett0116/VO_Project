# VO Fusion Model

A learned fusion model that combines **wheel odometry** with **visual odometry (VO)** to localize a mobile robot. At each timestep, a GRU emits per-step blend weights `(alpha_v, alpha_w)` in `[0, 1]`, and the fused twist is

```
fused_twist = alpha * vo_twist + (1 - alpha) * wheel_twist
```

The fused twist is integrated through a differentiable unicycle model and the network is trained to minimize Absolute Trajectory Error (ATE) against ground-truth `fused_pose` over whole runs.

See [`MISSION.md`](MISSION.md) for the broader research goal (active camera control); this repo contains only the fusion-model pipeline.

## Results

Mean ATE in metres across 5 seeds, vs. wheel-only and a GT-tuned constant blend (which peeks at the answer and is a reference *ceiling* for non-learned fusion, not a deployable estimator):

| Split | Wheel | Const blend | **Fused (ours)** | Gain vs wheel |
|------:|:----:|:-----------:|:----------------:|:-------------:|
| train | 1.95 | 1.01 | **0.92 ± 0.10** | +53% |
| val   | 1.86 | 1.15 | **1.38 ± 0.06** | +25% |
| test  | 2.24 | 1.19 | **1.71 ± 0.07** | +24% |

The learned fusion beats wheel-only on every split and approaches the GT-tuned constant-blend reference, which it does not see during training.

## Pipeline

Three stages, each producing artifacts the next consumes:

**Stage 0 — `scripts/stage0_eval.py`**
Picks the kinematic integrator (`midpoint` vs `euler`), writes `splits.json` (stratified by camera condition, whole runs held out), and computes the wheel / VO / constant-blend baselines into `artifacts/baselines.json`.

**Stage 1 — `scripts/train_stage1.py`** (`vo.models.ReliabilityNet`)
A small CNN over the 120×160 camera image, concatenated with scalar features, predicts `log1p(|VO twist error|)` per timestep. After training, runs inference over every sample and writes `artifacts/stage1/reliability.npy` (one prediction per sample, full-length).

**Stage 2 — `scripts/train_stage2.py`** (`vo.models.FusionNet`)
A causal GRU over per-step features (wheel twist, VO twist, Stage 1 reliability, feature counts, VO-validity flag) emits the blend weights. Training is over whole runs so long-horizon drift is optimized directly; window-based training and heavier regularization were both tried and worsened held-out ATE. Several seeds are trained, the best-validation model is kept, and results are reported as mean ± std across seeds.

## Reproducing

Requires the dataset at `VO_Research/comprehensive_dataset/training_data.hdf5` (not in repo; see `MISSION.md` for schema).

```bash
.venv/bin/python scripts/stage0_eval.py        # baselines + splits
.venv/bin/python scripts/run_overnight.py      # Stage 1 then Stage 2
# or run the stages individually:
.venv/bin/python scripts/train_stage1.py
.venv/bin/python scripts/train_stage2.py
```

Add `--smoke` to any training script for a quick end-to-end sanity run.

The trained fusion model lands at `artifacts/stage2/model.pt`, with per-seed and per-split metrics in `artifacts/stage2/summary.json`.

## Repo layout

```
vo/
  data.py          # HDF5 loading, run splits
  dataset.py       # feature building, prepare_runs, Stage1Dataset
  kinematics.py    # numpy unicycle integration + twist extraction
  torchkin.py      # differentiable counterpart (used by Stage 2 loss)
  metrics.py       # ATE / RMSE / loop-closure
  baselines.py     # wheel / VO / constant-blend (Stage 0)
  models.py        # ReliabilityNet (Stage 1), FusionNet (Stage 2)
scripts/
  stage0_eval.py
  train_stage1.py
  train_stage2.py
  run_overnight.py
splits.json        # locked train/val/test assignment
```
