<h1 align="center">struct-x</h1>

<p align="center">Physics-aware event vision for robust non-contact structural vibration monitoring.</p>

Struct-X reconstructs the vibration waveform a collocated laser Doppler
vibrometer (LDV) would have recorded — from an event-camera stream — and holds
up when sensing conditions (illumination, contrast threshold, bias) drift.
Single rig only: XJTU-DV-Beam hammer-impact free decay (30 trials). A frozen
single-degree-of-freedom physics residual regularises a spatiotemporal
event-vision encoder. The headline metric is the robustness gap `Δ`, with a
pre-registered falsification rule: if physics doesn't beat trial-to-trial
noise, that negative result is reported, not hidden.

## What it does

- Reconstructs the LDV waveform and dominant frequency from events, per
  acquisition condition.
- Measures stability as `Δ = metric(shifted) − metric(matched)` across
  illumination/threshold shifts — and tries to minimise it.
- Works with 30 independent trials: splits are trial-level, never window-level.
- Ablates what helps: SDOF physics residual vs temporal smoothness vs both.

Out of scope: damage classification (the dataset carries no damage labels).
No claims beyond this rig.

## Quickstart

All commands run via `make` from the project root. Requires `uv`.

```bash
make setup       # install dependencies
make download    # fetch XJTU-DV-Beam.zip, verify checksum, print inventory
make preprocess  # build voxel store + window index
make smoke       # shape/memory smoke test
make train       # leave-one-condition-out training, all ablations
make eval        # test-trial metrics, Δ estimators, falsification verdict
make test        # unit tests
make style       # format + lint
make clean       # remove build artifacts
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development workflow.

## Layout

```
src/                  # importable library (every module has a test in tests/)
  data_loader.py      # trial inventory, LDV units, windowing, LOCO splits
  voxel.py            # event filtering, ROI crop, polarity-split voxels
  physics.py          # SDOF residual, fp32 finite differences
  mck_fit.py          # log-decrement fit (m=1) on one train trial
  models/             # TimeSformer port, 3D-ResNet control, waveform/frequency heads
  train.py            # LOCO folds, ablation grid, gradient-ratio controller
  evaluate.py         # metrics, Δ/se estimators, falsification rule
configs/base.yaml     # bins, loss weights, learning rate, batch sizes, seeds
scripts/              # thin CLIs over src (download, preprocess, eval, Kaggle helpers)
tests/                # mirrors src, one module per module
kernel/               # Kaggle entry points (smoke, train, timing)
data/                 # local dataset staging (never committed)
```

## Key rules

- Trial-level splits only; physics fit on one train-condition trial.
- Physics math runs in fp32; `black` + `ruff`; snake_case.
- The ported TimeSformer is CC-NC 4.0, research-only — deployment paths use
  the ResNet control.
- Datasets and kernels are public; the dataset stays CC-BY-4.0.

## Data

XJTU-DV-Beam only (`XJTU-DV-Beam.zip`, ~1.6 GB, Zenodo record 17447083).
Rotor/Pump subsets are out of scope. Reference loaders are bundled in the
archive under `Code/`.
