<h1 align="center">struct-x</h1>

<p align="center">Physics-aware event vision for robust non-contact structural vibration monitoring.</p>

Struct-X learns `f_θ: E_s → q̂(t)`: from an event-camera stream to the vibration
waveform a collocated laser Doppler vibrometer (LDV) would have recorded — robust
to sensing-condition shifts (illumination, threshold/bias). Single rig only:
XJTU-DV-Beam hammer-impact free decay (30 trials). A frozen SDOF physics residual
regularises a spatiotemporal event-vision encoder. Headline metric: the robustness
gap `Δ`, with a pre-registered falsification rule (O5).

Sole spec: [`docs/design.md`](docs/design.md). It takes precedence over everything.
If it is ambiguous, flag it rather than improvising.

## What it does

- **O1 (accuracy):** reconstruct LDV waveform `q̂(t)` + dominant frequency `f̂_n`
  from events under the same acquisition condition.
- **O2 (stability):** quantify `Δ = M(shifted) − M(matched)` across illumination /
  threshold shifts and minimise it.
- **O3 (scarce data):** 30 independent trials only; trial-level splits, never
  window-level.
- **O4 (mechanism isolation):** ablate SDOF residual vs temporal smoothness vs both.
- **O5 (falsification):** if physics reduces `Δ` by less than trial-to-trial std
  `σ_Δ`, report the physics term as non-identifiable. A negative result counts.

Out of scope: damage classification (XJTU-DV-Beam has no damage labels — never
claimed). No claim beyond this rig; generalisation is future work.

## Quickstart

All commands run via `make` from the project root. Requires `uv`.

```bash
make setup       # installs all dependencies (uv sync)
make download    # fetches XJTU-DV-Beam.zip, verifies md5, prints inventory
make preprocess  # builds voxels + window index (trial-level, no leakage)
make smoke       # shape/memory smoke test (encoder -> heads, logs max_memory)
make train       # LOCO training, all O4 ablations
make eval        # TEST-trial metrics, Δ/σ estimators, O5 verdict
make test        # runs all unit tests
make style       # formats + lints all code
make clean       # removes build artifacts, caches, __pycache__ etc.
```

## Layout

```
src/
  data_loader.py, voxel.py      # trial inventory, LDV units, voxels, LOCO splits
  physics.py, mck_fit.py        # fp32 SDOF residual (fp16 forbidden), m=1 fit
  models/timesformer.py         # ported divided-attention encoder (CC-NC 4.0, research-only)
  models/resnet3d.py            # 3D-ResNet18 control (commercial-safe path)
  models/heads.py               # waveform (T=1000) + frequency heads
  train.py                      # LOCO folds, O4 grid, gradient-ratio controller
  evaluate.py                   # metrics, Δ/se estimators, O5 rule, latency
configs/base.yaml               # B, λp, λs, lr, micro/eff batch, splits, seeds
scripts/                        # download.sh, preprocess.py, run_folds.sh
tests/                          # mirrors src structure, one test module per module
kernel/, data/  # Kaggle kernel/dataset dirs (both PUBLIC)
```

## Rules that matter

- **TDD first.** No physics/math code ships without a unit test (collapse guard,
  finite differences, `m,c,k` fit, Δ/σ estimators).
- **Trial-level splits only.** Window-level splits are forbidden. `m,c,k` fit on
  one train-condition trial; tune on val-Δ only, never matched RMSE.
- **Physics in fp32.** Never compute `L_phys` or finite differences in fp16.
- **Conventions:** `uv` (never `pip`), `black` + `ruff`, snake_case everywhere.
- **License fence:** the ported TimeSformer is CC-NC 4.0, research-only. Any
  deployment path uses the ResNet control, never ported TimeSformer weights.
- **Kaggle:** datasets and kernels are PUBLIC; dataset license is CC-BY-4.0;
  auth (`kaggle.json`) is never committed.
- **After every feature:** update `docs/features.json`, commit
  (conventional commits), push to git.

## Data

Zenodo 17447083, `XJTU-DV-Beam.zip` only (1.7 GB). Rotor/Pump subsets are out of
scope — never download. Loader mirror: https://gr.xjtu.edu.cn/web/lixiang/xjtu-dv.

True window budget is ~90–270 windows (1.0 s / 0.5 s stride over 2–5 s trials),
not 2–3k. `L_phys` is amplitude-blind by design; `L_LDV` anchors amplitude.
