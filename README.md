<h1 align="center">struct-x</h1>

<p align="center">
  <a href="LICENSE.md"><img src="https://img.shields.io/badge/license-MIT-green" alt="License: MIT" /></a>
  <img src="https://img.shields.io/badge/python-3.10+-blue" alt="Python 3.10+" />
  <a href="https://www.kaggle.com/code/eros483/struct-x-train"><img src="https://img.shields.io/badge/NVIDIA_T4-76B900?logo=nvidia&logoColor=white" alt="Runs on NVIDIA T4" /></a>
  <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code_style-black-000000" alt="Code style: black" /></a>
</p>

<p align="center">Event cameras watch vibration as a flicker of pixels. Struct-X turns
that flicker into the waveform a collocated laser Doppler vibrometer (LDV) would
have recorded — and measures whether it stays accurate when lighting or camera
settings drift.</p>

## Abstract

Reconstruct LDV displacement waveforms and dominant frequency from event streams
on 30 hammer-impact trials (XJTU-DV-Beam), under a pre-registered robustness
protocol: trial-level leave-one-condition-out splits, `Δ = M(shifted) − M(matched)`,
and an O5 falsification rule that accepts "no measurable effect" as the answer.
Full method and analysis live in the report, not here.

## Key result

**O5: non-identifiable on all 8 backbone×fold settings** — the SDOF physics
residual does not clear trial-to-trial noise anywhere, and costs +0.1 to +0.5
matched RMSE on most folds. ResNet control (test mean ± std, `results/final_verdict.json`):

| Fold | data matched | data Δ | both matched | both Δ |
|---|---|---|---|---|
| On_set1 | 1.98±3.10 | −1.65 | 2.39±2.85 | −1.61 |
| Off_set1 | 1.95±2.80 | −1.49 | 2.39±2.55 | −1.33 |
| Off_set2 | 2.43±3.43 | −1.79 | 2.53±3.21 | −1.58 |
| Off_set3 | 0.37±0.07 | −0.07 | 0.63±0.10 | −0.02 |
| On_set2 | 0.52±0.10 | +2.23 | 0.99±0.21 | +1.98 |
| On_set3 | 0.47±0.10 | −0.23 | 0.37±0.07 | +0.20 |

RidgeCV on pooled voxels still beats deep models on RMSE (0.29–0.66, `results/ridge_verdict.json`)
with Pearson r ≈ 0.2 on both sides — shape is unreconstructed everywhere (deep r 0.03–0.19 matched).

## Quickstart

```bash
git clone https://github.com/arnabwithab/struct-x.git
cd struct-x
make setup       # install uv (if needed) + dependencies
make download    # fetch XJTU-DV-Beam.zip, verify md5, print inventory
make preprocess  # build voxel store + window index (trial-level, no leakage)
make train       # LOCO training, all O4 ablations
make eval        # test-trial metrics, Δ/se estimators, O5 verdict
make test        # unit tests (required before any claim)
```

Reproducibility: frozen tag `frozen-2026-11-04`, seeds + trial IDs + `m,c,k` + checkpoint SHA logged per run, `make mini-e2e` for a CPU smoke run.

## License fence

Repo code is MIT, but the ported TimeSformer is **CC-NC 4.0, research-only**.
Any deployment path uses the ResNet control or Ridge baseline, never ported
TimeSformer weights.

## Links

- Project site: https://arnabwithab.github.io/struct-x/
- Full report: [docs/results.md](docs/results.md)
- Design doc (sole source of truth): [docs/design.md](docs/design.md)
- Data: XJTU-DV-Beam, Zenodo record [17447083](https://zenodo.org/records/17447083) (CC-BY-4.0)
