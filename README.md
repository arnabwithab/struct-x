<h1 align="center">struct-x</h1>

<p align="center">
  <a href="LICENSE.md"><img src="https://img.shields.io/badge/license-MIT-green" alt="License: MIT" /></a>
  <img src="https://img.shields.io/badge/python-3.10+-blue" alt="Python 3.10+" />
  <a href="https://www.kaggle.com/code/eros483/struct-x-train"><img src="https://img.shields.io/badge/NVIDIA_T4-76B900?logo=nvidia&logoColor=white" alt="Runs on NVIDIA T4" /></a>
  <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code_style-black-000000" alt="Code style: black" /></a>
</p>

<p align="center">Event cameras watch vibration as a flicker of pixels. Struct-X turns
that flicker into the waveform a laser vibrometer would have recorded — and
stays accurate when lighting or camera settings change.</p>

## What it does

- Rebuilds the vibration waveform and its dominant frequency from an event
  stream, matching what a collocated laser vibrometer measures.
- Holds that accuracy steady across illumination and threshold shifts instead
  of degrading silently.
- Learns from only 30 hammer-impact trials, split trial-wise so nothing leaks.
- Tests whether a physics residual is what actually helps, with a
  pre-registered rule that also accepts "no measurable effect" as the answer.

## Quickstart

```bash
git clone https://github.com/arnabwithab/struct-x.git
cd struct-x
make setup       # install uv (if needed) + dependencies
make download    # fetch the vibration dataset, verify checksum
make preprocess  # build voxel store + window index
make train       # leave-one-condition-out training, all ablations
make eval        # test-trial metrics and robustness verdict
```

## Data

Vibration data from the XJTU-DV open-source dataset (Xi'an Jiaotong
University), via Zenodo record 17447083.
