# AGENTS.md

## Project Overview

Struct-X learns `f_θ: E_s → q̂(t)` — reconstructing the vibration waveform a
collocated laser Doppler vibrometer (LDV) would have recorded, from an
event-camera stream, robust to sensing-condition shifts (illumination,
threshold/bias). For SHM teams who need low-cost non-contact vibration sensing.
Single rig only: XJTU-DV-Beam hammer-impact free decay (30 trials). A frozen
SDOF physics residual regularises a spatiotemporal event-vision encoder; the
headline metric is the robustness gap `Δ`, with a pre-registered falsification
rule (O5). Sole spec: `docs/design.md` — it takes precedence over everything.

## Development Philosophy

- Design doc rules: `docs/design.md` is the sole source of truth. Never
  contradict it; if it is ambiguous, flag it rather than improvising.
- TDD first: write the test, then the implementation. Never skip.
- Tests mirror the structure of the module they test.
- No physics/math code ships without a unit test (collapse guard, finite
  differences, `m,c,k` fit, Δ/σ estimators).
- Explicit over clever — readable code beats smart code.
- Forbidden shortcuts are listed in design §11 — never take one (window-level
  splits, tuning λ on matched RMSE, `m,c,k` on test trials, training the
  Transformer from scratch, damage-classification claims, hardcoded sensor
  resolution).
- If it isn't runnable via `make`, it isn't done.
- After every feature run finishes, update `docs/features.json` and push to git.

## Tech Stack

- Language: Python (3.10+)
- ML: PyTorch, numpy, scipy (FFT, log-decrement), pyyaml
- Event IO: dv-processing / aedat (pin version after first successful load)
- Configs: YAML in `configs/`
- Package Manager: **uv** — all deps via `uv add` / `uv sync`. Never `pip` directly.
- Build/Task Runner: **Make** — root `Makefile` is mandatory, single entry point.
- Compute: Kaggle T4 notebooks via Kaggle CLI (datasets + kernels PUBLIC) or
  local RTX 4060. Binding constraints in design §10/§10b.

## Key Commands

All commands MUST run via `make <target>` from the project root.

```bash
make setup                       # installs all dependencies (uv sync)
make download                    # fetches XJTU-DV-Beam.zip, verifies md5, prints inventory
make preprocess                  # builds voxels + window index (trial-level, no leakage)
make smoke                       # shape/memory smoke test (encoder → heads, logs max_memory)
make train                       # LOCO training, all O4 ablations
make eval                        # TEST-trial metrics, Δ/σ estimators, O5 verdict
make test                        # runs all unit tests
make style                       # formats + lints all code
make clean                       # removes build artifacts, caches, __pycache__ etc.
```

## Directory Structure

```
struct-x/
├── src/
│   ├── data_loader.py           # trial inventory, LDV/event loading, windowing
│   ├── voxel.py                 # filters → ROI crop → polarity-split voxels
│   ├── physics.py               # fp32 finite differences + normalised L_phys
│   ├── mck_fit.py               # log-decrement fit (m=1) on one train trial
│   ├── models/
│   │   ├── timesformer.py       # ported divided-attention encoder + adapter
│   │   ├── resnet3d.py          # 3D-ResNet control (commercial-safe path)
│   │   └── heads.py             # waveform head + frequency head
│   ├── train.py                 # LOCO folds, O4 grid, gradient-ratio controller
│   └── evaluate.py              # metrics, Δ/se estimators, O5 rule, latency
├── configs/
│   └── base.yaml                # B, λp, λs, lr, micro/eff batch, splits, seeds
├── scripts/
│   ├── download.sh
│   ├── preprocess.py
│   └── run_folds.sh
├── tests/
│   └── test_physics.py          # collapse guard, stencil, sine-vs-noise, Δ math
├── docs/
│   ├── design.md                # SOLE source of truth
│   └── features.json            # canonical feature tracker — always up to date
├── kernel-structx/              # Kaggle kernel dir (kernel-metadata.json, PUBLIC)
├── data/                # Kaggle dataset dir (dataset-metadata.json, PUBLIC, CC-BY-4.0)
├── Makefile
├── README.md
└── AGENTS.md
```

## Conventions

### Makefile (required)

- Root-level `Makefile` is **mandatory** and the canonical control surface. No
  setup/download/train/eval step lives only as prose — it belongs in the Makefile.
- Required targets: `setup`, `download`, `preprocess`, `smoke`, `train`,
  `eval`, `test`, `style`, `clean`.
- Each target is a thin wrapper over `uv run` / scripts — no business logic in Make.
- `make setup` is idempotent and safe to re-run.
- Every target has a `## short description` comment for `make help`.

### Python

- **Package manager: `uv`** — never invoke `pip` directly.
- Formatter: `black`, Linter: `ruff` (includes import sorting).
- Naming: snake_case for everything — files, variables, functions.
- Physics/math in `physics.py` / `mck_fit.py` runs in **fp32** (AMP autocast
  exclusion) — never compute `L_phys` or finite differences in fp16.
- Units: displacement (integrate LDV velocity once if needed) → detrend →
  low-pass → per-trial z-score with frozen train stats. See design §5.
- Seeds, trial IDs, `m,c,k` + `fd/fn`, λs, checkpoint URL+SHA logged per run.

### Kaggle (binding)

- Datasets AND kernels are **PUBLIC** (`datasets create -u`,
  `"is_private": "false"`). Never omit.
- `dataset-metadata.json` must declare `CC-BY-4.0` (template defaults CC0-1.0).
- `~/.kaggle/kaggle.json` or `KAGGLE_USERNAME`/`KAGGLE_KEY` for auth — never
  committed. Pull `kernels output` before session expiry; version checkpoints
  back to a Dataset for resume.

### License fence (binding)

- The ported TimeSformer is **CC-NC 4.0, research-only**. Any deployment path
  uses the ResNet control, never ported TimeSformer weights.

### General

- After every feature/large sub-feature completion, commit and push to github.
- Commits: conventional commits (`feat:`, `fix:`, `chore:`, `docs:`, `test:`, `refactor:`).
- If something feels out of scope, flag it rather than silently doing it.

## Multi-Agent Workflow

When `docs/features.json` contains 3 or more independent features (different
modules, no shared state), parallelize with builder subagents (max 3 at a time;
spawn the next when one completes). Same-file features batch sequentially.

1. **Plan**: pick independent features from `features.json`.
2. **Build**: TDD per feature — tests first, then implementation.
3. **Review**: adversarial review of each feature diff before its checkpoint
   push; bloat audit on the combined diff every 3–4 features.
4. **Verify**: `make test && make style` + feature checklist in `features.json`.

Fewer than 3 independent features: implement directly, no subagents.

## Agent Guidelines

- Always check `docs/design.md` before starting any task — it takes precedence.
- Never modify files in `/docs` unless explicitly asked (updating `features.json`
  status after a completed task is explicitly required — see below).
- Always run `make style` before considering any code done.
- Always run `make test` after making changes — if tests fail, fix before moving on.
- Always use snake_case for Python files/variables/functions.
- Always use `uv` — never `pip` directly.
- Always update `docs/features.json` after completing any task — mark features
  done, update test status, add new features if introduced.
- Any new setup/download/train/eval step must be a Makefile target, not prose.
- Never take a design-§11 forbidden shortcut; never claim damage classification.
- Trial-level splits only; `m,c,k` from one train-condition trial; tune on
  val-Δ only. If the archive yields unexpected conditions, stop and flag.

## Project-Specific Notes

- Dataset: Zenodo 17447083, `XJTU-DV-Beam.zip` only (1.7 GB, md5 in design §11).
  Rotor/Pump subsets are out of scope — never download.
- Loader mirror: https://gr.xjtu.edu.cn/web/lixiang/xjtu-dv ; contact
  lixiang@xjtu.edu.cn for data questions.
- TimeSformer repo is archived read-only, py3.7-era — port, don't pip-install it.
  Pin Kinetics weight URL+SHA at first fetch; ResNet-only fallback if unfetchable.
- Known gotchas: window estimate in early notes was 10× too high (true budget
  ~90–270 windows); `L_phys` is amplitude-blind by design (`L_LDV` anchors
  amplitude); NAG Polytec price link is dead (cite as unverified); arXiv HTML
  abstract says "1.6%" but the PDF says 1.6× — PDF governs.
