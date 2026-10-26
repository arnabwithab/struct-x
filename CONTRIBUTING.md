# Contributing to Struct-X

Sole spec: `docs/design.md` — it takes precedence over everything. If it is
ambiguous, flag it rather than improvising.

## Layout rule: `src/` vs `scripts/`

- `src/` is the tested library. Everything importable lives here; every module
  has a mirror test module under `tests/`.
- `scripts/` holds thin, untested CLIs over `src/` (`preprocess.py`,
  `evaluate_campaign.py`, `stamp_kernel.py`, shell drivers). No math, no model
  code, no physics.
- If a script grows logic, promote that logic into `src/` with unit tests.

## Workflow

1. TDD first: write the test, then the implementation. Never skip.
2. No physics/math code ships without a unit test (collapse guard, finite
   differences, `m,c,k` fit, Δ/σ estimators).
3. Snake_case for everything. Formatter `black`, linter `ruff`.
4. Run `make test && make style` before pushing.
5. Commit with conventional commits (`feat:`, `fix:`, `chore:`, `docs:`,
   `test:`, `refactor:`), push after every feature (`docs/features.json`
   updated in the same commit).

## Forbidden shortcuts (design §11)

Window-level splits, tuning λ on matched RMSE, `m,c,k` on test trials,
training the Transformer from scratch, damage-classification claims, hardcoded
sensor resolution. Trial-level splits only; `m,c,k` from one train-condition
trial; tune on val-Δ only.

## Data and Kaggle

- Never commit data, zips, checkpoints, or kernel outputs (see `.gitignore`).
- Datasets AND kernels are PUBLIC. Dataset license is CC-BY-4.0.
- `~/.kaggle/kaggle.json` or env vars for auth — never committed.
- One kernel dir (`kernel/`); scripts stay self-contained because Kaggle
  executes `code_file` standalone. Push via `make kernel-smoke`,
  `make kernel-train`, `make kernel-timing` (stamps metadata, then pushes).
- CPU smoke before GPU training; training needs a working smoke kernel first.
