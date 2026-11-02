# F012 Kaggle-T4 GPU queue: B-grid -> AMP -> surface (PREP ONLY)

Parent runs these; subagent did prep only (no pushes, no long runs, no commit).
Quota cited: ~14 h T4 left. Design doc secs 5,7,8,10 govern; never contradict them.

## Entry points (all three exist; verified `--help` only, no training runs)

| # | Flag path | Local | Kaggle (`kernel/train.py` top block; script kernels take no CLI args) |
|---|---|---|---|
| 1 | B override `--b {3,7,9}` | `bash scripts/run_folds.sh --b 7 --folds On_set1 --variants data_only,both` | `B = 7` (temporal pos-embed re-interps 8->B in `load_kinetics_weights`; `TimeSformerEncoder(num_frames=B_EFF)`) |
| 2 | AMP `--use-amp` | same + `--use-amp` (fp32 carve-out already in `_train_single_config`) | `USE_AMP = True` -> `run_fold(..., use_amp=USE_AMP)` |
| 3 | repr `--repr voxel\|timesurface` | `scripts/preprocess.py --b 5 --repr timesurface` + train `--repr timesurface` | `REPR = "timesurface"` (`B_EFF=1`; `voxel_to_model_input` maps (2,H,W)->(2,1,224,224), no interp needed) |

Local data path: `scripts/preprocess.py --config configs/base.yaml --b B --repr R`
decodes with `n_bins=B, repr_kind=R`. `--b/--bins` alias added; `--config`
supplies the bins default. `make train` now works (`PYTHONPATH=src` fix in
`run_folds.sh`); queue flags are source of truth for B/AMP/repr, base.yaml for
lr/epochs/patience/micro/grids. Tests: `tests/test_train_cli.py` (15 tests).

`src/train.py` re-decodes from the archive on every run — it does NOT consume
`scripts/preprocess.py` `.npz` files. Pass `--b/--repr` identically to both so
the cached voxels match what train actually uses.

## Bounded defaults (quota protection)

Every comparison push below runs **1 fold x 2 variants** (`FOLDS + VARIANTS`
in the kernel block). Full 4-variant x K-fold grids are F012-2/F012-6, NOT
this queue. B=5 fp32 voxel baseline already exists (F011 pilot) — compare
against it, do not re-run it. **Local GPU/CPU runs >15 min are BANNED.**

## Exact push order (edit block, stamp, push, pull, version)

Preconditions: `kaggle.json` auth present, dataset `eros483/struct-x-beam`
PUBLIC and ready (`kaggle datasets status eros483/struct-x-beam`).

Per run: edit the `F012 queue knobs` block in `kernel/train.py`, then:

```bash
python3 scripts/stamp_kernel.py train && kaggle kernels push -p kernel --accelerator gpu -t 720
kaggle kernels status eros483/struct-x-train
kaggle kernels output eros483/struct-x-train -p outputs/<run-tag>   # BEFORE session expiry
kaggle datasets version -p data -m "<run-tag> ckpts"                 # resume/version per sec 10
```

1. **B-grid** (3 pushes): `B=3`, then `7`, then `9`; `FOLDS=["On_set1"]`,
   `VARIANTS=["data_only","both"]`; `USE_AMP=False`, `REPR="voxel"`.
   Run tags: `outputs/b3`, `outputs/b7`, `outputs/b9`.
2. **AMP compare** (1 push): `B=5`, `USE_AMP=True`, same bounded folds/variants.
   Tag `outputs/amp`. Compares against the fp32 pilot at identical B/fold/variants.
3. **Surface compare** (1 push): `REPR="timesurface"`, `B=5` (ignored, `B_EFF=1`),
   same bounded folds/variants. Tag `outputs/surface`.

`results_{held}.json` now records `b/repr/use_amp/micro` per push for the paper log.

## VRAM / exfil expectations

- T4: micro-batch probe starts at **8** (`micro_batch_t4`), OOM ladder -> 4,
  eff batch 16 via accumulation; 4060: 4. Logged per run with `nvidia-smi`.
- `L_phys` + finite differences + eps stay **fp32** under AMP (sec 7 carve-out);
  `as_fp32` covered by `tests/test_amp.py` (green, untouched).
- Checkpoints: best-per-fold ONLY; never accumulate best+last (sec 10 disk cap).

## Do NOT

- No full-fold / full-variant pushes in this queue; no local training runs.
- No window-level splits, no `m,c,k` on test trials, no tuning on matched RMSE
  (tune on val-Delta_RMSE only), no damage-classification claims.

## Known follow-ups (NOT done here)

- `scripts/evaluate_campaign.py` hardcodes `n_bins=5` + voxel + `num_frames=5`;
  B-grid/surface ckpts need a `--b/--repr` pass there before scoring.
- `make eval` (`python -m evaluate --config`) still has no CLI (`src/evaluate.py`
  is functions-only); pre-existing gap, out of F012 scope.
- `tests/test_models.py` needs `torchvision` (absent in this venv; pre-existing).
- Ruff debt outside the touched files is pre-existing; touched files are clean
  except `kernel/train.py` PLW1510 on the untouched `sh()` helper (pre-exists
  on main, out of scope here).
