# Struct-X handoff — 2026-11-04 (morning, IST)

Session left in a clean, idle state. Nothing is running anywhere (verified:
local GPU 0 procs, remote queue gated-idle, Kaggle COMPLETE, no background
watches alive). Every result below is committed and pushed to `main`.

## Headline findings (all new since the last checkpoint)

1. **TimeSformer arm was void (CLS bug, now fixed).** The ported encoder's CLS
   token never mixed with patch tokens (`src/models/timesformer.py`), so every
   TimeSformer output was an input-independent constant (matched RMSE ≈ 2 on
   unit-variance data = the constant predictor; r ≈ 0.05). Fixed by mean-pool
   readout + an input-dependence regression test (`tests/test_models.py`).
   Full suite: **152 passed, 7 skipped**, black+ruff clean.
2. **First VALID pretrained numbers exist** (remote F012-2 redo, 6/6, fixed
   code): Off_set1 +0.20/+0.33, Off_set2 −1.05/−0.19, On_set3 −0.69/−0.12
   (data Δ / both Δ). Physics loses on all three folds — a real negative,
   not an artifact. B-grid/AMP/surface Kaggle verdicts are VOID (old code)
   and must be rerun; their "null" pattern is unclaimed until then.
3. **Ridge is the standing bar** (F013, `make ridge`, 9 s): matched RMSE
   0.29–0.66 beats deep `data_only` every fold; r ≈ 0.2 both sides.
   `L_phys` carries no usable per-window uncertainty (null on pretrained,
   anti-correlated on ridge, F014-4).
4. **B-grid so far (void, for the record)**: B=3/5/7/9 valΔ identical to 4
   decimals — exactly what constant predictors produce. No conclusion drawn.

## Feature state (see `docs/features.json`, synced 2026-11-04)

- Done: F000–F010, F013, F016 (site live), F017 (CI green), F018 (v2e sweep,
  gate FAIL reported as-is). Checklist 6, 14, 19 closed.
- In progress: **F011** (TimeSformer arm — fix landed, rerun owed),
  **F012** (F012-1 metrics ✓, F012-2 redone valid ✓, F012-6 ✓; B/AMP/surface
  redos owed).
- Planned: F014 (1+4 done; 2/3 staged in worktrees, 5 = report section),
  F015 (1–4 done; 5 = frozen tag, blocked on F012–F014), F019 (Box 404,
  blocked on owner re-share — do not touch).

## Machine / infra state

- **Local 4060**: idle. Worktrees: `wt-f014-attr` (F014-2: rollout + bounded
  IG/occlusion + variant spectra; ran on constants — diagnostic evidence),
  `wt-f014-probe` (F014-3: probes + CKA, `stability.md` present),
  `wt-f018-v2e` (done, merged). Nothing running; integrate F014-2/3 next.
- **Remote cselab11-41** (`snu@100.112.181.46`): idle, DONE_F012-2 +
  DONE_F012-6 + valid redo ckpts in `/var/tmp/sx-ckpt/F012-2-redo/`.
  Sleep masked (all 4 targets + GNOME idle off). Cron: `@reboot` + `*/15`
  queue runner (currently gated-idle on missing GO_F014 — correct).
  `/home` is 100% full; checkpoints live in `/var/tmp` (33G free).
  Stale `@reboot …/arnab/structx-run.sh` removed (backup `~/cron.bak.*`).
- **Kaggle**: auth = OAuth `~/.kaggle/credentials.json` (expires — if calls
  fail with "Authentication required", re-login; `-t` flag is SECONDS,
  `-t 720` killed two runs at ~734 s; use `-t 43200`). Last state: train
  kernel COMPLETE (old-code b9). Queue script: `scripts/nightly_watch.sh`
  (committed, currently NOT looping — relaunch after deciding redos).
  Datasets PUBLIC: `eros483/struct-x-beam`, `eros483/struct-x-vox`
  (CC-BY-4.0). Checkpoints: local `outputs/{b3,b7,b9,amp,surface}/ckpts`
  (old code); valid remote ckpts NOT exfiltrated (no longer needed locally —
  F014-2/3 run here).
- **CI**: green on `main`; runs full suite (torchvision pinned).

## Ordered next steps

1. Integrate F014-2 + F014-3 from worktrees (review, merge to main, push).
2. Decide Kaggle redo scope (full B/AMP/surface ≈ 12h T4 vs B=5 + B=7 +
   amp + surface ≈ 8h) and push b3-redo with the fixed code.
3. F014-5 report section (taxonomy + future-work/deployment mapping).
4. F015-5 frozen tag → closes checklist 17. 5. GO_F014 + JOB3 when
   cross-ckpt work needs the server again (optional).
6. F019 only if the Box link is re-shared.

## Resume commands

```bash
git log --oneline -3                                  # where main is
ls ~/sx-queue/DONE_* 2>/dev/null                      # remote: ssh snu@100.112.181.46 '...'
kaggle kernels status eros483/struct-x-train          # kaggle (re-auth if 401)
nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader   # local 4060
```

## Open risks / decisions for the human

- Kaggle redo quota: ~12h T4 for the full grid; confirm budget before firing.
- F014-2 Stage A ran on constants: keep as diagnostic or rerun post-fix
  (cheap on 4060 now that the code is fixed — recommend rerun, 1–2h).
- `/home` on the remote is full: do not write checkpoints there; the
  `~/sx-ckpt` symlinks already point at `/var/tmp` — do not "fix" them.
- F019 is dead until the Box owner re-shares; F018 is fully done.
