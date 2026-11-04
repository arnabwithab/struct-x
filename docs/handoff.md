# Struct-X handoff — 2026-11-04 (~09:10 IST / 03:40 UTC)

Session is ACTIVE — nothing is finished. Two GPU lanes + three watchers running.
Everything below is committed and pushed to `main` (`cc05c54` + this handoff).

## What changed since the morning handoff

1. **F014-2/3 integrated to main** (`c4d6054`, `ef8d1af`, fix `cc05c54`).
   Merged `feat/f014-2-attr` + `feat/f014-3-probe`, added `f014-layers2`
   Makefile target. Full suite: **171 passed, 7 skipped**, green.
2. **Stale-test contradiction found and fixed.** `test_timesformer_readout_is_
   input_independent` pinned the old CLS-constant bug and contradicted the
   mean-pool fix — renamed/flipped to `..._is_input_dependent` (same
   fixture). Stale CLS-never-mixes prose remains in layers2 headers
   (`src/f014_layers2.py`, `scripts/f014_layers2.py`); left intact, JOB8
   rerun refreshes the evidence.
3. **Remote queue JOB5–8 staged, hardened, running.** Runner backed up
   (`run_queue.sh.bak.2026-11-04` + `.bak2.0327`), `bash -n` clean. JOB8
   patched to skip missing `smoothness/sdof` variants (exit 0, no watchdog
   retry loop). JOB5 started 03:24:59 UTC, GPU at 100%.
4. **Kaggle amp-redo v1 pushed under `eros438/struct-x-train`** (fresh
   account, GPU, `-t 43200`, PUBLIC), RUNNING since 02:42 UTC. Fix
   containment verified: kernel clones `origin/main`, which has the
   mean-pool fix. Surface redo follows on COMPLETE. Staged in
   `/tmp/opencode/kp-amp/`; `kernel/` committed files untouched.
5. **Watchers (all detached, read-only):** `scripts/poll_kaggle.sh` (pid
   803351 → `~/sx-poll-kaggle.log`), `scripts/poll_remote.sh` (pid 803418 →
   `~/sx-poll-remote.log`) — 60 s ticks ×45, then 30-min ticks; exit +
   notify on COMPLETE/ERROR/DONE_JOB8. Supervisor loop
   `scripts/nightly_watch.sh` every 14 min (pid 803909 →
   `/tmp/opencode/nwatch.log`). Remote cron verified active (2 entries).

## Feature state (see `docs/features.json`, synced 03:40 UTC)

- Done: F000–F010, F013, F016, F017, F018. F014-2/3 code done (rerun owed).
- In progress: **F011** (F011-2 = JOB5 running), **F012** (F012-2 = JOB5
  running; F012-3 = JOB6 queued; F012-4 = JOB7 queued; F012-5 code done,
  comparison = JOB7 queued; F012-6 done).
- Planned: F014-5 (report section), F015-5 (frozen tag), F019 (Box 404 —
  do not touch).

## Machine / infra state

- **Local 4060**: effectively idle (pollers + supervisor are seconds/tick).
  Constraint stands: no local compute >30 min.
- **Remote cselab11-41** (`snu@100.112.181.46`): BUSY — JOB5 training.
  Expected chain (serial): JOB5 ≈11:00 → JOB6 ≈13:30 → JOB7 ≈15:00 →
  JOB8 ≈16:00 IST. `/home` still 100% full; ckpts → `/var/tmp/sx-ckpt`.
  Queue log: `~/sx-queue/logs/queue.log`; per-job logs `JOB{5..8}*.log`.
- **Kaggle** (active auth: `eros438`; old slugs under `eros483`): amp v1
  RUNNING, COMPLETE expected ≈09:30–11:30 IST → surface push + run
  (≈12:30–14:30). Datasets: PUBLIC `eros483/struct-x-beam` (cross-account
  readable; no `struct-x-vox` referenced anywhere — does not exist).
  `nightly_watch.sh` still points at the old `eros483` slug — its Kaggle
  leg is informational only until the slug is updated.
- **CI**: green through `cc05c54`.

## Triggers (nothing else to launch — react only)

1. Amp COMPLETE → validate `results_On_set1.json` (RMSE ≠ ~2), push
   surface redo (same knobs, `REPR=timesurface`, `USE_AMP=0`).
2. ERROR/CANCEL anywhere → pull logs, diagnose, report.
3. DONE_JOB8 → exfiltrate results JSONs, F014-5 report section, F015-5
   frozen tag, site figs refresh, push.
4. Pollers auto-relax to 30-min ticks ≈09:45 IST.

## Resume commands

```bash
git log --oneline -3                                    # main = cc05c54+
tail -2 ~/sx-poll-kaggle.log ~/sx-poll-remote.log       # watcher state
tail -3 /tmp/opencode/nwatch.log | cut -c1-160          # supervisor tick
kaggle kernels status eros438/struct-x-train            # amp redo
ssh snu@100.112.181.46 'ls ~/sx-queue/ | grep DONE; tail -1 ~/sx-queue/logs/queue.log | cut -c1-120'
```

## Open risks / decisions for the human

- Kaggle logs were empty at ~50 min RUNNING — likely API lag; poller
  confirms or flags by ≈09:45 IST.
- JOB8 runs layers2 Stage A only if `smoothness/sdof` ckpts existed (they
  don't) — skips with exit 0 by design; Stage-A nulls stay pre-fix
  diagnostic evidence, not findings.
- F014 outputs land on remote `/home` (1.8G free) — should fit; poller
  flags disk errors.
- F019 dead until Box re-share; no action.
