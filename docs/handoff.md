# Struct-X handoff — 2026-11-04 (~17:30 IST)

Session COMPLETE. All lanes finished. Checklist 20/20. `main` @ `6ba1998`,
tag `frozen-2026-11-04` pushed, site live (200).

## Final state

- **F011/F012 closed, all valid on fixed code.** JOB5 (3 folds), JOB6
  B-grid (valΔ 0.327/0.332/0.328/0.330 — B irrelevant), Kaggle amp v2
  (1.91/2.63) + surface v3 (2.09/2.61) SHA-gated. Kaggle v1 was a stale
  clone (origin/main=45a86d4) — void, recorded, never cited as a finding.
  Kernel now fail-fasts on SHA mismatch (`STRUCTX_SHA`).
- **F014 closed.** Probes/stability redone on fixed ckpts (fn R²≈−0.2,
  slope R²≈0.55, cosine≈1.0, CKA≈0.04); report §11 taxonomy (T1–T4) + §9
  refresh written. Attribution (layers2) code integrated; full rerun
  blocked by design (needs 4-variant v8; zero-new-training premise).
  Amp/surface dropped from JOB8 scope (verdicts already committed nulls).
- **F015 closed.** Tag `frozen-2026-11-04` covers results+report+F014 code
  (site/tracker docs followed in `6ba1998`). CI green.
- **Site live** with valid numbers; `pretrained_vs_control.png` still shows
  pilot data (figure regen needs verdict-JSON rebuild — flagged, not done).
- **Remote queue idle** (gated on GO_F014 placeholder); cron intact.
  Kaggle lane closed (nothing left worth pushing). Poller shells dead by
  design (all exited on terminal states); supervisor loop may still tick
  (harmless, old slug).

## Resume commands

```bash
git log --oneline -3; git tag -l | tail -2
kaggle kernels status eros438/struct-x-train   # COMPLETE (v3); needs re-login if 401 (OAuth expires ~hours)
curl -s -o /dev/null -w "%{http_code}\n" https://arnabwithab.github.io/struct-x/
```

## Known warts (deferred, all recorded)

1. Site `pretrained_vs_control.png` = pilot data; regen needs
   final_verdict_pretrained.json rebuild.
2. Stale CLS prose in layers2 headers (flagged in code review note).
3. nwatch supervisor still points at old `eros483` slug.
4. F019 dead (Box 404). `wt-f014-*` worktrees merged; `wt-f018-v2e`
   merged earlier. Leaves `beam-vox/`, `outputs/`, `results/f014_redo/
   outputs/` untracked locally (regenerable staging).
