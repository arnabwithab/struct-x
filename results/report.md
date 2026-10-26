# Struct-X results report (F008): physics cost/benefit + event-camera survey

## 1. O5 falsification verdict (design sec 9)

Per-trial aggregation over TEST trials (6 LOCO folds, ResNet3D control).
`R = Delta_base − Delta_phys`, physics wins iff `R > sigma_R`.

| Fold | R (RMSE) | sigma_R | Verdict | RQ4 | Matched harm |
|------|----------|---------|---------|-----|--------------|
| On_set1 | -0.033 | 1.885 | NON-IDENTIFIABLE | False | +0.412 |
| Off_set1 | -0.160 | 1.698 | NON-IDENTIFIABLE | False | +0.438 |
| Off_set2 | -0.214 | 2.101 | NON-IDENTIFIABLE | False | +0.108 |
| Off_set3 | -0.051 | 0.086 | NON-IDENTIFIABLE | False | +0.264 |
| On_set2 | +0.247 | 3.179 | NON-IDENTIFIABLE | True | +0.475 |
| On_set3 | -0.434 | 0.056 | NON-IDENTIFIABLE | False | -0.108 |

**Verdict line:** the SDOF physics term does not clear the trial-to-trial
noise bar on any fold — **regularisation benefit with no measurable robustness
gain** (consistent with the allowed O5 non-identifiable outcome). Physics also
costs matched RMSE on 5/6 folds (+0.1 to +0.5), reported separately per sec 9,
never netted. RQ4 (freq degrades less than waveform) holds in 1/6 folds only.

Caveat: matched test RMSE itself is unstable across folds (0.2–2.5, trial stds
up to 5 on unit-variance data) — 5-trial test sets with outlier trials
dominate. The matched upper bound needs more trials before any Delta claim
could be load-bearing. `m,c,k` stable across folds (fd 6.7–6.9 Hz, k≈1800).

## 2. 15a physics cost/benefit (measured on T4, same micro-batch/GPU/seeds)

| Pair | ms/window/step | Overhead |
|------|----------------|----------|
| (a) data-only 122.0 vs (c) +SDOF 113.9 | — | −6.7% (noise) |
| (b) +smooth 116.8 vs (d) both 119.7 | — | +2.6% (noise) |

- FLOP prior: `~6T` multiply-adds/window vs `~1e10` encoder — measurement agrees:
  **no measurable per-step cost** (run-to-run variance dominates; (c) < (a)).
- Savings in convergence (epochs-to-target / epochs-run): physics reaches target
  in fewer epochs on most folds — e.g. On_set2 data 39/49 vs sdof 5/16;
  Off_set3 data 1/12 vs sdof 1/11; exceptions (Off_set2 both 2/50) show no
  systematic GPU-hour saving. Campaign ≈ 12 h GPU across v5–v7 runs.
- Accuracy price: matched-harm column in sec 1 (+0.1 to +0.5 on 5/6 folds).
- **Verdict line:** "Physics cost ~0% per step, saved no consistent GPU-hours
  (faster target on 4/6 folds, full-budget runs on 2/6), moved matched RMSE by
  +0.1 to +0.5 on 5/6 folds — regularisation benefit with no compute saving
  and no robustness gain" (consistent with the O5 non-identifiable outcome).

## 3. 15b event-camera usage survey (use → trick → our sec)

| Other use | Their trick | Struct-X decision |
|-----------|-------------|-------------------|
| Event–frame VDM (Wu et al. 2026) | IWE contrast max + NCC, survival-rate protocol | HW trigger sync (sec 5); trained encoder, not per-sequence optimisation (sec 6) |
| Event optical flow / TimeLens | Short-term motion, frame interpolation | Voxels over time surfaces (sec 6); surfaces kept as fallback flag (sec 5) |
| HDR/high-speed video (Rebecq) | 140 dB / µs-timestamp edge over CMOS | Sensor-motivation numbers (sec 1); frames are the "fails >100 Hz" column (sec 9) |
| Representation taxonomy (Gallego) | Frames vs surfaces vs voxels vs learned | Voxels + polarity + frozen train stats (sec 6); surfaces fallback only |
| Limitations literature (Wu sec 4.5) | Continuity/brightness-constancy failures | Δ exists to measure exactly this (sec 9); ROI crop + refractory filter (sec 5) |
