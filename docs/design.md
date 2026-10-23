# Struct-X — Design Document (sole source of truth)

> This file is the **only** reference for building Struct-X. All other reference
> files are deleted after this is written.
> A future agent must be able to implement, train, and evaluate the project from
> this file alone.

## 0. One-paragraph summary

Struct-X learns `f_theta: E_s -> q_hat(t)`: from an asynchronous event stream
`E_s = {(x,y,t,p)}, p in {-1,+1}` captured under sensing configuration `s` to the
vibration waveform `q(t)` that a co-located laser Doppler vibrometer (LDV) would
have recorded, plus the dominant natural frequency `f_n`. The thrust is
**robustness**: the same physical motion must give the same `q(t)` when `s`
(illumination, contrast threshold `C`, bias currents, sensor variant / pose)
changes. A reduced-order single-degree-of-freedom (SDOF) physics residual
regularises a spatiotemporal event-vision encoder. Central metric is the
robustness gap `Delta = M(shifted) - M(matched)`. Pre-registered falsification:
if physics reduces `Delta` by less than trial-to-trial std `sigma_Delta`, report
it as non-identifiable. Data: 30 hammer-impact free-decay trials from
XJTU-DV-Beam only.

## 1. Research question and formal problem

**Question:** can a deep spatiotemporal model watch the flicker of events from a
vibrating beam and reconstruct what a collocated LDV would have recorded, even
as lighting or bias settings drift?

- Input: event stream `E_s = {(x,y,t,p)}`, `t` in microseconds, `p in {-1,+1}`,
  under sensing configuration `s = {illumination, contrast threshold C,
  bias currents (e.g. refractory period), camera distance/angle, sensor variant}`.
- Ground truth: LDV waveform `q(t)` (displacement or velocity, kHz CSV),
  synchronised by hardware trigger.
- Learn: `f_theta: E_s -> q_hat(t)` (Eq. 1) trained with physics constraint.
- Invariance requirement: changing `s` must not change physical motion, so
  `q_hat` must not move. This is measured, not assumed.

Event generation model (motivation, not fitted): a pixel fires iff
`|Delta log I| > C`. Effective `C ~ N(C_nom, 2–4% contrast)` across pixels
(Gallego et al.); refractory/bias tuning shifts event rates ~15–35%. Sensor is
~140 dB dynamic range, microsecond timestamps, per-pixel kHz bandwidth,
negligible motion blur — suited to beam vibration at tens to thousands of Hz,
vs ~60 dB CMOS frame cameras that blur above ~100 Hz.

Stakeholders: SHM teams (civil infrastructure, factory machinery — bridges,
beams, wind turbines, pumps). They use bonded accelerometers (add mass/wiring,
change dynamics) or LDVs (stationary head, line of sight) or frame cameras
(blur). Struct-X is a low-cost non-contact optical trade-off: ~10x cheaper than
LDV, kHz-capable, field-lighting-tolerant. It must survive field lighting at kHz
bandwidth.

## 2. Scope and non-goals (binding)

- **In scope:** vibration waveform estimation + sensing-robustness only.
- **Out of scope:** damage classification. XJTU-DV-Beam provides **no damage
  labels**; never claim classification.
- Single physical system only: H-beam under hammer impact and free decay.
  No claim beyond this rig; generalisation to other rigs is future work.
- Target is a **continuous waveform**, so robustness needs a continuous
  definition (`Delta` on RMSE / freq error), not classification accuracy.
- Ignore XJTU-DV Rotor and Pump subsets (fault diagnosis, 2.8 GB + 7.8/8.7 GB).
  Use Beam only.
- Contrastive embedding term `L_c` for illumination invariance is **optional,
  not core** (no definition exists — do not implement unless the core pipeline
  works and a definition is written first).

## 3. Objectives O1–O5 (acceptance criteria)

- **O1 (accuracy):** reconstruct LDV waveform `q_hat(t)` and dominant frequency
  `f_hat_n` from events under the *same* acquisition condition.
- **O2 (stability):** quantify `Delta = M(shifted) - M(matched)` for
  `M in {RMSE, |f_hat_n - f_n|/f_n}` across illumination and threshold shifts;
  minimise `Delta`.
- **O3 (scarce data):** feasibility on **only 30 independent trials**. Windowed
  splits at **trial level only**. Do NOT train a large Transformer from scratch
  without pretraining.
- **O4 (mechanism isolation):** ablate what helps: SDOF residual vs temporal
  smoothness vs both. `m,c,k` must be **estimated from data**, never given.
- **O5 (falsification):** if `Delta` reduction from physics `< sigma_Delta`
  (trial-to-trial std), report physics term as **non-identifiable**. A negative
  result counts. Also report matched-condition degradation separately — a
  regulariser that helps shifted data but harms matched data must be reported
  as such, not hidden.

Related testable claim (RQ4): relative frequency error degrades less
under shift than waveform RMSE (spectral peak is coarser than pointwise trace).
Test and report it.

## 4. Literature survey (12 sources, 3 families)

Group by technique, not per-paper. Every future-agent design choice must cite
which family/gap motivates it.

### Theme A — Geometric training-free event–frame deformation measurement

- Wu et al. 2026 (arXiv:2602.14376, full PDF v1 read 2026-10-23): fuse temporally-dense /
  spatially-sparse events with co-registered frames; solve dense displacement
  field. Affine Invariant Simplicial (AIS) framework tiles field into
  low-parameter triangular sub-regions, affine `x(X)=AkX+bk` per sub-region
  (lossless via anchor vertices from the reproduction property `I[f]=f`),
  joint event-contrast maximisation (IWE, Eq.9–10) + image cross-correlation
  (zero-mean NCC) coarse-to-fine (rigid → coarse → split-refine); greedy
  neighbourhood guides poorly-converged tiles from well-converged neighbours
  (strain-continuity constraint, anchors fixed greedily). Benchmark: >120
  aligned event+frame sequences (Prophesee EVK4 + 210 fps grayscale via 50:50
  beam splitter, synced by square wave), GT from VDM on high-speed frames.
  PDF-verified results: **65.7% survival on 100+ px displacements (1.6× SOTA)**
  vs CoTrackerV3 45.2% (Table 1; EPE 3.204, SEPE 0.813); **18.9% data
  storage/compute** of high-speed video (106.4 MB vs 562.5 MB @100 fps,
  Table 2); optimisation ablation on single RTX 4090 24 GB, Adam/PyTorch:
  7.2 s vs 26.5 s mean converge time (Table 3). NOTE: the arXiv HTML abstract
  page renders "1.6%" but the PDF (abstract + §4 contributions) says 1.6× —
  PDF governs; the 65.7% / 1.6× / 18.9% numbers stand VERIFIED. Stated limitations (support
  our critique): spatial-continuity + brightness-constancy assumptions, failure
  on topological change/specularities.
  https://arxiv.org/abs/2602.14376 — LIVE.
- Gallego et al. 2022 (TPAMI Q1): taxonomy used here — event frames, time
  surfaces, voxel grids, learned representations; threshold-variation
  convention. https://doi.org/10.1109/TPAMI.2020.3008413
- Gehrig & Scaramuzza 2023 (TPAMI): simulator convention for stress-testing
  camera parameters without new capture.
- Rebecq et al. 2017 (CVPR): high-speed/HDR video from events (cited background).

### Theme B — Physics-informed / physics-guided learning for structural dynamics

- Raissi et al. 2019 (JCP Q1): generic PINN — data fidelity + PDE residual,
  continuous or discrete (arbitrary-order implicit Runge-Kutta).
  https://doi.org/10.1016/j.jcp.2018.10.045
- Karniadakis et al. 2021 (Nature Rev Phys): warning — training can fail for
  non-physics reasons; a bad residual *harms* accuracy. Justifies O4/O5
  ablations and gradient-ratio monitoring.
- Wang et al. 2024 (Eng Struct Q1, PhyResNet): ResNet maps accelerations to
  stiffness-reduction states; FEM decoder with Newmark-beta reconstructs
  response; loss includes `M x_ddot + C x_dot + K x = F(t)`. Uses only 5% data.
  10-element beam, 1–3% model error + 10% Gaussian noise: R +13.1% vs pure
  ResNet (0.789→0.870 at 10% noise; 0.808 at 20% noise); MSE 2.0e-4→1.6e-4
  under model error alone. 7-storey frame same trend. Key precedent for
  data-efficiency via physics, but assumes known M,C,K / FEM. We have neither.
  https://doi.org/10.1016/j.engstruct.2024.118703
- MSSP 2025 (Q1, vision+PINN, closest to Struct-X): fuses accelerometer + frame-
  camera displacement with physics loss. Inherits frame blur on fast small
  motion; residual lives on accelerometer channel, not a high-bandwidth optical
  channel. https://doi.org/10.1016/j.ymssp.2025.112961

### Theme C — Spatiotemporal Transformers + robustness-aware SHM

- Bertasius et al. 2021 (ICML A*, TimeSformer, verified live 2026-10-23): patch
  self-attention video models; **divided** space-time attention (temporal then
  spatial per block) beats joint on accuracy *and* memory; joint overflows GPU
  at target resolutions. Large hits 80.7% top-1 Kinetics-400, handles >60 s
  clips, but conv bias still helps when data scarce and gains need pretraining
  scaled with resolution. Direct constraint: our 30 recordings force
  pretrained init + CNN control. Code:
  https://github.com/facebookresearch/TimeSformer — LIVE but **archived
  read-only 2025-01-01, license CC-NC 4.0 (non-commercial, research OK)**;
  original stack Python 3.7 + 32×V100; HR/L variants need ~32 GB GPUs, so on
  T4/4060 use ONLY base 8-frame 224 (`TimeSformer_divST_8x32_224`) + finetune.
  Pretrained zoo (K400/K600/SSv2/HowTo100M) hosted on that repo. Port to modern
  torch — do not pip-install the 2021 env blindly.
  https://proceedings.mlr.press/v139/bertasius21a.html — LIVE.
- Zhou & Lai 2024 (MSSP Q1, open-set DA): SHM classification under varying
  environments + unknown classes. Transferable insight: train/test must differ
  **by construction** (leave-one-condition-out), not random split.
  https://doi.org/10.1016/j.ymssp.2024.111561
- He et al. 2016 (CVPR): 2D ResNet — the §6/§8 control uses a 3D inflation for video input.
- XJTU-DV companion: Li et al., CJME 2026 (Q1).
  https://doi.org/10.1016/j.cjme.2025.100163

### Comparative table (P1=accuracy, P2=stability under shift, P3=30 recordings)

| Dimension | A. Geometric event-frame VDM | B. Physics-guided structural DL | C. Spatiotemporal Transformer + robust SHM |
|---|---|---|---|
| Input | Raw events + co-registered frames (voxel / time-surface) | 1-D sensor series (accel/displ), sometimes FEM features | Stacked voxel or frame patches |
| Supervision | None (optimisation); GT only for eval | Supervised on indices/traces; low-data gains | Supervised, large labels, pretrain needed |
| Physics | Implicit linear-elastic + simplicial prior | Explicit PDE residual + FEM decoder (Newmark-beta) | None |
| Needs FE model? | No | Yes (or identified m,c,k) | No |
| Data regime | Any (per sequence) | Modest; biggest gains at 5–10% data | Large; degrades on small sample |
| Handles sensing shift? | Not evaluated | Not evaluated (noise/model-error only) | Yes by design but categorical only |
| Meets P1? | Yes for displacement | Yes for damage index | Unproven for waveforms |
| Meets P2? | Unknown | Partial (noise only) | Partial, wrong target type |
| Meets P3? | Yes | Yes | Only if pretrained |
| Strength | Training-free, SOTA accuracy | Data-efficient, correct inductive bias proven on beams | Best long-range modelling, memory-efficient divided attention |
| Limitation | Needs frames, brightness constancy, spatial continuity, no prior | Requires FE/M,C,K, no event fusion, no shift metric | Data-hungry, fails at 30 trials, classification robustness ≠ waveform |

No family gives calibrated confidence. No paper reports degradation under
changed sensing configuration as a headline number; A/B test zero-mean additive
noise, C tests categorical shift. Defining the continuous-waveform analogue
(trace/f_n movement) is itself a contribution.

### Research gaps G1–G5 (each maps to a novelty claim)

- **G1:** no event→LDV vibration pipeline exists; Wu stops at deformation field.
- **G2:** physics + events never combined (disjoint literatures).
- **G3:** sensing-condition shift unmeasured (no gap under changed C /
  illumination / sensor).
- **G4:** no waveform-domain robustness definition (only classification/noise).
- **G5:** physics priors unidentified under scarce data; Wang gives parameters,
  but estimating from the same 30 trials risks collapse `q_hat→0` driving the
  residual to zero. No treatment exists → normalised residual guard.

## 5. Dataset — XJTU-DV-Beam (exact spec)

- **Paper:** Li et al., "XJTU-DV: Open-Source Dynamic Vision Dataset for
  Non-Contact Vibration Measurement and Fault Diagnosis of Mechanical Systems",
  *Chinese J. Mech. Eng.*, 2026. DOI above.
- **Download:** Zenodo record 17447083, DOI `10.5281/zenodo.17447083`,
  https://zenodo.org/records/17447083 (CC-BY-4.0). Mirror/info:
  https://gr.xjtu.edu.cn/web/lixiang/xjtu-dv . Contact: lixiang@xjtu.edu.cn.
  Files verified Oct 2025: `XJTU-DV-Beam.zip` **1.7 GB** (+ Rotor 2.8 GB,
  Pump 7.8 + 8.7 GB — do NOT download). Page lists 21.0 GB total unpacked.
- **Sub-dataset:** XJTU-DV-Beam (vibration measurement) only.
- **System:** H-shaped beam rig, hammer-excited free decay until damped.
- **Sensors:** event camera (DVS, µs timestamps; nominal ~346×260 to
  640×480 — confirm from actual headers on load, do not hardcode) + LDV
  ground truth.
- **Records used:** 30 independent hammer-impact trials available for this beam
  configuration, each 2–5 s free decay, dominant mode tens to a few hundred Hz.
- **Conditions `s`:** varying illumination (low/high), contrast threshold +
  bias, camera distance/angle. Exact label names must be read from archive
  directory/file names + author example loaders; do not invent condition IDs.
- **Raw types:** events `{(x,y,t,p)}` (aedat4 or CSV, `p in {-1,+1}`, `t` in µs);
  LDV `q(t)` CSV at kHz. **Binding unit convention (fixes velocity/displacement
  punt):** treat LDV as displacement. If the file header shows velocity
  (m/s), integrate once via cumulative trapezoid then linear-detrend to obtain
  displacement BEFORE any other step. All RMSE/MAE are reported in these
  post-conversion displacement units (normalised scale below).
- **Derived input:** polarity-split voxel grid `V` with shape channels-first
  `(2B, H, W)` per temporal stack — i.e. B temporal bins × 2 polarities = 2B
  channels (channels-last notation `R^{B×H×W×2}`). B=5 default,
  ablate 5–9. Standardise per channel with **frozen train-set mean/std**
  (eps 1e-6), not per-window statistics.
- **Derived target:** LDV resampled to fixed **Fs = 1000 Hz, T = 1000 samples
  per 1.0 s window** (`dt = 1 ms`). `f_n` target = FFT peak on the FULL trial
  (Hann, zero-pad 4×, parabolic interpolation, search 5–500 Hz; resolution
  ~1 Hz native, interpolation required to meet the <2% budget).
- **Preprocessing (per trial, binding numbers):** hot-pixel mask (>5× median
  event count → drop) + refractory filter (drop same-pixel events <100 µs
  apart), ROI crop = contiguous y-band where mean event-count projection ≥50%
  of peak, pad 8 px, resize longest side ≤256 (do NOT hardcode sensor size);
  time-surface fallback behind `--repr` flag; LDV = linear detrend →
  4th-order Butterworth low-pass at 500 Hz (Nyquist of the 1 kHz resample) →
  per-TRIAL z-score with frozen train stats. Sync by hardware trigger.
- **Impact transient (binding):** the SDOF residual assumes free decay, so
  **exclude all windows starting <100 ms after the hammer trigger** of each
  trial. Window 0 is never scored.
- **Splits (binding, replaces "k-fold LOCO"):** deterministic
  leave-one-condition-out over the conditions found in the archive inventory.
  Both source wordings preserved: Step 7 list = matched, illumination shift,
  **camera shift** (distance/angle/pose), combined shift; Metrics list =
  held-out trials of A = matched upper bound, illumination shift,
  threshold/bias shift, combined shift. Treat camera shift and threshold/bias
  shift as separate axes; expect 4 folds but if the archive yields K≠4,
  folds=K — never force a fixed 5-fold (LOCO folds = conditions found). Within matched condition split trials into
  train/val-matched/test-matched; within EACH shift condition split trials
  50/50 into **val-shift (tuning + early-stop ONLY)** and **test-shift (final
  Delta ONLY — never touched during tuning)**. This separation fixes the
  val-shift = test-shift leakage. **Window-level random splits are forbidden.**
  Holdout for physics: `m,c,k` are fit on ONE trial of the TRAIN condition
  only (see §7), never on val/test or shift trials.
- **Example code:** author loaders/visualisation at gr.xjtu.edu.cn link above;
  vendor it or pin commit if reused.

CORRECTION (verification 2026-10-23): the "~2,000–3,000
windows" estimate is arithmetically wrong. 1.0 s windows / 0.5 s stride over 2–5 s
trials gives 3–9 windows/trial → **~90–270 windows total for 30 trials**
(before 100 ms exclusion). Future agent: count actual windows after loading
and use the real number; do not plan storage/compute for 2–3k windows.

## 6. System architecture (5 blocks)

```
E={(x,y,t,p)}, condition s
  → (1) Representation: polarity-split voxel, B bins × 2 polarities (see §5 shape note)
  → (2) Spatiotemporal encoder: divided space-time attention (pretrained, ported) OR lightweight 3D-CNN fallback → latent z_t
  → (3) Heads: waveform q_hat(t) in R^T (T=1000) + frequency f_hat_n (trained, see §7)
  → (4) Physics layer: m q_ddot + c q_dot + k q = 0 (m=1 convention), f_n = (1/2π)√(k/m), frozen m,c,k
  → (5) Output: q_hat(t), f_hat_n, normalised physics residual + uncertainty proxy
```

1. **Representation — voxel grids, not time surfaces.** Voxels preserve temporal
   resolution, feed standard spatiotemporal backbones, keep polarity (critical
   when C shifts +/− events together). Default B=5, ablate 5–9.
2. **Encoder — divided attention primary, CNN control.** Divided (temporal-then-
   spatial per block, TimeSformer-style) beats joint on accuracy and fits in
   memory at event-tensor H×W; joint overflows and is **banned**. Because
   Transformers need pretraining to beat CNNs on small data, initialise from
   Kinetics-pretrained weights and always report side-by-side with a 3D-ResNet18
   with identical heads (low-data ablation).
   **Adapter (binding, fixes B=5 vs 8-frame mismatch):** feed voxels as
   T=B frames with C=2 channels (polarity = channels, NOT extra frames), so the
   default net runs at T=5, C=2, 224×224. Surgery on the ported TimeSformer:
   (a) replace patch-embed conv in-channels 3→2, init by channel-mean inflation
   of the Kinetics weights; (b) bilinear-interpolate spatial pos-embed to 224²
   and linearly interpolate temporal pos-embed to B (re-interpolate per B in the
   5–9 ablation); (c) ROI preprocessing resizes longest side to 224 with bilinear
   interp then zero-pads to 224×224 square (aspect preserved; record pad amounts
   per trial — never stretch the beam axis). Shape smoke test required before
   training: `(microB, 2, B, 224, 224) → (microB, T=1000) + (microB,)` with no
   shape error and `torch.cuda.max_memory_allocated()` logged.
3. **Heads:** two heads from `z` predict `q_hat(t)` (T=1000, same 1 kHz sampling
   as resampled LDV) and `f_hat_n` (Hz, trained by `L_freq` in §7 — the old
   "FFT-peak as diagnostics" is inference-only fallback when the head is
   disabled, never the training target).
4. **Physics layer:** SDOF free-decay residual of the dominant mode
   `m q_ddot + c q_dot + k q = 0` with **m=1 by convention** (log-decrement gives
   only frequency + damping ratio, so absolute scale is unidentifiable — see
   §7). No forcing term: valid only on free decay (windows <100 ms post-trigger
   excluded per §5). Generic smoothness `||q_dot||` is a DC-seeking shrinker, not
   true smoothness — keep λs small and gate with the matched-RMSE rule in §9.
   Freeze `m,c,k` from the single global TRAIN-condition trial (§7).
   Residual normalised to block G5 collapse; `L_phys` is **amplitude-blind by
   design** (q→αq cancels) and amplitude is anchored by `L_LDV` alone — the §11
   smoke test must therefore check matched RMSE jointly, never `L_phys` alone.
5. **Output per window:** (1) `q_hat(t) in R^T`, (2) `f_hat_n`, (3) normalised
   `L_phys` as self-consistency score. No contrastive term (deleted — see §7).

## 7. Equations, losses, training procedure

Total loss (no contrastive term — deleted; the `L_c` term was optional/not
core and never defined, so it is NOT implemented):

```
L = L_LDV + λp · L_phys + λs · L_smooth + λf · L_freq,   λf = 0.1 fixed (not tuned)
L_LDV    = mean_T ||q_hat − q||²          (mean over T=1000, then mean over batch)
L_smooth = mean_T ||q_hat_dot||²
L_phys   = mean_T ||m·q_hat_ddot + c·q_hat_dot + k·q_hat||² / (mean_T ||q_hat||² + ε)
L_freq   = (f_hat_n − f_target)² / f_target²   (relative; f_target = LDV full-trial FFT peak, one value per trial)
f_n = (1/2π)·√(k/m)
```

- Reduction (binding): `mean_T` = 1/T sum over the window's T=1000 samples at
  Fs=1000 Hz (`dt` = 1 ms); per-window losses then averaged over the batch.
  All λ grids in §8 are calibrated to this mean-reduced scale.
- Derivatives (binding): 2nd-order central differences interior
  (`q_dot[i]=(q[i+1]−q[i−1])/2dt`, `q_ddot[i]=(q[i+1]−2q[i]+q[i−1])/dt²`),
  1st-order one-sided at samples 0 and T−1. No pre-filtering of `q_hat`
  (accepted 1/dt² noise amplification is what λs controls).
- `m,c,k` (binding): **m=1 by convention** — log-decrement/modal fit on free
  decay yields only damped frequency `fd` + damping ratio `ζ`, so absolute
  scale is unidentifiable. Fit on ONE global trial of the TRAIN condition only:
  `ζ` from log decrement, `fd` from zero-crossings/FFT, then
  `fn = fd/√(1−ζ²)`, `k = (2π·fn)²`, `c = 2ζ√k` (m=1). Report both `fd`
  (measured) and `fn` (undamped, enters the loss); the <2% freq target in §9
  is evaluated against LDV `fn` estimated the same way, so damping bias
  cancels to first order.
- Division by `mean_T||q_hat||²` is the **G5 collapse guard** — without it
  `q_hat→0` trivially minimises the physics term. Never remove the
  denominator. Consequence (binding): `L_phys` is **amplitude-blind by design**
  (`q→αq` cancels); amplitude is anchored by `L_LDV` alone.
- `ε`: the loss definition carries `+ε` with no fixed value — placeholder `ε=1e-6`
  (relative to the z-scored unit-variance scale from §5). **Precision carve-out (binding): compute `L_phys`,
  all finite differences, and `ε` in fp32** (AMP autocast exclusion) — fp16
  (min normal ≈6e-5) would underflow ε and NaN the second differences.
- Gradient-ratio controller (binding, Alg step 7b is actionable, not logging):
  track `r = ||∇L_phys||/||∇L_LDV||` per epoch; warm up with λp×0.1 for the
  first 5 epochs; thereafter if r>1.0 for 3 consecutive epochs halve λp
  (floor 1e-4), if r<0.01 for 3 epochs double λp (cap 0.1).

Pseudocode (one fold, 9 steps):

```
1: fit ζ, fd on ONE train-condition trial → m=1, k, c; freeze; record fd AND fn
2: for epoch (max 50):
3:   for batch (V,q) from train-recordings only (micro-batch §10, effective batch §8):
4:     z ← Encoder(V);  q_hat ← WaveHead(z);  f_hat_n ← FreqHead(z)
5:     compute L_LDV, L_freq, L_phys (fp32), L_smooth
6:     L ← L_LDV + λp·L_phys + λs·L_smooth + 0.1·L_freq
7:     Adam update; update r-controller above
8:   early-stop (patience 10) on VALIDATION Delta_RMSE (val-shift vs val-matched, §9)
9: report all §9 metrics as mean±std over TEST trials only
```

## 8. Hyperparameters and tuning protocol (binding)

- Representation: B=5 default (ablate 5–9); H×W = ROI-cropped beam edge
  (longest side resized to 224, zero-padded square per §6 adapter); 2 polarity
  channels (C=2, T=B frames).
- Encoder: divided Transformer 8–12 layers, hidden 384–768, Kinetics-pretrained
  (ported per §6 adapter); CNN control = 3D inflation of ResNet18 for video
  input (He et al. 2016 proposed 2D ResNet; the 3D inflation is defined here).
- Loss weights: `λp in [1e-3, 1e-1]`, `λs in [1e-4, 1e-2]`, tuned on
  **validation Delta_RMSE only, not matched RMSE** (tuning on matched RMSE
  invalidates O2). λf=0.1 fixed, never tuned.
- Optimiser: Adam, lr 1e-4–3e-4. **Batch language (binding):** micro-batch =
  what fits in VRAM per step (4060: 4; T4: 8–16); effective batch = micro ×
  accumulation steps (target eff 16; 32 only via accumulation). Max 50 epochs,
  early-stop patience 10 on validation Delta_RMSE.
- Physics: m=1, `k,c` via log decrement + modal fit on one global
  TRAIN-condition trial (§7), frozen; residual normalised as above.
- O4 ablations (binding, non-gameable): (a) data-only (λp=λs=0),
  (b) +smoothness only (λp=0, λs from grid), (c) +SDOF only (λs=0, λp from
  grid), (d) both (λp,λs jointly from grid). Each variant gets the SAME fixed
  budget: 3×3 λ grid × identical epoch cap, best-by-val-Delta_RMSE. "Both" uses
  its own jointly-tuned pair from the same grid — never hand-picked, never
  borrowed from (b)/(c). Report matched + all shifts for each.
- Seeds, exact trial IDs per fold, `m,c,k` + `fd/fn` values, λs, pretrain
  checkpoint URL+SHA, GPU type + `nvidia-smi` + `pip freeze` hash must be logged
  per run (env versions logged, never pinned to this doc's snapshot values).

## 9. Metrics — input/output and evaluation

### Input spec
Primary input event stream `E={(x,y,t,p)}` (µs, ±1) from the XJTU-DV-Beam rig
(hardware trigger synced; hammer strikes the H-beam externally; irregular
event cloud → voxel grid so CNN/Transformer backbones consume it without
custom event kernels). Supervision LDV `q(t)` CSV at kHz (unit convention §5).
Model input is the voxel in §5 (T=B frames, C=2 channels, frozen train stats).
No hand-crafted features; encoder learns from voxel occupancy + polarity
distribution over time.

### Output spec (per window)
`q_hat(t) in R^T` (T=1000 @1 kHz); `f_hat_n` (Hz, trained head; FFT-peak of
`q_hat` is inference-only fallback); normalised `L_phys` (self-consistency).
No contrastive output (deleted per §7).

### Evaluation metrics (report ALL as mean ± std across independent TEST trials)

| Metric | What it measures | Unit / range |
|---|---|---|
| Waveform RMSE | Pointwise fidelity q_hat vs q | same as q, lower better |
| MAE | Outlier-robust pointwise error | same as q |
| Pearson r | Shape/phase alignment | [−1,1], higher better |
| Rel. freq. error \|f_hat−f\|/f | Resonance error from LDV FFT peak | %, lower better |
| Normalised L_phys | SDOF self-consistency; catches degenerate drift | dimensionless, amplitude-blind |
| Robustness gap Delta | Stability under s-shift (estimators below; RMSE + freq error; r sign-flipped) | same as M |
| sigma terms | Falsification yardstick O5 (formula below) | same as M |
| Inference latency / throughput | Deployability; soft bar <50 ms/window on T4 (not pass/fail) | ms and Hz |

Delta estimators (binding — trials are UNPAIRED across conditions, so Delta is
a difference of group means, never per-trial):
`Delta_M = mean(M | shift, test) − mean(M | matched, test)` for lower-better M
(RMSE, freq error). For Pearson r (higher-better):
`Delta_r = mean(r | matched, test) − mean(r | shift, test)`. Standard error:
`se_Delta = √(s_shift²/n_shift + s_matched²/n_matched)` (ddof=1). Tuning and
early-stop use **Delta_RMSE on VAL trials only**; final reported Delta uses
TEST trials only (§5 split). O5 rule: `R = Delta_base − Delta_phys` (same M),
`sigma_R = √(se_base² + se_phys²)`; physics wins iff `R > sigma_R`, else report
**non-identifiable** (negative result counts). Report RQ4 (freq error degrades
less than waveform RMSE — spectral peak coarser than pointwise trace) and
matched-condition degradation separately: a regulariser helping shifted data
but harming matched data is reported as such, never hidden.

Report **accuracy (M_matched) and stability (Delta) separately**; never average
them away. RMSE+MAE+r cover absolute/shape errors; freq error is the
engineering quantity expected to move less under shift (test as RQ4);
Delta is the central O2 metric (`M(matched)` is the matched upper bound);
sigma terms are the bar physics must clear (O5).

### Expected performance (targets, not promises)

Matched (same `s`): waveform RMSE within 5–10% of LDV, Pearson r > 0.90,
rel. freq. error < 2%. Shifted: RMSE Delta smaller *with* physics than without;
freq error < 5% shifted and degrades less than waveform error. If Delta
reduction < sigma_Delta at n=30 → report non-identifiable.

### Cost/performance context (indicative 2025–2026 US catalogue, single point)

| | Accelerometer | LDV (gold) | Frame camera | Struct-X target |
|---|---|---|---|---|
| Example | MEMS/IEPE | Polytec PSV/OFV-5000 | Indust. CMOS | DAVIS346/Prophesee |
| Cost USD | 427–3,840 | 12,728 / 70k–140k | 500–2k (+light) | 4,730 (4,400 EUR) |
| Approx INR @96.39 | Rs 41k–3.7L | Rs 12.3L / 67L–1.35Cr | Rs 48k–1.93L | Rs 4.56L (acad. 4.04L) |
| Contact | Bonded (changes m,c) | Non-contact | Non-contact | Non-contact |
| Temporal | 0.1–10 kHz | DC–100 kHz | 30–1000 fps | µs (~1 MHz/px) |
| Dyn. range | ~90 dB | >120 dB | ~60 dB | ~140 dB |
| Blur/alias | None | None | Severe >100 Hz | Negligible |
| Field | Cabled, per-point | Tripod, still head | Needs bright light | Low power, low BW |
| Drift | Recal/mount | Speckle/alignment | Exposure/rolling shutter | Threshold/bias |
| Waveform | — (baseline) | ≈0 (truth) | High at >100 Hz | Within 5–10% of LDV |
| Freq err | <1% | <0.5% | >5% (blur) | <2% matched, <5% shifted |
| Stability | Mount-sensitive | LOS-sensitive | Very sensitive | Minimise Delta |

Sources (with verification status): DAVIS346 commercial €4,400, academic
€3,900; PCB general-purpose $427, ICP Shock 350C03 $915 to 350B42
$3,840, US only (live page is a category index — prices NOT
re-verifiable, treat as indicative catalogue snapshot); Polytec OFV-5000 set
$12,728 via NAG link **DEAD (404 on 2026-10-23, UNVERIFIED — do not cite as
verified)**; Polytec scanning 500k–1M RMB corroborated by OSP guide (verified
live, OFV-505/5000 class + 500k–1M RMB band); Aaral Tech exclusive Polytec
India distributor (verified live); EUR→USD 1.075 ECB avg; INR Rs 96.39/$
(Morningstar 6 Oct 2025 14:32 UTC). No India list — Polytec India via Aaral
Tech, PCB via GEM tender. Landed India ≈ +20% duty + 18% GST + shipping.
Context: lab rig and single point. Events do NOT beat LDV on
raw accuracy; the claim is cost/performance + field stability.

## 10. Hardware and compute budget (binding constraints)

PROVENANCE: this section is a user-supplied execution-environment addendum,
verified where noted 2026-10-23. It never overrides §§1–9.

Future agent runs on **either** Kaggle T4 **or** local RTX 4060 Laptop, **via
Kaggle CLI**. Design for the smaller one.

- **Kaggle T4:** 16 GB VRAM (confirm per run via `nvidia-smi`; quotas/sessions
  from the Kaggle UI at runtime — old `kaggle.com/docs/gpu` URL is DEAD, do not
  cite quota numbers from memory). Internet must be ON (Zenodo fallback +
  pretrained weights). Paths: read-only input mounts at
  `/kaggle/input/<owner>/<dataset>/`; ONLY `/kaggle/working` persists into the
  output bundle — exfil checkpoints via `kernels output` BEFORE expiry, then
  version back to a Dataset for resume.
- **Local RTX 4060 Laptop (verified 2026-10-23):** 8 GB VRAM
  (`nvidia-smi`: 8188 MiB), torch 2.14.1+cu130 + CUDA available (log `pip
  freeze` per run; never pin to this snapshot). This is the bottleneck — all
  configs must fit in **≤8 GB**.
- **Memory (binding numbers):** "batch 16 fits 8 GB" for TimeSformer-Base is NOT
  credible as stated (Base ≈121 M params → Adam m+v ≈1 GB fp32 before
  activations; 16 clips × T=5 × 224² through 12×768 layers overflows 8 GB even
  with AMP). Binding rule: **micro-batch** = what fits (4060: start at 4; T4:
  start at 8) with `torch.cuda.max_memory_allocated()` logged; **effective
  batch** = micro × accumulation (target eff 16). Fallback ladder, in order:
  (1) shrink micro to 1 + raise accumulation, (2) B=5→B=3, (3) layers 12→8 /
  hidden 768→384, (4) ResNet-3D control only. VRAM math must be measured on-GPU
  in the smoke test (§6), not assumed from this doc.
- **Rules for ≤8 GB:** default B=5, ROI longest-side 224 + square pad (§6),
  AMP fp16 for encoder/heads with **fp32 carve-out for `L_phys`/derivatives/ε
  (§7)**, accumulation to eff 16, precompute voxels to `.pt`/memmap once (never
  decode aedat4 in `__getitem__`; then `workers=2, pin_memory=true,
  persistent_workers=true, prefetch_factor=2`). Gradient checkpointing is
  OPTIONAL (the 2021 TimeSformer has no hooks — insertion point would be the
  per-block wrapper in the ported `models/timesformer.py`, ~20–30% slower);
  ResNet control needs none. Joint space-time attention is **banned**; HR/L
  TimeSformer variants banned (repo-lore ~32 GB requirement, unmeasured here).
- **Disk (binding):** Beam.zip 1.7 GB; unpacked Beam share UNKNOWN (21 GB figure
  was whole-record, do not assume) — inventory + `df -h` BEFORE unzip, then
  `rm` the zip. Voxel store at ~90–270 windows (§5 correction) is small
  (order 100s of MB fp32, not GBs). Checkpoints: Base ≈484 MB fp32 each — keep
  ONLY best-per-fold (sharded per ablation), exfil to output + Dataset version,
  never accumulate best+last × folds × ablations in working (20 GB naive total
  overflows the ~20 GB output cap).
- **Run budget (binding):** K LOCO folds × 4 O4 ablations trainings (K=4 →
  16 trainings); max 50 epochs, patience 10, each fold targeted <2 h; total
  must fit session/quota — resume across pushes via Dataset-versioned
  checkpoints (`kernels push -t <timeout>` + `output -p` + `datasets version`).
  If over budget: cut B/layers per the ladder first, never window-level splits.
- **License fence (binding):** TimeSformer code is CC-NC 4.0 (research-only) —
  the §1 "10× cheaper than LDV" deployment story is therefore RESEARCH
  FRAMING ONLY. Any product path must use the ResNet control (or a clean-room
  replacement), never the ported TimeSformer weights. Kinetics zoo URLs+SHAs
  must be pinned at first successful fetch with a ResNet-only fallback if
  weights 404.
- **Reproducibility:** fix seeds, log GPU type + VRAM + micro/eff batch + AMP +
  commit hash per run; latency per GPU separately (ms/window, windows/s).

### 10b. Kaggle CLI runbook (binding — verified CLI 2.2.4 present, auth missing)

PROVENANCE: same user-supplied addendum as §10. Correct as of CLI 2.2.4 in
this env (2026-10-23); re-check `--help` if the CLI version changes.

Checked 2026-10-23: `kaggle` 2.2.4 installed, but `~/.kaggle/kaggle.json`
**absent** — future agent must authenticate first:

```bash
# 1. Get token from https://www.kaggle.com/settings → API → Create New Token,
#    OR use env vars (no desktop download path on cloud runners):
export KAGGLE_USERNAME=<user> KAGGLE_KEY=<key>   # alternative to kaggle.json
mkdir -p ~/.kaggle && cp /path/to/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
kaggle datasets list -s "event camera vibration"  # connectivity smoke test
```

Dataset flow — PUBLIC dataset required. Beam.zip 1.7 GB exceeds practical
notebook-git limits, so use a Kaggle Dataset, not repo files):

```bash
kaggle datasets init -p beam-dataset   # creates dataset-metadata.json (exact name)
# edit dataset-metadata.json: set id "eros483/struct-x-beam", title, and
# licenses: [{name: "CC-BY-4.0"}] (template defaults CC0-1.0 — WRONG for Beam)
cp /path/to/XJTU-DV-Beam.zip beam-dataset/
kaggle datasets create -p beam-dataset -u   # -u = PUBLIC (verified flag); datasets default to private, never omit -u
kaggle datasets status eros483/struct-x-beam   # poll until ready
kaggle datasets files eros483/struct-x-beam    # verify listing
kaggle datasets version -p beam-dataset -m "v2 inventory fix"  # later updates
```

Kernel flow — PUBLIC notebooks required. Do NOT `pip install` the 2021 TimeSformer env; port it:

```bash
kaggle kernels init -p kernel-structx   # creates kernel-metadata.json (exact name)
# edit kernel-metadata.json: set id "eros483/structx-run" (MUST match the slug
# used below — init dir name alone does not set it), code_file entry script,
# language, kernel_type, is_private "false" (= PUBLIC notebook; template defaults
# "true"), enable_gpu "true", enable_internet "true",
# machine_shape, dataset_sources ["eros483/struct-x-beam"], plus
# competition/kernel/model_sources as needed — incomplete metadata fails schema validation
kaggle kernels push -p kernel-structx --accelerator gpu -t 720  # T4 pin attempt + timeout (mins); UI fallback: Settings → Accelerator → GPU T4
kaggle kernels status eros483/structx-run
kaggle kernels logs eros483/structx-run [-f for streaming]  # NO -s flag exists
kaggle kernels output eros483/structx-run -p ./outputs [--file-pattern, --page-size as needed]
```

Mounts/verification: dataset lands at `/kaggle/input/<user-slug>/<dataset-slug>/`
(slugs are lowercased, dashes preserved — verify with `ls` first). Every run
logs `nvidia-smi` (confirms T4 vs P100 actually assigned) + `pip freeze`; pull
`output` BEFORE session expiry and version checkpoints to a Dataset (§10 disk
policy) for resume.

## 11. Implementation plan for the future agent

Suggested repo layout (adapt, don't over-engineer):

```
src/structx/{data_loader.py, voxel.py, physics.py, models/{timesformer.py,resnet3d.py,heads.py}, train.py, evaluate.py, mck_fit.py}
configs/{base.yaml}   # B, λp, λs, lr, batch, splits, seeds
scripts/{download.sh, preprocess.py, run_folds.sh}
tests/test_physics.py # collapse-guard + finite-difference smoke test
```

- Deps: torch, numpy, scipy (FFT, log-decrement), aedat/dv-processing (read
  actual files; pin version after first successful load), pyyaml.
- `download.sh`: fetch only `XJTU-DV-Beam.zip` from Zenodo DOI (1.7 GB), verify
  md5 `d3af5b54a227be70fa932d93323d5920` (source: Zenodo file table, seen live
  2026-10-23). On mismatch: retry with `curl -C -`
  resume, then recompute-and-record the new hash in the run log instead of
  bricking; never silently skip verification. `unzip -q`, then print inventory
  as `trial_id × condition_label` table + header-driven sensor-resolution
  check (§5 "do not hardcode", enforced here).
- `voxel.py`: refractory/hot-pixel filter → ROI crop → B-bin polarity-split
  voxel + per-bin standardisation; time-surface fallback flag.
- `mck_fit.py`: log-decrement/modal fit on one held-out training trial → frozen
  m,c,k JSON (never touch test trials).
- `physics.py`: finite-difference derivatives + normalised `L_phys`; unit test
  that `q_hat=0` does NOT give zero loss and that a pure sine near `f_n` gives
  lower `L_phys` than white noise.
- `train.py` / `evaluate.py`: trial-level k-fold leave-one-condition-out,
  early-stop on Delta, log gradient ratio, emit mean±std table for §9 metrics
  across matched + 3 shift conditions × 4 ablations.
- Forbidden shortcuts: window-level shuffle, tuning λ on matched RMSE,
  estimating m,c,k on test, training Transformer from scratch, claiming damage
  classification, hardcoding sensor resolution.

## 12. Risks and open questions

- Condition labels in the archive may be coarser than "illumination/threshold/
  bias/combined" (e.g. only distance/angle varies) — inventory first; if a
  shift axis is missing, construct it with the Gehrig-Simulator convention
  (threshold/bias sweep) and label it synthetic.
- 30 trials may leave physics non-identifiable (O5) — that is an allowed,
  publishable outcome.
- LDV may be velocity while model predicts displacement (or vice versa) —
  check headers; differentiate/integrate consistently before RMSE.
- Time-surface vs voxel fallback and FFT-peak vs learned `f_hat_n` must both
  be runnable; keep the non-default path one flag away.

## 13. Novelty claims (map to G1–G5)

1. First LDV-referenced event vibration pipeline (G1).
2. First event-vision + SDOF physics-residual fusion; prior physics lived on
   accelerometers/FEM (G2).
3. Robustness gap Delta as first-class continuous target metric for sensing
   shift (G3+G4).
4. Normalised-residual collapse guard for scarce-data physics (G5).
5. Pretrained-vs-lightweight duality with trial-level splits (anti-leakage).

## 14. References (full, self-contained)

Link status verified 2026-10-23 via fetch unless marked otherwise.

Project: **Struct-X: Physics-Aware Event Vision for Robust Non-Contact
Structural Vibration Monitoring** — Arnab Mandal, Department of Computer
Science and Engineering, am483@snu.edu.in. Keywords: event cameras,
non-contact vibration measurement, structural health monitoring,
physics-informed neural networks, spatiotemporal Transformers, domain shift,
robustness. Acknowledgment (carries into the final paper): XJTU-DV is open
source from Xi'an Jiaotong University (Li et al., CJME 2026) via Zenodo
17447083 — rig, LDV ground truth, and example loaders courtesy of the authors.

1. Gallego et al., Event-based vision: A survey, TPAMI 44(1), 2022. (Q1)
   https://doi.org/10.1109/TPAMI.2020.3008413 — LIVE (DOI resolves).
2. Gehrig & Scaramuzza, Event-based vision meets computer vision, TPAMI 45(10), 2023.
3. Rebecq et al., High speed and HDR video with an event camera, CVPR 2017.
4. Wu et al., Event-based visual deformation measurement, arXiv:2602.14376, 2026.
   https://arxiv.org/abs/2602.14376 — LIVE. All three numbers PDF-verified (see §4).
5. Raissi et al., PINNs, J. Comput. Phys. 378, 2019. (Q1)
   https://doi.org/10.1016/j.jcp.2018.10.045 — LIVE (DOI resolves).
6. Wang et al., Structural damage identification by physics-guided residual NN,
   Eng. Struct. 318:118703, 2024. (Q1)
   https://doi.org/10.1016/j.engstruct.2024.118703 — LIVE (DOI resolves).
7. Wu/Tang et al., Accurate structural displacement reconstruction from
   acceleration + vision via PINNs, MSSP 235:112961, 2025. (Q1)
   https://doi.org/10.1016/j.ymssp.2025.112961 — LIVE (DOI resolves).
8. Zhou & Lai, Structural damage classification under varying environments via
   open-set DA, MSSP 218:111561, 2024. (Q1)
   https://doi.org/10.1016/j.ymssp.2024.111561 — LIVE (DOI resolves).
9. Bertasius et al., Is space-time attention all you need for video
   understanding? ICML 2021. (A*)
   https://proceedings.mlr.press/v139/bertasius21a.html — LIVE;
   code https://github.com/facebookresearch/TimeSformer — LIVE (archived, CC-NC 4.0).
10. Li et al., XJTU-DV, Chinese J. Mech. Eng. 39:100163, 2026. (Q1)
    https://doi.org/10.1016/j.cjme.2025.100163 — LIVE (DOI resolves).
11. He et al., Deep residual learning, CVPR 2016.
12. Karniadakis et al., Physics-informed ML, Nature Rev. Phys. 3, 2021.
13. iniVation DAVIS346 MONO €4,400. https://shop.inivation.com/products/davis346
    — LIVE (price + 346×260 specs confirmed).
14. PCB accelerometers (indicative $427–$3,840 US catalogue snapshot).
    https://www.pcb.com/sensors-for-test-measurement/accelerometers — LIVE
    (category page; exact prices unverified, treat as indicative).
15. Polytec OFV-5000 set $12,728. https://www.nagw.com/product/20260609/356607.html
    — **DEAD (404 on 2026-10-23, UNVERIFIED).**
16. Polytec scanning 500k–1M RMB band corroborated by
    https://ospmotiongo.com/archives/4004 — LIVE (selection guide, OFV-505/5000 class).
17. Aaral Tech (Polytec India distributor). https://www.aaraltech.com — LIVE.
- Dataset: Zenodo 17447083, XJTU-DV-Beam.zip 1.7 GB, md5
  d3af5b54a227be70fa932d93323d5920, CC-BY-4.0.
  https://zenodo.org/records/17447083 — LIVE (file table confirms sizes).
- Loader mirror: https://gr.xjtu.edu.cn/web/lixiang/xjtu-dv — LIVE (faculty page,
  dataset announced 2025-12-01). Contact lixiang@xjtu.edu.cn.
- DEAD at verification: `kaggle.com/docs/gpu`, NAG Polytec URL above. Everything
  else in this list returned 2xx or a resolving DOI.

## 15. Professor-required report analyses (binding — the final paper MUST contain both)

### 15a. What the physics loss cost us vs saved us (estimation method, however vague)

The report must contain a cost/benefit table for the SDOF physics term,
measured — not argued — on identical hardware/seeds using the O4 ablations:

- **Cost per step:** time 20 train steps/window-batch with and without `L_phys`
  (ablation (c) vs (a), (d) vs (b)), same micro-batch, same GPU. Report
  ms/window train-step overhead. Expectation (state as prior, then replace with
  measurement): the residual is O(T)=O(1000) finite differences per window —
  ~1–5% of a Transformer step — so per-step cost is near-zero; the FLOP count
  is `~6T` multiply-adds/window vs ~10¹⁰ for the encoder. If measurement
  disagrees, measurement wins.
- **Savings in convergence:** epochs-to-target (first epoch reaching
  val-Delta_RMSE within 5% of that variant's best) for (a) vs (c) and (b) vs
  (d); multiply by min/epoch → GPU-hours per fold per variant. Report
  Δepochs, ΔGPU-hours, and whether physics reached the same/better Delta in
  less total compute.
- **Accuracy price:** Δmatched-RMSE and Δmatched-freq-error for (c)−(a),
  (d)−(b). A physics term that saves 30% GPU-hours but costs 2% matched RMSE
  is reported as exactly that trade — never netted into one score.
- **Verdict line (one sentence):** "Physics cost X% per step, saved/lost Y
  GPU-hours per fold, moved matched RMSE by Z%." If Y ≤ 0 within noise, the
  honest finding is "regularisation benefit with no compute saving" (or a net
  cost) — consistent with the O5 non-identifiable outcome.

### 15b. How others use event cameras + transferable teachings

The report's related-work must survey event-camera usage BEYOND the Struct-X
pipeline and extract one actionable teaching per use (each mapped to the §
where Struct-X applies it):

- **Event–frame VDM (Wu et al. 2026, read §4):** hybrid rig (Prophesee EVK4 +
  210 fps grayscale via 50:50 beam splitter, square-wave sync), IWE contrast
  maximisation + NCC cross-correlation, survival-rate/EPE/SEPE protocol.
  Teaching → our sync/calibration (§5 hardware trigger), our coarse
  free-decay-then-refine mindset, and why we REJECT per-sequence optimisation
  (no learned prior across trials) in favour of a trained encoder.
- **Event optical flow (E-RAFT) + frame interpolation (TimeLens, both cited in
  Wu §4.2):** short-term motion from events; synthesising intermediate frames
  from events+frames to help image methods. Teaching → our voxel-grid choice
  over time surfaces (§6: preserve temporal resolution, keep polarity), and the
  time-surface fallback flag (§5) exists because flow literature shows surfaces
  win when event rates collapse under bad illumination.
- **HDR/high-speed video from events (Rebecq et al. 2017):** events' 140 dB /
  µs-timestamp edge over CMOS. Teaching → the §1 sensor-motivation numbers and
  why frame cameras are the "fails on fast" column in §9 (blur >100 Hz).
- **Representation taxonomy (Gallego et al. 2022):** event frames vs time
  surfaces vs voxel grids vs learned representations. Teaching → §6 decision
  log: voxels (temporal resolution + backbone compatibility + polarity) with
  per-channel frozen train stats; surfaces only as fallback.
- **Limitations literature (Wu §4.5: continuity/brightness-constancy failures,
  specularities):** Teaching → §12 risk register (narrow beam edge under
  varying light breaks geometric methods; our robustness gap Δ exists to
  measure exactly this) and the ROI-crop + refractory-filter steps in §5.

Each teaching gets one row: {other use → their trick → our § decision}. No
new implementation is required — this is a report section, roughly one page.

## 16. Future-agent acceptance checklist

- [ ] Only Beam.zip downloaded + md5 verified; Rotor/Pump untouched.
- [ ] Trial/condition inventory printed; m,c,k trial isolated from test.
- [ ] Micro-batch sized by on-GPU measurement (start 4 on 4060 / 8 on T4) + accumulation to eff 16, AMP with fp32 physics carve-out.
- [ ] All 4 ablations × LOCO folds reported as mean±std on TEST trials; Delta + se_Delta with r sign-flip; RQ4 + matched-degradation reported.
- [ ] O5 falsification statement present (even if negative: R < sigma_R → non-identifiable).
- [ ] §15a physics cost/benefit table (ms/window overhead, ΔGPU-hours, Δmatched) in the paper.
- [ ] §15b event-camera usage survey with per-use teaching rows in the paper.
- [ ] Kaggle dataset AND all kernels PUBLIC (`-u` flag, `"is_private": "false"`).
- [ ] No window-level split, no damage-classification claim, no joint attention, no unnormalised physics loss, no outside-file references.
