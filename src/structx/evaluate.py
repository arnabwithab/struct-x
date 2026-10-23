"""Metrics, Delta/se estimators, O5 verdict, latency (design sec 9).

Trials are UNPAIRED across conditions: every Delta is a difference of GROUP
means, never per-trial. All metrics report mean +- std over TEST trials only.
Self-consistency reuses physics.physics_loss (never reimplement the stencil).
"""

import time

import numpy as np
import torch

from structx.physics import DT, physics_loss


def rmse(q_hat, q):
    """Pointwise waveform fidelity, same units as q (lower better)."""
    return float(
        np.sqrt(np.mean((np.asarray(q_hat, dtype=float) - np.asarray(q, dtype=float)) ** 2))
    )


def mae(q_hat, q):
    """Outlier-robust pointwise error, same units as q."""
    return float(np.mean(np.abs(np.asarray(q_hat, dtype=float) - np.asarray(q, dtype=float))))


def pearson_r(q_hat, q):
    """Shape/phase alignment in [-1, 1] (higher better); 0.0 if either side is constant."""
    a = np.asarray(q_hat, dtype=float).ravel()
    b = np.asarray(q, dtype=float).ravel()
    if a.std() == 0.0 or b.std() == 0.0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def rel_freq_error(f_hat, f):
    """|f_hat - f| / f (lower better); f is the LDV full-trial FFT peak."""
    return float(abs(float(f_hat) - float(f)) / abs(float(f)))


def l_phys(q_hat, m, c, k, dt=DT):
    """Normalised SDOF self-consistency (dimensionless, amplitude-blind)."""
    q = q_hat if isinstance(q_hat, torch.Tensor) else torch.as_tensor(np.asarray(q_hat))
    return float(physics_loss(q, m, c, k, dt).item())


def summarize_trials(values):
    """Mean +- sample std (ddof=1, same convention as se_delta) over TEST trials."""
    v = np.asarray(list(values), dtype=float)
    return {"mean": float(v.mean()), "std": float(v.std(ddof=1))}


def delta_m(matched, shift):
    """Robustness gap for lower-better M: mean(shift) - mean(matched)."""
    return float(
        np.mean(np.asarray(list(shift), dtype=float))
        - np.mean(np.asarray(list(matched), dtype=float))
    )


def delta_r(matched_r, shift_r):
    """Gap for higher-better Pearson r (sign-flipped): mean(matched) - mean(shift)."""
    return float(
        np.mean(np.asarray(list(matched_r), dtype=float))
        - np.mean(np.asarray(list(shift_r), dtype=float))
    )


def se_delta(matched, shift):
    """Standard error of Delta: sqrt(s_shift^2/n_shift + s_matched^2/n_matched), ddof=1."""
    a = np.asarray(list(matched), dtype=float)
    b = np.asarray(list(shift), dtype=float)
    return float(np.sqrt(b.var(ddof=1) / b.size + a.var(ddof=1) / a.size))


def o5_verdict(delta_base, delta_phys, se_base, se_phys):
    """Falsification rule: physics wins iff R > sigma_R, else NON-IDENTIFIABLE."""
    r = float(delta_base) - float(delta_phys)
    sigma = float(np.sqrt(float(se_base) ** 2 + float(se_phys) ** 2))
    wins = r > sigma
    if wins:
        statement = f"physics wins: R={r:.4g} > sigma_R={sigma:.4g}"
    else:
        statement = f"non-identifiable: R={r:.4g} <= sigma_R={sigma:.4g} (negative counts)"
    return {"R": r, "sigma_R": sigma, "wins": wins, "statement": statement}


def rq4_compare(freq_delta, rmse_delta):
    """True iff relative freq error degrades less under shift than waveform RMSE."""
    return bool(float(freq_delta) < float(rmse_delta))


def matched_degradation_report(matched_base, matched_phys):
    """Matched-condition cost of physics, reported separately -- never netted into R."""
    base = float(np.mean(np.asarray(list(matched_base), dtype=float)))
    phys = float(np.mean(np.asarray(list(matched_phys), dtype=float)))
    harm = phys - base  # lower-better M: positive harm = physics hurt matched
    harmed = harm > 0
    if harmed:
        statement = f"physics harms matched performance by {harm:.4g} (reported separately)"
    else:
        statement = f"no matched degradation (harm={harm:.4g})"
    return {
        "matched_base": base,
        "matched_phys": phys,
        "harm": harm,
        "harmed": harmed,
        "statement": statement,
    }


def measure_latency_ms(model_fn, sample, reps=20):
    """Mean ms/window over reps calls (one untimed warmup); soft bar <50ms on T4, not pass/fail."""
    model_fn(sample)
    start = time.perf_counter()
    for _ in range(int(reps)):
        model_fn(sample)
    return (time.perf_counter() - start) / int(reps) * 1000.0
