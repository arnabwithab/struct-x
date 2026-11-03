"""F014 layers 1+4 helpers: behavioral figures + L_phys-vs-error calibration.

Layer 1 (behavioral): best/worst-test-trial overlays, spectra, error table.
Layer 4 (calibration): per-window L_phys vs |error| scatter + correlation,
testing the design-promised uncertainty proxy (expect weak/null; report it).
Pure functions only (no IO, no checkpoints) so figure regeneration is pinned
by tests/test_f014_layers14.py. Inference driver: scripts/f014_layers14.py.
"""

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


def pick_best_worst(trial_rmse):
    """(best, worst) trial ids by trial-mean RMSE (lower better)."""
    ids = sorted(trial_rmse)
    vals = [float(trial_rmse[t]) for t in ids]
    return ids[int(np.argmin(vals))], ids[int(np.argmax(vals))]


def calibration_corr(lphys, err):
    """Pearson + Spearman between per-window L_phys and |error|.

    Returns {"pearson": r, "spearman": rho, "n": n}; nan when either side
    is constant (no discriminative signal — itself a calibration finding).
    """
    lp = np.asarray(lphys, dtype=float).ravel()
    er = np.asarray(err, dtype=float).ravel()
    n = int(min(lp.size, er.size))
    lp, er = lp[:n], er[:n]
    if n < 3 or lp.std() == 0.0 or er.std() == 0.0:
        return {"pearson": float("nan"), "spearman": float("nan"), "n": n}
    with np.errstate(invalid="ignore"):
        rho = float(spearmanr(lp, er).statistic)
    return {"pearson": float(np.corrcoef(lp, er)[0, 1]), "spearman": rho, "n": n}


def overlay_fig(q_best, qh_best, q_worst, qh_worst, title):
    """True vs predicted waveform overlays: best (top) / worst (bottom) trial."""
    fig, ax = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    for a, q, qh, tag in (
        (ax[0], q_best, qh_best, "best"),
        (ax[1], q_worst, qh_worst, "worst"),
    ):
        a.plot(np.asarray(q).ravel(), label="LDV true", linewidth=1.0)
        a.plot(np.asarray(qh).ravel(), label="model pred", linewidth=1.0, alpha=0.8)
        a.set_title(tag)
        a.legend(fontsize=8)
    ax[1].set_xlabel("sample @1kHz")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig


def spectrum_fig(q, qh, fs, f_target, title):
    """Hann magnitude spectra (true vs pred) with FFT-peak + LDV-fd markers."""
    from baseline_ridge import window_fft_peak

    q = np.asarray(q, dtype=float).ravel()
    qh = np.asarray(qh, dtype=float).ravel()
    n = q.size
    freqs = np.fft.rfftfreq(4 * n, d=1.0 / fs)
    spec = lambda v: np.abs(np.fft.rfft(v * np.hanning(n), n=4 * n))
    fig, a = plt.subplots(figsize=(8, 3.5))
    a.plot(freqs, spec(q), label="LDV true", linewidth=1.0)
    a.plot(freqs, spec(qh), label="model pred", linewidth=1.0, alpha=0.8)
    a.axvline(window_fft_peak(qh, fs=fs), linestyle="--", label="pred FFT peak")
    a.axvline(float(f_target), linestyle=":", label="LDV fd")
    a.set_xlim(0, 250)
    a.set_xlabel("Hz")
    a.legend(fontsize=8)
    a.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig


def scatter_fig(lphys, err, pearson, spearman, title):
    """Per-window L_phys vs |error| scatter with correlations in the title."""
    fig, a = plt.subplots(figsize=(5, 4))
    a.scatter(
        np.asarray(lphys, dtype=float).ravel(),
        np.asarray(err, dtype=float).ravel(),
        s=8,
        alpha=0.6,
    )
    a.set_xlabel("L_phys (per-window)")
    a.set_ylabel("|error| RMSE (per-window)")
    a.set_title(f"{title}\npearson={pearson:.3f} spearman={spearman:.3f}", fontsize=10)
    fig.tight_layout()
    return fig


def fig_to_png_bytes(fig):
    """Render fig to PNG bytes (Agg) and close it."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    return buf.getvalue()
