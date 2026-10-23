"""Log-decrement m,c,k fit (m=1) on one train trial. See design sec 7.

Takes an LDV array + fs and returns frozen JSON-serialisable params. The
caller enforces the trial policy (ONE global trial of the TRAIN condition);
this module never sees val/test/shift trials.

Absolute scale is unidentifiable from free decay alone: log decrement yields
only damped frequency fd + damping ratio zeta, so m=1 by convention and
k = (2*pi*fn)^2, c = 2*zeta*sqrt(k) with fn = fd/sqrt(1-zeta^2).
"""

import json
from pathlib import Path

import numpy as np
from scipy.signal import detrend, find_peaks


def _damped_frequency_fft(q, fs, fmin=5.0, fmax=500.0):
    """Damped frequency via Hann-windowed FFT peak + parabolic interpolation."""
    n = q.size
    spectrum = np.abs(np.fft.rfft(q * np.hanning(n), n=4 * n))
    freqs = np.fft.rfftfreq(4 * n, d=1.0 / fs)
    hi = min(fmax, 0.999 * fs / 2.0)
    band = np.nonzero((freqs >= fmin) & (freqs <= hi))[0]
    if band.size < 3:
        raise ValueError("signal too short for requested frequency band")
    peak = band[np.argmax(spectrum[band])]
    if peak <= 0 or peak >= spectrum.size - 1:
        return float(freqs[peak])
    prev, centre, nxt = spectrum[peak - 1 : peak + 2]
    denom = prev - 2.0 * centre + nxt
    shift = 0.5 * (prev - nxt) / denom if denom != 0.0 else 0.0
    return float(freqs[peak] + shift * (freqs[1] - freqs[0]))


def _damping_ratio_log_decrement(q, fs, fmax=500.0):
    """Zeta from log decrement: linear fit of ln(positive-peak heights)."""
    distance = max(1, int(fs / fmax))
    peaks, _ = find_peaks(q, distance=distance)
    if peaks.size == 0:
        raise ValueError("no peaks found: signal has no free-decay oscillation")
    heights = q[peaks]
    kept = heights >= 0.05 * heights.max()  # decay tail below noise floor
    amps = heights[kept]
    if amps.size < 3:
        raise ValueError("fewer than 3 usable decay peaks")
    slope = np.polyfit(np.arange(amps.size), np.log(amps), 1)[0]
    delta = max(float(-slope), 0.0)
    return float(delta / np.sqrt(4.0 * np.pi**2 + delta**2))


def fit_mck(q, fs, fmin=5.0, fmax=500.0):
    """Fit SDOF params on one free-decay LDV trial; m=1 by convention."""
    q = np.asarray(q, dtype=np.float64).ravel()
    fs = float(fs)
    if q.size < 3 * fs / fmin:
        raise ValueError("signal shorter than 3 cycles of fmin")
    detrended = detrend(q, type="linear")
    if not np.any(np.isfinite(detrended)) or np.all(detrended == 0.0):
        raise ValueError("signal is empty or constant")
    fd = _damped_frequency_fft(detrended, fs, fmin, fmax)
    zeta = min(_damping_ratio_log_decrement(detrended, fs, fmax), 0.999)
    fn = fd / np.sqrt(1.0 - zeta**2)
    k = (2.0 * np.pi * fn) ** 2
    c = 2.0 * zeta * np.sqrt(k)
    return {
        "m": 1.0,
        "c": float(c),
        "k": float(k),
        "zeta": float(zeta),
        "fd": float(fd),
        "fn": float(fn),
        "fs": fs,
    }


def save_mck(params, path):
    """Freeze params to JSON (frozen m,c,k file)."""
    Path(path).write_text(json.dumps(params, indent=2) + "\n")


def load_mck(path):
    """Load frozen params from JSON."""
    return json.loads(Path(path).read_text())
