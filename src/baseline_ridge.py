"""F013: classical linear baseline — RidgeCV on pooled voxel features.

Features: per-channel spatial means of each (2B,H,W) voxel window -> (2B,).
Fit on TRAIN-trial windows only per LOCO fold; StandardScaler frozen from
train (never pooled stats). Closed-form, CPU minutes. No torch needed.
"""

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
FS = 1000.0


def pooled_features(vox):
    """Per-channel bin means: (n,C,H,W) -> (n,C); single (C,H,W) -> (C,)."""
    v = np.asarray(vox, dtype=np.float64)
    if v.ndim == 3:
        return v.mean(axis=(1, 2))
    return v.mean(axis=(2, 3))


def fit_ridge_fold(trial_data, train_trials, alphas=ALPHAS):
    """Fit scaler + RidgeCV on TRAIN trials only. Returns fitted dict."""
    xs = [pooled_features(trial_data[t][0]) for t in train_trials]
    ys = [np.asarray(trial_data[t][1], dtype=np.float64) for t in train_trials]
    x_train = np.concatenate(xs, axis=0)
    y_train = np.concatenate(ys, axis=0)
    scaler = StandardScaler().fit(x_train)
    model = RidgeCV(alphas=list(alphas)).fit(scaler.transform(x_train), y_train)
    return {"scaler": scaler, "model": model, "train_trials": list(train_trials)}


def predict_ridge(fitted, x):
    """Predict waveforms for pooled feature rows."""
    x = np.asarray(x, dtype=np.float64)
    return np.asarray(fitted["model"].predict(fitted["scaler"].transform(x)))


def window_fft_peak(q, fs=FS, fmin=5.0, fmax=500.0):
    """Hann FFT peak + parabolic interpolation (same convention as mck_fit)."""
    q = np.asarray(q, dtype=np.float64).ravel()
    n = q.size
    spectrum = np.abs(np.fft.rfft(q * np.hanning(n), n=4 * n))
    freqs = np.fft.rfftfreq(4 * n, d=1.0 / fs)
    band = np.nonzero((freqs >= fmin) & (freqs <= min(fmax, 0.999 * fs / 2.0)))[0]
    peak = band[int(np.argmax(spectrum[band]))]
    if peak <= 0 or peak >= spectrum.size - 1:
        return float(freqs[peak])
    prev, centre, nxt = spectrum[peak - 1 : peak + 2]
    denom = prev - 2.0 * centre + nxt
    shift = 0.5 * (prev - nxt) / denom if denom != 0.0 else 0.0
    return float(freqs[peak] + shift * (freqs[1] - freqs[0]))
