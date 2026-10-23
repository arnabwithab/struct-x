"""Trial inventory, LDV/event loading, windowing (trial-level, no leakage). See design sec 5."""

import numpy as np
from scipy import signal

FS_TARGET = 1000  # Hz resample for every LDV target
WINDOW_S = 1.0  # window length
STRIDE_S = 0.5  # window stride
EXCLUSION_S = 0.1  # drop windows starting <100 ms after hammer trigger
LP_CUTOFF_HZ = 500.0  # 4th-order Butterworth low-pass at resample Nyquist
LP_ORDER = 4
EPS = 1e-6  # frozen z-score guard


def is_velocity_unit(unit):
    """True when an LDV header unit (e.g. 'm/s', 'mm/s') means velocity."""
    return "/s" in str(unit)


def velocity_to_displacement(q_vel, dt_s):
    """Integrate velocity once (cumulative trapezoid) then linear-detrend.

    Binding unit convention (design sec 5): if the header shows velocity,
    integrate BEFORE anything else.
    """
    q_vel = np.asarray(q_vel, dtype=np.float64)
    q_disp = np.zeros_like(q_vel)
    q_disp[1:] = np.cumsum((q_vel[1:] + q_vel[:-1]) / 2.0) * dt_s
    return signal.detrend(q_disp, type="linear")


def preprocess_ldv(t_raw_s, q_raw, unit="m", fs_target=FS_TARGET):
    """Per-trial LDV pipeline: integrate-if-velocity -> detrend -> low-pass -> resample.

    Returns (t_out_s, q_out) at fs_target Hz. Z-scoring is separate so frozen
    train stats can be applied (see compute_train_stats / apply_frozen_zscore).
    """
    t_raw_s = np.asarray(t_raw_s, dtype=np.float64)
    q_raw = np.asarray(q_raw, dtype=np.float64)
    dt_raw = float(np.median(np.diff(t_raw_s)))
    fs_raw = 1.0 / dt_raw
    q = velocity_to_displacement(q_raw, dt_raw) if is_velocity_unit(unit) else q_raw
    q = signal.detrend(q, type="linear")
    nyq = fs_raw / 2.0
    cutoff = min(LP_CUTOFF_HZ, 0.9 * nyq)
    sos = signal.butter(LP_ORDER, cutoff, btype="low", fs=fs_raw, output="sos")
    q = signal.sosfiltfilt(sos, q)
    n_out = int(round((t_raw_s[-1] - t_raw_s[0]) * fs_target))
    q_out = signal.resample(q, n_out)
    t_out = t_raw_s[0] + np.arange(n_out) / fs_target
    return t_out, q_out


def compute_train_stats(signals):
    """Mean/std over concatenated TRAIN trials only; frozen for val/test."""
    all_q = np.concatenate([np.asarray(s, dtype=np.float64).ravel() for s in signals])
    return float(all_q.mean()), float(all_q.std())


def apply_frozen_zscore(q, mean, std, eps=EPS):
    """Per-trial z-score with frozen train stats (never per-window stats)."""
    return (np.asarray(q, dtype=np.float64) - mean) / (std + eps)


def window_starts(duration_s, window_s=WINDOW_S, stride_s=STRIDE_S, exclusion_s=EXCLUSION_S):
    """Window start times (s); excludes all starts <100 ms post-trigger."""
    starts = []
    s = 0.0
    while s + window_s <= duration_s + 1e-9:
        if s >= exclusion_s - 1e-9:
            starts.append(round(s, 9))
        s = round(s + stride_s, 9)
    return starts


def count_windows(durations_s, **kwargs):
    """Total windows over trials (design correction: ~90-270 for 30 trials)."""
    return sum(len(window_starts(d, **kwargs)) for d in durations_s)


def make_loco_splits(trial_conditions, seed=0):
    """Deterministic leave-one-condition-out over conditions found (folds=K).

    Per fold held out on condition H: all other conditions pool as matched
    (train/val-matched/test-matched); H splits 50/50 into val-shift (tuning
    only) vs test-shift (final Delta only). Trial-level only — window-level
    shuffle is FORBIDDEN.
    """
    rng = np.random.RandomState(seed)
    conditions = sorted(set(trial_conditions.values()))
    folds = {}
    for held in conditions:
        matched = sorted(t for t, c in trial_conditions.items() if c != held)
        shift = sorted(t for t, c in trial_conditions.items() if c == held)
        rng.shuffle(matched)
        rng.shuffle(shift)
        n = len(matched)
        n_test = max(1, int(round(0.2 * n))) if n >= 3 else 0
        n_val = max(1, int(round(0.2 * n))) if n >= 3 else 0
        n_val = min(n_val, n - n_test) if n >= 3 else 0
        folds[held] = {
            "train": matched[: n - n_val - n_test],
            "val_matched": matched[n - n_val - n_test : n - n_test],
            "test_matched": matched[n - n_test :],
            "val_shift": shift[: len(shift) // 2],
            "test_shift": shift[len(shift) // 2 :],
        }
    return folds


def check_no_leakage(split_to_trials):
    """Raise if any trial appears in more than one split set."""
    seen = {}
    for name, trials in split_to_trials.items():
        for t in trials:
            if t in seen:
                raise AssertionError(f"trial {t} in both {seen[t]} and {name}")
            seen[t] = name
    return True


def assign_windows_to_split(windows, trial_to_split):
    """Map (trial_id, start_s) windows to split sets via their trial id."""
    by_set = {}
    for trial, start in windows:
        by_set.setdefault(trial_to_split[trial], []).append((trial, start))
    return by_set


def pick_mck_fit_trial(train_trial_ids):
    """The ONE trial for m,c,k fit: deterministic pick from TRAIN only."""
    trials = sorted(train_trial_ids)
    if not trials:
        raise ValueError("no train trials to pick m,c,k fit trial from")
    return trials[0]
